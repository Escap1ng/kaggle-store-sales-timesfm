import os
import time

import numpy as np
import pandas as pd
import torch

from timesfm3 import TimesFM3Forecaster

from timesfm_v1 import DATA, HORIZON, OUT, ROOT, load_series_matrix, rmsle

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(os.cpu_count() or 8)

BATCH_CHUNK = 256


def load_onpromo_matrix(mi, dates):
    train = pd.read_csv(
        os.path.join(DATA, "train.csv"),
        parse_dates=["date"],
        usecols=["date", "store_nbr", "family", "onpromotion"],
    )
    train = train.groupby(["date", "store_nbr", "family"], as_index=False)["onpromotion"].mean()
    pivot = train.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion").reindex(dates)
    pivot = pivot.reindex(columns=mi)
    return pivot.to_numpy(dtype=np.float32, na_value=np.nan)  # (T, S)


def load_test_onpromo(mi, fut_dates):
    test = pd.read_csv(os.path.join(DATA, "test.csv"), parse_dates=["date"])
    pivot = test.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion").reindex(fut_dates)
    pivot = pivot.reindex(columns=mi)
    return pivot.to_numpy(dtype=np.float32, na_value=np.nan)  # (H, S)


def run(forecaster, X, label, cov=None, pf_future=None):
    """Returns quantile forecasts (S, H, Q). cov: (T_ctx, S) past covariates.
    pf_future: (H, S) future covariate values aligned with horizon days."""
    t0 = time.time()
    quantile_preds = []
    for start in range(0, X.shape[1], BATCH_CHUNK):
        cols = range(start, min(start + BATCH_CHUNK, X.shape[1]))
        contexts = [np.asarray(X[:, i], dtype=np.float32) for i in cols]
        if cov is not None:
            pfs = []
            for i in cols:
                seq = np.concatenate([cov[:, i], pf_future[:, i]]).astype(np.float32)
                pfs.append(seq[None, :])  # (1, T_ctx + H)
        else:
            pfs = None
        preds = []
        for out in forecaster.predict_batch(
            contexts=contexts,
            horizon=HORIZON,
            past_future_covariates=pfs,
            return_quantiles=True,
            make_positive=True,
            padding_mode="edge" if pfs is not None else "none",
        ):
            preds.append(np.asarray(out.quantiles))
        quantile_preds.append(np.stack(preds))
        done = start + len(contexts)
        print(f"[{label}] {done}/{X.shape[1]} series, {time.time() - t0:.1f}s", flush=True)
    return np.concatenate(quantile_preds, axis=0)  # (S, H, Q)


def sweep(name, q_preds, truth_T):
    """q_preds (S,H,Q), truth (S,H)."""
    qlevels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return [(name, q, rmsle(truth_T, q_preds[:, :, qi])) for qi, q in enumerate(qlevels)]


def main():
    X, mi, dates = load_series_matrix()

    forecaster = TimesFM3Forecaster.from_pretrained(
        "google/timesfm-3.0-pytorch", device="cpu", per_core_batch_size=32
    )
    print(f"model loaded on {forecaster.device}")

    val_end = pd.Timestamp("2017-07-15")
    ctx_mask = dates <= val_end
    val_mask = (dates > val_end) & (dates <= val_end + pd.Timedelta(days=HORIZON))
    truth = X[val_mask, :].T  # (S, H)
    print(f"validation: context <= {val_end.date()}, target {val_mask.sum()} days")

    X_op = load_onpromo_matrix(mi, dates)

    print("=== pass 1: no covariates ===")
    q_no = run(forecaster, X[ctx_mask, :], "no-cov")

    print("=== pass 2: onpromotion covariate ===")
    q_cov = run(forecaster, X[ctx_mask, :], "cov", cov=X_op[ctx_mask, :], pf_future=X_op[val_mask, :])

    qlevels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n=== quantile sweep, validation RMSLE ===")
    print(f"{'q':>5} {'no-cov':>9} {'cov':>9}")
    rows_no = sweep("no-cov", q_no, truth)
    rows_cov = sweep("cov", q_cov, truth)
    for (nm1, q, e1), (nm2, _, e2) in zip(rows_no, rows_cov):
        print(f"{q:>5.2f} {e1:>9.4f} {e2:>9.4f}")
    med_no = rmsle(truth, q_no[:, :, 4])
    med_cov = rmsle(truth, q_cov[:, :, 4])
    print(f"median: no-cov {med_no:.4f}, cov {med_cov:.4f}")

    best_no = min(rows_no, key=lambda r: r[2])
    best_cov = min(rows_cov, key=lambda r: r[2])
    print(f"best: no-cov q{best_no[1]} = {best_no[2]:.4f}; cov q{best_cov[1]} = {best_cov[2]:.4f}")

    np.save(os.path.join(OUT, "val_q_no.npy"), q_no)
    np.save(os.path.join(OUT, "val_q_cov.npy"), q_cov)

    if best_cov[2] < best_no[2]:
        use_cov, use_q = True, qlevels.index(best_cov[1])
    else:
        use_cov, use_q = False, qlevels.index(best_no[1])
    print(f"submission settings: covariate={use_cov}, quantile={qlevels[use_q]}")

    test = pd.read_csv(os.path.join(DATA, "test.csv"), parse_dates=["date"])
    fut_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    assert fut_dates[0] == test["date"].min() and fut_dates[-1] == test["date"].max()

    if use_cov:
        print("=== pass 3: submission with covariate ===")
        test_op = load_test_onpromo(mi, fut_dates)
        q_sub = run(forecaster, X, "sub-cov", cov=X_op, pf_future=test_op)
    else:
        print("=== pass 3: submission without covariate ===")
        q_sub = run(forecaster, X, "sub")

    preds = q_sub[:, :, use_q]  # (S, H)
    pos = {(int(s), f): i for i, (s, f) in enumerate(mi)}
    doff = {d: i for i, d in enumerate(fut_dates)}
    sales = np.zeros(len(test), dtype=np.float64)
    not_found = 0
    for r, row in enumerate(test.itertuples(index=False)):
        key = (int(row.store_nbr), row.family)
        if key not in pos or row.date not in doff:
            not_found += 1
            continue
        sales[r] = preds[pos[key], doff[row.date]]
    if not_found:
        print(f"warning: {not_found} test rows unmatched")
    sub = pd.DataFrame({"id": test["id"].to_numpy(), "sales": sales})
    out_path = os.path.join(OUT, "sub_timesfm_v2.csv")
    sub.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
