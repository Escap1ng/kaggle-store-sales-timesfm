"""v3: oil (no lookahead) + holiday flags as covariates.

Dual-window validation (2017-05-01..16, 2017-07-16..31). The covariate run is
accepted only if it beats the no-covariate median in BOTH windows; only then
is a submission produced. Pass `promo` as argv[1] to append the onpromotion
channel (v2's channel) on top.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from timesfm3 import TimesFM3Forecaster

from timesfm_v1 import DATA, HORIZON, OUT, load_series_matrix, rmsle

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(os.cpu_count() or 8)

BATCH_CHUNK = 256
WINDOWS = [pd.Timestamp("2017-05-01"), pd.Timestamp("2017-07-16")]
WINDOW_LABELS = ["valMay", "valJul"]

use_promo = len(sys.argv) > 1 and sys.argv[1] == "promo"


def load_oil(dates):
    oil = pd.read_csv(os.path.join(DATA, "oil.csv"), parse_dates=["date"])
    oil = oil.groupby("date", as_index=False)["dcoilwtico"].mean().set_index("date")["dcoilwtico"]
    return oil.reindex(pd.DatetimeIndex(dates)).to_numpy(dtype=np.float32)


def load_holiday_active():
    hol = pd.read_csv(os.path.join(DATA, "holidays_events.csv"), parse_dates=["date"])
    hol = hol.groupby("date", as_index=False).first()
    return hol[(~hol["transferred"].astype(bool)) & (hol["type"] != "Work Day")]


def build_national_flag(active, dates_xt):
    hol_dates = pd.DatetimeIndex(active.loc[active["locale"] == "National", "date"])
    return np.isin(pd.DatetimeIndex(dates_xt).to_numpy(), hol_dates.to_numpy()).astype(np.float32)


def build_local_flags(active, mi, dates_xt):
    """(T_xt, S): local holiday matches store city, regional matches state."""
    stores = pd.read_csv(os.path.join(DATA, "stores.csv")).set_index("store_nbr")
    date_pos = {d: i for i, d in enumerate(dates_xt)}
    local = np.zeros((len(dates_xt), len(mi)), dtype=np.float32)
    for i, (store_nbr, _family) in enumerate(mi):
        city, state = stores.loc[store_nbr, "city"], stores.loc[store_nbr, "state"]
        loc = active[
            ((active["locale"] == "Local") & (active["locale_name"] == city))
            | ((active["locale"] == "Regional") & (active["locale_name"] == state))
        ]
        for d in loc["date"]:
            pos = date_pos.get(pd.Timestamp(d))
            if pos is not None:
                local[pos, i] = 1.0
    return local


def oil_noleak(oil_raw, dates_xt, ctx_end):
    """Known oil through ctx_end only (NaNs forward-filled from first valid value);
    future repeats the last known value. Same rule in validation and submission
    so the channel never sees future covariate values it would not have had."""
    known = oil_raw[dates_xt <= ctx_end]
    known_filled = pd.Series(known).ffill().bfill().to_numpy(dtype=np.float32)
    return np.concatenate([known_filled, np.full(HORIZON, known_filled[-1], dtype=np.float32)])


def run(forecaster, X, channel_fn, label):
    """channel_fn(s) -> (V, T_ctx + H) covariate block for series s, or None."""
    t0 = time.time()
    quantile_preds = []
    S = X.shape[1]
    for start in range(0, S, BATCH_CHUNK):
        cols = range(start, min(start + BATCH_CHUNK, S))
        contexts = [np.asarray(X[:, i], dtype=np.float32) for i in cols]
        if channel_fn is None:
            pfs, mode = None, "none"
        else:
            pfs, mode = [channel_fn(i) for i in cols], "edge"
        preds = []
        for out in forecaster.predict_batch(
            contexts=contexts,
            horizon=HORIZON,
            past_future_covariates=pfs,
            return_quantiles=True,
            make_positive=True,
            padding_mode=mode,
        ):
            preds.append(np.asarray(out.quantiles))
        quantile_preds.append(np.stack(preds))
        print(f"[{label}] {start + len(contexts)}/{S} series, {time.time() - t0:.1f}s", flush=True)
    return np.concatenate(quantile_preds, axis=0)  # (S, H, Q)


def validate_window(forecaster, X, oil_raw, nat, local, dates_train, X_op, wstart, label):
    ctx_end = wstart - pd.Timedelta(days=1)
    fut_dates = pd.date_range(wstart, periods=HORIZON, freq="D")
    assert fut_dates[-1] <= dates_train[-1] + pd.Timedelta(days=HORIZON)
    ctx_mask = dates_train <= ctx_end
    val_mask = (dates_train >= wstart) & (dates_train <= fut_dates[-1])
    truth = X[val_mask, :].T  # (S, H)
    flags_cut = ctx_mask.sum()

    def chan_no(i):
        oil = oil_noleak(oil_raw, dates_train, ctx_end)[: flags_cut + HORIZON]
        return np.stack([oil, nat[: flags_cut + HORIZON], local[: flags_cut + HORIZON, i]])

    def chan_promo_base(i):
        promo = np.concatenate([X_op[ctx_mask, i], X_op[val_mask, i]])
        return np.concatenate([chan_no(i), promo[None, :]], axis=0)

    print(f"=== {label}: context <= {ctx_end.date()}, target {val_mask.sum()} days ===")
    q_no = run(forecaster, X[ctx_mask, :], None, f"{label}/no-cov")
    fn = chan_promo_base if use_promo else chan_no
    q_cov = run(forecaster, X[ctx_mask, :], fn, f"{label}/cov")
    med_no, med_cov = float(rmsle(truth, q_no[:, :, 4])), float(rmsle(truth, q_cov[:, :, 4]))
    print(f"{label}: median no-cov {med_no:.4f} | cov {med_cov:.4f} | delta {med_cov - med_no:+.4f}")
    return med_no, med_cov


def main():
    X, mi, dates = load_series_matrix()
    dates_xt = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=HORIZON), freq="D")

    oil_raw = load_oil(dates)
    active = load_holiday_active()
    nat = build_national_flag(active, dates_xt)
    local = build_local_flags(active, mi, dates_xt)
    X_op = None
    if use_promo:
        from timesfm_v2 import load_onpromo_matrix
        X_op = load_onpromo_matrix(mi, dates)

    assert nat.shape[0] == X.shape[0] + HORIZON, "flag axis must cover train dates + 16 submission days"
    assert not np.isnan(nat).any() and not np.isnan(local).any()

    forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=64)
    print(f"model loaded on {forecaster.device}, channels: oil/hol" + ("/promo" if use_promo else ""))

    results = {}
    for wstart, lab in zip(WINDOWS, WINDOW_LABELS):
        med_no, med_cov = validate_window(
            forecaster, X, oil_raw, nat, local, dates, X_op, wstart, lab
        )
        results[lab] = (med_no, med_cov)

    both_better = all(c < n for n, c in results.values())
    print(f"dual-window acceptance: {'PASS' if both_better else 'FAIL'}")

    if not both_better:
        print("no submission generated; v1 (0.39257) stays current best on leaderboard")
        return

    print("=== submission pass with covariates ===")
    fut_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    test = pd.read_csv(os.path.join(DATA, "test.csv"), parse_dates=["date"])
    assert fut_dates[0] == test["date"].min() and fut_dates[-1] == test["date"].max()

    def chan_sub(i):
        oil = oil_noleak(oil_raw, dates_xt, dates[-1])
        block = np.stack([oil, nat, local[:, i]])
        if X_op is not None:
            from timesfm_v2 import load_test_onpromo
            test_op = load_test_onpromo(mi, fut_dates)
            block = np.concatenate([block, np.concatenate([X_op[:, i], test_op[:, i]])[None, :]], axis=0)
        return block.astype(np.float32)

    q_sub = run(forecaster, X, chan_sub, "sub")
    preds = q_sub[:, :, 4]
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
    out_path = os.path.join(OUT, "sub_timesfm_v3.csv")
    sub.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
