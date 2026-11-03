import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from timesfm3 import TimesFM3Forecaster

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

HORIZON = 16
BATCH_CHUNK = 256

torch.set_num_threads(os.cpu_count() or 8)


def rmsle(y_true, y_pred):
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def load_series_matrix():
    train = pd.read_csv(
        os.path.join(DATA, "train.csv"), parse_dates=["date"], usecols=["date", "store_nbr", "family", "sales"]
    )
    dup = train.duplicated(subset=["date", "store_nbr", "family"]).sum()
    if dup:
        print(f"warning: {dup} duplicate rows, averaging")
        train = train.groupby(["date", "store_nbr", "family"], as_index=False)["sales"].mean()
    dates = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    pivot = train.pivot(index="date", columns=["store_nbr", "family"], values="sales").reindex(dates)
    X = pivot.to_numpy(dtype=np.float32, na_value=np.nan)
    print(f"matrix: dates {dates[0].date()}..{dates[-1].date()} ({len(dates)}), series {X.shape[1]}")
    print(f"NaN cells: {np.isnan(X).mean():.4%}")
    return X, pd.MultiIndex.from_tuples(pivot.columns.tolist(), names=["store_nbr", "family"]), dates


def run_forecasts(forecaster, X, label):
    t0 = time.time()
    forecasts = []
    for start in range(0, X.shape[1], BATCH_CHUNK):
        chunk = [np.asarray(X[:, i], dtype=np.float32) for i in range(start, min(start + BATCH_CHUNK, X.shape[1]))]
        chunk_preds = []
        for out in forecaster.predict_batch(
            contexts=chunk,
            horizon=HORIZON,
            make_positive=True,
        ):
            chunk_preds.append(out.forecast)
        forecasts.append(np.stack(chunk_preds))
        print(f"[{label}] {start + len(chunk)}/{X.shape[1]} series done, {time.time() - t0:.1f}s", flush=True)
    return np.concatenate(forecasts, axis=0)  # (S, H)


def main():
    X, mi, dates = load_series_matrix()

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    forecaster = TimesFM3Forecaster.from_pretrained(
        "google/timesfm-3.0-pytorch",
        device="cpu",
        per_core_batch_size=32,
    )
    print(f"model loaded on {forecaster.device}, global_context={forecaster.global_context}")

    if mode in ("all", "val"):
        val_end = pd.Timestamp("2017-07-15")
        ctx_mask = dates <= val_end
        val_mask = (dates > val_end) & (dates <= val_end + pd.Timedelta(days=HORIZON))
        print(f"validation: context <= {val_end.date()}, target {val_mask.sum()} days")
        preds = run_forecasts(forecaster, X[ctx_mask, :], "val")  # (S, H)
        preds = np.maximum(preds, 0)
        truth = X[val_mask, :]  # (H, S)
        if truth.shape[0] != HORIZON:
            raise RuntimeError("validation window outside data range")
        err = rmsle(truth.T, preds)
        print(f"RMSLE (store x family, day1..16 pooled): {err:.4f}")
        fam = mi.get_level_values("family")
        fam_arr = np.array(fam.tolist())
        df_f = pd.DataFrame(
            {"family": fam_arr, "rmsle": np.sqrt(((np.log1p(preds) - np.log1p(truth.T)) ** 2).mean(axis=1))}
        )
        print(df_f.groupby("family")["rmsle"].mean().sort_values().to_string())

    if mode in ("all", "sub"):
        train_end = dates[-1]
        print(f"submission: context <= {train_end.date()}")
        preds = run_forecasts(forecaster, X, "sub")  # (S, H)
        preds = np.maximum(preds, 0)
        test = pd.read_csv(os.path.join(DATA, "test.csv"), parse_dates=["date"])
        fut_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        assert fut_dates[0] == test["date"].min() and fut_dates[-1] == test["date"].max()
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
        out_path = os.path.join(OUT, "sub_timesfm_v1.csv")
        sub.to_csv(out_path, index=False)
        print(f"saved {out_path} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
