"""Stage 1 knob sweep: symmetric averaging and input z-norm, dual-window.

Baseline is rerun in the same process so the table is self-consistent.
Adopt a config only if it beats baseline by >= 0.002 on BOTH windows.
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
MIN_GAIN = 0.002

CONFIGS = [
    ("baseline", {}),
    ("sym", {"use_symmetric_averaging": True}),
    ("zn", {"use_znorm": True}),
    ("sym+zn", {"use_symmetric_averaging": True, "use_znorm": True}),
]


def run_window(forecaster, X, wstart, label, **kwargs):
    ctx_mask = X_dates <= wstart - pd.Timedelta(days=1)
    val_mask = (X_dates >= wstart) & (X_dates <= wstart + pd.Timedelta(days=HORIZON - 1))
    context = X[ctx_mask, :]
    truth = X[val_mask, :].T
    t0 = time.time()
    preds = []
    S = context.shape[1]
    for start in range(0, S, BATCH_CHUNK):
        cols = range(start, min(start + BATCH_CHUNK, S))
        contexts = [np.asarray(context[:, i], dtype=np.float32) for i in cols]
        for out in forecaster.predict_batch(
            contexts=contexts, horizon=HORIZON, make_positive=True, return_quantiles=True, **kwargs
        ):
            preds.append(np.asarray(out.quantiles))
    q = np.stack(preds)  # (S, H, Q)
    err = float(rmsle(truth, q[:, :, 4]))
    print(f"[{label}] {err:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return err, q


def main():
    global X_dates
    X, mi, X_dates = load_series_matrix()

    forecaster = TimesFM3Forecaster.from_pretrained(
        "google/timesfm-3.0-pytorch", per_core_batch_size=64
    )
    print(f"model loaded on {forecaster.device}")

    results = {}
    for wstart, lab in zip(WINDOWS, WINDOW_LABELS):
        print(f"=== {lab}: context <= {(wstart - pd.Timedelta(days=1)).date()} ===")
        for name, kw in CONFIGS:
            err, _q = run_window(forecaster, X, wstart, f"{lab}/{name}", **kw)
            results[(lab, name)] = err

    print("\n=== Stage 1 table (median RMSLE) ===")
    print(f"{'config':8}" + "".join(f"{l:>10}" for l in WINDOW_LABELS) + f"{'max delta':>12}")
    base = {lab: results[(lab, "baseline")] for lab in WINDOW_LABELS}
    winners = []
    for name, _kw in CONFIGS:
        deltas = [results[(lab, name)] - base[lab] for lab in WINDOW_LABELS]
        print(f"{name:8}" + "".join(f"{results[(lab, name)]:>10.4f}" for lab in WINDOW_LABELS) + f"{max(deltas):>+12.4f}")
        if name != "baseline" and all(d <= -MIN_GAIN for d in deltas):
            winners.append(name)
    print(f"adoptable configs: {winners if winners else 'none'}")

    if not winners:
        print("no submission generated; v1 (0.39257) stays current best")
        return

    best = min(winners, key=lambda n: max(results[(lab, n)] - base[lab] for lab in WINDOW_LABELS))
    kw = dict(CONFIGS[[c[0] for c in CONFIGS].index(best)][1])
    print(f"winner: {best} {kw}; running submission pass")

    fut_dates = pd.date_range(X_dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    test = pd.read_csv(os.path.join(DATA, "test.csv"), parse_dates=["date"])
    assert fut_dates[0] == test["date"].min() and fut_dates[-1] == test["date"].max()

    preds = []
    for start in range(0, X.shape[1], BATCH_CHUNK):
        cols = range(start, min(start + BATCH_CHUNK, X.shape[1]))
        contexts = [np.asarray(X[:, i], dtype=np.float32) for i in cols]
        for out in forecaster.predict_batch(
            contexts=contexts, horizon=HORIZON, make_positive=True, return_quantiles=True, **kw
        ):
            preds.append(np.asarray(out.quantiles))
    q_sub = np.stack(preds)
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
    out_path = os.path.join(OUT, "sub_timesfm_v4.csv")
    sub.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
