"""Blend evaluation: TimesFM median vs LGBM v2 log-space weighted average.

Uses the four calibration windows; prints RMSLE per weight so a robust weight
can be chosen before spending a submission.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from timesfm3 import TimesFM3Forecaster

from timesfm_v1 import HORIZON, OUT, load_series_matrix, rmsle

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(os.cpu_count() or 8)
BATCH_CHUNK = 256
LGBM_SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""

WINDOWS = [
    (pd.Timestamp("2017-05-01"), "May17"),
    (pd.Timestamp("2017-07-16"), "Jul17"),
    (pd.Timestamp("2016-08-16"), "Aug16"),
    (pd.Timestamp("2015-08-16"), "Aug15"),
]
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def main():
    X, mi, dates = load_series_matrix()
    forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=64)

    data = {}
    for wstart, lab in WINDOWS:
        ctx = X[np.asarray(dates <= wstart - pd.Timedelta(days=1)), :]
        truth = X[np.asarray((dates >= wstart) & (dates <= wstart + pd.Timedelta(days=HORIZON - 1))), :]
        tf_path = os.path.join(OUT, f"val_tf_{lab}.npy")
        if os.path.exists(tf_path):
            tf = np.load(tf_path)
        else:
            preds = []
            for start in range(0, ctx.shape[1], BATCH_CHUNK):
                cols = range(start, min(start + BATCH_CHUNK, ctx.shape[1]))
                contexts = [np.asarray(ctx[:, i], dtype=np.float32) for i in cols]
                for out in forecaster.predict_batch(contexts=contexts, horizon=HORIZON,
                                                    make_positive=True, return_quantiles=True):
                    preds.append(np.asarray(out.quantiles)[:, 4])
            tf = np.stack(preds).T  # (H, S)
            np.save(tf_path, tf)
        lgbm = np.load(os.path.join(OUT, f"val_lgbm_v2{LGBM_SUFFIX}_{lab}.npy"))
        data[lab] = (truth, tf, lgbm)
        print(f"{lab}: tf {rmsle(truth, tf):.4f} | lgbm_v2 {rmsle(truth, lgbm):.4f}", flush=True)

    print("\n=== blend RMSLE (w = weight on LGBM v2, log1p space) ===")
    print("    w " + "".join(f"{lab:>9}" for _, lab in WINDOWS))
    for w in WEIGHTS:
        row = []
        for _, lab in WINDOWS:
            truth, tf, lgbm = data[lab]
            blend = np.expm1((1 - w) * np.log1p(np.maximum(tf, 0)) + w * np.log1p(np.maximum(lgbm, 0)))
            row.append(rmsle(truth, np.maximum(blend, 0)))
        print(f"{w:5.1f} " + "".join(f"{e:>9.4f}" for e in row))


if __name__ == "__main__":
    main()
