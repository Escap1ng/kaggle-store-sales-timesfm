"""Diagnostic: same-calendar-window check on Aug 16-31 of 2015/2016 for LGBM vs TimesFM."""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from timesfm3 import TimesFM3Forecaster

from timesfm_v1 import DATA, HORIZON, OUT, load_series_matrix, rmsle
from timesfm_v2 import load_onpromo_matrix, load_test_onpromo
from timesfm_v3 import build_local_flags, build_national_flag, load_holiday_active, load_oil
from lgbm_v1 import CAT_COLS, PARAMS, build_feature_arrays, gather

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(os.cpu_count() or 8)
BATCH_CHUNK = 256
WINDOWS = [pd.Timestamp("2015-08-16"), pd.Timestamp("2016-08-16")]


def timesfm_window(forecaster, X, dates, wstart):
    ctx = X[np.asarray(dates <= wstart - pd.Timedelta(days=1)), :]
    truth = X[np.asarray((dates >= wstart) & (dates <= wstart + pd.Timedelta(days=HORIZON - 1))), :].T
    preds = []
    for start in range(0, ctx.shape[1], BATCH_CHUNK):
        cols = range(start, min(start + BATCH_CHUNK, ctx.shape[1]))
        contexts = [np.asarray(ctx[:, i], dtype=np.float32) for i in cols]
        for out in forecaster.predict_batch(contexts=contexts, horizon=HORIZON,
                                            make_positive=True, return_quantiles=True):
            preds.append(np.asarray(out.quantiles))
    q = np.stack(preds)
    return float(rmsle(truth, q[:, :, 4]))


def main():
    X, mi, dates = load_series_matrix()
    T, S = X.shape
    dates_xt = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=HORIZON), freq="D")
    fut_dates = dates_xt[T:]

    promo_xt = np.vstack([load_onpromo_matrix(mi, dates), load_test_onpromo(mi, fut_dates)])
    oil_xt = pd.Series(load_oil(dates)).reindex(pd.DatetimeIndex(dates_xt)).ffill().bfill().to_numpy(dtype=np.float32)
    active = load_holiday_active()
    nat = build_national_flag(active, dates_xt)
    local = build_local_flags(active, mi, dates_xt)
    sales_xt = np.vstack([X, np.full((HORIZON, S), np.nan, dtype=np.float32)])
    feats = build_feature_arrays(dates_xt, sales_xt, promo_xt, oil_xt, nat, local, mi)
    names = list(feats.keys())
    cat_idx = [names.index(c) for c in CAT_COLS]

    di_dates = np.repeat(np.arange(len(dates_xt)), S)
    si_series = np.tile(np.arange(S), len(dates_xt))
    target = sales_xt.reshape(-1)
    valid_row = np.ones(len(target), dtype=bool)
    for k in ("lag_364", "roll_mean_28"):
        valid_row &= ~np.isnan(feats[k].reshape(-1))
    valid_row &= ~np.isnan(target)
    date_of_row = dates_xt.to_numpy()[di_dates]

    forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=64)
    print(f"timesfm on {forecaster.device}")

    for wstart in WINDOWS:
        train_rows = valid_row & (date_of_row < np.datetime64(wstart))
        val_rows = valid_row & (date_of_row >= np.datetime64(wstart - pd.Timedelta(days=HORIZON))) & (date_of_row < np.datetime64(wstart))
        pred_dates = pd.date_range(wstart, periods=HORIZON, freq="D")
        pred_rows = np.isin(date_of_row, pred_dates.to_numpy())

        X_tr = gather(feats, names, di_dates[train_rows], si_series[train_rows])
        X_va = gather(feats, names, di_dates[val_rows], si_series[val_rows])
        dtr = lgb.Dataset(X_tr, np.log1p(target[train_rows]), categorical_feature=cat_idx, free_raw_data=True)
        dva = lgb.Dataset(X_va, np.log1p(target[val_rows]), reference=dtr, categorical_feature=cat_idx)
        booster = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        X_pred = gather(feats, names, di_dates[pred_rows], si_series[pred_rows])
        pred = np.expm1(booster.predict(X_pred, num_iteration=booster.best_iteration)).reshape(HORIZON, S)
        truth = X[np.asarray((dates >= wstart) & (dates <= wstart + pd.Timedelta(days=HORIZON - 1))), :]
        err_lgbm = rmsle(truth, np.maximum(pred, 0))
        err_tf = timesfm_window(forecaster, X, dates, wstart)
        print(f"{wstart.date()}: lgbm {err_lgbm:.4f} | timesfm {err_tf:.4f} | delta {err_lgbm - err_tf:+.4f}", flush=True)


if __name__ == "__main__":
    main()
