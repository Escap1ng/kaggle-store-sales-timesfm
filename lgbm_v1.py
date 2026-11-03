"""Stage 3 v1: LightGBM with leakage-safe features.

Features use only values at least 16 days old, so the same feature builder works
for training rows, held-out-window rows and the 16-day submission horizon.
Validation: dual windows (2017-05-01..16, 2017-07-16..31). Submission is only
written if LGBM beats TimesFM's baseline on BOTH windows.
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from timesfm_v1 import DATA, HORIZON, OUT, load_series_matrix, rmsle
from timesfm_v2 import load_onpromo_matrix, load_test_onpromo
from timesfm_v3 import build_local_flags, build_national_flag, load_holiday_active, load_oil

WINDOWS = [pd.Timestamp("2017-05-01"), pd.Timestamp("2017-07-16")]
WINDOW_LABELS = ["valMay", "valJul"]
TIMESFM_REF = {"valMay": 0.4464, "valJul": 0.4026}

LAGS = [16, 21, 28, 35, 42, 364]
ROLLS = [7, 14, 28, 56, 140, 364]
CAT_COLS = ["store_nbr", "type", "cluster", "city", "state", "family"]

PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=255,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    num_threads=32,
    verbosity=-1,
    seed=42,
)


def build_feature_arrays(dates_xt, sales_xt, promo_xt, oil_xt, nat, local, mi,
                         lags=LAGS, rolls=ROLLS, shift_base=16):
    T, S = sales_xt.shape
    feats = {}
    df = pd.DataFrame(sales_xt)
    for k in lags:
        feats[f"lag_{k}"] = df.shift(k).to_numpy(dtype=np.float32)
    base = df.shift(shift_base)
    for w in rolls:
        r = base.rolling(w, min_periods=max(3, w // 4))
        feats[f"roll_mean_{w}"] = r.mean().to_numpy(dtype=np.float32)
        feats[f"roll_std_{w}"] = r.std().to_numpy(dtype=np.float32)
    del df, base

    d = pd.DatetimeIndex(dates_xt)
    month_end = d.days_in_month
    dist_end = month_end - d.day
    dist_15 = np.abs(d.day.to_numpy() - 15)
    dist_pay = np.minimum(dist_end.to_numpy(), dist_15)
    cal = {
        "dow": d.day_of_week.to_numpy(),
        "dom": d.day.to_numpy(),
        "month": d.month.to_numpy(),
        "year": d.year.to_numpy() - 2013,
        "doy": d.day_of_year.to_numpy(),
        "woy": d.isocalendar().week.to_numpy(),
        "is_weekend": (d.day_of_week.to_numpy() >= 5).astype(np.float32),
        "is_payday": ((d.day.to_numpy() == 15) | d.is_month_end).astype(np.float32),
        "dist_payday": dist_pay.astype(np.float32),
    }
    for name, vals in cal.items():
        feats[name] = np.repeat(vals.astype(np.float32)[:, None], S, axis=1)

    stores = pd.read_csv(os.path.join(DATA, "stores.csv")).set_index("store_nbr")
    store_nbrs = np.array([s for s, _ in mi])
    families = [f for _, f in mi]
    type_codes = pd.Categorical(stores.loc[store_nbrs, "type"]).codes.astype(np.float32)
    city_codes = pd.Categorical(stores.loc[store_nbrs, "city"]).codes.astype(np.float32)
    state_codes = pd.Categorical(stores.loc[store_nbrs, "state"]).codes.astype(np.float32)
    fam_codes = pd.Categorical(families).codes.astype(np.float32)
    per_series = {
        "store_nbr": store_nbrs.astype(np.float32),
        "type": type_codes,
        "cluster": stores.loc[store_nbrs, "cluster"].to_numpy(dtype=np.float32),
        "city": city_codes,
        "state": state_codes,
        "family": fam_codes,
    }
    for name, vec in per_series.items():
        feats[name] = np.tile(vec, (T, 1))

    feats["onpromotion"] = promo_xt.astype(np.float32)
    feats["oil"] = np.repeat(oil_xt.astype(np.float32)[:, None], S, axis=1)
    feats["holiday_nat"] = np.repeat(nat.astype(np.float32)[:, None], S, axis=1)
    feats["holiday_local"] = local.astype(np.float32)
    return feats


def gather(feats, names, di, si):
    X = np.empty((len(di), len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = feats[name][di, si]
    return X


def main():
    t_start = time.time()
    X, mi, dates = load_series_matrix()
    T, S = X.shape
    dates_xt = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=HORIZON), freq="D")
    fut_dates = dates_xt[T:]

    promo = load_onpromo_matrix(mi, dates)
    promo_xt = np.vstack([promo, load_test_onpromo(mi, fut_dates)])
    oil = load_oil(dates)
    oil_xt = pd.Series(oil).reindex(pd.DatetimeIndex(dates_xt)).ffill().bfill().to_numpy(dtype=np.float32)
    active = load_holiday_active()
    nat = build_national_flag(active, dates_xt)
    local = build_local_flags(active, mi, dates_xt)

    sales_xt = np.vstack([X, np.full((HORIZON, S), np.nan, dtype=np.float32)])
    feats = build_feature_arrays(dates_xt, sales_xt, promo_xt, oil_xt, nat, local, mi)
    names = list(feats.keys())
    cat_idx = [names.index(c) for c in CAT_COLS]
    print(f"features: {len(names)} ({len(cat_idx)} categorical), build {time.time() - t_start:.0f}s", flush=True)

    di_dates = np.repeat(np.arange(len(dates_xt)), S)
    si_series = np.tile(np.arange(S), len(dates_xt))
    target = sales_xt.reshape(-1)

    valid_row = np.ones(len(target), dtype=bool)
    for k in ("lag_364", "roll_mean_28"):
        valid_row &= ~np.isnan(feats[k].reshape(-1))
    valid_row &= ~np.isnan(target)
    date_of_row = dates_xt.to_numpy()[di_dates]

    results = {}
    for wstart, lab in zip(WINDOWS, WINDOW_LABELS):
        t0 = time.time()
        train_rows = valid_row & (date_of_row < np.datetime64(wstart))
        val_rows = valid_row & (date_of_row >= np.datetime64(wstart - pd.Timedelta(days=HORIZON))) & (date_of_row < np.datetime64(wstart))
        pred_dates = pd.date_range(wstart, periods=HORIZON, freq="D")
        pred_rows = np.isin(date_of_row, pred_dates.to_numpy())

        X_tr = gather(feats, names, di_dates[train_rows], si_series[train_rows])
        y_tr = np.log1p(target[train_rows])
        X_va = gather(feats, names, di_dates[val_rows], si_series[val_rows])
        y_va = np.log1p(target[val_rows])
        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=True)
        dva = lgb.Dataset(X_va, y_va, reference=dtr, categorical_feature=cat_idx)
        booster = lgb.train(
            PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        X_pred = gather(feats, names, di_dates[pred_rows], si_series[pred_rows])
        log_pred = booster.predict(X_pred, num_iteration=booster.best_iteration)
        pred = np.expm1(log_pred).reshape(len(pred_dates), S)
        truth = X[np.asarray(dates >= wstart), :][:HORIZON, :]
        err = rmsle(truth, np.maximum(pred, 0))
        results[lab] = err
        print(f"{lab}: RMSLE {err:.4f} (best_iter {booster.best_iteration}, {time.time() - t0:.0f}s)", flush=True)
        np.save(os.path.join(OUT, f"val_lgbm_{lab}.npy"), pred)

    print("\n=== vs TimesFM ===")
    for lab in WINDOW_LABELS:
        print(f"{lab}: lgbm {results[lab]:.4f} | timesfm {TIMESFM_REF[lab]:.4f} | delta {results[lab] - TIMESFM_REF[lab]:+.4f}")
    better_both = all(results[lab] < TIMESFM_REF[lab] for lab in WINDOW_LABELS)
    if not better_both:
        print("LGBM loses to TimesFM on at least one window; no submission written")
        return

    print("=== training final model for submission ===")
    t0 = time.time()
    train_rows = valid_row & (date_of_row < np.datetime64(dates[-1] + pd.Timedelta(days=1)))
    val_rows = valid_row & (date_of_row >= np.datetime64(dates[-1] - pd.Timedelta(days=HORIZON))) & (date_of_row <= np.datetime64(dates[-1]))
    X_tr = gather(feats, names, di_dates[train_rows], si_series[train_rows])
    y_tr = np.log1p(target[train_rows])
    X_va = gather(feats, names, di_dates[val_rows], si_series[val_rows])
    y_va = np.log1p(target[val_rows])
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=True)
    dva = lgb.Dataset(X_va, y_va, reference=dtr, categorical_feature=cat_idx)
    booster = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    X_sub = gather(feats, names, di_dates[np.isin(date_of_row, fut_dates.to_numpy())], si_series[np.isin(date_of_row, fut_dates.to_numpy())])
    pred = np.expm1(booster.predict(X_sub, num_iteration=booster.best_iteration)).reshape(HORIZON, S)
    print(f"final model: best_iter {booster.best_iteration}, {time.time() - t0:.0f}s")

    test = pd.read_csv(os.path.join(DATA, "test.csv"), parse_dates=["date"])
    pos = {(int(s), f): i for i, (s, f) in enumerate(mi)}
    doff = {d: i for i, d in enumerate(fut_dates)}
    sales = np.zeros(len(test), dtype=np.float64)
    not_found = 0
    for r, row in enumerate(test.itertuples(index=False)):
        key = (int(row.store_nbr), row.family)
        if key not in pos or row.date not in doff:
            not_found += 1
            continue
        sales[r] = max(pred[doff[row.date], pos[key]], 0.0)
    if not_found:
        print(f"warning: {not_found} test rows unmatched")
    sub = pd.DataFrame({"id": test["id"].to_numpy(), "sales": sales})
    out_path = os.path.join(OUT, "sub_lgbm_v1.csv")
    sub.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(sub)} rows), total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
