"""Stage 3 v2: recursive LightGBM.

Fixes the root cause of v1's August failure: v1 lagged every feature by 16 days,
so early-horizon predictions used stale information during fast-moving seasonal
periods. v2 trains on shift-1 features and rolls predictions forward day by day
over the 16-day horizon, feeding each day's forecast back as history.

Calibration windows include Aug 16-31 of 2016 and 2015, which match the test
period's calendar slot (school season) and exposed v1's weakness.
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from lgbm_v1 import CAT_COLS, PARAMS, build_feature_arrays, gather
from timesfm_v1 import DATA, HORIZON, OUT, load_series_matrix, rmsle
from timesfm_v2 import load_onpromo_matrix, load_test_onpromo
from timesfm_v3 import build_local_flags, build_national_flag, load_holiday_active, load_oil

V2_LAGS = [1, 2, 3, 7, 14, 28, 56, 364]
V2_ROLLS = [7, 14, 28, 56, 140]

WINDOWS = [
    (pd.Timestamp("2017-05-01"), "May17"),
    (pd.Timestamp("2017-07-16"), "Jul17"),
    (pd.Timestamp("2016-08-16"), "Aug16"),
    (pd.Timestamp("2015-08-16"), "Aug15"),
]
V1_REF = {"May17": 0.4377, "Jul17": 0.3943, "Aug16": 0.7044, "Aug15": 0.5024}
TF_REF = {"May17": 0.4464, "Jul17": 0.4026, "Aug16": 0.7685, "Aug15": 0.6623}


def main():
    t_start = time.time()
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
    feats = build_feature_arrays(dates_xt, sales_xt, promo_xt, oil_xt, nat, local, mi,
                                 lags=V2_LAGS, rolls=V2_ROLLS, shift_base=1)
    names = list(feats.keys())
    cat_idx = [names.index(c) for c in CAT_COLS]
    print(f"features: {len(names)} ({len(cat_idx)} categorical), build {time.time() - t_start:.0f}s", flush=True)

    stores = pd.read_csv(os.path.join(DATA, "stores.csv")).set_index("store_nbr")
    store_nbrs = np.array([s for s, _ in mi]).astype(np.float32)
    static = {
        "store_nbr": store_nbrs,
        "type": pd.Categorical(stores.loc[store_nbrs.astype(int), "type"]).codes.astype(np.float32),
        "cluster": stores.loc[store_nbrs.astype(int), "cluster"].to_numpy(dtype=np.float32),
        "city": pd.Categorical(stores.loc[store_nbrs.astype(int), "city"]).codes.astype(np.float32),
        "state": pd.Categorical(stores.loc[store_nbrs.astype(int), "state"]).codes.astype(np.float32),
        "family": pd.Categorical([f for _, f in mi]).codes.astype(np.float32),
    }

    def step_features(work, t_idx):
        values = {}
        hist = pd.DataFrame(work[max(0, t_idx - 400):t_idx, :])
        for k in V2_LAGS:
            values[f"lag_{k}"] = hist.iloc[-k].to_numpy(dtype=np.float32)
        for w in V2_ROLLS:
            r = hist.rolling(w, min_periods=max(3, w // 4))
            values[f"roll_mean_{w}"] = r.mean().iloc[-1].to_numpy(dtype=np.float32)
            values[f"roll_std_{w}"] = r.std().iloc[-1].to_numpy(dtype=np.float32)
        d = dates_xt[t_idx]
        dist_pay = min(d.days_in_month - d.day, abs(d.day - 15))
        cal = {
            "dow": d.day_of_week,
            "dom": d.day,
            "month": d.month,
            "year": d.year - 2013,
            "doy": d.day_of_year,
            "woy": int(d.isocalendar()[1]),
            "is_weekend": float(d.day_of_week >= 5),
            "is_payday": float(d.day == 15 or d.is_month_end),
            "dist_payday": dist_pay,
        }
        for name, v in cal.items():
            values[name] = np.full(S, v, dtype=np.float32)
        for name, vec in static.items():
            values[name] = vec
        values["onpromotion"] = promo_xt[t_idx, :]
        values["oil"] = np.full(S, oil_xt[t_idx], dtype=np.float32)
        values["holiday_nat"] = np.full(S, nat[t_idx], dtype=np.float32)
        values["holiday_local"] = local[t_idx, :]
        return np.column_stack([values[n] for n in names]).astype(np.float32)

    di_dates = np.repeat(np.arange(len(dates_xt)), S)
    si_series = np.tile(np.arange(S), len(dates_xt))
    target = sales_xt.reshape(-1)
    valid_row = np.ones(len(target), dtype=bool)
    for k in ("lag_364", "roll_mean_28"):
        valid_row &= ~np.isnan(feats[k].reshape(-1))
    valid_row &= ~np.isnan(target)
    date_of_row = dates_xt.to_numpy()[di_dates]

    def train_booster(train_end, verbose=False):
        train_rows = valid_row & (date_of_row < np.datetime64(train_end))
        val_rows = valid_row & (date_of_row >= np.datetime64(train_end - pd.Timedelta(days=HORIZON))) & (date_of_row < np.datetime64(train_end))
        X_tr = gather(feats, names, di_dates[train_rows], si_series[train_rows])
        X_va = gather(feats, names, di_dates[val_rows], si_series[val_rows])
        dtr = lgb.Dataset(X_tr, np.log1p(target[train_rows]), categorical_feature=cat_idx, free_raw_data=True)
        dva = lgb.Dataset(X_va, np.log1p(target[val_rows]), reference=dtr, categorical_feature=cat_idx)
        return lgb.train(PARAMS, dtr, num_boost_round=8000, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(100, verbose=False)])

    def rollout(booster, origin_idx):
        work = sales_xt.copy()
        preds = np.zeros((HORIZON, S), dtype=np.float32)
        for k in range(1, HORIZON + 1):
            t_idx = origin_idx + k
            p = np.expm1(booster.predict(step_features(work, t_idx), num_iteration=booster.best_iteration))
            p = np.maximum(p, 0).astype(np.float32)
            work[t_idx, :] = p
            preds[k - 1] = p
        return preds

    results = {}
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    for wstart, lab in (WINDOWS if mode != "sub" else []):
        t0 = time.time()
        booster = train_booster(wstart)
        origin_idx = int(np.where(dates_xt == wstart - pd.Timedelta(days=1))[0][0])
        pred = rollout(booster, origin_idx)
        truth = X[np.asarray((dates >= wstart) & (dates <= wstart + pd.Timedelta(days=HORIZON - 1))), :]
        err = rmsle(truth, pred)
        results[lab] = err
        np.save(os.path.join(OUT, f"val_lgbm_v2_{lab}.npy"), pred)
        print(f"{lab}: v2 {err:.4f} | v1 {V1_REF[lab]:.4f} | timesfm {TF_REF[lab]:.4f} | "
              f"delta_v1 {err - V1_REF[lab]:+.4f} | best_iter {booster.best_iteration} | {time.time() - t0:.0f}s", flush=True)

    worse = [lab for lab in results if results[lab] >= V1_REF[lab]]
    print(f"windows not improved over v1: {worse if worse else 'none'}")
    if worse and mode != "sub":
        print("no submission written (pass 'sub' to override)")
        return

    print("=== training final model for submission ===")
    t0 = time.time()
    booster = train_booster(dates[-1] + pd.Timedelta(days=1))
    pred = rollout(booster, T - 1)
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
        sales[r] = pred[doff[row.date], pos[key]]
    if not_found:
        print(f"warning: {not_found} test rows unmatched")
    sub = pd.DataFrame({"id": test["id"].to_numpy(), "sales": sales})
    out_path = os.path.join(OUT, "sub_lgbm_v2.csv")
    sub.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(sub)} rows), total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
