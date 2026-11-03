# Kaggle Store Sales × TimesFM-3

Code and findings for the Kaggle competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting): forecast daily sales for 54 Favorita stores × 33 product families over a 16-day horizon. Metric: RMSLE.

Final score: **public LB 0.38980** — a log-space blend of TimesFM-3 zero-shot forecasts (70%) and a recursive LightGBM (30%).

中文版：[README.md](README.md)

## Score history

| Approach | Public LB RMSLE |
|---|---|
| TimesFM-3 zero-shot, per (store, family) series | 0.39257 |
| TimesFM-3 + onpromotion as a future covariate | 0.39563 |
| LightGBM, all features lagged 16 days | 0.49587 |
| LightGBM, lag-1 features with day-by-day recursive rollout | 0.47320 |
| **Blend: TimesFM 70% + recursive LightGBM 30% (log1p space)** | **0.38980** |

Blend weight curve: w=0 (pure TimesFM) 0.39257 → w=0.3 **0.38980** → w=0.4 0.39450. The optimum for this window sits at w≈0.3.

## Method

### TimesFM-3 zero-shot
- Model: Google Research TimesFM-3.0 (330M parameters, weights `google/timesfm-3.0-pytorch`, non-commercial license)
- Training data is reshaped into a 1782-column wide table (54 stores × 33 families); each column is one daily series fed its full history (1688 days) to produce a single 16-day forecast
- The median of 9 output quantiles is used as the point forecast; negatives are clipped to 0
- GPU inference: one pass over all 1782 series takes about 40 s with `per_core_batch_size=64`. The default of 4 leaves the GPU as slow as the CPU

### Recursive LightGBM
- 37 features: lags 1/2/3/7/14/28/56/364 days; 7–140 day rolling means and standard deviations (all computed from data at or before the forecast origin); calendar features (weekday, month, day of year, year, payday flag, distance to payday); store metadata (type, cluster, city, state); onpromotion; oil price; national and local holiday flags
- Target is log1p sales, L2 regression; store/family/city etc. are declared as categorical features
- Training and inference are consistent: forecasts are fed back day by day, i.e. the prediction for day k becomes an input feature for day k+1, rolling 16 steps forward
- Early-stopping validation set: the 16 days immediately before each window

### Blending
- Weighted average in log1p space: `expm1((1-w)·log1p(TF) + w·log1p(LGBM))`
- Weight was scanned on four held-out windows (May and July 2017, August 16–31 of 2016 and 2015), then verified on the public leaderboard

## Findings

1. **TimesFM's covariate channels do not help here.** onpromotion (future values known) improved validation from 0.4026 to 0.3939 but worsened the public score from 0.39257 to 0.39563; oil and holiday flags showed zero gain across two windows. Stacking covariates was abandoned.
2. **`use_znorm` has no effect at all** (the model's built-in RevIN already covers it); `use_symmetric_averaging` improves both windows by about 0.001–0.0015 consistently but below the noise threshold, so it was not adopted.
3. **Single-window validation is misleading.** LightGBM beat TimesFM by ~0.009 on the May/July 2017 windows but lost by 0.10 on the public leaderboard. Re-checking the same calendar slot in earlier years (Aug 16–31, 2015/2016): TimesFM collapses there too (0.66 / 0.77) while LGBM fares relatively better. Late August (Sierra school season plus promotion peaks) varies enormously year to year; any single-window tuning is unreliable.
4. **LightGBM capacity is not the bottleneck.** Raising the round cap from 3000 to 8000 made three of four windows worse and left the blend unchanged; early stopping always hit the cap, meaning the internal 16-day validation split is not representative.
5. **The two models' errors are complementary.** In the four-window weight scan, w=0.1–0.6 beats pure TimesFM on every window; on the public leaderboard w=0.3 helped and w=0.4 hurt. Blending was the only change in this project that beat a single model.

## Reproducing

```bash
# 1. Data (requires a configured kaggle CLI and accepted competition rules)
kaggle competitions download -c store-sales-time-series-forecasting -p data
cd data && unzip store-sales-time-series-forecasting.zip && cd ..

# 2. Environment (Python 3.13+; tested on 3.14.5)
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# In mainland China: export HF_ENDPOINT=https://hf-mirror.com

# 3. TimesFM-3 zero-shot baseline (validation + submission)
python timesfm_v1.py all

# 4. Recursive LightGBM (four-window validation; append "sub" to write a submission)
python lgbm_v2.py

# 5. Blended submission (w=0.3)
python blend_submission.py outputs/sub_timesfm_v1.csv outputs/sub_lgbm_v2.csv -w 0.3 -o outputs/sub_blend_tf70_lgbm30.csv
```

The first run downloads the TimesFM-3 weights from Hugging Face (about 1.3 GB).

## Layout

| File | Contents |
|---|---|
| `timesfm_v1.py` | TimesFM-3 zero-shot baseline; its data-loading and wide-table helpers are reused by the other scripts |
| `timesfm_v2.py` | onpromotion covariate experiment and quantile sweep |
| `timesfm_v3.py` | oil/holiday covariate experiment (no look-ahead, dual-window acceptance rule) |
| `timesfm_v4.py` | inference knob sweep (symmetric averaging, z-norm) |
| `lgbm_v1.py` | LightGBM feature engineering (16-day-lag variant) |
| `lgbm_v2.py` | recursive LightGBM (day-by-day rollout) |
| `blend_eval.py` | four-window blend weight sweep |
| `blend_submission.py` | build a blended submission file |
| `diag_aug.py` | year-over-year August window diagnostic |

`data/` and `outputs/` are not committed.

## Possible next steps (not done)

- Rolling multi-window cross-validation (16 windows) — without it, any calibration or tuning remains unreliable
- Calibration layer: per-family quantile choice, per-family × weekday multiplicative bias correction
- More LightGBM features: transactions, consecutive promotion days, holiday windows, store × family rolling statistics

## License

This repository exists as a competition archive. The TimesFM-3 model weights are released under Google's `timesfm-non-commercial-license-v1.0` and may not be used commercially; read the model card before use.
