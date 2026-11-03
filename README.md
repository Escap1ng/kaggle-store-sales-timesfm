# Kaggle Store Sales × TimesFM-3

English: [README.en.md](README.en.md)

本仓库保存 Kaggle「Store Sales - Time Series Forecasting」赛题的代码与实验结论。任务是对厄瓜多尔 Favorita 超市 54 家门店 × 33 个商品家族预测未来 16 天的日销售额，指标 RMSLE。

最终成绩：**公榜 0.38980**（TimesFM-3 零样本预测 70% + 递归 LightGBM 30% 的加权融合）。

## 成绩轨迹

| 方案 | 公榜 RMSLE |
|---|---|
| TimesFM-3 零样本，逐 (门店, 家族) 序列预测 | 0.39257 |
| TimesFM-3 + onpromotion 作为未来协变量 | 0.39563 |
| LightGBM，特征全部滞后 16 天 | 0.49587 |
| LightGBM，特征滞后 1 天 + 逐日回灌递归预测 | 0.47320 |
| **融合：TimesFM 70% + 递归 LightGBM 30%（log1p 空间加权）** | **0.38980** |

融合权重曲线：w=0（纯 TimesFM）0.39257 → w=0.3 **0.38980** → w=0.4 0.39450。该窗口最优点在 w≈0.3。

## 方法

### TimesFM-3 零样本
- 模型：Google Research TimesFM-3.0（330M 参数，权重 `google/timesfm-3.0-pytorch`，非商用许可）
- 把训练数据转成 1782 列宽表（54 门店 × 33 家族），每列作为一条日序列，喂全部历史（1688 天），单次预测 16 天
- 取 9 个输出分位的中位数作为点预测，负值截断为 0
- GPU 推理：`per_core_batch_size=64` 时单趟（1782 序列）约 40 秒；默认值 4 会让 GPU 与 CPU 同速

### 递归 LightGBM
- 特征 37 个：lag 1/2/3/7/14/28/56/364 天、7–140 天滚动均值/标准差（全部只用到预测原点及之前的数据）、日历（星期、月、年内日、年、是否发薪日、距发薪日）、门店元数据（type/cluster/city/state）、onpromotion、油价、全国/地方假日 flag
- 目标取 log1p，L2 回归；类别特征按门店/家族/城市等声明
- 训练与推理一致：预测时逐日回灌，即第 k 天预测作为第 k+1 天特征的输入，16 步滚动完成
- 早停验证集为窗口前 16 天

### 融合
- 在 log1p 空间按 w 加权：`expm1((1-w)·log1p(TF) + w·log1p(LGBM))`
- 权重由 4 个留出窗口（2017 年 5 月、7 月，2016/2015 年 8 月 16–31）扫描确定，再用公榜验证

## 已验证的结论

1. **TimesFM 的协变量通道在本赛题无效**。onpromotion（未来实值）使验证集从 0.4026 降到 0.3939，但公榜反而从 0.39257 恶化到 0.39563；oil 与假日 flag 在双窗口验证中零增益。堆积协变量的方向已放弃。
2. **内部归一化参数 `use_znorm` 完全无效果**（模型内置 RevIN 已覆盖）；`use_symmetric_averaging` 两窗口稳定改善约 0.001–0.0015，方向一致但小于噪声门槛，未采纳。
3. **单窗口验证会骗人**。LightGBM 在 2017 年 5/7 月窗口领先 TimesFM 约 0.009，公榜却落后 0.10。用 2015/2016 年同日历窗口（8 月 16–31）复核：这两个窗口 TimesFM 同样崩溃（0.66 / 0.77），LGBM 相对更好。结论是 8 月中下旬（Sierra 学区开学季 + 促销高峰）年际方差极大，任何单窗口调参都不可靠。
4. **LightGBM 的容量不是瓶颈**。轮数上限 3000→8000 后，四个窗口三劣一优，融合结果不变；早停始终顶到上限，说明内部 16 天验证集本身代表性不足。
5. **两个模型的误差互补**。四窗口权重扫描中 w=0.1–0.6 在每个窗口都优于纯 TimesFM；公榜也确认 w=0.3 有效、w=0.4 反噬。融合是本次唯一超过单模型的改动。

## 复现

```bash
# 1. 数据（需已配置 kaggle CLI 并接受赛题规则）
kaggle competitions download -c store-sales-time-series-forecasting -p data
cd data && unzip store-sales-time-series-forecasting.zip && cd ..

# 2. 环境（Python 3.13+，实测 Python 3.14.5）
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# 国内网络建议：export HF_ENDPOINT=https://hf-mirror.com

# 3. TimesFM-3 零样本基线（验证 + 提交）
python timesfm_v1.py all

# 4. 递归 LightGBM（四窗口验证；加 sub 直接出提交）
python lgbm_v2.py

# 5. 融合提交（w=0.3）
python blend_submission.py outputs/sub_timesfm_v1.csv outputs/sub_lgbm_v2.csv -w 0.3 -o outputs/sub_blend_tf70_lgbm30.csv
```

首次运行会从 Hugging Face 下载 TimesFM-3 权重（约 1.3GB）。

## 目录

| 文件 | 内容 |
|---|---|
| `timesfm_v1.py` | TimesFM-3 零样本基线；数据加载与宽表构建函数被其余脚本复用 |
| `timesfm_v2.py` | onpromotion 协变量实验 + 分位扫描 |
| `timesfm_v3.py` | oil/假日协变量实验（无前瞻用法，双窗口判据） |
| `timesfm_v4.py` | 推理旋钮扫描（对称平均、z-norm） |
| `lgbm_v1.py` | LightGBM 特征工程（滞后 16 天版） |
| `lgbm_v2.py` | 递归 LightGBM（逐日回灌） |
| `blend_eval.py` | 四窗口融合权重扫描 |
| `blend_submission.py` | 生成加权融合提交文件 |
| `diag_aug.py` | 8 月类窗口的年际差异诊断 |

`data/` 与 `outputs/` 不入库。

## 后续可以做的（未完成）

- 滚动多窗交叉验证体系（16 个窗口）——没有它，任何校准/调参都不可信
- 校准层：按家族选择分位、按家族×星期做乘性偏差修正
- LightGBM 特征强化：transactions、促销连续天数、假日前后窗口、门店×家族滚动统计

## 许可

本仓库代码仅用于赛事存档。TimesFM-3 模型权重遵循 Google 的 `timesfm-non-commercial-license-v1.0`，禁止商用；使用前请阅读模型页许可。
