# 配置参数说明

配置文件为 YAML。未填写的字段使用下表默认值；所有路径相对于配置文件所在目录解析。

## 数据与时间切分

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `data.panel_path` | 必填 | CSV 或 Parquet 多因子长表路径。 |
| `data.date_column` | 必填 | 整数日期列，格式为 `YYYYMMDD`。 |
| `data.ticker_column` | 必填 | 标的代码列。 |
| `data.label_column` | 必填 | 训练与验证标签列。 |
| `data.feature_columns` | 必填 | 用于建模的数值因子列列表。 |
| `splits.train` | 必填 | `[开始日期, 结束日期]` 训练区间。 |
| `splits.valid` | 必填 | 验证区间。 |
| `splits.test` | 必填 | 信号导出区间。三段区间按时间顺序且不重叠。 |

## 预处理

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `preprocessing.lookback` | `20` | 每次预测使用的历史交易日数量，最小值为 `2`。 |
| `preprocessing.standardize_features` | `true` | 按日期对每个因子做横截面标准化。 |
| `preprocessing.standardize_labels` | `true` | 按日期对标签做横截面标准化。 |

## 模型

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `model.hidden_size` | `64` | GRU 特征编码器与内部网络的隐藏维度。 |
| `model.latent_factors` | `8` | 潜在因子数量。 |
| `model.portfolios` | `16` | 动态组合层的组合数量。 |
| `model.dropout` | `0.1` | 特征编码器 dropout 比例。 |
| `model.kl_weight` | `0.01` | 后验与先验因子分布 KL 项的最终权重。 |
| `model.kl_warmup_epochs` | `10` | KL 权重从零递增到最终权重的 epoch 数。 |

## 训练与输出

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `training.epochs` | `20` | 训练 epoch 数。 |
| `training.learning_rate` | `0.001` | AdamW 学习率。 |
| `training.weight_decay` | `0.0001` | AdamW 权重衰减。 |
| `training.seed` | `42` | 随机种子。 |
| `training.device` | `cpu` | `cpu`、`cuda` 或 `auto`。`auto` 优先使用可用 CUDA。 |
| `training.selection_metric` | `mean_pearson` | 验证集 checkpoint 选择指标；当前版本固定为验证集平均 Pearson IC。 |
| `output_root` | `runs` | 运行目录根路径。 |
| `run_name` | 自动时间戳 | 可选固定运行目录名；同名目录已存在时运行会停止。 |

## 导出不确定度

默认命令导出三列标准信号：

```bash
python scripts/run_model.py predict --run-dir <run_dir> --split test
```

加入 `--include-uncertainty` 后，另行写入 `signals_test_with_uncertainty.parquet`，其中增加 `prediction_std` 列：

```bash
python scripts/run_model.py predict   --run-dir <run_dir> --split test --include-uncertainty
```
