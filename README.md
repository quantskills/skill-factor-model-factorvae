# FactorVAE Alpha Model

一个可独立安装的 FactorVAE 多因子合成模型。输入用户提供的多因子面板，使用概率动态因子模型训练股票横截面预测信号，并输出标准 `date,ticker,signal` 文件。

```bash
python -m pip install -e .
python examples/make_synthetic_panel.py --output examples/generated/synthetic_panel.csv
python scripts/run_model.py check --config examples/smoke.yaml
python scripts/run_model.py train --config examples/smoke.yaml
```

训练完成后，命令会打印运行目录。导出测试信号：

```bash
python scripts/run_model.py predict --run-dir outputs/smoke/<run_id> --split test
```

详细输入、输出和论文映射见 `references/`。

## 研究使用说明

**数据来源**：运行时由用户提供多因子面板和标签；仓库只提供合成样例。**假设与参数**：时间切分、标签、回看期、模型维度与训练参数均由 YAML 明确配置。**限制与风险边界**：输出是横截面排序信号，不包含组合构建、交易执行或收益保证。该项目仅用于研究与教育，不构成任何投资建议。
