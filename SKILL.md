---
name: factor-model-factorvae
description: Use when training a standalone cross-sectional stock model that synthesizes a user-supplied multi-factor panel into portable ranking signals with validation-based checkpoint selection.
quantSkills:
  schema_version: 2.0.0
  organization: quantskills
  organization_url: https://github.com/quantskills
  repository: skill-factor-model-factorvae
  repository_url: https://github.com/quantskills/skill-factor-model-factorvae
  project_type: skill
  license: GPL-3.0-only
  maintainer: wangxueliang
  collection: research-models
  catalog: {category: "06", subcategory: 06.statistical-ml-models}
  workflow: {primary_stage: modeling, workflow_stages: [modeling]}
  tags: [factor-model-factorvae, factor-model, alpha-model, signal-export]
  requires: []
  summary_zh: "使用 FactorVAE 概率动态因子模型将用户多因子面板合成为股票横截面预测信号。"
  summary_en: "Synthesizes user-supplied multi-factor panels into cross-sectional stock signals with FactorVAE."
  status: active
  validation_level: runnable
  maintainer_type: community
  platforms: [cursor, claude-code, codex, hermes, openclaw]
  interface:
    mode: structured
    envelope: {name: quantskills-envelope, version: 1.0.0}
    inputs:
      - {profile: factor-panel, version_range: ">=1.0.0 <2.0.0", required: true}
    outputs:
      - {profile: factor-signal, version: 1.0.0}
      - {profile: model-run-manifest, version: 1.0.0}
    adapters: [factor-model-training, factor-signal-export]
---

```json qsh-form
{
  "version": 1,
  "task": {"placeholder": "提供多因子面板、标签定义、时间切分与输出目录", "required": true},
  "fields": [
    {"key": "run_mode", "label": "运行模式", "type": "select", "options": [{"value": "train", "label": "训练"}, {"value": "predict", "label": "导出信号"}]}
  ],
  "prompt_template": "{{task}}\n\n使用 FactorVAE 执行 {{run_mode}}。先校验 YAML 配置与面板数据，再运行对应命令并检查输出清单。"
}
```

# FactorVAE Alpha Model

Use this skill to train a probabilistic dynamic factor model on a user-supplied
multi-factor panel. It is self-contained: it does not import local PandaAI
modules, other skills, or a bundled backtest engine.

## Workflow

1. Read [the input contract](references/input_contract.md) and prepare one
   CSV or Parquet panel with explicit `date`, `ticker`, feature, and label columns.
2. Start from `examples/smoke.yaml`, set paths, feature columns, label, and
   chronological train/validation/test splits.
   Read [the configuration reference](references/configuration.en.md) before changing model or training parameters.
3. Validate before training:

```bash
python scripts/run_model.py check --config config.yaml
```

4. Train and select the checkpoint with validation mean Pearson IC:

```bash
python scripts/run_model.py train --config config.yaml
```

5. Export the selected split as `date,ticker,signal`:

```bash
python scripts/run_model.py predict --run-dir runs/<run_id> --split test
python scripts/run_model.py predict --run-dir runs/<run_id> --split test --include-uncertainty
python scripts/run_model.py verify --run-dir runs/<run_id>
```

## Model Boundary

This repository is an independent, non-official FactorVAE reproduction and generic factor-panel adaptation.

The implementation follows the FactorVAE prior-posterior dynamic-factor
structure: historical features produce a prior factor distribution; future
labels are used only during training to form a posterior distribution; the
prediction API uses only historical features. Read
[paper mapping](references/paper_mapping.md) when comparing settings with the
original paper.

## Outputs

Every run writes a resolved configuration, input manifest, training metrics,
best checkpoint, validation summary, and exported signals. Read
[the output contract](references/output_contract.md) before integrating the
signals into another system.

## Research Use

**Data source:** users supply factor panels and labels at runtime; the repository contains only synthetic examples. **Assumptions and parameters:** splits, labels, lookback, model dimensions, and training settings are explicit YAML parameters. **Limitations and risk boundary:** output is a cross-sectional ranking signal, not portfolio construction, execution, or a return guarantee. This project is for research and education and does not constitute investment advice.
