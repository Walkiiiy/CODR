# 配置说明

该目录提供在本项目中启动训练时常用的配置示例。所有路径均相对于项目根目录，可根据需要自行修改。

## 1. 领域权重配置 (`domain_weights_example.json`)

`train.py` 与 `train_modern.py` 都依赖一个包含训练/评估领域权重的 JSON 文件，其格式需满足：

```json
{
  "train_domain_weights": {
    "domain_a": 0.6,
    "domain_b": 0.4
  },
  "eval_domain_weights": {
    "domain_a": 0.5,
    "domain_b": 0.5
  }
}
```

* `train_domain_weights`：训练时各领域的初始采样权重。
* `eval_domain_weights`：评估阶段各领域的采样权重。

请确保权重字典中的键与预处理数据集中领域目录名称一致。若使用更多领域，可在字典中继续添加键值对，并保证权重归一化（脚本会自动按比例规范化）。

## 2. 训练参数示例 (`train_example.json`)

`train_example.json` 展示了如何通过 JSON 文件一次性传入 `ModelArguments`、`DataTrainingArguments` 与 `FullTrainingArguments`。你可以复制后按需调整模型、数据与训练超参，然后执行：

```bash
python -m src.proportion.train_modern configs/train_example.json
```

该命令会：

1. 使用 Hugging Face Hub 上的 `distilgpt2` 作为初始权重并复用其分词器；
2. 读取 `data/sample_corpus` 下的预处理语料，数据应组织为 `domain_a`、`domain_b` 两个子目录，每个子目录再包含一组或多组 Hugging Face `datasets` 格式的分片；
3. 将训练输出写入 `outputs/example-run`；
4. 在训练结束时保存平均领域权重。

若要启用 DoReMi 领域重加权，请将 `reweight_domains` 设为 `true` 并提供 `reference_model_name_or_path`。当不需要重加权时，可保持当前默认。

## 3. 数据目录结构示例

无论是训练集还是验证集，预处理后的数据目录需遵循以下层级：

```
data/sample_corpus/
├── train/
│   ├── domain_a/
│   │   ├── shard_00000/
│   │   └── shard_00001/
│   └── domain_b/
│       └── shard_00000/
└── validation/
    ├── domain_a/
    │   └── shard_00000/
    └── domain_b/
        └── shard_00000/
```

其中 `shard_xxxxx` 可以是由 `datasets.Dataset.save_to_disk` 生成的 Arrow 数据集目录，或任何可被 `datasets.load_from_disk` 正确加载的路径。

## 4. 自定义技巧

- 如果需要在命令行而非 JSON 中传参，可直接使用 `python -m src.proportion.train_modern --help` 查看完整参数列表；
- 评估数据目录默认与训练数据目录一致，如需单独指定，请设置 `eval_dataset_dir` 与 `eval_dataset_name`；
- 若想将训练迁移到 `src/proportion/train.py`（闪存注意力版），可复用相同的配置文件，仅需确认模型类型与硬件支持匹配。
