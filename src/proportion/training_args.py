"""
DoReMi 训练脚本使用的训练参数。

该模块定义了模型、数据与训练配置所需的数据类。
所有参数均兼容 HuggingFace Transformers 的训练参数。
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any
from transformers import TrainingArguments, MODEL_FOR_CAUSAL_LM_MAPPING

MODEL_CONFIG_CLASSES: List[Any] = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
"""Transformers 支持的模型配置类列表。"""
MODEL_TYPES: Tuple[str, ...] = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
"""支持的模型类型元组。"""


@dataclass
class ModelArguments:
    """
    与将要微调或从零训练的模型/配置/分词器相关的参数。

    此数据类定义了 DoReMi 训练中加载或初始化语言模型所需的全部参数，
    同时支持使用预训练模型或从头训练。
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self) -> None:
        """
        初始化后校验参数组合是否有效。

        异常：
            ValueError: 当 config_overrides 与 config_name 或 model_name_or_path 同时使用时抛出。
        """
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    与用于训练和评估的数据相关的参数。

    此数据类涵盖 DoReMi 训练中涉及的数据集加载、预处理与数据增强的全部参数。
    """

    dataset_dir: str = field(
        default='.', metadata={"help": "Path to the dataset directory."}
    )
    dataset_name: str = field(
        default='pile', metadata={"help": "Name of the dataset."}
    )
    eval_dataset_dir: str = field(
        default=None, metadata={"help": "Path to the eval dataset directory. Defaults to dataset_dir"}
    )
    eval_dataset_name: str = field(
        default=None, metadata={"help": "Name of the eval dataset. Defaults to dataset_name."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_downstream_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For quicker downstream evaluation, limit the number of examples if set."
            )
        },
    )
    max_token_length: int = field(
        default=1024,
        metadata={
            "help": (
                "Input sequence length after tokenization. "
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    do_padding: bool = field(
        default=False, metadata={"help": "Pad the inputs."}
    )
    add_domain_id: bool = field(
        default=False, metadata={"help": "Add domain id to examples (when it's not already in the data)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    shuffle: bool = field(
        default=True, metadata={"help": "Shuffle the training data on the fly"}
    )
    keep_in_memory: bool = field(
        default=False, metadata={"help": "keep data in memory"}
    )


@dataclass
class FullTrainingArguments(TrainingArguments):
    """
    DoReMi 训练用的扩展训练参数。

    该类在 HuggingFace TrainingArguments 基础上新增 DoReMi 特有参数：
    - 领域重加权配置
    - 参考模型路径
    - 自定义学习率调度器
    - 下游评估设置

    同时继承 Transformers 的全部标准训练参数。
    """
    domain_config_path: str = field(
        default='.', metadata={"help": "Path to the domain config file."}
            )
    lr_end: float = field(
            default=1e-3,
            metadata={"help": "The final learning rate of the learning rate scheduler."},
    )
    reweight_domains: bool = field(
        default=False, metadata={"help": "Do reweighting."}
    )
    reweight_eta: float = field(
            default=1.0,
            metadata={"help": "Learning rate for reweighting."},
    )
    reweight_eps: float = field(
            default=1e-4,
            metadata={"help": "Smoothing parameter for reweighting."},
    )
    doremi_optimizer: str = field(
        default='doremiv1', metadata={"help": "Optimizer for DoReMi."}
    )
    reference_model_name_or_path: str = field(
        default='.', metadata={"help": "Path to the reference model."}
    )
    lr_scheduler_name: str = field(
        default=None, metadata={"help": "Custom LR scheduler name (linear_warmup_exponential, linear_warmup_cosine)"}
    )
    skip_perplexity_eval: bool = field(
        default=False, metadata={"help": "Don't evaluate perplexity."}
    )
    downstream_datasets: str = field(
            default=None, metadata={"help": "Comma-delimited list of dataset names from: {trivia_qa, web_questions, lambada, natural_questions, squad_v2}"}
    )
    eval_all_checkpoints: bool = field(
        default=False, metadata={"help": "Evaluate all the checkpoints at once."}
    )
    downstream_num_shots: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of in-context examples for downstream tasks. Defaults to 1"
            )
        },
    )
