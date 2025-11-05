import dataloader
import models
import trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
import json
from pathlib import Path
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    DoReMi数据配比算法的训练流程
    
    功能：
    - 加载模型和参考模型
    - 初始化域权重缓冲区
    - 加载并处理数据（支持prompt/target分离的微调模式）
    - 初始化训练器并执行训练
    """
    # ========== 配置参数 ==========
    model_path = 'models/Qwen2.5-0.5B-Instruct'
    reference_model_path = 'models/reference_model_SFT_5000'
    dataset_path = 'data/KodCode_splited/BySubset'  # 数据集目录
    domain_config_path = 'configs/domain_config.json'  # 域配置文件路径
    output_dir = 'data/proportion_training'  # 输出目录
    domain_output_dir = 'data'  # 域权重输出目录
    
    # 训练参数
    max_length = 2048  # 最大序列长度
    per_device_batch_size = 1  # 每设备批次大小
    gradient_accumulation_steps = 4  # 梯度累积步数
    num_train_epochs = 3  # 训练轮数
    learning_rate = 5e-5  # 学习率
    warmup_ratio = 0.03  # 预热比例
    weight_decay = 0.01  # 权重衰减
    
    # DoReMi参数
    reweight_domains = True  # 是否启用域重加权
    reweight_eta = 1.0  # 域重加权学习率
    reweight_eps = 1e-4  # 域重加权平滑参数
    doremi_optimizer = 'doremiv1'  # DoReMi优化器类型
    
    # ========== 加载域配置 ==========
    if Path(domain_config_path).exists():
        with open(domain_config_path, 'r', encoding='utf-8') as f:
            domain_config = json.load(f)
    else:
        # 如果没有配置文件，创建默认配置
        logger.warning(f"域配置文件不存在: {domain_config_path}，使用默认配置")
        # 从数据集目录获取域列表
        dataset_files = [f for f in Path(dataset_path).iterdir() if f.is_file() and f.suffix == '.json']
        domain_names = [f.stem for f in dataset_files]
        num_domains = len(domain_names)
        domain_config = {
            'train_domain_weights': {domain: 1.0 / num_domains for domain in domain_names},
            'eval_domain_weights': {domain: 1.0 / num_domains for domain in domain_names}
        }
    
    train_domain_weights_dict = domain_config['train_domain_weights']
    # 将字典转换为数组时始终按键排序
    domain_list = list(sorted(train_domain_weights_dict.keys()))
    
    # ========== 加载分词器 ==========
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # ========== 加载数据集 ==========
    logger.info("加载数据集...")
    ds = dataloader.get_mixed_dataset(
        dataset_dir=dataset_path,
        domain_config=domain_config,
        tokenizer=tokenizer,
        is_finetune=True  # 微调模式：支持prompt/target分离
    )
    
    # ========== 加载模型 ==========
    logger.info(f"加载模型: {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    
    # 创建ProportionGPT2LMHeadModel
    # 注意：ProportionGPT2LMHeadModel继承自GPT2LMHeadModel
    # 如果模型不是GPT2架构，需要先加载base model，然后复制权重
    try:
        # 尝试使用from_pretrained（如果ProportionGPT2LMHeadModel支持）
        model = models.ProportionGPT2LMHeadModel.from_pretrained(
            model_path,
            config=config
        )
    except (AttributeError, TypeError):
        # 如果不支持，先加载base model，然后创建ProportionGPT2LMHeadModel并复制权重
        base_model = AutoModelForCausalLM.from_pretrained(model_path)
        model = models.ProportionGPT2LMHeadModel(config)
        # 加载预训练权重（忽略不匹配的层，如新添加的loss函数等）
        model.load_state_dict(base_model.state_dict(), strict=False)
        del base_model  # 释放内存
    
    # ========== 加载参考模型并初始化域权重 ==========
    if reweight_domains:
        logger.info(f"加载参考模型: {reference_model_path}")
        reference_model = AutoModelForCausalLM.from_pretrained(reference_model_path)
        # 冻结参考模型参数
        for param in reference_model.parameters():
            param.requires_grad = False
        reference_model.eval()
        
        # 设置参考模型
        model.reference_model = reference_model
        
        # 初始化域权重缓冲区
        total_domain_weight = sum(train_domain_weights_dict.values())
        model.register_buffer('train_domain_weights', torch.tensor(
            [train_domain_weights_dict[domain] / total_domain_weight for domain in domain_list]
        ))
        model.register_buffer('avg_domain_weights', model.train_domain_weights.clone())
        model.register_buffer('perdomain_scores', torch.ones(len(train_domain_weights_dict)) * np.log(len(tokenizer)))
        model.register_buffer('update_counter', torch.tensor(1))
        
        logger.info(f"初始化域权重: {model.train_domain_weights}")
    else:
        reference_model = None
    
    # ========== 创建训练参数 ==========
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy='no',  # 暂时不评估
        fp16=False,  # 根据GPU情况调整
        dataloader_num_workers=0,
        remove_unused_columns=False,
        # 添加DoReMi相关参数（通过自定义属性）
    )
    
    # 添加DoReMi相关属性到training_args
    training_args.domain_config_path = domain_config_path
    training_args.reweight_domains = reweight_domains
    training_args.reweight_eta = reweight_eta
    training_args.reweight_eps = reweight_eps
    training_args.doremi_optimizer = doremi_optimizer
    training_args.gradient_accumulation_steps = gradient_accumulation_steps
    
    # ========== 创建数据整理器 ==========
    data_collator = dataloader.get_data_collator(
        tokenizer=tokenizer,
        max_length=max_length,
        do_padding=True,
        is_finetune=True  # 微调模式：支持prompt/target分离
    )
    
    # ========== 初始化训练器 ==========
    logger.info("初始化训练器...")
    proportion_trainer = trainer.ProportionTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # ========== 训练 ==========
    logger.info("开始训练...")
    train_result = proportion_trainer.train()
    proportion_trainer.save_model()
    
    # ========== 保存域权重 ==========
    if reweight_domains:
        metrics = train_result.metrics
        avg_domain_weights_dict = {}
        for i in range(len(model.avg_domain_weights)):
            domain_name = domain_list[i]
            metrics[f'avg_domain_weight:{domain_name}'] = model.avg_domain_weights[i].item()
            avg_domain_weights_dict[domain_name] = model.avg_domain_weights[i].item()
        
        # 将平均领域权重保存为 JSON
        avg_domain_weights_file = Path(domain_output_dir) / 'avg_domain_weights.json'
        avg_domain_weights_file.parent.mkdir(parents=True, exist_ok=True)
        with open(avg_domain_weights_file, 'w', encoding='utf-8') as f:
            json.dump(avg_domain_weights_dict, f, indent=2)
        
        logger.info(f"平均域权重已保存到: {avg_domain_weights_file}")
        logger.info(f"最终域权重: {avg_domain_weights_dict}")
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main()