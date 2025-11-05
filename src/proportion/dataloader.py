from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
import json
import os
from typing import Optional, List
from collections.abc import Generator
from typing import Dict, Any
import numpy as np
import torch
from datasets import IterableDataset
from datasets import interleave_datasets
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def get_full_dataset(dataset_dir: str):
    dataset_files = os.listdir(dataset_dir)
    ds={}
    for dataset_file in dataset_files:
        domain=dataset_file.split('.')[0]
        with open(os.path.join(dataset_dir, dataset_file), 'r') as f:
            dataset = json.load(f)
        ds[domain] = dataset
    return ds


def get_mixed_dataset(
    dataset_dir: str, 
    proportion_dict: Optional[dict]=None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    domain_config: Optional[Dict[str, Any]] = None,
    is_finetune: bool = True
):
    """
    获取混合数据集，支持微调模式的prompt/target分离
    
    参数:
        dataset_dir: 数据集目录
        proportion_dict: 领域权重字典
        tokenizer: 分词器（微调模式必需）
        domain_config: 领域配置（包含train_domain_weights）
        is_finetune: 是否为微调模式（True=微调，False=预训练）
    """
    ds = get_full_dataset(dataset_dir)
    
    # 从domain_config获取proportion_dict
    if domain_config is not None and 'train_domain_weights' in domain_config:
        proportion_dict = domain_config['train_domain_weights']
    
    if proportion_dict is None:
        proportion_dict = {domain: 1/len(ds) for domain in ds}
    
    # 确保所有领域都在proportion_dict中
    for domain in ds.keys():
        if domain not in proportion_dict:
            proportion_dict[domain] = 0.0
    
    # 过滤掉比例为0的领域
    active_domains = [domain for domain in ds.keys() if proportion_dict.get(domain, 0) > 0]
    
    # 计算采样概率
    probabilities = np.array([proportion_dict[domain] for domain in active_domains])
    probabilities = probabilities / probabilities.sum()
    
    # 获取排序后的域列表（用于domain_id映射）
    domain_list = list(sorted(proportion_dict.keys()))
    domain_to_id = {domain: idx for idx, domain in enumerate(domain_list)}
    
    def domain_generator(domain: str, data: list) -> Generator[Dict[str, Any], None, None]:
        """单个领域的数据生成器"""
        domain_id = domain_to_id[domain]
        for item in data:
            if is_finetune and tokenizer is not None:
                # 微调模式：处理prompt和target
                if 'prompt' in item and 'target' in item:
                    prompt = item['prompt']
                    target = item['target']
                    # 分别编码prompt和target
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    target_ids = tokenizer.encode(target, add_special_tokens=False)
                    yield {
                        'prompt_ids': prompt_ids,
                        'target_ids': target_ids,
                        'domain_id': domain_id,
                        'domain': domain
                    }
                else:
                    # 如果没有prompt/target字段，尝试使用其他字段或跳过
                    continue
            else:
                # 预训练模式：直接返回原始数据
                item['domain_id'] = domain_id
                item['domain'] = domain
                yield item
    
    # 创建各领域的IterableDataset
    domain_datasets = []
    for domain in active_domains:
        domain_ds = IterableDataset.from_generator(
            domain_generator,
            gen_kwargs={'domain': domain, 'data': ds[domain]}
        )
        domain_datasets.append(domain_ds)
    # 交织数据集（按比例采样）
    mixed_ds = interleave_datasets(
        domain_datasets,
        probabilities=probabilities.tolist(),
        seed=42  # 固定种子以保证可重现性
    )
    
    return mixed_ds


def get_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    return_tensors: str = 'pt',
    do_padding: bool = True,
    max_length: int = 1024,
    is_finetune: bool = True
) -> callable:
    """
    创建用于批处理样本的数据整理函数。

    该整理器负责处理分词、填充、标签生成以及 DoReMi 训练批次中的领域 ID。
    支持微调模式（prompt/target分离）和预训练模式。

    参数：
        tokenizer: 用于编码文本的 HuggingFace 分词器。
        return_tensors: 返回张量的格式（'pt' 表示 PyTorch）。
        do_padding: 若为 True 则始终填充，否则仅在必要时填充。
        max_length: 用于填充对齐的最大序列长度。
        is_finetune: 是否为微调模式（True=微调，False=预训练）

    返回：
        callable: 接受特征列表并返回批次字典的数据整理函数。
    """
    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        将样本列表整理为一个批次。

        参数：
            features: 来自数据集的样本字典列表。

        返回：
            Dict[str, torch.Tensor]: 批次字典，包含以下键：
                - input_ids: 词 ID（long 张量）
                - attention_mask: 注意力掩码（long 张量）
                - labels: 语言模型标签（long 张量，prompt部分和填充位置为 -100）
                - domain_ids: 每个样本的领域 ID（long 张量，若存在）
        """
        # 检查是否为微调模式（有prompt_ids和target_ids）
        if is_finetune and 'prompt_ids' in features[0] and 'target_ids' in features[0]:
            # 微调模式：处理prompt和target分离
            input_ids_batch = []
            labels_batch = []
            attention_mask_batch = []
            domain_ids_batch = []

            for feat in features:
                prompt_ids = feat['prompt_ids']
                target_ids = feat['target_ids']
                
                # 拼接prompt和target
                input_ids = prompt_ids + target_ids
                
                # 截断到max_length
                if len(input_ids) > max_length:
                    # 优先保留prompt，如果prompt太长则截断
                    if len(prompt_ids) > max_length:
                        prompt_ids = prompt_ids[:max_length]
                        target_ids = []
                    else:
                        # 保留prompt，截断target
                        remaining_length = max_length - len(prompt_ids)
                        target_ids = target_ids[:remaining_length]
                    input_ids = prompt_ids + target_ids
                
                # 构建labels：prompt部分为-100，target部分为真实token
                labels = [-100] * len(prompt_ids) + target_ids
                labels = labels[:max_length]
                
                # attention mask
                attention_mask = [1] * len(input_ids)
                
                input_ids_batch.append(input_ids)
                labels_batch.append(labels)
                attention_mask_batch.append(attention_mask)
                
                # 处理domain_id
                if 'domain_id' in feat:
                    domain_ids_batch.append(feat['domain_id'])
            
            # 找到batch的最大长度进行padding
            max_len = max(len(x) for x in input_ids_batch)
            if do_padding:
                # 对齐到max_length
                max_len = max(max_len, max_length)
            
            # padding
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            for i in range(len(input_ids_batch)):
                pad_len = max_len - len(input_ids_batch[i])
                input_ids_batch[i] = input_ids_batch[i] + [pad_token_id] * pad_len
                attention_mask_batch[i] = attention_mask_batch[i] + [0] * pad_len
                labels_batch[i] = labels_batch[i] + [-100] * pad_len
            
            batch = {
                'input_ids': torch.tensor(input_ids_batch, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask_batch, dtype=torch.long),
                'labels': torch.tensor(labels_batch, dtype=torch.long),
            }
            
            if domain_ids_batch:
                batch['domain_ids'] = torch.tensor(domain_ids_batch, dtype=torch.long)
        else:
            # 预训练模式：使用原有逻辑
            if not do_padding:
                try:
                    # 尝试不做填充直接转换为张量
                    batch: Dict[str, torch.Tensor] = {
                            k: torch.tensor([f[k] for f in features])
                            for k in features[0].keys()
                            }
                except Exception:
                    # 无法直接转换时退回到填充方案
                    batch = tokenizer.pad(
                            [{k: v for k, v in f.items() if k not in {'domain_id', 'domain_ids'}} for f in features],
                            return_tensors=return_tensors,
                            pad_to_multiple_of=max_length)

                    if 'domain_id' in features[0]:
                        batch['domain_id'] = torch.tensor([f['domain_id'] for f in features])
                    elif 'domain_ids' in features[0]:
                        batch['domain_ids'] = torch.tensor([f['domain_ids'] for f in features])
            else:
                batch = tokenizer.pad(features, return_tensors=return_tensors, pad_to_multiple_of=max_length)

            # 确保 input_ids 为 long 类型张量
            batch['input_ids'] = batch['input_ids'].long()

            # 如果缺少注意力掩码则创建
            if 'attention_mask' not in batch:
                batch['attention_mask'] = torch.ones_like(batch['input_ids']).long()
            else:
                batch['attention_mask'] = batch['attention_mask'].long()

            # 移除特殊符号掩码（训练中不需要）
            batch.pop("special_tokens_mask", None)

            # 使用 input_ids 生成语言模型训练所需的标签
            if 'labels' not in batch:
                labels: torch.Tensor = batch['input_ids'].clone()
                batch["labels"] = labels

            # 将标签中的填充位置设置为 -100（损失计算中忽略）
            if tokenizer.pad_token_id is not None:
                batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100

        # 将 domain_id 字段统一转换为 domain_ids
        if 'domain_ids' not in batch and 'domain_id' in batch:
            batch['domain_ids'] = batch['domain_id']  # 兼容旧字段名
            batch.pop('domain_id')
        
        return batch
    return data_collator


class ProportionDataLoader(RandomlyCyclingMultiSourcesExamplesIterable):
    pass


if __name__ == "__main__":
    dataset = get_mixed_dataset(
        dataset_dir='data/KodCode_splited/BySubset',
        # proportion_dict={'code': 0.6, 'text': 0.3, 'math': 0.1}
        proportion_dict={'Package': 0, 'Algorithm': 0, 'Code': 0, 'Leetcode': 0, 'Filter': 0, 'Prefill': 0, 'Data': 0, 'Taco': 0, 'Docs': 0, 'Codeforces': 0, 'Evol': 0, 'Apps': 1}
    )

    # 迭代使用
    i=0
    domain_count={}
    for example in dataset:
        if i>100:
            break
        i+=1
        domain = example['question_id'].split('_')[0]
        if domain[:4]=='Docs':
            domain = 'Docs'
        if domain not in domain_count:
            domain_count[domain] = 0
        domain_count[domain]+=1
        print(example)
    print(domain_count)