from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
import json
import os
from typing import Optional
from collections.abc import Generator
from typing import Dict, Any
import numpy as np
from datasets import IterableDataset
from datasets import interleave_datasets

def get_full_dataset(dataset_dir: str):
    dataset_files = os.listdir(dataset_dir)
    ds={}
    for dataset_file in dataset_files:
        domain=dataset_file.split('.')[0]
        with open(os.path.join(dataset_dir, dataset_file), 'r') as f:
            dataset = json.load(f)
        ds[domain] = dataset
    return ds


def get_mixed_dataset(dataset_dir: str, proportion_dict: Optional[dict]=None):
    ds = get_full_dataset(dataset_dir)
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
    
    def domain_generator(domain: str, data: list) -> Generator[Dict[str, Any], None, None]:
        """单个领域的数据生成器"""
        for item in data:
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
    do_padding: bool = False,
    max_length: int = 1024
) -> callable:
    """
    创建用于批处理样本的数据整理函数。

    该整理器负责处理分词、填充、标签生成以及 DoReMi 训练批次中的领域 ID。

    参数：
        tokenizer: 用于编码文本的 HuggingFace 分词器。
        return_tensors: 返回张量的格式（'pt' 表示 PyTorch）。
        do_padding: 若为 True 则始终填充，否则仅在必要时填充。
        max_length: 用于填充对齐的最大序列长度。

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
                - labels: 语言模型标签（long 张量，填充位置为 -100）
                - domain_ids: 每个样本的领域 ID（long 张量，若存在）
        """
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