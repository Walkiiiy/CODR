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

class ProportionDataLoader(RandomlyCyclingMultiSourcesExamplesIterable):
    pass


if __name__ == "__main__":
    dataset = get_mixed_dataset(
        dataset_dir='../data/proportion/KodCode_splited/BySubset',
        # proportion_dict={'code': 0.6, 'text': 0.3, 'math': 0.1}
    )

    # 迭代使用
    for example in dataset:
        print(example)