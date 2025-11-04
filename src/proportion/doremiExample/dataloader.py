"""
DoReMi（通过极小极大优化进行领域重加权）的数据加载工具。

该模块提供支持在训练过程中动态更新领域权重的可迭代数据集，
使大语言模型训练能够采用自适应的数据混合策略。
"""
from pathlib import Path
from collections import Counter
from collections.abc import Iterator
import random
from copy import deepcopy
from typing import Optional, List, Dict, Any, Generator
import numpy as np
import torch
from datasets import IterableDataset
from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
from datasets.utils.logging import get_logger
from datasets import load_from_disk
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = get_logger(__name__)


RANDOM_BATCH_SIZE: int = 8192
"""抽取领域索引时使用的随机采样批大小。"""
DEFAULT_SEED: int = 111
"""数据集操作默认使用的随机种子。"""


class UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
        RandomlyCyclingMultiSourcesExamplesIterable):
    """
    在 RandomlyCyclingMultiSourcesExamplesIterable 的基础上加入动态领域权重更新能力。

    该类支持在训练期间实时调整采样概率，使 DoReMi 可以根据模型表现自适应地重新加权各领域。

    属性：
        probabilities_handle: 指向模型参数中领域权重缓冲区的句柄（训练中会更新）。
        probabilities: 用于领域采样的静态概率向量（在没有句柄时使用）。
    """

    def __init__(
        self,
        ex_iterables: List[Any],
        generator: np.random.Generator,
        probabilities: Optional[np.ndarray] = None,
        probabilities_handle: Optional[torch.Tensor] = None,
        stopping_strategy: str = "all_exhausted"
    ) -> None:
        """
        按给定的领域采样配置初始化可更新的迭代器。

        参数：
            ex_iterables: 示例迭代器列表，每个领域一个。
            generator: 用于采样的 NumPy 随机数生成器。
            probabilities: 可选的领域静态概率向量，若为 None 则使用均匀分布。
            probabilities_handle: 可选的领域权重缓冲区句柄，用于动态更新。
            stopping_strategy: 迭代终止策略（'all_exhausted' 或 'first_exhausted'）。
        """
        super().__init__(ex_iterables, generator, stopping_strategy=stopping_strategy)
        self.probabilities_handle: Optional[torch.Tensor] = probabilities_handle
        self.probabilities: Optional[np.ndarray] = probabilities

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator,
        num_sources: int,
        probabilities_handle: Optional[torch.Tensor] = None,
        probabilities: Optional[np.ndarray] = None,
        random_batch_size: int = RANDOM_BATCH_SIZE
    ) -> Iterator[int]:
        """
        根据当前概率生成随机的领域索引。

        参数：
            rng: NumPy 随机数生成器。
            num_sources: 领域（数据源）数量。
            probabilities_handle: 领域权重缓冲区句柄（每个批次都会检查）。
            probabilities: 静态概率向量（在没有句柄时使用）。
            random_batch_size: 每个批次生成的索引数量。

        产出：
            int: 按照当前概率采样得到的领域索引。
        """
        while True:
            # 如果可行，从模型缓冲区读取领域权重
            if probabilities_handle is not None:
                probabilities = probabilities_handle.detach().cpu().numpy()

            yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=probabilities))

    def _give_indice_iterator(self) -> Iterator[int]:
        """
        使用当前概率创建新的领域索引迭代器。

        返回：
            Iterator[int]: 按当前采样权重产出领域索引的迭代器。
        """
        rng = deepcopy(self.generator)
        return self._iter_random_indices(
            rng,
            len(self.ex_iterables),
            probabilities_handle=self.probabilities_handle,
            probabilities=self.probabilities
        )

    def shard_data_sources(self, shard_indices: Any) -> 'UpdatableRandomlyCyclingMultiSourcesExamplesIterable':
        """
        分片支持的占位实现（当前未实现实际逻辑）。

        参数：
            shard_indices: 分片索引（未使用）。

        返回：
            self: 原样返回自身。
        """
        return self

    @property
    def n_shards(self) -> int:
        """
        获取分片数量。

        返回：
            int: 始终返回 1（仅单个分片）。
        """
        return 1

    def shuffle_data_sources(self, seed: int) -> 'UpdatableRandomlyCyclingMultiSourcesExamplesIterable':
        """
        打乱各领域内部的数据源顺序。

        参数：
            seed: 用于打乱的随机种子。

        返回：
            self: 返回自身以便链式调用。
        """
        self.ex_iterables = [ex_iterable.shuffle_data_sources(seed) for ex_iterable in self.ex_iterables]
        return self


def interleave_datasets(
    datasets: List[Any],
    probabilities: Optional[np.ndarray] = None,
    probabilities_handle: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    stopping_strategy: str = 'all_exhausted'
) -> IterableDataset:
    """
    按领域权重交织多个数据集。

    该函数将多个领域数据集合并为单个可迭代数据集，
    按给定概率从不同领域采样，既支持静态概率向量，也支持动态更新的概率。

    参数：
        datasets: 待交织的数据集列表，每个领域一个。
        probabilities: 可选的领域静态概率，需满足和为 1。
        probabilities_handle: 可选的领域权重缓冲区句柄，用于动态更新。
        seed: 交织时使用的随机种子，若为 None 则使用默认种子。
        stopping_strategy: 终止策略（'all_exhausted' 或 'first_exhausted'）。

    返回：
        IterableDataset: 按领域权重采样的交织数据集。
    """
    iterable_datasets: List[IterableDataset] = []
    for dataset in datasets:
        if not isinstance(dataset, IterableDataset):
            iterable_datasets.append(dataset.to_iterable_dataset())
        else:
            iterable_datasets.append(dataset)

    ex_iterables = [d._ex_iterable for d in iterable_datasets]

    generator = np.random.default_rng(seed)
    ex_iterable = UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables, generator=generator,
            probabilities=probabilities, probabilities_handle=probabilities_handle,
            stopping_strategy=stopping_strategy)

    return IterableDataset(ex_iterable=ex_iterable)


def simulate_data_skip_per_domain(
    num_skip_examples: int,
    probabilities: np.ndarray,
    rng: np.random.Generator,
    random_batch_size: int = RANDOM_BATCH_SIZE
) -> List[int]:
    """
    模拟恢复训练时需要跳过哪些领域的样本。

    该函数根据领域概率确定每个领域需跳过的样本数，
    以确保恢复训练后仍保持相同的数据分布。

    参数：
        num_skip_examples: 需要跳过的样本总数。
        probabilities: 各领域的概率向量（需满足和为 1）。
        rng: NumPy 随机数生成器。
        random_batch_size: 随机采样的批大小。

    返回：
        List[int]: 每个领域对应的跳过数量（按领域 ID 索引）。
    """
    num_sources: int = len(probabilities)
    sampled_domain_idxs: List[np.ndarray] = [
        rng.choice(num_sources, size=random_batch_size, p=probabilities)
        for _ in range(num_skip_examples // random_batch_size + 1)]
    sampled_domain_idxs = np.concatenate(sampled_domain_idxs)[:num_skip_examples]
    counts: Counter = Counter(sampled_domain_idxs)
    return [counts.get(i, 0) for i in range(num_sources)]


def determine_skip_per_domain(
    num_skip_examples: int,
    seed: Optional[int],
    domain_weights: Optional[np.ndarray],
    domain_names: Optional[List[str]]
) -> Dict[str, int]:
    """
    恢复训练时，确定每个领域应跳过的样本数量。

    参数：
        num_skip_examples: 需要跳过的样本总数。
        seed: 用于跳过模拟的随机种子。
        domain_weights: 各领域的概率向量。
        domain_names: 领域名称列表（需与 domain_weights 对应顺序一致）。

    返回：
        Dict[str, int]: 领域名称到跳过样本数的映射。

    异常：
        ValueError: 当 num_skip_examples > 0 但必要参数为 None 时抛出。
    """
    if num_skip_examples == 0:
        return {name: 0 for name in domain_names} if domain_names else {}

    if domain_weights is None or seed is None or domain_names is None:
        raise ValueError("If num_skip_examples > 0 then domain_weights, domain_names, and seed must not be None")

    rng: np.random.Generator = np.random.default_rng(seed)
    skip_per_domain: List[int] = simulate_data_skip_per_domain(num_skip_examples, domain_weights, rng)
    domain_name_to_skip_num: Dict[str, int] = {name: num for name, num in zip(domain_names, skip_per_domain)}
    return domain_name_to_skip_num


def skippable_data_gen(
    shards: List[Path],
    num_skip_examples: int = 0,
    loop: bool = True,
    seed: int = 111,
    shuffle: bool = False,
    keep_in_memory: bool = False
) -> Generator[Dict[str, Any], None, None]:
    """
    从分片中生成样本，并支持跳过部分样本（用于恢复训练）。

    该生成器从数据集分片中产出样本，可在开始时跳过指定数量的样本，
    便于从检查点恢复训练时保持进度。

    参数：
        shards: 数据集分片目录路径列表。
        num_skip_examples: 在产出之前需要跳过的样本数量（用于恢复）。
        loop: 若为 True，则无限循环遍历分片；否则只遍历一次。
        seed: 用于打乱的随机种子。
        shuffle: 是否在产出样本前打乱分片。
        keep_in_memory: 是否将分片保存在内存中（基于哈希的启发式策略）。

    产出：
        Dict[str, Any]: 来自数据集分片的样本字典。
    """

    def get_shard_ds(
        shard_dir: Path,
        num_skipped: int,
        seed: int,
        shuffle: bool
    ) -> tuple[Any, int]:
        """
        加载单个分片并应用跳过/打乱逻辑。

        参数：
            shard_dir: 分片目录路径。
            num_skipped: 当前已跳过的样本数量。
            seed: 用于打乱的随机种子。
            shuffle: 是否对分片执行打乱。

        返回：
            tuple: (加载后的分片数据集, 更新后的已跳过数量)。
        """
        # TODO 临时方案：基于分片路径哈希的启发式判断是否驻留内存
        if keep_in_memory:
            curr_keep_in_memory = (hash(str(shard_dir)) % 2 == 0)
        else:
            curr_keep_in_memory = False

        shard = load_from_disk(dataset_path=str(shard_dir), keep_in_memory=curr_keep_in_memory)
        if shuffle:
            shard = shard.shuffle(seed=seed)
        if num_skipped < num_skip_examples:
            # 尝试跳过样本
            if len(shard) < (num_skip_examples - num_skipped):
                num_skipped += len(shard)
            else:
                shard = shard.select(range(num_skip_examples - num_skipped, len(shard)))
                logger.info(f"Skipped {num_skip_examples} examples in {shard_dir}")
                num_skipped = num_skip_examples
        return shard, num_skipped

    num_skipped: int = 0
    if loop:
        while True:
            for shard_dir in shards:
                shard, num_skipped = get_shard_ds(shard_dir, num_skipped, seed, shuffle)
                if num_skipped < num_skip_examples:
                    continue

                for ex in shard:
                    yield ex
                seed += 1
    else:
        for shard_dir in shards:
            shard, num_skipped = get_shard_ds(shard_dir, num_skipped, seed, shuffle)
            if num_skipped < num_skip_examples:
                continue

            for ex in shard:
                yield ex
            seed += 1


def get_pile_datasets(
        preprocessed_dir: str,
        cache_dir: Optional[str] = None,
        split: str = 'train',
        seed: int = DEFAULT_SEED,
        domain_weights: Optional[np.ndarray] = None,
        domain_names: Optional[List[str]] = None,
        num_skip_examples: int = 0,
        shuffle: bool = False,
        shard_reversal: bool = False,
        keep_in_memory: bool = False
) -> Dict[str, IterableDataset]:
    """
    以领域为单位加载符合 PILE 格式的数据集。

    PILE 目录结构要求如下：
    - preprocessed_dir/split/domain_name/shard_1/
    - preprocessed_dir/split/domain_name/shard_2/
    - ...

    参数：
        preprocessed_dir: 含预处理数据切分的根目录。
        cache_dir: 可选的缓存目录（为兼容保留，实参未使用）。
        split: 数据划分（'train' 或 'validation'）。
        seed: 用于打乱分片的随机种子。
        domain_weights: 可选的领域概率向量（用于跳过样本计算）。
        domain_names: 可选的领域名称列表（用于跳过样本计算）。
        num_skip_examples: 每个领域需跳过的样本数（用于恢复训练）。
        shuffle: 是否在分片内部打乱样本。
        shard_reversal: 若为 True，则反转分片顺序以优先处理未见样本。
        keep_in_memory: 是否将分片保存在内存中（启发式判断）。

    返回：
        Dict[str, IterableDataset]: 领域名称到可迭代数据集的映射。
    """
    domain_name_to_skip_num: Dict[str, int] = determine_skip_per_domain(
        num_skip_examples, seed, domain_weights, domain_names
    )

    preprocessed_dir = Path(preprocessed_dir) / split

    all_ds: Dict[str, IterableDataset] = {}
    for domain_dir in preprocessed_dir.iterdir():
        if split == 'train':
            shards: List[Path] = list(domain_dir.iterdir())
            random.Random(seed).shuffle(shards)
            if shard_reversal:
                shards = list(reversed(shards))
        else:
            shards = [domain_dir]
        ds: IterableDataset = IterableDataset.from_generator(
                skippable_data_gen,
                gen_kwargs={'shards': shards,
                            'num_skip_examples': domain_name_to_skip_num[domain_dir.name],
                            'loop': (split == 'train'),
                            'seed': seed,
                            'shuffle': shuffle}
                )
        all_ds[domain_dir.name] = ds
        seed += 1
    return all_ds


def get_perdomain_datasets(
        preprocessed_dir: str,
        domain_weights_dict: Dict[str, float],
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
        seed: int = DEFAULT_SEED,
        domain_weights: Optional[np.ndarray] = None,
        domain_names: Optional[List[str]] = None,
        num_skip_examples: int = 0,
        shuffle: bool = False,
        shard_reversal: bool = False,
        keep_in_memory: bool = False
) -> Dict[str, IterableDataset]:
    """
    Load per-domain datasets from preprocessed directory.
    
    Returns a dictionary mapping domain names to IterableDataset objects.
    Supports flexible directory structures: either a single dataset file per domain
    or multiple shards per domain.
    
    参数：
        preprocessed_dir: Root directory containing domain subdirectories.
        domain_weights_dict: Dictionary mapping domain names to their sampling weights.
        cache_dir: Optional cache directory (unused, kept for compatibility).
        split: Optional dataset split name (if None, uses root directory).
        seed: Random seed for shuffling shards.
        domain_weights: Optional probability vector for domain sampling (for skip calculation).
        domain_names: Optional list of domain names (for skip calculation).
        num_skip_examples: Number of examples to skip per domain (for checkpoint resumption).
        shuffle: Whether to shuffle examples within each shard.
        shard_reversal: If True, reverse shard order to prioritize unseen examples.
        keep_in_memory: Whether to keep shards in memory.
        
    返回：
        Dict[str, IterableDataset]: Mapping from domain name to iterable dataset.
    """
    domain_name_to_skip_num: Dict[str, int] = determine_skip_per_domain(
        num_skip_examples, seed, domain_weights, domain_names
    )

    preprocessed_dir = Path(preprocessed_dir)
    if split is not None and (preprocessed_dir / split).exists():
        preprocessed_dir = preprocessed_dir / split
    else:
        logger.warn("No split used or split directory not found: using same data for all splits.")

    domains: List[str] = list(sorted(domain_weights_dict.keys()))

    all_ds: Dict[str, IterableDataset] = {}
    for domain in domains:
        domain_dir: Path = preprocessed_dir / domain

        if (domain_dir / 'dataset_info.json').exists():
            curr_shards: List[Path] = [domain_dir]
        else:
            curr_shards = list(domain_dir.iterdir())
            # 打乱分片的顺序
            random.Random(seed).shuffle(curr_shards)
            if shard_reversal:
                curr_shards = list(reversed(curr_shards))

        ds: IterableDataset = IterableDataset.from_generator(
                skippable_data_gen,
                gen_kwargs={'shards': curr_shards,
                            'num_skip_examples': domain_name_to_skip_num[domain],
                            'loop': (split == 'train'),
                            'seed': seed,
                            'shuffle': shuffle,
                            'keep_in_memory': keep_in_memory}
                )
        all_ds[domain] = ds
        seed += 1
    return all_ds


def get_preprocessed_mixed_dataset(
        preprocessed_dir: str,
        domain_weights_dict: Dict[str, float],
        dataset_name: str = 'pile',
        cache_dir: Optional[str] = None,
        split: str = 'train',
        seed: int = DEFAULT_SEED,
        max_samples: Optional[int] = None,
        add_domain_id: bool = False,
        domain_weight_buffer_handle: Optional[torch.Tensor] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        no_interleave: bool = False,
        shuffle: bool = False,
        num_skip_examples: int = 0,
        shard_reversal: bool = False,
        keep_in_memory: bool = False
) -> IterableDataset:
    """
    从预处理后的领域数据集中创建混合数据集。

    这是加载并交织领域数据集的主要函数。preprocessed_dir 需满足以下结构：
    - 第一级：领域目录
    - 第二级：每个领域的分片（各领域分片数量应一致）

    参数：
        preprocessed_dir: 包含领域分片子目录的根目录。
        domain_weights_dict: 领域名称到采样权重的映射。
        dataset_name: 数据集格式名称（'pile' 或自定义，新格式需在此函数中扩展）。
        cache_dir: Arrow 文件的缓存目录（若需要）。
        split: 数据划分（'train' 或 'validation'）。
        seed: 控制数据分片顺序的随机种子。
        max_samples: 可选的样本数量上限。
        add_domain_id: 若为 True，将在生成时为样本添加领域 ID。
        domain_weight_buffer_handle: 模型参数中领域权重缓冲区的句柄，用于动态更新。
        tokenizer: HuggingFace 分词器（用于设置 pad token）。
        no_interleave: 若为 True，则不交织领域，按顺序遍历数据集。
        shuffle: 若为 True，则使用大小为 100k 的缓冲区进行在线打乱。
        num_skip_examples: 迭代器中需跳过的样本数量（用于恢复训练）。
        shard_reversal: 若为 True，则反转分片顺序以优先处理未见样本。
        keep_in_memory: 是否将分片保存在内存中。

    返回：
        IterableDataset: 按领域权重采样的交织数据集。

    异常：
        ValueError: 当 dataset_name 不受支持时抛出。
    """
    domain_names: List[str] = list(sorted(domain_weights_dict.keys()))
    domain_to_idx: Dict[str, int] = {domain_names[i]: i for i in range(len(domain_names))}
    domain_weights: np.ndarray = np.asarray([domain_weights_dict[domain_name] for domain_name in domain_names])
    domain_weights = domain_weights / domain_weights.sum()

    probabilities: np.ndarray = domain_weights

    # TODO：通过 get_perdomain_datasets 方式加载 pile
    if dataset_name == 'pile':
        all_ds: Dict[str, IterableDataset] = get_pile_datasets(
                preprocessed_dir,
                cache_dir=cache_dir,
                split=split,
                seed=seed,
                domain_weights=domain_weights,
                domain_names=domain_names,
                num_skip_examples=num_skip_examples,
                shuffle=shuffle,
                shard_reversal=shard_reversal,
                keep_in_memory=keep_in_memory)
    else:
        try:
            all_ds = get_perdomain_datasets(
                preprocessed_dir, 
                domain_weights_dict,
                cache_dir=cache_dir,
                split=split,
                seed=seed,
                domain_weights=domain_weights,
                domain_names=domain_names,
                num_skip_examples=num_skip_examples,
                shuffle=shuffle,
                shard_reversal=shard_reversal,
                keep_in_memory=keep_in_memory)
        except Exception:
            raise ValueError(f"dataset_name {dataset_name} not implemented.")

    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def add_domain_id_generator(ds: IterableDataset, domain_idx: int) -> Generator[Dict[str, Any], None, None]:
        """
        为每个样本添加 domain_id 字段。

        参数：
            ds: 输入的可迭代数据集。
            domain_idx: 需添加到样本中的领域索引。

        产出：
            Dict[str, Any]: 添加了 domain_id 字段的样本。
        """
        for ex in ds:
            ex['domain_id'] = domain_idx
            yield ex

    domain_ds_ls: List[IterableDataset] = []
    for domain_name in domain_names:
        domain_idx: int = domain_to_idx[domain_name]
        domain_ds: IterableDataset = all_ds[domain_name]
        # 如有需要，补充 domain_id 字段
        if add_domain_id:
            domain_ds = IterableDataset.from_generator(
                add_domain_id_generator,
                gen_kwargs={'ds': domain_ds, 'domain_idx': domain_idx}
            )
        domain_ds_ls.append(domain_ds)

    if no_interleave:
        # 不进行交织，直接按顺序遍历各个数据集
        def data_generator(shards: List[IterableDataset]) -> Generator[Dict[str, Any], None, None]:
            """
            按顺序遍历各数据集的数据生成器。

            参数：
                shards: 领域数据集列表。

            产出：
                Dict[str, Any]: 顺序产出的样本。
            """
            for shard in shards:
                for ex in shard:
                    yield ex
        ds: IterableDataset = IterableDataset.from_generator(
            data_generator, gen_kwargs={'shards': domain_ds_ls}
        )
        logger.info("Not interleaving dataset - will not sample according to domain weights")

    else:
        ds = interleave_datasets(
                domain_ds_ls,
                probabilities=probabilities,
                probabilities_handle=domain_weight_buffer_handle,
                seed=seed)

    def take_data_generator(ds: IterableDataset, max_samples: Optional[int]) -> Generator[Dict[str, Any], None, None]:
        """
        限制产出样本数量的数据生成器。

        参数：
            ds: 输入的可迭代数据集。
            max_samples: 允许产出的最大样本数（None 表示不限制）。

        产出：
            Dict[str, Any]: 不超过 max_samples 的样本。
        """
        idx: int = 0
        for ex in ds:
            yield ex
            idx += 1
            if max_samples is not None and idx >= max_samples:
                return

    ds = IterableDataset.from_generator(
        take_data_generator, gen_kwargs={'ds': ds, 'max_samples': max_samples}
    )
    return ds


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
