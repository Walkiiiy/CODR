#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
微调reference模型，用于计算超额损失

功能：
- 加载一个 HF 上的自回归模型（如 gpt2、Qwen、LLaMA 类开源权重）
- 使用自定义 dataset，区分 prompt 和 target
- 在 collate 时把 prompt 段的 label 设成 -100，只让模型学 target
- 支持简单的单机单卡训练
"""

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    AdamW,
)


# ====== 1. 数据集部分 ======
class PromptTargetDataset(Dataset):
    """
    期望数据格式（举例）：
    每一行是一个 json，对应一个样本：
    {
        "prompt": "你是一个助手，请回答问题：\n问题：1+1=?\n答案：",
        "target": "2"
    }
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.data.append(obj)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        target = item["target"]

        # 这里先分别编码，后面在 collate 里再拼更灵活
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        # 截断逻辑放在 collate 里统一做
        return {
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
        }


# ====== 2. collator：在这里决定哪些 token 参与 loss ======

@dataclass
class PromptTargetCollator:
    tokenizer: Any
    max_length: int = 1024
    pad_to_multiple_of: int = None  # 可以为 None

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        for feat in features:
            prompt_ids = feat["prompt_ids"]
            target_ids = feat["target_ids"]

            # 拼成模型实际看到的序列： [prompt] + [target]
            input_ids = prompt_ids + target_ids

            # 截断到 max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]

            # 构建 labels
            # prompt 部分 label = -100；target 部分为真实 token
            labels = [-100] * len(prompt_ids) + target_ids
            labels = labels[: self.max_length]

            # attention mask
            attention_mask = [1] * len(input_ids)

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attention_mask)

        # 找到本 batch 的最大长度做 pad
        max_len = max(len(x) for x in input_ids_batch)
        if self.pad_to_multiple_of is not None:
            # 有些模型希望长度是 8 or 16 的倍数
            if max_len % self.pad_to_multiple_of != 0:
                max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        # pad
        for i in range(len(input_ids_batch)):
            pad_len = max_len - len(input_ids_batch[i])
            input_ids_batch[i] = input_ids_batch[i] + [self.tokenizer.pad_token_id] * pad_len
            attention_mask_batch[i] = attention_mask_batch[i] + [0] * pad_len
            labels_batch[i] = labels_batch[i] + [-100] * pad_len

        batch = {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }
        return batch


# ====== 3. 训练主逻辑 ======
def train(
    model_name_or_path: str,
    train_file: str,
    output_dir: str = "./output",
    max_length: int = 1024,
    per_device_batch_size: int = 2,
    num_epochs: int = 3,
    lr: float = 5e-5,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    device: str = None,
):
    # 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    # 自回归模型可能没有 pad token，手动指定一下
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # dataset
    dataset = PromptTargetDataset(train_file, tokenizer, max_length=max_length)
    collator = PromptTargetCollator(tokenizer, max_length=max_length, pad_to_multiple_of=8)

    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # 优化器 & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    num_update_steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    max_train_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * warmup_ratio)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % 10 == 0:
                print(f"Epoch {epoch} step {step} / {len(dataloader)} - loss: {loss.item():.4f}")

        # 每个 epoch 存一次
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(output_dir, f"epoch-{epoch}"))
        tokenizer.save_pretrained(os.path.join(output_dir, f"epoch-{epoch}"))

    # 最终再存一份
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training finished.")


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", type=str, required=True, help="如 'gpt2' 或本地模型路径")
    # parser.add_argument("--train_file", type=str, required=True, help="jsonl 格式的训练数据")
    # parser.add_argument("--output_dir", type=str, default="./output")
    # parser.add_argument("--max_length", type=int, default=1024)
    # parser.add_argument("--per_device_batch_size", type=int, default=2)
    # parser.add_argument("--num_epochs", type=int, default=3)
    # parser.add_argument("--lr", type=float, default=5e-5)
    # parser.add_argument("--warmup_ratio", type=float, default=0.03)
    # parser.add_argument("--weight_decay", type=float, default=0.01)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # args = parser.parse_args()

    train(
        model_name_or_path='models/Qwen2.5-0.5B-Instruct',
        train_file='data/KodCode_all.jsonl',
        output_dir='models/reference_model_SFT',
        max_length=4096,
        per_device_batch_size=2,
        num_epochs=3,
        lr=5e-5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
    )
