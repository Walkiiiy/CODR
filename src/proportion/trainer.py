"""
简化核心功能版本（单GPU/CPU）

本模块实现了 DoReMi (Domain Reweighting with Minimax Optimization) 的核心训练逻辑：
1. 在训练过程中动态更新各域的权重
2. 计算当前模型相对于参考模型的逐 token 超额损失
3. 根据超额损失更新域权重，使模型更关注困难域
4. 使用更新后的域权重重新加权训练损失

核心原理：
- 如果某个域在当前模型上的损失比参考模型高很多（超额损失大），说明该域较难学习
- DoReMi 会增加该域的权重，让模型在该域上训练更多
- 通过这种方式实现域之间的平衡，避免某些域被忽略

使用前提：
- 模型需要有缓冲区：train_domain_weights, avg_domain_weights, perdomain_scores, update_counter
- 模型输出需要包含：pertoken_loss, reference_pertoken_loss, token_mask
- args 需要包含：domain_config_path, reweight_domains, doremi_optimizer, reweight_eta, reweight_eps, gradient_accumulation_steps
"""
import json
from typing import Dict, Any, Optional, List
import torch
from transformers import Trainer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ProportionTrainer(Trainer):
    """
    DoReMi 训练器 - 简化核心功能版本（单GPU/CPU）
    
    核心功能：
    1. 域权重初始化：从配置文件中读取初始域权重
    2. 超额损失计算：计算当前模型相对于参考模型的逐 token 损失差值
    3. 域权重更新：根据各域的平均超额损失更新域权重
    4. 损失重加权：使用更新后的域权重重新加权训练损失
    
    属性：
        domain_config: 域配置字典（从 JSON 文件加载）
        train_domain_weights_dict: 训练域名称到权重的字典
        domain_list: 排序后的域名称列表
        pertoken_scores: 累积的逐 token 超额损失列表（用于域权重更新）
        token_masks: 累积的 token 掩码列表（标记哪些 token 是有效的，非 padding）
        domain_ids: 累积的域 ID 列表（标记每个样本属于哪个域）
        sampling_weights: 采样权重（用于损失重加权时的归一化）
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        初始化 DoReMi 训练器
        
        从配置文件中加载域配置，初始化域权重相关的数据结构
        """
        super().__init__(*args, **kwargs)
        
        # 加载域配置文件
        # 配置文件格式：{"train_domain_weights": {"domain1": 0.5, "domain2": 0.5}, ...}
        if hasattr(self.args, 'domain_config_path') and self.args.domain_config_path:
            with open(self.args.domain_config_path, 'r', encoding='utf-8') as f:
                self.domain_config: Dict[str, Any] = json.load(f)
            
            # 提取训练域权重字典
            self.train_domain_weights_dict: Dict[str, float] = self.domain_config['train_domain_weights']
            
            # 获取排序后的域名称列表（确保顺序一致）
            self.domain_list: List[str] = list(sorted(self.train_domain_weights_dict.keys()))
            
            # 初始化采样权重（用于损失重加权时的归一化）
            self.sampling_weights: torch.Tensor = torch.tensor(
                [self.train_domain_weights_dict[domain] for domain in self.domain_list]
            )
        else:
            # 如果没有配置文件，使用默认值
            self.domain_config = {}
            self.train_domain_weights_dict = {}
            self.domain_list = []
            self.sampling_weights = torch.tensor([])
        
        # 用于累积域权重更新所需的数据
        # 这些列表会在梯度累积步骤中累积数据，然后一次性更新域权重
        self.pertoken_scores: List[torch.Tensor] = []  # 逐 token 超额损失
        self.token_masks: List[torch.Tensor] = []       # token 掩码（标记有效 token）
        self.domain_ids: List[torch.Tensor] = []         # 域 ID（标记每个样本的域）
        
        # 告诉 Trainer 跳过数据跳过逻辑，因为我们在 dataloader 中处理
        if hasattr(self.args, 'ignore_data_skip'):
            self.args.ignore_data_skip = True
    
    def read_weights(self) -> torch.Tensor:
        """
        读取当前域权重
        
        域权重存储在模型的缓冲区中，可以通过这种方式读取
        
        返回：
            torch.Tensor: 当前域权重（形状: [num_domains]）
        """
        if hasattr(self.model, 'train_domain_weights'):
            return self.model.train_domain_weights.clone()
        else:
            raise AttributeError("模型缺少 train_domain_weights 缓冲区。请确保模型已初始化域权重。")
    
    def write_weights(self, weights: torch.Tensor) -> None:
        """
        更新域权重到模型缓冲区
        
        同时更新：
        1. train_domain_weights: 当前训练使用的域权重
        2. avg_domain_weights: 域权重的移动平均（用于记录和日志）
        
        参数：
            weights: 新的域权重（形状: [num_domains]，已归一化）
        """
        if not hasattr(self.model, 'train_domain_weights'):
            raise AttributeError("模型缺少 train_domain_weights 缓冲区。请确保模型已初始化域权重。")
        
        # 更新计数器（用于计算移动平均）
        if hasattr(self.model, 'update_counter'):
            self.model.update_counter += 1
        else:
            # 如果没有计数器，初始化为 1
            self.model.register_buffer('update_counter', torch.tensor(1))
        
        # 更新当前域权重
        self.model.train_domain_weights[:] = weights.float()
        
        # 更新移动平均：avg = (avg * (n-1) + new) / n
        if hasattr(self.model, 'avg_domain_weights'):
            self.model.avg_domain_weights[:] = (
                (self.model.avg_domain_weights * (self.model.update_counter - 1) + weights)
                / self.model.update_counter
            )
        else:
            # 如果没有移动平均缓冲区，初始化它
            self.model.register_buffer('avg_domain_weights', weights.clone())
    
    def update_domain_weights(
        self,
        scores: torch.Tensor,
        scores_mask: torch.Tensor,
        domain_ids: torch.Tensor
    ) -> None:
        """
        根据超额损失更新域权重 - DoReMi 核心算法
        
        算法步骤：
        1. 计算每个域的平均超额损失（per-domain score）
        2. 使用指数更新规则更新域权重：
           log(w_new) = log(w_old) + eta * score
           w_new = softmax(log(w_new)) + eps * uniform
        3. 归一化域权重
        
        参数：
            scores: 逐 token 超额损失（形状: [batch_size, seq_len]）
            scores_mask: token 掩码，标记哪些 token 是有效的（形状: [batch_size, seq_len]）
            domain_ids: 每个样本的域 ID（形状: [batch_size]）
        """
        # 读取当前域权重
        train_domain_weights = self.read_weights()
        
        # 分离计算图（不参与梯度计算）
        scores = scores.detach()
        domain_ids = domain_ids.detach()
        
        # 获取更新参数（带默认值）
        doremi_optimizer = getattr(self.args, 'doremi_optimizer', 'doremiv1')#只支持doremiv1算法
        reweight_eta = getattr(self.args, 'reweight_eta', 1.0)
        reweight_eps = getattr(self.args, 'reweight_eps', 1e-4)
        
        # DoReMi v1 更新规则
        if doremi_optimizer == 'doremiv1':
            # 步骤 1: 计算每个域的平均超额损失
            perdomain_scores: List[torch.Tensor] = []
            
            # 确保模型有 perdomain_scores 缓冲区
            if not hasattr(self.model, 'perdomain_scores'):
                # 初始化为零（或使用对数词汇表大小）
                self.model.register_buffer('perdomain_scores', torch.zeros(len(train_domain_weights)))
            
            for domain_id in range(len(train_domain_weights)):
                # 找到属于当前域的所有样本
                domain_mask = (domain_ids == domain_id)
                
                if domain_mask.sum() > 0:
                    # 提取该域的 token 掩码
                    perdomain_scores_mask = scores_mask[domain_mask]
                    
                    # 计算该域的平均超额损失（只考虑有效 token，且截断负值）
                    # 超额损失 = 当前模型损失 - 参考模型损失
                    # 如果超额损失为负，说明该域比参考模型表现好，截断为 0
                    curr_domain_scores = torch.clip(
                        scores[domain_mask][perdomain_scores_mask],
                        min=0
                    ).mean()
                else:
                    # 如果该域没有样本，使用之前的分数
                    curr_domain_scores = self.model.perdomain_scores[domain_id]
                
                perdomain_scores.append(curr_domain_scores)
            
            # 更新模型的 per-domain scores
            self.model.perdomain_scores[:] = torch.stack(perdomain_scores).float()
            
            # 步骤 2: 使用指数更新规则更新域权重
            # log(w_new) = log(w_old) + eta * score
            # eta 是学习率，控制更新幅度
            log_new_train_domain_weights = (
                torch.log(train_domain_weights + 1e-10) +  # 加小值避免 log(0)
                reweight_eta * self.model.perdomain_scores
            )
            
            # 步骤 3: 归一化（softmax 归一化）
            # 减去 logsumexp 实现数值稳定的 softmax
            log_new_train_domain_weights = (
                log_new_train_domain_weights -
                torch.logsumexp(log_new_train_domain_weights, dim=0)
            )
            
            # 步骤 4: 应用平滑项（防止权重过于极端）
            # w_new = (1 - eps) * softmax(...) + eps * uniform
            # eps 是平滑参数，确保每个域都有最小权重
            train_domain_weights = (
                (1 - reweight_eps) * torch.exp(log_new_train_domain_weights) +
                reweight_eps / len(log_new_train_domain_weights)
            )
            
            # 步骤 5: 写入更新后的权重
            self.write_weights(train_domain_weights)
        else:
            raise ValueError(f"DoReMi optimizer {doremi_optimizer} not supported")
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        return_pertoken_losses: bool = False
    ) -> torch.Tensor:
        """
        计算损失
        
        如果 return_pertoken_losses=True，会计算逐 token 损失（用于域权重更新）
        否则计算标准损失（用于反向传播）
        
        参数：
            model: 模型
            inputs: 输入字典，包含 input_ids, labels, domain_ids 等
            return_outputs: 是否返回模型输出
            return_pertoken_losses: 是否计算逐 token 损失
            
        返回：
            torch.Tensor: 损失值（如果 return_outputs=True，返回 (loss, outputs) 元组）
        """
        # 如果使用 label smoother，需要特殊处理
        labels = None
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        
        # 告诉模型计算逐 token 损失
        inputs['return_pertoken_losses'] = return_pertoken_losses
        
        # 前向传播
        outputs = model(**inputs)
        
        # 计算损失
        if labels is not None and self.label_smoother is not None:
            # 使用 label smoother
            # 对于因果语言模型，需要 shift_labels=True
            if hasattr(self.label_smoother, '__call__'):
                try:
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                except TypeError:
                    # 如果 label_smoother 不支持 shift_labels，使用默认调用
                    loss = self.label_smoother(outputs, labels)
            else:
                loss = self.label_smoother(outputs, labels)
        elif labels is not None:
            # 不使用 label smoother，直接从 outputs 获取损失
            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
                if loss is None:
                    raise ValueError(f"Model did not return loss. Keys: {list(outputs.keys())}")
            else:
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
        else:
            # 从模型输出中提取损失
            if isinstance(outputs, dict):
                if "loss" not in outputs:
                    raise ValueError(
                        f"Model did not return loss. Keys: {list(outputs.keys())}"
                    )
                loss = outputs["loss"]
            else:
                # 如果输出是元组，第一个元素是损失
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
        
        if return_outputs:
            return (loss, outputs)
        else:
            return loss
    
    def training_step(#batch级别的训练步骤
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        执行一个训练步骤 - DoReMi 核心训练逻辑
        
        流程：
        1. 如果启用域重加权（reweight_domains=True）：
           a. 计算逐 token 损失（当前模型和参考模型）
           b. 计算超额损失（当前模型损失 - 参考模型损失）
           c. 累积超额损失（用于域权重更新）
           d. 当累积足够的梯度步数后，更新域权重
           e. 使用更新后的域权重重新加权训练损失
        2. 否则：使用标准损失计算
        
        参数：
            model: 要训练的模型
            inputs: 输入字典，包含 input_ids, labels, domain_ids 等
            
        返回：
            torch.Tensor: 训练损失（已分离计算图）
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 检查是否启用域重加权
        reweight_domains = getattr(self.args, 'reweight_domains', False)
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        
        # 如果启用域重加权（DoReMi 模式）
        if reweight_domains:
            # 计算逐 token 损失（包含当前模型和参考模型的损失）
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(# 前向传播包含在这个方法中
                    model, inputs,
                    return_outputs=True,
                    return_pertoken_losses=True
                )
            #这里的ouputs是在models.py里面重定义的原始ouputs的子类，添加了pertoken_loss, reference_pertoken_loss, token_mask等域相关的信息
            # 提取逐 token 损失和相关信息
            pertoken_loss = outputs.pertoken_loss          # 当前模型的逐 token 损失
            reference_pertoken_loss = outputs.reference_pertoken_loss  # 参考模型的逐 token 损失
            token_mask = outputs.token_mask                # token 掩码（标记有效 token）
            
            # 计算超额损失：当前模型损失 - 参考模型损失
            # 超额损失 > 0 表示当前模型在该 token 上表现比参考模型差
            excess_loss = pertoken_loss - reference_pertoken_loss
            
            # 累积数据（用于域权重更新）
            self.pertoken_scores.append(excess_loss.detach())
            self.token_masks.append(token_mask.detach())
            self.domain_ids.append(inputs['domain_ids'].detach())
            
            #每个accumulation_steps更新一次域权重
            # 当累积足够的梯度步数后，更新域权重
            # 这样可以减少更新频率，提高稳定性
            if len(self.pertoken_scores) == gradient_accumulation_steps:
                # 拼接所有累积的数据
                pertoken_scores = torch.cat(self.pertoken_scores, dim=0)
                token_masks = torch.cat(self.token_masks, dim=0).bool()
                domain_ids = torch.cat(self.domain_ids, dim=0)
                
                # 更新域权重
                self.update_domain_weights(pertoken_scores, token_masks, domain_ids)
                
                # 清空累积列表
                self.pertoken_scores = []
                self.token_masks = []
                self.domain_ids = []
            
            # 使用更新后的域权重重新加权训练损失
            doremi_optimizer = getattr(self.args, 'doremi_optimizer', 'doremiv1')
            if doremi_optimizer == 'doremiv1':
                # 读取当前域权重
                train_domain_weights = self.read_weights().to(pertoken_loss.device).float()
                
                # 如果采样权重不均匀，需要归一化
                # 例如：如果某个域采样概率是 0.8，但权重是 0.5，需要调整
                if len(self.sampling_weights) > 0:
                    train_domain_weights = train_domain_weights / self.sampling_weights.to(train_domain_weights.device)
                    train_domain_weights = train_domain_weights / train_domain_weights.sum()
                
                # 为每个样本分配对应的域权重
                # domain_ids 标记每个样本属于哪个域
                curr_domain_weights = train_domain_weights[inputs['domain_ids']].unsqueeze(-1)
                curr_domain_weights = curr_domain_weights.expand_as(pertoken_loss).detach()
                
                # 只对有效 token 应用权重
                curr_domain_weights = curr_domain_weights * token_mask
                
                # 计算归一化因子（所有有效 token 的权重之和）
                normalizer = curr_domain_weights.detach().sum().clamp(min=1e-10)
                
                # 计算重加权后的损失
                # loss = sum(pertoken_loss * domain_weight) / normalizer
                token_mask = token_mask.detach().type(pertoken_loss.dtype)
                loss = (pertoken_loss * curr_domain_weights.detach()).sum() / normalizer
            else:
                raise ValueError(f"doremi_optimizer {doremi_optimizer} is not supported")
        else:
            # 标准训练模式：不使用域重加权
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        
        # 梯度累积：除以累积步数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        
        # 反向传播,每一步都优化模型参数,acc_step后的参数更新不在这里，由父类自动完成
        loss.backward()
        
        return loss.detach()
