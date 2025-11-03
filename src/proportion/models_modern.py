"""
无需 Flash-Attention 编译依赖的现代 DoReMi 模型。

该模块基于标准 Transformers（含内置优化）实现 DoReMi 模型，
在保持 DoReMi 功能的同时避免 flash-attn 的编译依赖。
"""
from collections import namedtuple
from contextlib import nullcontext
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import AutoModelForCausalLM, AutoConfig, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import logging

logger = logging.getLogger(__name__)


@dataclass#直接将类下定义的变量变成init的参数
class CausalLMOutputWithDomainIDs(CausalLMOutputWithCrossAttentions):#模型预测输出的结构化数据类型
    """
    包含 DoReMi 特定字段的扩展模型输出。

    属性：
        domain_ids: 批次中每个样本的领域 ID 张量。
        reference_pertoken_loss: 参考模型的逐 token 损失。
        pertoken_loss: 当前模型的逐 token 损失。
        token_mask: 标记有效（非填充）token 的掩码。
        hidden_states: 进入最终线性+softmax 层之前的隐藏状态。
    """
    domain_ids: Optional[torch.LongTensor] = None
    """批次中每个样本的领域 ID 张量。"""
    reference_pertoken_loss: Optional[torch.FloatTensor] = None
    """参考模型的逐 token 损失（用于超额损失计算）。"""
    pertoken_loss: Optional[torch.FloatTensor] = None
    """当前模型的逐 token 损失。"""
    token_mask: Optional[torch.FloatTensor] = None
    """标记有效（非填充）token 的掩码（有效为 1.0，填充为 0.0）。"""
    hidden_states: Optional[torch.FloatTensor] = None
    """进入最终线性+softmax 层之前的隐藏状态（嵌入）。"""


class DoReMiGPT2LMHeadModel(GPT2LMHeadModel):#完整的端到端 GPT2 语言模型
    """
    面向 DoReMi 数据混合优化的现代 GPT2 模型。

    该模型在 GPT2LMHeadModel 上加入 DoReMi 能力，并借助标准 Transformers 优化
    （梯度检查点、 高效注意力等），无需依赖 flash-attn 编译。

    属性：
        ignore_index: 损失计算时忽略的标签值（-100）。
        loss_fct: 标准训练使用的损失函数（取平均）。
        pertoken_loss_fct: 逐 token 损失函数（不做约简）。
        reference_model: 用于超额损失计算的参考模型（可选）。
    """
    
    def __init__(self, config):
        super().__init__(config)#只有当子类没有__init__时，才会自动调用父类的__init__，否则需要super().__init__(config)显式调用
        self.ignore_index = -100
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)#，忽略标签为-100的token
        self.pertoken_loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        self.reference_model = None#就是用默认配比训练的小参数模型

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        domain_ids: Optional[torch.LongTensor] = None,
        return_pertoken_losses: Optional[bool] = False,
        return_reference_hidden_states: Optional[bool] = None,
        reference_gradients: Optional[bool] = False,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithDomainIDs]:#返回元祖或自定义的CausalLMOutputWithDomainIDs类型
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 标准前向过程
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            labels=None,  # 逐 token 损失手动计算，这里不传入标签（有lable会返回loss，没有lable返回logits）
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )

        lm_logits = outputs.logits
        hidden_states = outputs.hidden_states[-1] if output_hidden_states else None

        # 计算损失
        loss = None
        pertoken_loss = None
        reference_pertoken_loss = None
        token_mask = None

        if labels is not None:
            labels = labels.to(lm_logits.device)

            if return_pertoken_losses:
                # 计算逐 token 损失
                with torch.autocast('cuda', enabled=False):#禁用自动混合精度
                    shift_logits = lm_logits[:, :-1, :].contiguous().float()
                    shift_labels = labels[:, 1:].contiguous()#对齐两个张量
                    pertoken_loss = self.pertoken_loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    pertoken_loss = pertoken_loss.view(shift_labels.size(0), shift_labels.size(1))
                    token_mask = shift_labels.ne(self.ignore_index).float()
                    loss = pertoken_loss.sum() / token_mask.sum().clamp(min=1e-10)#除以有效token数量得到平均损失，clamp(min=1e-10) 防止除零错误

                # 若提供参考模型则执行前向
                if self.reference_model is not None:
                    self.reference_model.train()#用参考模型计算参考损失
                    context_mgr = nullcontext() if reference_gradients else torch.no_grad()
                    with context_mgr:
                        ref_outputs = self.reference_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            output_hidden_states=return_reference_hidden_states,
                            return_pertoken_losses=True,
                        )
                        reference_pertoken_loss = ref_outputs.pertoken_loss
                        if return_reference_hidden_states:
                            hidden_states = ref_outputs.hidden_states
            else:
                # 标准损失计算
                with torch.autocast('cuda', enabled=False):
                    shift_logits = lm_logits[:, :-1, :].contiguous().float()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = self.loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

        if not return_dict:
            output = (lm_logits,)
            if output_hidden_states:
                output = output + (hidden_states,)
            output = output + (None, None, domain_ids, pertoken_loss, reference_pertoken_loss, token_mask)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithDomainIDs(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            domain_ids=domain_ids,
            pertoken_loss=pertoken_loss,
            reference_pertoken_loss=reference_pertoken_loss,
            token_mask=token_mask
        )


# 模型类型注册表
MODEL_REGISTRY = {
    'gpt2_mod': DoReMiGPT2LMHeadModel,
    'gpt_flash': DoReMiGPT2LMHeadModel,
    'gpt_neox_flash': DoReMiGPT2LMHeadModel,
}


def get_model_class(model_type: str) -> type:
    """
    根据类型获取模型类，默认回退到 AutoModelForCausalLM。

    参数：
        model_type: 模型类型标识字符串。

    返回：
        type: 对应类型的模型类。
    """
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type]
    return AutoModelForCausalLM


def create_modern_config(
    model_type: str,
    config_overrides: Optional[Union[str, Dict[str, Any]]] = None
) -> Any:
    """
    创建针对 2×24GB GPU 微调优化的现代配置。

    参数：
        model_type: 模型类型标识。
        config_overrides: 配置覆写字典或字符串。
    """
    if model_type in ['gpt2_mod', 'gpt_flash', 'gpt_neox_flash']:
        config = GPT2Config(
            vocab_size=50277,
            n_positions=2048,
            n_embd=768,
            n_layer=12,
            n_head=12,
            rotary_emb_fraction=0.25,
            tie_word_embeddings=True,
            scale_attn_by_inverse_layer_idx=False,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            eos_token_id=0,
            bos_token_id=0,
        )
    else:
        # 其他模型类型使用 AutoConfig
        config = AutoConfig.for_model(model_type) if hasattr(AutoConfig, 'for_model') else AutoConfig.from_dict({'model_type': model_type})

    # 应用覆写项
    if config_overrides:
        if isinstance(config_overrides, str):
            # 解析 "n_positions=2048,n_embd=768" 形式的字符串
            overrides = {}
            for item in config_overrides.split(','):
                key, value = item.split('=')
                overrides[key] = eval(value)
            config_overrides = overrides
        
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Set config.{key} = {value}")

    # 启用现代化优化选项
    if hasattr(config, 'use_cache'):
        config.use_cache = True

    return config
