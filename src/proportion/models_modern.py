"""
Modern DoReMi models without flash-attention compilation dependency.
Uses standard transformers models with built-in optimizations.
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


@dataclass
class CausalLMOutputWithDomainIDs(CausalLMOutputWithCrossAttentions):
    domain_ids: Optional[torch.LongTensor] = None
    reference_pertoken_loss: Optional[torch.FloatTensor] = None
    pertoken_loss: Optional[torch.FloatTensor] = None
    token_mask: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None


class DoReMiGPT2LMHeadModel(GPT2LMHeadModel):
    """
    Modern GPT2-based model for DoReMi data mixing optimization.
    Uses gradient checkpointing and efficient attention from transformers.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.ignore_index = -100
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
        self.pertoken_loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        self.reference_model = None

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
    ) -> Union[Tuple, CausalLMOutputWithDomainIDs]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Standard forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            labels=None,  # Compute loss manually for pertoken losses
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )

        lm_logits = outputs.logits
        hidden_states = outputs.hidden_states[-1] if output_hidden_states else None

        # Compute loss
        loss = None
        pertoken_loss = None
        reference_pertoken_loss = None
        token_mask = None

        if labels is not None:
            labels = labels.to(lm_logits.device)
            
            if return_pertoken_losses:
                # Compute per-token loss
                with torch.autocast('cuda', enabled=False):
                    shift_logits = lm_logits[:, :-1, :].contiguous().float()
                    shift_labels = labels[:, 1:].contiguous()
                    pertoken_loss = self.pertoken_loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    pertoken_loss = pertoken_loss.view(shift_labels.size(0), shift_labels.size(1))
                    token_mask = shift_labels.ne(self.ignore_index).float()
                    loss = pertoken_loss.sum() / token_mask.sum().clamp(min=1e-10)

                # Run reference model if provided
                if self.reference_model is not None:
                    self.reference_model.train()
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
                # Standard loss computation
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


# Registry for model types
MODEL_REGISTRY = {
    'gpt2_mod': DoReMiGPT2LMHeadModel,
    'gpt_flash': DoReMiGPT2LMHeadModel,
    'gpt_neox_flash': DoReMiGPT2LMHeadModel,
}


def get_model_class(model_type: str):
    """Get model class by type, falling back to AutoModelForCausalLM."""
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type]
    return AutoModelForCausalLM


def create_modern_config(model_type, config_overrides=None):
    """
    Create a modern config optimized for fine-tuning on 2x24GB GPUs.
    
    Args:
        model_type: Model type identifier
        config_overrides: Dict of config overrides
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
        # For other model types, use AutoConfig
        config = AutoConfig.for_model(model_type) if hasattr(AutoConfig, 'for_model') else AutoConfig.from_dict({'model_type': model_type})
    
    # Apply overrides
    if config_overrides:
        if isinstance(config_overrides, str):
            # Parse string format like "n_positions=2048,n_embd=768"
            overrides = {}
            for item in config_overrides.split(','):
                key, value = item.split('=')
                overrides[key] = eval(value)
            config_overrides = overrides
        
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Set config.{key} = {value}")
    
    # Enable modern optimizations
    if hasattr(config, 'use_cache'):
        config.use_cache = True
    
    return config
