#!/usr/bin/env python
# coding=utf-8

"""
Modern training script for DoReMi fine-tuning.
Optimized for small-scale training on 2x24GB GPUs with modern transformers.
"""

import logging
from pathlib import Path
import os
import sys
import json
import numpy as np

import datasets
import torch

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import TrainerState
from transformers.trainer import TRAINER_STATE_NAME
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from doremi.training_args import ModelArguments, DataTrainingArguments, FullTrainingArguments
import doremi.dataloader as data_utils
from doremi.trainer import DoReMiTrainer
from doremi.models_modern import (
    MODEL_REGISTRY, 
    get_model_class, 
    create_modern_config,
    DoReMiGPT2LMHeadModel
)

# Modern transformers version check
check_min_version("4.40.0")

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FullTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Checkpoint detection
    last_checkpoint = None
    num_skip_examples = 0
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )
            state = TrainerState.load_from_json(str(Path(last_checkpoint) / TRAINER_STATE_NAME))
            global_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
            num_skip_examples = state.global_step * global_batch_size
            logger.info(f"Skipping {num_skip_examples} examples")

    set_seed(training_args.seed)

    # Load config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        # Create from scratch with modern defaults
        config = create_modern_config(model_args.model_type, model_args.config_overrides)
        logger.warning("Creating new config from scratch.")
        
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        )
    
    # Set max length
    tokenizer.model_max_length = data_args.max_token_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load or create model
    model_class = get_model_class(model_args.model_type)
    
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        
        # Simplified: just load the model directly
        try:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
            )
            logger.info(f"Loaded pretrained model from {model_args.model_name_or_path}")
        except Exception as e:
            logger.warning(f"Failed to load as {model_class.__name__}, trying AutoModelForCausalLM: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
            )
    else:
        # Create from scratch
        model = model_class(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    # Load domain config
    with open(training_args.domain_config_path, 'r') as f:
        domain_config = json.load(f)

    train_domain_weights_dict = domain_config['train_domain_weights']
    eval_domain_weights_dict = domain_config['eval_domain_weights']
    domain_list = list(sorted(train_domain_weights_dict.keys()))

    # Setup reference model if doing reweighting
    if training_args.reweight_domains:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        
        if model_args.model_type in MODEL_REGISTRY:
            ref_model_class = MODEL_REGISTRY[model_args.model_type]
        else:
            ref_model_class = AutoModelForCausalLM
            
        reference_model = ref_model_class.from_pretrained(
            training_args.reference_model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
        )
        for param in reference_model.parameters():
            param.requires_grad = False
        model.reference_model = reference_model
        
        total_domain_weight = sum(train_domain_weights_dict.values())
        model.register_buffer('train_domain_weights', torch.tensor(
                [train_domain_weights_dict[domain] / total_domain_weight for domain in domain_list]))
        model.register_buffer('avg_domain_weights', model.train_domain_weights.clone())
        model.register_buffer('perdomain_scores', torch.ones(len(train_domain_weights_dict)) * np.log(len(tokenizer)))
        model.register_buffer('update_counter', torch.tensor(1))
    else:
        reference_model = None

    # Prepare datasets
    if training_args.do_train:
        train_dataset = data_utils.get_preprocessed_mixed_dataset(
                preprocessed_dir=data_args.dataset_dir,
                domain_weights_dict=train_domain_weights_dict,
                dataset_name=data_args.dataset_name,
                cache_dir=model_args.cache_dir,
                split='train',
                max_samples=data_args.max_train_samples,
                add_domain_id=data_args.add_domain_id,
                domain_weight_buffer_handle=None,
                seed=training_args.seed,
                tokenizer=tokenizer,
                shuffle=data_args.shuffle,
                num_skip_examples=num_skip_examples,
                shard_reversal=training_args.reweight_domains,
                keep_in_memory=data_args.keep_in_memory)

    if training_args.do_eval:
        if data_args.eval_dataset_dir is None:
            data_args.eval_dataset_dir = data_args.dataset_dir
        if data_args.eval_dataset_name is None:
            data_args.eval_dataset_name = data_args.dataset_name

        eval_dataset = data_utils.get_preprocessed_mixed_dataset(
                preprocessed_dir=data_args.eval_dataset_dir,
                domain_weights_dict=eval_domain_weights_dict,
                dataset_name=data_args.eval_dataset_name,
                cache_dir=model_args.cache_dir,
                split='validation',
                add_domain_id=data_args.add_domain_id,
                max_samples=data_args.max_eval_samples,
                tokenizer=tokenizer,
                no_interleave=True,
                keep_in_memory=data_args.keep_in_memory)

    training_args.ddp_find_unused_parameters = False

    torch.cuda.empty_cache()

    # Initialize trainer
    trainer = DoReMiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_utils.get_data_collator(tokenizer, do_padding=data_args.do_padding, max_length=data_args.max_token_length),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        if training_args.reweight_domains:
            avg_domain_weights_dict = {}
            for i in range(len(model.avg_domain_weights)):
                domain_name = domain_list[i]
                metrics[f'avg_domain_weight:{domain_name}'] = model.avg_domain_weights[i].item()
                avg_domain_weights_dict[domain_name] = model.avg_domain_weights[i].item()

            # Save avg domain weights to json
            avg_domain_weights_file = Path(training_args.output_dir) / 'avg_domain_weights.json'
            with open(avg_domain_weights_file, 'w') as f:
                json.dump(avg_domain_weights_dict, f, indent=2)

            # Also save to configs dir
            config_dict = {"train_domain_weights": avg_domain_weights_dict,
                           "eval_domain_weights": avg_domain_weights_dict}
            config_dict_file = Path(__file__).parent.parent / 'configs' / f"{Path(training_args.output_dir).name}.json"
            with open(config_dict_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if training_args.eval_all_checkpoints:
            checkpoint_dir_list = trainer.get_all_checkpoints(training_args.output_dir)
        else:
            checkpoint_dir_list = [get_last_checkpoint(training_args.output_dir)]

        for checkpoint_dir in checkpoint_dir_list:
            if checkpoint_dir is None:
                continue
            trainer.load_checkpoint(checkpoint_dir)
            state = TrainerState.load_from_json(str(Path(checkpoint_dir) / TRAINER_STATE_NAME))
            trainer.state.global_step = state.global_step

            if not training_args.skip_perplexity_eval:
                metrics = trainer.evaluate()
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

            if training_args.downstream_datasets is not None:
                dataset_names = training_args.downstream_datasets.split(',')
                downstream_metrics = trainer.evaluate_fewshot(
                        dataset_names,
                        max_samples=data_args.max_downstream_samples,
                        num_shots=training_args.downstream_num_shots)
                trainer.log_metrics("eval", downstream_metrics)
                trainer.save_metrics("eval", downstream_metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
