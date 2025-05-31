import os
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

import hydra
import torch
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig as HFSFTConfig
from peft import LoraConfig as PEFTLoraConfig
from huggingface_hub import login

from src.train_grpo import ModelConfig
from src.utils.git import resolve_git_commit_hash
from src.data.swe_gym import get_swe_gym_formatted_sft_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy loggers
for noisy in ("httpx", "LiteLLM", "transformers.tokenization_utils_base"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)


@dataclass
class RunConfig:
    wandb_project: str = "SWE-Gym-SFT"
    dataset_name: str = "bjarni/swe-gym-lite-sft"
    max_seq_length: int = 8192
    reward_min: float = 0.2
    output_model_name: str = "bjarni/qwen3-8b-swe-gym-sft"
    push_to_hub: bool = True
    commit_hash: str = ""  # added at runtime


@dataclass 
class SFTConfig:
    """SFT Configuration using TRL's SFTConfig as base"""
    # TrainingArguments parameters that are part of SFTConfig
    output_dir: str = "outputs/sft_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    report_to: str = "wandb"
    run_name: Optional[str] = None
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = False
    
    # SFT-specific parameters that belong in SFTConfig
    dataset_text_field: str = "text"
    max_length: int = 8192
    packing: Optional[bool] = False
    dataset_num_proc: Optional[int] = None
    dataset_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)


# Register the config schema
cs = ConfigStore.instance()
cs.store(name="base_sft_config", node=Config, group="")
# OmegaConf.register_new_resolver("resolve_git_commit_hash", resolve_git_commit_hash)


@hydra.main(version_base="1.1", config_path="conf", config_name="sft_config")
def main(cfg: Config) -> None:
    os.environ["WANDB_PROJECT"] = cfg.run.wandb_project

    # Login to HuggingFace if pushing to hub
    if cfg.run.push_to_hub:
        login()  # Will use token from environment or cache
    
    # Log precision settings
    precision_mode = torch.bfloat16 if cfg.sft.bf16 else torch.float16 if cfg.sft.fp16 else torch.float32
    logger.info(f"Training with {precision_mode} precision")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {cfg.model.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, 
        torch_dtype=precision_mode,
        # attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=True
    )
    
    # Load tokenizer with fixed template for Qwen3
    if "Qwen3" in cfg.model.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name, 
            chat_template=open("fixed_qwen3.jinja").read(),
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # For SFT, we typically pad on the right
    
    # Configure LoRA if enabled
    peft_config = None
    if cfg.model.lora:
        lora_params = OmegaConf.to_container(cfg.model, resolve=True)
        peft_config = PEFTLoraConfig(
            r=lora_params["r"],
            lora_alpha=lora_params["lora_alpha"],
            target_modules=lora_params["target_modules"],
            task_type="CAUSAL_LM",
            lora_dropout=0.1
        )
        logger.info(f"Using LoRA with r={cfg.model.r}, alpha={cfg.model.lora_alpha}")
    
    # Load and prepare dataset using the swe_gym function
    train_dataset = get_swe_gym_formatted_sft_dataset(
        dataset_name=cfg.run.dataset_name,
        tokenizer=tokenizer,
        max_seq_length=cfg.run.max_seq_length,
        reward_min=cfg.run.reward_min
    )
    
    if len(train_dataset) == 0:
        raise ValueError("No training examples after preprocessing!")
    
    # Convert SFT config to dict for SFTConfig creation
    sft_params = OmegaConf.to_container(cfg.sft, resolve=True)
    
    # Update with run-specific parameters
    sft_params["max_length"] = cfg.run.max_seq_length
    
    # Add hub model ID if pushing to hub
    if cfg.run.push_to_hub:
        sft_params["hub_model_id"] = cfg.run.output_model_name
        sft_params["push_to_hub"] = True
    
    # Create SFTConfig instance
    training_args = HFSFTConfig(**sft_params)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Train the model
    logger.info("Starting SFT training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Push to hub if requested
    if cfg.run.push_to_hub:
        logger.info(f"Pushing model to HuggingFace Hub: {cfg.run.output_model_name}")
        trainer.push_to_hub(commit_message="SFT training completed")
        logger.info("Successfully pushed model to HuggingFace Hub")
    
    logger.info("SFT training completed successfully!")


if __name__ == "__main__":
    main()