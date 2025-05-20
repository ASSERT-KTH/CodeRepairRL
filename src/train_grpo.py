import os
import logging
from typing import Optional
from dataclasses import dataclass, field

import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from peft import LoraConfig as PEFTLoraConfig
from trl import GRPOConfig as HFGRPOConfig, GRPOTrainer as HFGRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel # Added for Unsloth

from src.agents import nano_rollout_func
from src.rewards import (
    # reasoning rewards
    partial_reasoning_format_reward_func,
    strict_reasoning_format_reward_func,
    # detection rewards
    categorical_correctness_reward_func,
    # mono repair rewards
    sr_diff_format_reward_func,
    sr_diff_similarity_reward_func,
    # repo repair rewards
    unified_diff_similarity_reward_func,
)
from src.data import get_stack_repair_dataset, get_primevul_repair_dataset, get_primevul_detection_dataset, get_swe_gym_repo_repair_dataset
from src.utils.git import resolve_git_commit_hash

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

for noisy in ("httpx", "LiteLLM"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

@dataclass
class RunConfig:
    wandb_project: str = "TTC"
    task_type: str = "repo_repair"
    dataset_type: str = "stack"
    agent_type: Optional[str] = None  # for repo repair, either "nano" or "simple"
    context_lines: int = 0  # number of context lines to include in diffs
    commit_hash: str = ""  # added at runtime
    resume_training: bool = False

    def __post_init__(self):
        if self.task_type not in ["detection", "repair", "repo_repair"]:
            raise ValueError("task_type must be either 'detection' or 'repair'")
        if self.dataset_type not in ["primevul", "stack", "swe_gym"]:
            raise ValueError("dataset_type must be either 'stack', 'primevul' or 'swe_gym'")
        if self.agent_type:
            if self.task_type != "repo_repair":
                raise ValueError("agent_type must be None for non-repo repair tasks")
            if self.agent_type not in ["nano", "simple"]:
                raise ValueError("agent_type must be either 'nano' or 'simple'")

@dataclass
class ModelConfig:
    # Transformers configuration
    model_name: str = "Qwen/Qwen3-8B"
    attn_implementation: str = "flash_attention_3"  # only on >Hopper GPUs # Unsloth typically manages attention. This may be ignored when using Unsloth.
    load_in_4bit: bool = True # Added for Unsloth
    unsloth_max_seq_length: Optional[int] = None # Added for Unsloth
    # LoRA configuration
    lora: bool = True
    # only used if run.lora is true
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0 # Added for Unsloth
    target_modules: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

@dataclass
class GRPOConfig:
    # vLLM generation settings
    use_vllm: bool = True
    vllm_mode: str = "async_server"
    
    # Optimizer settings
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "constant_with_warmup"  # or linear, cosine learning rates have been shown to be bad for GRPO, see discussion: https://x.com/kalomaze/status/1895549497692090573
    optim: str = "paged_adamw_8bit"
    
    # Model settings - these will be automatically determined based on GPU architecture
    # when using the custom resolvers in the YAML config
    bf16: bool = True
    fp16: bool = False 

    # Generation and Training settings
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_generations: int = 4
    max_prompt_length: int = 256
    max_completion_length: int = 256

    # Reward settings
    scale_rewards: bool = False  # from Dr. GRPO, reward scaling introduces question-level difficulty bias
    use_liger_loss: bool = False # Added for LIGER loss
    
    # Training loop settings
    logging_steps: int = 1
    max_steps: int = 250
    save_steps: int = 250
    max_grad_norm: float = 0.1
    
    # Logging settings
    report_to: str = "wandb"
    run_name: Optional[str] = None
    output_dir: str = "outputs"
    log_completions: bool = True

    # silence peft warnings
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    gradient_checkpointing: bool = True # Added for gradient checkpointing

@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

# Register the config schema
cs = ConfigStore.instance()
cs.store(name="base_grpo_config", node=Config, group="")
OmegaConf.register_new_resolver("resolve_git_commit_hash", resolve_git_commit_hash)


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    os.environ["WANDB_PROJECT"] = cfg.run.wandb_project

    # Log precision settings
    precision_mode = "BF16" if cfg.grpo.bf16 else "FP16" if cfg.grpo.fp16 else "FP32"
    logger.info(f"Training with {precision_mode} precision based on GPU architecture")

    # Determine max_seq_length for Unsloth
    if cfg.model.unsloth_max_seq_length is not None:
        max_seq_length_model = cfg.model.unsloth_max_seq_length
        logger.info(f"Using unsloth_max_seq_length from ModelConfig: {max_seq_length_model}")
    else:
        max_seq_length_model = cfg.grpo.max_prompt_length + cfg.grpo.max_completion_length
        if max_seq_length_model < 512: # Ensure a minimum reasonable length
            logger.warning(f"Derived max_seq_length_model ({max_seq_length_model}) is less than 512. Setting to 512.")
            max_seq_length_model = 512
        else:
            logger.info(f"Derived max_seq_length_model as {max_seq_length_model} from GRPOConfig prompt/completion lengths.")

    tokenizer_kwargs = {}
    if "Qwen3" in cfg.model.model_name:
        try:
            with open("fixed_qwen3.jinja", "r") as f:
                tokenizer_kwargs["chat_template"] = f.read()
            logger.info("Applied fixed_qwen3.jinja chat template.")
        except FileNotFoundError:
            logger.error("fixed_qwen3.jinja not found. Qwen3 may not use the intended chat template.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.model_name,
        max_seq_length=max_seq_length_model,
        dtype=None,  # Unsloth auto-detects based on model config or system capabilities
        load_in_4bit=cfg.model.load_in_4bit, # Using 4-bit quantization as per Unsloth's recommendation
        token=os.environ.get("HF_TOKEN"),
        tokenizer_kwargs=tokenizer_kwargs,
        # attn_implementation from cfg.model.attn_implementation is not directly passed here,
        # as Unsloth handles optimal attention mechanism.
    )
    # Unsloth's FastLanguageModel.from_pretrained typically sets tokenizer.pad_token = tokenizer.eos_token
    # and tokenizer.padding_side = "left". We rely on this behavior.
    # If issues arise, explicit setting can be re-added:
    # if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    logger.info(f"Loaded model {cfg.model.model_name} with Unsloth. Max sequence length: {max_seq_length_model}. Tokenizer padding side: {tokenizer.padding_side}.")


    if cfg.model.lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.model.r,
            lora_alpha=cfg.model.lora_alpha,
            target_modules=list(cfg.model.target_modules), # Ensure it's a list if it's a tuple in config
            lora_dropout=cfg.model.lora_dropout, # Default, can be configured
            bias="none",      # Default, can be configured
            use_gradient_checkpointing="unsloth",
            random_state=3407, # From Unsloth examples, for reproducibility
            # max_seq_length is already set on the base model.
        )
        lora_config = None # Model is now a PeftModel, GRPOTrainer doesn't need separate lora_config
        logger.info("Applied LoRA configuration using Unsloth's get_peft_model.")
    else:
        lora_config = None
        logger.info("LoRA is disabled. Using base model.")

    rollout_func = None
    # Get dataset based on the task
    if cfg.run.task_type == "repair":
        get_repair_dataset = get_stack_repair_dataset if cfg.run.dataset_type == "stack" else get_primevul_repair_dataset
        dataset = get_repair_dataset(
            tokenizer=tokenizer,
            max_prompt_length=cfg.grpo.max_prompt_length,
            context_lines=cfg.run.context_lines
        )
        reward_functions = [
            partial_reasoning_format_reward_func,
            strict_reasoning_format_reward_func,
            sr_diff_format_reward_func,
            sr_diff_similarity_reward_func, 
        ]
        reward_weights = [0.1, 0.2, 0.3, 0.4]
    elif cfg.run.task_type == "detection":  # primevul only
        if not cfg.run.dataset_type == "primevul": raise ValueError("Only primevul supports detection task")
        dataset = get_primevul_detection_dataset(
            tokenizer=tokenizer, 
            max_prompt_length=cfg.grpo.max_prompt_length
        )
        reward_functions = [
            partial_reasoning_format_reward_func,
            strict_reasoning_format_reward_func,
            categorical_correctness_reward_func,
        ]
        reward_weights = [0.1, 0.2, 0.7]
    elif cfg.run.task_type == "repo_repair":
        dataset = get_swe_gym_repo_repair_dataset()
        rollout_func = nano_rollout_func  #if cfg.run.agent_type == "nano" else None
        reward_functions = [
            unified_diff_similarity_reward_func,
        ]
        reward_weights = [1.0]
    else:
        raise ValueError(f"Unknown task: {cfg.run.task_type}")  # can't happen but looks nice

    # Convert grpo config from OmegaConf to regular Python dict to ensure JSON serialization works
    grpo_params = OmegaConf.to_container(cfg.grpo, resolve=True)
    grpo_params["reward_weights"] = reward_weights
    training_args = HFGRPOConfig(**grpo_params)

    # Initialize trainer with task-specific reward functions
    trainer = HFGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        rollout_func=rollout_func,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config
    )

    trainer.train(resume_from_checkpoint=cfg.run.resume_training)

    # Save with task-specific name
    model_save_path = f"grpo_{cfg.run.task_type}_model"
    trainer.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main() 