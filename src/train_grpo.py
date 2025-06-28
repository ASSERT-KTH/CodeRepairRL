import os
import logging
from functools import partial
from datetime import datetime
from dataclasses import dataclass, field

import hydra
import torch
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from peft import LoraConfig as PEFTLoraConfig, PeftModel
from trl import GRPOConfig as HFGRPOConfig, GRPOTrainer as HFGRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import whoami

from src.agents.nano_agent import nano_rollout_func, NanoConfig
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
    unified_diff_file_match_reward_func,
    unified_diff_similarity_reward_func,
    unified_diff_similarity_reward_func_test,
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
    context_lines: int = 0  # number of context lines to include in diffs
    commit_hash: str = ""  # added at runtime
    resume_training: bool = False
    output_model_name: str = ""  # HuggingFace Hub model name
    push_to_hub: bool = False

    def __post_init__(self):
        if self.task_type not in ["detection", "repair", "repo_repair"]:
            raise ValueError("task_type must be either 'detection' or 'repair'")
        if self.dataset_type not in ["primevul", "stack", "swe_gym"]:
            raise ValueError("dataset_type must be either 'stack', 'primevul' or 'swe_gym'")

@dataclass
class ModelConfig:
    # Transformers configuration
    model_name: str = "Qwen/Qwen3-8B"
    attn_implementation: str = "flash_attention_3"  # only on >Hopper GPUs
    # LoRA configuration
    lora: bool = True
    r: int = 32
    lora_alpha: int = 64
    target_modules: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

@dataclass
class GRPOConfig:
    # vLLM generation settings
    use_vllm: bool = True
    vllm_mode: str = "async_server"
    # whether completions are multi-turn or single-turn
    multi_turn: bool = True
    # whether to mask tool responses in the loss
    mask_tool_responses: bool = False

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
    
    # Loss type
    loss_type: str = "dr_grpo"  # been shown to have less sequence-length bias
    
    # Attention kernel
    use_liger_loss: bool = True  # should cut memory footprint

    # Gradient checkpointing
    gradient_checkpointing: bool = True  # offload gradient to CPU for better memory utilization
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": True})

    # Training loop settings
    logging_steps: int = 1
    max_steps: int = 250
    save_steps: int = 250
    max_grad_norm: float = 0.1

    # Logging settings
    run_name: str = ""  # automatically set at runtime
    report_to: str = "wandb"
    output_dir: str = "outputs"
    log_completions: bool = True

    # silence peft warnings
    label_names: list[str] = field(default_factory=lambda: ["labels"])

@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    agent: NanoConfig = field(default_factory=NanoConfig)

# Register the config schema
cs = ConfigStore.instance()
cs.store(name="base_grpo_config", node=Config, group="")
OmegaConf.register_new_resolver("resolve_git_commit_hash", resolve_git_commit_hash)
OmegaConf.register_new_resolver("now", lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))


@hydra.main(version_base="1.1", config_path="conf", config_name="grpo_config")
def main(cfg: Config) -> None:
    # Validate that run_name is provided and not empty
    if not cfg.grpo.run_name or cfg.grpo.run_name.strip() == "":
        raise ValueError(
            "run_name is required and cannot be empty. "
            "Please provide a unique run name to prevent model overwriting. "
            "Example: grpo.run_name='my-grpo-experiment-v1'"
        )
    
    os.environ["WANDB_PROJECT"] = cfg.run.wandb_project

    # Check HuggingFace login if pushing to hub
    if cfg.run.push_to_hub:
        try:
            whoami()
        except Exception:
            raise ValueError("Not logged in to HuggingFace. Please run 'huggingface-cli login' first.")

    # Log precision settings
    precision_mode = torch.bfloat16 if cfg.grpo.bf16 else torch.float16 if cfg.grpo.fp16 else torch.float32
    logger.info(f"Training with {precision_mode} precision based on GPU architecture")
    
    # Load base model
    logger.info(f"Loading model: {cfg.model.model_name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, attn_implementation=cfg.model.attn_implementation, torch_dtype=precision_mode)
    
    # Load tokenizer
    if "Qwen3" in cfg.model.model_name:  # Qwen3's jinja template is bugged
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, chat_template=open("fixed_qwen3.jinja").read())
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # by padding a batch of prompts on the left side we can generate many completions in parallel (padding tokens are masked away)

    if cfg.model.lora:
        lora_params = OmegaConf.to_container(cfg.model, resolve=True)
        lora_config = PEFTLoraConfig(
            r=lora_params["r"],
            lora_alpha=lora_params["lora_alpha"],
            target_modules=lora_params["target_modules"],
            task_type="CAUSAL_LM"
        )
    else:
        lora_config = None

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
        # Update agent config with model and token_limit
        cfg.agent.model = f"hosted_vllm/{cfg.model.model_name}"
        cfg.agent.token_limit = cfg.grpo.max_prompt_length + cfg.grpo.max_completion_length - 512
        # Convert OmegaConf to NanoConfig dataclass
        agent_config = NanoConfig(**OmegaConf.to_container(cfg.agent, resolve=True))
        rollout_func = partial(nano_rollout_func, config=agent_config)
        reward_functions = [
            unified_diff_file_match_reward_func,
            unified_diff_similarity_reward_func,
            unified_diff_similarity_reward_func_test,
        ]
        reward_weights = [0.2, 0.4, 0.4]
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
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"LoRA adapters saved to {model_save_path}")
    
    # If using LoRA, also save the merged model for simplified VLLM deployment
    if cfg.model.lora:
        merged_model_dir = f"{model_save_path}_merged"
        logger.info(f"Merging LoRA adapters and saving merged model to {merged_model_dir}")
        
        # Load the trained LoRA model
        peft_model = PeftModel.from_pretrained(model, model_save_path)
        # Merge and unload the adapters to get a standard model
        merged_model = peft_model.merge_and_unload()
        
        # Save the merged model
        merged_model.save_pretrained(merged_model_dir)
        tokenizer.save_pretrained(merged_model_dir)
        logger.info(f"Successfully saved merged model to {merged_model_dir}")
    
    # Push to hub if requested
    if cfg.run.push_to_hub:
        if cfg.model.lora:
            logger.info(f"Pushing merged model to HuggingFace Hub: {cfg.run.output_model_name}")
            merged_model.push_to_hub(cfg.run.output_model_name, tokenizer=tokenizer, commit_message="GRPO training completed")
            logger.info("Successfully pushed merged model to HuggingFace Hub")
        else:
            logger.info(f"Pushing model to HuggingFace Hub: {cfg.run.output_model_name}")
            trainer.push_to_hub(commit_message="GRPO training completed")
            logger.info("Successfully pushed model to HuggingFace Hub")

if __name__ == "__main__":
    main() 