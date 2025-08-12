import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from trl import SFTTrainer, SFTConfig, create_reference_model

@dataclass
class KLSFTConfig(SFTConfig):
    kl_lambda: float = field(
        default=0.01,
        metadata={"help": "KL divergence regularization weight"},
    )

# TODO: This is pretty shitty, when we are training a peft model, we do not need the reference model (just unmerge the adapters)
class KLSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # build a frozen reference model in a way that works for every backend
        if (
            self.accelerator.distributed_type == "DEEPSPEED"
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
        ):
            # ZeRO-3: create a brand-new model instead of cloning.
            from transformers import AutoModelForCausalLM

            base_id = self.model.config._name_or_path
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                torch_dtype=self.model.dtype,
                trust_remote_code=True,
            )
        else:
            # single GPU, DDP, ZeRO-1/2, FSDP shard-aware clone
            self.reference_model = create_reference_model(self.model)

        self.reference_model.eval().requires_grad_(False)
        # wrap it the same way the main model is wrapped (DDP, FSDP, ZeRO, …)
        self.reference_model = self.accelerator.prepare(self.reference_model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        with torch.no_grad():
            ref_logits = self.reference_model(**inputs, use_cache=False).logits

        # Compute KL divergence with proper token shifting and masking
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_ref = ref_logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()

        log_p = F.log_softmax(shift_logits, dim=-1)
        log_ref = F.log_softmax(shift_ref, dim=-1)
        kl_per_token = F.kl_div(log_p, log_ref.exp(), reduction="none").sum(-1)  # (B, T-1)

        mask = (shift_labels != -100).float()  # (B, T-1)
        kl = (kl_per_token * mask).sum() / mask.sum()  # mean over valid tokens

        self._metrics[mode]["kl"].append(kl.item())

        # Add KL regularization to loss
        loss = loss + self.args.kl_lambda * kl
    

        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss
