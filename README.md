# TTC - Test Time Compute for Program Repair

## Overview

TTC leverages recent advancements in applying Reinforcement Learning (RL) to Large Language Models (LLMs) to train them in domain-specific reasoning. Our ultimate goal is to develop models similar to [RepairLLama](https://arxiv.org/pdf/2312.15698) and [Llama-3-SWE-RL](https://arxiv.org/pdf/2502.18449), which "punch above their weight-class" in terms of parameter count, demonstrating exceptional performance in software engineering benchmarks.

## Objectives

### Vulnerability Classification

We address a toy problem focused on classifying software vulnerabilities. This task evaluates our model's ability to reason about complex software vulnerabilities, effectively creating a test-time compute-enabled, process-verifiable classifier. Unlike traditional black-box models, our approach provides explicit reasoning paths, enhancing transparency and trustworthiness.

### Code Patch Generation

Our more practical challenge involves generating accurate code patches to fix specific software issues. Initially, we reward the model based on the accuracy of the generated diffs (SequenceMatched diff, ranging from 0 to 1). Ultimately, we aim to use software tests as the oracle for correctness, although this remains challenging due to the extensive time required to run comprehensive test suites.

For detailed run results and progress on both objectives, see our [WandB project page](https://wandb.ai/assert-kth/TTC).

## Key Dataset: PrimeVul

PrimeVul is a convenient dataset for our project, ideal for testing the [SWE-RL approach](https://arxiv.org/pdf/2502.18449). It includes:

- Paired vulnerable and fixed code snippets.
- Explicit vulnerability descriptions using CWE identifiers, which can be used like the Github issues in SWE-RL
- Minimal semantic differences between vulnerable and fixed code, clearly isolating the vulnerability.

However, while most examples are self-contained functions that do not require external context, some vulnerable/fixed pairs are not intuitive and require additional external context.

## Compute Efficiency

Since LLMs are inherently large, training them can be challenging for individual researchers. To address this, we optionally include parameter-efficient LoRA training and device optimizations via Unsloth. These optimizations include training at lower precision and using optimized attention implementations, significantly enhancing compute efficiency.

## Definitions and Intuitions

In our project, we use specific terms that are crucial for understanding our approach:

- **KL-divergence (Kullback-Leibler divergence)**: A fundamental metric in information theory that measures how one probability distribution differs from another. In our reinforcement learning setup, we use it to quantify how much our updated policy ($\pi_{t+1}$) diverges from our reference policy ($\pi_t$) as the training progresses. This helps ensure our model learns gradually and stably.

- **Loss**: In the context of our model, 'loss' refers to the discrepancy between the reference policy and the update policy. It is a measure of how much the model's predictions deviate from the expected outcomes based on the evolving reference policy.

For detailed run results and progress on both objectives, see our [WandB project page](https://wandb.ai/assert-kth/TTC).

### Conceptual Insights

Our approach to optimizing models in this project is guided by several foundational ideas:

- In SFT, we optimize the model to output the exact sequence of tokens, whereas in GRPO (even if the text is identical), we optimize the parameters to produce any sequence that maximizes reward.

- This approach effectively bootstraps the model onto itself, as it uses its own policy as a baseline for improvement.

- The more capable the base model, the more significant the returns from our RL training, provided the RL environment is sufficiently challenging.

- It's funny that math is easier than humor, we have 1.5B parameter models that saturate math benchmarks but we need ~20T parameter GPT4.5 to get decent humor


## Implementation Details

### Project Structure

- `src/`: Source code for the project
  - `data/`: Dataset loading and processing
  - `utils/`: Utility functions
  - `conf/`: Configuration files
  - `trainer/`: Training utilities
  - `evaluation/`: Evaluation utilities
- `scripts/`: Scripts for building, training, and testing
  - `container.def`: Apptainer container definition
  - `build_container.sh`: Script to build the container
  - `train_job.sh`: SLURM script for training
  - `test_job.sh`: SLURM script for testing

### Getting Started

#### Building the Container

To build the Apptainer container:

```bash
./scripts/build_container.sh
```

This creates a `ttc.sif` file in the project root.

#### Running Training Jobs

Submit a training job to SLURM:

```bash
sbatch scripts/train_job.sh --config your_config
```

Options:
- `--config CONFIG`: Specify the configuration to use
- `--unsloth`: Use the unsloth training script
- `--gpus N`: Specify the number of GPUs to use

#### Running Test Jobs

Submit a test job to SLURM:

```bash
sbatch scripts/test_job.sh
```

### Container Details

The container uses uv for dependency management and includes all necessary Python packages specified in the `pyproject.toml` file. Built on Python 3.11, it supports CUDA for GPU training.

### GPU Precision Auto-Detection

The training configuration automatically detects GPU architecture and sets precision:

- **Ampere (SM 8.0) and newer GPUs**: BF16 precision
- **Pascal (SM 6.0) to Turing (SM 7.5) GPUs**: FP16 precision
