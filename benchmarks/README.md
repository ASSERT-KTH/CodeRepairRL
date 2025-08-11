# Benchmark Suite

## Setup

First, build the benchmark container:

```bash
cd benchmarks/
apptainer build benchmark_container.sif benchmark_container.def
```

This container includes vLLM with flash-attn and all benchmark dependencies.

## Running Benchmarks

Each benchmark is a SLURM job that:
1. Launches a vLLM server with the trained LoRA adapter
2. Runs before/after comparisons against baseline models
3. Saves results to JSON

### Tau Bench
```bash
sbatch tau_bench/tau_job.sh /path/to/lora/adapter
```

### Terminal Bench
```bash
sbatch terminal_bench/terminal_job.sh /path/to/lora/adapter
```

### SWE-bench (with nano_agent)
```bash
sbatch swe_bench/swe_nano_job.sh /path/to/lora/adapter
```

### SWE-bench (with minisweagent)
```bash
sbatch swe_bench/swe_minisweagent_job.sh /path/to/lora/adapter
```

## vLLM Server Details

The jobs launch vLLM with LoRA support:
- Base model: Specified in job (e.g., Qwen/Qwen2.5-14B-Instruct)
- LoRA adapter: Loaded with `--enable-lora --lora-modules nano=/path/to/adapter`
- Access: Before uses base model name, after uses "nano" as model name

## Results

Results are saved in each benchmark's subdirectory:
- `tau_bench/results/` - Tau bench metrics
- `terminal_bench/results/` - Terminal bench scores
- `swe_bench/results_nano/` - SWE-bench with nano_agent predictions