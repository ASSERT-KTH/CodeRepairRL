# Benchmark Suite

## Setup

Build the benchmark container (includes vLLM and all deps):

```bash
cd benchmarks/
apptainer build benchmark_container.sif benchmark_container.def
```

## Run Flow (no SLURM)

1) Allocate a GPU node:
```bash
salloc --nodes 1 --gpus 1 -C "fat" --time 08:00:00
```

2) Start tmux and launch vLLM in one pane on port 8000:
```bash
srun --pty bash -l
tmux
```
then run server
```bash
bash benchmarks/vllm.sh Qwen/Qwen3-14B /path/to/nano_lora
# or without LoRA:
bash benchmarks/vllm.sh Qwen/Qwen3-14B
```

3) In another pane, run a benchmark (assumes vLLM is up on http://localhost:8000/v1):

- Tau Bench
```bash
bash benchmarks/tau_bench/tau_job.sh
```

- SWE-bench (nano_agent)
```bash
bash benchmarks/swe_bench/swe_nano_job.sh
```

- SWE-bench (minisweagent)
```bash
bash benchmarks/swe_bench/swe_minisweagent_job.sh
```

## Results

Outputs are saved under:
- `benchmarks/tau_bench/results/`
- `benchmarks/swe_bench/results_nano/`
- `benchmarks/swe_bench/results_mini/`