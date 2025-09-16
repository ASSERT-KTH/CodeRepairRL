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

### SWE-bench (two-phase: GPU predict -> CPU evaluate)

On the GPU node (to generate predictions JSONL):
```bash
bash benchmarks/swe_bench/swe_nano_job.sh
# or
bash benchmarks/swe_bench/swe_minisweagent_job.sh
# or
bash benchmarks/swe_bench/swe_aider_job.sh
```

Then on a CPU server with Docker and the SWE-bench harness installed, run evaluation:
```bash
# Install harness once on CPU machine:
# git clone https://github.com/princeton-nlp/SWE-bench
# cd SWE-bench && pip install -e .

# Evaluate a preds.jsonl file (example for nano):
benchmarks/swe_bench/run_harness_eval.sh \
  --subset verified --split test \
  --preds /ABS/PATH/TO/repo/benchmarks/swe_bench/results_nano/preds.jsonl \
  --run-id nano_test
```

## Results

Outputs are saved under:
- `benchmarks/tau_bench/results/`
- `benchmarks/swe_bench/results_nano/`
- `benchmarks/swe_bench/results_mini/`
Evaluation logs and reports (from the harness) will be written under the harness working directory (e.g., `evaluation_results/`). See the harness docs for details.