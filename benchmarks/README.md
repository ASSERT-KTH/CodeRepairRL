# Benchmark Suite

## Setup

Build the benchmark container (includes vLLM and all deps):

```bash
cd benchmarks/
apptainer build benchmark_container.sif benchmark_container.def
```

## Run Flow

Submit a single SLURM job that launches vLLM (optional) and runs the SWE-bench Nano evaluation to produce only `preds.jsonl`.

1) Ensure `CRRL_WORKDIR` is set (used by Apptainer bindings and caches):
```bash
export CRRL_WORKDIR="/proj/<project>/users/<user>"
```

2) Submit the job (starts its own vLLM server):
```bash
sbatch benchmarks/swe_nano_infer_job.sh \
  --base-model Qwen/Qwen3-8B
# optionally add LoRA (adapter name "nano")
sbatch benchmarks/swe_nano_infer_job.sh \
  --base-model Qwen/Qwen3-8B \
  --lora-path /path/to/nano_lora
```

3) If you already have a vLLM server running, skip starting a new one:
```bash
sbatch benchmarks/swe_nano_infer_job.sh \
  --no-server \
  --model-name hosted_vllm/Qwen/Qwen3-8B
```

Outputs: `benchmarks/swe_bench/results_nano/preds.jsonl`

### SWE-bench (two-phase: GPU predict -> CPU evaluate)

On the GPU node (to generate predictions JSONL):
```bash
bash benchmarks/swe_bench/swe_nano_job.sh --model-name hosted_vllm/Qwen/Qwen3-8B
# or
bash benchmarks/swe_bench/swe_minisweagent_job.sh --model-name hosted_vllm/Qwen/Qwen3-8B
# or
bash benchmarks/swe_bench/swe_aider_job.sh --model-name hosted_vllm/Qwen/Qwen3-8B
```

Then on a CPU server with Docker and the SWE-bench harness installed, run evaluation:
```bash
# Install harness once on CPU machine:
pip install swebench

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