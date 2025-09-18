# Benchmark Suite

## Setup

Build the benchmark container (includes vLLM and all deps):

```bash
cd benchmarks/
apptainer build benchmark_container.sif benchmark_container.def
```

## Run Flow

### Inference (on the GPU cluster)

Submit a single SLURM job that launches vLLM (optional) and runs the SWE-bench Nano evaluation to produce only `preds.jsonl`.

1) Ensure `CRRL_WORKDIR` is set (used by Apptainer bindings and caches):
```bash
export CRRL_WORKDIR="/proj/<project>/users/<user>"
```

2) Submit the job (starts its own vLLM server):
```bash
sbatch benchmarks/swe_bench_nano_infer_job.sh \
  --base-model Qwen/Qwen3-8B
# With LoRA (adapter name "nano"); include adapter path for tagging
sbatch benchmarks/swe_bench_nano_infer_job.sh \
  --base-model Qwen/Qwen3-8B \
  --lora-path /path/to/nano_lora
```

Outputs are organized per model and scaffold under:

```
benchmarks/swe_bench/results/<scaffold>-<model_tag>/shard_<array_id>/preds.jsonl
```

Examples:
- Base model (no LoRA): `benchmarks/swe_bench/results/nano-agent-hosted_vllm__Qwen__Qwen3-8B/shard_0/preds.jsonl`
- With LoRA (adapter basename "nano_lora"):
  `benchmarks/swe_bench/results/nano-agent-Qwen__Qwen3-8B__lora__nano_lora/shard_0/preds.jsonl`

Notes:
- `<model_tag>` is sanitized to be filesystem-safe. For base-only runs it derives from `hosted_vllm/<BASE_MODEL>`; for LoRA runs it derives from `<BASE_MODEL>__lora__<adapter_basename>`.
- The job supports SLURM arrays. Each task writes to its own `shard_<array_id>` and auto-selects a dataset slice (default shard size 50).

### Eval (on the CPU server)

Then on a CPU server with Docker and the SWE-bench harness installed, run evaluation:
```bash
# Install harness once on CPU machine:
pip install swebench

# Evaluate a preds.jsonl file:
benchmarks/swe_bench/run_harness_eval.sh \
  --subset verified --split test \
  --preds /PATH/TO/preds.jsonl \
  --run-id nano_test
```

## Results

Outputs are saved under:
- `benchmarks/tau_bench/results/`
- `benchmarks/swe_bench/results/<scaffold>-<model_tag>/shard_<array_id>/`
Evaluation logs and reports (from the harness) will be written under the harness working directory (e.g., `evaluation_results/`). See the harness docs for details.