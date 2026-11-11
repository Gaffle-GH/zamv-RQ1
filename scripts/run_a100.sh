#!/usr/bin/env bash
# Full RQ1 run on A100s: A4 attacks by Qwen3-32B, then 5 verifiers x 7 conditions x 3 seeds, then exports.
# Run from the repo root. Resumable: re-running skips calls already in runs/vllm/<model>.jsonl.
# Faster: run the 1-GPU models in parallel, one per GPU, e.g.
#   CUDA_VISIBLE_DEVICES=2 python -m rq1.harness --model Qwen3-8B --backend vllm &
set -euo pipefail

uv sync --extra vllm
PY=".venv/bin/python"

[ -f data/items.jsonl ] || $PY -m rq1.dataset
$PY -m rq1.attacks --generator Qwen3-32B --backend vllm          # needs 2 GPUs (tp=2)

# Smoke test first: 2 items, 1 seed. Check the printed tok/s and that nothing is unparsed.
$PY -m rq1.harness --model Qwen3-4B --backend vllm --n-items 2 --seeds 0 --batch-size 512

MODELS_TO_RUN="Qwen3-4B Qwen3-8B Qwen3-14B R1-Distill-Llama-8B Qwen3-32B"
for m in $MODELS_TO_RUN; do
  $PY -m rq1.harness --model "$m" --backend vllm --batch-size 512
done

# Held-out A0 seeds for the flip-rate baseline (+800 calls per model, about 11% more).
# Seeds 0-2 pick the eligible items; measuring the baseline on those same seeds biases it low.
# Optional: export falls back to seeds 0-2 (and says so) for any model without them.
for m in $MODELS_TO_RUN; do
  $PY -m rq1.harness --model "$m" --backend vllm --batch-size 512 --seeds 3,4 --conditions A0
done

$PY -m rq1.export --runs runs/vllm --out exports
