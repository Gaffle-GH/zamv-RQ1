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

for m in Qwen3-4B Qwen3-8B Qwen3-14B R1-Distill-Llama-8B Qwen3-32B; do
  $PY -m rq1.harness --model "$m" --backend vllm --batch-size 512
done

$PY -m rq1.export --runs runs/vllm --out exports
