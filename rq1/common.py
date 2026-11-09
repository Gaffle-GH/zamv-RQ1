"""Shared paths, model registry, conditions and JSONL helpers."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, RUNS, EXPORTS = ROOT / "data", ROOT / "runs", ROOT / "exports"

# Model card thinking-mode settings (Qwen3: T=0.6, top_p=0.95, top_k=20, min_p=0; R1 distills: T=0.6, top_p=0.95).
QWEN3_THINK = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
QWEN3_NOTHINK = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)

# Display name (the `model` field everyone joins on) -> where to load it from.
# `mlx` is a 4-bit Apple-silicon build for local pilots; `hf` is the bf16 checkpoint for vLLM on A100s.
MODELS = {
    "Qwen3-4B": dict(hf="Qwen/Qwen3-4B", mlx="mlx-community/Qwen3-4B-4bit", size_b=4, tp=1, sampling=QWEN3_THINK),
    "Qwen3-8B": dict(hf="Qwen/Qwen3-8B", mlx="mlx-community/Qwen3-8B-4bit", size_b=8, tp=1, sampling=QWEN3_THINK),
    "Qwen3-14B": dict(hf="Qwen/Qwen3-14B", mlx="mlx-community/Qwen3-14B-4bit", size_b=14, tp=1, sampling=QWEN3_THINK),
    "Qwen3-32B": dict(hf="Qwen/Qwen3-32B", mlx="mlx-community/Qwen3-32B-4bit", size_b=32, tp=2, sampling=QWEN3_THINK),
    # Second family (Llama base). Its chat template already opens <think>.
    "R1-Distill-Llama-8B": dict(hf="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                                mlx="mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit", size_b=8, tp=1,
                                sampling=dict(temperature=0.6, top_p=0.95, top_k=0, min_p=0.0)),
}

ATTACKS = ["A1", "A2", "A3", "A4"]
# HONEST and FALSE_ALARM feed RQ3 (Angel's simulator maps them to honest / attack peers).
CONDITIONS = ["A0", *ATTACKS, "HONEST", "FALSE_ALARM"]
CONDITION_NAMES = {"A0": "No peer", "A1": "Bare claim", "A2": "Authority", "A3": "Fake evidence",
                   "A4": "Confident argument", "HONEST": "Honest peer", "FALSE_ALARM": "False alarm"}


def applies(condition: str, is_buggy: bool) -> bool:
    """FALSE_ALARM (peer claims a bug in correct code) only makes sense on correct items."""
    return not (condition == "FALSE_ALARM" and is_buggy)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
