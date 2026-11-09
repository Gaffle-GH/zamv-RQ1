"""Two interchangeable generation backends with the same interface.

  generate(prompt_ids, seeds, sampling, max_tokens) -> [{"text", "n_tokens", "hit_limit"}]
  score(prefix_ids, conts) -> [[sum log P(cont | prefix) for cont in row] for row]

MLX: Apple-silicon pilots (4-bit). Batches share one RNG seed (seeds[0]); per-request seeds are not possible.
vLLM: A100 runs (bf16). Per-request seeds. Untested on this machine (no CUDA) -- run the smoke test first.
"""

from huggingface_hub import HfApi, snapshot_download


def revision_of(repo):
    try:
        return HfApi().model_info(repo).sha
    except Exception:
        return None


class MLXBackend:
    name = "mlx"

    def __init__(self, repo):
        from mlx_lm import load
        self.repo, self.revision = repo, revision_of(repo)
        self.model, self.tok = load(snapshot_download(repo, revision=self.revision))

    def generate(self, prompt_ids, seeds, sampling, max_tokens, batch_size=16):
        import mlx.core as mx
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=sampling["temperature"], top_p=sampling["top_p"],
                               top_k=sampling.get("top_k", 0), min_p=sampling.get("min_p", 0.0))
        mx.random.seed(seeds[0])
        r = batch_generate(self.model, self.tok, prompt_ids, max_tokens=max_tokens, sampler=sampler,
                           return_token_ids=True, completion_batch_size=batch_size, prefill_batch_size=min(8, batch_size))
        return [dict(text=t, n_tokens=len(ids), hit_limit=len(ids) >= max_tokens)
                for t, ids in zip(r.texts, r.token_ids)]

    def score(self, prefix_ids, conts):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        out = []
        for prefix, row in zip(prefix_ids, conts):
            cache = make_prompt_cache(self.model)
            logits = self.model(mx.array(prefix)[None], cache=cache)[0, -1]
            first = logits - mx.logsumexp(logits)
            scores = []
            for c in row:
                lp = first[c[0]].item()
                if len(c) > 1:
                    lg = self.model(mx.array(c[:-1])[None], cache=cache)[0]
                    lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                    lp += sum(lg[j, c[j + 1]].item() for j in range(len(c) - 1))
                    trim_prompt_cache(cache, len(c) - 1)
                scores.append(lp)
            out.append(scores)
            mx.clear_cache()
        return out


class VLLMBackend:
    name = "vllm"

    def __init__(self, repo, tp=1, max_model_len=16384):
        from vllm import LLM
        self.repo, self.revision = repo, revision_of(repo)
        self.llm = LLM(model=repo, revision=self.revision, tensor_parallel_size=tp, max_model_len=max_model_len,
                       enable_prefix_caching=True, gpu_memory_utilization=0.90, seed=0)
        self.tok = self.llm.get_tokenizer()

    def generate(self, prompt_ids, seeds, sampling, max_tokens, batch_size=None):
        from vllm import SamplingParams
        params = [SamplingParams(temperature=sampling["temperature"], top_p=sampling["top_p"],
                                 top_k=sampling.get("top_k", 0) or -1, min_p=sampling.get("min_p", 0.0),
                                 max_tokens=max_tokens, seed=s) for s in seeds]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompt_ids], params, use_tqdm=False)
        return [dict(text=o.outputs[0].text, n_tokens=len(o.outputs[0].token_ids),
                     hit_limit=o.outputs[0].finish_reason == "length") for o in outs]

    def score(self, prefix_ids, conts, k=20):
        # One max_tokens=1 request per continuation position; prefix caching makes these cheap.
        # A token outside the top-k gets the k-th logprob (an upper bound).
        from vllm import SamplingParams
        reqs = sorted({tuple(p) + tuple(c[:j]) for p, row in zip(prefix_ids, conts) for c in row for j in range(len(c))})
        outs = self.llm.generate([{"prompt_token_ids": list(r)} for r in reqs],
                                 SamplingParams(max_tokens=1, temperature=0, logprobs=k), use_tqdm=False)
        top = {r: {t: lp.logprob for t, lp in o.outputs[0].logprobs[0].items()} for r, o in zip(reqs, outs)}
        res = []
        for p, row in zip(prefix_ids, conts):
            scores = []
            for c in row:
                lp = 0.0
                for j in range(len(c)):
                    d = top[tuple(p) + tuple(c[:j])]
                    lp += d.get(c[j], min(d.values()))
                scores.append(lp)
            res.append(scores)
        return res


def load_backend(name, spec):
    return MLXBackend(spec["mlx"]) if name == "mlx" else VLLMBackend(spec["hf"], tp=spec.get("tp", 1))
