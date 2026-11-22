#!/usr/bin/env python3
"""Check the adapter's batched single-token scoring against eval-framework's per-completion path.

Two modes, both for the GPU machine before a sweep:

- ``--tokenizer <dir>`` (no model, seconds): with the real checkpoint's tokenizer, count how many
  prompts of the two lettered tasks qualify for the fast path (every letter a single token that the
  joint prompt+letter tokenisation keeps). Expect all of them; a prompt that does not qualify is scored
  by the reference path instead, which is correct but 30 times slower.
- default (small model, a minute): load a small public model, score spread-out prompts of the lettered
  tasks both ways, and fail if any letter's log-likelihood differs by more than ``--tolerance``, the
  argmax differs anywhere, or any prompt fell back to the reference path.

    uv run python scripts/nonexistent_entities_mcq/check_fast_logprobs.py --tokenizer /workspace/weights/<label>
    uv run python scripts/nonexistent_entities_mcq/check_fast_logprobs.py [--model <small model>] [--dtype bfloat16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

LETTERED_TASKS = ("NonexistentEntitiesMCQ", "NonexistentEntitiesMCQIdk")


def _tasks():
    from abstention_over_training_evals.plugins.nonexistent_entities_mcq import nonexistent_entities_mcq as module

    for name in LETTERED_TASKS:
        yield name, getattr(module, name).with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)


def tokenizer_coverage(tokenizer_dir: str) -> bool:
    from template_formatting.formatter import ConcatFormatter
    from transformers import AutoTokenizer

    from abstention_over_training_evals.adapters.olmo_hf import single_token_ids

    class Shim:  # single_token_ids only needs .tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    shim = Shim()
    print(
        f"tokenizer {tokenizer_dir}: pad={shim.tokenizer.pad_token_id} eos={shim.tokenizer.eos_token_id} bos={shim.tokenizer.bos_token_id}"
    )
    formatter = ConcatFormatter()
    all_ok = True
    for name, task in _tasks():
        total = eligible = 0
        examples = []
        for sample in task.iterate_samples():
            total += 1
            prompt = formatter.format(sample.messages, output_mode="string")
            if single_token_ids(shim, prompt, sample.possible_completions or []) is not None:
                eligible += 1
            elif len(examples) < 3:
                examples.append((sample.subject, sample.id, prompt[-40:], sample.possible_completions))
        print(f"{name}: {eligible} of {total} prompts take the fast path", examples or "")
        all_ok &= eligible == total
    return all_ok


def model_check(model: str, dtype: str, per_task: int, stride: int, batch_size: int, tolerance: float) -> bool:
    import torch
    from eval_framework.llm.huggingface import BaseHFLLM
    from template_formatting.formatter import ConcatFormatter

    from abstention_over_training_evals.adapters.olmo_hf import fast_single_token_logprobs, single_token_ids

    class SmallModel(BaseHFLLM):
        LLM_NAME = model
        DEFAULT_FORMATTER = ConcatFormatter

    llm = SmallModel()
    llm.model = llm.model.to(getattr(torch, dtype))
    ok = True
    for name, task in _tasks():
        samples = [s for i, s in enumerate(task.iterate_samples()) if i % stride == 0][:per_task]
        slow = BaseHFLLM.logprobs(llm, samples)
        fast = fast_single_token_logprobs(llm, samples, batch_size=batch_size)
        worst = 0.0
        flips = 0
        eligible = 0
        for s, f, sample in zip(slow, fast, samples):
            prompt = llm._formatter.format(sample.messages, output_mode="string")
            eligible += single_token_ids(llm, prompt, sample.possible_completions or []) is not None
            assert s.loglikelihoods.keys() == f.loglikelihoods.keys(), "completion sets differ"
            worst = max(worst, max(abs(s.loglikelihoods[k] - f.loglikelihoods[k]) for k in s.loglikelihoods))
            flips += max(s.loglikelihoods, key=s.loglikelihoods.get) != max(f.loglikelihoods, key=f.loglikelihoods.get)
        print(
            f"{name} ({model}, {dtype}): {len(samples)} prompts, fast-eligible {eligible}, worst |diff| {worst:.2e}, argmax flips {flips}"
        )
        ok &= eligible == len(samples) and worst <= tolerance and flips == 0
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default=None, help="Tokenizer-only coverage check on this checkpoint dir.")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument(
        "--dtype", default="float32", help="float32 isolates batching effects; bfloat16 mimics the live run."
    )
    parser.add_argument("--samples", type=int, default=48, help="Prompts per task to compare.")
    parser.add_argument("--stride", type=int, default=7, help="Take every stride-th sample, for item variety.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Max |difference| in log-likelihood.")
    args = parser.parse_args(argv)
    if args.tokenizer:
        ok = tokenizer_coverage(args.tokenizer)
    else:
        ok = model_check(args.model, args.dtype, args.samples, args.stride, args.batch_size, args.tolerance)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
