#!/usr/bin/env python3
"""Pilot: how a Think checkpoint answers the lettered MCQ prompts when it may reason first.

For a few items, in both lettered forms (4 options, and 5 with "I don't know"), the question and
options are sent through the chat template and the model generates freely (sampling at the model's
recommended temperature, a large token budget), so we learn how long the reasoning trace is, how the
final answer is phrased and whether it is parseable. A second, cheap condition closes the thinking
block immediately (``<think></think>``) and records the first tokens and the top next-token
probabilities, to see whether an immediate chat-format answer could be scored by log-likelihood.

    uv run --no-sync python scripts/nonexistent_entities_mcq/pilot_think.py --checkpoint think \\
        --items pharma-01 elec-01 osha-02 proc-02 ctrl-pharma-01 --out /workspace/pilot/pilot.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

INSTRUCTION = "Answer with the letter of the best option."
LETTER_RE = re.compile(
    r"(?:answer\s*(?:is|:)?\s*\**\s*\(?([A-E])\)?|^\s*\(?([A-E])\)?(?:[.):\s]|$)|\*\*([A-E])\*\*)", re.I | re.M
)


def parse_letter(answer: str) -> str | None:
    """The letter the answer settles on: the last explicit 'Answer: X' / bold / line-initial letter."""
    found = [next(g for g in m.groups() if g) for m in LETTER_RE.finditer(answer)]
    return found[-1].upper() if found else None


def build_prompts(items: list[str]) -> list[dict]:
    from abstention_over_training_evals.plugins.nonexistent_entities_mcq.nonexistent_entities_mcq import (
        NonexistentEntitiesMCQ,
        NonexistentEntitiesMCQIdk,
    )

    prompts = []
    for cls, task in ((NonexistentEntitiesMCQ, "MCQ"), (NonexistentEntitiesMCQIdk, "MCQIdk")):
        t = cls.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
        rows = {r["item_id"]: r for r in t.sample_manifest() if r["order"] == 0}
        for item in items:
            r = rows[item]
            letters = "ABCDE"[: len(r["choices"])]
            options = "\n".join(f"{letter}. {c}" for letter, c in zip(letters, r["choices"]))
            prompts.append(
                {
                    "item_id": item,
                    "task": task,
                    "user": f"Question: {r['question']}\n{options}\n\n{INSTRUCTION}",
                    "choices": r["choices"],
                    "gold_letter": r["gold"].strip(),
                    "abstention_letter": (r["abstention"] or "").strip() or None,
                }
            )
    return prompts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="think")
    ap.add_argument("--items", nargs="+", default=["pharma-01", "elec-01", "osha-02", "proc-02", "ctrl-pharma-01"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from abstention_over_training_evals.adapters.olmo_weights import download_weights, parse_checkpoint

    ckpt = parse_checkpoint(args.checkpoint)
    local = download_weights(ckpt, args.weights_dir)
    tok = AutoTokenizer.from_pretrained(local)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(local, device_map="auto", dtype=torch.bfloat16)
    prompts = build_prompts(args.items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # Condition A: free generation with reasoning, all prompts batched, sampled at the recommended settings.
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p["user"]}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with args.out.open("w", encoding="utf-8") as f:
        for sample in range(args.samples):
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                )
            elapsed = time.time() - t0
            for p, text, ids in zip(prompts, texts, out):
                gen = ids[enc["input_ids"].shape[1] :]
                gen = gen[gen != tok.pad_token_id]
                completion = tok.decode(gen, skip_special_tokens=False)
                think, sep, answer = completion.partition("</think>")
                answer = answer.replace(tok.eos_token, "")
                record = {
                    **p,
                    "condition": "reasoned",
                    "sample": sample,
                    "prompt": text,
                    "think": think,
                    "answer": answer if sep else None,
                    "completion_tokens": int(gen.numel()),
                    "think_tokens": len(tok.encode(think, add_special_tokens=False)),
                    "finished": bool(sep) and completion.rstrip().endswith(tok.eos_token),
                    "parsed_letter": parse_letter(answer) if sep else None,
                    "batch_seconds": round(elapsed, 1),
                }
                record["correct"] = record["parsed_letter"] == p["gold_letter"] if record["parsed_letter"] else None
                f.write(json.dumps(record) + "\n")
                print(
                    f"[reasoned] {p['task']:6s} {p['item_id']:14s} think_tokens={record['think_tokens']:5d} "
                    f"finished={record['finished']} letter={record['parsed_letter']} gold={p['gold_letter']} "
                    f"idk={p['abstention_letter']}"
                )

        # Condition B: the thinking block closed immediately; greedy continuation plus top next tokens.
        for p, text in zip(prompts, texts):
            prefilled = text + "</think>\n\n"
            ids = tok(prefilled, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                logits = model(**ids).logits[0, -1].float()
                probs = torch.softmax(logits, dim=-1)
                top = torch.topk(probs, 10)
                gen = model.generate(**ids, do_sample=False, max_new_tokens=24, pad_token_id=tok.pad_token_id)
            continuation = tok.decode(gen[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
            letter_probs = {}
            for letter in "ABCDE"[: len(p["choices"])]:
                for variant in (letter, f" {letter}"):
                    tid = tok.encode(variant, add_special_tokens=False)
                    if len(tid) == 1:
                        letter_probs[repr(variant)] = round(float(probs[tid[0]]), 4)
            record = {
                **p,
                "condition": "no_think",
                "prompt": prefilled,
                "continuation": continuation,
                "top_next_tokens": [
                    (tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)
                ],
                "letter_probs": letter_probs,
                "parsed_letter": parse_letter(continuation),
            }
            record["correct"] = record["parsed_letter"] == p["gold_letter"] if record["parsed_letter"] else None
            f.write(json.dumps(record) + "\n")
            print(f"[no_think] {p['task']:6s} {p['item_id']:14s} -> {continuation[:60]!r} letters={letter_probs}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
