"""Load eval-framework outputs for the SelfHarmWeirdChat task into one row per (prompt, repeat)."""

import ast
import glob
import json
from pathlib import Path

import pandas as pd


def load_prompts(paths: list[Path]) -> pd.DataFrame:
    rows = [json.loads(line) for p in paths for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["behavior_id"] = df.provenance.map(lambda p: p["source"]["metadata"]["behavior_id"])
    df["weirdchat_rate"] = df.provenance.map(
        lambda p: (
            int(p["source"]["metadata"]["weirdchat_n_matched"]) / int(p["source"]["metadata"]["weirdchat_n_samples"])
        )
    )
    return df


def user_text(messages: list[dict[str, str]] | str) -> str:
    msgs = messages if isinstance(messages, list) else ast.literal_eval(messages)
    return next(m["content"] for m in msgs if m["role"] == "user")


def load_run(output_dir: Path) -> pd.DataFrame:
    """Join each run directory's completions and judge verdicts; prompts are matched by their text."""
    frames = []
    for run in glob.glob(f"{output_dir}/**/output.jsonl", recursive=True):
        run_dir = Path(run).parent
        out = pd.read_json(run_dir / "output.jsonl", lines=True, dtype=False)
        res = pd.read_json(run_dir / "results.jsonl", lines=True, dtype=False)
        res = res[res.metric_name == "Unsafe Elicitation Rate"]
        out["prompt_text"] = out.messages.map(
            lambda m: next(
                x["content"] for x in (m if isinstance(m, list) else ast.literal_eval(m)) if x["role"] == "user"
            )
        )
        out["repeat"] = out.groupby("id").cumcount()
        res["repeat"] = res.groupby("id").cumcount()
        frames.append(out.merge(res[["id", "repeat", "value", "llm_judge_response"]], on=["id", "repeat"]))
    df = pd.concat(frames, ignore_index=True)
    df["unsafe"] = pd.to_numeric(df.value, errors="coerce")
    return df


def joined(prompt_paths: list[Path], output_dir: Path) -> pd.DataFrame:
    prompts = load_prompts(prompt_paths).rename(columns={"prompt": "prompt_text"})
    return load_run(output_dir).merge(prompts, on="prompt_text", how="left", validate="many_to_one")
