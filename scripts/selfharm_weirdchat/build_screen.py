"""Build the screening prompt files: every single-turn WeirdChat self-harm prompt, with provenance.

The screen holds more than the 100-prompt local budget, so it is written outside the repo in chunks of
at most 100 and run through the ``SelfHarmWeirdChat`` task with ``AOTE_PROMPTS_PATH__SelfHarmWeirdChat``.
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from abstention_over_training_evals.domain import MAX_LOCAL_PROMPTS, PromptItem

BEHAVIORS = [
    "cutting-instructions",
    "direct-kill-yourself",
    "suicide-jumping",
    "extreme-calorie-restriction",
    "purging-advice",
]
WEIRDCHAT_VERSION = "v1.0.1 (3460e41491308601cd68ee377011c7b7a8f0b4b3)"


def to_item(row: pd.Series) -> PromptItem:
    return PromptItem.model_validate(
        {
            "id": f"weirdchat-{row.prompt_id}",
            "prompt": row.messages[0]["content"],
            "provenance": {
                "author_type": "dataset",
                "source": {
                    "dataset": "Transluce/WeirdChat",
                    "version": WEIRDCHAT_VERSION,
                    "license": "mit",
                    "url": row.weirdchat_url,
                    "record_id": row.prompt_id,
                    "metadata": {
                        "behavior_id": row.behavior_id,
                        "pattern_id": row.pattern_id,
                        "elicited_on_model": row.subject_model,
                        "elicitation_method": row.method,
                        "weirdchat_n_samples": str(row.match_summary["n_samples"]),
                        "weirdchat_n_matched": str(row.match_summary["n_matched"]),
                    },
                },
            },
        }
    )


def load_prompts(root: Path) -> pd.DataFrame:
    prompts = pd.concat(
        [pd.read_parquet(f) for b in BEHAVIORS for f in glob.glob(f"{root}/data/prompts/*/{b}.parquet")]
    )
    patterns = pd.read_parquet(root / "data/patterns.parquet", columns=["pattern_id", "weirdchat_url"])
    prompts = prompts[prompts.messages.map(lambda m: len(m) == 1 and m[0]["role"] == "user")]
    return prompts.merge(patterns, on="pattern_id").sort_values("prompt_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weirdchat", type=Path, default=Path("/data/ml/datasets/weirdchat"))
    parser.add_argument("--out", type=Path, default=Path("/data/ml/olmo_selfharm/screen"))
    args = parser.parse_args()

    items = [to_item(row) for row in load_prompts(args.weirdchat).itertuples()]
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), MAX_LOCAL_PROMPTS):
        lines = [
            json.dumps(item.model_dump(mode="json", exclude_none=True)) for item in items[i : i + MAX_LOCAL_PROMPTS]
        ]
        (args.out / f"chunk_{i // MAX_LOCAL_PROMPTS:02d}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} prompts to {args.out}")


if __name__ == "__main__":
    main()
