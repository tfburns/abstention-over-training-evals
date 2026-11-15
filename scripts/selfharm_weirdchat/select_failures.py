"""Pick the screened prompts Olmo 3 32B Think answered unsafely and write them as the task's local JSONL.

If more than the local budget failed, a seeded random subset is kept, stratified by WeirdChat behavior.
"""

import argparse
import json
import sys
from pathlib import Path

from results import joined

from abstention_over_training_evals.domain import MAX_LOCAL_PROMPTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, default=Path("/data/ml/olmo_selfharm/screen"))
    parser.add_argument("--screen-out", type=Path, default=Path("/data/ml/olmo_selfharm/screen_out"))
    parser.add_argument("--out", type=Path, default=Path("data/local/selfharm_weirdchat.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = joined(sorted(args.screen.glob("chunk_*.jsonl")), args.screen_out)
    if df.id_y.isna().any():
        sys.exit(f"{df.id_y.isna().sum()} completions did not match a screened prompt")
    failed = df[df.unsafe == 1.0]
    print(f"screened {len(df)}, judged {df.unsafe.notna().sum()}, unsafe {len(failed)}")
    print(df.groupby("behavior_id").unsafe.agg(["count", "sum", "mean"]).to_string())
    if len(failed) > MAX_LOCAL_PROMPTS:
        frac = MAX_LOCAL_PROMPTS / len(failed)
        failed = failed.groupby("behavior_id", group_keys=False).apply(
            lambda g: g.sample(n=round(len(g) * frac), random_state=args.seed)
        )
        failed = failed.head(MAX_LOCAL_PROMPTS)
    rows = [
        json.dumps({"id": r.id_y, "prompt": r.prompt_text, "provenance": r.provenance})
        for r in failed.sort_values("id_y").itertuples()
    ]
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
