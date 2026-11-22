#!/usr/bin/env python3
r"""Run the NonexistentEntitiesMCQ tasks on a list of Olmo 3 32B checkpoints, one download per checkpoint.

On the GPU machine, after ``scripts/nonexistent_entities_mcq/pod_setup.sh``::

    uv run python scripts/nonexistent_entities_mcq/run_checkpoints.py --checkpoints base sft dpo think
    uv run python scripts/nonexistent_entities_mcq/run_checkpoints.py \\
        --checkpoints base@stage1-step72000 base@stage1-step335000 \\
        --tasks NonexistentEntitiesMCQ NonexistentEntitiesMCQIdk

Each checkpoint is downloaded once (about 65 GB), loaded once, scored on every requested task, then its
weights are deleted (``--keep-weights`` keeps them; they are also kept when any of its tasks failed, so
a retry does not download again). Every run is appended to ``<output-dir>/runs.jsonl`` with its timing
and output directory; a run counts as ``ok`` only when every sample scored without error (the
framework is told to fail on the first error rather than store error rows). Rerunning with the same
arguments skips the (checkpoint, task) pairs already recorded as ``ok``, so an interrupted sweep
resumes. ``--dry-run`` prints the plan only and needs no GPU stack.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO / "src" / "abstention_over_training_evals" / "plugins" / "nonexistent_entities_mcq"
ALL_TASKS = (
    "NonexistentEntitiesMCQ",
    "NonexistentEntitiesMCQIdk",
    "NonexistentEntitiesMCQCloze",
    "NonexistentEntitiesMCQClozeIdk",
)
RUNS_FILE = "runs.jsonl"

sys.path.insert(0, str(REPO / "src"))

from abstention_over_training_evals.adapters.olmo_weights import (  # noqa: E402
    checkpoint_label,
    parse_checkpoint,
    remove_weights,
    weights_root,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint specs: base, sft, dpo, think, or a catalogued Hub id, each optionally @<revision>.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(ALL_TASKS), choices=ALL_TASKS, metavar="TASK")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "nonexistent_entities_mcq")
    parser.add_argument("--weights-dir", type=Path, default=None, help="Defaults to $AOTE_WEIGHTS_DIR.")
    parser.add_argument(
        "--keep-weights", action="store_true", help="Do not delete a checkpoint's weights after scoring."
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Cap on samples per subject (debugging only).")
    parser.add_argument("--batch-size", type=int, default=1, help="eval-framework samples per logprobs call.")
    parser.add_argument(
        "--letter-batch-size", type=int, default=8, help="Prompts per forward pass when scoring letters."
    )
    parser.add_argument("--no-fast-letters", action="store_true", help="Score letters one pass per letter (slow).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return parser.parse_args(argv)


def load_done(runs_file: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if runs_file.exists():
        for line in runs_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("status") == "ok":
                    done.add((record["checkpoint"], record["task"]))
    return done


def append_record(runs_file: Path, record: dict[str, Any]) -> None:
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    with runs_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoints]
    runs_file = args.output_dir / RUNS_FILE
    done = load_done(runs_file)
    plan = [(c, [t for t in args.tasks if (checkpoint_label(c), t) not in done]) for c in checkpoints]

    print(f"weights dir: {weights_root(args.weights_dir)}")
    print(f"output dir:  {args.output_dir}")
    for checkpoint, tasks in plan:
        status = ", ".join(tasks) if tasks else "nothing left (already in runs.jsonl)"
        print(f"  {checkpoint_label(checkpoint)}: {status}")
    if args.dry_run:
        return 0

    import torch
    from eval_framework.main import main as run_eval
    from eval_framework.result_processors.result_processor import generate_output_dir
    from eval_framework.tasks.eval_config import EvalConfig
    from eval_framework.tasks.task_loader import load_extra_tasks

    from abstention_over_training_evals.adapters.olmo_hf import OlmoHFCheckpoint

    load_extra_tasks(PLUGIN_DIR)
    failures = 0
    for checkpoint, tasks in plan:
        if not tasks:
            continue
        label = checkpoint_label(checkpoint)
        t0 = time.time()
        try:
            llm = OlmoHFCheckpoint(
                checkpoint,
                weights_dir=str(args.weights_dir) if args.weights_dir else None,
                fast_letters=not args.no_fast_letters,
                letter_batch_size=args.letter_batch_size,
            )
        except Exception as exc:  # a download or load failure must not end the sweep
            failures += 1
            message = f"{type(exc).__name__}: {exc}"
            print(f"[{label}] FAILED to download or load: {message}")
            traceback.print_exc()
            for task in tasks:
                append_record(
                    runs_file,
                    {
                        "checkpoint": label,
                        "model_id": checkpoint.model_id,
                        "revision": checkpoint.revision,
                        "stage": checkpoint.stage.value,
                        "task": task,
                        "status": "error",
                        "seconds": round(time.time() - t0, 1),
                        "error": f"download/load: {message}",
                    },
                )
            continue
        load_seconds = time.time() - t0
        print(f"[{label}] downloaded and loaded in {load_seconds:.0f}s")
        checkpoint_ok = True
        for task in tasks:
            config = EvalConfig(
                llm_class=OlmoHFCheckpoint,
                llm_args={"checkpoint": checkpoint.model_id, "revision": checkpoint.revision},
                task_name=task,
                num_fewshot=0,
                num_samples=args.num_samples,
                output_dir=args.output_dir / label,
                batch_size=args.batch_size if args.no_fast_letters else max(args.batch_size, args.letter_batch_size),
                wandb_upload_results=False,
                fail_on_error=True,
                description=f"{label} on {task}",
            )
            record: dict[str, Any] = {
                "checkpoint": label,
                "model_id": checkpoint.model_id,
                "revision": checkpoint.revision,
                "stage": checkpoint.stage.value,
                "task": task,
                "load_seconds": round(load_seconds, 1),
                "fast_letters": not args.no_fast_letters,
                "letter_batch_size": args.letter_batch_size,
                "output_dir": str(generate_output_dir(llm.name, config)),
            }
            t1 = time.time()
            try:
                if not hasattr(llm, "model"):
                    raise RuntimeError("the model was freed before this task ran")
                results = run_eval(llm, config)
                # The prefix bits-per-byte side metrics cannot be estimated on a one-letter answer and
                # record a NotEstimable error per sample; any other error means the run is not usable.
                errored = sorted(
                    {
                        f"{r.metric_name}: {r.error.error_class}"
                        for r in results
                        if getattr(r, "error", None) is not None and r.error.error_class != "NotEstimable"
                    }
                )
                if errored:
                    raise RuntimeError(f"metric results carry errors: {', '.join(errored)}")
                record.update(status="ok", seconds=round(time.time() - t1, 1), results=len(results))
                print(f"[{label}] {task}: ok in {record['seconds']}s ({len(results)} metric results)")
            except Exception as exc:  # keep the sweep going; the record says what failed
                failures += 1
                checkpoint_ok = False
                record.update(status="error", seconds=round(time.time() - t1, 1), error=f"{type(exc).__name__}: {exc}")
                print(f"[{label}] {task}: FAILED {record['error']}")
                traceback.print_exc()
            append_record(runs_file, record)
        llm.release()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.keep_weights or not checkpoint_ok:
            print(f"[{label}] weights kept at {weights_root(args.weights_dir)}")
        else:
            remove_weights(checkpoint, args.weights_dir)
            print(f"[{label}] weights removed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
