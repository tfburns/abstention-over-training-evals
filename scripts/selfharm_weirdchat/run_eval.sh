#!/usr/bin/env bash
# Run the SelfHarmWeirdChat task against an Olmo 3 32B Think server and a Gemma 4 31B judge server.
# Usage: run_eval.sh <prompts.jsonl> <output_dir> <repeats>
set -euo pipefail
export AOTE_PROMPTS_PATH__SelfHarmWeirdChat="$1"
export AOTE_JUDGE_MODEL=google/gemma-4-31b-it
export AOTE_JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:18001/v1}"
export AOTE_JUDGE_API_KEY=EMPTY
uv run eval_framework \
  --extra-tasks-dir src/abstention_over_training_evals/plugins/selfharm_weirdchat \
  --task-name SelfHarmWeirdChat \
  --llm-name abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B \
  --llm-args base_url="${OLMO_BASE_URL:-http://127.0.0.1:18000/v1}" api_key=EMPTY model_name=allenai/Olmo-3-32B-Think \
  --output-dir "$2" \
  --num-fewshot 0 \
  --repeats "$3" \
  --batch-size 256
