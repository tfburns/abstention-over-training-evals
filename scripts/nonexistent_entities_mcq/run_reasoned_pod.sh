#!/usr/bin/env bash
# Run a reasoned MCQ task on a list of checkpoints, one vLLM server at a time, on a pod prepared by
# pod_setup_vllm.sh. Usage:
#   bash run_reasoned_pod.sh NonexistentEntitiesMCQReasoned sft@5e-5-step5000 sft
# Each checkpoint: `vllm serve` (with the Olmo reasoning parser) -> wait for /health -> eval_framework
# through the OpenAI-compatible adapter -> stop the server. Records one line per run in
# /workspace/outputs/runs.log (`<label> <task> ok|error <seconds>`); already-ok pairs are skipped, so
# rerunning resumes. Downloaded weights stay in $HF_HOME so a second task on the same checkpoints
# does not download again.
set -uo pipefail
TASK="$1"; shift
export PATH="$HOME/.local/bin:$PATH" HF_HOME="${HF_HOME:-/workspace/hf}"
REPO=/workspace/abstention-over-training-evals
OUT="${OUT:-/workspace/outputs}"
BATCH="${BATCH:-400}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
# Requests in flight (client) and sequences the server runs at once; 256 thrashes the KV cache on
# the long-tracing checkpoints, 160 keeps it below capacity on an H200 at this prompt length.
MAX_WORKERS="${MAX_WORKERS:-160}"
MAX_SEQS="${MAX_SEQS:-160}"
mkdir -p "$OUT"
cd "$REPO"
for spec in "$@"; do
  read -r MODEL REV LABEL <<<"$(uv run --no-sync python -c "
import sys; sys.path.insert(0, 'src')
from abstention_over_training_evals.adapters.olmo_weights import parse_checkpoint, checkpoint_label
c = parse_checkpoint(sys.argv[1]); print(c.model_id, c.revision, checkpoint_label(c))" "$spec")"
  if [ -z "${LABEL:-}" ]; then echo "bad checkpoint spec: $spec" | tee -a "$OUT/runs.log"; continue; fi
  if grep -q "^$LABEL $TASK ok" "$OUT/runs.log" 2>/dev/null; then echo "[$(date -u +%H:%M:%S)] skip $LABEL $TASK (done)"; continue; fi
  echo "[$(date -u +%H:%M:%S)] $LABEL: starting vllm serve $MODEL --revision $REV"
  vllm serve "$MODEL" --revision "$REV" --reasoning-parser olmo3 --max-model-len 20480 --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization 0.92 --port 8000 --served-model-name "$MODEL" > "$OUT/vllm_${LABEL}.log" 2>&1 &
  VPID=$!
  up=0
  for _ in $(seq 1 600); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then up=1; break; fi
    if ! kill -0 "$VPID" 2>/dev/null; then break; fi
    sleep 10
  done
  if [ "$up" -ne 1 ]; then
    echo "$LABEL $TASK error server-did-not-start" >> "$OUT/runs.log"
    echo "[$(date -u +%H:%M:%S)] $LABEL: vllm did not come up (see vllm_${LABEL}.log)"
    kill "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null
    continue
  fi
  echo "[$(date -u +%H:%M:%S)] $LABEL: server up, running $TASK"
  t0=$(date +%s)
  uv run --no-sync eval_framework \
    --extra-tasks-dir src/abstention_over_training_evals/plugins/nonexistent_entities_mcq_reasoned \
    --task-name "$TASK" \
    --llm-name abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B \
    --llm-args base_url=http://127.0.0.1:8000/v1 api_key=EMPTY "model_name=$MODEL" timeout=3600 "max_workers=$MAX_WORKERS" \
    --output-dir "$OUT/$LABEL" --num-fewshot 0 --batch-size "$BATCH" --max-tokens "$MAX_TOKENS" \
    > "$OUT/eval_${LABEL}_${TASK}.log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then echo "$LABEL $TASK ok $secs" >> "$OUT/runs.log"; else echo "$LABEL $TASK error rc=$rc $secs" >> "$OUT/runs.log"; fi
  echo "[$(date -u +%H:%M:%S)] $LABEL $TASK: rc=$rc in ${secs}s"
  kill "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null
  pkill -f "vllm serve" 2>/dev/null
  for _ in $(seq 1 24); do  # wait for the engine to release the GPU before the next server starts
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 5
  done
done
echo "[$(date -u +%H:%M:%S)] ALL DONE for $TASK: $*"
echo "ALL DONE $TASK" >> "$OUT/runs.log"
