#!/usr/bin/env bash
# One-time setup of a `vllm/vllm-openai` pod (entrypoint overridden to `sleep infinity`) for the
# reasoned MCQ runs: installs uv, clones the repo and installs the eval extra (no torch needed: the
# client talks to the local vLLM server). Usage: bash pod_setup_vllm.sh [branch]
set -euo pipefail
BRANCH="${1:-feat/nonexistent-entities-mcq-rotations}"
REPO_URL="${REPO_URL:-https://github.com/daria-teodora-harabor/abstention-over-training-evals.git}"
WORK="${WORK:-/workspace}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
mkdir -p "$WORK" "$HF_HOME" "$WORK/outputs"
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq git curl >/dev/null 2>&1 || true
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d "$WORK/abstention-over-training-evals" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$WORK/abstention-over-training-evals"
fi
cd "$WORK/abstention-over-training-evals"
git fetch -q origin "$BRANCH" && git checkout -q "$BRANCH" && git pull -q
uv python install 3.12
uv sync --extra eval
uv run --no-sync python -c "import eval_framework, openai; print('client ok')"
vllm --version
echo "Setup done."
