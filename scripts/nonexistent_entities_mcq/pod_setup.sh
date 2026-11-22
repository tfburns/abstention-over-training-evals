#!/usr/bin/env bash
# One-time setup of a fresh GPU pod (RunPod PyTorch template or any CUDA image with Python) for the
# NonexistentEntitiesMCQ checkpoint runs. Clones the repo, installs the eval extra plus torch, and prints
# the run plan. Usage, from the pod's shell:
#   bash pod_setup.sh [branch]            (defaults to the branch this script was written on)
# Then:
#   cd /workspace/abstention-over-training-evals
#   uv run python scripts/nonexistent_entities_mcq/run_checkpoints.py --checkpoints base sft dpo think
set -euo pipefail

BRANCH="${1:-feat/nonexistent-entities-mcq-rotations}"
REPO_URL="${REPO_URL:-https://github.com/daria-teodora-harabor/abstention-over-training-evals.git}"
WORK="${WORK:-/workspace}"
export AOTE_WEIGHTS_DIR="${AOTE_WEIGHTS_DIR:-$WORK/weights}"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$WORK" "$AOTE_WEIGHTS_DIR"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d "$WORK/abstention-over-training-evals" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$WORK/abstention-over-training-evals"
fi
cd "$WORK/abstention-over-training-evals"
uv python install 3.12
uv sync --extra eval
# torch is deliberately not a dependency of the package; add it (and accelerate for device_map="auto",
# hf_transfer for fast downloads) to the project venv.
uv pip install torch accelerate hf_transfer
uv run python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), torch.cuda.mem_get_info()[1] // 2**30, 'GiB')"
uv run python scripts/nonexistent_entities_mcq/run_checkpoints.py --checkpoints base sft dpo think --dry-run
echo "Setup done. Keep AOTE_WEIGHTS_DIR=$AOTE_WEIGHTS_DIR and HF_HUB_ENABLE_HF_TRANSFER=1 exported in the shell that runs the sweep."
