"""Where checkpoint weights live on the evaluation machine, and how they get there.

The log-likelihood tasks need the model in-process, so unlike the vLLM path the runner fetches weights
itself: one Hub revision at a time into ``AOTE_WEIGHTS_DIR`` (default ``~/.cache/aote-weights``),
scored, then deleted, because a 32B checkpoint is about 65 GB and a sweep visits many. Nothing here
imports torch, so the runner's ``--dry-run`` and the tests work on a machine without a GPU stack.

A checkpoint is named by a short spec: a stage alias (``base``, ``sft``, ``dpo``, ``think``) or a
catalogued Hub id, optionally followed by ``@<revision>`` for an intermediate checkpoint, e.g.
``base@stage1-step72000`` or ``think@step_100``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from abstention_over_training_evals.adapters.checkpoints import OLMO3_32B_CHECKPOINTS, OlmoCheckpoint

WEIGHTS_DIR_ENV = "AOTE_WEIGHTS_DIR"
_DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "aote-weights"

# Only what loading needs: the safetensors shards and index, the configs and the tokenizer files.
WEIGHT_FILE_PATTERNS: tuple[str, ...] = ("*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja")

STAGE_ALIASES: dict[str, OlmoCheckpoint] = {c.stage.value: c for c in OLMO3_32B_CHECKPOINTS}
STAGE_ALIASES["think"] = STAGE_ALIASES["rlvr"]
_BY_MODEL_ID: dict[str, OlmoCheckpoint] = {c.model_id.lower(): c for c in OLMO3_32B_CHECKPOINTS}


def parse_checkpoint(spec: str) -> OlmoCheckpoint:
    """Resolve ``<alias or Hub id>[@<revision>]`` against the catalog.

    Raises:
        ValueError: when the name is neither a stage alias nor a catalogued Hub id.
    """
    name, _, revision = spec.strip().partition("@")
    key = name.strip().lower()
    checkpoint = STAGE_ALIASES.get(key) or _BY_MODEL_ID.get(key)
    if checkpoint is None:
        known = ", ".join(sorted(STAGE_ALIASES)) + ", " + ", ".join(c.model_id for c in OLMO3_32B_CHECKPOINTS)
        raise ValueError(f"unknown checkpoint {name!r}; use one of: {known} (optionally followed by @<revision>)")
    return checkpoint.at_revision(revision.strip()) if revision.strip() else checkpoint


def checkpoint_label(checkpoint: OlmoCheckpoint) -> str:
    """A file-system-safe name for a checkpoint: ``<org>_<repo>@<revision>``."""
    return f"{checkpoint.model_id.replace('/', '_')}@{checkpoint.revision}"


def weights_root(root: str | os.PathLike[str] | None = None) -> Path:
    """The directory checkpoints are downloaded into: the argument, else ``$AOTE_WEIGHTS_DIR``, else the default."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get(WEIGHTS_DIR_ENV) or _DEFAULT_WEIGHTS_DIR)


def weights_dir(checkpoint: OlmoCheckpoint, root: str | os.PathLike[str] | None = None) -> Path:
    """Where this checkpoint's weights are (or will be) on disk."""
    return weights_root(root) / checkpoint_label(checkpoint)


def download_weights(checkpoint: OlmoCheckpoint, root: str | os.PathLike[str] | None = None) -> Path:
    """Fetch the checkpoint's weights from the Hub into :func:`weights_dir` (resumes a partial download)."""
    from huggingface_hub import snapshot_download  # heavy import, only when downloading

    target = weights_dir(checkpoint, root)
    snapshot_download(
        repo_id=checkpoint.model_id,
        revision=checkpoint.revision,
        local_dir=str(target),
        allow_patterns=list(WEIGHT_FILE_PATTERNS),
    )
    return target


def remove_weights(checkpoint: OlmoCheckpoint, root: str | os.PathLike[str] | None = None) -> None:
    """Delete the checkpoint's local weights, if present."""
    shutil.rmtree(weights_dir(checkpoint, root), ignore_errors=True)
