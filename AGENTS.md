# abstention-over-training-evals: engineering standards

This repo runs single-turn evaluations on Olmo 3 32B checkpoints, built as
[`eval-framework`](https://github.com/Aleph-Alpha-Research/eval-framework) task plugins. The
immediate goal is to find single-turn prompts that elicit unsafe behaviour in the final
post-trained checkpoint (`allenai/Olmo-3-32B-Think`), then to find earlier post-training and
pre-training checkpoints whose behaviour predicts that final-checkpoint result.

## Rules

- **No git, `gh`, or Hugging Face actions from this repo.** This package does not run git or
  `gh`, and it does not upload, create, or download Hub repositories. Eval results and prompts
  stay on disk. A human runs any git step and creates any Hugging Face dataset.
- **Local eval data is capped at 100 prompts.** Prompts live in `data/local/*.jsonl` as
  `{"id": ..., "prompt": ..., "provenance": {...}}` lines. Past 100 prompts, a human creates an
  Apache-2.0 Hugging Face dataset and the task is pointed at it. The loader enforces the cap.
- **Every prompt is human-authored or from a permissively-licensed LLM, with provenance.** Each
  row carries a `provenance` object. Human prompts: `{"author_type": "human"}`. LLM prompts:
  `{"author_type": "llm", "generator_model": ..., "generator_license": ...}`, where the license
  is in the permissive allowlist (`domain/provenance.py::PERMISSIVE_LICENSES`). The loader
  rejects rows that violate this. Extend the allowlist deliberately and only with genuinely
  permissive licenses.
- **Everything is Apache-2.0 and must not infringe copyright.** Eval prompts must be original.
  They must not copy benchmark items or other copyrighted text.

## Architecture: hexagonal-lite (domain vs adapters vs plugins)

- `domain/` is orchestration- and IO-agnostic. It may import `pydantic` and `loguru` and no
  other third-party libraries. No `eval_framework`, `openai`, `vllm`, `transformers`,
  `datasets`, `huggingface_hub`, or `httpx`. It is unit-testable with no GPU and no network.
  The import-linter contract enforces this; do not weaken it.
- `adapters/` is the only place disk layout, the checkpoint catalog, and the vLLM-backed
  OpenAI-compatible client live.
- `plugins/` holds the `eval-framework` task plugins, **one task per subdirectory** (e.g.
  `plugins/unsafe_elicitation/`, `plugins/smoke/`) so `--extra-tasks-dir` can target a single task
  in isolation. `eval-framework` loads every `.py` under the directory it is given and calls each
  module's `register_tasks(registry)` entrypoint (modules without one, like the shared
  `plugins/base.py`, are imported but register nothing). New tasks subclass `LocalJsonlSafetyTask`,
  set a unique `NAME` + `PROMPTS_FILENAME`, and expose `register_tasks`. The shared base classes in
  `plugins/base.py` are imported by package path, never via `--extra-tasks-dir`.
- Layering: `plugins -> adapters -> domain`. Keep modules deep: push complexity into
  lower-level functions. Layers use different vocabulary (`PromptItem` in the domain;
  `BaseTask`/`vLLM`/OpenAI client in adapters and plugins).

## Judging

- Unsafe behaviour is scored through the domain `SafetyJudge` port. The default abstains;
  `LlmSafetyJudge` is configured from the environment (`AOTE_JUDGE_MODEL`, `AOTE_JUDGE_BASE_URL`,
  `AOTE_JUDGE_API_KEY`), so the CLI needs no code change. A judge that errors or returns malformed
  output must abstain (excluded from the mean), never score as safe.
- Each task reads its own prompt file `data/local/<task>.jsonl`. Use `AOTE_PROMPTS_PATH__<NAME>`
  for a per-task override (tests, CI); overrides are per-task only, so none can redirect every task
  at once.

## Language and tooling

- Python `>=3.12,<3.14`. Ruff line length 120, double quotes, Google docstrings
  (`D100`-`D107` off). McCabe <= 10. **Strict mypy.** Pydantic models for payloads. Loguru for
  logs. Do not `print` in the domain.
- `uv run validate` = ruff check + ruff format --check + mypy + import-linter.
  `uv run test` = pytest. Both must pass.
- Tests mirror `src/` one-to-one under `tests/`.

## Merging

- Do not merge a pull request unless `uv run validate` and `uv run test` pass.
- CI runs both on every pull request and on pushes to `main` (`.github/workflows/ci.yml`).
- Enable branch protection on `main` to require the CI check before merging (a one-time repo setting
  in the host UI; this package runs no git or host-admin actions itself).

## Comments and style

- Comments explain why. Do not narrate what the code does, anthropomorphise, or record
  "changed from X to Y". The code is the current state.

## Inference

All inference uses the recommended Olmo 3 settings through a local `vllm serve` endpoint, with
`eval-framework`'s OpenAI-compatible client talking to it. See [`README.md`](README.md) for the
serve and eval commands. Recommended sampling for the Think checkpoints: `temperature=0.6`,
`top_p=0.95`, `max_tokens=32768`, served with `--reasoning-parser olmo3` so the `<think>` trace
is captured separately from the answer.
