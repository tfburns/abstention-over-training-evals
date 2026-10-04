# abstention-over-training-evals

Single-turn safety-elicitation evaluations for the [Olmo 3](https://arxiv.org/abs/2512.13961) 32B
checkpoints, built as [`eval-framework`](https://github.com/Aleph-Alpha-Research/eval-framework)
task plugins.

The goal is to find single-turn prompts that elicit unsafe behaviour in the final post-trained
checkpoint (`allenai/Olmo-3-32B-Think`), then to find earlier post-training and pre-training
checkpoints whose behaviour correlates with, and predicts, that final-checkpoint result.

## Layout

```
src/abstention_over_training_evals/
  domain/     # IO- and framework-agnostic: PromptItem, the 100-prompt budget, the SafetyJudge port
  adapters/   # disk, the Olmo checkpoint catalog, and the vLLM-backed OpenAI client
  plugins/    # eval-framework task plugins; shared base in plugins/base.py
    base.py                           # shared task base + metric (imported by package path)
    unsafe_elicitation/               # one task per directory, so --extra-tasks-dir can target one
      unsafe_elicitation.py
    smoke/
      smoke.py
data/local/   # local prompt JSONL, one file per task (capped at 100 prompts each)
tests/        # mirrors src/ by module name
```

Layering is strict: `plugins -> adapters -> domain`. The domain may import only `pydantic` and
`loguru`; an import-linter contract enforces that. See [`AGENTS.md`](AGENTS.md) for the full
standards.

## Install

```bash
uv sync --extra eval
```

The `eval` extra pulls `eval-framework[openai]`. Inference runs out of process, so vLLM and torch
are not dependencies of this package.

## Serve the model

Inference uses the recommended Olmo 3 32B Think settings through a local `vllm serve` endpoint:

```bash
vllm serve allenai/Olmo-3-32B-Think \
  --max-model-len 65536 \
  --reasoning-parser olmo3
```

`--reasoning-parser olmo3` separates the `<think>` trace from the answer, so a judge can read both.
The recommended sampling settings (`temperature=0.6`, `top_p=0.95`, `max_tokens=32768`) are set in
the model adapter and the task. To evaluate an earlier checkpoint, serve that checkpoint; the
catalog is in
[`adapters/checkpoints.py`](src/abstention_over_training_evals/adapters/checkpoints.py).

## Configure a safety judge

A `SafetyJudge` scores responses. With none configured the judge abstains: generation runs, but the
"Unsafe Elicitation Rate" is empty (abstentions are excluded, not counted as safe). To score
responses, point the judge at an OpenAI-compatible endpoint (a second `vllm serve` on another port):

```bash
export AOTE_JUDGE_MODEL=<an independent judge model>   # NOT the checkpoint under evaluation
export AOTE_JUDGE_BASE_URL=http://127.0.0.1:8001/v1    # a second server, distinct from inference
export AOTE_JUDGE_API_KEY=EMPTY                         # default; EMPTY for a local server
```

> **Use an independent judge.** Do not set `AOTE_JUDGE_MODEL` to the checkpoint under evaluation. A
> model grading its own output under-reports its unsafe behaviour and biases both the rate and the
> cross-checkpoint correlation. Use a capable, permissively-licensed judge from another model
> family, and keep it fixed across the checkpoints you compare.

The judge reads the answer and the `<think>` trace and returns a JSON verdict. Each call has a
request timeout and a token cap; a judge that errors or returns malformed output abstains (with the
reason recorded) instead of failing the run. See
[`adapters/llm_judge.py`](src/abstention_over_training_evals/adapters/llm_judge.py).

## Run an evaluation

Point `--extra-tasks-dir` at the single task's directory, so only that task loads:

```bash
uv run eval_framework \
  --extra-tasks-dir src/abstention_over_training_evals/plugins/unsafe_elicitation \
  --llm-name abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B \
  --llm-args base_url=http://127.0.0.1:8000/v1 api_key=EMPTY \
  --task-name UnsafeElicitation \
  --output-dir ./outputs \
  --num-fewshot 0
```

Pointing `--extra-tasks-dir` at one task's directory loads only that task, so a broken sibling task
cannot block it. Point it at the parent
`src/abstention_over_training_evals/plugins` to load every task.

Results stay in `./outputs`. Do not pass `--hf-upload-dir` or `--wandb-project`: this project does
not upload anywhere.

## Add your own task

A task is one small plugin plus its own prompt file. Shared loading, provenance, and judging live in
[`plugins/base.py`](src/abstention_over_training_evals/plugins/base.py); you write the task class.

1. **Create the prompt file** `data/local/<your_task>.jsonl` (see the schema below). Keep it under
   100 prompts; every row needs provenance.
2. **Create the plugin** in its own directory,
   `src/abstention_over_training_evals/plugins/<your_task>/<your_task>.py`:

   ```python
   from eval_framework.tasks.registry import Registry, register_task

   from abstention_over_training_evals.plugins.base import LocalJsonlSafetyTask


   class MyTask(LocalJsonlSafetyTask):
       NAME = "MyTask"  # must be unique; names are compared case-/separator-insensitively
       PROMPTS_FILENAME = "my_task.jsonl"


   def register_tasks(registry: Registry) -> None:
       register_task(MyTask, registry=registry)
   ```

   Add an `__init__.py` in that directory (a one-line docstring is fine).
3. **Add a test** `tests/plugins/test_<your_task>.py` and run `uv run validate && uv run test`.
4. **Run it**, pointing `--extra-tasks-dir` at your task's directory:

   ```bash
   uv run eval_framework \
     --extra-tasks-dir src/abstention_over_training_evals/plugins/<your_task> \
     --task-name MyTask \
     --llm-name abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B \
     --llm-args base_url=http://127.0.0.1:8000/v1 api_key=EMPTY \
     --output-dir ./outputs --num-fewshot 0
   ```

Each task is isolated: its own directory, its own `data/local/` file, and a unique `NAME`
(duplicates fail at load). To read a different prompt file (e.g. in CI), set
`AOTE_PROMPTS_PATH__<NAME>` (e.g. `AOTE_PROMPTS_PATH__MyTask=/path/to/my_task.jsonl`); the override
applies only to that task.

## Evaluation data

Prompts live in [`data/local/unsafe_elicitation.jsonl`](data/local/unsafe_elicitation.jsonl), one
object per line. It starts empty; add your own prompts. To check the serve/eval pipeline first, run
the separate `SafetyPipelineSmokeTest` task, which ships one benign prompt in
[`data/local/smoke.jsonl`](data/local/smoke.jsonl):

```bash
uv run eval_framework \
  --extra-tasks-dir src/abstention_over_training_evals/plugins/smoke \
  --llm-name abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B \
  --llm-args base_url=http://127.0.0.1:8000/v1 api_key=EMPTY \
  --task-name SafetyPipelineSmokeTest --output-dir ./outputs --num-fewshot 0
```

Every prompt must be human-authored or generated by a permissively-licensed LLM, and each row
records its provenance:

```json
{"id": "h1", "prompt": "original single-turn user text", "provenance": {"author_type": "human"}}
{"id": "l1", "prompt": "another single-turn prompt", "provenance": {"author_type": "llm", "generator_model": "OLMo-2-1124-7B-Instruct", "generator_license": "apache-2.0"}}
```

For LLM-generated prompts, `generator_license` must be in the permissive allowlist in
[`domain/provenance.py`](src/abstention_over_training_evals/domain/provenance.py)
(`apache-2.0`, `mit`, `bsd-2-clause`, `bsd-3-clause`, `cc-by-4.0`, `cc0-1.0`); the loader rejects
anything else. Prompts must be original and must not copy benchmark items or other copyrighted
text. Each file is capped at **100 prompts**; the loader raises past that. Beyond the cap, move the
prompts to an Apache-2.0 Hugging Face dataset and point the task at it. This repo runs no git or
Hugging Face actions for you.

## Development

```bash
uv run validate   # ruff check + ruff format --check + mypy + import-linter
uv run test       # pytest
```

CI runs both on every pull request and on pushes to `main`
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Do not merge unless both pass; enable
branch protection on `main` to require the CI check.

## License

Apache-2.0. See [`LICENSE`](LICENSE). Eval datasets must be Apache-2.0 and must not infringe
copyright.
