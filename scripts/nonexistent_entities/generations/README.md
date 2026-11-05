# First-run generations

`olmo3-32b-think_2026-10-26.jsonl`: 250 completions (50 prompts x 5 samples) from
`allenai/Olmo-3-32B-Think` served by `vllm serve ... --max-model-len 65536 --reasoning-parser olmo3`
(vLLM 0.31.0, one H200), temperature 0.6, top_p 0.95, max 32768 tokens. One JSON object per line:
`id`, `sample` (0-4), `prompt`, `answer` (final answer), `reasoning` (the `<think>` trace),
`finish_reason`, `completion_tokens`, `model`, `params`, `timestamp`.

These were produced with a small OpenAI-compatible runner rather than `eval_framework` (the
in-repo run with a judge model follows once the group settles on a judge); the prompts, model and
sampling settings are identical to the task's. The labels in `../labels/` refer to these rows by
`id` and `sample`.
