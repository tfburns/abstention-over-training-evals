# NonexistentEntitiesMCQReasoned: the same multiple choice, answered with reasoning

The log-likelihood tasks in `../nonexistent_entities_mcq/` read a checkpoint's immediate preference
over the option letters. These two tasks ask the post-trained (Think) checkpoints the same questions,
with the same option text and the same option orders, but let the model reason first through its chat
template and parse the letter it settles on. The comparison between the two is paired: every reasoned
answer has an immediate-answer twin for the same item, order and checkpoint.

| Task | Options | Orders per item | Samples |
|---|---|---:|---:|
| `NonexistentEntitiesMCQReasoned` | 4 | 24 | 1,200 |
| `NonexistentEntitiesMCQReasonedIdk` | 5, with a visible `I don't know` | 120 | 6,000 |

What differs from the log-likelihood prompt, both unavoidable for a reasoning model: the chat template
(the server applies it; the model's own system line and thinking block), and one instruction line,
`Answer with the letter of the best option.`, so the answer is a parseable letter. Everything else is
the same text. Answers are sampled at the model card's settings (temperature 0.6, top-p 0.95) with a
16k-token cap, so each answer is one draw; the pilot on five items gave traces of 600 to 5,000
tokens and a bare letter after the thinking block every time.

Metrics per answer: `Letter Accuracy` (unparseable counts as wrong), `Abstention`, `Unparseable`,
`Reasoning Tokens`, `Reasoning Chars`. The sample context (`item_id`, `order`, `gold_letter`,
`abstention_letter`) is copied into `output.jsonl`, so results map back to items without the manifest.

## Run

Serve the checkpoint with vLLM and the Olmo reasoning parser, then run the task through the
OpenAI-compatible adapter; `scripts/nonexistent_entities_mcq/run_reasoned_pod.sh` does this for a
list of checkpoints on one pod (`pod_setup_vllm.sh` prepares a `vllm/vllm-openai` pod):

```bash
vllm serve allenai/Olmo-3-32B-Think --revision step_400 --reasoning-parser olmo3 \
  --max-model-len 20480 --max-num-seqs 256 --port 8000
uv run eval_framework \
  --extra-tasks-dir src/abstention_over_training_evals/plugins/nonexistent_entities_mcq_reasoned \
  --task-name NonexistentEntitiesMCQReasoned \
  --llm-name abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B \
  --llm-args base_url=http://127.0.0.1:8000/v1 api_key=EMPTY model_name=allenai/Olmo-3-32B-Think timeout=3600 \
  --output-dir ./outputs/reasoned --num-fewshot 0 --batch-size 400 --max-tokens 16384
```

`--batch-size` is the number of answers in flight at once (the client runs them concurrently; vLLM
queues what it cannot run); `timeout` on the adapter is the per-request limit in seconds, which a
16k-token trace needs. Base and pre-training checkpoints have no chat template and cannot run these.
