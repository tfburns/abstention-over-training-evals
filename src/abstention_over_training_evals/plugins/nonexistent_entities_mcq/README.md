# NonexistentEntitiesMCQ: the base-model variant of NonexistentEntities

The generative `NonexistentEntities` task (its own plugin, under review in
[tfburns/abstention-over-training-evals#5](https://github.com/tfburns/abstention-over-training-evals/pull/5))
asks about entities that sound real but are not (made-up medicines, non-existent code articles and
regulation numbers, invented procedures and clinical scores) and judges whether the final answer
presents specifics about them as real. Pre-trained (base) checkpoints are not instruction-tuned and
do not answer such a question, so this plugin asks the same questions as **multiple-choice items
scored by log-likelihood**: the model is shown lettered options and scored on the likelihood of each
letter, following Pan and Yang's framing of a task as a label space with a conditional distribution
over it, with the label space chosen to be multiple-choice answer sets with gold answers.

## Tasks

Both read `data/local/nonexistent_entities_mcq.jsonl`.

| task | options per item | metrics |
|---|---|---|
| `NonexistentEntitiesMCQ` | 4: the gold answer and three wrong options, in a fixed per-item shuffle | accuracy (raw, length-normalised, Bayesian), bits per byte |
| `NonexistentEntitiesMCQIdk` | 5: the same four plus a visible `E. I don't know` | the same, plus Confidence-weighted Accuracy, Distributional Correctness Score and Ternary Score |

For a fake entity the gold option says it does not exist and the three wrong options treat it as real
(a fabricated specific, a second fabricated specific, a silent substitution by a real neighbour). For
the five real-entity **controls** the gold option is the true specific and one wrong option denies the
entity, so a model cannot score well by always choosing "does not exist". Items are reported under two
subjects, `main` (fake entities) and `control`, as well as overall.

The abstention variant exists because the two scorings answer different questions: the plain task asks
whether the model knows the entity is fake; the abstention task asks whether a model that does not
know says so rather than guessing. The `I don't know` option is always the last letter because the
confidence-aware metrics treat the last scored completion as the abstention. The shuffle of the four
real options is seeded by the item id and question, so both tasks show them in the same order.

## Item file

One JSON object per line, validated by `McqItem` (`domain/models.py`), at most 100 items:

```json
{"id": "pharma-01", "category": "prescription_drug", "control": false,
 "question": "...the question, close to the generative prompt...",
 "gold": "...the option that says the entity does not exist...",
 "distractors": ["...fabricated specific...", "...second fabricated specific...", "...silent substitution..."],
 "provenance": {"author_type": "human"}}
```

`category` follows the generative task (`prescription_drug`, `paediatric_dosing`, `electrical_code`,
`workplace_safety`, `medical_procedure_or_score`); `control` is `true` for the real-entity items.
Keep the options matched in length and style: a base model scored by log-likelihood will otherwise
pick by surface form, and the length-normalised accuracy only partly corrects for that.

## Workflow

1. The author writes the items in the curation workbook (one row per prompt, with every labelled
   Olmo-3-32B-Think answer on a second sheet as raw material for the wrong options; the wrong options
   are written by hand, not copied).
2. Export the workbook's `mcq` tab as CSV and convert it:

   ```bash
   uv run python scripts/nonexistent_entities_mcq/sheet_to_jsonl.py path/to/mcq.csv
   ```

   The converter matches columns by the start of their header, expands the short category slugs,
   validates every row with `McqItem`, and writes nothing if any row has a problem.
3. Run the tasks (below) and commit the item file with the results.

## Run

Log-likelihoods need the model in-process (eval-framework's OpenAI-compatible client cannot score
prompt tokens through vLLM), so run on a GPU machine with the Hugging Face extra of eval-framework
installed. The same plain prompt format is used for every checkpoint, so the stages are comparable:

```bash
uv run eval_framework \
  --extra-tasks-dir src/abstention_over_training_evals/plugins/nonexistent_entities_mcq \
  --task-name NonexistentEntitiesMCQ \
  --llm-name eval_framework.llm.huggingface.HFLLM \
  --llm-args model_name=allenai/Olmo-3-1125-32B formatter_name=ConcatFormatter \
  --output-dir ./outputs/mcq --num-fewshot 0
```

Run `NonexistentEntitiesMCQIdk` the same way, and the post-training stages by swapping `model_name`
(`allenai/Olmo-3-32B-Think-SFT`, `allenai/Olmo-3-32B-Think-DPO`, `allenai/Olmo-3-32B-Think`; see
`adapters/checkpoints.py`). `--num-fewshot k` draws k other items from the same file as examples
(the current item is excluded); because 45 of the 50 items are fake, examples mostly show the "does
not exist" answer, so zero-shot is the primary setting.

## Status

Plugin, converter and tests are in place; the item file is being written.
