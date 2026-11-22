# NonexistentEntitiesMCQ: the base-model variant of NonexistentEntities

The generative `NonexistentEntities` task (its own plugin, under review in
[tfburns/abstention-over-training-evals#5](https://github.com/tfburns/abstention-over-training-evals/pull/5))
asks about entities that sound real but are not (made-up medicines, non-existent code articles and
regulation numbers, invented procedures and clinical scores) and judges whether the final answer
presents specifics about them as real. Pre-trained (base) checkpoints are not instruction-tuned and
do not answer such a question, so this plugin asks the same questions as **items scored by
log-likelihood**, following Pan and Yang's framing of a task as a label space with a conditional
distribution over it, with the label space chosen to be multiple-choice answer sets with gold answers.
The items are offered in the two formulations that OLMES ([Gu et al., 2024](https://arxiv.org/abs/2406.08446),
section 2.1) evaluates side by side: the **multiple-choice formulation**, where the options are shown
as lettered lines and the letters are scored, and the **cloze formulation**, where only the question is
shown and each option's text is scored. OLMES reports that base models learn the lettered format only
part-way through pre-training and that the cloze formulation gives a signal before that, so early
checkpoints are expected to read mainly from the cloze tasks.

## Tasks

All four read `data/local/nonexistent_entities_mcq.jsonl` and score the same per-item option order.

| task | formulation | scored per item | metrics |
|---|---|---|---|
| `NonexistentEntitiesMCQ` | lettered options shown | the 4 letters | accuracy (raw, length-normalised, Bayesian), bits per byte |
| `NonexistentEntitiesMCQIdk` | lettered, plus a visible `E. I don't know` | the 5 letters | the same, plus Confidence-weighted Accuracy, Distributional Correctness Score and Ternary Score |
| `NonexistentEntitiesMCQCloze` | question only | the 4 option texts | as for `NonexistentEntitiesMCQ` |
| `NonexistentEntitiesMCQClozeIdk` | question only, hidden abstention | the 4 option texts plus `I don't know` | as for `NonexistentEntitiesMCQIdk` |

For a fake entity the gold option says it does not exist and the three wrong options treat it as real
(a fabricated specific, a second fabricated specific, a silent substitution by a real neighbour). For
the five real-entity **controls** the gold option is the true specific and one wrong option denies the
entity, so a model cannot score well by always choosing "does not exist". Items are reported under two
subjects, `main` (fake entities) and `control`, as well as overall.

The abstention variants exist because the two scorings answer different questions: the plain tasks ask
whether the model knows the entity is fake; the abstention tasks ask whether a model that does not
know says so rather than guessing. The `I don't know` option is always the last scored completion
because the confidence-aware metrics treat the last one as the abstention. In the lettered task it is
a visible fifth option; in the cloze task it is scored like any other answer text but never shown.
The shuffle of the four real options is seeded by the item id and question, so all four tasks score
them in the same order.

**Which accuracy to read.** In the lettered tasks every scored completion is a single letter, so option
length never enters the score and the three accuracies agree. In the cloze tasks the option texts
themselves are scored, and a longer text collects more negative log-probability regardless of
content; the gold options here are often two sentences and the wrong options one line, so read the
**Bayesian accuracy**, which fits one per-byte length penalty across the run and removes it before
picking the best option, and treat the raw accuracy as a lower bound.

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

1. The author writes the items in the curation sheet (one row per prompt, with every labelled
   Olmo-3-32B-Think answer on a second tab as raw material for the wrong options; the wrong options
   are written by hand, not copied). Every fake item's gold option says the entity does not exist,
   and names the real counterpart where there is one; the wrong options attribute a rule or a specific
   to the fake entity, one of them usually the true rule under the wrong name. Options carry no
   terminal full stop; generic drug names are lower case except at the start of an option.
2. Export the sheet's `mcq` tab as CSV and convert it:

   ```bash
   uv run python scripts/nonexistent_entities_mcq/sheet_to_jsonl.py path/to/mcq.csv
   ```

   The converter matches columns by the start of their header (`Edited questions`, `Gold option`,
   `Wrong option 1` and up, optional `IDK option`), expands the short category slugs, validates every
   row with `McqItem`, and writes nothing if any row has a problem.
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

Run the other three tasks the same way, and the post-training stages by swapping `model_name`
(`allenai/Olmo-3-32B-Think-SFT`, `allenai/Olmo-3-32B-Think-DPO`, `allenai/Olmo-3-32B-Think`; see
`adapters/checkpoints.py`). `--num-fewshot k` draws k other items from the same file as examples
(the current item is excluded); because 45 of the 50 items are fake, examples mostly show the "does
not exist" answer, so zero-shot is the primary setting.

## Status

The 50 items (45 fake, 5 controls, 10 per category) are in `data/local/nonexistent_entities_mcq.jsonl`,
converted from the curation sheet on 15 November 2026. Runs on the checkpoints are pending.
