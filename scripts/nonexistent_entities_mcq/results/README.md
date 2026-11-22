# NonexistentEntitiesMCQ results: Olmo 3 32B across training

Results of the 50-item multiple-choice task (45 fake entities, 5 real-entity controls) on 13 Olmo 3 32B
checkpoints, run 15–16 November 2026. Raw per-sample outputs (about 0.5 GB for the sweep and 1.3 GB of
reasoned traces) are not committed; these files are the summaries computed from them and the figures
drawn from the summaries.

## Setup

- **Checkpoints**, in training order: `allenai/Olmo-3-1125-32B` at `stage1-step136000`, `…270000`,
  `…398000`, `…528000`, `…656000`, `stage2-ingredient1+2-step23842` and `main` (the released base);
  `allenai/Olmo-3-32B-Think-SFT` at `5e-5-step5000` and `main`; `allenai/Olmo-3-32B-Think-DPO`;
  `allenai/Olmo-3-32B-Think` at `step_100`, `step_400` and `main`.
- **Immediate answers** (`summary.json`, `summary.md`): the four log-likelihood tasks in
  `plugins/nonexistent_entities_mcq/` on all 13 checkpoints, zero-shot, plain-text prompt, bfloat16,
  in-process Hugging Face (`adapters/olmo_hf.py`) on A100 80 GB pods. The lettered tasks score every
  option order (24 per item, 120 with the visible "I don't know"). Driver: `run_checkpoints.py`;
  summariser: `summarise_runs.py`.
- **Reasoned answers** (`reasoned_summary.json`, `reasoned_summary.md`): the two tasks in
  `plugins/nonexistent_entities_mcq_reasoned/` on the 6 post-training checkpoints, served with vLLM and
  the `olmo3` reasoning parser on H200 pods, temperature 0.6, top-p 0.95, 16,384-token cap, one answer
  per item and option order (43,200 answers: 1,200 plain and 6,000 with "I don't know" per checkpoint). An answer without a parseable letter counts as wrong.
  Driver: `run_reasoned_pod.sh`; summariser: `summarise_reasoned.py`.
- **Intervals**: 95% bootstrap over items (2,000 resamples of the items with replacement, seed 0),
  each item's value being its mean over option orders. Chance accuracy is 0.25 (0.20 with "I don't know").
- **Framings** come from `scripts/nonexistent_entities/prompt_metadata.jsonl` (same item ids).

## Figures (`figures/`)

Regenerate from this folder with `uv run --with matplotlib python scripts/nonexistent_entities_mcq/<script> …`;
the commands are in each script's docstring.

| File | Script | What it shows |
|---|---|---|
| `fig1_accuracy_over_training.png` | `make_figures.py` | Accuracy per task over the 13 checkpoints, fakes and controls; cloze series length-corrected (Bayesian) |
| `fig2_abstention_over_training.png` | `make_figures.py` | "I don't know" rate over training for the two abstention tasks |
| `fig3_consistency_over_training.png` | `make_figures.py` | Share of items whose chosen option is the same under every option order |
| `r1_reasoned_accuracy_abstention.png` | `make_figures_reasoned.py` | Reasoned accuracy, abstention (with the immediate abstention overlaid) and unparseable answers per post-training checkpoint |
| `r2_reasoning_length.png` | `make_figures_reasoned.py` | Median trace length by outcome (correct, wrong); traces cut at the cap excluded |
| `r3_think_by_category.png` | `make_figures_reasoned.py` | Final Think model's reasoned accuracy per domain |
| `c1_immediate_vs_reasoned.png` | `make_figures_reasoned.py` | Immediate vs reasoned accuracy on the same checkpoints, with the paired per-item difference |
| `c2_paired_difference.png` | `make_figures_reasoned.py` | Per-item immediate vs reasoned accuracy for the final Think model |
| `formulations_accuracy_abstention.png` | `make_figure_formulations.py` | The six question formulations (lettered, cloze, reasoned; each with and without "I don't know") on accuracy and abstention over training |
| `prompt_framings_accuracy_abstention.png` | `make_figure_prompt_framings.py` | Accuracy and abstention per prompt framing over training, fake items only |
| `prompt_framings_bars_last3.png` | `make_figure_framing_bars.py` | Framings compared at RL 100, RL 400 and Think, immediate vs reasoned |

All error bars are the 95% bootstrap intervals above. The per-framing and per-domain intervals are wide:
10 items per framing (5 half-remembered) and 9 fake items per domain.
