# PurposeFraming on base checkpoints

Can a checkpoint from before post-training tell us which harmful requests the final
`allenai/Olmo-3-32B-Think` will answer? This plugin tests that on the 29 [PurposeFraming](../purpose_framing/README.md)
cores (fraud and cyber). It uses earlier checkpoints of `allenai/Olmo-3-1125-32B`: pretraining (stage 1), mid-training (stage 2),
long-context training (stage 3) and `main`.

## Targets

The final Think model's per-core results on the plain condition, from
[`purpose_framing/results/olmo3-32b-think/per_core.csv`](../purpose_framing/results/olmo3-32b-think/per_core.csv)
(5 samples per prompt):

- **delivers**: share of samples that handed over the requested content;
- **StrongREJECT**: the graded StrongREJECT score (Souly et al., 2024).

Each checkpoint score is compared with these by Spearman rho over the 29 cores. Negative is the expected direction: more refusal early, less delivery at the end. CIs are bootstrap over cores, and p values are permutation.

## Measures

Base checkpoints do not chat, so we score likelihoods instead of judging generations. The domain stays the same (the same requests) and the task changes (Pan and Yang's domain/task split).

1. **Cloze.** After `User: {request}\nAssistant:`, compare the log-likelihood of refusal openers with
   compliance openers. The score is the PMI margin, which removes how likely a generic refusal is
   regardless of the request. The result of record uses openers
   ([`human_openers.json`](human_openers.json)), earlier runs used refusals taken from the Think model.
2. **Multiple choice (MCQ).** The request is shown with three options ([`mcq_options.json`](mcq_options.json),
   [`MCQ_ITEMS.md`](MCQ_ITEMS.md)): decline, fully answer, or comply with a warning. The letters are scored
   in eval-framework's MCStyle layout, as in the NonexistentEntities MCQ (#6). Each item is scored in all 6 option
   orders and averaged, so letter preference cannot reach the per-core numbers.

We picked the measures and checked their noise using FineTasks (Kydlíček et al., 2024), the Hugging Face evaluation guidebook, and Heineman et al. (2025) on signal-to-noise across checkpoints.

## Main results

| Checkpoint | Cloze rho vs delivers / StrongREJECT | MCQ accuracy | MCQ rho vs StrongREJECT | Cloze + MCQ vs StrongREJECT |
|---|---|---|---|---|
| `stage1-step66000` (early pretraining) | −0.12 / −0.18 | 0.03 | −0.33 | — |
| `stage1-step656000` (end of pretraining) | −0.14 / −0.36 | 0.88 | −0.39 | −0.48 |
| ingredient 1, step 2000 | −0.42 / −0.53 | 1.00 | −0.51 | −0.59 |
| merge `stage2-ingredient1+2-step23842` | **−0.54 / −0.63** | 0.98 | **−0.65** | **−0.71** [−0.86, −0.48] |
| `main` | −0.53 / −0.46 | 0.78 | −0.60 | −0.60 |
| Think (MCQ only, no chat template) | — | 0.62 | −0.64 | — |

Cloze numbers use the Think-derived openers; with human openers the merge gives −0.56 / −0.56 and `main`
−0.59 / −0.47.

1. **The per-core predictor forms in mid-training, not pretraining.** Pretraining checkpoints do not reliably
   predict the final model (3 of 26 stage-1 tests reach p < 0.05). Within 1,000–2,000 mid-training steps,
   refusal mass appears and the ordering moves most of the way to `main`'s. The merge is the strongest point.
   Long-context training keeps the ordering and halves the refusal mass.
2. **Opener choice and prompt length don't explain it.** It holds with human-written openers, with each
   opener on its own, and after controlling for prompt length.
3. **Knowing to decline also comes early.** MCQ accuracy goes from below chance (0.03) to 0.88 at the end of
   pretraining, 1.00 at step 2000, then drops to 0.78 at `main`, as the cloze refusal mass does.
4. **The two measures agree and combine.** They rank cores alike (0.62–0.64 from step 2000 on). The
   pre-registered combined score is the strongest predictor.
5. **Framing.** From 656000 on, p(decline) is lower on framed items (Wilcoxon p ≤ 0.002 except step 1000), as the
   final model delivers more when framed. Framing *lift* is not a usable target: the final model's framed scores
   sit at the ceiling (withdrawn, trajectory findings).

## Caveats

- **29 cores.** CIs are about ±0.3.
- **MCQ no-skill baseline.** 66000 cannot do the MCQ but still ranks cores somewhat like the final model. No
  checkpoint beats it with a CI excluding zero (merge −0.30 [−0.76, +0.10]), so the MCQ ranking is not yet shown to
  add much beyond the request wording. The cloze baseline is near zero, so cloze is the cleaner per-core
  predictor.
- **Think in the MCQ** was run without its chat template or thinking step. It is a format probe.
- **Not tested.** Harmful vs harmless contrast; which mid-training data creates the signal.

## Runs

All runs were done in Colab with Hugging Face `transformers`; the plugin registers no eval-framework task. Each
results folder has a `FINDINGS.md`.

| Results | What it does |
|---|---|
| [`results/p0b/`](results/p0b/FINDINGS.md) | Cloze on `main`, 656000, 66000 with Think-derived openers; framing controls |
| [`results/trajectory/`](results/trajectory/FINDINGS.md) | Cloze across 22 revisions; noise and SNR |
| [`results/stage2_openers/`](results/stage2_openers/FINDINGS.md) | Early stage-2 bracket; human vs Think openers |
| [`results/purpose_framing_mcq/`](results/purpose_framing_mcq/FINDINGS.md) | First MCQ run, one option order (per-core results superseded) |
| [`results/purpose_framing_mcq_perm/`](results/purpose_framing_mcq_perm/FINDINGS.md) | MCQ in all 6 orders; combined test; MCQ result of record |
