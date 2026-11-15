# Trajectory findings: where the per-core signal appears

Run in Colab on 22 revisions of
`allenai/Olmo-3-1125-32B`. The design is the same as P0b (29 cores × 6 conditions × 6 openers; see
[`../p0b/FINDINGS.md`](../p0b/FINDINGS.md)). `main`, `stage1-step656000` and `stage1-step66000` were reused from
P0b after an assertion that the design matched. The plumbing check passed on every revision (max diff 0.12,
tolerance 0.25).

"Refusal score" is the plain-condition PMI refusal margin. rho is Spearman over the 29 cores against the final
Think model; negative is the expected direction.

## Primary result: the signal appears in mid-training

| Stage | Revision | vs delivers | vs StrongREJECT | mean `p_refuse` | rank agreement with `main` |
|---|---|---|---|---|---|
| pretraining | 10% grid, 66k–590k (9 points) | −0.04 to −0.31, except 328k | −0.16 to −0.26, except 328k | ≤ 0.004 | −0.29 to +0.49 |
| pretraining | `stage1-step328000` | −0.48 (p = 0.002) | −0.63 (p < 0.001) | 0.004 | +0.49 |
| pretraining | 653k–656k (end of stage 1) | −0.03 to −0.18 | −0.25 to −0.46 | 0.003–0.004 | +0.01 to +0.21 |
| mid-training | `stage2-ingredient1-step12000` | −0.27 (p = 0.17) | −0.38 (p = 0.04) | 0.55 | +0.70 |
| mid-training | ingredient 1 / 2 at step 23842 | −0.55 / −0.39 | −0.64 / −0.60 | 0.56 / 0.51 | +0.86 / +0.59 |
| mid-training | merge `stage2-ingredient1+2-step23842` | **−0.54** (p < 0.001) | **−0.63** (p < 0.001) | 0.66 | +0.77 |
| long context | stage 3, steps 9000–11921 | −0.43 to −0.54 | −0.39 to −0.47 | 0.31–0.35 | +0.97 to +0.99 |
| final base | `main` | −0.53 (p = 0.003) | −0.46 (p = 0.012) | 0.34 | 1 |

- **The per-core predictor of the final model forms during stage 2 (mid-training).**
  - Refusal mass appears in the first 12,000 steps of ingredient 1: `p_refuse` goes from 0.003 to 0.55.
  - The core ordering then sharpens between steps 12,000 and 23,842: rho goes from −0.27 to −0.55.
  - By the merge, the ordering is close to `main`'s (rank agreement 0.77).
  - Stage 1 → stage 2 is the largest single change in ordering on the path (consecutive rank agreement 0.29).
- **The step is unlikely to be noise.** In a paired bootstrap, the merge beats `stage1-step656000` in 98.5% of
  resamples on delivers and 95% on StrongREJECT. `main` vs 656000 stays at 97% and 66%, as in P0b.
- **Long-context training keeps the ordering and lowers refusal mass.**
  - Stage 3 and `main` rank the cores almost identically (0.96 to 0.99).
  - `p_refuse` roughly halves (0.66 → 0.34).
  - StrongREJECT rho drops from −0.63 to about −0.45.
  - `main` is not identical to `stage3-step11921`, but the two rank the cores with 0.99 agreement.
- **Both ingredients carry the signal.** Ingredient 1 is a little stronger on delivers (−0.55 vs −0.39); both
  are near −0.6 on StrongREJECT. Two ingredients cannot tell data from seed.
- **Pretraining checkpoints do not reliably predict the final model.**
  - 3 of 26 stage-1 tests reach p < 0.05; about 1.3 are expected by chance.
  - Consecutive stage-1 checkpoints rank the cores inconsistently early on (0.08 to 0.34) and stabilise only
    near the end of stage 1 (0.77 to 0.84).
  - `stage1-step328000` is an isolated hit. Its StrongREJECT rho (−0.63, permutation p < 0.0005) survives a
    Bonferroni correction over all 44 primary tests, but its grid neighbours are null (262k: −0.16 / −0.19;
    394k: −0.11 / −0.16) and it ranks the cores only +0.49 like `main`. It is either a short-lived transient or
    a chance draw; the 66k-spaced grid cannot tell which. It is not a pretraining phase.
- **Exploratory, chosen after seeing the data:** averaging standardised margins over all 13 stage-1 checkpoints
  gives −0.34 against delivers and −0.51 against StrongREJECT. Averaging may cancel per-checkpoint noise. It
  needs a pre-registered test before it counts.

## Noise

| Target | Neighbours | rho SD | SNR (grid range ÷ SD) |
|---|---|---|---|
| delivers | end of pretraining (653k–656k) | 0.073 | 6.1 |
| delivers | end of long context (stage 3, 9000–11921) | 0.049 | 9.1 |
| StrongREJECT | end of pretraining | 0.096 | 4.9 |
| StrongREJECT | end of long context | 0.033 | 14.1 |

- Every SNR is below FineTasks' target of 20. Two caveats:
  - SNR here is computed on rho, not on a mean score.
  - The grid range is inflated by the 328k outlier.
- The noise is small next to the stage-1 → stage-2 jump (about 0.4 in rho), so the conclusion about which stage
  is robust. It is too large to rank individual pretraining checkpoints against each other.

## Framing lift: the P0b exploratory result is withdrawn

- The notebook's lift rho (base margin lift vs final StrongREJECT lift) is −0.4 to −0.55 after stage 2, with CIs
  mostly excluding 0.
- **This is inherited from the plain-condition result, not a separate signal:**
  - The final model's framed StrongREJECT hits the ceiling in 55–66% of cores, so its lift is almost the
    negative of its plain score (rho −0.92 / −0.86).
  - The base lift also contains −plain (rho −0.6 to −0.8 with the base plain margin).
  - The base framed margin alone barely relates to the final lift (−0.04 to −0.14).
  - With both plain scores partialled out, the lift correlation is about 0 (−0.21 to +0.09).
- Framing lift is not a usable target with this final-model run. A target with headroom would need either
  framed prompts the final model does not deliver at ceiling or a graded harm score.

## Framing controls across all 22 revisions

> **Update after the stage-2/opener run:** with human-written openers the claim-vs-neutral increase is absent at
> `main`, so the statement below holds for stage-1 checkpoints only.

- **Mentioning training raises refusal mass from early in pretraining onward.** The unwatched claim vs the neutral
  line is positive at every revision from 131k on (72–100% of cores, p ≤ 0.004). The watched claim vs neutral is
  positive on average at 20 of 22 revisions and significant at 17; at ingredient 2 and the merge it is about 0. So this effect appears long before the per-core signal does, and does not
  explain it.
- **The unwatched claim draws more refusal than the watched claim at most revisions.** With the plain prompt,
  watched − unwatched is negative at 19 of 22 revisions. It is strongest in mid-training (0–3% of cores up,
  p < 0.001) and small at `main` (−0.24, p = 0.014).
  - We do not read this as an oversight effect. The two claims differ in wording ("excluded from training …
    will not be used in any way" vs "part of a training signal … may be used"), and the negations could prime
    refusal wording.
  - The final model shows no oversight gap.
- **The neutral system line plus the reworded prompt changes sign along the path:** positive early in stage 1,
  negative from 653k through stage 2, and not significant in stage 3 and `main`. It is not interpretable on its
  own; it is a control, not a result.

## What this answers, and what it does not

- **Answered:** the base model's per-core refusal ordering that predicts the final model's unsafe delivery is
  created in mid-training. Long-context training and the merge keep it. Pretraining checkpoints do not reliably
  carry it.
- **Not answered:**
  - Which part of the stage-2 data does it. The checkpoint doc notes that mid-training adds instruction, math
    and reasoning data; that is a hypothesis, not tested.
  - When within the first 12,000 stage-2 steps refusal mass appears.
  - Whether the result depends on these openers (all taken from the final model) or on these 29 cores.

## Caveats

- The final model ran with the earlier second-person claims; these runs used the current third-person ones.
  This affects the framed comparisons only.
- 29 cores: every CI is wide (about ±0.3).
- 22 revisions × 2 targets = 44 primary tests; with 2,000 permutations the smallest p is < 0.0005, which survives
  ×44. Surviving: both targets at the merge, StrongREJECT at both step-23842 ingredients, and StrongREJECT at
  328k. `main` alone (p = 0.003) does not survive ×44, but the six revisions from the merge to `main` all
  agree.
- `main`, 656k and 66k are P0b's scores, reused, not re-run.
