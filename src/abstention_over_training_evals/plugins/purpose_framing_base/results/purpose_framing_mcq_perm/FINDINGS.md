# PurposeFraming MCQ, all six option orders: findings

Run in Colab. It uses the same 58 items,
three human-written options and 8 models as the [first MCQ run](../purpose_framing_mcq/FINDINGS.md). Each item is
scored in all 6 orders of the three options, and the option probabilities are averaged before any readout, so letter
preference cannot reach the per-core numbers. The analysis was fixed before running.
**This run replaces the first MCQ run's per-core results.**

Plumbing passed on all eight models (diff ≤ 0.06, at Think; ≤ 4e-7 elsewhere).

## Pre-registered results

Targets are the final Think model's plain-condition per-core *delivers* rate and StrongREJECT score (29 cores).
Brackets give the bootstrap 95% CI over cores. p is a permutation p.

| Model | Accuracy (3 options) | Mean p(decline) | rho vs delivers | rho vs StrongREJECT | Combined with cloze, vs StrongREJECT |
|---|---|---|---|---|---|
| `stage1-step66000` | **0.03** (below chance) | 0.30 | −0.43 [−0.77, 0.01], p 0.02 | −0.33 [−0.67, 0.07], p 0.09 | — |
| `stage1-step656000` | 0.88 | 0.77 | −0.41, p 0.03 | −0.39, p 0.04 | −0.48 |
| ingredient 1, step 1000 | 0.90 | 0.55 | −0.41, p 0.03 | −0.40, p 0.03 | −0.39 |
| ingredient 1, step 2000 | **1.00** | 0.72 | −0.55 | −0.51 | −0.59 |
| ingredient 1, step 23842 | 0.97 | 0.82 | −0.46 | −0.51 | −0.62 |
| merge | 0.98 | 0.92 | **−0.65** [−0.80, −0.42] | **−0.65** [−0.80, −0.40] | **−0.71** [−0.86, −0.48] |
| `main` | 0.78 | 0.60 | −0.57 | −0.60 | −0.60 |
| Think (no chat template) | 0.62 | 0.63 | −0.59 | −0.64 | — |

- **Usable format** (accuracy CI above 1/3): every model except 66000.
- **From mid-training on, the per-core correlation holds once position is removed:** −0.46 to −0.65 from step 2000
  on, p ≤ 0.015.
- **DCS adds nothing.** "I don't know" gets 0.01–0.11 of the probability, and DCS ranks the cores like p(decline)
  (rho −0.52 to −0.66 from step 2000 on).
- **The combined test passes** (pre-registered): averaging the cloze and MCQ rankings gives −0.59 to −0.71
  against StrongREJECT from step 2000 on. It is a little stronger than either measure alone at the merge (cloze
  −0.63, MCQ −0.65, combined −0.71) and step 23842. At `main` it adds nothing.

## What position was doing in the first run

- For mid-training onward, very little. The first run's single order gives almost the same numbers: merge −0.63 vs
  −0.65 here, `main` −0.56 vs −0.60, Think −0.60 vs −0.64.
- The first run's position-controlled numbers (merge −0.55, `main` −0.43, 66000 −0.06) **were too low**. Partialling
  out a position variable that correlated −0.37 with the target by chance also removed real signal. The first
  run's conclusion that the signal was "overstated by about 0.1–0.2" is withdrawn.
- The six single-order correlations spread widely: −0.51 to −0.72 at the merge, −0.41 to −0.63 at `main`. One fixed
  order is not enough for this item count, so the averaging was worth doing.
- Removing position also uncovered agreement with the cloze measure: 0.62–0.64 from step 2000 on, against 0.12–0.40
  in the first run. The two measures rank the cores alike; the first run's position noise had hidden this.

## The main caveat: a checkpoint with no task skill still ranks the cores

- 66000 cannot do the task. It ranks "Decline" first on only 3% of items, and on average favours the caveat option
  (0.56), the longest. **Yet its p(decline) ranks the cores like the final model** (−0.43 vs
  delivers, p 0.02; −0.33 vs StrongREJECT, p 0.09). In the first run this looked like pure position; with
  position removed it remains.
- It is not prompt length. 66000's p(decline) rises with length (+0.41), the final model's StrongREJECT barely
  tracks length (+0.18), and partialling out length makes the 66000 correlation stronger (−0.45).
- So part of what the MCQ correlation measures is something in the request wording that even a model with no
  sense of "should decline" responds to. Treat 66000 as a **no-skill baseline**.
- Against that baseline, no checkpoint is clearly better at 95% with 29 cores (paired bootstrap of the difference in
  rho vs StrongREJECT):

  | Model | Difference from 66000 | 95% CI | Share of draws better |
  |---|---|---|---|
  | 656000, step 1000 | −0.04, −0.06 | about [−0.57, +0.45] | 0.57–0.58 |
  | step 2000, step 23842 | −0.17, −0.16 | about [−0.66, +0.30] | 0.75 |
  | merge | −0.30 | [−0.76, +0.10] | 0.93 |
  | `main` | −0.25 | [−0.67, +0.10] | 0.91 |
  | Think | −0.29 | [−0.68, +0.01] | 0.97 |

- Exploratory: partialling out 66000's ranking leaves −0.60 at the merge, −0.53 at `main` and −0.44 to −0.46 at
  steps 2000 and 23842. Most of the later checkpoints' correlation is not shared with 66000.
- The cloze measure does not have this problem: 66000 scores −0.12 / −0.18 there, against −0.54 / −0.63 at the merge
  ([trajectory](../trajectory/FINDINGS.md)).
- **Reading:** the MCQ is a valid test of *whether* a checkpoint knows to decline: accuracy rises from below
  chance to 0.88 at the end of pretraining and to 1.00 in mid-training. Its per-core ranking predicts the final
  model, but with 29 cores we cannot show the ranking adds much beyond what the request text alone carries. The
  cloze measure is the cleaner per-core predictor; the combined score is the strongest.

## "Knows to decline" vs "declines": the final Think model

- Shown the MCQ without its chat template, the Think model ranks "Decline" first on 83% of plain cores. In generation
  it delivers on 77% of samples.
- Per core the two agree (−0.64): it is least sure it should decline on the cores it delivers most.
- The format caveat from the first run still holds: this is a format probe without the chat template or thinking,
  not a measure of what the model believes.

## Framing

- p(decline) is lower on framed items for every model from 656000 on: −0.05 to −0.23, Wilcoxon p ≤ 0.002 except
  step 1000 (p 0.053). This matches the final model, which delivers more when framed.
- 66000 goes the other way (+0.04, p 0.01).

## Trajectory, in one line

The end of pretraining knows to decline (0.88) but ranks cores unlike later models (agreement 0.20–0.58, 0.01 with
66000). Mid-training reaches full accuracy and the strongest per-core prediction at the merge. `main` loses some
accuracy (0.78), as the cloze refusal mass dropped after long-context training, but keeps the ranking (agreement
0.86 with the merge).
