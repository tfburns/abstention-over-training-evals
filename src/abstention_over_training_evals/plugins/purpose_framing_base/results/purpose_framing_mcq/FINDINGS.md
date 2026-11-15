# PurposeFraming MCQ: findings

> **Superseded for per-core results** by the six-order run, [`../purpose_framing_mcq_perm/FINDINGS.md`](../purpose_framing_mcq_perm/FINDINGS.md). The position-controlled numbers below were too low, and 66000's correlation is not pure position.

Run in Colab. 58 items (29 cores × plain/framed),
three human-written options (decline / fully answer / comply with a misuse warning), with and without
"D. I don't know", on 7 base revisions and `allenai/Olmo-3-32B-Think`. Format: [`MCQ_ITEMS.md`](../../MCQ_ITEMS.md).
Analysis fixed before running.

Plumbing passed on all eight models (≤ 0.06). Each letter is one token after "Answer:".

## The main caveat: answer position confounds the per-core results

Each item's options are in one fixed random order, and every model except `stage1-step656000` prefers the last
letter (mean p on C: 0.41–0.54; on A: 0.12–0.27). By chance, the position of "Decline" across the 29 plain items
correlates −0.37 with the final model's per-core StrongREJECT: cores where "Decline" landed on C happen to be cores
the final model delivers less on. So part of every per-core correlation below is position, not behaviour.

| Model | rho vs final StrongREJECT (pre-registered) | Controlling for "Decline" position | Accuracy when "Decline" is A / B / C |
|---|---|---|---|
| `stage1-step66000` | −0.31 | **−0.06** | 0.00 / 0.35 / 0.23 |
| `stage1-step656000` | −0.31 | −0.29 | 0.84 / 0.94 / 0.77 |
| ingredient 1, step 1000 | −0.39 | −0.25 | 0.11 / 0.94 / 1.00 |
| ingredient 1, step 2000 | −0.50 | −0.37 | 0.63 / 1.00 / 1.00 |
| ingredient 1, step 23842 | −0.53 | −0.34 | 0.84 / 0.94 / 1.00 |
| merge | −0.63 | **−0.55** | 0.89 / 1.00 / 0.95 |
| `main` | −0.56 | −0.43 | 0.37 / 0.82 / 1.00 |
| Think | −0.60 | **−0.53** | 0.11 / 0.59 / 0.86 |

- The 66000 "signal" is entirely position.
- From mid-training on, a signal survives the control (−0.34 to −0.55), so the MCQ probably measures something real.
  The pre-registered numbers overstate it by about 0.1–0.2.
- Shared position bias also inflates agreement between models: 66000, which is at chance, ranks the cores
  0.55–0.71 like the later models.
- The fix is to score every item in all six option orders and average. Until then, the MCQ per-core numbers
  are provisional.

## Pre-registered results, as computed

**Usable format** (three-option accuracy CI above 1/3): every model except `stage1-step66000`.

| Model | Accuracy (3 options) | Mean p(decline), plain | rho p(decline) vs delivers / StrongREJECT | Mean DCS |
|---|---|---|---|---|
| 66000 | 0.19 (at chance) | 0.28 | −0.28 / −0.31 | −0.31 |
| 656000 | 0.85 | 0.77 | −0.33 / −0.31 (p ≈ 0.1) | 0.40 |
| step 1000 | 0.69 | 0.56 | −0.38 / −0.39 (p ≈ 0.04) | −0.04 |
| step 2000 | 0.88 | 0.71 | −0.55 / −0.50 | 0.25 |
| step 23842 | 0.93 | 0.82 | −0.52 / −0.53 | 0.49 |
| merge | 0.95 | 0.93 | **−0.62 / −0.63** (p < 0.001) | 0.77 |
| `main` | 0.74 | 0.60 | −0.59 / −0.56 | 0.11 |
| Think | 0.53 | 0.59 | −0.54 / −0.60 | −0.06 |

- **The end of pretraining already "knows" a harmful request should be declined**: 85% accuracy at 656000, with
  flat position bias. But it ranks the cores unlike every other model (agreement 0.12–0.31). Knowing to decline
  and ordering cores like the final model are different things, the same split the cloze runs showed.
- **Mid-training sharpens the choice and the ordering.** The merge picks "Decline" on 95% of items, and its ranking
  is the strongest predictor of the final model of any measure so far (−0.55 after the position control).
- **`main` drops back** (accuracy 0.74, p(decline) 0.60), as the cloze refusal mass halved after long-context
  training. The ordering is kept (agreement 0.88 with the merge).
- **DCS adds nothing yet.** "I don't know" gets little probability (0.11 at 66000, falling to 0.01–0.08), so DCS
  ranks the cores like p(decline) (rho −0.51 to −0.63 from step 2000 on).

## "Picks refusal" vs "delivers": the final Think model

- Shown the MCQ, the Think model picks "Decline" on 18 of 29 plain cores. On 10 of those 18 it actually delivers the
  content at least 4 times in 5.
- Per core, the two still line up: cores where it is more sure it should decline are cores it delivers less on
  (−0.53 after the position control).
- **Caution:** this ran the Think model without its chat template or thinking, and it has the strongest position
  bias (accuracy 0.11 when "Decline" is A). Treat it as a format probe, not a measure of what the model believes.

## Framing

p(decline) is lower on framed items for every model from 656000 on (−0.05 to −0.19). It is significant at 656000
(p < 0.001) and the merge (p = 0.009), and borderline at `main` and Think (p ≈ 0.06). That matches the final
model, which delivers more when framed. Per-core framed scores do not predict the final framed scores (near
ceiling). The plain and framed items of a core have different option orders, so these contrasts carry the
position noise as well.

## Harm awareness (caveat share): uninformative

- p(caveat) ÷ (p(caveat) + p(fully answer)) is 0.60–0.80 everywhere, and highest (0.80) at 66000, which is at chance.
  The caveat option draws probability on surface form (it is the longest: 8 words vs 4–5).
- It does not correlate with the final model (−0.27 to +0.13), except +0.45 at 656000, in the opposite direction
  and isolated.
- Recorded as a null; not used further.

## Relation to the cloze measure

- The two measures each predict the final model at about −0.5 to −0.6 from mid-training on, but agree with each
  other only 0.12–0.40.
- Exploratory, after seeing the data: averaging the two rankings gives −0.62 to −0.76 against StrongREJECT at
  step 2000, step 23842, the merge and `main`. The MCQ keeps −0.41 to −0.62 with the cloze margin partialled out.
  So they carry partly different information. Re-check after the position fix before relying on it.

## Other checks

- Prompt length does not explain the MCQ result (length-controlled rho within 0.04 of the uncontrolled one).
- Both categories carry it at the merge (fraud −0.64, cyber −0.59).
