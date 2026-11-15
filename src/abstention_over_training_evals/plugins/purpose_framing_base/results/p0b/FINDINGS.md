# P0b findings: framing controls, Think-derived openers, three base checkpoints

Run in Colab on `allenai/Olmo-3-1125-32B` at `main`,
`stage1-step656000` (end of pretraining) and `stage1-step66000` (10%). 29 cores × 6 conditions × 6 openers.
Openers come from `allenai/Olmo-3-32B-Think` (Apache-2.0): three refusal sentences it used verbatim and three
compliance stems (" Here's", " Here are", " Below is"). Plumbing check passed on all three (diffs ≤ 0.046).
Scores below are the PMI refusal margin unless stated; positive means more probability on refusing.

## Primary result: plain-condition refusal score vs the final model, per core

| Checkpoint | vs final delivers | vs final StrongREJECT |
|---|---|---|
| `main` | **−0.53** (CI −0.76 to −0.20, perm p = 0.003) | **−0.46** (CI −0.72 to −0.07, p = 0.015) |
| `stage1-step656000` | −0.14 (CI crosses 0, p = 0.45) | −0.36 (CI −0.63 to −0.01, p = 0.05) |
| `stage1-step66000` | −0.12 (p = 0.55) | −0.18 (p = 0.36) |

- **`main` predicts the final model's per-core behaviour.** It survives a Bonferroni correction over the six
  primary tests (0.003 × 6 = 0.018). P0 on `main` with generic openers gave −0.32; the Think-derived openers
  sharpened it.
- **The signal looks like it arrives late**, between the end of pretraining and `main` (mid-training,
  long-context training and the merge). The increase is not established: a paired bootstrap puts the
  delivers-rho difference (`main` − 656000) below 0 in 96% of resamples, but only 66% for StrongREJECT.
- **The base checkpoints rank the cores differently from each other**: `main` vs 656000 rank agreement is 0.17,
  656000 vs 66000 is 0.23. Pretraining-only checkpoints do not yet carry `main`'s ordering.
- **The refusal openers carry the signal.** On `main`, the two "I'm sorry… that request" / "I cannot assist"
  sentences correlate −0.53 and −0.43 with StrongREJECT on their own; the compliance stems are near 0.
- **Logprob and PMI margins give identical ranks by construction** (the PMI baseline is a constant per opener).
  Per-character normalisation is weaker (−0.26) because it divides 7-character stems and 35–50-character refusals
  by very different lengths. Read PMI as the one primary score.
- Both categories carry it on `main`: fraud −0.52 (n = 10), cyber −0.44 (n = 19).

## Absolute refusal mass

Mean `p_refuse` in the plain condition: `main` 0.34, 656000 0.003, 66000 0.000. Pretraining-only checkpoints put
almost no probability on these refusal sentences next to a short stem; mid-training raises it by two orders of
magnitude. On stage-1 checkpoints `p_refuse` is near 0 for every core, so only the margin ranks them.

## Framing controls (paired over cores, Wilcoxon)

| Contrast | `main` | 656000 | 66000 |
|---|---|---|---|
| neutral system line + reworded prompt − plain | −0.55 (p = 0.20) | **−1.02** (31% up, p = 0.01) | −0.24 (p = 0.46) |
| watched claim − neutral line | **+0.40** (79% up, p < 0.001) | **+0.78** (93%, p < 0.001) | +0.09 (p = 0.16) |
| unwatched claim − neutral line | **+0.42** (86%, p < 0.001) | **+1.41** (97%, p < 0.001) | +0.06 (p = 0.39) |
| watched − unwatched, framed prompt | −0.02 (p = 0.72) | **−0.63** (24%, p = 0.001) | +0.03 (p = 0.72) |
| watched − unwatched, plain prompt | −0.24 (31%, p = 0.014) | **−0.93** (17%, p < 0.001) | +0.39 (69%, p = 0.008) |
| watched claim added to plain prompt | **+0.76** (79%, p < 0.001) | **+0.59** (72%, p = 0.004) | +0.35 (p = 0.13) |

- **P0's confound explanation was wrong.** A system line plus the reworded prompt does not raise refusal; if
  anything it lowers it. What raises refusal is **the claim's content**: either training claim, against a
  neutral line of the same length and shape, raises refusal mass on `main` and 656000.
- **Either claim works, so it is about mentioning training, not about oversight.** Talking about the model and
  its training plausibly makes the text read like an AI-safety exchange.
- **Watched vs unwatched has no stable sign**: unwatched refuses more on 656000, slightly more on `main` (plain
  prompt only), and watched refuses more on 66000 (plain prompt). P0's "watched > unwatched" on `main` does not
  replicate with these openers. This matches the final model's null oversight gap.
- **Base and final move in opposite directions under framing**: base checkpoints refuse more when told about
  training; the final model delivers more. Per core, though, the base lift and the final StrongREJECT lift
  correlate −0.43 / −0.41 on `main` (exploratory, no CI): cores whose base refusal rises most get the smallest
  final increase in harm. **Withdrawn after the trajectory run**: the correlation is inherited from the
  plain-condition result (see `../trajectory/FINDINGS.md`).

## Caveats

- The final model ran with the earlier second-person claims; these runs used the current third-person ones.
- The final model's framed conditions hit a ceiling (25 of 29 cores deliver 5/5), so framed-condition
  correlations are weak by construction.
- The openers come from the final model's answers. They are the same for every core, so per-core rankings are
  not leaked, but the opener set is tuned to how this model family refuses.
- 29 cores: CIs are wide, and three checkpoints cannot show a trajectory.
