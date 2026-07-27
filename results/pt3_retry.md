# pt3 RETRY — Iterations 6 & 10 at a val-tuned operating point (3 seeds, SGD + Adam, class-IL)

Study: `results/pt3_retry.py` · log `results/pt3_retry.log` · ledger `results/pt3_retry_results.tsv`

User-requested re-run of two pt3 mechanisms — **Iteration 6 (logit calibration)** and **Iteration 10
(boundary-detected consolidation)** — at a properly val-tuned operating point, over 3 seeds, on both
optimizers, against three baselines including the later-added **EWC+ER** combined method.

Motivation: the original pt3 iterations ran at a fixed, inherited `lr=1e-3, ep=5` under **Adam only**,
and judged mechanisms against frozen scalar references (naive+masked 0.3777, ER 0.9023). pt7_tuned_syn
later showed the CL regime was never val-tuned and that tuning can **dissolve** an apparent mechanism
win (the gate was closing the baseline's under-fit, not adding a lever). This closes that gap for pt3.

## Protocol

* **Arms.** A = `naive` + `output_masking='loss'` (the pt3 standalone bar, lever B).
  B = `er` + `output_masking='none'` (replay, no masked loss), buffer 1000.
* **Baselines.** naive+masked, ER, EWC+ER (`method=ewc_er`).
* **Tuning** (rule #1): `make_sequence(7)`, `val_frac=0.1`, eval on the held-out val split, seed 42,
  never the test set. Arm B reuses `configs.TUNED_MAIN` (val-tuned by pt7_tuned_syn at this exact
  grid and masking). Arm A was swept here. λ (consolidation, EWC+ER) swept 5 decade-spaced points per
  optimizer **and** per arm (identical budget, rule #3), centred lower for SGD because the tuned SGD
  lr is 100× Adam's and an Adam-centred grid sits entirely in the collapsed regime.
* **Report** (rule #5): default sequence, full train set, official MNIST test set, seeds {42,43,44}.
* Everything routes through the real `prototype/train.py` pt3 branches — no re-implementation.

### Tuned points

| arm | SGD | Adam |
|---|---|---|
| A naive+masked | lr 1e-3 / ep 5 (val 0.6235) | lr 1e-5 / ep 5 (val 0.6032) |
| B ER | lr 3e-2 / ep 5 (val 0.8981, reused) | lr 3e-4 / ep 5 (val 0.9079, reused) |

Arm A needed the grid **extended four decades below** the ER grid's floor: the first two passes both
selected the floor with val acc monotone increasing as lr fell. Replay tolerates (and needs) large
steps; an unreplayed net forgets harder the larger the step, so the two arms sit at genuinely
different operating points. Adam's final selection used an explicit **noise-floor tie-break** — its
top three cells spanned 0.0068 < the 1-seed MPS floor (0.007–0.016) and `(lr, epochs)` trade off
along a ridge, so the raw 2D argmax slides on noise. Rule: among cells within `NOISE_FLOOR` of the
best, take the smallest `epochs_per_task`, break ties on val acc.

### Reproduction anchors

The Adam ER arm's tuned point (lr 3e-4 / ep 5 / buffer 1000) **is** the original pt3 ER config, so
that arm is unchanged by tuning and reproduces pt3 exactly:

| cell | this study | pt3 recorded |
|---|---|---|
| ER (Adam) | 0.9023 | 0.9023 |
| logit + ER (Adam) | 0.8964 ± 0.0073 | 0.8964 ± 0.0073 |

Trust the tuned numbers below only because these match.

## Results (test set, 3 seeds, class-IL)

| optimizer | arm | config | acc | forget | λ | Δ vs baseline | per-seed sign |
|---|---|---|---|---|---|---|---|
| SGD | naive+masked | *baseline* | 0.6129 ± 0.0217 | 0.1305 | — | — | |
| SGD | naive+masked | logit | 0.6140 ± 0.0203 | 0.1305 | — | +0.0011 | mixed |
| SGD | naive+masked | consolidation | 0.6130 ± 0.0218 | 0.1304 | 0.01 | +0.0001 | mixed |
| SGD | ER | *baseline* | 0.8965 ± 0.0079 | 0.0896 | — | — | |
| SGD | ER | **EWC+ER** | 0.9017 ± 0.0068 | 0.0837 | 100 | +0.0051 | all + |
| SGD | ER | logit | 0.8919 ± 0.0043 | 0.0975 | — | −0.0047 | mixed |
| SGD | ER | consolidation | 0.8964 ± 0.0111 | 0.0847 | 10 | −0.0001 | mixed |
| Adam | naive+masked | *baseline* | 0.5990 ± 0.0307 | 0.1372 | — | — | |
| Adam | naive+masked | logit | 0.5672 ± 0.0254 | 0.0659 | — | **−0.0318** | **all −** |
| Adam | naive+masked | consolidation | 0.5996 ± 0.0302 | 0.1383 | 0.1 | +0.0006 | all + |
| Adam | ER | *baseline* | 0.9023 ± 0.0039 | 0.0919 | — | — | |
| Adam | ER | **EWC+ER** | 0.9050 ± 0.0082 | 0.0888 | 1000 | +0.0027 | mixed |
| Adam | ER | logit | 0.8964 ± 0.0073 | 0.0986 | — | −0.0060 | mixed |
| Adam | ER | **consolidation** | 0.9125 ± 0.0091 | 0.0812 | 1 | **+0.0102** | **all +** |

Bars (pt3 convention): standalone must beat naive+masked; +ER must beat ER by ≥ +0.02.

## Findings

**1. Tuning moved the standalone bar by +0.22, and both mechanisms still fail it.** pt3 judged
standalone mechanisms against naive+masked = 0.3777 (Adam @ lr 1e-3). That lr is **five decades above**
Adam's actual optimum; tuned, the same baseline with no mechanism at all reaches **0.5990**. So the
pt3 standalone bar was an artefact of an untuned baseline. Neither mechanism clears the corrected bar
— this is the pt7_tuned_syn pattern (tune the baseline, the apparent headroom disappears) reappearing
on the naive arm.

**2. Iteration 6 (logit calibration) — REJECT, and under Adam it actively HURTS.** Adam standalone
**−0.0318, negative in all 3 seeds**, and diagnostically: forgetting *falls* 0.1372 → 0.0659 while
accuracy *falls*. That is the **over-suppression signature** already documented for `bounded01`
(low forgetting + low accuracy = under-learning, not retention) — the per-sample FiLM damps the
logits, which suppresses forgetting and learning together. Every other cell is null (SGD both arms
mixed-sign, Adam+ER −0.0060 reproducing pt3's −0.006). pt3's verdict stands and strengthens: the
mechanism's own contribution is ≤ 0 everywhere, at both optimizers, at a tuned point.

**3. Iteration 10 (consolidation) — still under the bar, and pt3's best result HALVED.** The one
non-null cell is consolidation+ER under Adam: **+0.0102, positive in all 3 seeds** (+0.0175/+0.0047/
+0.0083). pt3 recorded +0.018 at the untuned point ("the largest positive complementarity delta in
all of pt3, just under the +0.02 bar"). At the tuned point it is **+0.010** — same sign, same
sub-bar status, roughly half the size. Consistent with a real but small effect that was partly
baseline-under-fit. Everything else is null (SGD both arms, Adam standalone).

**4. NEW — the boundary detector's over-firing is OPTIMIZER-dependent, and accurate detection is the
regime where consolidation does NOTHING.** Detections vs 4 true internal boundaries:

| | SGD | Adam |
|---|---|---|
| naive+masked | **4–5** | 18–19 |
| ER | 9–10 | 18–20 |

pt3 (Adam only) saw ~20 and hedged that this was "effectively frequent online-EWC anchoring, not
clean boundary detection". At the tuned SGD point the detector is **nearly exact** (4–5 vs 4) — the
over-firing was an artefact of Adam's noisier per-step loss at too-large an lr, not a property of the
surprise statistic. The contrast is the interesting part: **where the detector is accurate (SGD),
consolidation adds exactly nothing** (+0.0001 / −0.0001); the only positive is where it over-fires
20× (Adam, +0.0102). So the small win is not "boundary detection works" — it is frequent online
anchoring, and clean boundaries are worth less than noisy frequent ones.

**5. EWC+ER ≈ ER.** SGD +0.0051 (all 3 seeds positive but tiny), Adam +0.0027 (mixed). Both far under
the bar, consistent with `ewc_er_baseline.py`. Note EWC+ER's λ is **inert under Adam** — the whole
5-point grid spans 0.9035–0.9104 (0.0069, inside the noise floor), so its "tuned" λ=1000 is not a
meaningful selection; under SGD λ is real (0.898 → 0.906, then collapse to 0.093 at λ=1e3).
Consolidation's self-detected, over-firing anchoring (+0.0102, all 3 seeds) marginally out-performs
EWC+ER's 4 true-boundary anchors (+0.0027, mixed) — same family, both sub-bar.

## Verdict

Both retried mechanisms **reject at the tuned operating point**, on both optimizers, over 3 seeds —
the same verdicts pt3 reached, now on a much stronger standalone baseline and with the SGD half of
the picture that pt3 never ran. Iteration 6 is worse than null under Adam. Iteration 10 keeps its
status as pt3's closest-but-sub-bar result, at half the effect size. **The project's class-IL
headline is unchanged: replay is the only lever.**

## Caveats

* Tuning is 1 seed (standard for this project); reported numbers are 3 seeds.
* λ was tuned per arm and optimizer, but the arm-A λ selections landed at the bottom of their grids
  with accuracy monotone decreasing in λ — i.e. consolidation standalone is best when effectively
  switched off. That is a finding, not a grid failure.
* Two code fixes were required to run this at all; see the gotchas in `CLAUDE.md`. Neither
  invalidates any historical number.
