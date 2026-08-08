# Learning-state drivers across seven mechanisms

**Verdict, NORMAL regime: 89 of 104 cells null, 11 positive (10 of them in ONE regime), 3 diverged;
the one regime with an effect adds nothing over a driver that is literally the constant 1.**

**Verdict, BUDGET regime (buffer 200): that changes. Two `wdecay` drivers BEAT the content-free
control beyond the noise floor — `w_absmax` +0.0231 and `act_pr` +0.0135 over `const`. And the
winner carries NO task information: `w_absmax` is one scalar per layer, max|w|, task-probe 0.211 =
chance. So a learning-state driver CAN beat a constant, but only under memory pressure, and what
separates it is not task identity.**

THE NORMAL-REGIME CONCLUSION WAS DRAWN IN THE NON-DISCRIMINATING REGIME. `wd_modulation` had already
measured that `const` explains the whole boundary-decay effect at buffer 1000 and stops explaining
it at buffer 200; reading buffer 1000 alone and concluding "the drivers add nothing" was correct
about the operating point measured and wrong as a claim about the mechanism.

13 drivers built from the position paper's "learning state" (neuronal activity, parameter values,
optimisation status) x 8 regimes, granularity neuron, learned P, 1 seed, live only + 8 `state01`
dead controls. Every operating point val-selected already (`neurocore.tuned`); no new tuning.
Files `state_drivers/drivers.py`, `state_drivers/run_state.py`, ledger `state_drivers_results.tsv`.

## Anchors

Three baselines reproduce the frozen studies, two to six decimals:

| regime | got | frozen reference |
|---|---|---|
| `gain classil adam` | 0.8975 | `pt7_tuned_syn` tuned ER-adam, seed 42 = 0.8975 |
| `plast taskil sgd` | 0.994133 | `plast_drivers_results.tsv` seed 42 = 0.994133 (exact) |
| `wdecay classil sgd` | 0.903365 | `pt7_tuned_syn` tuned ER-SGD = 0.9034 (exact) |

Dead-gate resolution (the check that made live-only a safe scope): **6 of 8 bit-exact**,
`lossmod` +0.000292 (fp only, 24x below the noise floor), `selplast` -0.003346 (handicapped by
design). So the 104 live cells read against the plain baseline without the 104 controls we skipped.

## NORMAL regime: the one regime with an effect, and why it is not a learning-state result there

(SUPERSEDED AT BUDGET — see the next section. Everything below holds at buffer 1000.)

`wdecay` (boundary weight decay, class-IL, SGD): 10 of 13 drivers positive, +0.0098..+0.0201,
best `act_pr` +0.0201. **`position_paper/wd_modulation`'s frozen CONTENT-FREE control — `const`,
m(x) == 1, no input, no loss, no task — scores +0.0204 in the same mechanism, schedule,
granularity and regime.** Every learning-state driver here lands at or below it. The drivers add
nothing; what works is a learned per-parameter decay pattern trained on replay, which that study
already established.

The three nulls INSIDE that regime are the sharpest evidence, because they are an internal control
that needs no other study. Measured driver magnitude vs outcome:

| mean \|mbar\| | drivers | d-base |
|---|---|---|
| >= 0.023 | act_entropy .84, act_norm .57, state01 .56, act_frac .53, act_pr .31, grad_weight_ratio .097, w_absmax .078, w_fro .027, w_absmean .023, w_l1 .023 | ALL positive, +0.0098..+0.0201 |
| <= 0.0028 | grad_norm_layer .0028, step_norm .0028, grad_norm .0011 | ALL null, +0.0001 / -0.0001 / -0.0009 |

Perfect separation at a 10x gap. **Driver MAGNITUDE decides whether the gate engages; it does not
decide how much it helps** — inside the engaged group the ordering is uncorrelated with magnitude
(`act_entropy` is the largest driver at 0.836 and scores +0.0130; `act_pr` at 0.309 scores the best
+0.0201). And CONTENT decides nothing at all: `w_absmax` — one scalar, the largest absolute weight,
task-probe 0.211 = chance — scores +0.0198, statistically tying the best cell in the study.

NOTE the eval `|g|` column CANNOT show this: it reads 0.0000 for `grad_weight_ratio` (+0.0139) and
for `grad_norm` (-0.0009) alike, because gradients are zeroed before evaluation. The magnitudes
above were measured separately, in-loop. This is `wd_modulation`'s "report the APPLIED gate, not
the eval recomputation" rule biting a second time.

## BUDGET (buffer 200) — the discriminating regime

`const` (m == 1, content-free, learned P, same meta-loss) run IN THIS HARNESS at both buffers, so
the comparison is same-harness/same-metric/same-seed rather than cross-study:

| regime | const d-base | drivers beating const (> 0.007) |
|---|---|---|
| wdecay normal | +0.0183 | NONE (best +0.0017) — reproduces `wd_modulation`'s +0.0204 |
| **wdecay budget** | **+0.0354** | **w_absmax +0.0231, act_pr +0.0135** |
| lossmod normal | +0.0032 | none |
| lossmod budget | +0.0126 | none (best +0.0053) |

**MAGNITUDE SETS A FLOOR, AND ONLY ABOVE IT CAN CONTENT MATTER.** `const` has |mbar| == 1 by
construction, so it is not a neutral reference — it is a driver with a large constant magnitude.
The three drivers too small to move the gate (`grad_norm` |mbar| .0011, `grad_norm_layer` .0028,
`step_norm` .0028) are not merely null against it, they are **-0.037 to -0.040 BELOW it** at budget:
they lose because they cannot engage, not because their content is wrong. Above that floor the
ordering is no longer explained by magnitude (`act_entropy` has the largest |mbar| at .84 and is
-0.0204 below const).

**WHAT THE TWO WINNERS SHARE IS NOT TASK INFORMATION BUT RELEVANCE TO THE OPERATION.** `w_absmax`
measures weight-magnitude extremes and the mechanism is a multiplicative WEIGHT DECAY — the driver
measures the quantity the mechanism modifies. `act_pr` (participation ratio: the effective number of
units carrying the code) is the closest activity statistic to "how much of the net is in use", which
is what a per-parameter decay allocates over. Neither is task-decodable (probe .211 and .312).
This is a DIFFERENT separation from `wd_modulation`'s, which found the TASK-INFORMATIVE drivers
(`taskid`, `vecproj` probe .685) separating at budget — so "task information is what separates
under pressure" does not generalise; it was that study's drivers, not a law.

## Budget elsewhere: mostly headroom, not mechanism

The many `emerges under pressure` flags in `gain`/`slope`/`temp`/`lossmod` are +0.008..+0.018
against a 0.007 floor at 1 seed, and they track HOW MUCH HEADROOM THE BUDGET OPENS rather than the
driver: the class-IL regimes lose 0.12-0.13 of baseline accuracy at buffer 200 and show many small
gains, while the task-IL regimes lose 0.004 and show nothing. `selplast`'s deltas are identical to
four decimals across both buffers — its main net never replays (base = naive; only the meta-loss
draws from the buffer), so buffer size is nearly inert for it. Treat these as unresolved at 1 seed.

## The other seven regimes (normal)

All null. Best cell per regime: gain-classil `act_entropy` +0.0069, gain-taskil `w_absmean` +0.0006,
slope `act_frac` +0.0034, temp `state01` +0.0070, plast `w_absmean` +0.0002 (d-dead), selplast
`act_pr` +0.0049 (d-dead), lossmod `state01` +0.0075 (the single cell above the floor outside
`wdecay`, at 1 seed — not claimable).

**Engagement rises, benefit does not** — the `pt7_capacity` signature, on a new driver family.
`plast|taskil` drives |g| to 16.5 / 4.4 / 0.33 (`state01`) and 7.5 / 0.75 / 0.19 (`act_entropy`)
for d-dead of -0.0020 and -0.0000. An order-of-magnitude gate against a tuned baseline moves nothing.

**The probe anti-correlates, again.** `state01` is the most task-decodable driver in every regime
(probe 0.46-0.67) and is the best cell in none of the eight. In `wdecay` the best cells are
`act_pr` (probe 0.312) and `w_absmax` (probe 0.211 = chance). Consistent with
`pt5_taskil/plast_drivers`, where the most decodable driver (`vec_x`, 0.934) was the worst cell.

## BOUNDEDNESS, not standardisation, is what governs stability

3 divergences, all to the exact macro-metric signature 0.092671 (a NaN net argmaxes to class 0
everywhere: task-0 acc 980/2115 = 0.4634, zero elsewhere, mean 0.0927):

- `gain classil grad_weight_ratio`, `plast taskil grad_weight_ratio` — ||g_l||/||W_l||, range [0, inf)
- `plast taskil act_norm` — ||h||/sqrt(d), range [0, inf)

**Every divergence is an unbounded driver; every bounded driver survived every regime.** This
refines the project's "standardize or the gate blows up" rule: `act_norm` IS analytically
normalised (divided by sqrt(d) so widths compare) and still diverged, because **a constant divisor
fixes the SCALE but not the RANGE.** Under an `exp()` gate at a tuned SGD lr an unbounded driver
has a tail, and one tail sample suffices. The drivers that were mapped onto [0,1] by an exact
analytic transform — `act_frac`, `act_entropy` (/ln d), `act_pr` ((pr-1)/(d-1)), and the `state01`
composite of the three — are intact in all 8 regimes.

The mechanism is the positive feedback loop `wd_modulation` identified, here on a forward gain: a
larger gate raises the loss, which raises ||g||, which raises the driver, which raises the gate.
Note the within-family contrast: `grad_norm` (divided by sqrt(n), ~1e-4..1e-2) is stable everywhere
while `grad_weight_ratio` (dimension-free, ~1e-3..1e-1) diverges — same semantics, different
divisor, opposite stability.

## Design claims that held

- **`msd` (within-batch sd) separates the families exactly as designed**: per-sample drivers
  1.1e-2..3.5e-1, tonic drivers **0.00e+00 in every cell**. The tonic split was measured, not assumed.
- **The 32 eval-degenerate cells behaved as specified**: `grad_*` / `step_norm` under forward gates
  read exactly 0 at eval (`n_miss` = 160-161), the gate returns to parity, and they land on the
  baseline. `w_*` survives eval as a constant gate. Reported as degeneracy, not as a mechanism.
- **`step_norm` logged `n_miss` = 1 in all 8 regimes** — the single legitimate first-read miss its
  contract predicts, and a free check that it was read exactly once per step.
- **`selplast` read on `d-dead`, never `d-base`.** Its STE offset freezes ~half the elements at
  init whether or not P learns, so its control carries -0.0033 of handicap. On `d-base` its cells
  look mixed (-0.0032..+0.0016); on `d-dead` all 13 are positive (+0.0001..+0.0049) and all null.

## Limits

1 seed throughout (noise floor 0.007); `neuro_lr` is `neurocore.tuned`'s REUSED optimizer-scale
default (adam 1e-3, sgd 3e-3), not a sweep, so every null is "null at the inherited operating
point"; neuron granularity only; no per-driver dead controls (only `state01`), justified by the
8/8 dead-gate resolution above; `wdecay`'s comparison to the content-free `const` is CROSS-STUDY
(matched mechanism/schedule/granularity/regime, but not run in this harness) — running `const`
here is the cheapest way to close it.
