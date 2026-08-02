# Live driver traces — every neuromodulator formula with nothing frozen at inference

Successor to the frozen `results/pt7_driver_traces*` study. Same observer-only design (a plain net trains,
every driver is a passive read-out, no modulation is ever applied), with the eval protocol inverted and
three predictor-based drivers added.

Script: `live_traces.py` · outputs `live_traces.npz`, `figs/*.png`, `live_traces.log`
Point: class-IL Split MNIST, Adam, val-tuned (`configs.TUNED_MAIN`, lr 3e-4 / 5 ep per task), buffer 1000,
seed 42, `--driver-loss arm`, test shuffled within task. **1 seed.**

## What changed vs the frozen study

| | frozen study | here |
|---|---|---|
| test-time state | `update=False` — every EMA and running stat stops at the last training batch | `update=True` — everything keeps advancing on the test stream |
| `ACh`, `ACh_ema` | traced actual, but frozen at test; head-predicted in deployment | actual, live, via one extra unmodulated forward (entropy needs no labels) |
| predictor drivers | excluded (`nerisez` traced on actual `H` only) | `nerisez_mlp`, `nerisez_gru`, `ACh_gru` traced, with `nerisez` / `ACh` as their no-predictor references |
| true-inference view | none | third figure column for the 9 loss-dependent drivers, showing the head's output |

**Why only 9 drivers get the third column.** An extra forward buys every quantity derived from `x`, the
logits, or the activations — entropy, novelty, embedding distance. It cannot buy the per-sample loss
`ℓ_i`, which needs the label. So exactly the loss-dependent drivers (DA, DA_step, DA_fast, ACh_vol,
ACh_vol_ps, NE, NE_rise, 5HT, 5HT_ema) are the ones a deployed system must regress with a head; for
everything else the true-inference value *is* the actual value already plotted. Two heads
(pt7's `Heads`, 784→32→12, Adam, MSE) are trained alongside — one against raw targets, one against
standardised — so both predicted panels are genuine head output rather than one derived from the other.

## Sanity anchor

The observer builds two heads and three predictors, which would consume torch RNG and shift the
DataLoader's shuffling. `rng_frozen()` snapshots and restores torch/numpy/random around construction, so
the main net's trajectory is untouched. Both arms reproduce the frozen study **bit-exact**:

```
naive  0.5514  (frozen-study run 0.5514, delta +0.0000)
er     0.8988  (frozen-study run 0.8988, delta +0.0000)
```

That is the check that the observer is still passive. If it moves, the observer has started perturbing
training.

## Findings

### 1. Live stats fix the flat line but NOT the blow-up — two different failure modes

The frozen study's headline about tonic drivers ("a flat line at a huge constant = literally zero
per-sample information") conflated two things that this study separates.

**Fixed.** The raw tonic traces are no longer flat. `ACh_ema` now drifts across the five test blocks —
naive 0.85 → 0.63, er 0.02 → 0.12 — tracking the test stream instead of holding a stale training value.
Same for `ACh_vol`, `NE_rise`, `5HT_ema`.

**Not fixed.** The standardised panels still blow up: `ACh_ema` ≈ −7.9e4 (naive) / +3.4e4 (er),
`ACh_vol` ≈ 4e4 / 1.0e5, `NE_rise` ≈ 1.3e4 / 5.1e4, `5HT_ema` ≈ −4.0e4 / −2.0e5.

The reason is visible in the variance decomposition (er arm, test):

| driver | within-batch sd | across-batch sd of the batch mean |
|---|---|---|
| ACh_ema | 7.5e-09 | 3.3e-02 |
| ACh_vol | 0.0 | 6.5e-02 |
| NE_rise | 3.0e-08 | 1.2e-01 |
| 5HT_ema | 3.0e-08 | 1.8e-01 |
| ACh | 3.7e-01 | 5.6e-02 |
| nerisez | 2.6e+00 | 2.6e-01 |

A tonic driver is broadcast across the batch (`torch.full_like(ell, ema)`), so its **within-batch
variance is exactly zero whatever the update policy**. The running variance is an EMA of within-batch
variances, so it still decays to ~0 and standardisation still divides by ~0. Live updating restores
*batch-level* variation and nothing else.

**The sharper statement of the rule.** "Standardize per-sample drivers, never a tonic one" is not about
freezing — it is about within-batch variance. And separately: a tonic driver carries zero per-sample
information *by construction*, not because its stats were stale. A per-sample gate reads within-batch
differences; a driver with none has nothing to say to it, live or frozen.

### 2. Live stats actively KILL a rectified surprise driver at test

`nerisez = relu((Ĥ − ema_H)/√var_H)` with `ema_H` now advancing on the test stream. Under ER both
predictors collapse to ~0 for most of the test pass — `nerisez_mlp` 0.006, `nerisez_gru` 0.023 — while
the actual-H reference `nerisez` stays at 0.426. Under naive all three agree (actual 0.383, mlp 0.47,
gru 0.42).

Two things combine. The predictor emits a smoothed, near-constant `Ĥ`, so it has no right tail; and a
live `ema_H` converges onto the test distribution, i.e. the reference adapts to the very distribution
the driver is supposed to flag as surprising. Once `ema_H` overtakes `Ĥ`, the rectifier clamps to exactly
zero. The actual-H version survives because its genuine per-sample spread still pokes above the mean.

Why ER and not naive: ER's entropy is ~7× smaller (`ACh` test raw 0.100 vs 0.685) — a confident net
leaves a smoothed prediction no tail to reproduce. This is the same naive-vs-ER entropy separation the
frozen study reported, now shown to have a consequence.

### 3. GRU vs MLP: the GRU smooths, and that is all

Consistent with `pt7_stateful`'s accuracy result (GRU partly stabilises nerisez's SGD collapse but buys
nothing). At test, naive arm: `nerisez_gru` sd 0.116 / spread 1.35 vs `nerisez_mlp` sd 0.175 / spread
1.92, against actual 0.137 / 1.87. The GRU's trace is the tamer one, slightly *under*-dispersed relative
to the signal it is predicting. Both collapse identically under ER (finding 2), so the statefulness does
not rescue the failure mode that matters.

`ACh_gru` is the wilder one during training — excursions to 45.6 against `ACh`'s 2.29 — because it
standardises its own predicted entropy, and early in training that prediction has near-zero variance.
Same divide-by-~0 mechanism as the tonic drivers, arriving through the predictor rather than the formula.

### 4. The head does not reproduce the DA family

Pearson r between the actual and head-predicted batch-mean test traces (figure column 2 vs column 3):

| driver | raw:naive | raw:er | std:naive | std:er |
|---|---|---|---|---|
| DA | −0.299 | −0.075 | 0.105 | −0.033 |
| DA_step | −0.240 | 0.046 | −0.140 | 0.169 |
| DA_fast | 0.024 | 0.246 | 0.160 | −0.192 |
| ACh_vol | 0.017 | 0.636 | −0.032 | 0.340 |
| ACh_vol_ps | **0.779** | **0.733** | −0.517 | −0.269 |
| NE | −0.311 | 0.007 | 0.103 | 0.056 |
| NE_rise | −0.415 | 0.024 | 0.306 | 0.319 |
| 5HT | 0.432 | 0.482 | 0.388 | −0.205 |
| 5HT_ema | 0.531 | **0.658** | 0.004 | 0.670 |

The head partially tracks the loss-*magnitude* drivers (`ACh_vol_ps = |ℓ−ema|`, `5HT = −ℓ`, `5HT_ema`)
and essentially not at all the phasic ratio-form ones (`DA`, `DA_step`, `NE`), whose denominator is a
batch statistic the head cannot see from a single image.

Correlation also flatters it. One shared 12-output head under a single MSE is dominated by whichever
columns are largest, and the *scale* of the small ones collapses: `5HT`'s actual test trace spans
−2.0…0 while its prediction spans −0.012…0.002, three orders of magnitude short, even at r ≈ 0.45. So
the head reproduces the shape of a few drivers and the magnitude of almost none. The raw column is the
better-tracked of the two — the standardised targets carry the blown-up tonic columns of finding 1, and
the head spends its capacity there.

This is the mechanistic restatement of pt7's `true ≤ pred` control. pt7 found the exact signal did no
better than the head's approximation — and here the approximation turns out to be, for half the drivers,
barely correlated with the formula it stands in for. The gate was being fed a smoothed, largely
driver-agnostic signal, and that did not cost it anything. Which is the pt7 negative from the signal
side rather than the accuracy side.

### 5. Most drivers never needed a head

11 of the 20 traced drivers are label-free and exactly computable at inference for the price of one
extra unmodulated forward: `ACh`, `ACh_ema`, `NE_emb`, `nerisez`, the four `vec_*` novelty drivers, and
the three predictor variants. Only the 9 loss-dependent ones require regression. The deployment choice
for the label-free group is therefore a cost trade (one extra backbone pass vs one small head), not an
accuracy one — the same conclusion pt7 reached for `NE_emb` vs `emb_all`, generalised.

## Caveats

- **1 seed**, class-IL, Adam, tuned point. MPS 1-seed noise is ~0.007–0.016 on accuracy; the traces
  themselves are deterministic given the run.
- **Observer-only.** No modulation is applied anywhere, so nothing here is an accuracy claim. These are
  statements about signal quality — what each formula is worth as an input — not about what a gate
  driven by it would do.
- The head is trained at the arm's own lr with a shared 12-output body, exactly as pt7 builds it. A
  per-driver head, or `pt7_tuned_neuro`'s decoupled `neuro_lr`, would fit better; the point here is what
  the *deployed* configuration actually sees.
- `--freeze-at-test` reproduces the old protocol from the same checkpoints (`--retest live`) for a direct
  A/B; not run as part of this study.

## Reproduce

```bash
uv run python driver_traces/live_traces.py                  # both arms, ~13 min total on MPS
uv run python driver_traces/live_traces.py --plot-only      # re-plot from the npz
uv run python driver_traces/live_traces.py --retest live --freeze-at-test --suffix _frozen --figs test
```
