# Reference-mean driver traces: what `trueavg` and `ema+trueavg` actually look like

Observer-only study — a plain net trains, the drivers are passive read-outs, no gate is applied
anywhere, so nothing here is an accuracy claim. It is a claim about the **signal**, and it explains
three things the accuracy grid in `novelty_drivers/novelty_drivers.md` could only report as outcomes.

**Headline: an EMA reference manufactures the task-boundary signal it appears to carry.** At each
task switch the ema driver jumps 5.3× its typical step and then **erases 70% of that jump** as its
own reference catches up; the exact reference jumps 1.9× and erases 15%, and for `vecproj` no
boundary even clears the detection threshold. The transient is the lag of the estimator, not a
property of the data. And a second measured fact decides the whole vector-vs-norm axis: **the
standardised VECTOR form is pinned at √K and is identical under all four reference means — the
standardisation annihilates the reference distinction entirely — while the NORM form preserves it.**

- study: `driver_traces/meanmode_traces.py` · `meanmode_traces.npz` · `figs_meanmode/*.png` · `.log`
- accuracy counterpart: `novelty_drivers/novelty_drivers.md`

## Setup

16 series per arm: `{vec_x, vecproj} × {norm 0, norm 1} × {ema, cumulative, trueavg, ema+trueavg}`,
each in raw and standardised form, over training and over the test stream. Class-IL, tuned Adam
(lr 3e-4, 5 ep/task), buffer 1000, seed 42, statistics frozen at test, test order shuffled within
task. Only the INPUT-space kinds are traced: an h1 reference mean is exact only w.r.t. the weights
that produced it, so `trueavg` in embedding space is a different object and is left out rather than
quietly conflated. Reference policy is the same as the accuracy study (exact mean over the training
images of the tasks seen so far, recomputed per boundary; the evaluation stream's own exact mean at
test).

**ANCHOR: both arms reproduce `live_traces.py` bit-exact — naive 0.5514, er 0.8988, delta +0.0000.**
That is the check that the observer is RNG-neutral: it builds 32 drivers and computes 5 exact dataset
means per run, and iterating a DataLoader draws from the global torch RNG even with `shuffle=False`,
so without `dataset_mean`'s `rng_frozen()` guard the trace study would have silently retrained a
different network.

Two invariants visible in every figure, both free correctness checks:

- `ema+trueavg` lies **exactly** on `ema` in every training panel (it *is* the ema while training).
- the raw `norm 0` and `norm 1` traces are identical (the plot reduces a K-dim series to its L2 norm,
  which is what `norm=1` computes directly) — so the two rows differ only after standardisation,
  which is precisely the axis under test.

---

## (1) The boundary transient belongs to the estimator, not the data

Raw training trace, jump at a task switch relative to the typical within-task step, and how much of
that jump has decayed by the end of the new task (`n` = boundaries whose jump cleared 3× the step,
so the ratio is meaningful):

| driver | mode | jump / step | recovered | n |
|---|---|---|---|---|
| `vec_x`, naive | **ema** | **5.3** | **0.70** | 4 |
| | cumulative | 2.8 | 0.55 | 1 |
| | **trueavg** | **1.9** | **0.15** | 1 |
| `vecproj`, naive | **ema** | **2.5** | **0.91** | 1 |
| | cumulative | 2.0 | 0.70 | 1 |
| | **trueavg** | **1.5** | — | **0** |

The ordering is the same in the ER arm and identical for `ema+trueavg` (which is the ema here). Read
it as: **the ema produces a large transient and then removes it; the exact mean produces a smaller,
persistent level shift; `cumulative` sits between the two and decays slowly because its lag shrinks
as 1/n.** For `vecproj` + `trueavg` not one of the four boundaries produces a jump worth measuring.

This matters for how a novelty driver is described. "Novelty spikes at a task boundary" is true of the
ema form, but what spikes is the *residual of a lagging estimator*: within a few hundred steps the
new digits read as normal again, because the reference has moved onto them. An absolute reference
cannot do that, so under it a task's novelty is a level, not an event. The two are different
mechanisms wearing the same formula, which is why they had to be measured rather than assumed
interchangeable.

## (2) Standardising the VECTOR form destroys the reference distinction; the NORM form keeps it

Standardised batch-mean level at test:

| series | ema | cumulative | trueavg | ema+trueavg | reference |
|---|---|---|---|---|---|
| `vecproj` **vector**, naive | 6.22 | 6.23 | 6.22 | 6.07 | √K = **5.66** |
| `vecproj` **vector**, er | 5.76 | 5.76 | 5.76 | 5.72 | √K = **5.66** |
| `vecproj` **norm**, naive | 0.515 | 0.210 | 0.218 | 0.377 | 0 |
| `vecproj` **norm**, er | 0.159 | 0.048 | 0.084 | 0.126 | 0 |
| `vec_x` **norm**, naive | 0.743 | 0.289 | 0.300 | 0.538 | 0 |
| `vec_x` **norm**, er | 0.236 | 0.090 | 0.139 | 0.194 | 0 |

The `vecproj` vector row is the measured form of a trap this project had recorded but never plotted:
standardising per DIMENSION and then taking the norm concentrates the result at √K, and it does —
6.22/5.76 against 5.66, with the four reference means **agreeing to within 0.01–0.15**. The
standardisation has removed the very difference the mean_mode axis is varying. The within-batch sd is
~1.0, so per-sample information is still present, but it rides a DC offset of 5.7: signal-to-offset
≈ 0.18.

The norm rows are a genuine z-score — level 0.05–0.74 on a within-batch sd of ~1.0 — and there the
four reference means separate cleanly (ema is 2–5× the level of trueavg). **`norm=True` is what makes
the reference choice observable at all.** It is also the only form that keeps the reference choice
observable *and* leaves the driver well-conditioned; see (4).

## (3) `ema+trueavg` sits between the two pure modes, which is the train/inference mismatch made visible

In **8 of 8** well-conditioned test cells the `ema+trueavg` standardised level falls strictly between
`ema` and `trueavg` (e.g. `vec_x` norm, naive: 0.743 → **0.538** → 0.300; er: 0.236 → **0.194** →
0.139). The cause is structural: its running mean and variance were fitted during training under the
**ema** reference, while its values at inference come from the **exact** one, so it is a value from
one distribution being z-scored by another's statistics. The offset is exactly the size of the
mismatch.

This is the trace-side reading of the accuracy result that `ema+trueavg` buys nothing: it is not a
neutral upgrade to an exact reference, it is a driver evaluated under statistics that were never
fitted to it. (In the *standard* regime the accuracy grid found it numerically identical to `ema` —
|Δ| ≤ 0.001, twice exactly 0.0000 — for the complementary reason: a stationary single-task
distribution makes the converged ema equal to the exact mean, so there is nothing to switch to.)

## (4) The `vec_x` vector form is ill-conditioned with no gate anywhere near it

Standardised `vec_x` vector at test, batch-mean level: **728 (er) to 2,790 (naive)**, max **2.8e4 to
4.9e4** — against √784 = 28, which is where a well-conditioned 784-d driver would sit. So the
concentration story of (2) does not even apply: the value is 25–100× above √K, and it is the ~9–17%
of MNIST border-pixel dimensions with (near-)zero running variance that put it there. `vecproj`
(which mixes all 784 pixels into every one of its 32 coordinates) and both norm forms (0.05–0.74)
are unaffected.

**This is the pathology measured in the observer, with no modulation applied at all** — which is what
makes the accuracy result in `novelty_drivers.md` a consequence rather than a coincidence: the same
configuration, once its gate is in the forward, collapses to chance. A driver whose standardised
value is four orders of magnitude off its expected scale was never going to survive being multiplied
into a network.

---

## Limits

- 1 seed, one arm pair, class-IL only; observer-only, so no accuracy claim is made or implied here.
- Input space only; the h1 kinds keep their existing panels in `live_traces.md`.
- Statistics are FROZEN at test (the frozen-study convention). The live-update protocol is
  `live_traces.py`'s premise and is deliberately not re-opened — note however that a frozen `ema` at
  inference is *not* an ema but a snapshot of the last training batches, i.e. a reference fitted to
  the last task. `novelty_drivers.md` §2 measures how far that is from the test distribution (5.61,
  against 0.71 for the exact mean).
- The `recovery` column is a ratio with the boundary jump in its denominator and is reported only
  where the jump clears 3× the typical step; `n` says how many of the four boundaries qualified. An
  earlier unguarded version produced readings like −77.9 from near-zero jumps.
