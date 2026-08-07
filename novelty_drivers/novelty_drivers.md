# Novelty-driver form: the difference VECTOR vs its NORM, and what "normal" is measured against

**Verdict: REJECT everywhere — but the two axes fail in different ways, and only one of them is a
null.** The NORM form is inert: `d-dead` within ±0.008 in **all 32** norm-form cells, across both
regimes, both projections and all four reference means. The VECTOR form is *harmful*, and the harm
grows with the driver's width: null at K=32 with a learned projection, −0.06 at K=32 frozen,
−0.03/−0.32 at K=784 (learned/frozen), and **−0.79 (chance) at K=784 once the reference mean is
exact**. Nothing beats its RNG-matched control; the class-IL headline is unchanged.

The reusable result is the *mechanism* of that collapse. It converts a diagnostic-only warning this
project already recorded into a measured accuracy failure, and — the part that was not predicted —
the reference that triggers it is the one that is demonstrably **better matched to the test
distribution**. What breaks is the standardisation, not the centring.

- study: `novelty_drivers/novelty_drivers.py` · ledger `novelty_drivers_results.tsv` (82 rows)
- observer-only traces of the same drivers: `driver_traces/meanmode_traces.py` / `.md`
- core changes: `neurocore/signals.py` (`NEDriver(norm=…, mean_mode=…)`, `dataset_mean`),
  `neurocore/utils.py` (`rng_frozen` promoted), `neurocore/tuned.py` (standard-regime point)

---

## What was varied

| axis | values |
|---|---|
| `kind` | `vec_x` = `x − reference` (784-d) · `vecproj` = `R(x − reference)`, R random 784→32 |
| `norm` | 0 = the difference VECTOR is the driver (K = its width) · 1 = its L2 NORM (K = 1) |
| `mean_mode` | `ema` · `cumulative` · `trueavg` · `ema+trueavg` |
| `proj` | `learned` (P trained by the main loss, zero-init) · `random` (P frozen at N(0, 0.1²)) · `dead` (P ≡ 0) |
| regime | class-IL er-own (buffer 1000) · standard (full MNIST, single task) |

Gain modulation, per-neuron gate over (h0, h1, out), Adam, standardised drivers (the norm is taken
**before** standardising). **No new tuning** — class-IL er-own at lr 3e-4 / 5 ep/task (the
`pt7_tuned_syn` val-selected ER point), standard at lr 1e-3 with ≤6 epochs early-stopped on the
held-out val split (`pt7_std_tuned`'s protocol, recorded in `neurocore.tuned` as the partial tune it
is — that study swept epochs but not lr). 1 seed; the 1-seed MPS noise floor is 0.007.

### Reference-mean policy (needed to read anything below)

- **training** — `trueavg` installs the exact mean of the training images of every task seen so far,
  recomputed at each task boundary: the causal, exact counterpart of `cumulative`, which builds the
  same quantity online and therefore lags it. `ema+trueavg` uses the ema here.
- **inference** — both true-mean modes install the exact mean of the evaluation stream (the pooled
  test set in class-IL; the val set for val passes and the test set for test passes in standard, so
  val never sees test). Label-free, one pass, computable on the way through — deployable, not an
  oracle.

---

## The three checks this study had to pass about itself

| check | result |
|---|---|
| class-IL plain ER vs the frozen `pt7_tuned_syn` ledger | **0.897549 vs 0.8975 — bit-exact** |
| `dead` gate vs plain, all 8 shapes × both regimes | **spread 0.000000, d-plain +0.000000** |
| `dead` under `trueavg` vs `dead` under `ema` | **bit-identical** |
| standard plain vs `pt7_std_tuned` vanilla (band, not anchor) | 0.9766 vs 0.9802 (−0.0036) |

Rows 2 and 3 are the load-bearing ones. Nothing in the modulator draws from the *global* torch RNG
(`NEDriver`'s projection and the frozen P both come from private generators; a zero-init P is
`torch.zeros`), so `dead` was *predicted* to equal the plain baseline exactly — and does, at every K
from 1 to 784. And because `trueavg` adds a full pass over the images, and **iterating a DataLoader
draws from the global torch RNG even with `shuffle=False`**, that pass had to be RNG-neutral or the
true-mean cells would have trained on a different data order than the `ema` cells and the whole axis
would have been confounded. `dataset_mean` wraps itself in `rng_frozen()` for that reason, and the
dead-gate `trueavg` == dead-gate `ema` equality is the check that it worked: **a `mean_mode` cell
differs from its `ema` twin only in the reference vector.**

The standard number is deliberately *not* an anchor: `pt7_std_tuned` evaluates val and test inside
its training loop without an RNG guard, and its test pass fires only on a val improvement — so its
RNG draw depends on its own accuracy trajectory. This study guards every eval, which is correct and
necessarily lands on a slightly different trajectory.

---

## Class-IL (er-own, buffer 1000). Dead gate = plain ER = 0.8975

`d-dead`, learned / frozen-random projection:

| kind | K | extra params | ema | cumulative | trueavg | ema+trueavg |
|---|---|---|---|---|---|---|
| `vec_x` **norm** | 1 | 810 (0.0017×) | −0.001 / +0.003 | +0.001 / +0.004 | +0.004 / +0.002 | −0.001 / +0.003 |
| `vecproj` **norm** | 1 | 810 (0.0017×) | −0.006 / +0.003 | +0.008 / +0.001 | +0.000 / −0.002 | −0.007 / +0.003 |
| `vecproj` vector | 32 | 25,920 (0.054×) | −0.002 / −0.057 | +0.002 / −0.064 | +0.001 / −0.056 | −0.009 / −0.054 |
| `vec_x` vector | 784 | 635,040 (**1.33×**) | −0.062 / −0.322 | −0.053 / −0.344 | **−0.775 / −0.797** | **−0.798 / −0.774** |

## Standard regime (full MNIST). Dead gate = plain = 0.9766

| kind | ema | cumulative | trueavg | ema+trueavg |
|---|---|---|---|---|
| `vec_x` **norm** | −0.004 / −0.003 | −0.002 / −0.003 | +0.000 / +0.001 | −0.003 / −0.003 |
| `vecproj` **norm** | +0.001 / +0.002 | +0.003 / −0.001 | +0.002 / −0.001 | +0.001 / +0.002 |
| `vecproj` vector | −0.001 / −0.020 | +0.003 / −0.018 | −0.002 / −0.020 | −0.001 / −0.019 |
| `vec_x` vector | −0.028 / −0.315 | −0.036 / −0.335 | −0.037 / **−0.794** | −0.028 / **−0.788** |

---

## (1) The norm form is a clean null; the vector form is not, and it is not about semantics

All 32 norm-form cells land on their dead control (min −0.0069, max +0.0076, every one inside the
noise floor) in **both** regimes and under **both** projections. So the honest reading of the norm
axis is: reducing the driver to "how unusual" costs nothing and gains nothing — the project's
standing gating null, reproduced on a new axis, in its cheapest possible form (810 parameters,
0.0017× the backbone).

The vector form is where the damage is, and under a *frozen* projection it is monotone in K:
K=32 → −0.06, K=784 → −0.32 (class-IL; standard agrees, −0.02 → −0.32). That ordering rules out the
appealing story — "the vector says *where* a sample is unusual, so it carries more information".
More directions make it strictly worse in the regime where the backbone cannot adapt the gate away.
With a *learned* P the network claws most of it back (vecproj vector → null; vec_x vector → −0.03 to
−0.06) by shrinking P, so the learned projection's contribution here is damage control, not signal
extraction.

**The most task-decodable driver is the worst, again.** `vec_x` vector probes at **0.933** — the
highest task-decodability measured anywhere in this project, above pt6's oracle-free selector
(0.884) — and is the only cell that reaches chance. `vecproj` vector probes 0.685 and is mildly
harmful; the norm forms probe 0.19–0.25 and are null. Across the table, probe and benefit are
*inversely* ordered. Third independent re-derivation (after `pt5_taskil/plast_drivers` on a
plasticity target and `fixedproj_scale` on a forward target) that **"the driver doesn't know the
task" was never the binding constraint**.

## (2) The headline: an EXACT reference tips vec_x's ill-conditioned vector form into a chance-level collapse — and it is the standardisation that breaks, not the centring

`vec_x` + vector form + exact reference = **0.1223 / 0.1009 / 0.1000 / 0.1236** in class-IL and
**0.1824 / 0.1883** in standard with a frozen P — where the *same driver* under an `ema` reference
reaches 0.84 and 0.66. Same code, same data order, same gate; one changed vector.

**The precondition is already on record.** `pt5_taskil/plast_drivers.py` measured that **212 of
`vec_x`'s 784 input dimensions are MNIST border pixels with `run_var == 0`**, so standardising the
vector divides by ~eps there (|m| up to 2e6 at test). That study used a *plasticity* target — the
gate multiplies gradients and never enters the forward — so the blow-up was visible and inert, and
CLAUDE.md recorded the prediction verbatim: *"a forward-gain target would have collapsed to chance."*
**This is that experiment, and the prediction is confirmed exactly.**

Three things measured here that go beyond it:

- **The ill-conditioning is intrinsic, not caused by the reference.** Re-measured directly on the
  standard stream: 8.7% of the 784 running variances end below 1e-8 (some exactly 0), and the
  standardised driver has a **max |m| of 3.055e5 at test under *every* reference**, including no
  swap at all. Every mode is riding a 1e5-scale tail.
- **The exact reference amplifies the magnitude, and that is what tips it.** At class-IL eval the
  mean |m| goes **1.64 → 2.88** (×1.75) and |g| follows to 16–90 (from 8.6–46). The tail was always
  there; the exact reference raises the whole distribution onto it.
- **It is NOT a distribution-mismatch effect — measured, the exact reference is 8× *better* matched.**
  A frozen `ema` at inference is not an ema at all: with `update=False` it is a snapshot of the last
  training batches, i.e. **a reference fitted to the LAST TASK**. Its distance to the pooled test
  mean is **5.61**, against **0.71** for the exact mean. So the reference that is 8× closer to the
  evaluation distribution is the one that destroys the network. The failure lives entirely in the
  frozen `run_mean`/`run_var` of the near-constant dimensions, not in where the driver is centred.

`vecproj` escapes completely (a random projection mixes all 784 pixels, so all 32 coordinates have
real variance) and the norm form escapes completely (‖x − x̄‖ has real variance whatever individual
pixels do) — which is exactly why neither moves under the same switch.

**Rule (extends the existing standardisation rules): standardise per-DIMENSION only when every
dimension has real variance. If any can be constant, reduce first — project it or take the norm —
and standardise the reduced scalar. And do not read a streaming reference as "safe": it was hiding a
numerical defect, not avoiding one, and a better-matched reference exposes it.**

## (3) With the numerics sound, the reference mean does not matter at all

Restricted to the well-conditioned cells (both norm forms and `vecproj` vector), the four reference
means are indistinguishable. Spread over `{ema, cumulative, trueavg, ema+trueavg}` at fixed
(kind, norm, proj): **0.0024–0.0145** in class-IL and **0.0016–0.0044** in standard, against a
0.007 floor, with no consistent ordering (the best mode is `cumulative` in 3 of 6 class-IL groups
and 3 of 6 standard groups; the worst is a different mode in almost every group).

- **`trueavg` vs `ema`: +0.0065 at best, −0.0047 at worst**, positive in 8 of 12 well-conditioned
  groups. Directionally mildly favourable, entirely inside the floor, not claimed.
- **`ema+trueavg` is, in the standard regime, numerically the same run as `ema`** — |Δ| ≤ 0.0010 in
  all six groups and **exactly 0.0000 in two**. That is structural rather than lucky: with a
  stationary single-task distribution the converged ema already *is* the exact mean, so the mode has
  nothing to switch to. It can only act where the training distribution is non-stationary — i.e. in
  class-IL — and there it is still inside the floor (−0.0071 to +0.0031). **A train/inference
  reference swap is not a free upgrade; on this problem it buys nothing at all in the standard
  regime by construction.**

The traces (`driver_traces/meanmode_traces.md`) show why the accuracy is flat while the *signal*
plainly is not: the reference changes the driver's level and its whole behaviour across task
boundaries, and the gate is absorbed either way — the conclusion `fixedproj_scale` reached from the
magnitude end of the same axis.

## (4) Cost, and the capacity flag doing its job

`neurocore.cost` flags `vec_x` vector at **1.33× the backbone** (635,040 extra parameters against
478,410) — the one configuration whose positive result could not have been trusted. It is instead
the worst cell in the table, so the flag costs nothing here; the contrast is what matters, since the
**cheapest** form (810 parameters, 0.0017×) is exactly as good as the most expensive one in both
regimes. These drivers are also free at inference — `vec_x` and `vecproj` are computed **pre-forward**
from the input alone, so `fwd_infer = 1`, `bwd_infer = 0`, no extra pass (unlike every loss- or
entropy-based driver). `trueavg` adds one amortised pass over the data per reference update, which is
not a per-step cost.

## (5) Standard regime: goal #2 is preserved wherever the driver is well-conditioned

Every norm-form cell is within ±0.004 of vanilla and `vecproj` vector-learned within ±0.003, so the
gate is harmless for plain single-task accuracy — the same answer pt4 and pt7's standard runs gave
for every other mechanism. The exceptions are exactly the ill-conditioned ones (`vec_x` vector:
−0.03 learned, −0.32/−0.79 frozen), which is a numerical failure, not a statement about gain control.

One diagnostic caveat visible in the log: the standard-regime cells report accuracy at the best-val
epoch, and for `vec_x` vector + frozen P + exact reference **val reads 0.58 while test reads 0.18**.
Val and test differ only in which exact mean is installed, so that gap is the same conditioning
failure measured across two evaluation sets rather than a selection artefact.

---

## Limits

- **1 seed** across a 64-cell grid. Every conclusion above is either far outside the 0.007 floor
  (the vector-form damage, the collapse) or is explicitly reported as a null *inside* it (the whole
  norm-form block, the reference-mean axis). The two sub-floor positives (+0.008) are not claimed.
- **Adam only, per-neuron only, one arm per regime** (er-own; no `nobuf` / `buf-own`). Per-synapse is
  not runnable for `vec_x` (K=784 × 400×784) and was not run for `vecproj`.
- **Input space only.** An h1 reference mean is exact only w.r.t. the weights at the moment it was
  computed, so `trueavg` in embedding space is a materially different object; `NEDriver` supports it
  (`dataset_mean(..., space="h1")`) but nothing here measures it.
- **Not retuned**, per the request. The standard point is a partial tune (epochs val-selected, lr not
  swept) and is recorded as such in `neurocore.tuned`.
- The standard-regime `|m|`/`|g|` diagnostics are read under the *training* reference (the loop
  restores it at the end of each epoch), so unlike the class-IL ones they do not show the eval-time
  magnitude. The class-IL diagnostics and the direct re-measurement in §2 are what the mechanism
  claim rests on.
- Three memory regimes (rule #12) are not swept: this is an ablation *inside* an existing arm, not a
  new direction, so it inherits `normal`. Cost columns are recorded per cell regardless.
