# Novelty-driver form: the difference VECTOR vs its NORM, the reference mean, and standardisation

**Verdict: REJECT everywhere. The NORM form is a null, the VECTOR form is harmful, and the
chance-level collapse the first pass found is ENTIRELY a standardisation artefact — measured, not
inferred.** Running the identical grid raw takes `vec_x`'s vector form from **0.1223 back to 0.8693**
in class-IL and **0.1824 to 0.7775** in standard. What survives removing standardisation is the
vector form's *ordinary* harm (−0.03 to −0.29), which is the real finding; what does not survive is
the catastrophe, which was division by the near-zero running variance of MNIST's constant border
pixels and nothing else.

Nothing in either arm beats its RNG-matched control. The class-IL headline is unchanged.

- study: `novelty_drivers/novelty_drivers.py` · ledger `novelty_drivers_results.tsv` (162 rows)
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
| `std` | 1 = per-dimension running standardisation · 0 = the raw driver |
| regime | class-IL er-own (buffer 1000) · standard (full MNIST, single task) |

Gain modulation, per-neuron gate over (h0, h1, out), Adam. The norm, when taken, is always taken
**before** standardising. **No new tuning** — class-IL er-own at lr 3e-4 / 5 ep/task (the
`pt7_tuned_syn` val-selected ER point), standard at lr 1e-3 with ≤6 epochs early-stopped on the
held-out val split (`pt7_std_tuned`'s protocol, recorded in `neurocore.tuned` as the partial tune it
is — that study swept epochs but not lr). 1 seed; the 1-seed MPS noise floor is 0.007.

`std` is a **key column**, not a constant. The first pass ran std=1 throughout, per CLAUDE.md's
"standardize per-sample drivers"; the raw arm was added because (a) the recent single-driver work
runs raw by direction (`position_paper/drivers.py`'s `STANDARDIZE` table is all-`False`;
`fixedproj_scale` likewise) and (b) the std=1 headline was a *conditioning* failure, so the arm
without any division is the control that claim needs. The std=1 rows were migrated in place with the
new column — a faithful annotation, since `build_driver` hardcoded `standardize=True` for all of them.

### Reference-mean policy (needed to read anything below)

- **training** — `trueavg` installs the exact mean of the training images of every task seen so far,
  recomputed at each task boundary: the causal, exact counterpart of `cumulative`, which builds the
  same quantity online and therefore lags it. `ema+trueavg` uses the ema here.
- **inference** — both true-mean modes install the exact mean of the evaluation stream (the pooled
  test set in class-IL; the val set for val passes and the test set for test passes in standard, so
  val never sees test). Label-free, one pass, computable on the way through — deployable, not an
  oracle.

---

## The checks this study had to pass about itself

| check | result |
|---|---|
| class-IL plain ER vs the frozen `pt7_tuned_syn` ledger | **0.897549 vs 0.8975 — bit-exact** |
| `dead` gate vs plain — 16 cells, both shapes × both std arms × both regimes | **spread 0.000000, d-plain +0.000000** |
| `dead` under `trueavg` vs `dead` under `ema` | **bit-identical** |
| standard plain vs `pt7_std_tuned` vanilla (band, not anchor) | 0.9766 vs 0.9802 (−0.0036) |

Nothing in the modulator draws from the *global* torch RNG (`NEDriver`'s projection and the frozen P
both come from private generators; a zero-init P is `torch.zeros`), so `dead` was *predicted* to
equal the plain baseline exactly — and does, at every K from 1 to 784 and under both standardisation
conventions. And because `trueavg` adds a full pass over the images, and **iterating a DataLoader
draws from the global torch RNG even with `shuffle=False`**, that pass had to be RNG-neutral or the
true-mean cells would have trained on a different data order than the `ema` cells and the whole axis
would have been confounded. `dataset_mean` wraps itself in `rng_frozen()` for that reason, and the
dead-gate `trueavg` == dead-gate `ema` equality is the check that it worked.

The standard number is deliberately *not* an anchor: `pt7_std_tuned` evaluates val and test inside
its training loop without an RNG guard, and its test pass fires only on a val improvement — so its
RNG draw depends on its own accuracy trajectory. This study guards every eval.

---

## Class-IL (er-own, buffer 1000). Dead gate = plain ER = 0.8975

`d-dead`, raw (std=0) / standardised (std=1), learned projection:

| kind | K | extra params | ema | cumulative | trueavg | ema+trueavg |
|---|---|---|---|---|---|---|
| `vec_x` **norm** | 1 | 810 | −0.009 / −0.001 | −0.004 / +0.001 | −0.005 / +0.004 | −0.009 / −0.001 |
| `vecproj` **norm** | 1 | 810 | −0.002 / −0.006 | −0.008 / +0.008 | +0.003 / +0.000 | −0.001 / −0.007 |
| `vecproj` vector | 32 | 25,920 | −0.003 / −0.002 | +0.002 / +0.002 | −0.000 / +0.001 | −0.008 / −0.009 |
| `vec_x` vector | 784 | 635,040 | −0.050 / −0.062 | −0.034 / −0.053 | **−0.028 / −0.775** | **−0.107 / −0.798** |

and with the frozen-random projection:

| kind | ema | cumulative | trueavg | ema+trueavg |
|---|---|---|---|---|
| `vec_x` **norm** | −0.021 / +0.003 | −0.013 / +0.004 | −0.012 / +0.002 | −0.022 / +0.003 |
| `vecproj` **norm** | −0.001 / +0.003 | +0.001 / +0.001 | +0.001 / −0.002 | −0.001 / +0.003 |
| `vecproj` vector | −0.032 / −0.057 | −0.029 / −0.064 | −0.026 / −0.056 | −0.028 / −0.054 |
| `vec_x` vector | −0.267 / −0.322 | −0.268 / −0.344 | **−0.279 / −0.797** | **−0.287 / −0.774** |

## Standard regime (full MNIST). Dead gate = plain = 0.9766

| kind | proj | ema | cumulative | trueavg | ema+trueavg |
|---|---|---|---|---|---|
| `vec_x` **norm** | learned | +0.000 / −0.004 | −0.000 / −0.002 | −0.005 / +0.000 | +0.000 / −0.003 |
| `vecproj` **norm** | learned | +0.005 / +0.001 | +0.002 / +0.003 | +0.002 / +0.002 | +0.005 / +0.001 |
| `vecproj` vector | learned | +0.002 / −0.001 | +0.001 / +0.003 | −0.004 / −0.002 | +0.001 / −0.001 |
| `vec_x` vector | learned | −0.018 / −0.028 | −0.011 / −0.036 | −0.012 / −0.037 | −0.018 / −0.028 |
| `vec_x` vector | random | −0.199 / −0.315 | −0.197 / −0.335 | **−0.199 / −0.794** | **−0.201 / −0.788** |

---

## (1) The collapse was standardisation, and only standardisation

The four class-IL cells that reached chance under std=1 all recover under std=0:

| cell | std=1 | std=0 |
|---|---|---|
| `vec_x` vector, trueavg, learned | 0.1223 | **0.8693** |
| `vec_x` vector, trueavg, random | 0.1009 | **0.6187** |
| `vec_x` vector, ema+trueavg, learned | 0.1000 | **0.7909** |
| `vec_x` vector, ema+trueavg, random | 0.1236 | **0.6102** |
| standard, trueavg, random | 0.1824 | **0.7775** |
| standard, ema+trueavg, random | 0.1883 | **0.7753** |

So the mechanism argued in the first pass is now *measured*: **212 of `vec_x`'s 784 input dimensions
are MNIST border pixels with (near-)zero running variance** (re-measured here: 8.7% of the running
variances end below 1e-8, some exactly 0), and standardising divides by them. With no division there
is no catastrophe. This also completes the prediction CLAUDE.md carried from
`pt5_taskil/plast_drivers.py` — *"a forward-gain target would have collapsed to chance"* — with its
converse: remove the standardisation and the same forward-gain target does not.

Worth keeping in view: **the exact reference is not what "caused" it.** Measured, a frozen `ema` at
inference is not an ema at all but a snapshot of the last training batches — a reference fitted to
the **last task**, at distance 5.61 from the pooled test mean, against 0.71 for the exact mean. The
reference 8× *better* matched to the evaluation distribution is the one that detonated the
ill-conditioned driver. The exact mean raises the whole distribution onto a 1e5-scale tail that was
present under every reference (max |m| = 3.055e5 at test in all of them); the ema was masking a
numerical defect, not avoiding one.

**Rule: standardise per-DIMENSION only when every dimension has real variance. If any can be
constant, reduce first — project it or take the norm — and standardise the reduced scalar. Do not
read a streaming reference as "safe"; and when a driver collapses, run the raw arm before attributing
the collapse to the driver.**

## (2) What survives: the vector form is harmful under BOTH conventions

Removing standardisation does not rescue the vector form, it only removes the catastrophe.
`vec_x` vector is outside the noise floor in **8 of 8** raw class-IL cells and 8 of 8 raw standard
cells (−0.028 to −0.287, −0.011 to −0.201). `vecproj` vector is outside it in 5 of 8 raw class-IL
cells, all negative. And under a frozen projection the harm is still monotone in K
(raw class-IL: K=32 → −0.03, K=784 → −0.27), which again refutes "the vector says *where* a sample is
unusual, so it carries more". More directions are strictly worse wherever the backbone cannot adapt
the gate away; with a learned P it claws most of it back by shrinking P, i.e. the learned projection
is doing damage control rather than signal extraction.

The **norm** form remains the null it was: at std=0, class-IL `vecproj` norm is inside the floor in 7
of 8 cells and standard is inside it in 8 of 8 for both kinds. The one systematic exception is
`vec_x` norm + **frozen** P (−0.021, −0.022), which is a scale effect and not a conditioning one:
the raw norm ‖x − x̄‖ has magnitude ≈ **23**, so a frozen P at σ = 0.1 produces |g| ≈ 1.9 against
std=1's 0.07. That is precisely the cost `position_paper/drivers.py` documents for running raw — the
usable gate scale becomes driver-specific — and it is visible here as ~2 points.

## (3) Which convention is better is decided by the driver's conditioning, and it splits cleanly

`std=1 − std=0`, averaged over the 8 mean_mode × proj cells:

| | class-IL | standard | direction |
|---|---|---|---|
| `vec_x` **vector** | **−0.326** (0/8 positive) | **−0.188** (0/8) | raw wins, decisively |
| `vecproj` **vector** | −0.014 (2/8) | −0.005 (2/8) | raw wins, mildly |
| `vec_x` **norm** | **+0.014** (8/8 positive) | +0.000 (4/8) | standardising wins, mildly |
| `vecproj` **norm** | +0.001 (3/8) | +0.000 (5/8) | tie |

Read as a single rule: **standardising helps only where the quantity being standardised is a
well-conditioned scalar of large raw magnitude** (`vec_x` norm, raw |m| ≈ 23 — the one row that is
8/8 positive, and the one with the largest raw scale to tame), **and hurts a vector with constant
dimensions, sometimes catastrophically.** `vecproj`, whose random projection gives every coordinate
real variance and a modest raw scale (|m| ≈ 0.69 vector, 4.8 norm), is indifferent either way.

## (4) The reference mean still does not matter, in either convention

Within the well-conditioned cells the four reference means stay indistinguishable at std=0 as they
were at std=1 — e.g. class-IL `vec_x` norm learned: 0.8889 / 0.8941 / 0.8929 / 0.8889, spread 0.005.
The one place a mean mode separates at std=0 is `vec_x` **vector** + learned P, where `ema+trueavg`
costs 0.06–0.08 relative to the other three (0.7909 vs 0.847–0.869, spread 0.078). That is the
train/inference reference swap paying a real price once it is not being swamped by a conditioning
failure — the same direction the std=1 arm and the traces both showed, now at a magnitude that
clears the floor. It does not appear under a frozen P (spread 0.020) or in either norm form.

`trueavg` vs `ema` remains inside the floor wherever both are sound (+0.0065 best, −0.0047 worst at
std=1; ±0.02 at std=0). And `ema+trueavg` is still numerically the same run as `ema` in the standard
regime — structurally, since a stationary distribution makes the converged ema equal the exact mean,
so a train/inference reference swap has nothing to switch to.

## (5) The probe ordering is unchanged, and unaffected by standardisation

`vec_x` vector probes **0.933** in both arms — the highest task-decodability measured anywhere in
this project, above pt6's oracle-free selector (0.884) — and is the worst driver in the table under
either convention. `vecproj` vector probes 0.685 and is mildly harmful; the norm forms probe
0.19–0.25 and are null. Probe and benefit remain *inversely* ordered. Third independent
re-derivation (after `plast_drivers` on a plasticity target and `fixedproj_scale` on a forward one)
that **"the driver doesn't know the task" was never the binding constraint**.

## (6) Cost, and the capacity flag doing its job

`neurocore.cost` flags `vec_x` vector at **1.33× the backbone** (635,040 extra parameters against
478,410) — the one configuration whose positive result could not have been trusted. It is instead
the worst cell under both conventions. The **cheapest** form (810 parameters, 0.0017×) is exactly as
good as the most expensive one in both regimes. These drivers are also free at inference — `vec_x`
and `vecproj` are computed **pre-forward** from the input alone, so `fwd_infer = 1`, `bwd_infer = 0`,
no extra pass (unlike every loss- or entropy-based driver). `trueavg` adds one amortised pass per
reference update, not a per-step cost.

## (7) Standard regime: goal #2 is preserved wherever the driver is well-conditioned

At std=0 every norm-form cell and every `vecproj` cell is within ±0.011 of vanilla (within ±0.005
with a learned P), so the gate is harmless for plain single-task accuracy — the same answer pt4 and
pt7's standard runs gave for every other mechanism. The exception is `vec_x` vector, in both arms.

---

## Limits

- **1 seed** across 128 grid cells. Conclusions are either far outside the 0.007 floor (the
  vector-form harm, the collapse and its recovery, the std1−std0 split) or explicitly reported as
  nulls inside it (the norm-form block, the reference-mean axis).
- **Adam only, per-neuron only, one arm per regime** (er-own; no `nobuf` / `buf-own`). Per-synapse is
  not runnable for `vec_x` (K=784 × 400×784) and was not run for `vecproj`.
- **Input space only.** An h1 reference mean is exact only w.r.t. the weights that produced it, so
  `trueavg` in embedding space is a materially different object; `NEDriver` supports it
  (`dataset_mean(..., space="h1")`) but nothing here measures it.
- **Not retuned**, per the request — and note that the raw arm arguably deserves its own operating
  point: a raw driver of magnitude 23 into an unbounded gate is a different scale regime, which is
  exactly the "usable range has to be measured per driver" cost `position_paper` records. The
  matched-budget comparison here is still valid, but the raw absolutes may be pessimistic.
- The standard-regime `|m|`/`|g|` diagnostics are read under the *training* reference (the loop
  restores it at the end of each epoch), so unlike the class-IL ones they do not show the eval-time
  magnitude.
- Three memory regimes (rule #12) are not swept: this is an ablation *inside* an existing arm, not a
  new direction, so it inherits `normal`. Cost columns are recorded per cell regardless.
- `driver_traces/meanmode_traces.md` traces both conventions but was produced before the raw
  accuracy arm; its conclusions are unaffected (it already plots raw and standardised side by side).
