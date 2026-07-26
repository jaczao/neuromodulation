# pt7 driver traces — what each neuromodulator formula actually *is*, over training and at test

Observer-only study. A **plain** MLP is trained under two continual-learning regimes and every pt7
single-neuromodulator formula is evaluated on each batch as a passive read-out. **No modulation is applied
anywhere**: there is no gate, no rank-K projection, no head. Nothing the observer computes touches the loss,
the parameters, or the RNG — so both runs reproduce the frozen pt7 baselines, and that equality is the
sanity anchor for the whole study.

Files: `pt7_driver_traces.py` (study), `pt7_driver_traces.npz` (all traces),
`pt7_driver_traces/*.png` (one figure per driver + a contact sheet), `pt7_driver_traces.log`.

## Setup

| | |
|---|---|
| Task | Split MNIST, 5 tasks × 2 classes, class-IL |
| Regime A | **naive + masked loss** — per-sample masked CE, current task only, no buffer |
| Regime B | **ER** — plain 10-way CE on `cat([current, replay])`, reservoir buffer 1000, **no masked loss** |
| Optimizer | Adam (`--opt sgd` for the SGD point) |
| Operating point | **val-tuned**, `--point tuned` — lr/epochs read per arm from `configs.TUNED_MAIN[("classil", arm, "adam")]` = **lr 3e-4, 5 ep/task** for both arms |
| Other | seed 42, batch 64, buffer 1000, 1 seed |
| Plotted quantity | batch mean of the per-sample driver value; **multidimensional drivers are reduced to the per-sample L2 norm first** |

**Sanity anchor** — the observer must not perturb training, and it doesn't:

| arm | this run | `pt7_tuned_syn` reference | delta |
|---|---|---|---|
| naive + masked loss | 0.5514 | 0.5545 | −0.0031 |
| ER | 0.8988 | 0.8975 | +0.0013 |

Both inside the documented ~0.007–0.016 MPS 1-seed noise floor.

**The operating point is not cosmetic for the naive arm.** The untuned pt7 point (lr 1e-3) gives naive
0.3900; tuning *lowers* Adam's lr to 3e-4 and naive rises to 0.5545, because a smaller lr forgets less. Every
loss-based driver is a direct function of the loss trajectory, so their naive traces differ materially
between the two points. ER is nearly insensitive (0.8946 untuned vs 0.8975 tuned). `--point untuned`
reproduces the original pt7 point if you need the traces that pt7's own numbers sat on.

Each driver gets one figure with four panels: `{non-standardised, standardised} × {training, test}`, both
regimes overlaid. Panels switch to a symlog y-axis automatically when the dynamic range demands it (the
title says so) — several drivers span four decades and a linear axis would show a flat line plus a spike.

### Standardisation

"Standardised" is pt7's own running transform, reproduced once here so both versions come from a single
pass: running per-dimension mean/variance updated `0.99/0.01` on every training batch (initialised from the
first batch), then `(v − run_mean) / (sqrt(run_var) + 1e-6)`. At **test time the statistics are frozen** —
exactly as pt7 freezes them — so the test panels show each driver under the statistics training left behind.

### One faithfulness choice worth stating

The loss-based formulas (`DA`, `DA_step`, `DA_fast`, `ACh_vol*`, `5HT*`) contain an inner per-sample loss
`ℓ_i`. pt7's `Signals` computes that with the **masked** per-sample CE in *both* arms — including `er-own`,
where the network itself trains on plain 10-way CE. This study keeps pt7's convention by default so the
traces are the values pt7 actually fed its heads. `--driver-loss arm` recomputes `ℓ_i` with the arm's own
training loss (plain CE under ER) if you want the other reading; combine with `--suffix` to write it beside
the default.

## The formulas traced

`all4` and UNIFY-12 are composites and the `free` / `5ht-const` controls are not neuromodulator formulas at
all, so none of them appear here. `emb_all` from `pt7_variants` is the same formula as `NE_emb` and is not
duplicated.

| Driver | Formula | Family | Dim |
|---|---|---|---|
| `DA` | `(ℓ_i − ema_slow) / std(ℓ)` | DA | 1 |
| `DA_step` | `(ℓ_i − L_{t−1}) / std(ℓ)` | DA | 1 |
| `DA_fast` | `(ℓ_i − ema_fast) / abs(ema_fast)` | DA | 1 |
| `ACh` | `H(logits_i)` | ACh | 1 |
| `ACh_ema` | `ema(H)` — **tonic** | ACh | 1 |
| `ACh_vol` | `sqrt(ema((L − ema_fast)²))` — **tonic** | ACh | 1 |
| `ACh_vol_ps` | `abs(ℓ_i − ema_fast)` | ACh | 1 |
| `NE` | `relu((abs(DA_i) − ACh_vol) / ACh_vol)` | NE | 1 |
| `NE_rise` | `max(ema_fast − ema_slow, 0)` — **tonic** | NE | 1 |
| `NE_emb` | `‖h1_i − mean_h1‖` | NE | 1 |
| `nerisez` | `relu((H_i − ema_H) / sqrt(var_H))` | NE | 1 |
| `5HT` | `−ℓ_i` | 5-HT | 1 |
| `5HT_ema` | `−ema_slow` — **tonic** | 5-HT | 1 |
| `vec_h1` | `‖h1_i − mean_h1‖` | NE | 400 |
| `vec_h1proj` | `‖R(h1_i − mean_h1)‖` | NE | 32 |
| `vec_x` | `‖x_i − mean_x‖` | NE | 784 |
| `vecproj` | `‖R(x_i − mean_x)‖` | NE | 32 |

`nerisez` is traced on the **actual** entropy. `pt7_stateful` predicts `H` with an MLP/GRU head; the formula
is identical and the predictor is a separate mechanism, out of scope for a formula trace.

## Results

Ranges below are over the whole training run, batch means, tuned Adam. `spread` = max|v| / median|v|, a
scale-free measure of how far a driver's extremes sit from its typical value.

### 1. The drivers split into three scale classes, and the split predicts every pt7 outcome

| class | drivers | raw range | standardised range |
|---|---|---|---|
| **benign** | `NE_emb`, `vec_h1`, `vec_h1proj`, `vec_x`, `vecproj`, `ACh` | `NE_emb` 15–35 (spread **1.7**) | ±2 |
| **spiky** | `DA`, `DA_step`, `DA_fast`, `NE`, `ACh_vol_ps`, `5HT` | `DA` ±3e3, `DA_step` ±1.8e4, `NE` up to **1.2e6** | ±10² |
| **tonic** | `ACh_ema`, `ACh_vol`, `NE_rise`, `5HT_ema` | 0.02–2.3 (small, smooth) | **±1e6** |

This is the mechanism behind two rules in the gotchas that were previously stated as empirical outcomes:

- **"standardize or the gate blows up"** — `NE` raw reaches **1.2e6**. `NE = relu((|DA| − ACh_vol)/ACh_vol)`
  divides by `ACh_vol`, whose raw mean is 0.016 and whose **minimum is exactly 0**. Standardisation pulls it
  back to ±10². A K=4 unbounded gate fed the raw mixture cannot survive this, which is exactly the observed
  `all4 std0` → NaN → 0.0980 collapse.
- **"standardize per-sample drivers, NEVER a tonic one"** — the four tonic drivers are the *best-behaved* raw
  signals (smooth, range ≈ 0–2.3) and become the *worst* standardised ones (±1e6). Their per-batch variance
  is near zero, so dividing by `sqrt(run_var)` divides by ≈0. `ACh_ema.png` shows this in one image: a clean
  decaying sawtooth raw, a ±1e6 mess standardised.

`DA`'s explosion has a separate cause worth noting: its denominator is `std(ℓ)` computed **within the
batch**. Once a batch's per-sample losses become near-uniform — which is the normal end-state of training —
that divisor collapses toward the 1e-6 epsilon. `DA` is therefore least stable exactly when the model is
doing well, which is the opposite of what a reward-prediction-error signal should do.

### 2. Test time is where the tonic drivers become pathological

At test the running statistics are frozen, so a tonic driver — constant within a batch by construction —
becomes a **flat line at a huge constant offset**: `ACh_ema` standardised sits at −16 000 (naive) and −3 600
(ER) across every test batch. A per-sample gate driven by a constant carries zero per-sample information,
which is precisely the "a signal with no per-sample content cannot drive a per-sample gate" finding.

The spiky drivers also degrade from train to test: `DA` in the ER arm goes from mean −5.7 during training to
mean **−736** at test, because `ema_slow` is frozen at its training value while test batches on well-learned
tasks have near-zero loss and near-zero spread.

### 3. Only the novelty drivers stay well-conditioned everywhere

`NE_emb` / `vec_*` are the only family that is stable in all four panels: raw 15–35, standardised ±2, on both
train and test, with no dependence on frozen loss statistics. They are also the drivers behind pt7's largest
observed effect (the `vec_h1`/`vec_x` +0.14 SGD+ER boost, later shown to be an SGD-underfit artifact). Their
conditioning is a plausible part of why they were the ones that did anything at all.

Note `vecproj` standardised sits at ≈5.6, not 0 — standardising a 32-d vector per dimension and *then*
taking the norm concentrates it at `sqrt(32) ≈ 5.66`. That is expected, not an offset bug.

### 4. The two regimes separate cleanly, and mostly through entropy

`ACh` (entropy) raw: **naive 0.50 ± 0.32 vs ER 0.056 ± 0.17** — an order of magnitude apart. The ER model is
far more confident, because it trains on the real 10-way problem with replay, while naive+masked-loss is only
ever asked to separate 2 classes and stays diffuse over the other 8. The task-boundary sawtooth in
`ACh_ema.png` is much larger in the naive arm and decays to ≈0 in the ER arm — visual confirmation that
replay is what keeps the shared head calibrated.

### Caveat that limits reading (2) and (3) for the ER arm

The loss-based drivers in the ER arm are computed with **masked** CE (pt7's convention), while that arm's
network trains on plain 10-way CE. So ER's `DA`/`5HT`/`ACh_vol*` traces are read under a loss the network is
not optimising, and their small magnitudes partly reflect that. The `--driver-loss arm` variant recomputes
them with plain CE; see `pt7_driver_traces_armloss/`.

## Reproduce

```
uv run python results/pt7_driver_traces.py --opt adam --point tuned   # the study above
uv run python results/pt7_driver_traces.py --point untuned            # pt7's inherited lr 1e-3 point
uv run python results/pt7_driver_traces.py --driver-loss arm --suffix _armloss   # plain-CE variant
uv run python results/pt7_driver_traces.py --plot-only                # re-plot from the npz
```
