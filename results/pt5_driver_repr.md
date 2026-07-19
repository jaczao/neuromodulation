# pt5 driver-representation study — is one-hot replaceable, and can we drop the eval oracle?

Companion to `pt5_driver_repr.py`. Question: the pt5 driver is a **one-hot task id** feeding a
projection `P` (so the gate is a per-task lookup row). What else could the driver be, and does any
alternative survive **without the task id at eval**? Gain-neuron, unbounded gain `g = 1+raw` on
`h0,h1`, class-IL Split MNIST, seed 42, lr 1e-3, ep 5, buffer 1000. **1 seed; ORACLE caveat carries
except where noted.** All numbers reproduce the pt5 harness (baselines match `naive-sgd 0.63 /
naive-adam 0.39 / er-sgd 0.72 / er-adam 0.90`).

## Drivers
- **onehot**: `raw_t = P[t]` — independent per-task rows (the pt5 default).
- **lin**: `raw_t = mu_t @ W` — shared linear map (784→800) over the task-mean image `mu_t`.
- **mlp**: `raw_t = gf(relu(gh(mu_t)))` — shared nonlinear map (784→128→800).
- **centered**: subtract `mean_t mu_t` (inter-task cosine of the mean images: **0.82 → −0.24**).

## 1. One-hot's edge is ORTHOGONALITY, not the lookup
One-hot gives each task a **private, independently-trained row** (the one-hot zeroes every other
row's gradient), so tasks differentiate for free. A **content** driver over the RAW mean images
collapses — the shared "average digit" component dominates, and a *learned* `W` amplifies it:

| driver (buf-own/sgd) | acc | cos(dev) h0/h1 |
|---|---|---|
| onehot | 0.868 | +0.21 / +0.21 |
| lin (raw) | ~0.35 | +0.99 / +0.98 (collapsed) |
| mlp (raw) | ~0.33 | +0.99 / +0.99 (collapsed) |

**Centering fixes the linear driver** (removes the shared component; a linear map then passes the
−0.24 geometry through): `lin_c` cos → **−0.23**, acc → **0.82** ≈ one-hot. The **mlp is only
partial** (cos → +0.40): its `relu` re-correlates the centered features into the positive orthant.

## 2. A centered lin driver MATCHES one-hot — under the oracle
16-cell grid, oracle eval (`lin_c` = centered lin):

| arm | opt | one-hot | **lin_c** | lin_c cos | note |
|---|---|---|---|---|---|
| buf-own | sgd | 0.868 | 0.832 | −0.23 | ✓ (high-variance arm; see §4) |
| buf-own | adam | 0.491 | 0.128 | −0.24 | ✗ Adam breaks BOTH |
| er-own | sgd | 0.729 | **0.966** | −0.23 | lin_c beats (one-hot gate inert) |
| er-own | adam | 0.991 | **0.989** | −0.23 | ✓✓ matches pt5's er+gain ~0.99 |

Per-synapse behaves the same in the working regimes (buf-own/sgd 0.974=0.974, er-own/adam 0.989),
but its `lin_c` projection is `784×n_syn` ≈ **125M params** (260× the main net) — the content driver
does not scale to per-synapse.

## 3. Remove the eval oracle → everything falls below plain ER  (the real result)
Same `lin_c` training; eval with **no task id**. `per-image` = gate from the test image;
`nearest` = nearest-prototype task inference (infer acc **0.759** = the ceiling).

| arm | opt | naive | ER | oracle | per-image | nearest |
|---|---|---|---|---|---|---|
| buf-own | sgd | 0.629 | (0.711) | 0.772 | 0.500 | 0.628 |
| er-own | sgd | 0.629 | 0.711 | 0.968 | 0.778 | 0.753 |
| er-own | adam | 0.357 | 0.894 | 0.991 | **0.708** | **0.755** |

- Strong cell (er-own/adam): oracle **0.99 → 0.71/0.76**, and `nearest ≈ oracle × infer`
  (0.991 × 0.759 = 0.752). The disjoint gate has **zero tolerance for the 24% misrouted samples**.
- **No non-oracle configuration beats plain ER-Adam (0.894)** — best non-oracle number is 0.778.
- MLP is the same story (er-own/adam oracle 0.985 → 0.67/0.75), `lin_c ≥ mlp` throughout.
- The gate's ~0.99 was **entirely oracle-dependent**: this is a **task-IL mechanism**, and cheap
  task inference (76%) is not accurate enough to sustain it (re-hits the pt3-Iter-8 wall).

## 4. Two methodology notes
- **Overlap is measured on `dev`(=raw), NOT `g`(=1+raw).** The shared parity `1` inflates `cos(g)`
  toward +1 for *gentle* gates: one-hot `cos(dev)=+0.21` but `cos(g)=+0.79`; er-own/sgd `lin_c`
  `cos(dev)=−0.22` but `cos(g)=+0.99` at `|dev|=0.06`. Inflation tracks `|dev|` (buf-own `|dev|≈5`:
  cos_dev≈cos_g). `cos(dev)` strips the task-independent offset and is invariant to gate strength.
- **buf-own/sgd is high-variance** (`lin_c` over seeds 42–46: **0.773 ± 0.083**, range 0.61–0.83;
  one-hot 0.895 ± 0.014). The differentiation (`cos ≈ −0.23`) is rock-stable every seed; only the
  standalone accuracy is noisy. Report buf-own numbers with ≥3 seeds; the er-own arms are stable.

## Verdict
The one-hot's essential property is that it makes the per-task gates **orthogonal / independent**;
you can recover that from a **decorrelated content driver** (centered mean image + linear map), and
it matches one-hot **under the oracle**. But no driver representation removes the **task-inference
dependency** — every oracle-free eval falls below plain ER. The reportable pt5 gain result stays the
one-hot disjoint cell, and the driver axis does not open a path to a genuine class-IL win.
