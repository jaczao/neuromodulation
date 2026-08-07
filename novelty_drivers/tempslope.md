# Novelty-driver form under the TEMPERATURE and SLOPE gates (class-IL)

**Verdict: REJECT — 111 of 128 cells are inside the noise floor and exactly ONE is positive beyond it
(+0.0073, at 1 seed). Every cell that moves, moves DOWN, and all 17 of them are the same driver:
`vec_x`'s raw 784-d vector form.** With the capacity confound removed (the widest gate here is 1,568
parameters, 0.0033× the backbone, against the linear gate's 635,040 = 1.33×), the norm axis gives a
cleaner answer than the gain study could: **the norm form is null in 64 of 64 cells, and the vector
form is harmful only for `vec_x` — `vecproj`'s vector form is null too.** So "784 raw dimensions"
was never carrying information; it was carrying a conditioning problem.

The two reusable results are (1) the temp-vs-slope asymmetry measured at *matched* gate magnitude,
and (2) what `|g−1| = inf` in the table actually is.

- study: `novelty_drivers/tempslope.py` · ledger `tempslope_results.tsv` (145 rows)
- sibling under the rank-K linear gain gate: `novelty_drivers/novelty_drivers.py` / `.md`

---

## Setup

`p` is a `(K,)` vector, so `m @ p` is one **scalar** per sample and the gain is uniform across a
layer — the structural difference from the linear gate, which gives each of 810 units its own
coefficient:

    temp   logits *= exp(m @ p_out)                  — out layer only
    slope  h0 *= exp(m @ p_h0);  h1 *= exp(m @ p_h1)  — hidden only

Axes: form {temp, slope} × kind {vec_x, vecproj} × norm {0,1} × std {0,1} × mean_mode {ema,
cumulative, trueavg, ema+trueavg} × proj {learned, random} = 128 grid cells. `mean_mode` was
included as a superset of the requested axes, so the table answers either reading.

class-IL er-own, Adam, buffer 1000, lr 3e-4 / 5 ep/task from `neurocore.tuned` — **not retuned**, the
same val-selected ER point the gain study used, so the two are directly comparable. 1 seed; floor
0.007.

**Anchors:** plain ER = **0.897549**, bit-exact against the frozen `pt7_tuned_syn` ledger; and all
**16 dead controls** (both forms × both std arms × both kinds × both norms) equal it exactly — `exp(0)
= 1`, so parity is structural here, and the RNG-matching claim is verified rather than assumed.

---

## d-dead, by block (out of 8 cells each)

| form | kind | norm | std | min | max | outside floor |
|---|---|---|---|---|---|---|
| temp | vec_x | 0 | 0 | −0.104 | +0.003 | 4/8 |
| temp | vec_x | 0 | 1 | **−0.805** | +0.006 | 5/8 |
| temp | vec_x | 1 | 0/1 | −0.002 | +0.006 | **0/8** |
| temp | vecproj | 0/1 | 0/1 | −0.005 | +0.005 | **0/8** |
| slope | vec_x | 0 | 0 | −0.469 | +0.006 | 4/8 |
| slope | vec_x | 0 | 1 | **−0.805** | **−0.431** | **8/8** |
| slope | vec_x | 1 | 0/1 | −0.003 | +0.006 | **0/8** |
| slope | vecproj | 0 | 0/1 | −0.009 | +0.007 | 2/16 |
| slope | vecproj | 1 | 0/1 | −0.002 | +0.007 | 1/16 |

The single positive cell beyond the floor in the whole grid is `slope · vecproj norm1 std1
cumulative random` at **0.9049 (+0.0073)** — one cell out of 128 at 1 seed, i.e. what you expect
from noise, and not claimed.

## (1) The temp-vs-slope asymmetry, at matched gate magnitude

pt7 measured this once, on different drivers (`ne` temp 0.39 vs slope 0.10), so form and driver were
confounded. Here the driver, its `|m|`, and the gate magnitude `|g−1|` are **identical** between the
two forms — only the layer differs:

`vec_x` norm0, std=0, frozen p (|g−1| ≈ 11–18 in both forms):

| mean_mode | temp | slope | temp − slope |
|---|---|---|---|
| ema | 0.8296 | 0.5627 | **+0.267** |
| cumulative | 0.7935 | 0.4285 | **+0.365** |
| trueavg | 0.8313 | 0.4833 | **+0.348** |
| ema+trueavg | 0.8296 | 0.5441 | **+0.286** |

A blown-up `temp` costs 0.07–0.10; the identically blown-up `slope` costs 0.33–0.47. **A uniform
positive scale on the logits is argmax-invariant, so a broken `temp` can only corrupt training (it
reweights the per-sample loss by novelty); a broken `slope` also corrupts the eval forward.** The
same contrast appears at its most extreme in the std=1 learned arm — `temp` reads **0.9034 (+0.0059)**
and `slope` **0.3479 (−0.5496)** with the *same* driver and both reporting `|g−1| = inf`.

The corollary is the one that matters for design: **`temp` has no inference-time lever at all.** Its
entire effect is a training-time loss weighting, which is why it is so hard to damage — and equally
why it has nothing to contribute.

## (2) What `|g−1| = inf` is (measured, not inferred)

`inf` is a **mean over a heavy tail**, not a dead run. Re-running the two cells with an overflow
counter (both reproduce their ledger accuracies exactly, 0.9034 / 0.3479):

| form | overflowed | median gain | p99 | ‖p‖ |
|---|---|---|---|---|
| temp | **28 / 10,000 = 0.28%** | 1.101 | 15.3 | 0.446 |
| slope | **34 / 10,000 = 0.34%** | 0.503 | **9.08e4** | 1.155 |

So ~0.3% of test samples overflow float32 in `exp()`, which makes the batch mean infinite while the
median gain sits near parity. The two forms overflow at nearly the same *rate*; what separates them
is the **tail among the finite samples — p99 of 15 versus 90,000, four orders of magnitude** — and
where that tail lands. **Report a median or a quantile alongside a mean for any exponential gate; the
mean is destroyed by a handful of samples and hides that the typical gain is ≈ 1.**

## (3) With no capacity confound, the norm axis answers cleanly — and blames conditioning

The gain study could not separate "the 784-d vector form is bad" from "635,040 extra trainable
parameters are bad". Here the widest gate is 1,568 parameters and there is no confound anywhere, and
the result is sharper:

- **norm form: 0/64 cells outside the floor.** Every kind, both forms, both std arms, all four
  reference means. K = 1 → `p` is a single scalar (or two).
- **`vecproj` vector form: essentially null too** (3 of 32 cells outside the floor, max |d| 0.009).
- **`vec_x` vector form: 17 of 32 cells outside the floor, all negative**, down to −0.805.

So it is not "the vector form" that is harmful — it is **`vec_x`'s** vector form specifically, which
is exactly the driver with 212 constant border-pixel dimensions. Under the linear gate the same
driver was also the only catastrophic one. Two different gate families, same culprit.

## (4) Standardisation is now the DOMINANT axis, and its sign flips versus the linear gate

Under the linear gate, standardising `vec_x`'s vector form was what detonated it (0.87 raw → 0.12
standardised). Under `exp()` it is worse still and in a wider set of cells: `slope · vec_x norm0
std1` is outside the floor in **8/8** cells (−0.43 to −0.80) versus 4/8 at std=0. Standardising a
vector with constant dimensions inflates the driver, and an exponential gate then amplifies what an
affine gate merely added. Everywhere the driver is well-conditioned, the std axis is inert
(all `vecproj` and all norm-form blocks, both signs, inside the floor).

## (5) The reference mean is inert here too

Within every well-conditioned block the four reference means are indistinguishable (spreads
0.001–0.009 against a 0.007 floor, no consistent ordering). `ema+trueavg` is **numerically identical
to `ema` to 4 dp in every `temp` and `slope` cell where the gate is small** — e.g. `temp · vecproj
norm1 std0 learned` 0.8939 both, `slope · vec_x norm1 std0 random` 0.9038 vs 0.9033. The reference
only separates where the driver is already broken (`vec_x` norm0), i.e. exactly where the number
means nothing about the reference.

## (6) The probe ordering survives a third gate family

`vec_x` vector probes **0.933**, `vecproj` vector 0.685, the norm forms 0.19–0.25 — unchanged from
the gain study, unaffected by standardisation, and *inversely* ordered against benefit. The most
task-decodable driver in the project is the only one that can destroy the network, under a linear
gate and under an exponential one.

---

## Limits

- **1 seed**, 128 cells. The one positive beyond the floor (+0.0073) is one cell in 128 and is not
  claimed. Everything else is either far outside the floor (the `vec_x` vector damage) or reported
  as a null inside it.
- **σ = 0.1 for the frozen projection**, matched to the gain study so "learned vs frozen" means the
  same thing in both. Under `exp()` that is *not* scale-free: a frozen `p` interacts multiplicatively
  with the driver's raw magnitude, so the frozen arm's collapses are a statement about σ = 0.1 for
  these drivers, not about frozen projections in general. A σ sweep is the obvious follow-up and was
  not run.
- class-IL only, Adam only, er-own only, not retuned. An exponential gate arguably deserves its own
  lr, since the usable range is driver-magnitude-dependent.
- The overflow diagnostic in §2 was measured on two cells (std=1, learned, ema, `vec_x` norm0), not
  across the grid; the `inf` entries elsewhere are reported as-is.
