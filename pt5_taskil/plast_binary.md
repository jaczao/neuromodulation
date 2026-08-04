# Selective plasticity — a hard {0,1} learned mask via a clipped STE

THESIS-PLAN B's `w_{t+1}_ij = w_t_ij − σ·1[(i,j) ∈ p]·∇L`, made learnable with the clipped
straight-through estimator (forward `1[z ≥ 0]`, backward `1[|z| ≤ 1]`). Standalone buf-cur, task-IL,
SGD, main lr 3e-3 / ep 5 / buffer 1000, `neuro_lr` tuned per granularity on val, 3 seeds. Code
`plast_binary.py`, log `plast_binary.log`, ledger `plast_binary_results.tsv`.

**Verdict: REJECT.** No cell beats naive (0.9784), let alone EWC (0.9821) or ER (0.9946). But the
study answers a sharper question than the previous plasticity rejections, because of *why* it was run.

## The prediction this was designed to test

Every graded plasticity gate in this project died the same way: it collapsed onto a **uniform global
LR knob** (pt7's `ach_ema` 2.6× rescale that dissolved at a tuned lr; `gate_stats`'s cos(dev)
0.97–0.99; `meta_schedule`'s α → 0.99 uniformly, whose entire measured effect was recovering its own
0.5 init). A *binary* gate cannot express a graded rescale — a uniform binary gate is either α≡1
(exactly vanilla) or α≡0 (no learning at all). So the artifact that explained away every prior result
is unavailable by construction, and **any** effect here would have to be genuine allocation.

**Result: no effect.** Which is the cleanest available confirmation that the prior positives really
were the artifact, and that the one-step lookahead meta-loss contains no allocation signal to find.

## Anchor

`--part anchor` runs the same loop with the *original sigmoid* modulator and reproduces
`plast_init_results.tsv` bit-exact (neuron nlr 1e-2 → 0.979672; dead → 0.978401), so the loop is
isolated from the estimator and only the gate differs in the mechanism cells.

## Result 1 — the `open` init self-disables in ~50 steps (measured)

With `b = 0` every element starts exactly on the threshold, α≡1 (all plastic, parity). Instrumented:

| step | z mean | z range | frozen frac | P.grad norm |
|---|---|---|---|---|
| 0 | +0.098 | [0.000, 0.100] | 0.000 | 5.4e-04 |
| 10 | +1.025 | [0.252, 1.100] | 0.000 | 4.6e-04 |
| 50 | +1.933 | [1.183, 2.065] | 0.000 | **0.000e+00** |
| 100 | +1.954 | [1.226, 2.087] | 0.000 | **0.000e+00** |

The meta-loss pushes z **uniformly upward** — a one-step lookahead always prefers a larger step — so
every element leaves the `|z| ≤ 1` window and the clipped STE then yields *exactly zero* gradient
forever. The gate welds at α≡1. Confirmed end-to-end: the fraction of elements with `|z| > 1` at each
task boundary is 1.00 (global), 0.98–1.00 (neuron), 0.46–0.60 (synapse), frozen fraction 0.00
throughout, and the runs are **byte-identical to naive** (global and neuron, seed 42: 0.980286 =
naive's 0.980286). It is its own dead control at any `neuro_lr`, which is why its val sweep returned
the same number to 4 dp across 1e-5…1e-1.

**Generalisable:** any STE gate initialised *exactly* at its threshold, under a meta-objective with a
consistent sign, drifts out of the clipped window and freezes there. The saturation that gives the
estimator its stability also makes that drift a one-way door — an unclipped STE would keep passing
gradient and could return.

## Result 2 — the `spread` init (~50% frozen at start) still allocates nothing

`b ~ N(0, 0.5)` puts elements across the threshold, so ~50% start frozen and ~95% sit inside the
window where gradient flows. The dead control (`neuro_lr = 0`) then freezes *the same random half*
forever, so `d-dead` asks the sharp question: **does learning which elements to freeze beat freezing
a random half of the same density?**

| gran | nlr | binary | dead (random half) | d-dead | pos | **vs naive** |
|---|---|---|---|---|---|---|
| global | 1e-1 | 0.9784±0.0016 | 0.9784±0.0016 | +0.0000 | 0/3 | **+0.0000** |
| neuron | 1e-2 | 0.9776±0.0025 | 0.9657±0.0062 | **+0.0119** | 3/3 | **−0.0008** |
| synapse | 1e-4 | 0.9760±0.0021 | 0.9761±0.0022 | −0.0001 | 1/3 | **−0.0024** |

**The +0.0119 is handicap recovery, not a win — read the last column, not `d-dead`.** The neuron cell
is 3/3 positive against its control only because I handicapped that control by freezing a random half
of the units (which costs 0.0127). The live gate climbs back to 0.9776, i.e. *to* naive and not past
it (−0.0008). The gate structure says the same thing directly: the frozen fraction falls from 0.50 at
init to **0.023 (h0) / 0.037 (h1)** — the gate learns to **unfreeze**, exactly as in `meta_schedule`
where it spent itself undoing its own 0.5 init.

This is the second time in this package that a large, all-seeds-positive `d-dead` has turned out to be
the mechanism repairing its own initialisation. **A control must be handicapped no more than the
mechanism is, or `d-dead` measures the handicap.**

**No per-task allocation.** IoU of the frozen sets across tasks is **0.96 (h0)** and **0.988
(synapse)** — every task freezes nearly the *same* elements. The shared offset `b` starts all tasks at
IoU 1.0, so differentiation would show as IoU falling; only h1 moves appreciably (0.65). The disjoint
reference for T=5 is frozen fraction 0.8 at IoU 0.60. This is the hard-gate analogue of the sigmoid
studies' cos(dev) ≈ 0.99.

**`global` behaves exactly as the degenerate control it is:** never freezes (freezing globally means
no learning at all), so both arms are numerically naive and `d-dead` is exactly 0.0000 in all 3 seeds.
That is THESIS-PLAN B's required scalar control returning the null it must return.

**`synapse` never engaged:** the frozen fraction stays at 0.494 (init 0.50) and its tuned `neuro_lr`
is 1e-4, the grid floor — the same "tuning selects the off-switch" signature `gate_stats` found for
per-synapse plasticity.

## Incidental result worth keeping

A **random** static freeze of half the network costs little in task-IL: **0.0127** for half the hidden
*neurons*, **0.0023** for half the *synapses* in net.0/net.2. The synapse figure in particular says
this backbone is massively redundant at this operating point — which is also why a per-synapse mask
has so little to gain here.

## Limits

- One STE window (|z| ≤ 1) and one spread (σ = 0.5); neither swept.
- The offset `b` is shared across tasks (per element, not per task), so tasks start at IoU 1.0 and
  must differentiate via P[t]. A per-task offset would start differentiated — untested.
- No sparsity/regularisation pressure toward freezing was applied. The meta-loss's only preference is
  "learn the current batch faster", so nothing ever rewards a freeze; an explicit retention term or a
  multi-step lookahead horizon is the untested variant that could change this.
- 1 seed for tuning, 3 seeds for every reported number; task-IL, SGD, one operating point.
