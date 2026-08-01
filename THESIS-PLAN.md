# THESIS-PLAN — neuromodulation across multiple problems

Roadmap for the phase after Split MNIST. Scoped by PROBLEM, not by architecture: the Split-MNIST
phase already established that the limiting factor is what the mechanism can reach, not how big the
network is, so the next axis is *which learning problem* the mechanism is asked to solve.

Repo layout this plan assumes:

```
neurocore/     shared core: drivers/signals, gate primitives, projections, controls, ledger, cost
prototype/     FROZEN, archival. Split MNIST. Reproducible, never extended.
results/       FROZEN, archival. ~76 self-contained pt3-pt8 study scripts + their ledgers.
<problem>/     one self-contained package per direction, owning its data, backbone, baselines, loop.
```

---

## 1. What the Split-MNIST phase established

Carry these forward as constraints on the new work, not as background reading. Each was paid for.

**Replay is the only lever that moved class-IL.** Nothing across pt2-pt8 beat a tuned ER by a
decisive margin: not four gate targets, not three projection types, not driver representation, not
the four classic neuromodulators, not a 23-signal feature net, not a factorized task-inference head.

**A jointly-trained multiplicative gate is absorbed by the backbone.** The h1-gate sat at mean 0.281
— far from parity — and landed exactly on ER, because the weights it multiplies simply rescale. L1
on the gate shifts magnitude between gamma and W without changing the function. Gate is not memory.
The one mechanism that worked (pt5 iter-1 disjoint gain) used a hard {0,1} freeze, which kills the
gradient and is therefore un-absorbable.

**It is not an over-parameterization artifact.** Shrinking hidden width to 5 units manufactured real
scarcity (ER 0.891 -> 0.575) and the gate delta never emerged, while gate *engagement* rose ~15x —
it engages exactly when a resource-allocation account says it should, and is still absorbed.

**Difficulty/novelty is not task identity.** The pt7 drivers probe at 0.21-0.52 task-decodability
against pt6's learned selector at 0.88. That gap is the mechanistic "why" behind the pt7 negative.

**Any task-conditioned decomposition is capped by its task-inference stage.** `pred ~= oracle x
infer` held for hard routing (pt3 iter-8, pt6) and again for a soft factorized posterior (pt8).
Softness buys no tolerance to misrouting.

**Methodological rules that cost real time to learn:**
- Tune the baseline before claiming headroom. Tuning ER lifted SGD 0.72 -> 0.90 and dissolved pt7's
  apparent "+0.14 SGD boost" — the gate was compensating for an underfit baseline.
- Compare against the RNG-matched `free` control, never the plain baseline.
- A zero-init module stacked on a zero-init projection is dead by construction.
- Standardize per-sample drivers; never standardize a tonic one.
- Report gate magnitude per layer, never as a single mean.
- Arms sit at different operating points; never compare arm A to arm B, only mechanism to its own
  baseline.

---

## 2. Baseline regimes (every CL direction reports all three)

| regime | stored samples | note |
|---|---|---|
| **normal** | replay allowed | ER and friends; the current headline bar (~0.90 class-IL) |
| **rehearsal-free** | none | **DGR is legal and is the real bar (~91%)** — a generative replay method stores no samples. Do not report a rehearsal-free win against a weaker bar. |
| **memory-budgeted** | tiny buffer | where buffer bytes actually bind |

`neurocore.cost` makes this legible: `regime` is a ledger KEY column and `buffer_bytes` a metric, so
"reaches X at zero stored bytes" is a statement the table can make.

## 3. Baseline families to build out (per problem package, as each is reached)

- **regularization** — EWC, SI, MAS, LwF
- **replay** — ER, DGR, MIR, GDumb, A-GEM
- **gating** — XdG, HAT
- **TTA** — frozen source, BN-stats, TENT, T3A, LAME

## 4. Problem directions

Each becomes a self-contained package owning its data, backbone, baselines and loop, assembled from
`neurocore` primitives. `neurocore/verify_anchors.py` is the reference example of that structure.

**A. Task-IL revisited.** Split MNIST at val-tuned SGD *and* Adam operating points (the pt5/6/7 CL
regime was never tuned; only class-IL ER has been), then task-IL on a new dataset (Split
Fashion-MNIST / Split EMNIST). Note the fixed `disjoint`/`shared` projections need a task id at eval
— legitimate here (the XdG convention), an oracle in class-IL. `neurocore.projections` carries that
label.

**B. Position-paper mechanisms** ("Neurotransmitters as a Missing Dimension"), plus own variants:
- loss modulation `L = sum_T f(s_T, T) L_T`
- modulated weight decay `w_{t+1} = f_w(s_t)(w_t - sigma grad_w L)`
- selective plasticity `w_{t+1}_ij = w_t_ij - sigma . 1[(i,j) in p] grad_{w_ij} L`

Prior that applies directly: pt7's plasticity study found a per-parameter LR gate reduced to a
*global* LR-scaling artifact — one scalar matched per-neuron and per-synapse structure, and the whole
effect vanished once the main lr was tuned. Run these against a TUNED main lr from the start, and
include the `global` scalar control, or a nonzero-mean driver will just recover the missing tune.

**C. Domain-IL** — Permuted / Rotated MNIST.

**D. TTA** — MNIST-C, frozen backbone. This is where `fwd_infer`/`bwd_infer` is expected to win
outright: a frozen source model costs no backward at inference, TENT-style adaptation does.

**E. Meta-learned gate** — Split EMNIST as a task distribution.

**F. Task-free / online** — single-pass streams, no task boundaries.

**G. Reward-modulated contextual bandit** — the one setting where a reward-prediction-error driver is
not a metaphor. DA has an actual reward signal here rather than a loss-derived proxy.

**H. Neuromodulated fast weights** — episodic memorize-and-recall.

---

## 5. Promotion policy (supersedes pt7's)

**Promote by future use.** A mechanism moves into `neurocore` when a second problem actually calls
for it — not on a schedule, and not because it was explored once.

This replaces pt7's "promote ALL pt5/pt6/pt7 mechanisms, winners and non-winners, into
`neuromod.py`". That policy was written for a scaffolding step that assumed the project stayed on
Split MNIST, where a single live registry could hold the full ablation set behind one flag. Across
several problems with different datasets and backbones, promoting non-winners costs maintenance on
every one of them and buys nothing: the value of a rejected mechanism is the writeup in
`prototype/iteration-notes.md`, which is complete and reproducible from the frozen `results/` scripts.

Consequences:
- `results/` stays runnable and frozen; it is the archive of record for every rejected mechanism.
- The first extraction likely to be called for is pt6's `soft_mlp` / `embedding` selector — the best
  oracle-free result (~0.88 = ER parity) and the natural fit for the task-IL and meta-learned-gate
  directions.
- Extract on second use, and copy-forward rather than cut, so archived numbers keep reproducing.

## 6. Reproducibility contract

Frozen anchors, all verified after the extraction:

| path | anchor |
|---|---|
| pt7 (seed 42, Adam, lr 1e-3 / ep 5 / buffer 1000) | naive 0.3900, er 0.8946, free 0.8760, all4 0.8816 |
| pt7_tuned_syn | ER-sgd 0.9034, ER-adam 0.8975 |
| pt5 iter-1 disjoint gain (SGD) | naive+gain 0.6225, er+gain 0.8264 (needs `--no-neuromod-er-task-id`) |

`uv run python neurocore/verify_anchors.py` checks the pt7 set through the extracted primitives;
the archived path reproduces the rest unchanged. `uv run pytest tests/` must pass throughout.

Tuned operating points live in `neurocore/tuned.py`, keyed by `(problem, metric, base, optimizer)`
and `(problem, metric, base, optimizer, mechanism, granularity)`. A missing key raises by design:
an un-tuned combination must be tuned on a validation split, never guessed and never inherited from
another problem.
