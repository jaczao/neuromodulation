# Modulated weight decay — `w ← f(s)·(w − σ∇L)`

THESIS-PLAN direction B, position-paper mechanism 2. Split MNIST, SGD, val-tuned main lr, 3 seeds,
every delta against the RNG-matched dead-gate control (rule #10).

**Verdict.** The paper's per-step form is null-or-harmful and has no stable engaged operating point.
A per-task-boundary variant of our own reaches **+0.020–0.025 over its dead control in class-IL /
er-own, 3/3 seeds** — but a **content-free control reproduces ~85% of it**, so the mechanism is
LEARNED STRUCTURED DECAY, not neuromodulation. The neuromodulator signal contributes nothing
measurable. That is a genuine CL result with a negative neuromodulation result inside it.

---

## Setup

| | |
|---|---|
| gate | `f = exp(m̄ @ P)`, `P` zero-init ⇒ `f = 1` parity. `DecayGate` subclasses pt7's `PlastGate` and overrides NOTHING |
| trained by | pt5 lookahead meta-loss on a replay-containing batch; the decay is applied under `no_grad`, so the main loss gives `P` no gradient |
| drivers | `taskid`, `ach`, `nerisez`, `vecproj` (raw, un-standardised) + `const` (content-free control) |
| granularity | `global` (1 scalar) / `neuron` (810) / `synapse` (477,600) |
| arms | `erown` (main net + gate on the ER batch) / `bufcur` (naive backbone, replay reaches only the gate) |
| main lr | val-tuned per (metric, arm) from `neurocore.tuned`; NOT swept here (rule #11) |
| anchors | task-IL ER-SGD **0.994133 bit-exact**; class-IL ER-SGD 0.903365 vs pt7_tuned_syn 0.9034 |

Only one lever differs from the plasticity studies in `pt5_taskil/`: where the multiplier lands
(weight vs gradient). Same gate, same loop, same meta-loss, same drivers.

---

## 1. The per-step form has no stable, engaged operating point

`step` applies `f` after every update — ~4750 times — so what reaches a weight is `f^4750`, while
the lookahead meta-loss is ONE step deep. A per-step `f = 1.001` is a 115× blow-up.

Measured consequence: divergence to chance that is **NON-MONOTONE in neuro_lr** (`nerisez` diverged
at 1e-5 but not at 1e-3; `vec_x` the reverse). That is a positive feedback loop — weights grow, the
driver moves, the gate pushes further — not a step-size problem, so **there is no stability
threshold to normalise against** and every driver needs its own sweep.

Tuning, 48 cells: 34 selected a gate at |f−1| ≤ 0.001; 20 selected the grid floor 1e-7; 37 had at
least one grid point collapse to chance. The selections are not a tie-break artifact — ties are
broken toward MORE engagement — the engaged cells simply score worse, often far outside the noise
floor (`classil/bufcur/ach` spans 0.268).

Test grid: everything inside ±0.007 except negatives (`classil/bufcur/taskid` −0.03 to −0.07) and
two cells with catastrophic per-seed instability (`taskil/erown/taskid/global` 0.693 ± 0.42).

### The one thing that looked like a win, and isn't

`taskil/bufcur`: `taskid/global` +0.0102 and `ach/global` +0.0080, both 3/3. Their |f−1| is
0.0001–0.0009, which reads as parity — but `exp(4750 × 9e-4) ≈ 60×`.

> **PER-STEP |f−1| IS THE WRONG ENGAGEMENT MEASURE FOR A COMPOUNDING GATE.** Report the compounded
> exponent `n·log f`, not the per-application deviation. A gate that looks inert per step can be a
> 60× aggregate weight rescale.

So those two cells are large global norm rescales in the arm whose baseline underfits — pt7's
global-LR-knob artifact in a new costume, and it appears in `global` granularity, which is where a
norm knob should appear.

---

## 2. The boundary variant works — and is not neuromodulation

`boundary` fires `f` once per task over that task's mean driver, trained on exactly that operation.
Five applications instead of 4750. It is stable out to nlr 1e-1 where `step` diverges at 1e-4 —
the compounding diagnosis confirmed from the other side.

**class-IL / er-own**, d-dead, 3 seeds (ungated ER 0.9019):

| driver | global | neuron | synapse |
|---|---|---|---|
| `const` (control) | +0.0005 | **+0.0204** | **+0.0197** |
| taskid | −0.0013 | +0.0232 | +0.0249 |
| ach | −0.0004 | +0.0250 | +0.0038 |
| nerisez | −0.0004 | +0.0242 | +0.0223 |
| vecproj | −0.0003 | +0.0219 | +0.0235 |

All neuron/synapse cells 3/3 seeds positive. Forgetting falls 0.088 → 0.051–0.071.

**Prediction 1 was REVERSED, and that is the finding.** It was pre-registered that
`global ≈ neuron ≈ synapse` would mean a global weight-norm knob. Instead **global is null and
neuron ≈ synapse are positive** — structure matters, down to per-neuron but no further. This is the
first mechanism in the project where granularity mattered at all.

**But the driver does not.** Four unrelated drivers — a task one-hot, an entropy head, a stateful
surprise z-score, a random input projection — land within 0.003 of each other, and `m(x) ≡ 1`
captures ~85% of the effect while reproducing the granularity pattern exactly. Every driver's
marginal contribution over `const` (+0.0015 to +0.0052) is inside the ±0.007 noise floor. The
task-decodability probe stays at chance (0.21–0.29) throughout.

> **The mechanism is: per-parameter decay coefficients, learned on replay, applied at task
> boundaries.** What matters is that `P` has per-neuron freedom and a retention objective. What the
> driver says is irrelevant.

It is not a capacity confound at the granularity that matters: `const`/neuron is an **810-param**
projection, 0.0017× the 478k backbone, and beats ER by ~2 pts. (`vecproj`/synapse at 31.9× IS
confounded and is reported with its ratio; it adds nothing over the 810-param control anyway.)

### 2b. Under MEMORY PRESSURE the driver starts to matter — the one real driver effect

Rule #12's regime axis turned out to carry the most interesting result in the study. At `budget`
(buffer 200, ER falls to 0.7913) every effect roughly quadruples, and the content-free control no
longer explains all of it. Paired per-seed, live-vs-live against `const`, neuron granularity:

| driver | normal (buf 1000) | budget (buf 200) | RNG-matched to `const`? |
|---|---|---|---|
| `taskid` | +0.0032 ± 0.0047 ~ | **+0.0233 ± 0.0083** (3/3) | yes |
| `vecproj` | +0.0019 ± 0.0029 ~ | **+0.0319 ± 0.0166** (3/3) | yes |
| `ach` | −0.0021 ~ | −0.0032 ~ | no |
| `nerisez` | −0.0029 ~ | −0.0055 ~ | no |

The two that gain are exactly the two carrying TASK information — `taskid` by construction, and
`vecproj` with the highest task-probe of the content drivers (0.685). The two difficulty/entropy
drivers add nothing at either buffer size. That is the project's "difficulty/novelty is not task
identity" line reproduced, but for the first time with a POSITIVE sign on the task-informative side.

`taskid` and `vecproj` build no parameters, so they consume the same construction RNG as `const` and
their dead controls come out **bit-identical** — the comparison is genuinely RNG-matched (checked,
not assumed). `ach`/`nerisez` build heads and are not, so their nulls are the weaker claim.

Read it as: **a content-free structured decay captures the whole effect when replay is plentiful,
and a task-informative driver adds ~2–3 pts on top of it once the buffer is tight.** 3 seeds, and
`vecproj`'s spread is large (per-seed +0.010 / +0.050 / +0.035), so this wants more seeds before it
carries weight.

### `rfree` is structurally degenerate — and says so cleanly

At buffer 0 every cell returns **0.1976 with |f−1| = 0.0000**, identical to naive. The boundary
meta-loss trains `P` on buffer samples, so with no buffer `P` never leaves zero and the gate is
exactly parity. The mechanism does not merely underperform without replay — it does not exist. Same
shape as the loss-modulation degeneracy, and worth stating rather than reporting a chance number as
if it were a result.

### Where it does not appear

- **task-IL / er-own**: null. ER is already 0.9934 and `trajectories.py`'s ceiling argument applies.
- **class-IL / bufcur**: null or negative — the gate needs replay in the MAIN net, not just its own.
- **task-IL / bufcur**: uniformly +0.003 to +0.007, 3/3 seeds, but at/inside the noise floor.

---

## 3. Methodology notes worth keeping

**A grid edge is only a truncation warning at ONE end.** `step` selected the floor 20 times,
`boundary` the ceiling 14 times. Extending is right for the ceiling; at the FLOOR of a mechanism's
own axis the edge IS the off-switch, and extending buys nothing but inertness. The existing rule
("a boundary selection is a truncated grid") is written for a main lr and is directionally wrong here.

**`g_*` is the gate recomputed AT EVAL, not the gate applied during training.** The raw
un-standardised novelty drivers read far larger on the test stream than in training, so
`vecproj`/boundary logged |f−1| up to 3.8e12 with accuracy at baseline. For a decay target the gate
never enters the forward, so an eval-time gate is diagnostic only. It mattered because it was
feeding the tie-break; capped at `ENGAGE_CAP = 10` for ranking purposes only. **An applied-gate
column is the right fix and is not yet implemented** — the boundary rows' engagement numbers should
be read as indicative, not as the operative factor.

**A dead control cannot separate "the driver spoke" from "the gate had freedom".** It has no learned
gate at all, so everything a learned gate buys shows up as mechanism. The content-free control is
what splits them, and it should be run as soon as a near-identical delta appears across unrelated
drivers — that pattern IS the signature.

---

## 4. Limits

1 seed for tuning, 3 for every reported number. `ach_ema` (the only tonic driver), `vec_x` and
`all5` (the only composite) were dropped for budget — prediction 2 is therefore untestable here.
`META_STEPS = 50` and the boundary-per-task schedule are single unswept points. The boundary
meta-loss is retention-only, so it has no pressure to decay; adding an explicit weight-norm term is
the untested variant that could change what the gate allocates. Regimes (`budget`, `rfree`) not yet
run. The `+0.02` is against tuned ER at THIS operating point on Split MNIST class-IL and should not
be assumed to survive a different dataset or a stronger baseline (DGR ~0.91).
