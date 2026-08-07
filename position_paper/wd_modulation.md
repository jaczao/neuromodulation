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
captures ~85% of the effect while reproducing the granularity pattern exactly. The
task-decodability probe stays at chance (0.21–0.29) throughout.

> **`d-dead` IS NOT COMPARABLE ACROSS DRIVERS THAT DO NOT SHARE A DEAD CONTROL.** Within a driver it
> is the right measure (mechanism vs its own no-op). Across drivers it smuggles the control
> difference into the comparison, and here that inverts a sign: `ach`/neuron scores d-dead +0.0250
> against `const`'s +0.0204, which reads as `ach` winning — but `ach`'s dead control is 0.8938
> against `const`'s 0.9004, a 0.0067 RNG shift from constructing the head, and `ach`'s LIVE accuracy
> is 0.9187 against `const`'s 0.9209. Paired live-vs-live, `ach` is **−0.0021** and `nerisez`
> **−0.0029**: they never beat the content-free control, they beat a suppressed baseline by more.
> Use paired live-vs-live for cross-driver claims, and only where the arms are RNG-matched.

Paired live-vs-live vs `const` at normal: `taskid` +0.0032, `vecproj` +0.0019, `ach_act` +0.0006,
`nerisez_act` −0.0024, `ach` −0.0021, `nerisez` −0.0029 — every cell null at this buffer size.

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

**RNG-matching, verified per regime rather than inferred.** At `budget` the dead controls of
`const`, `const5`, `taskid`, `vecproj` and `vecproj_norm` are all **bit-identical**
(0.775812 / 0.771833 / 0.794895), so the `taskid`-vs-`const` and `vecproj`-vs-`const` comparisons
above are genuinely RNG-matched. `ach`/`nerisez` build heads and are not.

At `normal` the grouping is DIFFERENT: `const` sits ~0.0005 apart from the other four, which are
bit-identical to each other. Re-running `const`'s cell under the current code reproduces its ledger
value exactly (0.901888), so this is deterministic and driver-dependent, not code drift.

> **Do not infer RNG-matching from "the driver builds no parameters" — verify it for the specific
> comparison and the specific regime.** The prediction (parameter-free ⇒ matched) is right at budget
> and wrong at normal, and the grouping does not follow K either (`const` and `vecproj_norm` are both
> K=1 and land in different groups). The normal-regime driver-vs-`const` deltas are all nulls, so
> nothing above depends on this; the budget deltas, which do, are verified matched.

### The entropy family, re-run under the ACTUAL-value convention

`ach`/`nerisez` were originally copy-forwarded as HEAD predictions, violating the project convention
that the entropy family must use actual values (one extra unmodulated forward). `ach_act` /
`nerisez_act` fix that and are kept as separate keys so both conventions are comparable. Paired
per-seed vs `const` at budget:

| driver | neuron | synapse |
|---|---|---|
| `ach` (head) | −0.0032 ± 0.0049 | **−0.0236 ± 0.0032** |
| `ach_act` (actual) | +0.0026 ± 0.0021 ~ | +0.0038 ± 0.0004 ~ |
| `nerisez` (head) | −0.0055 ± 0.0089 | −0.0099 ± 0.0039 |
| `nerisez_act` (actual) | −0.0005 ± 0.0110 ~ | −0.0109 ± 0.0062 |
| `taskid` | +0.0233 ± 0.0083 | +0.0204 ± 0.0043 |
| `vecproj` | +0.0319 ± 0.0166 | +0.0198 ± 0.0037 |

**The convention mattered, and it mattered in the direction that makes the null cleaner.** The
head-based `ach` at synapse was actively HURTING (−0.0236); the actual-value version is neutral
(+0.0038) — a 0.027 swing from replacing a prediction with the quantity it was predicting. So part of
what the head-based cells measured was head distortion, not the driver.

With the convention corrected the split is sharper than before: **task-informative drivers
(+0.020 to +0.032) versus the entropy/difficulty family (−0.011 to +0.004, every cell null).** That
is the project's "difficulty/novelty is not task identity" line with the entropy family finally
measured under the right convention, so the null is a proper claim rather than a confounded one.

Read it as: **a content-free structured decay captures the whole effect when replay is plentiful,
and a TASK-INFORMATIVE driver adds ~2–3 pts on top of it once the buffer is tight — while the
entropy family adds nothing under either convention.** 3 seeds, and
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

## 2c. Four follow-up controls, and what each rules out

**A tuned scalar decay with NO learning is a null.** `fixed` applies one val-selected `f` globally at
every boundary — classic weight decay at task boundaries, no gate, no meta-loss. At its selected
f = 0.9 it reads **−0.0021** against its own f = 1.0 no-op (0.8999 vs 0.9020), with the val grid
spanning 0.0018 (unresolved). So "decay at boundaries" per se does nothing: the +0.020 needs the
learning *and* the per-parameter structure, not just the operation.

**Rank is not what makes `taskid` win.** `const5` is content-free at K=5 — taskid's exact rank and P
shape. Budget/neuron: **+0.0465 vs `const`'s +0.0583 and `taskid`'s +0.0815**. Five rows do not help;
if anything K=1 is better. So taskid's budget margin is the task signal, not the projection's rank.

**`vecproj`'s margin needs its 32 dimensions.** The 1-D norm form collapses to content-free
performance — budget/neuron **+0.0465**, tracking `const5` almost exactly, against `vecproj`'s
+0.0901. What it contributes is the *direction* structure of the input-novelty vector (task-correlated,
probe 0.685), not its magnitude.

**`boundary_last` ≈ `boundary`, except where it destabilises.** Using the task's LAST driver value
instead of its mean is *identical by construction* for `taskid` and `const` (constant within a task —
a useful check that the variant fired correctly: +0.0232/+0.0249 both ways). For content drivers it
is within noise, except `vecproj`/neuron: **+0.0219 → −0.2649**. One raw unstandardised novelty batch
is far noisier than a ~950-batch mean, and the exp gate amplifies it.

## 3. Methodology notes worth keeping

**A grid edge is only a truncation warning at ONE end.** `step` selected the floor 20 times,
`boundary` the ceiling 14 times. Extending is right for the ceiling; at the FLOOR of a mechanism's
own axis the edge IS the off-switch, and extending buys nothing but inertness. The existing rule
("a boundary selection is a truncated grid") is written for a main lr and is directionally wrong here.

**`g_*` is the gate recomputed AT EVAL, not the gate applied during training — now MEASURED**
(`--part applied`, own ledger to avoid schema drift on 1,236 finished rows). `_apply_boundary`
returns the per-layer |f−1| it actually applied; the diagnostic prints it beside the eval
recomputation. Ratio = eval / applied, class-IL er-own boundary, seed 42:

| driver | global | neuron | synapse | applied |f−1| (neuron) |
|---|---|---|---|---|
| `taskid` | 1.34 | 0.78 | 0.96 | 0.143 / 0.143 / 0.278 |
| `ach` | 1.22 | 0.82 | 0.75 | 0.045 / 0.047 / 0.096 |
| `nerisez` | 2.02 | 0.90 | 0.87 | 0.057 / 0.062 / 0.095 |
| `const` | 1.70 | 1.28 | 1.31 | 0.034 / 0.038 / 0.069 |
| **`vecproj`** | **94.3** | **1.9e11** | **36.9** | 0.188 / 0.188 / 0.326 |

**The hypothesis is confirmed and the scope of the problem is exactly one driver.** For four of the
five, eval is a fair proxy (ratio 0.75–2.02). For `vecproj` it is off by up to **eleven orders of
magnitude** — and its APPLIED gate is 0.19/0.19/0.33, entirely ordinary and in line with `taskid`'s
0.14/0.14/0.28. So the 3.8e12 was never a runaway gate; it was the raw un-standardised headless
novelty driver reading out of distribution on the test stream.

This retroactively strengthens the accuracy results rather than qualifying them: `vecproj`/neuron
reaches 0.9221 with a perfectly sane applied gate, so its budget-regime +0.032 over `const` is not an
artifact of a pathological magnitude. It also shows `ENGAGE_CAP = 10` was the right call — it
excluded precisely the artifact readings (11 to 1e11) while leaving every genuine gate (applied
≤ 0.33, eval ≤ 0.57 elsewhere) untouched.

The discriminator is the driver FAMILY, not the granularity: only the raw headless input-novelty
driver can read out of distribution at test. Head-based drivers (`ach`, `nerisez`) are functions of
the image through a trained head, and `taskid`/`const` are constants — none of them can.

> **RULE: for a training-time-only gate, report the APPLIED magnitude. The eval recomputation is a
> fair proxy except for raw headless drivers, where it can be wrong by eleven orders of magnitude.**

**The `nlr` column means different things per schedule, and a control has to follow.** `fixed`
reuses `nlr` to carry `f`, so the usual `nlr = 0` control multiplied every weight by zero and read
0.0927 = chance. `control_nlr(schedule)` now returns 1.0 for `fixed` and 0.0 for the learned
schedules, and is used by the runner AND the report selectors. When a key column is overloaded, the
no-op value is overloaded with it.

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
