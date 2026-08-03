# pt5 iter-3 learned plasticity — TASK-IL, SGD, both nets val-tuned

Study `plast_taskil.py` (ledger `plast_taskil_results.tsv`, log `plast_taskil.log`), gate diagnostics
`gate_stats.py`, per-task curves `trajectories.py` (`trajectories.tsv`/`.log`).
Default class order for reporting, `make_sequence(7)` for tuning. Buffer 1000. 3 seeds {42,43,44}.
New code in its own package; `prototype/` and `results/` imported read-only (rule #9).

## Verdict

**Reject, in all five arms and both granularities.** At a val-tuned operating point, pt5 iter-3
learned-projection plasticity is negative against its own RNG-matched dead-gate control in all ten
mechanism cells (−0.0003 to −0.0017; 8 of 10 negative in every seed), and sits ~1.7 points below ER.

The gate diagnostics say *why*, and the two granularities fail differently: per-neuron **engages hard
but learns a global LR knob** (identical across tasks), per-synapse **never leaves parity at all**.
The trajectories add a ceiling argument: in task-IL at a tuned point there is only ~0.006 of
forgetting to recover, so no retention mechanism can close ER's 0.016 gap — most of that gap is
better *acquisition*, not better retention.

## Result (test set, 3 seeds)

| cell | acc (mean±std) | forget | lr | ep | λ | neuro_lr |
|---|---|---|---|---|---|---|
| naive | 0.9784±0.0016 | 0.0061 | 3e-3 | 5 | — | — |
| naive_unmasked | 0.9769±0.0028 | 0.0077 | 3e-3 | 5 | — | — |
| **er** | **0.9946±0.0004** | 0.0023 | 1e-1 | 5 | — | — |
| ewc | 0.9821±0.0011 | 0.0024 | 3e-3 | 5 | 100 | — |
| ewc (λ=0 control) | 0.9786±0.0017 | — | 3e-3 | 5 | 0 | — |
| ewc_er | 0.9936±0.0007 | 0.0034 | 1e-1 | 5 | 0.1 | — |
| dead-nobuf-neuron | 0.9784±0.0016 | 0.0061 | 3e-3 | 5 | — | 0 |
| plast-nobuf-neuron | 0.9779±0.0019 | 0.0063 | 3e-3 | 5 | — | 1e-2 |
| dead-nobuf-synapse | 0.9784±0.0016 | 0.0061 | 3e-3 | 5 | — | 0 |
| plast-nobuf-synapse | 0.9767±0.0017 | 0.0054 | 3e-3 | 5 | — | 1e-4 |
| dead-neuron (bufcur) | 0.9784±0.0016 | 0.0061 | 3e-3 | 5 | — | 0 |
| plast-neuron (bufcur) | 0.9778±0.0019 | 0.0063 | 3e-3 | 5 | — | 1e-2 |
| dead-synapse (bufcur) | 0.9784±0.0016 | 0.0061 | 3e-3 | 5 | — | 0 |
| plast-synapse (bufcur) | 0.9767±0.0017 | 0.0054 | 3e-3 | 5 | — | 1e-4 |
| plast-bufown-neuron | 0.9778±0.0019 | 0.0063 | 3e-3 | 5 | — | 1e-2 |
| plast-bufown-synapse | 0.9767±0.0017 | 0.0054 | 3e-3 | 5 | — | 1e-4 |
| dead-ercur-neuron | 0.9946±0.0004 | 0.0024 | 1e-1 | 5 | — | 0 |
| plast-ercur-neuron | 0.9936±0.0004 | 0.0028 | 1e-1 | 5 | — | 1e-2 |
| dead-ercur-synapse | 0.9946±0.0004 | 0.0025 | 1e-1 | 5 | — | 0 |
| plast-ercur-synapse | 0.9942±0.0004 | 0.0025 | 1e-1 | 5 | — | 1e-2 |
| dead-erown-neuron | 0.9946±0.0002 | 0.0024 | 1e-1 | 5 | — | 0 |
| plast-erown-neuron | 0.9938±0.0004 | 0.0027 | 1e-1 | 5 | — | 1e-2 |
| dead-erown-synapse | 0.9945±0.0005 | 0.0025 | 1e-1 | 5 | — | 0 |
| plast-erown-synapse | 0.9943±0.0004 | 0.0024 | 1e-1 | 5 | — | 1e-2 |

Deltas against each arm's own RNG-matched inert gate (rule #10):

| arm | gran | d-dead | per seed | positive |
|---|---|---|---|---|
| nobuf | neuron | −0.0005 | −0.0006, −0.0011, +0.0001 | 1/3 |
| nobuf | synapse | −0.0017 | −0.0017, −0.0019, −0.0016 | 0/3 |
| bufcur | neuron | −0.0006 | −0.0006, −0.0012, +0.0001 | 1/3 |
| bufcur | synapse | −0.0017 | −0.0016, −0.0019, −0.0017 | 0/3 |
| bufown | neuron | −0.0006 | (identical to bufcur) | 1/3 |
| bufown | synapse | −0.0017 | (identical to bufcur) | 0/3 |
| ercur | neuron | −0.0010 | −0.0015, −0.0010, −0.0006 | 0/3 |
| ercur | synapse | −0.0004 | −0.0004, −0.0004, −0.0004 | 0/3 |
| erown | neuron | −0.0008 | −0.0014, −0.0008, −0.0003 | 0/3 |
| erown | synapse | −0.0003 | −0.0005, −0.0002, −0.0001 | 0/3 |

Both design axes the arms exist to separate are null: **buffer vs no buffer** (nobuf −0.0005/−0.0017
vs bufcur −0.0006/−0.0017) and **correct vs wrong task routing** (erown −0.0008/−0.0003 vs ercur
−0.0010/−0.0004). Notably `er_task_id` was worth **+0.356** for per-synapse *gain* in pt5 — routing
replayed samples through the wrong mask scrambled their forward — but gating the *gradient* instead of
the forward makes it worth ~0.0004: a wrong-task LR gate merely mis-scales a step, it does not corrupt
the features the step is computed from.

## 1. Why it is null — the gate diagnostics (`gate_stats.py`, buf-cur, seed 42)

α = sigmoid(init_bias + P[t]) ∈ [0,1] multiplies the per-parameter LR; parity is α=0.5 (P zero-init),
so α−0.5 is the learned task-specific modulation. Reported **per layer**, per the pt6-followup rule.

| | layer | \|α−0.5\| | α mean | α sd | α>0.75 | cos(dev) | frac |
|---|---|---|---|---|---|---|---|
| plast-neuron | h0 | 0.433 | 0.933 | 0.073 | 0.968 | **0.991** | 0.997 |
| | h1 | 0.377 | 0.877 | 0.125 | 0.859 | **0.968** | 0.925 |
| plast-synapse | net0 | **0.0007** | 0.5007 | 0.0016 | 0.000 | 0.48 | 0.033 |
| | net2 | **0.0007** | 0.5007 | 0.0015 | 0.000 | 0.41 | 0.038 |

**plast-neuron engaged hard, and learned a global LR knob.** α moved from 0.5 to ~0.93/0.88 with
86–97% of units above 0.75 — far from parity, so this is *not* the dead-gate/saddle failure mode. But
`cos(dev)` = 0.97–0.99 means every task learned essentially the same deviation, with 92–100% of units
engaged. That is not allocation; it is the gate discovering that its 0.5 init throttles learning and
opening everything up. Per-task means confirm it (h0: 0.82 / 0.98 / 0.97 / 0.93 / 0.96 — the scale
varies slightly, the pattern across units does not).

This is pt7's `ach_ema` result reproduced on a different mechanism: a plasticity gate that looks like
it is doing something is a **global LR rescale**. There it inflated accuracy by +0.11–0.13 against an
*untuned* SGD baseline and dissolved once the main lr was tuned. Here the main lr is tuned from the
start, so opening the gate merely returns to naive's effective step — hence the tie, by construction.

**plast-synapse never engaged.** |α−0.5| = 0.0007, α confined to [0.499, 0.527], nothing below 0.25 or
above 0.75. Its tuned neuro_lr was 1e-4, the grid floor: tuning selected the gate's off-switch. Its
`cos(dev)` of 0.41–0.48 is noise around zero given |dev|≈0.0007 — cosine must be read together with
magnitude (the pt5_mask_overlap rule). So that arm is a uniform half-LR naive run, which is also why
it is the most negative cell (−0.0017): a 0.5× rescale of an already-tuned lr is slightly worse.

Neither granularity performs per-task allocation, which is the thing the mechanism exists to do.

## 2. The trajectories show there is almost nothing to win (`trajectories.py`, seed 42)

All 25 cells reproduced their ledger rows exactly (0 mismatches), which validates the re-run.

**Task 0 retention is perfect for every cell**: 0.999 at every checkpoint, drop +0.0000, mechanisms and
baselines alike. Split-MNIST task-IL forgetting is not a task-0 phenomenon at a tuned point.

Per-task accuracy after the final task:

| cell | T0 | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|
| naive | 0.999 | **0.953** | 0.987 | 0.991 | 0.972 |
| ewc | 0.999 | 0.967 | 0.989 | 0.991 | 0.971 |
| er | 0.999 | 0.989 | 0.997 | 0.996 | 0.994 |
| plast-neuron | 0.999 | 0.952 | 0.986 | 0.991 | 0.971 |

Everything is decided on task 1 (classes 2/3, the hardest pair). Tracking it through training:

| | after T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| naive | 0.972 | 0.949 | 0.950 | 0.953 |
| er | 0.993 | 0.989 | 0.985 | 0.989 |

**ER's advantage is mostly better ACQUISITION, not better retention.** It reaches 0.993 on task 1 at
the moment of learning against naive's 0.972 — a 0.021 gap before any forgetting can occur — and then
decays 0.004 where naive decays 0.019. Replay in task-IL is acting substantially as extra data /
regularisation, not only as an anti-forgetting device.

**The ceiling argument.** naive's total forgetting is 0.0061, so a *perfect* retention mechanism could
reach at most 0.9784 + 0.0061 = 0.9845 — still 0.010 below ER. A plasticity gate protects old weights;
it cannot make the current task fit better. So in this regime the mechanism is competing for a
0.006 prize with a ±0.002 measurement floor, and could not have closed the ER gap even in principle.
That is a limitation of the regime as a testbed, and it should be stated alongside the negative
result: this experiment has low power to detect a retention win, and none to detect an acquisition one.

## 3. The masked training loss is inert in task-IL (it is a class-IL lever)

`naive_unmasked` (full 10-way CE training, task-IL 2-way masked eval) = 0.9769±0.0028 vs masked
`naive` 0.9784±0.0016 — inside the noise floor. In pt3 Iteration 5 the masked loss was worth
0.198 → 0.389 on class-IL because it suppresses the 10-way output competition that dominates class-IL
forgetting; task-IL eval removes that competition by construction, so the lever has nothing to pull.

`output_masking` cannot express "unmasked train + masked eval"; the arm uses a shim that no-ops
`MaskedCE.pairs` (documented plain-CE behaviour) for one `cl_train` call.

**A val-stage reading that did not survive**: during tuning, naive forgot 0.0437 while the mechanism
arms read 0.002–0.005, which looked like a 10× retention effect. It did not reproduce on test (naive
0.0061, plast-neuron 0.0063). The val figure is specific to the seed-7 sequence and split, and
forgetting is far noisier than mean final accuracy. Do not read forgetting off the tuning stage.

## 4. The dead-gate controls, and where rule #10 does and does not bite

All eight dead controls land within ±0.0001 of their plain baselines, and the naive-backbone ones are
**bit-exact to naive on every seed** despite constructing the modulator, filling a 1000-sample
reservoir, sampling it every step and running the full lookahead. `P` is zero-init (no random draws),
the meta buffer uses python `random.choices` while DataLoader shuffling runs off the torch generator
(disjoint streams), and the meta-loop consumes no torch RNG.

So rule #10's RNG shift does not reach this path and `naive` happens to be a valid baseline here —
knowable only because the control was run, and the argument does not transfer to the gain path (whose
P lives in `model.parameters()`). The controls remain load-bearing for a different reason: they are
what makes `−0.0006` interpretable as "the gate did nothing" rather than "the arm was constructed
differently".

## 5. EWC: λ inert in tuning, yet a small real effect at 3 seeds

Val sweep at naive's point: λ ∈ {0.1, 1, 10, 100} → 0.9762 / 0.9764 / 0.9766 / 0.9769 (span 0.0007,
a tenth of the noise floor), λ=1000 → 0.0927 (chance). By val alone, λ=100 is an argmax over noise.

On test against an **RNG-matched λ=0 control** (Fisher pass runs, penalty off): λ=0 = 0.9786±0.0017 ≈
naive, λ=100 = 0.9821±0.0011 — **+0.0035, positive in all 3 seeds**. The Fisher pass alone does
nothing; the penalty carries a small real effect, invisible during tuning because it is half the
1-seed noise floor. The trajectories localise it: EWC's gain is almost entirely task 1 (0.967 vs
naive's 0.953), the same task that decides everything else.

Lesson: a 1-seed val sweep can only *bracket* a hyperparameter (it did correctly find the λ=1000
cliff); "inert on val" is not "inert".

**EWC+ER has no usable λ at a tuned ER lr=0.1**: λ=0.1 → 0.9921, λ=1 → 0.9909, λ ≥ 10 → chance. The
selected λ=0.1 is the "off" end (as λ→0 the arm converges analytically to ER). On test `ewc_er`
0.9936±0.0007 is marginally *below* `er` 0.9946±0.0004.

## 6. Tuning notes

Selected: naive 3e-3/ep5 (val 0.9761), naive_unmasked 3e-3/ep5 (0.9776), er 1e-1/ep5 (0.9909).
`TUNED_MAIN` has no `("taskil", *, "sgd")` entry, so every arm was swept from scratch.

**ER's grid needed an upward extension.** It first selected the grid top (1e-1) with val monotone
increasing in lr at every epoch budget — the truncated-grid signature. Extending to {3e-1, 1.0}
resolved it: 3e-1 plateaus (0.9902), 1.0 diverges to chance, so 1e-1 is a genuine interior maximum.
Only ER was extended; naive and naive_unmasked turn over inside the grid and collapse at 1e-1
(0.9332, 0.7076), so upward cells there are strictly worse. Same asymmetry pt3_retry used.

**Grid-edge warnings are not all equal.** Four fired that were correctly *not* acted on: `ep=5` fires
on nearly every selection because the tie-break prefers the cheapest budget among noise-floor ties;
`ewc_er` λ=0.1 walks toward an analytically known limit; and both neuro_lr grids span 0.0013–0.0017
(a quarter of the noise floor) while trending to **opposite** edges (neuron→1e-2, synapse→1e-4) — the
signature of noise, not a mechanistic optimum. Extend only when the trend is real and unresolved.

The gate diagnostics later confirmed both neuro_lr readings independently: the neuron gate saturates
open regardless of neuro_lr, and the synapse gate's floor selection really is "off" (|α−0.5|=0.0007).

## 7. Pattern across the whole study

Every hyperparameter belonging to a **mechanism** — EWC's λ, both neuro_lrs — is inert within tuning
noise or selects its own off-switch. Both hyperparameters belonging to the **optimizer** — lr and
epochs — move results by 0.02 to 0.9. The one arm that moves the number is replay, and even that is
mostly acquisition rather than retention.

## Limits

- SGD main net only (requested). The pt5 plasticity meta-optimizer is hardcoded Adam in the frozen
  `train.py`, so the neuromod net is Adam at the tuned `neuro_lr`.
- `neuromod_plasticity_init` held at 0.5 (`results/pt5_plast_init.py` established it as the SGD
  standalone optimum). The gate diagnostics show this init is itself a handicap the per-neuron gate
  spends its capacity undoing — an init sweep *at a tuned main lr* is the untested cell.
- `bufown` is not expressible for plasticity (see below), so four distinct arms were run, not five.
- Gate stats and trajectories are seed 42 only (shape, not reported numbers); every reported
  accuracy is 3 seeds.
- Task-IL on Split-MNIST at a tuned point has ~0.006 of forgetting available — a weak testbed for a
  retention mechanism, as §2 argues.
- ORACLE task id at train and eval; task-IL supplies it anyway, so this is not the class-IL oracle
  caveat, but the gate is task-conditioned by construction.

## Appendix: `bufown` is not expressible for plasticity

`neuromod_er_task_id` reaches two places in `cl_train`: `er_task_id_on`, which is replay-gated and so
cannot fire on a naive run, and `meta_task_id_on`, read *only* inside `if gain_meta_replay_on` — the
**gain** meta-loop. Plasticity's standalone meta-loss (train.py ~1332) builds its batch from the
buffer and gates it with the current task's `factors`, consulting neither. It was run as a real
3-seed row to demonstrate rather than assert the identity: all six runs match `bufcur` to the
ledger's precision. This pins down the mechanism behind CLAUDE.md's existing note that "`cur` is the
accurate label" for this arm.
