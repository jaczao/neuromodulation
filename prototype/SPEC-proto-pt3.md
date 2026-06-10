# SPEC: Prototype Iteration pt3 — retry every mechanism aimed at the actual bottleneck

## Context
SPEC-proto-pt2 ran four neuromodulation iterations on MLP + Split MNIST (class-IL):
plasticity (1), weight mask (2), driver comparison (3), stateful modulator (4). **All four
rejected at ~ Naive (0.1979 ± 0.0003)**, every debugging checklist clean. The sprint's
activation-gain variant (0) also did not help. See `prototype/iteration-notes.md`.

The pt2 "failure across all four" stop condition was reached and the framing was discussed.
pt3 is the agreed continuation: retry the mechanisms with changes that target the **one**
reason they all failed.

## The single diagnosis that governs pt3
Every pt2 mechanism acts on a **hidden layer** (its activations, its learning rates, or its
weights). But catastrophic forgetting on **class-IL** Split MNIST is dominated by the **shared
output head's logit competition** between tasks (van de Ven & Tolias 2019): one 10-way softmax,
no task ID, training on 2 classes at a time with no old-class negatives in the loss, drives the
head and shared features toward the most recent task. A hidden-layer neuromodulator cannot reach
that. Positive control in our own data: **ER (replay) = 0.9023**, every hidden-layer neuromod
variant ~ 0.198 (chance-level retention of the last task only). The weight_mask diagnostic also
showed masks that *do* differentiate across tasks yet still fail, because they only gate one
hidden layer.

**Therefore the governing principle of pt3:** every retry must either (a) **reach the output
head / the logits**, or (b) supply a genuine **retention/importance signal** (what to protect),
or (c) change the regime. Re-running any mechanism unchanged on a hidden layer is out of scope:
we already know the answer.

## Repo housekeeping (once, before the first pt3 iteration)
- This file is `SPEC-proto-pt3.md` in `prototype/`. `SPEC-proto-pt1.md` (sprint) and
  `SPEC-proto-pt2.md` (iterations 1-4) remain historical.
- Continue appending to the existing `prototype/iteration-notes.md` and its running results
  table. pt3 iterations continue the numbering from pt2 (next is Iteration 5).
- Do NOT scaffold the definitive repo, add new architectures, or add new benchmarks. Stay in
  the `prototype/` layout. Migration happens only after pt3, per pt2's "after all iterations".
- Update the `## Specs` block in `CLAUDE.md` to list pt3 as the governing SPEC for current work.

## Goal
Find a neuromodulation mechanism that **clearly beats Naive standalone (≥5 absolute points)**,
OR that **clearly complements ER** (neuromod+ER beats ER), on class-IL Split MNIST with the MLP.
The mechanism comparison is itself the contribution even if none wins outright.

## Methodology — non-negotiable (carried from pt2, plus the pt3 additions)
1. **Change one substantive thing at a time.** Never combine two new mechanisms in one
   experiment (the head-reaching change IS the one substantive change per iteration).
2. **Keep all baseline numbers frozen** from the sprint and never re-tune them. References:
   - Naive: **0.1979 ± 0.0003** (forgetting 0.7979)
   - ER (buffer=1000): **0.9023 ± 0.0039**
   - EWC: 0.2014 (context only; known to fail class-IL)
3. **Same Split MNIST setup** (5 tasks × 2 classes, same order, same metrics, same protocol) as
   the sprint, unless an iteration explicitly changes the regime (e.g. the task-IL diagnostic),
   in which case that change is the iteration's single substantive change and is labelled as
   such.
4. **`--use-neuromod` OFF must still numerically reproduce vanilla.** Preserve every iteration.
5. **Tune only on the validation sequence** `make_sequence(7)`, never on the test sequence or the
   official test set. Same tuning budget shape as the sprint (2×2 lr × epochs grid) per
   iteration. No expanding sweeps to rescue a failing method.
6. **Report mean ± std over 3 seeds (42/43/44)** for any final number. Run the validation-sequence
   sanity/selection run first.
7. **No hardcoded hyperparameters** in training code; everything routes through `configs.py`.
8. **No em dashes** in any output (commas, parentheses, separate sentences instead).
9. **Run the pt2 debugging checklist before declaring any iteration a failure** (modulator output
   distribution, OFF parity, gradient flow into the modulator, LR ratio, where modulation is
   applied, capacity, init). A clean negative result is reportable; a silently broken one is not.

### NEW pt3 rule — every iteration reports BOTH comparisons
For every pt3 iteration, run and report **two** comparisons, not just one:
- **(A) Standalone:** `neuromod + naive`  vs  **Naive** (0.1979). Tests whether the mechanism
  reduces forgetting on its own.
- **(B) Complementarity:** `neuromod + ER`  vs  **ER** (0.9023). Tests whether the mechanism adds
  anything on top of the best baseline (replay handles the head; can neuromod still help the
  representation, sample efficiency, or forgetting beyond what ER already gives?).

Both rows go in the results table for every iteration. ER composition must use the **frozen ER
config** (buffer=1000, lr/epochs as in `BEST_CL_ER`); do not re-tune ER. This means the
mechanisms must **compose with the ER training loop**, not only the naive loop (an implementation
requirement for pt3, since pt2 wired plasticity/weight_mask/stateful to naive only).

## Iteration order
Run sequentially. Move on only after the current iteration is fairly tested AND the debugging
checklist is clean. **Iteration 5 (diagnostic) runs first and gates the rest.**

### Iteration 5 — Diagnostic: confirm the output head is the bottleneck
**Not a new mechanism; a regime/verification iteration.** Cheapest, highest-information step.
**What to implement.** Run the existing pt2 mechanisms (at minimum weight_mask, ideally also the
sprint activation gain) in a **task-IL / masked-output** setting: at train and eval, only the
current task's 2 logits are active (mask the other 8 logits out of the softmax). Equivalently,
extend the weight_mask target to also mask `net.4` (the 10×400 output weights).
**Why.** If forgetting drops sharply once the head competition is removed, that nails the
diagnosis and justifies every head-reaching retry below. If it does NOT drop, the diagnosis is
wrong and pt3 must be re-planned before building anything.
**Report.** Forgetting and avg final acc in the masked-output regime vs the class-IL numbers, for
the same mechanisms. (No accept/reject; this is a decision gate.)
**Files.** `train.py` (output-logit masking option, behind a config flag), `configs.py`,
`iteration-notes.md`. Keep class-IL the default; the mask is opt-in for this diagnostic.

### Iteration 6 — Activation gain on the logits (retry of mechanism 0)
**What to implement.** Move FiLM-style gain from hidden activations to the **output logits**: a
context-driven additive bias and/or scale on the 10 logits (learned **logit calibration** to
counteract the recency bias). Pair with a retention term (see Iteration 8's self-distillation, or
Iteration 5's masked loss) since trained on the current task alone the calibration just favors
current classes.
**Accept (A) or (B).** See accept rules below.
**Honest note.** Hidden-layer gain is NOT to be re-run unchanged. The logit version is the retry.

### Iteration 7 — Plasticity gating on the output head (retry of mechanism 1)
**What to implement.** Extend the plasticity (per-neuron/per-output) gradient gate to `net.4` and
have it learn to **not update the logit rows for classes absent from the current batch** ("don't
push down old-class logits when they aren't present"). This is the cheapest retry with a real
shot. Second sub-option: feed the modulator a **per-weight importance estimate** (Fisher/SI-style)
so it freezes weights important to past tasks (neuromodulated importance gating), trained via the
existing lookahead.
**Files.** `neuromod.py` (output-head plasticity gate + optional importance driver), `train.py`
(extend `_plasticity_train_task`; compose with ER), `configs.py`.

### Iteration 8 — Hard, all-layer, task-inferred weight masks (retry of mechanism 2)
**What to implement.** Three changes together (this is the single coherent "hard routing" change):
(a) **harden** the mask toward binary (Gumbel-sigmoid / straight-through) so tasks get near-disjoint
subnetworks instead of soft overlapping ones; (b) **extend to all layers including `net.4`** so the
disjointness reaches the output weights; (c) **infer task/group from context** so the right
subnetwork is selected at test without a task ID. This is HAT/PackNet recast for class-IL with
inferred task. **Measure the task-inference accuracy directly** (a single MNIST digit may not
identify its task reliably; if inference is poor, routing cannot work and that is the finding).
**Files.** `neuromod.py` (hard mask + task-inference head), `model.py` (mask all `ModulatedLinear`
layers), `train.py`, `configs.py`.

### Iteration 9 — Retention/importance drivers (retry of mechanism 3)
**What to implement.** Replace the pt2 novelty drivers (surprise/uncertainty/activation_stats,
which told the modulator *that* things changed, not *what to protect*) with
**retention/importance drivers**: per-parameter importance, a per-class "has this class been seen
/ does it own this logit" estimate, and an explicit **task-change detector**. Evaluate these
**on top of the best head-reaching target** from Iterations 6-8, not on the dead hidden-layer
mask. All drivers detached, as in pt2.
**Files.** `neuromod.py` (new driver computations), `train.py`, `configs.py`.

### Iteration 10 — Stateful boundary/consolidation controller (retry of mechanism 4)
**What to implement.** Use the GRU state for an **action**, not just tracking: detect task
boundaries from surprise/dynamics (no task ID) and trigger a **consolidation event** (snapshot +
freeze / importance-gate) on the output head; carry a running "which classes seen" summary in the
state to drive the head-level gating from Iterations 6-8. On its own the GRU did nothing; as a
boundary/consolidation controller for the head-reaching mechanisms it may help.
**Files.** `neuromod.py` (boundary detector + consolidation hook), `train.py`, `configs.py`.

## Cross-cutting levers (the actual content; referenced by the iterations above)
- **A. Reach the output head.** Every retry is a variation on this. It is the change that matters.
- **B. Output-gradient / masked-loss gating.** Cheapest real win; natural for the plasticity target.
- **C. Task inference from context.** The legal class-IL substitute for HAT's task input; enables
  disjoint routing without a task ID. Always measure inference accuracy.
- **D. Self-distillation (LwF-style) as the retention signal the modulator scales.** No buffer
  needed (regularizes current data through a frozen old-model snapshot); "modulate distillation
  strength per context" keeps it a neuromodulation story. The buffer-free analogue of what ER does
  for the head.
- **(Out of scope as a lever to invent here: replay itself. ER is the frozen baseline in
  comparison (B); we do not build a new replay method.)**

## Accept / reject rules (per iteration)
A pt3 iteration is **accepted** if EITHER:
- **(A) Standalone:** beats Naive avg final accuracy by **≥5 absolute points** over 3 seeds; OR
- **(B) Complementarity:** `neuromod+ER` beats ER by **≥2 absolute points**, OR matches ER with
  **materially less forgetting**, over 3 seeds.

**Reject** when, after a fair sweep at the sprint budget, neither (A) nor (B) holds AND the
debugging checklist is clean. **Record both comparisons regardless** of outcome; the comparison
across mechanisms (and standalone-vs-complementary) is the contribution.

## Per-iteration deliverables
- Code changes for the retry (single, revertable commit, message `iter<N>: <summary>`).
- A note in `iteration-notes.md`: what was tried, both comparison results, checklist outcome,
  decision (accepted / rejected / moved on).
- Results-table rows for **both** comparisons, e.g.:
  `iter | mechanism | target | standalone acc±std | +ER acc±std | beats Naive? | beats ER?`
- One-line status at completion:
  `Iteration N: <accept|reject>, standalone=X.X±Y.Y (beats_naive=<y/n>), +ER=X.X±Y.Y (beats_er=<y/n>)`.

## Stop conditions
- **Success.** Any iteration satisfies (A) or (B), consistent across 3 seeds. Keep that mechanism;
  do NOT halt, continue the remaining iterations as additional comparisons (the comparison between
  mechanisms is the contribution).
- **Failure across all pt3 iterations.** All complete, checklists clean, none satisfies (A) or (B).
  Do NOT add more iterations ad hoc. A clean negative result across the head-reaching design space
  is a valid finding. Pause and discuss framing with the supervisor before continuing.

## Honest caveats (state these in the writeup)
- Several retries recast known CL ideas (HAT, masked/expanded-output loss, LwF, bias correction)
  as neuromodulation. The framing question for the supervisor: is the thesis "is neuromodulation a
  useful lens" (then this is the point) or "a novel mechanism" (then novelty vs reframing must be
  stated explicitly)?
- Class-IL without replay or task ID is genuinely hard (EWC fails too). The realistic bar may be
  "beats Naive and the other neuromod variants, and/or complements ER", not "approaches ER".
- The task-inference idea (lever C) can fail outright if a single digit does not identify its task;
  that is itself a reportable result, so measure inference accuracy directly.

## Execution rules for Claude Code
- Read this SPEC at the start of each pt3 iteration session.
- Run **Iteration 5 (diagnostic) first**; it gates whether the rest is worth building.
- Implement one iteration end-to-end (both comparisons) before starting the next.
- Commit after each iteration with a clear message.
- If a debugging-checklist item triggers, fix it within the current iteration; do not move on with
  a known-broken implementation.
