# SPEC: Prototype Iteration pt4 — every neuromod mechanism in the STANDARD learning regime

## Context
pt2 (Iterations 1-4) and pt3 (Iterations 5-10) ran nine neuromodulation mechanisms on
**class-IL Split MNIST**. All rejected: no variant beats Naive by >=5pts standalone, none
complements ER by >=2pts (Iter 10 closest at +0.018). See `prototype/iteration-notes.md`
("pt3 SUMMARY"). That closes the *continual-learning* front (project goal #1).

pt4 turns to the **second project front** (CLAUDE.md goal #2, so far only tested for the
default activation-gain modulator): **does each neuromodulation mechanism improve, preserve, or
degrade plain MNIST accuracy vs a vanilla MLP, in a standard (single-task) learning regime?**
This is the natural standard-learning analogue of the pt2/pt3 CL sweep: same mechanisms, same
MLP, different regime. It is a comparative *study*, not a search for a new mechanism, so it
introduces NO new neuromod variant (rule 8 / iteration discipline is satisfied: zero new
mechanisms).

## The governing classification (the intellectual content of pt4)
Standard learning is **one stationary task** (full MNIST, all 10 classes always present, no task
boundaries, no task sequence). Several pt2/pt3 mechanisms are *intrinsically continual* and have
no well-defined single-task form. pt4 therefore splits the nine mechanisms into two groups, and
the split is itself a reportable finding.

### Group R — runnable and meaningful in standard
These act on the forward pass or the per-task optimization, so they have a direct single-task
form. Run all of them.

| # | mechanism (iteration) | target | standard form |
|---|------------------------|--------|----------------|
| R1 | activation gain (sprint / Iter "0") | `activation` | FiLM `(1+m)⊙h` on hidden units, trained by backprop |
| R2 | weight mask (Iter 2) | `weight_mask` | per-synapse mask `M⊙W` on net.2, trained by backprop |
| R3 | logit calibration (Iter 6) | `logit` | per-sample FiLM on the 10 logits, trained by backprop |
| R4 | plasticity / meta-LR (Iter 1) | `plasticity` | per-neuron LR gate via the lookahead meta-gradient (SGD main net) |
| R5 | importance gating (Iter 7) | `importance` | online omega=sum(grad^2) gates per-param LR via grad-hooks |

### Group N — not definable (or degenerate) in single-task standard
Report these as **N/A by construction** with the reason; do NOT manufacture a degenerate number.
The "why" is the result.

| mechanism (iteration) | target | why N/A in standard |
|------------------------|--------|----------------------|
| weight_mask + drivers (Iter 3) | `weight_mask`+driver | surprise/uncertainty/activation_stats are cross-task novelty/retention signals fed lag-1 by the CL loop; on one stationary task the signal is degenerate and the mask reduces to R2 |
| stateful / GRU (Iter 4) | `weight_mask` stateful | the GRU state tracks cross-task dynamics and is never reset between tasks; with one task there is nothing to track -> reduces to R2 |
| task-inferred routing (Iter 8) | `task_route` | routing selects among >=2 tasks; standard has exactly one task -> routing is undefined / trivial |
| logit + recency (Iter 9) | `logit`+`recency` | the recency driver is per-class presence across tasks; all classes are present at all times -> constant driver -> reduces to R3 |
| consolidation (Iter 10) | `consolidation` | boundary detector + EWC anchors fire at task boundaries; there are none -> no anchor ever fires -> reduces to vanilla |

## Goal
For every Group-R mechanism, report **vanilla MLP vs neuromod MLP** test accuracy on full
MNIST, mean +- std over 3 seeds. The question is preserve / improve / degrade, NOT "beat a
baseline by X points". Neuromodulation passing here means it does **not materially hurt** standard
accuracy (the precondition for using it as a CL add-on without a standard-accuracy tax).

## Methodology — non-negotiable
1. **Standard regime only.** Full MNIST, `--standard`. No Split MNIST, no CL methods.
2. **Frozen shared config.** Use the existing tuned standard config `BEST_STANDARD_VANILLA`
   (lr=3e-4, epochs=20, batch=64), the SAME config behind the published `standard_mnist_table.md`
   (vanilla 0.9796 +- 0.0008, activation-gain 0.9806 +- 0.0006). Every Group-R mechanism is
   evaluated at this config so the comparison is apples-to-apples with the existing baseline. We
   do NOT re-tune per mechanism: the question is "at the vanilla-optimal hyperparameters, does
   adding the modulator change standard accuracy?", which is exactly the right standard-learning
   question and keeps the tuning budget identical (rule 3) and zero new test-set tuning (rule 1).
3. **3 seeds (42/43/44).** Report test accuracy mean +- std. Standard tuning, where referenced,
   is on the held-out 10k val split of the MNIST *train* set, never the test set (rule 1).
4. **`--use-neuromod` OFF must reproduce vanilla** numerically (carried rule).
5. **No hardcoded hyperparameters** in training code; route through `configs.py` (rule 7).
6. **No new neuromod mechanism** (rule 8). pt4 only runs existing targets in the standard loop.
7. **The plasticity (R4) confound, stated explicitly.** The plasticity lookahead trains the main
   net with **plain SGD** (the Adam-moments caveat in CLAUDE.md), whereas the vanilla standard
   baseline uses Adam. So R4 must also report a **plain-SGD vanilla reference** at the same config,
   and the R4 verdict is plasticity-SGD vs vanilla-SGD (the modulator's effect), with Adam-vanilla
   shown only for context. Do not compare meta-LR-SGD directly against Adam-vanilla.
8. **No em dashes** in any output (commas, parentheses, separate sentences instead).

## Implementation
- `train_standard` currently builds R1/R2/R3 directly (they are forward-graph modulators wrapped
  by `_build_model`; the single Adam optimizer over `model.parameters()` trains the modulator).
  Verify R1/R2/R3 run unchanged; if any needs a flag it is added to `configs.py` only.
- R5 (importance) needs `_install_importance_gates(model, lambda)` installed in `train_standard`
  before the epoch loop and removed after (mirror `cl_train`). Pure addition, behind
  `target=importance`.
- R4 (plasticity) needs a standard lookahead loop in `train_standard` mirroring
  `_plasticity_train_task` (SGD main net + Adam modulator, commit fast weights). Add an
  `optimizer` field to `StandardConfig` so the SGD-vanilla reference is config-selected, not
  hardcoded.
- Runner script `results/pt4_standard.py` (mirrors `results/iterN_*.py`): loops Group-R x 3 seeds,
  prints a vanilla-vs-neuromod table; log to `results/pt4_standard.log`.

## Deliverables
- Code changes (single revertable commit, message `pt4: <summary>`).
- `iteration-notes.md`: a pt4 section with the Group-R / Group-N classification, the
  vanilla-vs-neuromod table, and a one-line conclusion per mechanism (improve / preserve /
  degrade).
- Update `results/standard_mnist_table.md` (or a pt4 table) with all Group-R rows.
- Update the `## Specs` block in `CLAUDE.md` to list pt4 as governing current work; add a
  "Known gotchas" line for any wiring lesson.

## Interpretation rule (per mechanism)
At the frozen config over 3 seeds, vanilla = 0.9796 +- 0.0008:
- **improves**: neuromod mean exceeds vanilla beyond the combined std.
- **preserves**: within ~1 std of vanilla (the target outcome; neuromod is "free").
- **degrades**: neuromod mean is materially below vanilla beyond the combined std.
A clean "preserves" across Group R is the expected and acceptable result (neuromod adds capacity
without a standard-accuracy tax). A degrade is a real finding (the mechanism costs standard
accuracy and that cost must be weighed against any CL benefit, of which pt2/pt3 found none).

## Out of scope
- Re-tuning per mechanism / wider sweeps (use the frozen vanilla-optimal config).
- Any Split MNIST / CL run (that is pt2/pt3, complete).
- New architectures, Permuted MNIST, repo scaffolding (post-iteration migration, unchanged).
- Inventing a new neuromod mechanism or a single-task form for the Group-N mechanisms.
</content>
</invoke>
