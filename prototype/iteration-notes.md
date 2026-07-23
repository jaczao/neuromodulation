# Iteration notes (SPEC-proto-pt2)

Results and decisions for each neuromodulation iteration are appended below.

## Running results table

| iteration | variant | target | driver | avg_final_acc ± std | forgetting ± std | beats Naive? |
|-----------|---------|--------|--------|---------------------|------------------|--------------|
| (baseline) Naive | feedforward | (none) | none | 0.1979 ± 0.0003 | 0.7979 ± 0.0004 | (reference) |
| 1 | feedforward | plasticity (per-neuron) | none | 0.1992 ± 0.0000 | 0.7986 ± 0.0001 | no |
| 2 | feedforward | weight_mask (per-synapse, layer 2) | none | 0.1979 ± 0.0000 | 0.7982 ± 0.0002 | no |
| 3a | feedforward | weight_mask | surprise | 0.1977 ± 0.0008 | 0.7979 ± 0.0003 | no |
| 3b | feedforward | weight_mask | uncertainty | 0.1975 ± 0.0008 | 0.7982 ± 0.0002 | no |
| 3c | feedforward | weight_mask | activation_stats | 0.1978 ± 0.0004 | 0.7978 ± 0.0002 | no |
| 4 | stateful (GRU) | weight_mask | surprise | 0.1979 ± 0.0002 | 0.7973 ± 0.0002 | no |

---

## Iteration 1 — Plasticity modulation (per-neuron, backward-pass LR gating)

**Status:** `Iteration 1: reject, avg_final_acc=0.1992 ± 0.0000, beats_naive=no`

**What was implemented.** New `--neuromod-target plasticity`. `PlasticityModulator` (neuromod.py):
same architecture family as the gain modulator (signal net 784→64→k=8, plus per-layer
fixed random projection P_l: k→hidden_dim), but its output is a per-neuron gate
`α ∈ [0,1]` (sigmoid) rather than an activation gain. One α per hidden unit, broadcast
to that unit's incoming and outgoing weights:
`net.0.weight[i,:]*=α0[i]`, `net.2.weight[i,j]*=α1[i]·α0[j]`, `net.4.weight[:,j]*=α1[j]`;
output bias unmodulated. Init: signal-net final layer zero-init plus a constant logit bias,
so `α ≈ 0.95` (~full plasticity) for every unit at the start, regardless of P_l.

**Key implementation decision (deviation from the literal SPEC).** The SPEC specifies an
in-place hook: multiply `param.grad` by α between `loss.backward()` and `optimizer.step()`,
forward untouched. That cannot train the modulator: with the forward untouched, the
same-step loss is independent of α. The next-step loss does depend on α via
`W_new = W_old − lr·(α⊙g)`, but `optimizer.step()` writes W in place under `no_grad`, so
`W_new` is a fresh leaf (`grad_fn=None`) and the autograd edge α→W_new is severed →
`α.grad = None`. (User-confirmed analysis: the dependency is real mathematically but cut in
the recorded graph.) Fix used: a **lookahead / first-order meta-gradient** step in
`_plasticity_train_task` (train.py). Per batch: `g = autograd.grad(L, params).detach()`;
`W_fast = W.detach() − lr·(α⊙g)` (differentiable in α); `L_meta = CE(functional_call(model,
W_fast), batch)`; `L_meta.backward()` trains the modulator; then commit `W ← W_fast.detach()`
as the real gated-SGD step. Main net uses plain SGD (Adam caveat option (a)) so the inner
step is linear in α. The in-place `modulate_gradients` hook is still provided for interface
completeness but is NOT used in the trained path.

**Tuning.** Validation sequence `make_sequence(7)` only (never test). Same 2×2 budget shape
as the sprint: lr ∈ {0.01, 0.1} × epochs_per_task ∈ {5, 10}, modulator LR fixed 1e-3.
Best val config: lr=0.1, epochs_per_task=10 (acc 0.1999, all four within noise of each other).
Final numbers are 3 test seeds (42/43/44) at that config.

**Result.** Plasticity = 0.1992 ± 0.0000 vs frozen Naive 0.1979 ± 0.0003 (+0.13 pt, noise;
needs +5 pt). Per-task final accuracies are `[0, 0, 0, 0, ~1]` every seed: total catastrophic
forgetting, identical to Naive. A matched **Naive-SGD control** (same lr/epochs, SGD, no
neuromod) gives 0.1991–0.2003 across the grid, i.e. plasticity tracks it to within noise at
every point. This isolates the mechanism: the Adam→SGD switch contributes nothing, and the
plasticity gating itself contributes nothing.

**Debugging checklist (run before declaring failure):**
1. Output distribution — α not collapsed; spans up to [0.000, 1.000] per-neuron at the higher
   modulator LR. Healthy and expressive. ✓
2. OFF parity — `--use-neuromod` off is bit-identical to vanilla naive (0.199295 == 0.199295). ✓
3. Gradient flow into modulator — nonzero modulator grad norms (~2e-4 to 1.7e-3); the
   lookahead path does train it (and model params correctly get zero grad from the meta loss). ✓
4. LR ratio — modulator LR ∈ {1e-4, 1e-3, 1e-2}: 0.1996 / 0.1999 / 0.1997. No effect. At 1e-2
   the modulator uses the full [0,1] gate range, still no retention. ✓
8. Adam vs SGD — used SGD; matched Naive-SGD control equals plasticity. ✓
   Items 5/6/7 (layer placement, granularity, which params) not separately swept: item 1 shows
   the modulator already exercises the full gate range per-neuron with zero effect, so the cause
   is the mechanism, not a placement/granularity detail.

**Why it fails (mechanism, not implementation).** A feedforward modulator driven by the
current input and trained on a current-task lookahead loss has no signal about which neurons
matter for *past* tasks, so it cannot protect them. Backward-only LR gating can only
re-weight how fast the current task is learned; with no retention objective it converges to
"learn the current task," which is exactly what Naive already does. This is a clean negative
result and directly motivates Iteration 3 (drivers such as surprise/uncertainty/activation
stats), which is where a retention signal could enter the modulator's input.

**Decision:** reject, move on to Iteration 2 (per-synapse weight mask). Implementation kept
(it is needed as the "best target so far" candidate for the Iteration 3 driver comparison).

---

## Iteration 2 — Weight mask (per-synapse, second linear layer)

**Status:** `Iteration 2: reject, avg_final_acc=0.1979 ± 0.0000, beats_naive=no`

**What was implemented.** New `--neuromod-target weight_mask`. The 2nd linear layer
(net.2, 400×400) is replaced by `ModulatedLinear` (model.py), which computes
`y = (M ⊙ W) x + b` from an externally-supplied per-synapse mask `M ∈ [0,1]^{400×400}`
(and behaves exactly like `nn.Linear` when no mask is supplied: verified
`torch.allclose` parity). `WeightMaskModulator` (neuromod.py) is context-driven: batch-mean
image → signal net (784→64→k=8) → mask head. Full-rank head `Linear(k → 400·400)` outputs
all 160k mask logits directly (SPEC "try full-rank first"); a low-rank fallback
`M = sigmoid(bias + A·diag(g(s))·Bᵀ)` (rank r) is available via `neuromod_mask_rank`.
Mask head zero-init + logit bias → `M ≈ 0.99` (near-vanilla) for every synapse at init.
`WeightMaskMLP` threads the mask into net.2 each forward.

**No lookahead needed (contrast with Iteration 1).** The mask is in the forward graph, so
the task loss depends on it directly: `∂L/∂W = M ⊙ (∂L/∂y ⊗ x)` and `∂L/∂M = (∂L/∂y ⊗ x) ⊙ W`.
A single mask gates both the forward pass and the gradient at W. The modulator therefore
trains by ordinary backprop under one optimizer over net+modulator (no meta-gradient).

**Tuning.** Validation sequence `make_sequence(7)` only. Same 2×2 budget as the sprint
neuromod sweep: lr ∈ {3e-4, 1e-3} × epochs_per_task ∈ {5, 10}, full-rank, single Adam.
All four within noise (0.1997–0.1998). Best val: lr=1e-3, ep=5. Final = 3 test seeds.

**Result.** weight_mask = 0.1979 ± 0.0000 vs frozen Naive 0.1979 ± 0.0003: bit-identical,
not within-noise-identical. Per-task finals `[0,0,0,0,~1]` every seed: total forgetting.

**Debugging checklist (results/iter2_diag.py):**
1. Output distribution / task differentiation — the mask is NOT degenerate. At the swept
   shared LR it moves off 0.99 (by task 4: min=0.000, max=1.000, std=0.171, mean=0.954)
   and differs across tasks (max cross-task mask diff 0.28). At modulator-LR×50 it is
   strongly bimodal (mean 0.58, std 0.49) with cross-task diff up to 0.69. ✓ (healthy)
2. OFF parity — off → plain MLP; `ModulatedLinear` with no mask is `allclose` to `nn.Linear`. ✓
3. Gradient flow into modulator — mask_head and signal_net grads nonzero (signal_net is
   zero only at step 0 because mask_head is zero-init; it trains from step 1). ✓
4. LR ratio — modulator LR ×50 moves the mask far more (full bimodal, large cross-task
   diff) but forgetting is unchanged (avg 0.1997). ✓
10. Capacity — low-rank r=16 (LR×50): mask fully bimodal, cross-task diff 0.51, still total
    forgetting (avg 0.1997). ✓

**Why it fails (mechanism, not implementation).** Unlike Iteration 1 the mask trains and DOES
produce task-differentiated, full-range masks, yet forgetting stays total. Two reasons,
both structural to class-IL Split MNIST: (a) the mask only gates ONE hidden layer (net.2);
the first layer (net.0) and especially the shared output head (net.4) are unmasked and get
overwritten by each task; (b) class-IL forgetting is dominated by output-logit competition
between tasks (van de Ven & Tolias 2019), which a hidden-layer weight mask cannot touch. The
per-task masks are also soft and overlapping (mean ≈ 0.95 at the swept LR, i.e. most synapses
near 1 for every task), not disjoint task routing, so W is shared and overwritten regardless.

**Comparison Iter 1 vs Iter 2.** Both reject at ≈ Naive. Iter 1 (plasticity, backward-only)
could not even train without a meta-gradient and had no retention signal; Iter 2 (weight mask,
forward+backward coupled) trains naturally and learns task-differentiated masks, but masking a
single hidden layer cannot overcome the shared-head class-IL bottleneck. Neither isolates
catastrophic forgetting on its own. This motivates Iteration 3 (drivers): a retention-relevant
input signal (surprise / uncertainty) on top of the more-promising weight_mask target.

**Decision:** reject, move on to Iteration 3 (driver comparison). weight_mask is the
"most life" target so far (it at least learns structured, task-differentiated masks), so it is
the natural base target for the Iteration 3 driver comparison.

---

## Iteration 3 — Driver comparison (on the weight_mask target)

**Status:** `Iteration 3: reject (3a/3b/3c), best avg_final_acc=0.1978 ± 0.0004, beats_naive=no`

**Setup.** Drivers are applied on top of the **weight_mask** target (the "most life" target
from Iterations 1-2). Matched conditions per SPEC: target (weight_mask, layer 2, full-rank),
modulator architecture, and hyperparameters are all FIXED at the Iteration 2 best
(lr=1e-3, epochs_per_task=5); **only the modulator's input (the driver) changes**. So drivers
are NOT separately re-tuned (that would break the matched comparison); `driver=none` is the
Iteration 2 baseline (0.1979). Each driver: one validation-sequence sanity run (seed=7) then
3 test seeds (42/43/44). Code: `results/iter3_drivers.py`.

**Implementation.** The mask is computed *before* the main forward, but surprise/uncertainty/
stats are products *of* a forward, so each driver is fed as a **lag-1 detached control signal**:
computed from step t's loss/logits/activations (all `.detach()`), stored on the modulator
(`current_driver` buffer, `requires_grad=False`), and concatenated onto the batch-mean image
context to drive step t+1's mask. Verified no gradient path from any driver source into the main
loss (the driver buffer is a constant input; the modulator still trains via the mask in the
forward graph). The surprise EMA persists across task boundaries (never reset). Driver dims:
surprise 1, uncertainty 1, activation_stats 8 (per hidden layer × [L2 norm, mean, var, sparsity]).

**Results (avg_final_acc ± std over 3 test seeds; forgetting):**

| driver | avg_final_acc | forgetting | mean \|driver\| |
|--------|---------------|------------|-----------------|
| none (Iter 2)    | 0.1979 ± 0.0000 | 0.7982 | – |
| 3a surprise      | 0.1977 ± 0.0008 | 0.7979 ± 0.0003 | ~0.19 |
| 3b uncertainty   | 0.1975 ± 0.0008 | 0.7982 ± 0.0002 | ~0.04 |
| 3c activation_stats | 0.1978 ± 0.0004 | 0.7978 ± 0.0002 | ~12 |

All three are within noise of `none` and of Naive (0.1979). Per-task finals `[0,0,0,0,~1]`
every run: total forgetting. Best driver (3c, 0.1978) misses the +5pt bar (Iter1-2 best 0.1992
→ 0.2492) by ~5 points.

**3a Surprise.** `surprise = (loss - ema_loss).detach()`, EMA β=0.99, persisted across tasks.
mean |surprise| ≈ 0.19 (a live, varying signal). No effect: 0.1977 ± 0.0008.

**3b Uncertainty.** Mean predictive entropy `H(p) = -Σ p log p`, detached. mean |driver| ≈ 0.04
(small but nonzero; entropy collapses fast once a task is learned). No effect: 0.1975 ± 0.0008.

**3c Activation statistics.** Per hidden layer (×2): [mean L2 norm, mean, variance, sparsity],
detached, via forward hooks on the two hidden ReLUs. Richest driver, mean |driver| ≈ 12
(dominated by the L2-norm components). No effect: 0.1978 ± 0.0004.

**Debugging checklist.** Detachment verified (`current_driver.requires_grad=False`, no grad path
to driver source). Driver magnitudes confirm all three were live signals (not collapsed to zero),
so this is not a "driver wasn't computed" bug. Modulator still trains (mask in forward graph, as
Iteration 2). Hyperparameters fixed by design (matched comparison), so no LR sweep per driver.

**Why none helps (mechanism, not implementation).** The drivers carry *novelty/difficulty*
information (surprise spikes at task boundaries, entropy is high on unfamiliar inputs, activation
norms shift), but NOT *retention/importance* information ("which weights matter for past tasks").
A novelty signal tells the modulator *that* the input changed, not *what to protect*; and even a
perfectly-informed mask still only gates layer 2, leaving the shared output head to overwrite
(the class-IL bottleneck, van de Ven & Tolias 2019). So adding any of these inputs to the
weight_mask modulator cannot move the result. The comparison itself is the contribution: under
matched conditions, surprise ≈ uncertainty ≈ activation_stats ≈ none on class-IL Split MNIST.

**Decision:** reject all three, move on to Iteration 4 (stateful modulator). Per SPEC, the
winning driver feeds Iteration 4; since none won, Iteration 4 defaults to **surprise**.

---

## Iteration 4 — Stateful modulator (GRU)

**Status:** `Iteration 4: reject, avg_final_acc=0.1979 ± 0.0002, beats_naive=no`

**What was implemented.** `StatefulModulator` (variant=`stateful`): the feedforward signal path
of the weight_mask modulator is replaced by a `nn.GRUCell`. Pipeline each step:
`x = [batch-mean image (784), surprise (1)]` → `h ← GRUCell(x, h_prev)` → `Linear(h→k)` →
`mask_head(k → 400·400)` → `sigmoid(bias + logits)`, i.e. the GRU hidden state drives the same
per-synapse mask on layer 2 as Iteration 2. Run on the best target so far (weight_mask) with the
default driver (surprise, since none won in Iteration 3). Hidden state is a buffer persisted
across steps AND across task boundaries (never reset); it is detached each step (truncated BPTT
length 1, via a `clone()` so the in-place state update does not corrupt autograd), so the graph
stays bounded while the state still carries information forward numerically. Matched config
(lr=1e-3, ep=5). Code: `results/iter4_stateful.py`.

**Result.** stateful (GRU, h=64) = 0.1979 ± 0.0002, forgetting 0.7973 ± 0.0002 vs Naive 0.1979
and best-so-far 0.1992. Per-task finals `[0,0,0,0,~1]`. Forgetting is marginally lower than
Iterations 2-3 (0.7973 vs ~0.798) but noise-level, not "materially less," so the alternative
accept clause is not met either.

**Debugging checklist.** State persists across tasks by design (buffer, never reset; verified the
state evolves: h moved ~0.4-0.7 per step in a smoke test). Modulator trains (GRU + mask_head grads
nonzero from step 1; mask_head zero-init gives the usual one-step slow start). Hidden size 32 vs 64
give identical validation accuracy (0.1998), so this is not a GRU-capacity issue. Detachment /
bounded BPTT verified (no autograd in-place error after the clone fix). Parity: weight_mask off →
plain MLP; `ModulatedLinear` no-mask is `allclose` to `nn.Linear`.

**Why it fails (mechanism, not implementation).** A stateful modulator can track training dynamics
("how much is shifting") via its hidden state, but that is still only an *input* to a mask on one
hidden layer. Tracking "what has been learned" is not the same as *protecting old-class
separability in the shared output head*, which is the actual class-IL bottleneck. The state gives
temporal context but no lever on net.4, so the result is unchanged.

**Decision:** reject. All four iterations complete.

---

## Summary across all four iterations — STOP CONDITION REACHED

| iteration | mechanism | avg_final_acc | beats Naive (0.1979)? |
|-----------|-----------|---------------|------------------------|
| 1 | plasticity (per-neuron LR gating, lookahead) | 0.1992 ± 0.0000 | no |
| 2 | weight_mask (per-synapse, layer 2) | 0.1979 ± 0.0000 | no |
| 3a/3b/3c | weight_mask + surprise / uncertainty / activation_stats | 0.1975-0.1978 | no |
| 4 | stateful GRU + weight_mask + surprise | 0.1979 ± 0.0002 | no |

**All four iterations reject. Debugging checklists clean for each.** This is the SPEC
"Failure across all four" stop condition: do NOT add more iterations or mechanisms ad hoc. A clean
negative result across this design space (activation gain from the sprint, plus plasticity, weight
mask, drivers, and a stateful modulator) is itself a valid finding. **Pause and discuss framing
with the supervisor before continuing.**

**The single unifying explanation.** Every mechanism here acts on a *hidden* layer (its
activations, its learning rates, or its weights), but catastrophic forgetting on **class-IL**
Split MNIST is dominated by the **shared output head's logit competition** between tasks (van de
Ven & Tolias 2019): with no task ID and one 10-way softmax, training on 2 classes at a time, with
no old-class negatives in the loss, drives the head and shared features toward the most recent
task. A hidden-layer neuromodulator cannot reach that. The contrast in the data: ER (replay, which
puts old-class negatives back in the loss) reaches 0.90, while every hidden-layer neuromod variant
sits at ~0.198 = chance-level retention of the last task only.

**Reportable contributions (Chapter 3) regardless of the negative outcome.**
- A clean, mechanism-by-mechanism negative result on class-IL Split MNIST with a fair, matched
  protocol and a debugging checklist clean for each.
- A precise diagnosis (the shared-head bottleneck), with ER as the positive control that confirms
  the bottleneck is the output competition, not optimisation.
- The driver comparison under matched conditions (surprise ≈ uncertainty ≈ activation_stats ≈ none)
  as a standalone finding.

**Suggested next steps for the supervisor discussion (NOT to be executed ad hoc now).**
- Verify the diagnosis directly: extend the mask/gate to the output head (net.4), or run the same
  mechanisms in a **task-IL** (masked-output) setting, and check whether forgetting drops. If it
  does, that confirms the head is the bottleneck.
- Reframe toward where neuromodulation could plausibly help: task-IL, or stacking neuromod with a
  replay/regularisation method that addresses the head (the "neuromod + best baseline"
  complementarity row in CLAUDE.md), or a mechanism that acts on the output competition itself.
- Only then consider the architecture roadmap (GRU/CNN/ViT) and scaffolding.

---

# pt3 — retry every mechanism aimed at the output-head bottleneck (`SPEC-proto-pt3.md`)

## Iteration 5 — Diagnostic: is the shared output head the class-IL bottleneck?

**Status:** `Iteration 5: decision gate — head_is_bottleneck = YES. Proceed with Iterations 6-10.`

**What was implemented.** A reusable `output_masking` config option (`none` | `loss` | `taskil`)
plus a `MaskedCE` loss and a masked `evaluate(allowed=...)` in train.py. `none` = class-IL
(default, unchanged; parity verified). `loss` = mask the TRAIN loss to the current task's 2
classes (do not push absent-class logits down), class-IL eval over all 10. `taskil` = mask train
AND eval to the task's 2 classes (full task-IL). Lever B (`loss`) is reused by later iterations.

**Result (seed=42, test sequence, lr=1e-3, ep=5; single seed — this is a gate, not a 3-seed final):**

| mechanism | regime | avg_final_acc | forgetting |
|-----------|--------|---------------|------------|
| naive | none (class-IL) | 0.1977 | 0.7981 |
| naive | loss (masked train, class-IL eval) | 0.3894 | 0.5355 |
| naive | taskil (masked train + eval) | **0.9286** | **0.0695** |
| weight_mask | none | 0.1979 | 0.7984 |
| weight_mask | taskil | 0.8685 | 0.1289 |
| activation_gain | none | 0.1968 | 0.7970 |
| activation_gain | taskil | 0.9009 | 0.0975 |

**Gate decision: the shared output head IS the dominant class-IL bottleneck.** Removing the head
competition (task-IL) takes Naive from 0.198 to 0.929; forgetting collapses 0.798 -> 0.070 (a drop
of 0.729). This is the positive confirmation that justified pt3: the pt2 mechanisms failed because
they act on hidden layers, one layer away from where the forgetting happens.

**Two findings that shape Iterations 6-10:**
1. **`loss` alone (masked training loss, class-IL eval) recovers a large but partial chunk**:
   0.198 -> 0.389 acc, 0.798 -> 0.536 forgetting. So "don't push down absent-class logits during
   training" (lever B, the core of Iteration 7) genuinely helps, but the eval-time 10-way
   competition still caps it. Conclusion: masked-loss alone will not fully solve class-IL; a
   head/eval-side mechanism (logit calibration in Iter 6, task inference in Iter 8) is needed for
   the rest of the gap. This sets a realistic standalone bar for Iter 7 (clearly beats Naive, but
   not ER-level on its own).
2. **In task-IL the hidden-layer neuromod mechanisms do not beat Naive** (weight_mask 0.869 < naive
   0.929; activation gain 0.901 ~ naive). Even with the head bottleneck removed they add nothing,
   and the mask slightly hurts. So the mechanisms were not silently helping the representation
   either; the head was the whole story.

**Decision:** gate passed, proceed. Iteration 6 (logit calibration), Iteration 7 (output-head
plasticity gating / masked-loss), Iteration 8 (hard task-inferred all-layer masks) are now
justified. Per the pt3 dual-comparison rule, Iterations 6+ report both `neuromod+naive vs Naive`
and `neuromod+ER vs ER`.

### Iteration 5 addendum — neuromod+ER vs ER under task-IL (results/iter5_er.py)

Frozen ER config (lr=3e-4, ep=5, buffer=1000), task-IL regime, seed=42, test sequence. Required a
fix: `MaskedCE` now masks **per sample** by label->task-pair (a single current-task mask would send
ER's replayed old-task samples' true logits to -inf). Per-sample masking is identical to per-task
for the naive loop, so the Iteration 5 naive/weight_mask/gain numbers above are unchanged (verified:
naive+loss reproduces 0.3894/0.5355).

| config | avg_final_acc | forgetting | vs ER |
|--------|---------------|------------|-------|
| ER (task-IL) | 0.9959 | 0.0025 | — |
| weight_mask + ER (task-IL) | 0.9953 | 0.0028 | -0.0006 |
| activation_gain + ER (task-IL) | 0.9954 | 0.0024 | -0.0005 |

Once the head bottleneck is removed AND replay is added, ER is at ceiling (~0.996) and the
hidden-layer mechanisms add nothing (marginally lower, within noise). This closes the iter5 picture:
the shared head was the whole story, and on top of replay the pt2 mechanisms are inert.

## pt3 running results table (dual comparison)

| iter | mechanism | target | standalone acc ± std | +ER acc ± std | beats Naive (0.1979)? | beats ER (0.9023)? |
|------|-----------|--------|----------------------|---------------|------------------------|---------------------|
| 6 | logit calibration (FiLM on logits) | logit | 0.3649 ± 0.0228 (logit+masked-loss) | 0.8964 ± 0.0073 | only via masked-loss, not the modulator | no (-0.006) |
| 7 | importance-gated plasticity (online omega) | importance | 0.1977 ± 0.0005 (naive); 0.3975 ± 0.0268 (+masked-loss) | 0.9035 ± 0.0037 | no (=Naive; +maskloss within noise of masked-loss) | no (+0.001) |
| 8 | task-inferred routing (simplified HAT / lever C) | task_route | 0.1990 ± 0.0003 (routing acc 0.20=chance) | 0.8840 ± 0.0089 (routing acc 0.89) | no (g forgets without replay) | no (-0.018; hard routing < ER soft classification) |
| 9 | retention driver (per-class recency) on logit calibrator | logit + recency | 0.1979 ± 0.0006 | 0.8948 ± 0.0033 | no (=Naive) | no (-0.008) |
| 10 | stateful boundary detector + EWC consolidation | consolidation | 0.1974 ± 0.0003 | 0.9205 ± 0.0074 | no (=Naive) | no, but closest (+0.018, just under +0.02 bar) |

## Iteration 6 — Logit calibration (FiLM on the output logits)

**Status:** `Iteration 6: reject, standalone=0.3649±0.0228 (beats_naive=via lever B only), +ER=0.8964±0.0073 (beats_er=no)`

**What was implemented.** New target `logit`: `LogitModulator` produces a context-driven per-sample
FiLM on the 10 output logits, `logits' = (1+γ(x))⊙logits + β(x)` (signal net 784→64→k, head k→2·10,
zero-init so γ=β=0 and it is identical to vanilla at init; parity verified). `LogitModulatedMLP`
wrapper applies it after the base MLP. It reaches the head directly (unlike pt2). Composes with the
naive and ER loops via the standard else branch. Per the SPEC, paired with a retention term
(`output_masking='loss'`, lever B). Class-IL, 3 test seeds. Code: `results/iter6_logit.py`.

**Results (class-IL, 3 seeds 42/43/44).**

| config | avg_final_acc | forgetting |
|--------|---------------|------------|
| Naive (frozen) | 0.1979 | 0.7979 |
| ER (frozen) | 0.9023 | — |
| naive + masked-loss (lever B, no neuromod) | 0.3777 ± 0.0331 | 0.5506 ± 0.0117 |
| logit, no retention | 0.1979 ± 0.0003 | 0.7974 ± 0.0003 |
| logit + masked-loss | 0.3649 ± 0.0228 | 0.5347 ± 0.0033 |
| logit + ER | 0.8964 ± 0.0073 | 0.0986 ± 0.0075 |

**Verdict (the controlled comparisons matter, not the raw accept box).**
- **(A) standalone:** `logit+masked-loss` (0.3649) technically clears "beats Naive by ≥5pts", but vs
  `naive+masked-loss` (0.3777) the modulator contributes **-0.013 (within noise)**. The **masked loss
  (lever B) does all the work; the logit modulator adds nothing** (slightly hurts).
- **logit with no retention = 0.1979, exactly Naive.** Confirms the SPEC prediction precisely: trained
  on the current task, the per-sample calibration just learns to favor the current classes (the very
  recency bias we wanted to counteract), so it nets to zero. A per-input calibration has no signal
  about old classes at test time.
- **(B) complementarity:** `logit+ER` (0.8964) is **below** ER (0.9023) by 0.006; no complementarity.

**Debugging checklist.** OFF parity holds (zero-init → vanilla; `logit+none` reproducing Naive to
±0.0003 corroborates the path is clean). Gradient flow OK (head trains immediately, signal net from
step 1; the usual zero-init slow start). The "no effect" of `logit+none` is not a bug, it is the
mechanism: a current-task-trained logit calibration cannot encode "protect old classes".

**Decision:** reject the logit modulator (no value over the retention term it is paired with; does
not complement ER). The masked-loss result (lever B roughly doubles Naive, 0.198 -> 0.38) is recorded
as a method finding, but it is NOT neuromodulation. This sharpens the bar for Iterations 7-8: a
useful mechanism must beat `naive+masked-loss` (~0.38) standalone, not just Naive. Proceed to
Iteration 7 (output-head plasticity gating, which IS a learned form of lever B and may go further).

## Iteration 7 — Importance-gated plasticity (online omega, all layers + head)

**Status:** `Iteration 7: reject, standalone=0.1977±0.0005 (beats_naive=no), +ER=0.9035±0.0037 (beats_er=no)`

**What was implemented.** New target `importance`: online per-parameter importance `omega` (running
sum of raw grad^2, never reset across tasks) installed as per-parameter autograd grad-hooks. Each
backward, the hook scales that param's gradient by `alpha_p = 1/(1+lambda*omega_p)` BEFORE the
optimizer step, then accumulates `omega += grad^2`. Params important to past tasks (large omega) are
frozen (alpha->0). Applied to ALL params including the head. omega=0 at start (alpha=1 = vanilla;
parity verified at lambda=0). Composes with any loop (naive, ER) since hooks fire during backward
regardless of method. This is online EWC/MAS recast as a hard per-parameter LR gate (the SPEC's
"neuromodulated importance gating"). No lookahead needed (the gate is a deterministic function of
the importance signal; the importance IS the retention signal iter6 lacked). Code:
`results/iter7_importance.py`. lambda tuned on the validation sequence (best lambda=10; all lambda
gave ~0.1997 val, i.e. tuning does not rescue it).

**Results (class-IL, 3 seeds, lambda=10).**

| config | avg_final_acc | forgetting |
|--------|---------------|------------|
| importance + naive (standalone) | 0.1977 ± 0.0005 | 0.7975 ± 0.0003 |
| importance + naive + masked-loss | 0.3975 ± 0.0268 | 0.5195 ± 0.0466 |
| importance + ER | 0.9035 ± 0.0037 | 0.0912 ± 0.0043 |

(reference: Naive 0.1979, naive+masked-loss 0.3777, ER 0.9023)

**Verdict.**
- **(A) standalone = 0.1977 = Naive.** Importance gating alone does NOT beat Naive. This reproduces
  the known EWC-fails-class-IL result (EWC was 0.2014) as an LR gate: protecting weights does not
  address the head's logit competition, which is where class-IL forgetting lives.
- **importance + masked-loss = 0.3975 vs masked-loss-alone 0.3777**: +0.02, but within one std
  (0.027/0.033), so at most a marginal, noisy improvement. Does not clear the iter6 bar convincingly.
- **(B) importance + ER = 0.9035 vs ER 0.9023**: +0.001, not >= 2pts. No complementarity.

**Debugging checklist.** Gates non-degenerate: at the best lambda=10, mean gate ~0.81, min ~0.001
(some params frozen, not all). Higher lambda over-freezes (min gate -> 0) but standalone acc stays
flat ~0.198, confirming the ceiling is the mechanism, not the tuning. Parity (lambda=0 == vanilla)
verified. omega accumulation / gating live (gates ramp with lambda). Clean negative.

**Decision:** reject. Weight-importance protection cannot fix class-IL (head bottleneck); marginal
and noisy on top of masked-loss; no complementarity with ER. Proceed to Iteration 8 (hard,
task-inferred, all-layer masks), the one remaining target-side idea that attacks the eval-time
competition via routing rather than weight protection.

## Iteration 8 — Task-inferred routing (simplified HAT / lever C)

**Status:** `Iteration 8: reject, standalone=0.1990±0.0003 (beats_naive=no), +ER=0.8840±0.0089 (beats_er=no, -0.018)`

**What was implemented (and the deliberate simplification).** A masked-loss main net (which already
supplies the per-task anti-forgetting that HAT's masks would) plus a `TaskInferenceNet` g(x) (5-way,
784->64->5). At eval each input is routed: t_hat = argmax g(x), output masked to task t_hat's classes,
then argmax. `g` is trained on the current task index. Two modes: **naive** (g trained sequentially,
no replay) and **er** (a shared reservoir buffer trains BOTH the main net and g, with g's targets for
buffered samples derived from their labels via label->task-pair). This isolates the binding constraint
of the SPEC's Iter 8 (lever C: infer the task from a single digit without a task ID) and measures g's
routing accuracy directly. Full per-task weight subnetworks (HAT/PackNet) were NOT built: the binding
constraint (test-time task inference) is shared with this version, and masked-loss already provides the
anti-forgetting, so this is the decisive, cheaper test. Code: `results/iter8_taskroute.py`.

**Results (class-IL, 3 seeds).**

| config | avg_final_acc | routing accuracy |
|--------|---------------|------------------|
| task_route + naive (standalone) | 0.1990 ± 0.0003 | 0.200 (= chance) |
| task_route + ER | 0.8840 ± 0.0089 | ~0.89 |

(refs: Naive 0.1979, naive+masked-loss 0.3777, ER 0.9023, task-IL oracle routing 0.9286)

**Verdict.**
- **(A) standalone fails at inference.** Without replay, g is itself a class-IL problem and
  catastrophically forgets: it routes everything to the most recent task (routing accuracy decays
  1.0 -> 0.5 -> 0.33 -> 0.25 -> 0.20 across tasks). CL acc = 0.199 = Naive. Lever C is blocked, exactly
  as the SPEC anticipated ("a single digit may not identify its task; that is itself a result").
- **(B) +ER is the sharp finding.** Replay fixes g's forgetting (routing accuracy ~0.89), BUT routed
  ER (0.8840) is BELOW plain ER (0.9023), delta -0.018. Hard routing turns g's ~11% soft errors into
  **unrecoverable** ones (a wrong task mask zeroes the true class), so it underperforms ER's direct
  soft 10-way classification. Task-inferred hard routing does not complement replay; it slightly hurts.

**Generalization to full HAT.** HAT's per-task weight subnetworks would face the same test-time
inference ceiling (~0.89 best case, with replay) and the same unrecoverable-misroute penalty, so the
negative conclusion is not an artifact of the simplification.

**Decision:** reject (neither A nor B). The eval-time competition cannot be fixed by hard routing in
class-IL: without replay inference forgets; with replay, direct classification beats routing. Proceed
to Iteration 9 (retention/importance drivers) and 10 (stateful), per the SPEC, though the pt3 picture
is now strongly negative.

## Iteration 9 — Retention driver (per-class recency) on the logit calibrator

**Status:** `Iteration 9: reject, standalone=0.1979±0.0006 (beats_naive=no), +ER=0.8948±0.0033 (beats_er=no, -0.008)`

**What was implemented.** Gives iter6's logit calibrator the retention signal it lacked: a per-class
presence EMA (presence_c = decayed recency of class c appearing, beta=0.95, persists across tasks) is
concatenated (detached) onto the logit modulator's input, so it can in principle learn to boost stale
classes. The presence is a clean recency signal (final value ~1 for the last task's classes, ~0 for
old ones). naive (standalone) and er (replay). Code: `results/iter9_recency.py`.

**Results (class-IL, 3 seeds).**
- (A) logit+recency + naive = 0.1979 ± 0.0006 = **Naive**. The recency driver does NOT rescue logit
  calibration. A global per-class bias cannot recover old classes because the bottleneck is also the
  drifted shared representation (old inputs no longer activate old-class structure), which no output
  bias can fix.
- (B) logit+recency + ER = 0.8948 vs ER 0.9023 (-0.008): slightly hurts, like the other head-bias
  attempts on top of ER.

**Decision:** reject. Replacing iter6's missing retention signal with a recency driver still fails:
the retention signal must re-supply old-class DATA/structure (replay), not just a recency hint.
Proceed to Iteration 10 (stateful boundary/consolidation).

## Iteration 10 — Stateful boundary detector + EWC consolidation

**Status:** `Iteration 10: reject, standalone=0.1974±0.0003 (beats_naive=no), +ER=0.9205±0.0074 (beats_er=no, +0.018 just under bar)`

**What was implemented.** A running surprise statistic (loss EMA) detects task boundaries with no task
ID (boundary when current loss > 2x EMA, with a 150-step cooldown); at each detected boundary the model
snapshots params and accumulates an importance anchor, adding an EWC penalty `0.5*lambda*sum omega*(theta-theta*)^2`
to subsequent losses (online EWC at detected boundaries). naive or er. lambda tuned on validation
(best=10). Code: `results/iter10_consolidation.py`. (Used a running-surprise state rather than the
pt2 GRU, since pt2 Iter 4 showed the GRU adds nothing; the binding question is whether
boundary-triggered consolidation helps.)

**Results (class-IL, 3 seeds, lambda=10).**
- (A) consolidation + naive = 0.1974 = **Naive**. EWC-style consolidation fails class-IL (the head
  bottleneck again). The boundary detector also **over-fires (~20 detections vs 4 true)**: within-task
  loss noise exceeds the threshold, so this is effectively frequent online-EWC anchoring, not clean
  boundary detection. Surprise does not cleanly segment tasks here.
- (B) consolidation + ER = 0.9205 vs ER 0.9023, **+0.018** (~2 sigma): the **largest positive
  complementarity delta in all of pt3**, but just under the +0.02 accept bar. "Online-EWC anchors +
  replay" gives a small stability bump on top of ER. Per the SPEC (no expanding sweeps to rescue a
  method), not chased further.

**Decision:** reject (A fails; B is the closest pt3 result but under the bar). Notably the one place
anything helped on top of ER is regularization-style consolidation (+0.018), consistent with replay
handling the head and a mild anchor adding stability.

---

# pt3 SUMMARY — failure across all pt3 iterations (STOP CONDITION REACHED)

| iter | mechanism | standalone (vs Naive 0.1979) | +ER (vs ER 0.9023) |
|------|-----------|------------------------------|--------------------|
| 5 | diagnostic: task-IL Naive 0.929 (forget 0.07) | (gate) confirmed head is the bottleneck | ER taskil 0.996, +neuromod inert |
| 6 | logit calibration | 0.3649 (= masked-loss alone; modulator ~0) | 0.8964 (-0.006) |
| 7 | importance-gated plasticity | 0.1977 (= Naive) | 0.9035 (+0.001) |
| 8 | task-inferred routing (HAT/lever C) | 0.1990 (routing forgets to chance) | 0.8840 (-0.018) |
| 9 | recency driver on logit calibrator | 0.1979 (= Naive) | 0.8948 (-0.008) |
| 10 | boundary-detected EWC consolidation | 0.1974 (= Naive) | 0.9205 (+0.018, closest) |

**All pt3 retries reject. Debugging checklists clean.** This is the SPEC "failure across all pt3
iterations" stop condition: do NOT add more iterations ad hoc; pause and discuss framing with the
supervisor. A clean negative across the head-reaching design space is a valid finding.

**What pt3 adds over pt2 (the contribution).**
- pt2 showed hidden-layer neuromodulation does not beat Naive on class-IL. pt3 confirmed WHY
  (Iteration 5: removing the shared-head competition takes Naive 0.198 -> 0.929) and then showed that
  **even mechanisms that reach the head/logits do not beat Naive standalone, nor complement ER**:
  - logit calibration (6) and recency-driven calibration (9): a per-input/global output bias cannot
    recover old classes once the shared representation has drifted; alone = Naive.
  - importance gating (7): weight protection fails class-IL (the EWC result, reproduced as an LR gate).
  - task-inferred routing (8): without replay the task-inference net itself forgets (routing -> chance);
    with replay it routes at ~0.89 but hard routing is worse than ER's soft 10-way classification
    (unrecoverable misroutes).
  - boundary-detected consolidation (10): online-EWC + ER is the only thing that nudges ER (+0.018),
    still under the bar; surprise does not cleanly detect boundaries (over-fires ~5x).
- **The single lever that helps is replay (ER, 0.90), and on top of it neuromodulation adds little to
  nothing (best +0.018, mostly 0 or negative).** Masked loss (lever B) roughly doubles Naive (0.38)
  but is a method change, not neuromodulation, and is capped by eval-time competition.

**Reportable conclusion.** Across pt2 (4 hidden-layer mechanisms) and pt3 (5 head-reaching/retention
mechanisms x {standalone, +ER}), no neuromodulation variant beats Naive by >=5pts on class-IL Split
MNIST, and none complements ER by >=2pts. The cause is structural (shared-head logit competition;
representation drift), addressable only by re-supplying old-class data (replay). This is a clean,
well-controlled negative result and a mechanism comparison; it is the basis for the supervisor
framing discussion (reframe toward task-IL, or neuromodulation as a lens / complementary stability
term, rather than a standalone class-IL cure).




# pt4 — every neuromod mechanism in the STANDARD (single-task) learning regime (`SPEC-proto-pt4.md`)

**Status:** `pt4: complete. Every runnable mechanism PRESERVES standard MNIST accuracy (R3 marginally improves, none degrades). 5 of 9 mechanisms are N/A by construction in single-task learning.`

**What this is.** pt2/pt3 answered project goal #1 (continual learning; all rejected). pt4 answers
goal #2: in plain single-task MNIST, does each neuromod mechanism improve, preserve, or degrade
accuracy vs a vanilla MLP? It is a comparative study at the frozen tuned standard config
(lr=3e-4, epochs=20, batch=64, the config behind `results/standard_mnist_table.md`), 3 seeds
(42/43/44), reporting test accuracy. No new mechanism (iteration discipline intact).

**Group classification (a finding in itself).** Standard learning is one stationary task (all 10
classes always present, no boundaries, no task sequence). Five of the nine mechanisms are
intrinsically continual and have no single-task form, so they are N/A by construction (not run):

| mechanism (iter) | target | why N/A in standard |
|------------------|--------|----------------------|
| weight_mask + drivers (3) | weight_mask+driver | surprise/uncertainty/activation_stats are cross-task novelty signals fed lag-1 by the CL loop; on one stationary task the signal is degenerate and the mask reduces to R2 |
| stateful / GRU (4) | weight_mask stateful | GRU state tracks cross-task dynamics, never reset between tasks; one task -> nothing to track -> reduces to R2 |
| task-inferred routing (8) | task_route | routing selects among >=2 tasks; standard has one task -> undefined |
| logit + recency (9) | logit+recency | recency = per-class presence across tasks; all classes always present -> constant driver -> reduces to R3 |
| consolidation (10) | consolidation | boundary detector + EWC anchors fire at task boundaries; none exist -> no anchor fires -> reduces to vanilla |

**Results (standard full MNIST, test acc, 3 seeds 42/43/44).**

| group | mechanism (iter) | target | test_acc ± std | vs fair vanilla | verdict |
|-------|------------------|--------|----------------|------------------|---------|
| - | vanilla (Adam) | - | 0.9796 ± 0.0008 | - | reference |
| R1 | activation gain (sprint) | activation | 0.9806 ± 0.0006 | +0.0010 | preserve (slight +) |
| R2 | weight mask (2) | weight_mask | 0.9805 ± 0.0013 | +0.0009 | preserve |
| R3 | logit calibration (6) | logit | 0.9811 ± 0.0006 | +0.0015 | marginal improve |
| R5 | importance gating (7) | importance | 0.9791 ± 0.0011 | -0.0005 | preserve |
| - | vanilla (SGD ref) | - | 0.8879 ± 0.0011 | - | R4 reference |
| R4 | plasticity / meta-LR (1) | plasticity | 0.8863 ± 0.0011 | -0.0016 (vs SGD ref) | preserve |

**Reading.**
- **R1/R2/R3/R5 all preserve** Adam-vanilla accuracy; deltas (-0.0005 to +0.0015) are within the
  combined seed std. R3 (logit calibration) is the only one whose +0.0015 edges past the combined
  std, a marginal improvement. None degrades. The extra modulator capacity is "free" in standard
  learning. R1 reproduces the published 0.9806 exactly (sanity).
- **R4 (plasticity) carries an Adam/SGD confound, handled explicitly.** The lookahead meta-gradient
  trains the main net with plain SGD (CLAUDE.md Adam-moments caveat), so its fair reference is the
  SGD-vanilla (0.8879), not the Adam-vanilla. SGD-vanilla itself sits ~9pts below Adam-vanilla
  because 20 epochs of SGD at lr=3e-4 underfits MNIST (an optimizer fact, not a modulator fact).
  Against its fair reference the meta-LR modulator is -0.0016 (within noise) -> preserve.

**Conclusion.** In the standard single-task regime every neuromodulation mechanism that is even
*definable* preserves vanilla MNIST accuracy (one marginal improvement, no degradation), confirming
project goal #2: neuromodulation imposes no standard-accuracy tax. Combined with pt2/pt3 (no
class-IL benefit), the overall picture is "neuromodulation is accuracy-neutral in standard learning
and does not cure class-IL forgetting on this MLP". The 5/9 N/A classification is the honest answer
to "run all the iteration methods in standard": the continual-only mechanisms have no single-task
form. Files: `results/pt4_standard.py`, `results/pt4_standard.log`.

# pt4/5 addendum — direct-gain modulator (user-requested): pt1 gain without the bottleneck/projection

**Status:** `direct_gain: complete. Standard: hidden-only gating preserves, two_hidden_output degrades (-0.40pt). Class-IL: no config beats Naive standalone; only last_hidden is ER-neutral, output gating hurts ER. Reject (consistent with pt2/pt3).`

**Mechanism.** New target `direct_gain` (`DirectGainModulator`). The pt1 `GainModulator` maps the
image to a low-dim signal s (k=8) then broadcasts it through a fixed projection P_l (k->hidden) to
get the per-neuron gain. This variant drops the bottleneck and the projection: each gated layer has
its own `Linear(784 -> layer_width)` head that emits the full gain vector directly (weight shape
in x out). Same FiLM gain `(1+m(x)) ⊙ h` (and `⊙ logits` for the output). Heads zero-init -> gain=1
-> exact vanilla parity at init (verified, all 4 gates allclose to vanilla). Composes with the
standard loop and the generic naive/ER CL loop (forward-graph modulator, single optimizer).

**Gate configs and neuromod-net size** (vs pt1 gain's 50,760 trainable + 6,400 frozen projection):

| gate config | layers gated | neuromod params |
|-------------|--------------|------------------|
| last_hidden | h2 | 314,000 |
| two_hidden | h1, h2 (the pt1 layout) | 628,000 |
| last_hidden_output | h2, logits | 321,850 |
| two_hidden_output | h1, h2, logits | 635,850 |

**Results (3 seeds 42/43/44).**

Standard (full MNIST). vanilla=0.9796, pt1 gain=0.9806.

| gate | test_acc | vs vanilla | verdict |
|------|----------|------------|---------|
| last_hidden | 0.9801 ± 0.0010 | +0.0005 | preserve |
| two_hidden | 0.9801 ± 0.0007 | +0.0005 | preserve |
| last_hidden_output | 0.9798 ± 0.0006 | +0.0002 | preserve |
| two_hidden_output | 0.9756 ± 0.0012 | -0.0040 | degrade (beyond noise) |

Class-IL Split MNIST. Naive=0.1979, ER=0.9023, accept bars: standalone +0.05, +ER +0.02.

| gate | (A) naive standalone | vs Naive | (B) +ER | vs ER |
|------|----------------------|----------|---------|-------|
| last_hidden | 0.1978 ± 0.0007 | -0.0001 | 0.9046 ± 0.0043 | +0.0023 |
| two_hidden | 0.1966 ± 0.0008 | -0.0013 | 0.8861 ± 0.0035 | -0.0162 |
| last_hidden_output | 0.1990 ± 0.0011 | +0.0011 | 0.8730 ± 0.0016 | -0.0293 |
| two_hidden_output | 0.1983 ± 0.0017 | +0.0004 | 0.8790 ± 0.0055 | -0.0233 |

**Findings.**
- **Standard: preserve for hidden-only gating; output gating starts to hurt.** Gating the two hidden
  layers (with or without a small output touch) is accuracy-neutral, like pt1 gain. But gating
  h1+h2+logits degrades standard by 0.40pt (beyond combined noise): a per-sample multiplicative gain
  on the 10 logits rescales the output unstably and slightly overfits.
- **Class-IL standalone: no config beats Naive (all ~0.197), even with output gating.** This is the
  same lesson as Iter 6 (logit FiLM): a gain trained on the current task alone just favors the
  current classes, so reaching the head does NOT supply the missing retention signal. Gain != memory.
- **Class-IL +ER: gating MORE hurts ER, output gating worst.** Only `last_hidden` is ER-neutral
  (+0.0023, within noise, under the +0.02 bar). Adding the second hidden layer (-0.0162) and
  especially the output logits (-0.0293) actively degrade ER: the per-sample logit gain fights the
  head calibration ER learns from replay (replayed old-class samples get their logits rescaled by a
  gain conditioned on the current input distribution).
- **Capacity is not the bottleneck.** Direct gain uses 6-12x pt1 gain's parameters (314k-636k vs
  ~51k) yet does not beat pt1 gain in standard and adds nothing in CL. The pt1 bottleneck+projection
  was never the limiting factor; removing it only adds cost and, at the output, harm.

**Decision.** Reject (consistent with pt2/pt3/pt4): gain modulation is accuracy-neutral at best in
standard (hidden-only) and provides no class-IL benefit standalone or on top of ER; output-layer
gain is mildly harmful in both regimes. Files: `results/directgain.py`, `results/directgain.log`.


# pt5 — the generalized driver system, first driver = task-id oracle (`SPEC-proto-pt5.md`)

New mechanism front: a driver -> bottleneck -> target architecture. pt5 slice is context=none, a
single driver `task_id=onehot` (dim = n_tasks = 5), so the bottleneck z IS the one-hot e_t and a
projection P (T x D) maps it to a per-element gate raw = e_t @ P = P[t]. The task id is an ORACLE
(fed at train AND eval); results are task-IL-style but reported on the class-IL 10-way metric so
they stay directly comparable to naive+masked-loss and ER. SGD main net throughout (Methodology 6:
no Adam/SGD confound). Legacy `--neuromod-driver` path untouched; new path behind `--neuromod-drivers`.

## Iteration 1 — disjoint per-task subnetworks (`projection=disjoint`)

Fixed binary P, each target element assigned to exactly one task (seeded even partition), so the
modulator is parameter-free and the main net trains under a hard per-task {0,1} gate. gain and
plasticity gate the two hidden layers; weight_mask masks net.0+net.2 with masked loss (lever B on
the head) and net.0+net.2+net.4 with ER (mask also reaches the head). Screening, 1 seed (seed=42),
test sequence, SGD lr=1e-3 ep=5, ER buffer=1000. Files: `results/pt5_taskid.py`,
`results/pt5_taskid.log`.

Baselines (SGD, same optimizer): naive-SGD+masked-loss = **0.6296** (forget 0.1245); ER-SGD =
**0.7226** (forget 0.2385). (ER-SGD is weaker than the tuned Adam-ER 0.90 from pt3, as expected for
SGD lr=1e-3; the +ER comparison is same-optimizer per Methodology 6.)

| target-config | naive+mask | neurom (delta) | ER | neurom+ER (delta) | verdict |
|---------------|-----------|----------------|-----|-------------------|---------|
| plasticity | 0.6296 | 0.4174 (-0.2122) | 0.7226 | 0.4483 (-0.2744) | reject / reject |
| weight_mask | 0.6296 | 0.4407 (-0.1890) | 0.7226 | 0.1127 (-0.6099) | reject / reject |
| gain | 0.6296 | 0.6225 (-0.0071) | 0.7226 | **0.8264 (+0.1037)** | reject / **accept-for-confirm** |

Per-task trajectories (final row, after task 5):
- gain neurom+ER: [0.997, 0.766, 0.532, 0.947, 0.890], forget 0.0089. Task 0 holds 0.997 the whole run.
- gain neurom (masked): [0.995, 0.576, 0.566, 0.975, 0.001], forget 0.0071. Last task collapses at class-IL eval.
- weight_mask neurom (masked): [0.995, 0.261, 0.221, 0.727, 0.000], forget 0.0000 (perfect hidden retention).
- weight_mask neurom+ER: [0.564, 0.000, 0.000, 0.000, 0.000], forget 0.0480 (head mask + replay collapse).

**Findings.**
- **gain+ER is the standout: +10.4pts over ER-SGD, the first neuromod cell across pt2/pt3/pt5 to
  clear the +2pt replay bar.** The disjoint gain gate gives each task a FROZEN private subnetwork:
  when a unit is gated off (gain 0) both its incoming weights and its outgoing weights (including
  its column of the shared head net.4) receive zero gradient, so a task's whole subnet is preserved
  after its task. The oracle selects that subnet at eval (task 0 stays 0.997 across all 5 tasks).
  ER supplies the missing piece: replay (plain CE, all 10 classes) calibrates the shared head's
  cross-task logit competition, which is exactly what gain cannot touch. Hidden protection (gain) +
  head calibration (replay) are complementary, hence the large gain.
- **gain standalone = naive+masked-loss (head competition caps it).** Hidden retention is near
  perfect (forget 0.007) but the newest task collapses to 0.001: masked loss isolates each task's
  logits, so at class-IL eval the newest classes lose to the accumulated magnitudes of earlier
  classes. This is the pt2/pt3 head-bottleneck diagnosis, and it is why standalone gain (0.6225)
  cannot beat naive+masked-loss (0.6296). Masked loss (lever B) reaches the head but does not
  calibrate it across tasks; replay does.
- **Disjoint plasticity and weight_mask are too aggressive and reject in every cell.** The hard
  {0,1} freeze at 1/5 capacity underperforms naive+masked-loss standalone (plasticity -0.21,
  weight_mask -0.19). weight_mask+ER is catastrophic (-0.61, near chance): masking the output head
  net.4 per-task while replaying under the current task's mask sends replayed old-class samples
  through the wrong head synapses and scrambles the shared logits. plasticity+ER also degrades
  (-0.27, forget 0.46): frozen old-task units cannot be refreshed by replay.
- **Structural read.** Only gain reaches a working retention mechanism because its gate freezes both
  the hidden units AND their head columns for free, without hard-masking the head weights the way
  weight_mask does. weight_mask's explicit per-task head mask fights replay; gain's implicit column
  freeze cooperates with it.

**Decision.** Iteration 1: reject plasticity (both), weight_mask (both), and gain standalone.
**gain+ER (0.8264, +0.104 vs ER-SGD) is accept-for-confirm** (SPEC accept bar: neurom+ER beats ER
by >=2pts). The 3-seed confirm is deferred per SPEC Methodology 3. Honest caveat (SPEC "Oracle"):
this is an oracle-task-conditioned, task-IL-style result reported on the class-IL metric; the
privileged task id at eval is what selects the disjoint subnetwork. Next: Iteration 2 (shared
backbone, `projection=shared`) tests whether partial sharing beats the full-disjoint extreme.

## Iteration 2 — shared backbone + private capacity (`projection=shared`, `shared_frac=0.5`)

Same as iter 1 but ~50% of the gated hidden units are all-ones columns (shared by every task); the
rest are disjointly assigned. Run as the exact replica of iter-1's last (gain-focused) run: gain
(`activation`) only, per-neuron, gate = (h0, h1), 3 seeds {42, 43, 44} x 2 optimizers {adam, sgd} x
{naive, naive+gain, er, er+gain} = 24 runs, lr=1e-3 ep=5 ER buffer=1000, class-IL eval, task-id
oracle. Only the projection changed vs iter 1. Files: `results/pt5_iter2_gain.py`,
`results/pt5_iter2_gain.log`.

Baselines reproduce iter-1 to 4 decimals (no-neuromod, same configs): Adam naive+masked-loss 0.3777,
Adam ER 0.8932, SGD naive+masked-loss 0.6129, SGD ER 0.7047. Confirms no regression / bit-consistency.

acc mean+-std over 3 seeds, with iter-1 (disjoint) alongside:

| optimizer | cell | disjoint (iter 1) | shared (iter 2) |
|-----------|------|-------------------|-----------------|
| Adam | naive+gain (delta vs naive+mask) | **0.9949** (+0.617) | 0.6827 +- 0.0081 (+0.3049) |
| Adam | er+gain (delta vs ER)            | **0.9901** (+0.097) | 0.9728 +- 0.0072 (+0.0796) |
| SGD  | naive+gain (delta vs naive+mask) | 0.652 (+0.039, NS)  | 0.6752 +- 0.0220 (+0.0622) |
| SGD  | er+gain (delta vs ER)            | 0.795 (+0.090)      | **0.8709 +- 0.0100 (+0.1662)** |

Forgetting (shared): naive+gain-Adam **0.2979** (vs disjoint ~0), er+gain-Adam 0.0209, naive+gain-SGD
0.1622, er+gain-SGD 0.0826.

**Findings.**
- **er+gain still clears the +2pt bar in BOTH optimizers** (Adam +7.96pt, SGD +16.62pt), so the
  headline "task-conditioned hidden gain + replay is complementary" survives partial sharing.
  accept-for-confirm, same verdict as iter 1.
- **Partial sharing does NOT beat the full-disjoint extreme; it mostly hurts.** Disjoint wins clearly
  on Adam in both cells (standalone 0.9949 vs 0.6827, +ER 0.9901 vs 0.9728). Shared wins only in the
  SGD +ER cell (0.8709 vs 0.795). So iter-2's hypothesis ("a shared backbone transfers common
  features while private capacity limits interference, beating full-disjoint") is REJECTED: the
  disjoint allocation of iter 1 is at least as good and usually better at equal oracle information.
- **Mechanism: the shared columns are not frozen, so they reintroduce forgetting.** With a fixed P
  the disjoint gate freezes a task's whole private subnet (gated-off unit -> zero grad on its
  incoming weights AND its head column). The ~50% all-ones columns are ON for every task, so every
  task writes them and none of that capacity is protected. Under fast-overwriting Adam this shows up
  directly: standalone naive+gain forgetting is 0.298 (shared) vs ~0 (disjoint), and standalone acc
  collapses from 0.995 to 0.683. Sharing trades away the exact freeze protection that made disjoint
  gain work, and the transfer it buys does not compensate.
- **Why SGD +ER is the lone shared win.** SGD overwrites the shared columns slowly and ER continually
  refreshes them from the buffer, so the shared backbone acts as extra jointly-trained capacity that
  helps rather than a fast-forgotten liability; the private half still supplies the per-task freeze.
  This is optimizer-specific and does not generalize to the (stronger) Adam cells.

**Decision.** Iteration 2: er+gain accept-for-confirm in both optimizers (beats same-optimizer ER by
>=2pts), but **shared (frac 0.5) does not improve on iter-1 disjoint** and degrades the standalone
Adam cell substantially. The best pt5 cell remains **disjoint gain** (Adam er+gain 0.9901, standalone
0.9949). 3-seed confirm of the accept cells is already in hand here (this IS the 3-seed run); the
reportable pt5 gain result stays the iter-1 disjoint numbers. Same oracle caveat (task-IL-style
result on the class-IL metric). Next: Iteration 3 (learned projection via modulator-only replay).


## Iteration 1 addendum — where the residual forgetting comes from (output-bias drift probe)

**Question.** Disjoint gain freezes each task's private subnet, so why is forgetting not exactly 0?
The SGD er+gain cell (seed 42, buffer 1000, gate (h0,h1), the exact iter-1 0.8264 run) forgets 0.0089;
the per-task trajectory `A[t,i]` shows task 0 dead-flat at 0.997 but task 1 sliding 0.802→0.766 as
tasks 2–4 train:

```
after task | task0  task1  task2  task3  task4
    1       | 0.997
    2       | 0.997  0.802
    3       | 0.997  0.785  0.536
    4       | 0.997  0.771  0.533  0.952
    5       | 0.997  0.766  0.532  0.947  0.890
```
(Absolute lows — task2 0.53, task1 0.77 — are SGD *undertraining*, not forgetting; forgetting is the
peak→final drop, essentially all of it on task 1.)

**Probe (results/scratch `bias_proof.py`: monkeypatch `evaluate` on the REAL `cl_train`, trajectory
reproduced bit-exact).** Snapshot `net.4.bias` right after task 1; at the final model, re-evaluate
task 1 with the drifted bias vs the restored post-task-1 bias, every other parameter untouched:
```
task-1 acc, final model, drifted bias  : 0.7659   (= trajectory final)
task-1 acc, final model, bias RESTORED : 0.8022   (= post-task-1 peak 0.802)
recovered by restoring ONLY the bias   : +0.0362   (task-1 forgetting was 0.802-0.766 = 0.036)
```

**Finding — forgetting here is ~100% output-bias drift.** Restoring one 10-vector recovers the entire
task-1 drop, proving every other parameter in a task's eval path is frozen: gain(h0,h1) zeros a
non-owned unit's activation, so its incoming weights, its **outgoing head columns**, AND its hidden
biases all get zero gradient during other tasks. The ONLY shared, never-frozen parameter is the
**output head bias** (the output layer is not gated), which every task and every ER-replay step
updates. The drift is pure recency — `b_final − b_afterT1`: later classes up, earlier down
(c0,c1 −0.11,−0.14 · c2,c3 −0.10,−0.08 · c4,c5 **+0.12,+0.09** · c6,c7 +0.08 · c8,c9 +0.03), ‖·‖₂=0.286.
For a task-1 input (true class 2/3) the competing later-class logits float up while its own sink,
tipping thin-margin cases. Task 0 (0.997, huge margin) is immune; task 1 (~0.80) is not — so
forgetting magnitude ∝ bias drift × fraction of predictions with margins thin enough to flip.

**Implications.** (1) This is the exact leak `--neuromod-modulate-bias --neuromod-mask-layers 4`
would close by freezing the head bias — but that bias is also what ER uses to keep old classes
competitive, so freezing it fights replay's head recalibration (cf. weight_mask+ER −0.61); the bias
is doing double duty. (2) It explains Adam ≫ SGD: same frozen columns and same bias leak, but Adam
drives each task's correct logit confident enough (margins ≈1.0) that the 0.286 drift can't flip
anything → forgetting ≈0.001 and acc ≈0.99. The mechanism (freeze) is identical; only whether the
one leak *bites* differs, and it bites only when the head competition was left marginal.


## Iter 1 + Iter 2 under TASK-IL eval (after the `--output-masking taskil` eval fix, commit 87a6b9e)

**Context.** The pt5 driver branch used to evaluate class-IL 10-way regardless of `--output-masking`,
so an earlier "taskil" table had true task-IL baselines (`naive`/`er`, non-pt5 branch) but class-IL
gain cells (`*+gain`, pt5 branch) — apples-to-oranges. The fix masks the pt5 eval to the oracle
task's 2 classes iff `output_masking=="taskil"` (else `allowed=None` → prior class-IL default, all
existing pt5 numbers unchanged). Re-ran iter-1's gain study verbatim (gain/activation, gate (h0,h1),
lr=1e-3, ep=5, buffer=1000, both optimizers, cells {naive, naive+gain, er, er+gain}) with
`output_masking=taskil` on BOTH sides, 1 seed=42, both projections. Files
`results/pt5_taskil_eval.py`/`.log`.

```
                 disjoint (iter 1)              shared frac=0.5 (iter 2)
 cell            acc     forget                 acc     forget
 -- ADAM --
 naive           0.9286  0.0695                 0.9286  0.0695
 naive+gain      0.9956  0.0015  (+0.067)       0.9793  0.0185  (+0.051)
 er              0.9942  0.0036                 0.9942  0.0036
 er+gain         0.9888  0.0090  (-0.005)       0.9952  0.0022  (+0.001)
 -- SGD --
 naive           0.9769  0.0016                 0.9769  0.0016
 naive+gain      0.9303  0.0000  (-0.047)       0.9640  0.0087  (-0.013)
 er              0.9740  0.0005                 0.9740  0.0005
 er+gain         0.8382  0.0236  (-0.136)       0.9613  0.0024  (-0.013)
```

**Finding — under genuine task-IL, gain adds ~0 on top of a strong baseline and hurts under SGD.**
Task-IL eval removes the cross-task head competition that was the class-IL bottleneck, so the
baselines are already near-ceiling (naive 0.93–0.98, er 0.97–0.99) and forgetting is ≈0 everywhere
(0.000–0.024). **er+gain ≈ er** in every cell (Adam disjoint −0.005 / shared +0.001; SGD shared
−0.013), and the SGD disjoint er+gain cell actively drops (−0.136 — the 1/T-capacity freeze
underfits without replay to refill). **naive+gain helps only as a standalone Adam retention fix**
(+0.067 disjoint, +0.051 shared: it repairs naive-Adam's 0.0695 forgetting but only reaches ER's
level, never beats it). So the pt5 disjoint-gain win was a **class-IL** result; the freeze and
task-IL masking attack the same bottleneck (head logit competition) and do not stack.

**Residual forgetting is NOT the class-IL bias leak** (that is gone under 2-way masked eval). The
clean case **SGD naive+gain = forgetting exactly 0.0000** (every task's row dead-flat: frozen subnet,
no momentum, no replay). The small nonzero forgetting elsewhere has two sources task-IL does not
remove: (a) **Adam optimizer state** nudges frozen (zero-grad) params from decaying m/v buffers →
the subnet isn't byte-frozen under Adam (naive: SGD 0.0000 vs Adam 0.0015, same setup); (b) **ER
replay** retunes each task's OWN two-class head biases — task-IL masks out *other* tasks' classes but
a task still discriminates its own pair, and replayed old samples run under the *current* task's gate
keep shifting that intra-pair boundary (er+gain: task 0 at margin 0.997 unmoved, lower-margin tasks
1–2 slide down every later task). 1 seed → directional, not reportable. Same oracle caveat.


## Iteration 3 — learned projection (`projection=learned`)

**Status:** `Iteration 3: reject (all cells). Learned allocation is WEAKER than the fixed disjoint
extreme of iter 1; no standalone cell beats its baseline by a clear margin, no neurom+ER clears the
+2pt bar, and learned plasticity+ER actively collapses. Confirms the SPEC "meta-loss < replay"
prediction.`

**What was run (user-directed slice, deviates from the SPEC's literal iter-3 target list).** Instead
of the SPEC's {gain-unbounded, gain-bounded01, plasticity, weight_mask}, the user asked for FOUR
granularity-organised mechanisms, each under the learned projection, across BOTH optimizers and BOTH
metrics, 1 seed (42), lr=1e-3, ep=5, buffer=1000:
- gain per-NEURON   (`activation`, neuron,  gate (h0,h1), gain_form=unbounded)
- gain per-SYNAPSE  (`activation`, synapse, layers net.0+net.2, gain_form=unbounded)
- plasticity per-NEURON  (`plasticity`, neuron,  layers 0,2,4 scope both)
- plasticity per-SYNAPSE (`plasticity`, synapse, layers net.0+net.2)
Layer sets held FIXED across cells (no per-condition head switching): per-synapse gain/plasticity
gate the two hidden layers only (an explicit head gate fights replay, cf. iter-1 weight_mask+ER
−0.61); per-neuron plasticity keeps 0,2,4 (reaches net.4 only via the implicit outgoing-column
coupling a1, the cooperative-with-ER kind). Files: `results/pt5_iter3.py`, `results/pt5_iter3.log`.

**Implementation added to make plasticity's learned P actually train (train.py, pt5 branch).** The
plasticity gate is applied to grads IN PLACE under no_grad, so a learned P got NO gradient (verified:
constant sigmoid(0)=0.5 gate, P.grad=None). Wired a per-batch **lookahead / first-order
meta-gradient** (mirrors the legacy `PlasticityModulator` loop) that runs ONLY for
`projection=learned`: `W_fast = W − lr·(gate⊙g)` with g detached (differentiable in P), a meta-loss
on the SAME (replay-augmented for ER) batch trains ONLY P via an Adam meta-optimizer
(`neuromod_lr`), then the real gated step commits with the same detached gate. For +ER the batch cx/cy
already carries replayed past-task samples, so the meta-loss IS the SPEC's modulator-only replay
meta-loss. Guarded by `plast_mod.fixed`: disjoint/shared plasticity are parameter-free and take the
unchanged in-place path (verified bit-exact: disjoint plasticity naive+masked reproduces iter-1
0.4174). Gain (a FORWARD target) needed no new code: its learned P sits in `model.parameters()` and
the pt5 main optimizer trains it via the ordinary main loss (one-hot → only P[t] gets a gradient, so
rows specialise per task).

**Baselines reproduce prior work exactly (sanity).** class-IL SGD naive 0.6296 / er 0.7226 (= iter
1); class-IL Adam naive 0.3894 / er 0.9053 (= pt3/iter5); task-IL Adam naive 0.9286 (= iter-5 task-IL
naive) / er 0.9942; task-IL SGD naive 0.9769 / er 0.9740.

**Results (avg_final_acc, 1 seed; delta vs same-opt/metric baseline).**

CLASS-IL:
| opt | mechanism | neurom (vs naive) | neurom+ER (vs er) |
|-----|-----------|-------------------|-------------------|
| SGD  | gain-neuron   | 0.6311 (+0.0015) | 0.7271 (+0.0045) |
| SGD  | gain-synapse  | 0.6295 (−0.0001) | 0.7266 (+0.0039) |
| SGD  | plast-neuron  | 0.6456 (+0.0160) | 0.5676 (**−0.1551**) |
| SGD  | plast-synapse | 0.6430 (+0.0134) | 0.5847 (**−0.1379**) |
| Adam | gain-neuron   | 0.3770 (−0.0124) | 0.8842 (−0.0211) |
| Adam | gain-synapse  | 0.4202 (**+0.0308**) | 0.9169 (+0.0116) |
| Adam | plast-neuron  | 0.3866 (−0.0029) | 0.9057 (+0.0004) |
| Adam | plast-synapse | 0.3820 (−0.0074) | 0.8900 (−0.0153) |

TASK-IL (near ceiling; nothing to add once the head competition is removed at eval):
| opt | mechanism | neurom (vs naive) | neurom+ER (vs er) |
|-----|-----------|-------------------|-------------------|
| SGD  | gain-neuron   | 0.9768 (−0.0001) | 0.9740 (+0.0000) |
| SGD  | gain-synapse  | 0.9771 (+0.0001) | 0.9755 (+0.0014) |
| SGD  | plast-neuron  | 0.9739 (−0.0030) | 0.9707 (−0.0033) |
| SGD  | plast-synapse | 0.9737 (−0.0032) | 0.9704 (−0.0036) |
| Adam | gain-neuron   | 0.9102 (−0.0184) | 0.9949 (+0.0007) |
| Adam | gain-synapse  | 0.9665 (**+0.0379**) | 0.9940 (−0.0002) |
| Adam | plast-neuron  | 0.9092 (−0.0194) | 0.9928 (−0.0014) |
| Adam | plast-synapse | 0.9191 (−0.0095) | 0.9925 (−0.0017) |

**Findings (mechanism, not implementation).**
- **Learned gain ≪ fixed disjoint gain (the iter-1 win does not survive learning the allocation).**
  The reportable pt5 result was iter-1 DISJOINT gain (Adam er+gain 0.9901, standalone 0.9949; SGD
  er+gain 0.795). Under the LEARNED projection, gain falls back to ≈ baseline: SGD er+gain 0.7271
  (+0.004 vs 0.7226), Adam er+gain 0.8842/0.9169 (≈ or below ER 0.9053). Cause: the fixed disjoint P
  is a HARD {0,1} gate that FREEZES each task's private subnet AND its head columns (zero grad); the
  learned P starts neutral (unbounded gain 1+0=1) and is trained per-task-row by the main loss, but
  it learns a SOFT, mostly-on gain that does not implement a hard disjoint freeze, so old capacity is
  overwritten and forgetting returns (same failure mode as iter-2 shared, where the all-ones shared
  columns were never frozen). Learning the allocation < fixing it disjoint, at equal oracle info.
- **gain-synapse is the only mildly-positive standalone cell, and only under Adam.** class-IL Adam
  0.4202 (+0.031 over naive+masked 0.3894), task-IL Adam 0.9665 (+0.038 over naive 0.9286). Its
  class-IL trajectory `[0.005, 0.111, 0.375, 0.633, 0.977]` is a graded recency curve (older tasks
  progressively, not totally, lost), a little better than naive+masked's near-total old-task loss but
  still recency-dominated. Its +ER is neutral (+0.0116 / −0.0002), i.e. ER already subsumes the small
  standalone gain. Not an accept.
- **Learned plasticity: a small standalone SGD bump, but +ER collapses.** class-IL SGD standalone
  0.6456 / 0.6430 (+0.016 / +0.013 over naive+masked 0.6296, and well above the frozen disjoint
  0.4174), so the meta-loop IS training P to protect a bit (trajectory plast-neuron `[0.943, 0.320,
  0.669, 0.753, 0.544]`, mild spread retention, no single-task collapse). But plasticity+ER is
  catastrophic under SGD: 0.5676 / 0.5847 (**−0.155 / −0.138** vs ER 0.7226), forgetting 0.39/0.37,
  trajectory `[0.916, 0.524, 0.031, 0.420, 0.947]` (task 2 → 0.031). The plasticity gate throttles
  the effective LR on the REPLAYED samples' grads too, so ER cannot refresh the down-weighted units;
  gating gradients fights replay (same family as iter-1 disjoint plasticity+ER −0.27, weight_mask+ER
  −0.61). Under Adam plast+ER ≈ ER (0.9057/0.8900 vs 0.9053): Adam's per-parameter moments partly
  cancel the gate, so it neither helps nor hurts.
- **task-IL is near ceiling; nothing moves.** With the head competition removed at eval, baselines are
  0.93–0.99 and every cell sits within ±0.01–0.04, mostly slightly negative (the gate mildly hurts
  optimisation). The lone standalone positive (gain-synapse task-IL Adam +0.038) only recovers what
  the undertrained naive-taskil-Adam (0.9286) leaves on the table; ER already reaches 0.9942.

**Verdict.** Iteration 3 rejects across all cells. The learned projection underperforms the fixed
disjoint extreme of iter 1 and does not clear either bar: no standalone cell beats its baseline by a
clear margin, no neurom+ER beats ER by ≥2pts, and learned plasticity+ER actively degrades replay.
This is exactly the SPEC's iter-3 prediction ("the meta-loss variant we predict is weaker than ER;
spending the buffer on the modulator, not the shared weights, leaves the head drifting"), and it is
direct evidence for the pt2/pt3 "replay is the lever" conclusion: the one thing that worked in pt5
(iter-1 disjoint gain) worked because of a HARD FIXED task-private freeze, not because the allocation
was learned. Ordering across pt5: **disjoint (iter 1) > shared (iter 2) > learned (iter 3)**; the
single reportable pt5 win stays iter-1 disjoint gain+ER (accept-for-confirm, 3-seed deferred). Same
oracle caveat (privileged task id at train+eval → task-IL-style even on the class-IL metric).

### Iteration 3 follow-up — init-bias (plasticity) and sparsity regularization (user-requested)

**Status:** `follow-up: (A) the plasticity +ER collapse WAS an init-value artifact (init 0.5 throttled
the replayed grads); raising init->0.99 cures it back to ~ER (no net win). (B) SPARSITY reg is a real
STANDALONE win for gain (learned gain finally beats its baseline, recovering a chunk of the disjoint
result), but NOT a +ER win, and it HURTS plasticity. Files results/pt5_iter3_followup.py/.log.`

Two probes the user asked for after iter-3, both class-IL, 1 seed (42), lr=1e-3 ep=5 buffer=1000.
New config knobs (both default to the iter-3 behaviour): `neuromod_plasticity_init` (initial LEARNED
plasticity gate via a logit bias, 0.5 = iter-3) and `neuromod_sparsity_lambda` (L1 penalty on the
projected GATE, `lambda*mean|gate|`, 0 = off). NB an L1 on P itself is degenerate here (the gate at
P=0 is 1.0 for gain / 0.5 for plasticity, not 0), so the meaningful sparsity target is the gate, not
P. gain trains P via the main loss (penalty added there); plasticity via the meta-loss (penalty there).

**(A) INIT-BIAS (plasticity, SGD).** Gain was NOT swept (its learned init is already 1.0 = parity;
its failure is mechanistic, not an init value). Table (delta vs naive+masked 0.6296 / ER 0.7226):

| mechanism | init | neurom (vs naive) | neurom+ER (vs er) |
|-----------|------|-------------------|-------------------|
| plast-neuron  | 0.50 (iter3) | 0.6456 (+0.016) | 0.5676 (−0.155) |
| plast-neuron  | 0.90 | 0.6313 (+0.002) | 0.7023 (−0.020) |
| plast-neuron  | 0.95 | 0.6304 (+0.001) | 0.7123 (−0.010) |
| plast-neuron  | 0.99 | 0.6299 (+0.000) | **0.7200 (−0.003)** |
| plast-synapse | 0.50 (iter3) | 0.6430 (+0.013) | 0.5847 (−0.138) |
| plast-synapse | 0.99 | 0.6298 (+0.000) | **0.7206 (−0.002)** |

**Finding: the iter-3 plasticity +ER collapse was an init-value artifact, now explained and removed.**
The learned gate starts at sigmoid(0)=0.5, i.e. every weight (incl. the REPLAYED samples' grads) is
throttled to half-LR from step 1, so ER could not refresh the down-weighted units (the −0.155). As the
initial gate rises 0.5 -> 0.9 -> 0.95 -> 0.99 the +ER collapse monotonically heals: plast-neuron
−0.155 -> −0.003 (0.7200 ~ ER), plast-synapse −0.138 -> −0.002. Confirmed for both variants. BUT
curing it just returns +ER to ~ER and drives the standalone bump to ~0: init->1 means the gate starts
~1 (full plasticity) so the learned P has little pressure to deviate and the mechanism approaches
vanilla. So plasticity remains a non-win; the value of (A) is diagnostic (the collapse was the 0.5
default, not the mechanism reaching the head), and 0.99 is the correct default if plasticity is ever
run under ER (do not leave it at 0.5 with replay).

**(B) SPARSITY (gate L1).** Adam grad-normalises, so a mean-normalised penalty bites near lambda~1
for gain; the per-synapse gate has ~larger fan-in so its useful lambda is ~10x higher (D-scaling).

| mechanism (opt) | lambda | neurom (vs baseline) | neurom+ER (vs er) |
|-----------------|--------|----------------------|-------------------|
| gain-neuron (adam)  | 0.0 (iter3) | 0.3770 (−0.012) | 0.8842 (−0.021) |
| gain-neuron (adam)  | 0.1  | 0.5165 (+0.127) | – |
| gain-neuron (adam)  | 0.3  | **0.6672 (+0.278)** | 0.7367 (−0.169) |
| gain-neuron (adam)  | 1.0  | 0.5668 (+0.177) | 0.8923 (−0.013) |
| gain-neuron (adam)  | 3.0  | 0.5034 (+0.114) | 0.8999 (−0.005) |
| gain-synapse (adam) | 0.0 (iter3) | 0.4202 (+0.031) | 0.9169 (+0.012) |
| gain-synapse (adam) | 1.0  | 0.5898 (+0.200) | 0.8352 (−0.070) |
| gain-synapse (adam) | 3.0  | 0.7330 (+0.344) | 0.7387 (−0.167) |
| gain-synapse (adam) | 10.0 | **0.7632 (+0.374)** | – (still rising) |
| plast-neuron (sgd)  | 1.0  | 0.4328 (−0.197) | 0.5455 (−0.177) |
| plast-synapse (sgd) | 1.0  | 0.5950 (−0.035) | 0.4557 (−0.267) |

(baselines: gain adam naive 0.3894 / er 0.9053; plast sgd naive 0.6296 / er 0.7226.)

**Finding 1: sparsity is a genuine STANDALONE win for gain (validates the "push the learned gate toward
disjoint" idea).** gain-neuron standalone rises 0.3770 -> 0.6672 (+0.278 over naive-adam) at lambda=0.3
(clean inverted-U: 0.1->0.517, 0.3->0.667, 1->0.567, 3->0.503; over-sparsify past the peak). gain-synapse
rises monotonically to 0.7632 at lambda=10 (+0.374, peak not bracketed; needs higher lambda for the
larger fan-in). These are the LARGEST standalone gains anywhere in pt5's learned work and clear the
standalone bar decisively. Mechanism: the L1 pushes each task's gate toward a sparse active subset, so
the learned gate approaches the iter-1 disjoint {0,1} freeze (a soft, learned version of it), partially
recovering the disjoint-gain standalone (disjoint gain-Adam standalone was 0.9949; sparsity gets a
chunk of the way, 0.67-0.76, not all the way because it is soft-learned, not hard-fixed).

**Finding 2: sparsity does NOT help +ER, and HURTS plasticity.** For gain+ER the best sparsity result is
~ER (gain-neuron lambda=3 -> 0.8999 ~ ER 0.9053; lower lambda hurts, e.g. lambda=0.3 -> −0.169), and
gain-synapse+ER only degrades from its iter-3 0.9169. Reason: ER already calibrates the shared head (the
exact class-IL bottleneck the standalone case lacked), so the extra sparsity constraint on top just costs
capacity / fights replay. Consistent with the pt2/pt3/pt5 conclusion that replay is the lever and the
modulator adds ~0 on top. For PLASTICITY sparsity is harmful in both cells: it drives the learned gate to
the frozen regime (plast-neuron standalone -> 0.4328 ~ the iter-1 disjoint-plasticity 0.4174, which was
already rejected as too aggressive), so more freezing is exactly the wrong direction there.

**Verdict.** The follow-up sharpens iter-3 rather than overturning it. (A) The plasticity +ER collapse was
an init artifact (0.5 gate throttling replay), fixable to ~ER but with no net benefit. (B) Sparsity
regularisation turns learned gain into a clear STANDALONE win (the learned projection partially recovering
the disjoint-gain standalone, the user's hypothesis confirmed), accept-for-confirm on the gain standalone
cells (3-seed deferred); but there is still NO learned +ER win (best ~ER), and sparsity hurts plasticity.
The reportable pt5 headline is unchanged (iter-1 disjoint gain+ER); the new, reportable standalone result
is "learned gain + gate-sparsity beats its class-IL baseline standalone under the oracle" (gain-neuron
+0.278, gain-synapse +0.37). Same oracle caveat throughout.

### Iteration 3 follow-up 2 — standalone modulator-only replay meta-loss (user-requested)

**Status:** `follow-up 2: the iter-3 STANDALONE plasticity meta-loop trained P on the current batch only
(no retention signal). Adding a modulator-only replay buffer (--neuromod-meta-replay; buffer trains ONLY
P, main net stays naive) helps under ADAM (class-IL plast-neuron +0.038, task-IL +0.041, forgetting
-0.04..-0.08) but does NOTHING under SGD. Modest, SGD-inert, still far below ER, oracle-conditioned. Files
results/pt5_iter3_metareplay.py/.log.`

New config knob `neuromod_meta_replay` (default OFF = iter-3, verified bit-exact: plast-neuron class-IL SGD
0.6456). ON: a reservoir buffer of past examples augments the META-loss batch (current + buffer sample), so
the lookahead trains P for retention (does the gated current step preserve past tasks?), while the MAIN net
still steps naive on the current task only. This is the SPEC's iter-3 "modulator-only replay meta-loss" for
the standalone condition. No effect for +ER (the meta already sees replay via cx) or fixed P.

Standalone (naive), both plasticity mechs x {class-IL, task-IL} x {SGD, Adam}, 1 seed, buffer=1000. OFF =
iter-3 (results/pt5_iter3.log).

| metric | opt | mech | acc OFF -> ON (delta) | forget OFF -> ON (delta) |
|--------|-----|------|-----------------------|---------------------------|
| class-IL | SGD  | plast-neuron  | 0.6456 -> 0.6454 (-0.000) | 0.1285 -> 0.1273 |
| class-IL | SGD  | plast-synapse | 0.6430 -> 0.6417 (-0.001) | 0.1130 -> 0.1106 |
| class-IL | Adam | plast-neuron  | 0.3866 -> **0.4244 (+0.038)** | 0.5841 -> **0.5059 (-0.078)** |
| class-IL | Adam | plast-synapse | 0.3820 -> 0.3861 (+0.004) | 0.5933 -> 0.5280 (-0.065) |
| task-IL  | SGD  | plast-neuron  | 0.9739 -> 0.9738 (-0.000) | ~0 |
| task-IL  | SGD  | plast-synapse | 0.9737 -> 0.9738 (+0.000) | ~0 |
| task-IL  | Adam | plast-neuron  | 0.9092 -> **0.9503 (+0.041)** | 0.0875 -> **0.0473 (-0.040)** |
| task-IL  | Adam | plast-synapse | 0.9191 -> 0.9366 (+0.018) | 0.0786 -> 0.0605 (-0.018) |

**Finding: the retention signal helps exactly where there is fast forgetting to prevent (Adam), and is
redundant where retention is already handled (SGD).** Under SGD the main net overwrites slowly and the
masked loss already supplies retention, so a retention-informed freeze protects units that were not being
lost anyway -> no change (+-0.001). Under Adam the main net overwrites fast (naive+masked-Adam 0.389 vs
SGD 0.630), so there is real forgetting to prevent, and the buffer now trains P to freeze the past-important
units: forgetting drops 0.04-0.08 and acc rises, most for plast-neuron (which gates net.4 HEAD COLUMNS via
the outgoing-column coupling a1, so freezing a past unit also freezes its head-weight column -> it reaches
the class-IL bottleneck enough to move the number; plast-synapse gates only net.0/net.2 hidden weights, so
it moves less: +0.004 class-IL).

**Honest limits.** Modest: class-IL Adam plast-neuron 0.4244 is still far below ER (0.9053) and only just
above naive+masked (0.3894, +0.035); it does NOTHING under SGD (Methodology-6's clean optimizer); it never
touches the head BIAS (the residual-forgetting leak, cf. the iter-1 bias probe); and it is oracle-conditioned.

**Correction to the point-2 framing.** Earlier the standalone plasticity failure was attributed to "no
retention signal." This experiment shows that was only part of it: supplying the retention signal (a proper
modulator-only replay buffer) DOES help where forgetting is fast (Adam), so the missing signal was a real
factor there. But the binding ceiling is still the lever, not the signal: even with a perfect retention
buffer, hidden/column LR-gating gets plasticity only to ~0.42 class-IL, because it protects hidden weights
and head-weight columns but not the head bias / full logit calibration, which only replay supplies.

### Iteration 3 follow-up 3 — modulator-only replay meta-loss for GAIN (user-requested; BIG standalone win)

**Status:** `follow-up 3: gain modulator-only replay meta-loss is the STRONGEST learned-projection result
in pt5. Standalone class-IL SGD gain-synapse 0.6295 -> 0.9871 (+0.358, forget -> 0.0015), gain-neuron 0.6311
-> 0.9074 (+0.276); the learned per-task gate trained by a per-task replay meta-loss reaches the
disjoint-oracle ceiling and OVERTURNS the iter-3 "learned < disjoint" ordering for standalone. Oracle +
replay-on-modulator caveats. Files results/pt5_iter3_gain_metareplay.py/.log.`

Extends `--neuromod-meta-replay` to gain (a FORWARD target). iter-3 trained gain's learned P via the MAIN
loss (current task only, so P learned a soft mostly-on gate = no freeze = fail). Now: a SEPARATE optimizer
trains ONLY P on a modulator-only replay meta-loss, main net trains naive on the current task (P excluded
from the main optimizer). Because gain gates the FORWARD, the meta-loss is PER-TASK: each seen task j is
forwarded under ITS OWN gate P[j] (fresh current batch for j=t, buffer samples for j<t), losses summed;
only P[j] gets a gradient (one-hot). `_pt5_gain_modulator_params` splits P from the backbone;
`label_to_task` routes buffer samples to their task's gate. Default OFF reproduces iter-3 bit-exact
(gain-neuron class-IL SGD 0.6311, Adam 0.3770). Standalone only, both gain mechs x {class-IL, task-IL} x
{SGD, Adam}, 1 seed, buffer=1000.

| metric | opt | mech | acc OFF -> ON (delta) | forget OFF -> ON |
|--------|-----|------|-----------------------|-------------------|
| class-IL | SGD  | gain-neuron  | 0.6311 -> **0.9074 (+0.276)** | 0.1242 -> 0.0205 |
| class-IL | SGD  | gain-synapse | 0.6295 -> **0.9871 (+0.358)** | 0.1239 -> 0.0015 |
| class-IL | Adam | gain-neuron  | 0.3770 -> 0.5075 (+0.131) | 0.5883 -> 0.3713 |
| class-IL | Adam | gain-synapse | 0.4202 -> 0.6304 (+0.210) | 0.5668 -> 0.3238 |
| task-IL  | SGD  | gain-neuron  | 0.9768 -> 0.9838 (+0.007) | ~0 |
| task-IL  | SGD  | gain-synapse | 0.9771 -> 0.9919 (+0.015) | ~0 |
| task-IL  | Adam | gain-neuron  | 0.9102 -> 0.9881 (+0.078) | 0.0860 -> 0.0091 |
| task-IL  | Adam | gain-synapse | 0.9665 -> 0.9950 (+0.029) | 0.0306 -> 0.0030 |

**Mechanism: the per-task gates are task-specific READOUT ADAPTERS, continuously re-calibrated by the replay
meta-loss to track the drifting shared backbone.** During task t, the main net drifts (naive); each past
task's gate P[j] is retrained (meta, its buffer term) to keep task j readable under the CURRENT weights, and
the oracle selects P[i] at eval. Under SGD the backbone drifts slowly so the gates keep up (forget ~0.02,
gain-synapse ~0.0015; trajectory gain-synapse `[0.999, 0.972, 0.996, 0.997, 0.972]`, no task collapses).
Under Adam the backbone overwrites fast so the gates lag (still +0.13..+0.21, but forget stays 0.32-0.37).

**This overturns the iter-3 ordering FOR STANDALONE, and shows HOW you train the learned P is decisive.**
iter-3 learned gain (P trained by the main loss) gave 0.63 SGD / 0.38 Adam (soft mostly-on gate, no freeze);
the SAME learned P trained instead by a per-task modulator-only replay meta-loss gives 0.91-0.99 SGD. So
"disjoint (iter1) > shared (iter2) > learned (iter3)" held only for main-loss-trained P; a replay-meta-trained
P beats even the fixed disjoint standalone (disjoint gain SGD standalone was 0.6225).

**Honest caveats (crucial).** (1) ORACLE: the task id selects P[i] at eval, so this is a task-IL-style result
on the class-IL metric, NOT a class-IL solution; the gates are per-task parameters and the oracle picks the
right one. (2) It USES the buffer (replay on the MODULATOR, not the backbone) - so it is "spend replay on
per-task gates," not replay-free; under SGD it exceeds ER-SGD (0.7226, main-net replay) but ER is true
class-IL with NO oracle, so that is not apples-to-apples. (3) gain-SYNAPSE's near-perfect 0.9871 comes with a
huge per-task parameter cost: its P is (T x d_out x d_in) per layer (~5.6M params, far larger than the ~478k
backbone) - essentially a per-task weight mask, so its capacity is the story; gain-NEURON (P ~4k params)
reaching 0.9074 is the parameter-efficient, more interesting result. (4) SGD-specific for the big win; 1 seed.

**Verdict.** accept-for-confirm on the gain standalone cells (strong, 3-seed deferred). This is the reportable
"neuromodulation as replay-calibrated per-task adaptation" result for pt5: under the oracle, spending the
buffer on a small per-task gain gate (gain-neuron, ~4k params), not the backbone, recovers standalone class-IL
from 0.63 to 0.91 under SGD. The +ER story is unchanged (not run here; the win is standalone).

---

## pt5 --neuromod-er-task-id (per-task masks for replayed samples under +ER, user-requested)

**Feature (not a new iteration).** Under +ER the mixed `cat([current, replay])` batch was processed under
a SINGLE current task `t`, so a replayed task-`j` sample got the WRONG mask `P[t]` instead of its own
`P[j]`. `--neuromod-er-task-id` (default OFF = parity) applies each sample's own task mask, split by task:
FORWARD targets (gain/weight_mask) forward each task subset under its `P[j]` and scatter the logits back
(the gate is in the forward); PLASTICITY backward each subset (weighted `n_j/N`), gate its gradient by
`P[j]`, accumulate, one step (the gate is on the gradient). Learned P: a per-task lookahead meta-loss over
the full ER batch trains `P[j]`. Verified autograd-correct (forward: byte-identical to a per-subset weighted
backward; plasticity: byte-exact `(1/N)·Σ_i P[task(i)]⊙g_i`, max|Δ|=7e-8), and genuinely ≠ the old `P[t]`
path.

**Study** `results/pt5_er_task_id.py`/`.log`: disjoint projection, class-IL (ER masked-loss OFF=`none`, the
pt5 replay convention — OFF reproduces iter-1 er+gain bit-exact), seed 42, lr=1e-3, ep=5, buffer=1000, no
sparsity, 1 seed. OFF (batch under `P[t]`) vs ON (each sample under its `P[j]`):

| mechanism    | opt  | OFF acc | ON acc | dAcc   | OFF forg | ON forg | dForg  |
|--------------|------|---------|--------|--------|----------|---------|--------|
| gain-neuron  | sgd  | 0.8264  | 0.8163 | -0.010 | 0.0089   | 0.0012  | -0.008 |
| gain-neuron  | adam | 0.9901  | 0.9948 | +0.005 | 0.0002   | 0.0005  | +0.000 |
| gain-synapse | sgd  | 0.2576  | 0.6133 | +0.356 | 0.0001   | 0.0050  | +0.005 |
| plast-neuron | sgd  | 0.4483  | 0.4833 | +0.035 | 0.4577   | 0.3480  | -0.110 |

**Reading.** er_task_id is **~neutral where the baseline already retains**: per-neuron gain's disjoint
subnet already isolates each task, so routing replayed samples through their own subnet barely moves
accuracy (only a small forgetting drop, SGD 0.0089->0.0012). It **substantially rescues the cases where the
WRONG-task mask was actively scrambling replayed samples**: per-synapse gain (= the weight_mask-equivalent
that collapses under ER, cf. weight_mask+ER -0.61) jumps 0.2576->0.6133 (+0.356) because ON sends each
replayed sample through its own synapse mask (correct features) instead of `P[t]` (which corrupted them);
and plasticity's forgetting drops 0.4577->0.3480 (-0.110) with a small acc gain. So the flag matters most
for the aggressive per-synapse / plasticity masks, not for gain-neuron.

**Caveats.** 1 seed, ep=5; oracle (task id at eval). Corrects an earlier ep=2 smoke that used ER
masked-loss ON (`loss`, non-standard for ER cells) and misleadingly showed gain-neuron +0.073 — under the
correct `none` masking the OFF baseline reproduces iter-1 (0.8264 / 0.9901) and gain-neuron is ~neutral.
The reportable pt5 gain headline is unchanged (iter-1 disjoint gain+ER, Adam 0.9901). accept-for-confirm on
the gain-synapse / plasticity rescue deferred (3-seed).

## pt5 gain FORMS × BUFFER (user-requested; `gain_form` refactor + `positive`, then the buffer arms)

**Code (not a new iteration).** Two changes precede the studies:
1. `TaskWeightMaskMLP.gate={mask,gain}` REMOVED — it was fully redundant with `gain_form`
   (`gate="mask"` == `gain_form="bounded01"`; under a fixed P every form returns raw `{0,1}`), so the two
   only ever differed at learned+unbounded. `weight_mask` now PINS `gain_form="bounded01"` (a mask is
   suppress-only by definition; reading the flag would let `--neuromod-gain-form unbounded` silently turn a
   mask into a gain). Third form added: `positive` = `softplus(raw + ln(e-1))`, range (0,+inf), init 1.0
   exactly (same zero-init parity as `unbounded`), amplifies but never inverts. Behaviour-preserving:
   disjoint gain seed42 SGD ep=5 reproduces 0.6225/0.0071 bit-exact.
2. `--neuromod-er-task-id` extended to the NON-replay gain meta-loop. It was replay-gated
   (`er_task_id_on = use_replay and ...`) so it could never fire on a naive run, and the meta-loop never
   read it. Now one flag means one thing everywhere ("gate a buffered sample by its own task id"):
   standalone it selects the meta arm, +ER it selects the main-batch arm (unchanged, verified byte-exact —
   disjoint er+gain SGD still 0.8264/0.0089). **POLARITY:** the meta-loop was per-task UNCONDITIONALLY
   before, so OFF is now the wrong-task ablation; `results/pt5_iter3_gain_metareplay.py` passes the flag
   explicitly to reproduce its numbers (verified 0.9074).

**Studies** (both class-IL, `projection=learned` — `gain_form` is INERT under a fixed P, so the forms only
exist here — seed 42, lr=1e-3, ep=5, buffer=1000, lambda=0, 1 seed):
`results/pt5_gain_forms.py`/`.log` (28 runs, no buffer; all 7 `unbounded` cells reproduce iter-3 bit-exact),
`results/pt5_gain_forms_buffer.py`/`.log` (40 runs, buffer arms), merged by
`results/pt5_gain_forms_table.py`/`.log`. Arms: `no-buf` | `meta-cur`/`meta-own` (standalone +
modulator-only replay buffer, meta batches under `P[t]` vs each task's own `P[j]`) | `er-cur`/`er-own`
(+ER, replayed sample under `P[t]` vs its own `P[j]`; `er-cur` = the no-buffer sweep's `neurom+er` cells,
valid since meta_replay is inert for +ER).

| opt  | mech         | form      | no-buf | meta-cur | meta-own | d-meta | er-cur | er-own | d-er   | own-er |
|------|--------------|-----------|--------|----------|----------|--------|--------|--------|--------|--------|
| sgd  | gain-neuron  | unbounded | 0.6311 | 0.7155   | 0.9074   | +0.192 | 0.7271 | 0.7376 | +0.011 | +0.015 |
| sgd  | gain-neuron  | bounded01 | 0.4638 | 0.7007   | 0.8592   | +0.159 | 0.2901 | 0.2898 | -0.000 | -0.433 |
| sgd  | gain-neuron  | positive  | 0.6303 | 0.6990   | 0.8694   | +0.170 | 0.7243 | 0.7279 | +0.004 | +0.005 |
| sgd  | gain-synapse | unbounded | 0.6295 | 0.7391   | 0.9871   | +0.248 | 0.7266 | 0.7282 | +0.002 | +0.006 |
| sgd  | gain-synapse | bounded01 | 0.4643 | 0.7793   | 0.9809   | +0.202 | 0.2855 | 0.2855 | +0.000 | -0.437 |
| sgd  | gain-synapse | positive  | 0.6291 | 0.7535   | 0.9832   | +0.230 | 0.7261 | 0.7270 | +0.001 | +0.004 |
| adam | gain-neuron  | unbounded | 0.3770 | 0.3696   | 0.5075   | +0.138 | 0.8842 | 0.9887 | +0.105 | **+0.083** |
| adam | gain-neuron  | bounded01 | 0.4073 | 0.3813   | 0.4293   | +0.048 | 0.9031 | 0.9879 | +0.085 | **+0.083** |
| adam | gain-neuron  | positive  | 0.4623 | 0.3344   | 0.4351   | +0.101 | 0.8960 | 0.9862 | +0.090 | **+0.081** |
| adam | gain-synapse | unbounded | 0.4202 | 0.3798   | 0.6304   | +0.251 | 0.9169 | 0.9900 | +0.073 | **+0.085** |
| adam | gain-synapse | bounded01 | 0.4271 | 0.3628   | 0.5925   | +0.230 | 0.9100 | 0.9911 | +0.081 | **+0.086** |
| adam | gain-synapse | positive  | 0.4389 | 0.3406   | 0.6047   | +0.264 | 0.8839 | 0.9895 | +0.106 | **+0.084** |

Baselines: SGD naive 0.6296 / er 0.7226; Adam naive 0.3894 / er 0.9053.

**Result 1 — er_task_id under ADAM is the first LEARNED-projection +ER win (accept-for-confirm).** All six
Adam `er-own` cells clear the +2pt bar by ~4x (0.9862-0.9911 vs ER 0.9053, +0.081..+0.086; forgetting
0.005-0.011). The flag effect itself (`d-er`, the clean SAME-ORACLE comparison) is +0.073..+0.106.

**Result 2 — the effect is OPTIMIZER-DEPENDENT, and the two regimes mirror each other.** SGD: `d-er`
+0.000..+0.011 (neutral) but standalone `meta-own` 0.859-0.987. Adam: `d-er` +0.073..+0.106 but standalone
`meta-own` only 0.429-0.630. Consistent with followup-3's mechanism: SGD drifts the backbone slowly so
per-task gates keep up (forget ~0), Adam drifts fast so standalone gates lag — but replay refreshes the
backbone underneath them. **Do NOT generalise an er_task_id reading from one optimizer.**

**Result 3 — challenges iter-3's `disjoint > shared > learned` ordering.** Learned + correct task ids under
Adam hits 0.9887, indistinguishable from iter-1's FIXED-disjoint headline 0.9901. Mechanism: disjoint's hard
`{0,1}` gate already isolates each task's subnet, so mis-routing a replayed sample barely matters (the
disjoint er_task_id study saw only +0.005); a LEARNED soft mostly-on gate scrambles replayed samples badly
under `P[t]`, so fixing the routing is worth ~20x more. Learned was not weaker than disjoint — it was being
fed the wrong gate.

**Result 4 — per-task META gating is most of followup-3's standalone win.** `d-meta` > 0 in all 12 cells
(+0.05..+0.26). Decomposing gain-synapse/SGD/unbounded: 0.6295 (no buffer) -> 0.7391 (buffer, wrong-task
gate; +0.110 = the buffer alone) -> 0.9871 (per-task gate; +0.248). Mechanism: the meta-loss trains ONLY P,
so a wrong gate routes every meta gradient to row `t` via the one-hot and the past rows `P[j]` get NO
gradient — the retention signal vanishes. Under ER the same mis-gating is ~free because replayed samples
train the BACKBONE regardless of which gate they pass.

**Result 5 — forms.** No buffer + Adam: `positive` > `bounded01` > `unbounded` consistently on BOTH
granularities (gain-neuron 0.4623/0.4073/0.3770), i.e. sign inversion HURTS and forbidding it while keeping
the 1.0 init helps; forgetting orders the same way (0.498/0.543/0.588). SGD + buffer: `unbounded` >
`positive` > `bounded01`, but the per-synapse margins (0.9871/0.9832/0.9809) are within 1-seed noise. **In
the winning regime (Adam er-own) the form does not matter at all** — all six land at 0.986-0.991. So there
is no universally best form; it is regime-dependent and moot where it counts.

**Result 6 — `bounded01` is NOT parity at init** (`sigmoid(0)=0.5`, unlike unbounded/positive's 1.0), so it
halves every gated activation from step one and is capped at 1 (it can never recover the scale). +ER under
SGD collapses (-0.43); under Adam it is fine (+0.083) because Adam's per-parameter normalisation absorbs a
uniform rescale. Correct routing changes the collapse by 0.000 (0.2901->0.2898, 0.2855->0.2855), which
RULES OUT task-id misrouting as the cause and confirms the init-suppression story.

**Caveats.** 1 seed, lambda=0, ORACLE task id at train+eval (task-IL-style on the class-IL metric).
`own-er` compares an ORACLE method to a non-oracle ER baseline and is NOT apples-to-apples; `d-er` is the
clean number (both arms carry the oracle, differing only in the flag). The standalone meta arms USE the
buffer (replay on the MODULATOR, not the backbone), so `meta-own` is not comparable to `no-buf`/naive.
3-seed confirm of the six Adam `er-own` cells deferred per SPEC Methodology 3.

## pt5 LEARNED-plasticity init sweep (user-requested; does the gate init rescue plasticity?)

**Question.** The learned plasticity gate is `sigmoid(init_bias + raw)`, `init_bias = logit(neuromod_plasticity_init)`,
so the init sets BOTH the starting LR throttle AND the gate's own trainability, and the two fight: init 0.5 has
the max gate gradient (s(1-s)=0.25) but halves every grad from step one; init 0.99 removes the throttle but
saturates (s'=0.0099, ~25x weaker) so the gate drifts to ~1 and the mechanism degenerates to vanilla. The recorded
iter3-followup ran only the two ENDS. This sweep fills the middle (0.8, 0.9) and, on request, extends below the
0.5 peak (0.1, 0.3).

**Runs.** class-IL, projection=learned (init is IGNORED under a fixed P — the gate is raw {0,1}), seed 42, lr=1e-3,
ep=5, buffer=1000, lambda=0, 1 seed. Main sweep `results/pt5_plast_init.py`/`.log` (36 runs: init {0.5,0.8,0.9,0.99}
x {plast-neuron, plast-synapse} x {sgd,adam} x {standalone, +ER}); low-init extension (0.1, 0.3) ad-hoc, logged in
`results/pt5_plast_init_low.log` (sgd plast-neuron), `_low_adam.log` (adam plast-neuron), `_low_adam_synapse.log`
(adam plast-synapse). standalone = `--neuromod-meta-replay` (buffer trains ONLY P via the lookahead meta-loss; note
plasticity gates the GRADIENT, so its meta-loss applies ONE gate to the whole summed grad — no meta-own/meta-cur
split, unlike gain); +ER = `--neuromod-er-task-id` (each replayed sample's grad gated by its own P[j]).
Reference: adam plast-neuron standalone init=0.5 = 0.4244 reproduces iter3-followup-2 (confirms the gain-meta flag
rewiring left plasticity untouched). GAP: sgd plast-synapse at 0.1/0.3 not run.

**SGD** — baselines naive 0.6296 / er 0.7226. init: bias / dgate = s(1-s).

| mech          | init | standalone | vs naive | er     | vs er   |
|---------------|------|------------|----------|--------|---------|
| plast-neuron  | 0.10 | 0.4751     | -0.1545  | 0.4047 | -0.3179 |
| plast-neuron  | 0.30 | 0.6408     | +0.0112  | 0.5847 | -0.1379 |
| plast-neuron  | 0.50 | 0.6454     | +0.0158  | 0.6598 | -0.0628 |
| plast-neuron  | 0.80 | 0.6346     | +0.0050  | 0.7087 | -0.0140 |
| plast-neuron  | 0.90 | 0.6312     | +0.0016  | 0.7154 | -0.0073 |
| plast-neuron  | 0.99 | 0.6298     | +0.0002  | 0.7215 | -0.0011 |
| plast-synapse | 0.50 | 0.6417     | +0.0121  | 0.6110 | -0.1116 |
| plast-synapse | 0.80 | 0.6346     | +0.0050  | 0.6907 | -0.0320 |
| plast-synapse | 0.90 | 0.6308     | +0.0012  | 0.7077 | -0.0149 |
| plast-synapse | 0.99 | 0.6297     | +0.0001  | 0.7206 | -0.0021 |

**Adam** — baselines naive 0.3894 / er 0.9053.

| mech          | init | standalone | vs naive | er     | vs er   |
|---------------|------|------------|----------|--------|---------|
| plast-neuron  | 0.10 | 0.3892     | -0.0002  | 0.8782 | -0.0271 |
| plast-neuron  | 0.30 | 0.3973     | +0.0079  | 0.8472 | -0.0581 |
| plast-neuron  | 0.50 | 0.4244     | +0.0349  | 0.8936 | -0.0117 |
| plast-neuron  | 0.80 | 0.3204     | -0.0690  | 0.9083 | +0.0030 |
| plast-neuron  | 0.90 | 0.3925     | +0.0031  | 0.8826 | -0.0227 |
| plast-neuron  | 0.99 | 0.3772     | -0.0122  | 0.8910 | -0.0143 |
| plast-synapse | 0.10 | 0.3984     | +0.0090  | 0.8977 | -0.0076 |
| plast-synapse | 0.30 | 0.3496     | -0.0398  | 0.8946 | -0.0107 |
| plast-synapse | 0.50 | 0.3861     | -0.0033  | 0.8935 | -0.0118 |
| plast-synapse | 0.80 | 0.3798     | -0.0096  | 0.8759 | -0.0294 |
| plast-synapse | 0.90 | 0.4305     | +0.0411  | 0.8871 | -0.0182 |
| plast-synapse | 0.99 | 0.3847     | -0.0047  | 0.8997 | -0.0056 |

**Result — REJECT at every init; the init never rescues learned plasticity.** No cell clears any bar (best
standalone +0.041, best +ER +0.003, both inside Adam's noise). Four readings:

1. **SGD standalone has a mild INTERIOR optimum at init 0.5 (+0.016), not a monotone trend.** Both sides fall off
   it: -> 0.99 decays to vanilla (gate lifts), -> 0.1 crashes BELOW naive (-0.155) as the freeze over-closes and the
   main net cannot learn the current task (forgetting -> 0.0435 but accuracy collapses with it — the classic
   over-regularization signature). The peak is real but tiny.
2. **SGD +ER is MONOTONE — only worse as init drops, to -0.318 at init 0.1.** No interior optimum; the best ER can
   do is init 0.99 (turn the mechanism off). The standalone peak (0.5) and the ER optimum (0.99) are at OPPOSITE
   ends: one gate parameter cannot serve both arms, so there is no init that helps standalone without hurting replay.
   This is the same one-parameter throttle-vs-replay conflict, now traced to its extreme.
3. **Adam is noisy and effectively INERT to the init** (standalone hovers near naive, non-monotone; +ER stays
   0.85-0.91 across the whole range). Adam's per-parameter normalization absorbs a uniform gate rescale.
4. **The SGD-vs-Adam contrast at the extreme is the cleanest single finding.** At init 0.1, SGD +ER craters
   (plast-neuron 0.4047, -0.318) while Adam +ER barely moves (plast-neuron 0.8782 -0.027; plast-synapse 0.8977
   -0.008). The throttle that dominates SGD is neutralized by Adam's normalization.

**Conclusion.** Corroborates pt3 rather than overturning it: the learned plasticity gate does no useful work at ANY
init on ANY optimizer/granularity — it either throttles (low init) or vanishes into vanilla (high init), and the
middle just interpolates. The binding constraint is the LEVER (gating gradients cannot fix class-IL head competition),
not the initialization. This also closes the reparameterization idea (`clamp(1+raw,0,1)` would fix saturation, but
saturation is not what limits this). Caveats: 1 seed, lambda=0, ORACLE task id at train+eval; plasticity+Adam is a
first-order surrogate (grads gated before .step()), so SGD is the clean read; sgd plast-synapse low-init not run.

## pt5 OUT-LAYER + BIAS modulation (user-requested; extend the learned-P arms to net.4, then to biases)

**Question.** Two extensions to the learned-projection mechanisms, applied cumulatively: (1) add the OUTPUT layer
to the modulated set — gain-neuron gains a learned per-class logit gate (`gain_layers 0,2,4`), the per-synapse
mechanisms gate the head weights (`mask_layers 0,2,4`); (2) also modulate the BIASES of the modulated layers
(`--neuromod-modulate-bias`, independent learned `P_bias` per layer, incl. the head bias — the documented residual
leak no weight gate can reach). Grid: {gain, plast} × {per-neuron, per-synapse} × {sgd, adam} × class-IL × two
arms: standalone with the modulator-only replay buffer (`buf-meta-own`) and +ER with own-task routing (`er-own`).
Files `results/pt5_out_bias.py`/`.log`. Seed 42, lr 1e-3, ep 5, buffer 1000, gain_form=unbounded, plast init 0.5;
12 comparison cells merged from `pt5_gain_forms_buffer.log` / `pt5_plast_init.log`; all 4 baselines reproduce
bit-exact. Notes: bias stage is per-SYNAPSE only (the toggle is not wired to per-neuron mechanisms by design);
plast-neuron already defaults to layers 0,2,4, so its NEW cell is the reverse ablation (hidden-only 0,2), isolating
the head-column coupling.

**Result table (acc; f = forgetting; deltas within-row).** See `results/pt5_out_bias.log` for the full table.
Headline cells:

| cell | hid (0,2) | out (0,2,4) | d-out | bias | d-bias |
|---|---|---|---|---|---|
| gain-neuron meta SGD | 0.9074 | **0.9625** (f 0.0009) | +0.055 | n/a | |
| gain-neuron er SGD | 0.7376 | **0.9466** (f 0.0265) | **+0.209** | n/a | |
| gain-neuron meta Adam | 0.5075 | 0.7229 | **+0.215** | n/a | |
| gain-neuron er Adam | 0.9887 | 0.9875 | −0.001 | n/a | |
| gain-synapse er Adam | 0.9900 | 0.9915 | +0.002 | **0.9923** (f 0.0044) | +0.001 |
| gain-synapse meta Adam | 0.6304 | 0.6907 | +0.060 | 0.5203 | **−0.170** |
| plast-neuron meta Adam | 0.3787 (NEW hid) | 0.4244 | +0.046 | n/a | |
| plast-synapse er SGD | 0.6110 | 0.5845 | −0.027 | 0.5837 | −0.001 |

**Reading 1 — the learned LOGIT gate is the win of this study; it rescues gain-neuron's weak cells.** Adding the
per-class logit gain moves every gain-neuron cell that wasn't already at ceiling: SGD er-own **0.7376 → 0.9466**
(+0.209 — the SGD+ER regime was gain's weak spot in every prior study, now +0.224 over ER), SGD standalone
0.9074 → 0.9625 (forget → 0.0009, approaching gain-synapse's 0.9894 with a ~4k-param P instead of ~5.6M — the
parameter-efficient cell), Adam standalone 0.5075 → 0.7229. Adam er-own stays at ceiling (0.9875 ≈ 0.9887). The
mechanism reaches the class-IL bottleneck (head logit competition) DIRECTLY, which the hidden gates never could —
and unlike the fixed-P layer-4 gate (random partition, collapses to chance), the learned P_out is zero-init
(γ=1 parity) and each task's row is trained by its own (meta or main) loss. gain-neuron out now clears the bar in
all four of its cells (1 seed): standalone SGD +0.333 / Adam +0.334 vs naive+masked, er SGD +0.224 / Adam +0.082.

**Reading 2 — explicit head-synapse gating no longer fights replay once routing is correct.** gain-synapse
mask_layers 0,2,4 under er-own is neutral-to-positive (SGD +0.007, Adam +0.002; Adam bias cell 0.9923 = the best
+ER number in pt5, though within 1-seed noise of the 0.9900 hid cell). The fixed-P-era catastrophe
(weight_mask+ER −0.61) is fully absent under learned P + own-task routing. plast-synapse head gating under er-own
SGD still hurts (−0.027) — gating head GRADIENTS throttles replay's head recalibration (the init-0.5 throttle),
unlike a forward gain which replay can pass through.

**Reading 3 — bias modulation is inert where mechanisms work; it is not the missing lever.** d-bias is ±0.004
in every working cell (gain-synapse er: SGD +0.000, Adam +0.001; plast-synapse: −0.001..+0.031). The one large
effect is NEGATIVE: gain-synapse meta Adam −0.170 (per-task bias gains trained only by the meta-loss lag Adam's
fast backbone drift and mis-scale the forward). The head-bias residual leak is real but small (er-own gain-synapse
forget 0.0067 → 0.0044 with bias); it does not translate into accuracy.

**Reading 4 — plasticity stays a non-win everywhere (consistent with every prior plast result).** Best plast
cell vs its bar: standalone SGD +0.016, Adam +0.035 (both ≪ any gain cell); ALL plast er-own cells are ≤ ER
(−0.003..−0.112). The plast-neuron reverse ablation shows the head-column coupling was doing real work only under
Adam meta (+0.046 of the 0.4244); under SGD it is worth +0.006.

**Verdict.** ACCEPT-for-confirm (3-seed deferred) the gain-neuron 0,2,4 cells — the logit gate makes gain-neuron
strong in 3 of 4 arms and is the new parameter-efficient standalone result (0.9625 @ ~4k params). Bias modulation:
REJECT (inert or harmful; ablation closed). plast out/bias: REJECT. The pt5 headline (+ER Adam ≈ 0.99) is
unchanged — er-own Adam cells were already at ceiling. Caveats: ORACLE task id at train+eval (task-IL-style on the
class-IL metric); meta arms use the buffer (modulator-only replay — not apples-to-apples with no-buffer naive, and
beating ER is not beating ER); 1 seed; plast init 0.5.

### Follow-up: hid+bias (layers 0,2 + modulate_bias, HEAD bias excluded)

The main study only ran `modulate_bias` on the OUT stage (0,2,4+bias), so the head bias was always in.
This slots the missing cell — layers (0,2)+bias, hidden biases only — to isolate hidden-bias modulation
from the head-bias leak. Per-synapse only (8 cells + 4 baselines, all reproduce bit-exact). Files
`results/pt5_out_bias_hid.py`/`.log`. `d-hidbias` = (0,2)+bias minus (0,2) no-bias:

| cell | hid | hid+bias | d-hidbias | (out+bias) |
|---|---|---|---|---|
| gain-syn SGD meta | 0.9871 | 0.9871 | +0.0000 | 0.9892 |
| gain-syn SGD er | 0.7282 | 0.7282 | +0.0000 | 0.7348 |
| gain-syn Adam meta | 0.6304 | 0.6563 | +0.0259 | **0.5203** |
| gain-syn Adam er | 0.9900 | 0.9918 | +0.0018 | 0.9923 |
| plast-syn SGD meta | 0.6417 | 0.6417 | −0.0000 | 0.6440 |
| plast-syn SGD er | 0.6110 | 0.6106 | −0.0004 | 0.5837 |
| plast-syn Adam meta | 0.3861 | 0.3834 | −0.0027 | 0.4008 |
| plast-syn Adam er | 0.8935 | 0.8971 | +0.0036 | 0.9018 |

**Reading — hidden-bias modulation is INERT, and this DECOMPOSES the main study's bias result.** Under
SGD d-hidbias is exactly ±0.0000 (4 dp) in every cell; under Adam it is within noise (±0.003) in three
of four. The one non-trivial move is gain-syn Adam meta +0.026 — and it is the KEY comparison: the main
study's large negative (gain-syn Adam meta out+bias 0.5203, −0.170) is FULLY RECOVERED by excluding the
head bias (hid+bias 0.6563 > hid 0.6304). So that −0.170 collapse was ENTIRELY the head-bias gate (a
meta-trained head-bias gain lags Adam's fast backbone drift and mis-scales the logits), not hidden-bias
modulation. **Verdict unchanged: bias modulation REJECT** — inert where it touches only hidden biases,
net-negative where it reaches the head bias. The head bias is the only bias that matters and gating it
hurts. Caveats carry (oracle, meta uses buffer, 1 seed, plast init 0.5).

### Out-layer gain for the FIXED projections (iter1 disjoint, iter2 shared) — LABEL-ALIGNED P_out

Extends the out-layer modulation to the two fixed projections, gain per-neuron only (bias variants are
N/A for per-neuron gain — γ=0 freezes biases implicitly). class-IL, {sgd,adam} x {standalone (naive+
masked), er-own} x {disjoint,shared} x {hid=(0,2), out=(0,2,4)}. Files
`results/pt5_out_gain_neuron_fixed{,_aligned}.py`/`.log`.

**FIX shipped first.** The fixed-projection output-logit gate used a RANDOM P_out (random column
partition -> zeroed the wrong classes -> chance: disjoint out 0.0647/0.1937). GainDriverModulator now
builds P_out LABEL-ALIGNED for fixed projections: task t keeps EXACTLY its own class columns (== task-IL
gating), threaded from split_mnist.sequence via _build_model/_build_pt5_model. Learned P unchanged
(zero-init trainable); sequence=None keeps the legacy random build (unit-test only). Test
`test_gain_output_gate_label_aligned`; 51 pt5 tests pass. All 4 baselines reproduce bit-exact.

| opt | proj | arm | hid | out (aligned) | d-out |
|---|---|---|---|---|---|
| SGD | disjoint | standalone | 0.6225 | **0.8562** (f 0.0000) | +0.234 |
| SGD | disjoint | er | 0.8163 | 0.8342 | +0.018 |
| SGD | shared | standalone | 0.6597 | **0.9578** (f 0.008) | +0.298 |
| SGD | shared | er | 0.9658 | 0.9649 | −0.001 |
| Adam | disjoint | standalone | 0.9953 | 0.9956 | +0.000 |
| Adam | disjoint | er | 0.9948 | 0.9960 | +0.001 |
| Adam | shared | standalone | 0.6817 (f 0.293) | **0.9768** (f 0.021) | +0.295 |
| Adam | shared | er | 0.9929 | 0.9928 | −0.000 |

**Reading.** (1) The aligned logit gate helps exactly where the shared head was NOT calibrated — the
STANDALONE (no-replay) arm: SGD disjoint +0.234, SGD shared +0.298, Adam shared +0.295 (forgetting
0.29→0.02). The hidden freeze already preserves each task's features; the missing piece is cross-task
head competition, which zeroing the 8 competitor logits supplies. (2) Where hid was already at ceiling
— every +ER cell (replay calibrates the head) and Adam-disjoint standalone (the freeze alone hit 0.995)
— the output gate adds ~0. (3) **The aligned gate == task-IL masking** (keep task t's 2 columns, zero
the other 8) under the oracle, so gain-neuron+out is a TASK-IL-style result on the class-IL metric, not
a class-IL solution. Its headline: standalone (NO replay) now reaches 0.86–0.996 because the disjoint/
shared freeze + task-IL output gate together need no replay. Same oracle + 1-seed caveats. This is the
`P_out must be built from the sequence` fix the earlier gotcha flagged, now done.

### Buffer-size sweep — standalone (gate-only) vs er-own (backbone replay), out stage, class-IL

Sweeps 4 buffer sizes {200, 500, 1000, 2000} for gain {per-neuron, per-synapse} x {sgd, adam} on the
out stage (learned P), for two arms with a KEY structural difference: standalone buf-meta-own trains
ONLY the gate P (main net naive, buffer-free), so size affects gate CALIBRATION; er-own feeds the
BACKBONE (replay, each sample via its own P[j]), so size affects what the network RELEARNS. Files
`results/pt5_out_meta_buffer_sweep.py`/`.log` (standalone) and `results/pt5_out_buffer_sweep_er.py`/`.log`
(reruns standalone buf=1000 + full er-own). All 4 CL baselines reproduce; every fresh buf=1000 cell
reproduces the prior logs bit-exact (8/8 cross-checks OK). seed 42, lr 1e-3, ep 5.

STANDALONE (gate-only): SGD gain-neuron 0.9459→0.9570→0.9625→0.9591, gain-synapse flat ~0.989; Adam
gain-neuron 0.6109→0.7227→0.7229→0.7256, gain-synapse flat ~0.71. ER-OWN (backbone): SGD gain-neuron
0.9189→0.9414→0.9466→0.9452, gain-synapse flat ~0.72-0.74; Adam gain-neuron 0.9665→0.9858→0.9875→**0.9938**,
gain-synapse 0.9810→0.9885→0.9915→0.9922.

**Reading.** (1) **Standalone saturates by ~500 and plateaus; er-own keeps climbing.** Under Adam,
standalone gain-neuron saturates at ~0.72 by buf=500 and never moves (gate calibrates but the un-replayed
backbone drifts), whereas er-own gain-neuron rises monotonically to 0.994 at buf=2000 — the buffer
refreshes the backbone, so more of it is genuinely better. This is the mechanism made visible:
gate-calibration saturates fast, backbone-replay quality scales with buffer. (2) **The Adam standalone
ceiling (~0.72) is backbone-drift-bound, not buffer-bound** — no size fixes an un-refreshed fast-drifting
backbone. (3) **SGD needs almost no buffer** (gain-synapse flat 0.989 even at 200; its 5.6M-param P
calibrates on few exemplars). (4) The apparent non-monotonicities (SGD gain-neuron peak at 1000; Adam
gain-synapse dip at 1000) are within 1-seed noise on saturated curves — past saturation, more buffer
just reshuffles samples, no signal. (5) **gain-synapse+ER-SGD is the dead cell** (~0.72-0.74, barely
above the ER-SGD floor 0.7226; the per-synapse head gate fights SGD replay). Oracle + 1-seed caveats.

### Compute + memory cost of each retention lever (vs the cheapest gain cell)

Reproducer `results/pt5_compute_memory.py`/`.log` (ep=1, 1 seed): marginal cost of each axis vs baseline
gain per-neuron SGD standalone no-buffer. Two axes — ep=1 wall-time, and TOTAL resident memory (float32
counts = base MLP 478,410 + modulator P + optimizer state [SGD ~0; Adam = 2x trained params] + buffer
[er_buffer_size x 784] + EWC state).

| config | wall (xtime) | total floats | xmem |
|---|---|---|---|
| baseline gain-neu SGD standalone | 12.7s (1.0x) | 482,410 | 1.0x |
| + Adam | 13.0s (1.03x) | 1,447,230 | 3.0x |
| + buffer, standalone meta | 23.5s (1.85x) | 1,274,410 | 2.6x |
| + buffer, ER | 59.3s (4.66x) | 1,266,410 | 2.6x |
| gain-synapse | 15.3s (1.20x) | 2,846,410 | 5.9x |
| **EWC** (SGD standalone) | 13.2s (1.04x) | **5,262,510** | **10.9x** |

**Reading.** Adam is ~free on time but **~3x memory** (2x the trained params in moments — scales with P).
The buffer costs compute + ~784k floats (not params); ER's compute is ~4.7x (batch-doubling + the
er_task_id per-task split), the standalone meta-loop ~1.9x. gain-synapse costs **params (~590x the
per-neuron P: 2.37M vs 4k)**, not time (+20%). **EWC is the memory-heaviest (~11x)** — from per-task
Fisher+theta* state (each 2x the model), which uniquely **scales LINEARLY with the task count** (Adam/
buffer are constant in T) — and it FAILS class-IL (~chance), so it is the one lever that is both most
expensive and non-working here, which is why pt5 went to the disjoint gain freeze (+4k params, ~free)
and replay instead. 1 seed, ep=1; wall-times indicative.

### Iteration 3 follow-up — sparsity (gate L1) x three gain arms x {sgd,adam} (user-requested)

**Status:** `continues iter3-followup (B), which swept the L1 gate-sparsity penalty for gain only under
ADAM and only {standalone,+ER} (+ER used the OLD er_task_id=OFF). This fills the gaps: SGD gain
sparsity, the buff-own meta arm with sparsity, and er-own (correct per-sample routing, now the
default). Verdict UNCHANGED: sparsity is a real STANDALONE lever and now also boosts the buff-own meta
arm, but there is STILL NO sparsity+ER win — er-own is inert to the penalty. Files
results/pt5_sparsity_arms.py/.log.`

Grid: gain {neuron,synapse} x {sgd,adam} x arms {standalone (no buffer, P via main loss), buff-own
(--neuromod-meta-replay --neuromod-er-task-id, P via a per-task modulator-only replay meta-loss), er-own
(method er --neuromod-er-task-id, P via replay-augmented main loss)} x per-granularity lambda grid.
class-IL, learned proj, gain_form=unbounded, 1 seed (42), lr=1e-3 ep=5 buffer=1000. Per-granularity
grids (per-synapse fan-in ~10x -> higher useful lambda): neuron {0,.1,.3,1,3}, synapse {0,1,3,10,30}.
All 12 lambda=0 cells run LIVE and reproduce results/pt5_gain_forms_buffer.log (unbounded) bit-exact.

Baselines: SGD naive 0.6296 / er 0.7226 ; Adam naive 0.3894 / er 0.9053. d0 = delta vs the arm's own
lambda=0 anchor.

| opt  | gran    | lambda | standalone (d0) | buff-own (d0)   | er-own (d0 / vs er) |
|------|---------|--------|-----------------|-----------------|---------------------|
| sgd  | neuron  | 0.0*   | 0.6311          | 0.9074          | 0.7376 (+0.015)     |
| sgd  | neuron  | 0.1    | 0.6311 (+0.000) | **0.9670 (+0.060)** | 0.7376 (+0.015) |
| sgd  | neuron  | 0.3    | 0.6311 (+0.000) | 0.9582 (+0.051) | 0.7376 (+0.015)     |
| sgd  | neuron  | 1.0    | 0.6312 (+0.000) | 0.5246 (−0.383) | 0.7370 (+0.014)     |
| sgd  | neuron  | 3.0    | 0.6311 (−0.000) | 0.3028 (−0.605) | 0.7358 (+0.013)     |
| sgd  | synapse | 0.0*   | 0.6295          | 0.9871          | 0.7282 (+0.006)     |
| sgd  | synapse | 1.0    | 0.6295 (+0.000) | 0.9844 (−0.003) | 0.7282 (+0.006)     |
| sgd  | synapse | 3.0    | 0.6295 (+0.000) | 0.9785 (−0.009) | 0.7282 (+0.006)     |
| sgd  | synapse | 10.0   | 0.6295 (+0.000) | 0.9762 (−0.011) | 0.7281 (+0.006)     |
| sgd  | synapse | 30.0   | 0.6295 (+0.000) | 0.9617 (−0.025) | 0.7281 (+0.006)     |
| adam | neuron  | 0.0*   | 0.3770          | 0.5075          | 0.9887 (+0.083)     |
| adam | neuron  | 0.1    | 0.5165 (+0.140) | **0.6990 (+0.192)** | 0.9900 (+0.085) |
| adam | neuron  | 0.3    | **0.6672 (+0.290)** | 0.5486 (+0.041) | 0.9867 (+0.081) |
| adam | neuron  | 1.0    | 0.5668 (+0.190) | 0.6374 (+0.130) | 0.9883 (+0.083)     |
| adam | neuron  | 3.0    | 0.5034 (+0.126) | 0.6477 (+0.140) | 0.9810 (+0.076)     |
| adam | synapse | 0.0*   | 0.4202          | 0.6304          | 0.9900 (+0.085)     |
| adam | synapse | 1.0    | 0.5898 (+0.170) | **0.8878 (+0.258)** | 0.9912 (+0.086) |
| adam | synapse | 3.0    | 0.7330 (+0.313) | 0.8384 (+0.208) | 0.9913 (+0.086)     |
| adam | synapse | 10.0   | **0.7632 (+0.343)** | 0.7677 (+0.137) | 0.9903 (+0.085) |
| adam | synapse | 30.0   | 0.5659 (+0.146) | 0.6237 (−0.007) | 0.9894 (+0.084)     |

**Finding 1: er-own is INERT to sparsity in all four cells (the robust +ER null).** er-own sits at its
lambda=0 value (~0.73 SGD / ~0.99 Adam) and the L1 moves it by <=+-0.008, sagging slightly at high
lambda. This confirms iter3-followup's "sparsity does NOT help +ER" now under the CORRECT per-sample
routing (er_task_id=ON, the current default) and BOTH optimizers — not just the old wrong-task Adam
pair. Once replay calibrates the shared head, the extra sparsity constraint only costs capacity. No
learned sparsity+ER win exists.

**Finding 2: buff-own is the arm where sparsity helps under SGD (new).** SGD gain-neuron buff-own peaks
at lambda=0.1 -> 0.9670 (+0.060 over an already-strong 0.9074), and Adam gain-synapse buff-own at
lambda=1 -> 0.8878 (+0.258). Mechanism: in buff-own P is trained by the modulator's OWN Adam
meta-optimizer, so the mean-normalised L1 bites regardless of the MAIN-net optimizer — unlike
standalone under SGD (see Finding 3). Bumps are modest on top of already-high anchors, and gain-synapse
buff-own is already near ceiling (0.987) so sparsity only hurts it there.

**Finding 3: standalone reproduces the followup and closes its open bracket.** Adam gain-neuron
inverted-U peak at lambda=0.3 (0.6672) confirmed; Adam gain-synapse — which the followup left "still
rising at lambda=10 (0.7632)" — now PEAKS at lambda=10 and drops at lambda=30 (0.5659), so the peak is
bracketed. SGD standalone is fully INERT to sparsity (0.6311/0.6295 across all lambda): P is trained by
the MAIN SGD optimizer, which under 5 epochs barely moves P, so the L1 changes nothing (this is WHY the
followup only ran sparsity under Adam).

**Finding 4: over-sparsification collapse past the peak.** buff-own SGD gain-neuron craters to 0.3028 at
lambda=3 with forgetting -> 0.000 — the over-suppression signature (under-learning masquerading as
retention). Every arm's curve is an inverted-U once its optimizer/granularity lets the L1 bite.

**Verdict.** Sharpens iter3-followup (B) without overturning it. Reportable pt5 headline unchanged
(iter-1 disjoint gain+ER). New reportable standalone peak is bracketed (Adam gain-synapse lambda=10
0.7632). buff-own+sparsity is a mild new lever (SGD gain-neuron 0.9670) but rides the buffer + oracle.
No sparsity+ER win at any lambda/opt/granularity. Same oracle caveat (task-id at train+eval); 1 seed;
SGD lambda scale was exploratory. The `positive` gain-form x lambda>0 cell (vanishing-L1-pull
prediction) is STILL untested — this sweep pinned gain_form=unbounded throughout.

## pt5 DRIVER REPRESENTATION — is one-hot replaceable, can we drop the eval oracle? (user-requested)

**Question.** The pt5 driver is a one-hot task id → projection `P` (gate = per-task lookup row). What
else could the driver be, and does any alternative survive **without the task id at eval**? Standalone
exploration; scripts consolidated in `results/pt5_driver_repr.py` + `.md`. Gain-neuron, unbounded
`g=1+raw` on h0,h1, class-IL Split MNIST, seed 42, lr 1e-3, ep 5, buffer 1000. **1 seed; ORACLE caveat.**

**Drivers.** `onehot` (`raw_t=P[t]`, independent rows), `lin` (`raw_t=μ_t@W`, shared 784→800 linear
over the task-mean image), `mlp` (`raw_t=gf(relu(gh(μ_t)))`, 784→128→800), each ±CENTERING
(`μ_t−mean_t μ_t`; inter-task cosine of the mean images 0.82→−0.24). Arms: buf-own (naive main + per-task
replay meta-loss on the gate) and er-own (main+gate jointly on the ER batch, own-task gating). Baselines
reproduce the pt5 harness (naive-sgd 0.63/adam 0.39, er-sgd 0.71/adam 0.89).

**Finding 1 — one-hot's edge is ORTHOGONALITY, not the lookup.** One-hot's independent rows differentiate
for free (the one-hot zeroes every other row's gradient). A RAW content driver COLLAPSES (cos(dev)→+0.99,
buf-own/sgd lin~0.35/mlp~0.33 ≪ onehot 0.87): the shared "average-digit" component dominates and a learned
`W` amplifies it (|dev|→~11). CENTERING rescues `lin` (a linear map then passes the −0.24 geometry
through): cos→−0.23, acc→0.82 ≈ onehot. `mlp` only partial (cos→+0.40) — its relu re-correlates the
centered features into the positive orthant, so `lin` > `mlp` throughout.

**Finding 2 — centered `lin` MATCHES one-hot UNDER THE ORACLE.** er-own/adam lin_c 0.989 ≈ onehot 0.991
(≈ pt5's er+gain ~0.99); er-own/sgd lin_c 0.966 > onehot 0.729 (one-hot gate inert under sgd). buf-own/sgd
is high-variance (lin_c over seeds 42–46 = 0.773±0.083, range 0.61–0.83; onehot 0.895±0.014 — the
differentiation cos≈−0.23 is stable every seed, only the standalone accuracy is noisy). Per-synapse same in
working regimes but its lin_c projection is 784×n_syn≈125M params (260× the net) — does not scale.

**Finding 3 — DROP THE ORACLE → below ER (the real result).** With no task id at eval, `per-image` (gate
from the test image) and `nearest` (nearest-prototype task inference; infer acc 0.759 = ceiling) both
collapse: er-own/adam oracle 0.991→0.708/0.755, and `nearest ≈ oracle×infer` (0.991×0.759=0.752) because
the disjoint gate has ZERO tolerance for the 24% misrouted samples. NO non-oracle cell beats plain ER-adam
(0.894); best is 0.778. mlp identical. The gate is TASK-CONDITIONED, so an oracle-free protocol must add a
task-inference stage (re-hits the pt3-Iter-8 routing wall) — the EVAL is class-IL throughout (10-way
argmax, no task id); only the `oracle` column is a task-IL-STYLE result on that class-IL metric. The
oracle, not the driver representation, carried the ~0.99.

**Methodology.** Overlap is measured on `dev(=raw)`, NOT `g(=1+raw)`: the shared parity 1 inflates
cos(g)→+1 for gentle gates (one-hot cos_dev+0.21 vs cos_g+0.79; er-own/sgd lin_c cos_dev−0.22 vs cos_g+0.99
at |dev|=0.06; inflation tracks |dev|). `cos(dev)` strips the task-independent offset and is invariant to
gate strength.

**Verdict.** The one-hot's essential property is that it makes the per-task gates orthogonal/independent;
you can recover that from a decorrelated content driver (centered mean image + linear map), matching
one-hot UNDER THE ORACLE. But no driver removes the task-inference dependency — every oracle-free eval
falls below plain ER. Reportable pt5 gain result stays iter-1 disjoint gain+ER; the driver axis opens no
class-IL win.

## pt6 — content & inference-net mechanisms + eval-resolution axis (user-requested; `@SPEC-proto-pt6.md`)

**Goal.** pt5's driver study showed a centered mean-image mechanism matches one-hot UNDER THE ORACLE but
every oracle-free eval (per-image, hard nearest) fell below ER — capped by nearest-prototype task
inference (~0.76). pt6 adds the SOFT resolution it lacked and two mechanisms that LEARN the task
selection, to test whether the gate can be made genuinely oracle-free. Study module
`results/pt6_driver_mechanisms.py` + `.md` + `.log`. Gain-neuron on (h0,h1,out), class-IL, seed 42,
lr 1e-3, ep 5, buffer 1000, 1 seed. Baselines reproduce the pt5 harness (naive-sgd 0.629/adam 0.390,
er-sgd 0.723/adam 0.895).

**Axes.** Driver = `task_id` (unchanged). Mechanisms: `mean_image` (proj(μ_t), lin/mlp, ±center),
`soft_mlp` (onehot-style gate P + inference net g(x):784→128→5 trained WITH REPLAY; eval blends
Σ softmax(g(x))·P[t]), `embedding` (gate = proj(g's 128-d hidden), per-image, oracle-free). New
EVAL-RESOLUTION axis for mean_image: oracle / per-image / nearest / soft-nearest(τ). Arms buf-own,
er-own; baselines naive/er; opt sgd/adam.

**Finding 1 — learned inference ≫ prototype inference.** soft_mlp's g(x) reaches 0.86–0.88 task-acc vs
nearest-prototype's fixed 0.759. The gate has ~zero misrouting tolerance (nearest ≈ oracle×infer), so
lifting infer 0.76→0.88 is what lifts the oracle-free acc ~0.75→~0.88. soft-nearest(τ) confirms softness
helps (sharp τ=0.03 beats hard nearest, mean_image/lin er-own/sgd 0.819 vs 0.775) but stays
prototype-capped (≤~0.82); the LEARNED selector is the real lever.

**Finding 2 — oracle-free ER parity (the headline).** soft_mlp er-own/adam soft-blend 0.885 (infer 0.884)
≈ er-adam 0.895; embedding er-own/sgd/lin 0.889 ≈ er-adam. Same-optimizer: under SGD the learned mechs
BEAT ER (embedding 0.889 / soft_mlp 0.856 vs er-sgd 0.723, +0.13..+0.17); under Adam they MATCH ER (0.88
vs 0.895). Parity, not a new lever over replay — but genuinely oracle-free, which pt5's ~0.75 ceiling was
not. This overturns the "well below ER" half of pt5's conclusion.

**Finding 3 — embedding is the cleanest.** Per-image continuous, oracle-free BY CONSTRUCTION (no discrete
task inference), 0.889 (er-own/sgd); lin ≈ mlp. Standalone also works: soft_mlp buf-own/sgd 0.856
oracle-free with a NAIVE backbone (replay only on gate + inference net) ≫ naive 0.629 / er-sgd 0.723.

**Finding 4 — mean_image reproduces pt5.** Centered lin er-own = onehot under oracle (0.98/0.99);
buf-own/adam and uncentered-lin collapse; mlp ≤ lin (relu re-correlation).

**Verdict.** The task-inference dependency pt5 flagged is real but NOT a hard wall — a replay-trained soft
inference (soft_mlp) or a learned per-image embedding reaches ER parity ORACLE-FREE (~0.88) vs pt5's
prototype-capped ~0.75. The reportable class-IL lever is still ER (matches, doesn't beat), but the gate
mechanism is now genuinely oracle-free. Synapse deferred (rationale corrected below). 1 seed; buf-own
high-variance (report ≥3 seeds if headlining).

**Synapse deferral — corrected rationale.** The original note ("per-sample gates conflict with
per-image/soft resolution + 374M-param content projection") welded two independent blockers together and
over-applied both. (a) The projection blow-up is `mean_image`/lin (784×477 600 ≈ 374M) and `embedding`/lin
(128×477 600 ≈ 61M) ONLY — `onehot` and `soft_mlp` gate via a `(T, n_syn)` ≈ 2.4M LOOKUP, no projection at
all (pt5 already ran per-synapse onehot). (b) The per-sample Γ expansion — `(B, d_out, d_in)`, ~160 MB for
layer 0 at B=128, vs one shared `(Γ⊙W)` matmul when the batch shares a gate — blocks TRAINING only for
`embedding`, whose `train_gate` IS `gate_per_sample`. `soft_mlp` trains on true task ids (`P[tids]`), so
even the mixed current+replay er-own batch holds ≤T=5 distinct gates and groups into ≤5 masked matmuls
(exactly the pt5 `er_task_id` path); only its soft-blend EVAL needs per-sample Γ, and that chunks down
under `no_grad` (bit-identical — eval is per-sample independent; the cost is wall-clock, not correctness).
**Net: per-synapse `soft_mlp` was RUNNABLE and simply not run** (oracle-only synapse cells were skipped as
redundant with pt5); the deferral is justified only for `mean_image`/`embedding`. The SPEC's rank-64
low-rank content projection cuts the parameter count but NOT the per-sample expansion (64 masked matmuls
per layer per batch).

## pt6 follow-ups — seven isolated probes (A–G) (user-requested)

Files `results/pt6_followups.py` / `pt6_followups2.py` (+ `.md`, `.log`). Each probe isolated (no
cross-product). soft_mlp unless stated; gain-neuron on (h0,h1,out); class-IL; seed 42, lr 1e-3, ep 5,
buffer 1000; **1 seed**. Refs naive-sgd 0.629, er-sgd 0.723, er-adam 0.895.

**NOISE FLOOR (measured).** Identical configs re-run differ by ~0.007–0.016 (MPS nondeterminism;
parity λ=10 gave oracle 0.962 then 0.978). At 1 seed, treat anything under ~0.02 as null. Several
probes below are null precisely on that basis.

**A. soft-nearest(τ) — interior peak, and τ→0 → hard-nearest.** mean_image/mlp/cen er-own/sgd:
0.757@.003, 0.767@.01, **0.777@.03**, 0.765@.05, 0.692@.1, hard 0.751. τ≈0.03–0.05 is a genuine peak;
τ→0 converges to hard exactly as theory predicts (validates the implementation). Soft beats hard only
under SGD (+0.026); under Adam it converges to hard. Nothing below τ=0.03 to gain.

**B. Replay is what makes the selector work.** no-buffer: `infer` → **0.198 ≈ chance (1/5)** and
oracle-free 0.463, while the ORACLE stays 0.933 — the gate is fine, the selector forgot. `buf-cur`
(wrong-task gating in the meta-loss) costs −0.17 soft / −0.22 oracle: own-task gating matters.

**C. soft ≈ hard for a LEARNED selector.** ±0.017, no consistent direction, all inside noise. A
well-trained selector is confident (softmax ≈ one-hot) so the blend ≈ the argmax. Softness only pays
when the posterior is diffuse (the prototype case A) — do not generalise "soft helps" to a learned net.

**D. The selector does not need true task labels.** Trained on pseudo-labels from the main net's out
layer (argmax(logits)//2, detached): infer 0.882 vs 0.884, soft 0.854 vs 0.856 (sgd). Caveat: this does
not make the method label-free — the gate table and backbone still use true labels.

**D2. Making the TRAIN driver the soft posterior does NOT help.** er-own null (±0.008), buf-own/sgd
−0.103, and the oracle drops in every cell (up to −0.14). `P[t_true]` gives each row a clean UNMIXED
gradient (the one-hot "independent rows" property); a blend smears each sample across all rows so they
differentiate less. The train/eval mismatch was never the problem.

**E. Sparsity λ·mean|1+P| is ineffective.** mean|P| 0.06→1.0 with accuracy flat (0.879–0.887): a scale
degeneracy — the jointly-trained backbone rescales W to absorb the gate's scale, so L1 on γ just shifts
magnitude between γ and W without changing the function.

**F/G. Per-layer gate study — corrects a mis-reading.** `mean|P|` averages 4050 entries and hides the
10-wide out gate. er-own/sgd: h0 0.001 / h1 0.002 / **out 0.107** — so the er-own gate is a pure
per-task LOGIT adjustment (the class-IL bottleneck), which explains the er-own/sgd gap (0.856 vs
er-sgd 0.723) that mean|P|=0.003 made look like "gate ≈ parity". The two arms use different layers:
with ER the gate is out-layer (replay handles features); standalone buf-own it is hidden-layer
(0.516/0.545 vs out 0.251). The parity penalty crushes hidden ~100× but the out gate resists (~6×) —
it is load-bearing; even fully crushed the oracle loses only 0.012 and oracle-free is flat.
**Always report gate magnitude per layer, never as a single mean.**

**Net.** The pt6 headline survives and is localised: the win needs (i) replay for the selector,
(ii) unmixed per-task training of the gate rows, (iii) a per-task out-layer logit adjustment under ER.
What does not matter: soft-vs-hard, the selector's label source, sparsity, and most of the gate's
magnitude. 1 seed; <0.02 is noise.

## pt6 gain-SYNAPSE for soft_mlp (soft-blend + hard-argmax) (user-requested)

Files `results/pt6_synapse.py` / `.md` / `.log`. Gated net0(400x784)+net2(400x400)+net4(10x400),
n_syn=477,600; gate table P:(T,n_syn)=2.4M (a LOOKUP — the 374M blow-up is mean_image/embedding only);
Gamma=1+P; class-IL, seed 42, lr 1e-3, ep 5, buffer 1000, 1 seed. Refs er-sgd 0.723, er-adam 0.895;
NEURON soft er-own/adam 0.886, er-own/sgd 0.856, buf-own/sgd 0.856.

**The per-sample-Gamma blocker was over-stated — the soft blend needs NO expansion.** Since
Gamma_i = sum_t p_it Gamma_t and (Gamma.W)x is LINEAR in Gamma:
    (Gamma_i.W)x_i + b = sum_t p_it [ (Gamma_t.W)x_i + b ]      (bias rides along, sum_t p_it = 1)
so a blended layer is the p-weighted sum of the T per-task gated outputs: T=5 matmuls per layer, no
(B,d_out,d_in) tensor, and EXACT — verified vs the grouped path, max|delta| = 2.1e-07. hard-argmax is
cheaper still (one task per sample -> <=5 masked matmuls, the training path). This supersedes the
earlier "only the soft-blend eval needs a per-sample Gamma, which chunks under no_grad" caveat.

Results (oracle / soft / hard / infer):
    er-own  sgd   .748 .745 .746 .892
    er-own  adam  .991 .890 .888 .892
    buf-own sgd   .989 .887 .878 .883
    buf-own adam  .649 .598 .605 .883

**Finding 1 — synapse matches neuron under ER, BEATS it standalone.** er-own/adam soft 0.890 ~ neuron
0.886 (inside the ~0.02 noise floor); buf-own/sgd soft 0.887 vs neuron 0.856 (+0.031) with oracle
0.989 vs 0.939 (+0.050). Per-synapse is the better STANDALONE mechanism. buf-own/adam collapses
(0.598) exactly as at neuron granularity (0.661) — that arm/optimizer pairing is bad regardless.

**Finding 2 — er-own/sgd is INERT for an OPTIMIZER-ROUTING reason, not a synapse one.** In er-own the
gate rides the MAIN optimizer, and SGD@1e-3 cannot move 2.4M gate params in 5 epochs: |P| stays
~1e-5..7e-4, so there is no modulation and every mode collapses to the same number (.748/.745/.746 ~
plain er-sgd .723). In buf-own the gate has its OWN Adam optimizer, which is why buf-own/sgd trains
(|P| 0.12-0.17) and reaches 0.887. If a per-synapse gate looks inert, check which optimizer owns it
before concluding the mechanism failed.

**Finding 3 — soft ~ hard replicates** (.890/.888, .887/.878, .745/.746), now confirmed at BOTH
granularities: a confident learned selector makes softmax ~ one-hot so blending buys nothing.

**Finding 4 — the per-layer localisation rule replicates.** With ER the gate concentrates in the OUT
layer (er-own/adam net4 7.9e-2 vs hidden 1.4-2.1e-2); STANDALONE it moves to the HIDDEN layers
(buf-own/sgd net2 1.7e-1 > net0 1.2e-1 > net4 6.2e-2). Same split as neuron — replay handles features
so the gate only recalibrates logits; without replay the gate must modulate features itself.

**Net.** Per-synapse soft_mlp is fully runnable and is the strongest standalone pt6 cell (buf-own/sgd
0.887 oracle-free, oracle 0.989), while under ER it ties neuron. It adds NO new oracle-free ceiling —
still ~ER parity (0.89) — consistent with the pt6 verdict. 1 seed; <0.02 is noise.

---

## pt7 — the four classic neuromodulators as PRE-FORWARD gate drivers (`@SPEC-proto-pt7.md`)

`results/pt7_neuromodulators.py` / `.md` / `.log` (+ `pt7_results.tsv` ledger, `pt7_make_table.py`).
class-IL Split MNIST, gain (h0,h1,out), seed 42, lr 1e-3, ep 5, buffer 1000, 1 seed. New DRIVER: replace
pt5/pt6's `task_id` with the four neuromodulators DA/ACh/NE/5-HT as difficulty/novelty/uncertainty/reward
signals. Constraint: every signal is computable BEFORE the forward (a head `m_k(x)` regresses a per-sample
biological `τ_k`, trained WITH REPLAY; NE_emb = last-hidden novelty gating the out layer), so both gain
granularities work via the **rank-K linear gate** `Γ_i = 1 + Σ_k m_ik P_k` (K+1 matmuls, `P` = `(K,n_syn)`
≈ 1.9M, no 374M content blow-up). Arms nobuf / buf-own / er-own; controls free + 5ht-const; eval pred
(oracle-free) / true (2-pass, labels) / probe (task-decodability). Baselines naive 0.629/0.390,
er 0.723/0.895 reproduce the harness.

**Pre-registered prediction CONFIRMED — a controlled NEGATIVE.** No biological driver beats ER: every
er-own cell is within ±0.02 of same-opt ER, none positive beyond noise (DA 0.723/0.879, ACh 0.713/0.884,
NE 0.721/0.891, 5HT 0.722/0.897, all4 0.706/0.882; synapse matches: DA 0.723/0.899, all4 0.726/0.892). And
NOTHING standalone: nobuf ≈ naive (best all4-sgd 0.635 ≈ 0.629) — a *difficulty* gate, unlike pt6's
*task-selective* soft_mlp (buf-own 0.856), does nothing without replay.

**Four controls close it.** (1) PROBE ≪ pt6: task-decodability of m(x) is 0.21–0.52 (all4 highest) vs pt6
infer 0.88 → the code is ~task-agnostic → no class-IL lever (the mechanistic "why"). (2) `free` → gate 0
(|g|=0.000, reproduces baseline bit-exact) → the ≈-baseline bio cells aren't riding gate capacity. (3)
`true` ≤ `pred`: the REAL signal is no better, often worse (NE true 0.638 vs pred 0.891; all4 0.691 vs
0.882) — no "better head" ceiling to chase; the head's smoothing is what holds pred at ER. (4) `5ht-const`
is a null (er-adam 0.886 ≈ ER).

**Per-layer emergence = the ARM, not the neuromodulator.** Theory's ACh→h0 / NE→out specialization does NOT
appear; the CL structure dominates (pt6-followup-F/G): er-own puts the gate in the OUT layer for every
driver (ACh-sgd out 0.083, all4-sgd out 0.138), buf-own moves it to HIDDEN (ACh-sgd 0.567/0.615/0.379).
Read |gate| per layer.

**Synapse rank-K validated** (≈ neuron in working regimes; the pre-forward linearity is the point). **Tonic
scalar variants degenerate**: ACh_vol / NE_rise have ~zero per-sample variance, so per-driver
standardization divides by ~0 → |g| blows to 10–17 and accuracy collapses (ACh_vol 0.184/0.098; NE_rise
0.098 nan under SGD, 0.857 under Adam whose moment-norm absorbs it) — a signal with no per-sample content
cannot drive a per-sample gate. DA_step (per-sample) is fine (≈ ER).

**Net.** The neuromodulator mechanism is sound and synapse-tractable, but the *signal* (difficulty/reward)
carries none of the retention/selection information the class-IL head bottleneck needs. Consistent with the
whole pt2→pt6 arc: replay (or a task-selective gate) is the lever; a modulatory difficulty code is not. The
one honesty gain over pt5: this is oracle-FREE by construction. Reportable class-IL headline unchanged
(pt6 oracle-free ER-parity selector / pt5 disjoint gain+ER under oracle). 1 seed; buf-own high-variance
(scattered >0.02 buf-own cells are naive-backbone noise, all ≪ ER).

### pt7 VARIANTS (user-requested follow-up: `results/pt7_variants.py`/`.md`/`.log`, 152 cells)

Standard regime + new drivers + NE multidim/double-forward + standardization & mean-mode ablations. Same
class-IL harness (seed 42, lr 1e-3, ep 5, buffer 1000, 1 seed); baselines reproduce (naive 0.629/0.390,
er 0.723/0.895).

**A. STANDARD regime (full MNIST):** all4-gate ≈ vanilla (sgd +0.0047, adam +0.0038) — the neuromodulation
gate does not hurt standard accuracy (goal #2). Untuned matched budget, not a reportable standard number
(rule #2 needs a separate standard sweep).

**B. New head drivers ≈ baselines:** DA_fast=(loss−ema_fast)/ema_fast, ACh_vol_ps=|loss−ema_fast|
(per-sample) are stable ≈ ER/naive. ACh_ema=ema(entropy), 5HT_ema=ema_slow(−loss) are tonic scalars → see C.

**C. STANDARDIZATION ablation (clean):** tonic scalar drivers (ACh_ema, 5HT_ema, and pt7's ACh_vol/NE_rise)
COLLAPSE under SGD *with* standardization (er-sgd 0.098, nan |g| — standardizing a near-constant divides by
~0), but are fine *without* it (ACh_ema std-OFF er-sgd 0.759). NE_rise std-OFF is INERT (|g|=0, = baseline):
the head learns a ≈0 gate. So standardization TURNS "inert" INTO "catastrophic" for a tonic driver. Rule:
standardize per-sample drivers; never a tonic/scalar one. Per-sample/multidim drivers: std-ON ≈ std-OFF.

**E. NE double-forward / multidim novelty — the one notable effect, still a NON-WIN.** Head-free novelty
gates (emb_all scalar, vec_h1 400-d, vec_h1proj h1→32, vec_x 784-d input, vecproj input→32) under SGD+er-own
lift ER-sgd 0.723 → 0.79–0.86 (vec_x +0.140, vec_h1 +0.139, emb_all +0.122, NE_emb-stdoff +0.108). BUT:
(1) below the best baseline (Adam-ER 0.895; under Adam these add nothing) — the pt5/pt6 SGD-underfitting-
closure pattern; (2) INPUT novelty (vec_x) ≈ EMBEDDING novelty (vec_h1) and more dims → more boost ⇒
CAPACITY, not novelty semantics; (3) hurts without replay (nobuf/buf-own below naive; vec_h1 nobuf-adam
0.19). emb_all (head-free) ≈ NE_emb (head-based) ⇒ for a novelty driver the head is a design choice, not a
necessity (only loss/entropy drivers need one at eval).

**F. Cumulative mean vs EMA:** the SGD+ER boost SURVIVES a cumulative (true) mean (vec_h1 sgd 0.863→0.861),
so it is not an EMA artifact. But cumulative mean is fragile for the DRIFTING h1 — vec_h1 cumulative+Adam
COLLAPSES to 0.101 (the cumulative mean_h1 lags the representation drift, so h1−mean_h1 grows biased and
destabilises Adam) — while the STATIONARY input vec_x is mean-mode-agnostic (0.836→0.811). EMA is required
for embedding novelty; input novelty is not — reinforcing the capacity reading.

**Verdict:** nothing beats Adam-ER 0.895; standard gate harmless; tonic drivers degenerate; NE-novelty gates
give a real but explained SGD-only capacity boost, harmful without replay. vec_h1/vec_x SGD+ER (+0.14) are
the strongest pt7 numbers, need 3 seeds. Project class-IL headline unchanged.

**G. Split optimizer (main=Adam, neuromod gate+head=SGD), standardized:** isolates the SGD boost's source.
vecproj/NE_emb er-own → 0.8914/0.8886 ≈ ER-adam (0.895), NOT the +0.07/+0.11 of full-SGD (0.795/0.832). So
the boost was ENTIRELY a MAIN-net-optimizer artifact (SGD under-fits the head; the gate closes the gap; with
Adam there is no gap so the gate — Adam or SGD — adds nothing, |g|~0.002-0.04). The gate optimizer is
irrelevant; confirms "capacity closing SGD's under-fit, not a class-IL lever." nobuf/buf-own ≈ naive-adam.

**A (extended) — STANDARD free/vecproj:** free drives the gate to 0 (|g|=0.000, ~vanilla: no capacity win in single-task either); vecproj is the only standard cell that mildly HURTS (-0.0018/-0.0030) and has the LARGEST gate (|g|~0.27) — a headless input-novelty gate injects noise a single-task net doesn't use; all4 (biological, |g|~0.10) mildly helps. All within +-0.005 of vanilla => gate neutral for standard (goal #2).

**H/I/J (user-requested follow-ups: gain-syn, tuned-standard, stateful/GRU).**
- **H. CL gain-SYNAPSE** (5ht-const/NE/vecproj/vec_h1proj): mirrors neuron — 5ht-const/NE er-own ≈ ER;
  vecproj/vec_h1proj er-own show the same small SGD bump, Adam ≈/below ER. buf-own eye-catchers (vecproj
  0.773/0.684, vec_h1proj 0.747/0.519) are the META-REPLAY buffer lifting a naive backbone, 1-seed + buf-own
  variance, still ≪ Adam-ER 0.895. Nothing beats the best baseline.
- **I. STANDARD TUNED** (`results/pt7_std_tuned.py`; val-selected epochs≤6, lr adam1e-3/sgd1e-2): tuned
  vanilla 0.9515/0.9802 (tuning fixes the untuned SGD 0.88→0.95). Gate NEUTRAL even tuned — nearly every cell
  within ±0.006 of vanilla across driver×gran×std. RECURRING INSTABILITY: NE/all4 std-OFF, NEURON, SGD
  collapse to 0.098 ("standardize or the SGD gate blows up", now in standard); std-ON/Adam/synapse avoid it.
- **J. STATEFUL / z-score entropy** (`results/pt7_stateful.py`): nerisez=relu((H−ema_H)/√(var_H+eps)), H
  predicted by MLP or GRU, ema_H/var_H from ACTUAL past entropies; ach-GRU=standardized predicted entropy via
  GRU. FROZEN ≈ RUNNING (±0.003) — inference-time hidden/stat updates are inert, freeze at end of training.
  nerisez is SGD-unstable (MLP er-own-sgd collapse 0.098; GRU partly stabilises to 0.643<ER-sgd); ach-GRU ≈ ER
  (0.722/0.882). Statefulness/entropy-surprise buy nothing. All Adam ≈ ER; nothing beats it.
- **K. TUNED CL regime, gain-SYNAPSE (`results/pt7_tuned_syn.py`/`.log`, ledger `pt7_tuned_syn_results.tsv`;
  user-requested).** Every prior pt5/6/7 CL number used a FIXED, inherited lr=1e-3/ep5 (never val-tuned). Tuned
  the ER reference per optimizer on the VAL sequence (make_sequence(7), val_frac=0.1, eval on val — never test),
  grid lr×ep {sgd:3e-3/1e-2/3e-2, adam:3e-4/1e-3/3e-3}×{5,10,20}; selected (lr*,ep*)=SGD 0.03/5 (val 0.8981),
  Adam 3e-4/5 (val 0.9079); transferred to the mechanisms (rule #3). Harness reproduces the frozen ledger
  bit-exact at ep5/lr1e-3 (sanity: er-adam 0.8946, all4-syn-er-own-adam 0.8919). **(1) The CL regime WAS badly
  under-tuned — but only for SGD:** tuned ER-SGD 0.723→**0.9034** (test), ER-Adam 0.895→**0.8975** (already near
  ceiling, barely moves). SGD wanted a 30× larger lr. **(2) The pt7 controlled-NEGATIVE HOLDS at the tuned point,
  cleaner than before:** gain-synapse std1 er-own {NE, vecproj, all4} all within ±0.011 of same-opt tuned ER —
  SGD NE 0.9026/vecproj 0.9032/all4 0.9011 (Δ −0.001/−0.000/−0.002), Adam NE 0.9081/vecproj 0.8976/all4 0.9074
  (Δ +0.011/+0.000/+0.010, 1 seed). **3-SEED confirm of the Adam er-own cells** ({42,43,44}, tuned lr=3e-4/ep5,
  `results/pt7_tuned_syn_seeds.py`/`.log`): ER 0.9029±0.0042, **NE 0.9045±0.0043 (Δ +0.0016 = NULL, per-seed Δ
  flips sign +0.010/−0.005/−0.001 — the 1-seed +0.011 was seed 42)**, **all4 0.9104±0.0022 (Δ +0.0075, POSITIVE
  in all 3 seeds +0.010/+0.009/+0.004, paired t≈3.9 p≈0.06 — small consistent edge, borderline-significant,
  <1pt)**. NE is a null; all4 is the one cell with a consistent (if modest) positive; headline unchanged (nothing
  DECISIVELY beats tuned ER). **The earlier SGD-only "+0.14" vec_* boost is GONE** — at
  lr=1e-3 the gate was compensating for ER's underfit; tune ER directly and there's nothing left to add. Confirms
  the "capacity closing SGD's under-fit, not a class-IL lever" reading as tightly as possible. **(3) all4 std0
  (UN-standardised, gain-syn er-own): standardization is MANDATORY under SGD** — at the tuned lr=0.03 the 4 raw
  mixed-scale τ blow the K=4 unbounded gate to NaN → **0.0980 = chance** (vs std1 0.9011, Δ −0.803); Adam absorbs
  the blow-up (0.8909, Δ −0.017 vs std1). Mirror image of the tonic-driver rule (tonic collapses WITH std ÷~0;
  the mixed-scale composite collapses WITHOUT it). Reportable pt7 class-IL headline UNCHANGED — nothing beats
  tuned ER (~0.90 both opts). 1 seed; naive drops at the tuned lr (SGD 0.63→0.55, higher lr forgets harder).
- **K-batch2. 9 MORE drivers, 3-seed, tuned Adam** (`results/pt7_tuned_syn_seeds2.py`/`.log`; gain-syn er-own std1,
  vs ER 0.9029±0.0043): free **0.9086±0.0021 (+0.0057)**, 5ht-const 0.9052±0.0007 (+0.0023), NE_emb 0.9034±0.0044,
  emb_all 0.9033±0.0009, ACh 0.9014±0.0016, NE_rise 0.9010±0.0039, DA_fast 0.8993±0.0021, vec_h1proj
  0.8958±0.0033, **ACh_ema 0.8753±0.0113 (−0.0276, the lone clear loser)**. Eight of nine are a WASH vs ER; the
  two top cells are the CONTENT-FREE controls (free = K=4 gate no bio target; 5ht-const = constant gate) — a gate
  with no neuromodulator signal matches-or-beats the real bio drivers, the cleanest restatement of the pt7
  negative. ACh_ema (tonic entropy-EMA) degrades even under Adam+std (standardizing a near-constant). NE_emb
  runs cleanly at SYNAPSE (out-only P2 gate, first time).
- **K-unify. UNIFY-12 — a new "all4-like" composite of 12 heterogeneous drivers** (`results/pt7_unify.py`/`.log`;
  user-requested). One rank-12 gate Γ=1+Σ m_k P_k over: 8 head-regressed Signals drivers (DA, ACh, NE, NE_emb,
  5HT, DA_step, DA_fast, ACh_vol_ps) + headless emb_all + 3 stateful predictors (nerisez-MLP, nerisez-GRU,
  ach-GRU); all 12 columns assembled per step, DETACHED, unified-standardised; predictors trained separately with
  replay; oracle-free at eval (stateful FROZEN). Adam tuned lr=3e-4/ep5 er-own std1, 1 seed: **neuron 0.9005
  (−0.0024 vs ER, −0.0099 vs all4), synapse 0.8953 (−0.0076 vs ER, −0.0151 vs all4)**. **The kitchen sink does
  NOT help — it slightly HURTS: more drivers ≠ better.** Adding 8 redundant/entropy-flavoured/unstable columns
  (emb_all≈NE_emb; ACh≈ach-GRU≈nerisez) onto all4 dilutes the gate after standardization and lands below both ER
  and the lean 4-driver all4 — driver COUNT/content was never the lever, replay is. Synapse<neuron again (the 12×
  larger P gives the noise more room). 1 seed.

### pt7 SIGNAL-NET + GRU + neuromod-RESET (user-requested; `results/pt7_signalnet.py`/`.md`, ledger `pt7_signalnet_results.tsv`)
Four user-requested mechanisms; class-IL, gain (h0,h1,out), er-own, ADAM, seed42 lr1e-3 ep5 buffer1000
(Adam-ER operating-point-insensitive: untuned 0.8946 ≈ tuned 0.8975). Baselines er-adam 0.8946, all4 er-own
adam 0.8816. **Headline: all four land in the pt7 controlled-negative — INERT → = ER, ENGAGED → < ER, never
> ER.**
- **Task 1. NEUROMODULATOR-NET RESET (all4, 3 seeds, standardised).** Reset the neuromod net (heads m_k(x) +
  gate P) to its start-of-training weights + fresh optimizer state at each task switch (t>0); main net NOT reset,
  buffer persists (task-2 training still sees task-1 replay under the freshly-reset gate); inference uses the last
  (end-of-task-5) net. Seeds 42/43/44 = 0.8906/0.8923/0.9048, **mean 0.8959±0.0078 ≈ ER 0.8946**. Mechanism:
  resetting P each task stops it accumulating magnitude → |g| collapses to ~0.0006 (60× smaller than the no-reset
  all4's 0.045) → gate ≈ parity → run = ER. It even nudges seed42 UP (0.8816→0.8906): the tiny nonzero gate the
  no-reset all4 builds was itself very mildly hurting, and killing it returns to clean ER. probe falls 0.46→0.29–
  0.34 (a per-task-reset gate is less task-decodable). Sanity: reset-OFF reproduces all4 er-own adam **0.8816
  bit-exact** (splitting P into its own same-lr Adam ≡ keeping it in main_opt; Adam is per-parameter).
- **Tasks 2–4 (zero-init). THE DOUBLE-ZERO-INIT SADDLE.** `gru-all4` (stateful GRU on the predicted all4 vector
  → gate), `signalnet` (MLP with 3 hidden layers ingesting a 23-dim difficulty/uncertainty/novelty feature vector
  → low-D code K∈{4,16} → upproject via gate P), `signalnet-gru` (signal net → GRU → gate): ALL pin |g| to
  EXACTLY 0.000 → **= ER** across {neuron,synapse}×K{4,16}×std{on,off} (0.8856–0.8985). Cause: stacking a zero-init
  module (module-output zero-init for γ=1 parity) BEFORE the zero-init gate P makes dL/dP ∝ m = 0 AND
  dL/d(module) ∝ P = 0 → neither bootstraps → the gate is frozen at parity the whole run. This is EXACTLY pt7's
  `free` control (end-to-end-trained gate → |g|=0, baseline bit-exact); plain all4 escapes only because its heads
  have an MSE target forcing m≠0. **So a zero-init learned gate on a zero-init projection is DEAD by construction —
  it doesn't test the mechanism, it trivially reproduces the baseline.** Std/K/granularity are irrelevant while
  inert.
- **ENGAGE re-run (`--part engage`).** Break the saddle: module OUTPUT layer normal-init while gate P stays
  zero-init (preserves γ=1 at step 0, but P bootstraps because m≠0). The gate now moves (|g| up to ~2.9) and
  **every cell is WORSE than ER**: gru-all4 0.8789 (small gate, −0.016), signalnet-gru K4 0.8657 / K16 0.8799,
  signalnet K16 0.8063, signalnet K4 **0.5215** (catastrophic collapse). More gate magnitude → more harm; the
  GRU's temporal smoothing partly stabilises a large gate (signalnet-gru > signalnet at matched K) but still < ER.
  When the signal net actually drives the gate it over-modulates / injects noise.
- **Task 5. H1-GATE (`--part h1gate`).** A sibling net with the main net's architecture up to h1
  (784→400→400, same input x, output σ→[0,1]) gates the main net's h1 by element-wise multiply; trained jointly
  with the main net by the ER loss (no separate target). **pred 0.8956 ≈ ER (+0.001), but the gate is GENUINELY
  ACTIVE (mean h1-gate 0.281, far from parity 1.0 — strongly suppresses h1)** — unlike the dead-saddle cells. It
  still = ER because the jointly-trained backbone ABSORBS the uniform [0,1] rescale (Adam compensates, cf. the
  bounded01 gotcha), the pt6-followup-(E) scale-degeneracy: a jointly-trained multiplicative gate is reabsorbed by
  the weights it multiplies — gate ≠ memory.
- **Task 6. H1-GATE in the STANDARD regime (`--part h1gate-std`).** Same sibling-net h1 gate, full-MNIST
  single-task 10-way CE (goal #2). seed42 adam ep5 untuned: **vanilla 0.9766 vs h1-gate 0.9759 (Δ −0.0007 =
  neutral)**; the gate is active (mean h1-gate 0.639) but standard accuracy is preserved — harmless, like every
  neuromod mechanism in pt4/pt7-standard.
- **Task 7. all4 with a FIXED RANDOM projection (`--part all4fixed`, 3 ways × 2 seeds).** all4 er-own adam, but
  the rank-K P is FIXED RANDOM (frozen, no gradient) instead of learned; heads still regress standardized bio τ
  (with replay), only m + backbone adapt. neuron gaussian scale{0.1,0.3} + synapse gaussian scale0.1, seeds
  {42,43}: means **0.8915 / 0.8923 / 0.8905 ≈ ER 0.8946** (−0.002..−0.004), across scale and granularity; gate
  genuinely active (|g| scales with the projection scale). **Fixed-random (0.891) ≈/marginally ABOVE the learned
  all4 (0.8816)** — injecting the bio signals along random frozen directions does as well as learning the
  directions, both = ER, so the projection STRUCTURE carries nothing on class-IL (the backbone absorbs whatever
  direction the gate points in). Tightest isolation yet that the projection was never the lever.
- **Verdict.** The rich 23-signal feature set, low-D bottleneck, K=16 capacity, GRU statefulness, per-task
  resetting, a direct active h1 gate, AND a fixed-random (vs learned) projection each add NOTHING over replay on
  class-IL — same as the drivers, tonic variants, and UNIFY-12 (difficulty/novelty is not task identity; the
  projection is not the lever; replay is the only lever); the h1 gate is harmless in standard (goal #2). 1 seed
  except reset (3 seeds) + all4fixed (2 seeds); oracle-free by construction; project class-IL headline unchanged.
