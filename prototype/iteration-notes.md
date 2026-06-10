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



