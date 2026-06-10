# Iteration notes (SPEC-proto-pt2)

Results and decisions for each neuromodulation iteration are appended below.

## Running results table

| iteration | variant | target | driver | avg_final_acc ± std | forgetting ± std | beats Naive? |
|-----------|---------|--------|--------|---------------------|------------------|--------------|
| (baseline) Naive | feedforward | (none) | none | 0.1979 ± 0.0003 | 0.7979 ± 0.0004 | (reference) |
| 1 | feedforward | plasticity (per-neuron) | none | 0.1992 ± 0.0000 | 0.7986 ± 0.0001 | no |
| 2 | feedforward | weight_mask (per-synapse, layer 2) | none | 0.1979 ± 0.0000 | 0.7982 ± 0.0002 | no |

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


