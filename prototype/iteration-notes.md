# Iteration notes (SPEC-proto-pt2)

Results and decisions for each neuromodulation iteration are appended below.

## Running results table

| iteration | variant | target | driver | avg_final_acc ± std | forgetting ± std | beats Naive? |
|-----------|---------|--------|--------|---------------------|------------------|--------------|
| (baseline) Naive | feedforward | (none) | none | 0.1979 ± 0.0003 | 0.7979 ± 0.0004 | (reference) |
| 1 | feedforward | plasticity (per-neuron) | none | 0.1992 ± 0.0000 | 0.7986 ± 0.0001 | no |

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

