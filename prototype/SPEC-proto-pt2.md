# SPEC: Prototype Iteration — making neuromodulation work on Split MNIST

## Context
The 1.5-day sprint produced baselines (Naive, Joint, EWC, ER) and one neuromodulation variant on MLP + Split MNIST. The neuromod variant did **not** beat Naive (or not convincingly). We are still in the prototype, not the scaffolded repo. This SPEC governs the iteration that follows.

## Repo housekeeping — do this once, before Iteration 1
- Rename the existing `SPEC.md` (the completed sprint SPEC) to `SPEC-proto-pt1.md` so it's clearly historical and doesn't get confused with the current SPEC.
- This file is `SPEC-proto-pt2.md` in prototype/.
- Create an empty `prototype/iteration-notes.md` — iteration results and decisions get appended here.
- Do NOT create any new folders (`src/`, `docs/`, `configs/`, etc.) or scaffold anything else. Stay in the existing `prototype/` layout. Migration to the definitive repo happens only AFTER all three iterations complete (see "After all 3 iterations" at the bottom).

## Goal
Find a neuromodulation variant that *clearly beats Naive* on Split MNIST with an MLP, before investing in any scaffolding, wider grids, sensitivity plots, or additional architectures. If none of the planned iterations succeeds, the project pivots (see "Stop conditions" at the bottom).

## Methodology — non-negotiable
1. **Change one substantive thing at a time.** Never combine two new mechanisms in the same experiment. If you do, you cannot attribute the result.
2. **Keep all baseline numbers (Naive, EWC, ER) frozen** from the sprint. Do not re-tune baselines during iteration; they are the reference.
3. **Use the same Split MNIST configuration** as the sprint — same task order, same metrics, same evaluation protocol. The only thing that changes between iterations is the neuromodulation mechanism.
4. **Sanity-check the modulator before declaring failure.** Before moving to the next iteration, run the debugging checklist (below). Adding more mechanisms on top of a silently-broken modulator just compounds the problem.
5. **`--use-neuromod` OFF must still numerically reproduce vanilla.** Every iteration must preserve this.
6. **Same tuning budget** for each iteration as for the original sprint variant. No expanding sweeps to rescue a failing method.

**Note on "≥X points" throughout this SPEC.** All accuracy gaps are stated in *absolute percentage points*, not relative percent. E.g. "≥5 points over Naive" means Naive 25% → method ≥30%, not 25% × 1.05.

## Iteration order
Run these sequentially. Move to the next only if the current one has been fairly tested AND the debugging checklist has been completed.

### Iteration 1 — Plasticity modulation (acting on weights)
**What to implement.** A new `--neuromod-target plasticity` option that gates per-parameter learning rates instead of scaling activations. The modulator outputs a per-parameter (or per-neuron, broadcast) mask `α ∈ [0,1]`. After `loss.backward()` and before `optimizer.step()`, multiply each parameter's gradient by its `α`. Start at **per-neuron** granularity (one `α` per hidden unit, broadcast to that unit's incoming and outgoing weights) — do not start per-synapse, do not start global.

**Note on the mask range.** `α ∈ [0,1]` (sigmoid) is the right choice here: it gates *learning rates*, not activations. Activations remain dense — every unit still fires; only gradient updates are gated. This is **not** an activation-sparsity mechanism. (If you ever reuse the activation-modulation target, prefer softplus or affine FiLM there instead of sigmoid, so activations don't get sparsified.)

**Files to touch.**
- `prototype/neuromod.py`: add `PlasticityModulator(Modulator)` subclass, register it under `"plasticity"`. It exposes a hook like `modulate_gradients(named_params, context) -> None` that mutates `param.grad` in place.
- `prototype/train.py`: in the training step, between `loss.backward()` and `optimizer.step()`, call the modulator's gradient hook when `--neuromod-target == plasticity`.
- `prototype/configs.py`: add `neuromod_target: str = "activation"` to the config dataclass; accept the new value `"plasticity"`.
- `prototype/iteration-notes.md`: append the Iteration 1 entry on completion.

**What stays the same.** Same modulator architecture, same context input, same hyperparameter budget. Only the target axis changes.

**Adam caveat.** If the base optimizer is Adam, multiplying `param.grad` by `α` interacts oddly with Adam's running first/second-moment estimates (the moments are computed *from the scaled gradient*, so a small `α` shrinks the moment estimates rather than the parameter update). Two options: (a) test with plain SGD for the first run to isolate the mechanism, or (b) scale the *update* (after `optimizer.step()` writes it) rather than the gradient. State which choice you made in `iteration-notes.md`.

**Accept when.** Beats Naive avg final accuracy by **≥5 absolute percentage points** on Split MNIST with 3 seeds.

**Reject and move on when.** After a fair sweep with the same budget as the sprint, it does not beat Naive *and* the debugging checklist is clean.

### Iteration 2 — Mask on weights (per-synapse)
**Run after Iteration 1 completes (accepted or fairly rejected).** Independent of Iteration 1's outcome.

**What to implement.** A new `--neuromod-target weight_mask` option where the modulator produces a per-synapse mask `M ∈ [0,1]^{d_out × d_in}` (sigmoid) for a chosen weight matrix in the main net. The forward pass uses `y = (M ⊙ W) x + b` instead of `y = W x + b`. **The mask must be per-synapse, not per-neuron.** A per-neuron mask (one scalar per row of W) is mathematically equivalent to pre-activation gain modulation and would not be a distinct mechanism.

**Why this is its own iteration.** With `y = (M ⊙ W) x`, the chain rule gives `∂L/∂W = M ⊙ (∂L/∂y ⊗ x)`, so a single mask gates both the forward pass (activation-like) AND the gradient at W (plasticity-like). This couples activation and plasticity modulation through one mask — distinct from Iteration 1, which gates only the backward pass and leaves the forward untouched.

**Target one weight matrix initially.** Modulating every weight matrix in the MLP would balloon the modulator's output dimension. Start by modulating only the **second linear layer** (hidden→hidden, `400×400 = 160k` mask values). This is the largest single matrix that's still tractable on a MacBook CPU.

**Files to touch.**
- `prototype/neuromod.py`: add a `WeightMaskModulator(Modulator)` subclass that outputs a (d_out, d_in) sigmoid mask. Register it under `"weight_mask"`.
- `prototype/model.py`: replace the targeted linear layer with a small `ModulatedLinear` wrapper that accepts an externally-supplied mask `M` per forward call and computes `y = (M ⊙ W) x + b`. With `--use-neuromod` off or no mask supplied, it must behave exactly like `nn.Linear` (parity check).
- `prototype/train.py`: each forward pass: query the modulator for `M`, pass it to `ModulatedLinear`. No gradient-hook plumbing — gradients flow naturally through the chain rule.
- `prototype/configs.py`: accept `neuromod_target = "weight_mask"`; add a config field for which layer's weights to modulate (default: layer 2).
- `prototype/iteration-notes.md`: append entry.

**Memory note.** Per-synapse mask for the 400×400 layer means the modulator outputs 160k values. If a small modulator becomes dominated by its output head, fall back to a low-rank factorization `M = sigmoid(A B^T)` with rank ~16 — but try the full-rank version first.

**Accept when.** Beats Naive avg final accuracy by **≥5 absolute percentage points** on Split MNIST with 3 seeds. The comparison with Iteration 1 is itself informative — does coupled forward+backward gating outperform backward-only gating?

### Iteration 3 — Driver comparison
**Run after Iterations 1 and 2 complete, regardless of their individual outcomes.**

This iteration tests *three different driver signals* feeding the modulator, applied on top of the best-performing target identified so far (sprint target, Iteration 1 plasticity, or Iteration 2 weight-mask — whichever showed the most life). The point is to compare drivers under matched conditions: the target, modulator architecture, and tuning budget are all fixed; only the modulator's input changes.

Run the three sub-iterations sequentially. **Every driver signal must be detached** — drivers are control signals, not backprop paths. Verify no gradient flow from any driver input back into the main loss.

**Sub-iteration 3a — Surprise.** `surprise = (loss - ema_loss).detach()`, with EMA update `ema_loss = β*ema_loss + (1-β)*loss.detach()` (β=0.99 default). High surprise → modulator can raise plasticity / loosen gates / etc.

**Sub-iteration 3b — Softmax uncertainty.** Predictive entropy of the model's output distribution: `H(p) = -Σ p_i log p_i`, detached and fed to the modulator. (A cheaper proxy is `1 - max_i p_i`.) High entropy → low confidence → modulator has a signal that the current input is hard.

**Sub-iteration 3c — Hidden activation statistics.** Per-layer statistics of the main net's hidden activations, computed and detached: at minimum the L2 norm and mean activation per layer; ideally also variance and sparsity (fraction near zero). Concatenate into the modulator's input. Richest of the three drivers, largest input dimension.

**Files to touch.**
- `prototype/neuromod.py`: extend the modulator's `forward` to accept a `driver` argument; route into the input concatenation. Add small helper functions to compute each driver signal.
- `prototype/train.py`: at the right point in the forward pass, compute the active driver (selected by config), detach, pass to the modulator. For activation stats, use forward hooks on the hidden layers.
- `prototype/configs.py`: add `neuromod_driver: str` with values `"none"` (baseline), `"surprise"`, `"uncertainty"`, `"activation_stats"`. Combinations like `"surprise+uncertainty"` are out of scope for this iteration.
- `prototype/iteration-notes.md`: append separate entries for 3a, 3b, 3c with their individual results.

**Accept when.** At least one of the three drivers beats the best result from Iterations 1 and 2 by **≥5 absolute percentage points** (or matches it with materially less forgetting). **Record all three results regardless** — the comparison between drivers is itself a contribution, even if none wins outright.

### Iteration 4 — Stateful modulator
**Run after Iterations 1, 2, and 3 complete, regardless of their individual outcomes.**

**What to implement.** Replace the modulator's feedforward architecture with a small recurrent one (a single GRU cell is sufficient; hidden size 32–64). It maintains state across training steps within a task *and* across task boundaries (do NOT reset the hidden state between tasks — that would secretly leak task identity). Inputs as in Iteration 3 (use the winning driver from 3a/3b/3c; if none won, default to surprise).

**Files to touch.**
- `prototype/neuromod.py`: add a `StatefulModulator` subclass containing a GRU cell; its `forward` updates the internal hidden state each call and returns the modulation. Register it under `"stateful"`.
- `prototype/configs.py`: add a config flag selecting the modulator architecture (`"feedforward"` vs `"stateful"`).
- `prototype/train.py`: ensure the trainer initializes the modulator's hidden state once at start of training (or with seed-dependent init), persists it across tasks, and never resets it on a task boundary.
- `prototype/iteration-notes.md`: append entry.

**Why it might help.** A stateful modulator can track "what has been learned so far" and "how much things are shifting" — a stateless one cannot. Likely more impactful in the later GRU phase, but cheap enough to test on MLP-CL as the fourth iteration.

**Accept when.** Beats the best result so far by **≥5 absolute percentage points**, OR matches it with materially less forgetting — either is a positive result worth keeping.

## Debugging checklist — run BEFORE giving up on an iteration
If an iteration doesn't beat Naive, check these in order before concluding the mechanism is wrong:

1. **Modulator output distribution.** Log the histogram of the modulator's outputs over a training run. If they're stuck near 0, near 1, or constant, the modulator is degenerate. Check the activation/normalization on the modulator's output head. For plasticity, near-1 means "everything is plastic" (no protection); near-0 means "nothing learns."
2. **`--use-neuromod` OFF parity.** Confirm OFF reproduces vanilla MLP exactly — if not, the modulator is leaking into the base path even when "off," which silently corrupts comparisons.
3. **Gradient flow into the modulator.** Print modulator parameter gradient norms. If they're zero, the modulator is not being trained (common bug: `detach` in the wrong place, or modulator outputs not actually used).
4. **Learning-rate ratio.** The modulator typically needs a different LR than the main net — try a 1/10× and 10× ratio. A common failure is the modulator either being overwhelmed by the main net's updates or dominating them.
5. **Where modulation is applied.** Try moving the target from one hidden layer to another (e.g. only the last hidden layer, or only the first). The "right" layer is mechanism-specific.
6. **Plasticity granularity (for the plasticity target).** Try the three granularities in order: per-layer (one `α` per layer), per-neuron (one `α` per unit, broadcast), per-synapse (one `α` per weight). Per-neuron is the default; per-layer is a more conservative fallback if per-neuron is noisy; per-synapse is the most flexible but the hardest to train.
7. **Which parameters are modulated (for plasticity target).** Try modulating only weights vs. weights+biases vs. weights of specific layers only. Modulating biases often adds noise without value; output-layer modulation interacts with classifier capacity.
8. **Adam vs. SGD interaction (for plasticity target).** As noted in Iteration 1, multiplying gradients before Adam computes its moments is *not* the same as scaling the per-parameter LR. Test once with plain SGD to isolate whether the mechanism works at all, then decide whether to keep SGD or move to scaling the post-Adam update.
9. **Modulator output range.** Sigmoid is right for plasticity (gate ∈ [0,1]) and for activation *gating* — but for activation *gain modulation* with dense activations preserved, prefer softplus (positive, unbounded above) or affine FiLM. A sigmoid on an activation modulator silently induces sparsity.
10. **Modulator capacity.** If the modulator is too small (e.g. <32 hidden units) it may underfit; too large and it overfits the current task's statistics. Try one step bigger and one smaller.
11. **Init.** If the modulator initially outputs near-zero gates or near-zero gradients, learning may never start. Initialize the modulator's output head so that initial behaviour ≈ vanilla (e.g. gate ≈ 1 at init for multiplicative gain; α ≈ 1 for plasticity, i.e. full plasticity at start).

Only after this checklist returns clean is it fair to say the *mechanism* (not the implementation) doesn't work, and move to the next iteration.

## Per-iteration deliverables
For each iteration, commit:
- Code changes implementing the new variant/target/driver (single commit, easily revertable).
- A short markdown note in `prototype/iteration-notes.md`: what was tried, what the result was, debugging-checklist outcome, decision (kept / discarded / moved on).
- A row in a running results table: `iteration | variant | target | driver | avg_final_acc ± std | forgetting ± std | beats Naive?`.

These notes are what will populate Chapter 3 of the interim report regardless of outcome.

## Stop conditions
- **Success at any iteration.** An iteration clearly beats Naive on Split MNIST by ≥5 absolute percentage points, consistent across 3 seeds. Commit that iteration's variant as the working baseline mechanism, but **do NOT halt** — continue with the remaining iterations. They are no longer rescue attempts; they are additional comparisons that contribute to the thesis's empirical contribution (the comparison *between* mechanisms is the contribution). Scaffolding and migration happen only after all four iterations are complete.
- **Failure across all four.** All four iterations complete, debugging checklists clean for each, none beats Naive. Do NOT add more iterations or mechanisms ad hoc — a clean negative result on this design space is itself a valid finding. Pause and discuss framing with the supervisor before continuing. Negative results are reportable; chasing a working configuration ad hoc is not.

## What NOT to do during iteration
- Do not scaffold the full repo (no `src/`, `configs/`, `docs/`, Hydra, Mammoth submodule, etc.). Scaffolding happens only after all four iterations are complete — see "After all 4 iterations" below.
- Do not add Permuted MNIST, CIFAR, GRU, or any additional architecture/benchmark.
- Do not produce sensitivity plots or wider hyperparameter grids.
- Do not combine multiple new mechanisms into one iteration.
- Do not re-tune the baselines.
- Do not create `THESIS-PLAN.md` yet; it will be created as part of the post-iteration migration (see below).

## After all 4 iterations — migration outline (do NOT execute until then)
Documented here so Claude Code knows the destination; not in scope for the iteration phase. Once all four iterations are done, regardless of outcome:

1. **Create `THESIS-PLAN.md`** — the living overview doc with: research questions, multi-architecture roadmap (MLP → GRU → CNN → ViT), variant × target × driver ablation grid, current status, references, **and the planned definitive repo structure**. This is where the repo layout for the scaffold gets specified, so Claude Code has a single source of truth to scaffold from.
2. **Scaffold from the structure stated in `THESIS-PLAN.md`** — typically `src/neuromod_cl/{models,methods,benchmarks,neuromod,trainers,eval,utils}/`, `configs/` (Hydra), `tests/`, `docs/`, `third_party/mammoth/` (submodule), and `scripts/`.
3. **Migrate prototype code file-by-file** into the new layout; do not rewrite from scratch.
4. **Regression test.** Rerun the MLP experiments under the new structure and confirm numbers match the prototype's within noise — this is the migration's correctness check.
5. **Archive the prototype.** Git-tag the final prototype commit (e.g. `v0.1-prototype`); move `prototype/` to `archive/prototype-mlp/` or rely on the tag and delete it.
6. **Archive the SPECs.** Move `SPEC-mlp-sprint.md` and `SPEC-prototype-iteration.md` to `docs/archive/specs/`.
7. **Update `CLAUDE.md`** to reflect the new layout, point at `@docs/THESIS-PLAN.md`, and add any gotchas accumulated during iteration.
8. **Add Permuted MNIST** as the second MLP benchmark, then move to GRU.

None of steps 1–8 are in scope for the current iteration phase.

## Execution rules for Claude Code
- Read this SPEC at the start of each iteration session.
- Implement one iteration end-to-end before starting the next.
- Print a one-line status at iteration completion: `Iteration N: <accept|reject>, avg_final_acc=X.X ± Y.Y, beats_naive=<yes|no>`.
- Commit after each iteration with a clear message: `iter<N>: <variant/target/driver summary>`.
- If a debugging-checklist item triggers (e.g. modulator outputs collapsed), fix it *within the current iteration* — do not move on with a known-broken implementation.