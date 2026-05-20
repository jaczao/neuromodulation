# SPEC: MLP Prototype for Neuromodulated Continual Learning

## Goal
A controlled comparison of a neuromodulated MLP vs. {Naive, Joint, EWC, ER} on **both** standard MNIST and Split MNIST, completed in 1.5 days. Two questions:
1. **Continual learning:** does neuromodulation reduce catastrophic forgetting on Split MNIST (vs. baselines, and combined with the best baseline)?
2. **Standard learning:** does neuromodulation improve, preserve, or degrade plain MNIST accuracy vs. a vanilla MLP?

Final deliverables: a standard-training table (vanilla vs. neuromod), a CL table (5+ methods × {avg final acc, forgetting}, mean ± std over 3 seeds), and a yes/no decision on whether the mechanism helps.

## Scope and non-goals
- **In scope:** MLP only; **standard MNIST** (full dataset) and **Split MNIST** (CL); four baselines (Naive, Joint, EWC, ER); one neuromodulation variant, evaluated standalone and combined with the best baseline.
- **Out of scope (for this prototype):** GRU / CNN / ViT, Permuted MNIST, Hydra, Mammoth integration, sensitivity plots, additional baselines.

## Architecture
- MLP: `784 → 400 → 400 → 10`, ReLU activations, no batchnorm, no dropout.
- Single shared classification head (class-IL, no task-id at inference).
- **Neuromodulation as a swappable component.** Define a small `Modulator` interface in `neuromod.py` with concrete variants (e.g. gain modulation, gating, learning-rate modulation), selected via `--neuromod-variant`. A separate `--neuromod-target` flag specifies which part of the main net it acts on (e.g. hidden activations, a specific layer, or the optimizer's per-parameter step). `--use-neuromod` toggles the whole mechanism on/off. This keeps every combination a config change, not a code fork, so the ablation grid (variant × target) runs from the same `train.py` and sweep machinery.
- It **must** be implementable as a sidecar/wrapper to the base MLP — do not tangle neuromod logic inside `model.py` — and must **compose with a CL method** (e.g. `--method er --use-neuromod --neuromod-variant gain --neuromod-target hidden`), since Phase 7 tests the combination.
- **For the 1.5-day sprint, ship exactly one variant × one target.** The interface exists so later expansion needs no rewrites, not so you build many up front.

## Benchmarks
- **Standard MNIST:** the full dataset (all 10 classes), standard train/test split. Used for the non-CL research question — vanilla MLP vs. neuromod MLP. For tuning, carve a **held-out validation split from the MNIST *training* set** (e.g. last 10k of 60k train images, fixed); the official MNIST test set is touched only at final evaluation.
- **Split MNIST:** 5 tasks, classes `(0,1), (2,3), (4,5), (6,7), (8,9)`. Each task uses standard MNIST train/test, filtered to its 2 classes.
- **Test sequence (CL):** class order as above, fixed with seed=42.
- **Validation sequence (CL):** shuffled class order with seed=7. Used only for CL hyperparameter tuning. Never touched at final-evaluation time.

## Phase 1 — Project setup
**Create:**
- `pyproject.toml` via `uv init`
- `prototype/{data,model,neuromod,methods,train,configs}.py`
- `tests/test_data.py`
- `CLAUDE.md` (already drafted by author)

**Add deps:** `torch`, `numpy`, `pytest`, `wandb`.

**Accept when:** `uv run pytest tests/` exits 0 (no tests yet is fine); `uv run python -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())"` returns True on the target machine.

## Phase 2 — Data and model
**`data.py`:**
- `class SplitMNIST` with `get_task_loaders(task_id, batch_size) -> (train_loader, test_loader)`.
- Normalization: whole-MNIST mean/std `(0.1307, 0.3081)`, applied uniformly across tasks. **Never** per-task normalization.
- Helper `task_indices(task_id, split)` returning dataset indices.
- Factory `make_sequence(seed) -> list[tuple[int, int]]` returning the per-task class pairs.

**`tests/test_data.py`:**
- Each task has exactly 2 classes.
- Pairwise task class-set intersection is empty.
- Union of task classes equals `{0,…,9}`.
- No train/test index leakage within a task.

**`model.py`:**
- `class MLP(nn.Module)` with the architecture above; `forward(x)` flattens `(B, 1, 28, 28)` → `(B, 784)`.

**Standard training in `train.py` (pipeline sanity check):**
- Train vanilla MLP on full MNIST with default hyperparameters (10 epochs, batch_size 64, Adam lr=1e-3), single seed, to confirm the pipeline works. Run with `--no-wandb` so this check needs no W&B auth.
- The *final* standard-MNIST numbers are produced in Phase 7 using the best hyperparameters from the Phase 6 standard sweep — not here.

**Accept when:** disjointness tests pass and this sanity run reaches **≈98% (anything ≥97.5% passes)**. This is a pipeline sanity check, not a tuning target — do NOT increase epochs or otherwise grind to chase the last fraction of a percent. If you're within ~0.5 points of 98%, the pipeline is correct; move on.

## Phase 3 — CL loop and bounds
**Extend `train.py`:**
- `cl_train(method, model, sequence, epochs_per_task, ...)`.
- After each task, evaluate on **all seen tasks**; log full accuracy matrix `A[t, i]` = acc on task `i` after training on task `t`.
- Compute:
  - `avg_final_acc = mean(A[T-1, :])`
  - `forgetting = mean over i<T of (max_{t≤T-1} A[t, i] - A[T-1, i])`

**`methods.py`:**
- `Naive`: standard sequential fine-tuning, no CL mechanism.
- `Joint`: train once on the union of all task data (oracle upper bound).

**Accept when:** Naive shows avg final acc ≈ 20–40% (severe forgetting); Joint shows ≥ 97%. If Naive does not catastrophically forget, the splits are leaking — stop and debug.

## Phase 4 — EWC and ER baselines
**`methods.py`:**

**EWC**
- Compute Fisher diagonal `F_i = E[(∂ log p(y|x; θ))²]` at end of task `i`, using ~200 samples drawn from that task's training data.
- Store snapshot `θ_i*`.
- Loss: `task_loss + (λ/2) · Σ_i F_i · (θ - θ_i*)²`.
- Accumulate across tasks (sum of per-task penalties).

**ER (Experience Replay)**
- Reservoir buffer of fixed size `M`.
- Each training step: sample mini-batch from buffer (size `B`, with replacement), concatenate with current-task batch, single gradient step on combined loss.
- Update buffer with current batch **before** the gradient step (reservoir sampling).

**Accept when:** with defaults (EWC λ=1000, ER buffer=200, 5 epochs/task), both beat Naive by ≥10 points avg final accuracy. If either doesn't, debug before moving to Phase 5.

## Phase 5 — Neuromodulation variant
**`neuromod.py`:** define a `Modulator` base interface and at least one concrete variant. Structure for extensibility:
- `Modulator` base class with a clean hook API (e.g. `modulate(activations, context)` and/or an optimizer-step hook).
- A registry mapping `--neuromod-variant` names → classes; a `--neuromod-target` arg selecting where it applies.
- For this sprint, implement **one** variant and **one** target (author's primary hypothesis). Leave the registry in place so adding more later is a few lines.

**Wire into `train.py`:** `--use-neuromod` adds the selected modulator on top of the base MLP and/or modifies the training procedure. Clean toggle: with the flag off, results must reproduce the corresponding vanilla baseline numerically.

**Accept when:** `--use-neuromod` is a one-flag swap; variant and target are config-selected (not hardcoded); with-neuromod vs. without numbers actually differ.

**Quick standard-MNIST check:** run neuromod-MLP once on standard MNIST (default hyperparameters, single seed) to confirm it trains and that `--use-neuromod` off reproduces vanilla. Final tuned standard numbers come in Phase 7.

## Phase 6 — Hyperparameter sweeps
Two **separate** sweeps, because standard and CL settings have different optima (epochs-per-task is meaningless in standard training; CL prefers smaller LR to limit drift).

**(a) Standard-MNIST sweep — on the held-out MNIST-train validation split:**
- Grid: `lr ∈ {3e-4, 1e-3}`, `epochs ∈ {10, 20}`.
- Run for **both** vanilla MLP and neuromod MLP, identical grid and budget.
- Select best config per model by validation accuracy. Record in `configs.py`.
- ~8 short runs total; a few minutes on CPU.

**(b) CL sweep — on the validation sequence (seed=7):**
- Grid (small by design — 1.5-day budget):
  - `lr ∈ {3e-4, 1e-3}`
  - `epochs_per_task ∈ {5, 10}`
  - EWC: `λ ∈ {100, 1000}`
  - ER: `buffer ∈ {200, 1000}`
  - Neuromod: 2–3 author-specified values per knob
- Same total trial budget per method (track this — fair comparison depends on it).
- 1 seed per trial during sweep is fine.
- Record the single best configuration per method in `configs.py`.

**Accept when:** best configs exist in `configs.py` for (a) vanilla and neuromod on standard MNIST, and (b) each CL method; and no config was selected by looking at any test set / test sequence.

## Phase 7 — Final evaluation
**Standard MNIST table (on the official MNIST test set, 3 seeds):**
- Run vanilla MLP and neuromod MLP, each with its Phase-6(a) best hyperparameters.
- Rows: `vanilla MLP`, `neuromod MLP`. Column: `test acc (mean ± std)` over 3 seeds.
- Save as `results/standard_mnist_table.{csv,md}`.
- Interpretation: neuromod should be within ~1 point of vanilla; a large drop is an important negative finding to report.

**CL on test sequence (seed=42), 3 seeds (42, 43, 44):**
- Run each of {Naive, Joint, EWC, ER, neuromod-MLP} with its Phase-6 best hyperparameters.
- **Also run a combined method:** neuromod + the single best-performing baseline from Phases 4/6 (e.g. neuromod+ER or neuromod+EWC). This tests whether neuromodulation is *complementary* (stacks with replay/regularization) or *redundant* (no gain on top) — a key result either way. On MLP this is a cheap extra ~3 runs.
- Produce a table with rows = methods (including the combined one), columns = `avg_final_acc (mean ± std)`, `forgetting (mean ± std)`.
- Save as `results/split_mnist_table.{csv,md}`.

**Accept when:** both tables exist. The neuromod row sits in a defensible position on Split MNIST (clear win, clear loss, or comparable to EWC/ER — any is a valid outcome), and on standard MNIST neuromod is within ~1 point of vanilla.

## Methodological rules (apply throughout)
1. **Never** tune on any test set: not the CL test sequence, not the official MNIST test set. CL tuning uses the validation sequence; standard tuning uses the held-out MNIST-train validation split.
2. **Equal** tuning budgets across methods, including neuromod.
3. **Mean ± std over 3 seeds** for any final number.
4. **Whole-dataset normalization stats**, never per-task.
5. **Reservoir sampling**, not FIFO, for ER buffer.
6. **Fisher computed after training** on a task, not during.
7. **All hyperparameters in `configs.py`** as `@dataclass`; CLI args override via argparse.
8. **Single seed source:** one `--seed` arg seeds `torch`, `numpy`, and `random`. Also call `torch.cuda.manual_seed_all(seed)` for portability — it's a harmless no-op on MPS/CPU but matters on CUDA. `torch.manual_seed` already covers MPS, so do not rely on the cuda call doing anything on this MacBook.
9. **Commit to git** after each accepted phase.
10. **Re-run `pytest tests/test_data.py`** after any edit to `data.py`.

## W&B conventions
- A `--no-wandb` flag must skip `wandb.init()` entirely and log metrics to stdout only, so sanity runs (Phase 2) and CI work without W&B auth. Equivalently, respect `WANDB_MODE=offline`.
- Project name: `neuromod-cl-prototype`.
- Per-run tags: `method=<name>`, `dataset=<standard_mnist|split_mnist>`, `seed=<n>`, `use_neuromod=<bool>`, `neuromod_variant=<name|none>`, `neuromod_target=<name|none>`.
- Log per task: `acc/task_{i}` after every task, plus final `avg_final_acc` and `forgetting`.

## Execution order for Claude Code
Run phases strictly in order. Do not start Phase N+1 until Phase N's accept-criteria are met. After each phase, print a one-line status (`Phase N accepted: <summary>`) and commit.

**On numeric accept-criteria:** all accuracy/forgetting thresholds in this SPEC are *approximate sanity bars*, not optimization targets. If a result is within ~0.5 points of a stated threshold, treat the criterion as met and move on. Never enter a tune-and-rerun loop to chase a fraction of a percent — that wastes the sprint budget and risks overfitting. If genuinely far off (e.g. MNIST sanity run stuck below 95%, or a CL baseline not beating Naive at all), that signals a real bug worth investigating; a 0.2-point miss does not.