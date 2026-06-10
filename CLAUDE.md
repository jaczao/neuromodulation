# Neuromodulation for Continual Learning — MLP Prototype

## Project
Thesis prototype testing a neuromodulation mechanism in an MLP, on two fronts:
1. Continual learning: does it reduce catastrophic forgetting on Split MNIST (alone, and stacked with the best baseline)?
2. Standard learning: does it improve, preserve, or degrade plain MNIST accuracy vs. a vanilla MLP?

The 1.5-day sprint is complete. Current work follows `@SPEC-proto-pt2.md` — four sequential iterations of neuromodulation variants on the prototype. Read that SPEC at the start of each iteration session. After all four iterations, the project scaffolds to the definitive repo (which is when `THESIS-PLAN.md` will be created) and moves to GRU/CNN/ViT.

## Stack
- Python 3.12 via uv
- PyTorch 2.x (CUDA if available, else CPU/MPS)
- W&B for logging, pytest for the one test that matters

## Layout
```
prototype/
├── data.py             # Split MNIST loader + disjointness test; standard MNIST + val split
├── model.py            # vanilla MLP [784→400→400→10]; ModulatedLinear wrapper for Iteration 2+
├── neuromod.py         # Modulator base + variants + drivers + registry (the contribution)
├── methods.py          # Naive, Joint, EWC, ER
├── train.py            # standard + CL loops + metrics
├── configs.py          # dataclass configs — ALL hyperparameters live here
└── iteration-notes.md  # appended per iteration: result, debugging outcome, decision
tests/
└── test_data.py        # task disjointness, no train/test leakage
```

## Specs
- `@SPEC-proto-pt1.md` — completed sprint SPEC; historical reference only.
- `@SPEC-proto-pt2.md` — governs all current work. Iterations 1–4: plasticity, weight mask, driver comparison, stateful modulator.
- `THESIS-PLAN.md` — does NOT exist yet. Created only as part of the post-iteration migration (will hold the multi-architecture roadmap and the definitive repo structure for scaffolding).

## Neuromodulation design
- `neuromod.py` holds a `Modulator` base class and a registry of variants, targets, and drivers.
- Three orthogonal axes, all config-selected, never hardcoded:
  - `--neuromod-variant {feedforward, stateful}` — modulator architecture
  - `--neuromod-target {activation, plasticity, weight_mask}` — where modulation acts
  - `--neuromod-driver {none, surprise, uncertainty, activation_stats}` — what feeds the modulator
- `--use-neuromod` toggles the whole mechanism; OFF must reproduce the vanilla baseline numerically.
- Must compose with CL methods, e.g. `--method er --use-neuromod --neuromod-target plasticity --neuromod-driver surprise`.
- Iterations explore this grid **one axis at a time** per `@SPEC-proto-pt2.md`. Never combine multiple new mechanisms in one experiment.

## Run
- `uv run pytest tests/` — must pass before any training
- CL: `uv run python prototype/train.py --method {naive,joint,ewc,er} --seed N [--use-neuromod --neuromod-variant V --neuromod-target T --neuromod-driver D] [--no-wandb]`
- Standard: `uv run python prototype/train.py --standard --seed N [--use-neuromod ...] [--no-wandb]`
- `--no-wandb` logs to stdout only — use it for sanity runs so they don't need W&B auth (or set `WANDB_MODE=offline`).
- W&B sweep configs live in `sweeps/` (create when needed).

## Non-negotiable rules
1. Never tune on any test set — not the CL test sequence, not the official MNIST test set. CL tuning uses the validation sequence (seed=7); standard tuning uses a held-out split of the MNIST training set.
2. Standard and CL get SEPARATE hyperparameter sweeps (different optima — epochs-per-task is meaningless for standard; CL wants smaller LR).
3. Identical tuning budget per method/variant.
4. Neuromodulation is config-selected and a one-flag swap (`--use-neuromod`) — never tangle neuromod logic inside `model.py`. The `ModulatedLinear` wrapper is the one exception, and it must behave exactly like `nn.Linear` when no mask is supplied. Variant, target, and driver are args, not hardcoded. Must compose with CL methods.
5. Report mean ± std over 3 seeds for any final number.
6. After any edit to `data.py`, re-run `pytest tests/test_data.py`.
7. No hardcoded hyperparameters in training code — everything routes through `configs.py`.
8. **Iteration discipline:** one new mechanism per iteration. Never combine. See `@SPEC-proto-pt2.md`.

## Metrics
### Standard learning (full MNIST)
- Test accuracy (mean ± std over 3 seeds), tuned via the standard sweep
- Two rows: vanilla MLP, neuromod MLP — neuromod must not materially hurt standard accuracy

### Continual learning (Split MNIST)
- Average final accuracy across all tasks (primary)
- Forgetting: mean over tasks of (max acc during training − final acc)
- Includes a combined "neuromod + best baseline" row to test complementarity
- Per-task accuracy trajectory (for plots, if time allows)

## CL setup
- Split MNIST: 5 tasks × 2 classes — (0,1), (2,3), (4,5), (6,7), (8,9)
- Optimizer: Adam (LR tuned separately for standard vs CL)
- Epochs per task: tuned — dominant forgetting knob
- Default batch size: 64

## Known gotchas
- Normalize MNIST with whole-dataset stats, not per-task — per-task normalization silently leaks task identity.
- EWC Fisher must be computed *after* finishing a task, before starting the next.
- ER buffer: update *before* the gradient step, sample with replacement, reservoir sampling for buffer fill.
- Standard val split comes from the TRAIN set, never the test set.
- `--use-neuromod` OFF must match vanilla exactly — if it doesn't, the modulator is leaking into the base path.
- **Plasticity target + Adam:** multiplying `param.grad` by `α` before Adam computes its moments is NOT the same as scaling per-parameter LRs — Adam's running first/second moments are computed *from the scaled gradient*. Test with plain SGD first to isolate the mechanism, or scale the post-Adam update instead of the gradient.
- **Plasticity target trains the modulator ONLY via lookahead:** the literal SPEC hook (multiply `param.grad` by α between `backward()` and `step()`, forward untouched) cannot train the modulator. With the forward untouched, the same-step loss is independent of α; the next-step loss depends on α through `W_new = W − lr·(α⊙g)`, but `optimizer.step()` writes W in place under `no_grad`, severing the α→W_new autograd edge → `α.grad = None`. Train it with a lookahead/first-order meta-gradient (`W_fast = W.detach() − lr·(α⊙g)`, `L_meta` on `functional_call(model, W_fast)`, backward), then commit `W ← W_fast.detach()`. (Iteration 1, rejected: even with α free over [0,1] per-neuron, no retention signal in a current-task meta-loss → no forgetting reduction.)
- **Weight-mask target:** the mask must be **per-synapse** (one scalar per weight). A per-neuron mask (one scalar per row of W) is mathematically equivalent to pre-activation gain modulation and is therefore not a distinct mechanism.
- **Weight-mask on one hidden layer does NOT fix class-IL forgetting (Iteration 2, rejected).** The mask is in the forward graph so it trains by ordinary backprop (no lookahead) and learns full-range, task-differentiated masks, yet forgetting stays total (= Naive). Masking only net.2 leaves the first layer and the shared output head (net.4) to be overwritten, and class-IL forgetting is dominated by output-logit competition (van de Ven & Tolias 2019), which a hidden-layer mask cannot touch. Modulator LR ×50 and low-rank r=16 do not change the result. `ModulatedLinear` with no mask must stay `allclose` to `nn.Linear` (parity).
- **Drivers:** every driver signal (surprise, uncertainty, activation_stats) must be **detached** before reaching the modulator. There must be no gradient path from a driver input back into the main loss.
- **Activation target output range:** sigmoid silently sparsifies activations. If activation modulation is ever revisited, prefer softplus (positive, unbounded above) or affine FiLM `(1+m) ⊙ h + β` to preserve dense activations. Plasticity and weight-mask targets are fine with sigmoid (they gate learning/multiplication, not activations directly).
- On this MacBook (MPS), `torch.cuda.manual_seed_all` is a no-op — `torch.manual_seed` covers MPS. Call both anyway for CUDA portability, but don't expect the cuda call to do anything here.
- Always seed torch, numpy, AND python's random module.

## W&B tags
`method`, `dataset` (standard_mnist|split_mnist), `seed`, `use_neuromod`, `neuromod_variant`, `neuromod_target`, `neuromod_driver` — so variant × target × driver ablations stay sortable.

## Out of scope for the iteration phase
- GRU/CNN/ViT (next phase, after iterations and scaffolding).
- Permuted MNIST (added only after iterations complete, before GRU).
- Scaffolding the definitive repo (`src/`, Hydra, Mammoth submodule, etc.) — happens only after all four iterations are complete.
- Creating `THESIS-PLAN.md` — created as part of the post-iteration migration.
- Combining multiple new mechanisms in one experiment.
- Wider hyperparameter grids beyond the sprint's budget; sensitivity plots.
- Revisiting activation modulation as its own iteration (held in reserve; the sub-variants `(1+m)⊙h+β`, residual init, pre- vs. post-activation, sign-bound `1+m≥0` are noted but parked).

## Update policy
Whenever Claude Code makes a mistake, add a one-liner under "Known gotchas" so it doesn't repeat.