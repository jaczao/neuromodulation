# Neuromodulation for Continual Learning — MLP Prototype

## Project
Thesis prototype testing a neuromodulation mechanism in an MLP, on two fronts:
1. Continual learning: does it reduce catastrophic forgetting on Split MNIST (alone, and stacked with the best baseline)?
2. Standard learning: does it improve, preserve, or degrade plain MNIST accuracy vs. a vanilla MLP?
1.5-day sprint. If results are promising, we scaffold a full repo for GRU/CNN/ViT.

## Stack
- Python 3.12 via uv
- PyTorch 2.x (CUDA if available, else CPU/MPS)
- W&B for logging, pytest for the one test that matters

## Layout
prototype/
├── data.py      # Split MNIST loader + disjointness test; standard MNIST + val split
├── model.py     # vanilla MLP [784→400→400→10] — NO neuromod logic here
├── neuromod.py  # Modulator interface + variants + registry (the contribution)
├── methods.py   # Naive, Joint, EWC, ER
├── train.py     # standard + CL loops + metrics
└── configs.py   # dataclass configs — ALL hyperparameters live here
tests/
└── test_data.py # task disjointness, no train/test leakage

## Neuromodulation design
- `neuromod.py` holds a `Modulator` base class + a registry of variants (gain / gating / lr-modulation / …).
- Two orthogonal choices, both config-selected, never hardcoded:
  - `--neuromod-variant <name>`  — which mechanism
  - `--neuromod-target <name>`   — where it acts (hidden activations / specific layer / optimizer step)
- `--use-neuromod` toggles the whole thing; OFF must reproduce the vanilla baseline numerically.
- Must compose with CL methods, e.g. `--method er --use-neuromod --neuromod-variant gain --neuromod-target hidden`.
- SPRINT SCOPE: ship exactly ONE variant × ONE target. The interface is for later expansion, not upfront breadth.
- FIRST (and only) net this sprint: --neuromod-variant gain --neuromod-target hidden — multiplicative gain h ← (1+mod)⊙h on both hidden layers; signal from a small MLP (784→64→k=8, sigmoid) broadcast through a fixed random P (buffer, not Parameter; randn/√k). Modulator's final layer zero-init so gain starts at 1.0 and --use-neuromod off matches vanilla exactly.

## Run
- `uv run pytest tests/` — must pass before any training
- CL:       `uv run python prototype/train.py --method {naive,joint,ewc,er} --seed N [--use-neuromod --neuromod-variant V --neuromod-target T] [--no-wandb]`
- Standard: `uv run python prototype/train.py --standard --seed N [--use-neuromod --neuromod-variant V --neuromod-target T] [--no-wandb]`
- `--no-wandb` logs to stdout only — use it for sanity runs so they don't need W&B auth (or set `WANDB_MODE=offline`).
- W&B sweep configs in `sweeps/`

## Non-negotiable rules
1. Never tune on any test set — not the CL test sequence, not the official MNIST test set. CL tuning uses the validation sequence (seed=7); standard tuning uses a held-out split of the MNIST training set.
2. Standard and CL get SEPARATE hyperparameter sweeps (different optima — epochs-per-task is meaningless for standard; CL wants smaller LR).
3. Identical tuning budget per method/model, including the neuromod variant.
4. Neuromodulation is config-selected and a one-flag swap (`--use-neuromod`) — never tangle neuromod logic inside `model.py`. Variant and target are args, not hardcoded. It must compose with CL methods.
5. Report mean ± std over 3 seeds for any final number.
6. After any edit to `data.py`, re-run `pytest tests/test_data.py`.
7. No hardcoded hyperparameters in training code — everything routes through `configs.py`.

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
- Normalize MNIST with whole-dataset stats, not per-task — per-task normalization silently leaks task identity
- EWC Fisher must be computed *after* finishing a task, before starting the next
- ER buffer: update *before* the gradient step, sample with replacement, reservoir sampling for buffer fill
- Standard val split comes from the TRAIN set, never the test set
- `--use-neuromod` OFF must match vanilla exactly — if it doesn't, the modulator is leaking into the base path
- On this MacBook (MPS), `torch.cuda.manual_seed_all` is a no-op — `torch.manual_seed` covers MPS. Call both anyway for CUDA portability, but don't expect the cuda call to do anything here.
- Always seed torch, numpy, AND python's random module
- EWC does not beat Naive on class-IL Split MNIST (shared head, no task-id) — this is a known result (van de Ven & Tolias 2019): output logit competition between tasks makes weight regularization ineffective; EWC works for task-IL but not class-IL. The Fisher computation must use per-sample gradients (not batch-mean), otherwise Fisher is underestimated by ~B×.
- `ewc_lambda` default changed to 1e5 (not 1000) because per-sample Fisher diagonal mean ≈ 1e-5 on this MLP; λ=1000 gives a penalty ≈0.01 which is negligible relative to the task loss.

## W&B tags
`method`, `dataset` (standard_mnist|split_mnist), `seed`, `use_neuromod`, `neuromod_variant`, `neuromod_target` — so variant×target ablations stay sortable.

## Out of scope for this prototype
- GRU/CNN/ViT (next phase)
- Permuted MNIST (add only if time at end of Day 2)
- More than one neuromod variant/target (interface only; one of each for now)
- Hydra (dataclass configs are enough here; Hydra when we scaffold the real repo)
- Mammoth (custom code for the prototype; integrate later)

## Update policy
Whenever Claude Code makes a mistake, add a one-liner under "Known gotchas" so it doesn't repeat.