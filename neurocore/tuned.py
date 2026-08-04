"""Tuned operating points — the live registry (prototype/configs.py holds the frozen Split-MNIST copy).

Same semantics as before, with PROBLEM added to the front of both keys so a second dataset cannot
silently inherit Split MNIST's tuned point. Existing entries migrate as problem="splitmnist" and are
unchanged in value.

NON-NEGOTIABLES CARRIED OVER:
  - Everything here was selected on a VALIDATION split — the CL validation sequence make_sequence(7)
    with val_frac=0.1, or a held-out split of the training set — and never on any test set.
  - A MISSING TUNED_MAIN KEY RAISES. That is the design: an un-tuned combination must be tuned, never
    guessed. Only Adam was swept for the neuromod studies; SGD is known only for class-IL er.
  - The optimizer is a config-DETERMINING axis (separate sweeps per optimizer), so it lives in the
    key rather than being folded away.
  - Arms sit at genuinely different operating points: replay tolerates and needs large steps, an
    unreplayed net forgets harder the larger the step. The naive arm's true Adam optimum was FIVE
    decades below the ER grid's floor. Never reuse one arm's grid for another, and never compare arm
    A to arm B directly — only mechanism-vs-its-own-baseline.
  - A point selected at a GRID BOUNDARY is a truncated grid, not an optimum: extend it. Stop
    extending once the top cells fall inside the noise floor, and tie-break by smallest
    epochs_per_task (cheapest, and CL average accuracy DECAYS with more epochs per task).
"""
from .ledger import NOISE_FLOOR

# (a) MAIN-net operating point: (problem, metric, base, optimizer) -> {lr, epochs_per_task}.
TUNED_MAIN = {
    # ---- Split MNIST (pt7_tuned_syn tuned the ER baseline PER optimizer) ----
    ("splitmnist", "classil", "er",    "adam"): dict(lr=3e-4, epochs_per_task=5),   # val 0.9079
    ("splitmnist", "classil", "er",    "sgd"):  dict(lr=3e-2, epochs_per_task=5),   # val 0.8981
    ("splitmnist", "classil", "naive", "adam"): dict(lr=3e-4, epochs_per_task=5),   # ER point transferred
    # ("splitmnist","classil","naive","sgd"): UNKNOWN — naive != er under SGD (masked-loss confound)
    # task-IL — pt7_tuned_neuro_taskil (ADAM only; SGD not swept):
    ("splitmnist", "taskil",  "naive", "adam"): dict(lr=3e-4, epochs_per_task=5),   # val 0.9560
    ("splitmnist", "taskil",  "er",    "adam"): dict(lr=3e-4, epochs_per_task=10),  # val 0.9948
    # task-IL SGD — pt5_taskil/plast_taskil.py swept these from scratch (they had no entry here);
    # persisted after the fact, so a later study does not re-sweep what is already val-selected.
    ("splitmnist", "taskil",  "naive", "sgd"):  dict(lr=3e-3, epochs_per_task=5),   # val 0.9761
    ("splitmnist", "taskil",  "er",    "sgd"):  dict(lr=1e-1, epochs_per_task=5),   # val 0.9909
    # ER selected the grid TOP on the first pass; the grid was extended to {3e-1, 1.0} and 3e-1
    # plateaued while 1.0 diverged to chance, so 1e-1 is a genuine interior max, not truncation.
    # ---- pt3 retry, naive+masked arm (its own grid; the ER grid was wrong by four decades) ----
    ("splitmnist", "classil", "naive_masked", "sgd"):  dict(lr=1e-3, epochs_per_task=5),
    ("splitmnist", "classil", "naive_masked", "adam"): dict(lr=1e-5, epochs_per_task=5),
    # ---- new problems: nothing tuned yet. Tune before use; do NOT copy a Split-MNIST row. ----
}

# (b) NEUROMOD-net learning rate: (problem, metric, base, optimizer, mechanism, granularity) -> lr.
#     Main net FROZEN at TUNED_MAIN[...]; the neuromod optimizer (gate P + heads, Adam) is decoupled
#     from the main lr. Studies pt7_tuned_neuro{,_taskil,_sgd}.py.
#     FINDING: at the selected neuro_lr the tuned all4 gate TIES or trails the dead-gate `free`
#     control in every cell. Persisted for provenance, not as a claimed win. `free`'s neuro_lr is
#     INERT (the zero-init gate never engages, |g|=0 at every lr).
TUNED_NEURO_LR = {
    ("splitmnist", "classil", "er", "adam", "all4", "neuron"):  1e-3,   # val 0.9080
    ("splitmnist", "classil", "er", "adam", "all4", "synapse"): 1e-3,   # val 0.9117
    ("splitmnist", "classil", "er", "adam", "free", "neuron"):  1e-4,   # dead gate; nlr inert
    ("splitmnist", "classil", "er", "adam", "free", "synapse"): 1e-4,   # dead gate; nlr inert
    ("splitmnist", "classil", "er", "sgd",  "all4", "synapse"): 3e-3,   # val 0.9000 (flat)
    ("splitmnist", "classil", "er", "sgd",  "free", "synapse"): 3e-3,   # dead gate; nlr inert
    ("splitmnist", "taskil", "naive", "adam", "all4", "synapse"): 3e-4,  # val 0.9640 (1-seed peak)
    ("splitmnist", "taskil", "naive", "adam", "free", "synapse"): 1e-4,  # dead gate; nlr inert
    ("splitmnist", "taskil", "er",    "adam", "all4", "synapse"): 1e-3,  # val 0.9943 (~ceiling)
    ("splitmnist", "taskil", "er",    "adam", "free", "synapse"): 1e-4,  # dead gate; nlr inert
}

# Fallback for an un-swept neuro_lr: reuse the one tuned value AT THAT OPTIMIZER'S SCALE (Adam and
# SGD want very different neuro_lr, so a single default is wrong). These are REUSED defaults, NOT
# independent tunes — tune per combination for a final number.
DEFAULT_NEURO_LR = 1e-3                                   # Adam scale (inherited pt7 default)
DEFAULT_NEURO_LR_BY_OPT = {"adam": 1e-3, "sgd": 3e-3}     # SGD reuses the one SGD tune


def tuned_main(problem, metric, base, optimizer):
    """Main-net operating point. RAISES on an un-tuned combination — that is intentional: tune it."""
    key = (problem, metric, base, optimizer)
    if key not in TUNED_MAIN:
        raise KeyError(
            f"no tuned main-net point for {key}. Tune it on the validation split first (never the "
            f"test set) and add it here; do not reuse another problem's or another arm's point.")
    return dict(TUNED_MAIN[key])


def tuned_neuro_lr(problem, metric, base, optimizer, mechanism, granularity):
    """Neuromod-net lr, falling back to the optimizer-scale default when the combo is un-swept."""
    return TUNED_NEURO_LR.get((problem, metric, base, optimizer, mechanism, granularity),
                              DEFAULT_NEURO_LR_BY_OPT.get(optimizer, DEFAULT_NEURO_LR))


def select_tuned(cells, acc_key="acc", epochs_key="epochs_per_task", noise_floor=NOISE_FLOOR):
    """Pick an operating point from validation cells, with the tie-break the sweeps converged on.

    `cells` is an iterable of dicts holding at least `acc_key` and `epochs_key`. Among cells within
    `noise_floor` of the best validation accuracy, take the smallest epochs_per_task (cheapest, and
    fewer epochs per task means less forgetting), breaking remaining ties on accuracy. Raw 2D argmax
    is NOT used: (lr, epochs) trade off along a ridge, so the argmax slides on noise.

    Also flags a boundary selection, which means the grid was truncated rather than optimised.
    """
    cells = list(cells)
    if not cells:
        raise ValueError("select_tuned: no validation cells")
    best = max(float(c[acc_key]) for c in cells)
    tied = [c for c in cells if best - float(c[acc_key]) <= noise_floor]
    pick = min(tied, key=lambda c: (float(c[epochs_key]), -float(c[acc_key])))
    return pick, {"n_tied": len(tied), "best_val": best}


def at_grid_boundary(pick, grid, field):
    """True if the selected value sits at an edge of the swept grid for `field` — a truncated grid,
    not an optimum. Extend the grid in that direction and re-select."""
    vals = sorted(set(grid))
    return len(vals) > 1 and pick[field] in (vals[0], vals[-1])
