"""pt5 projection primitives: driver -> projection -> per-target gate.

Extracted verbatim from prototype/neuromod.py (the fixed/learned projection builders only; the
Modulator class hierarchy and the pt5 training-loop machinery wrapped around them deliberately stay
frozen in prototype/).

P has shape (n_tasks, D). A one-hot driver z = e_t selects row t, so raw = z @ P = P[t] is a gate
over the target's D elements.

EVAL-TIME TASK ID (keep this label attached — see REQUIRES_TASK_ID_AT_EVAL): the fixed projections
`disjoint` and `shared` index P by task, so they need a task id AT EVAL to pick the gate. That is
legitimate in task-IL (it is the XdG convention) but is an ORACLE in class-IL — every pt5 class-IL
number obtained under a fixed projection is a task-IL-STYLE result on the class-IL metric, never a
class-IL solution. The `learned` projection carries the same caveat whenever its row is selected by
a true task id rather than inferred from the input.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Projections that index P by a task id, and therefore require that id to be available at eval.
REQUIRES_TASK_ID_AT_EVAL = frozenset({"disjoint", "shared"})

PROJECTIONS = ("disjoint", "shared", "learned")

# Per-neuron BIAS projections live in a separate seed namespace so a layer's bias partition is
# independent of (never aliased to) its per-synapse weight partition, and of adjacent layers' seeds.
BIAS_PROJ_SEED_OFFSET = 10_000


def build_disjoint_proj(n_tasks: int, D: int, seed: int = 0) -> torch.Tensor:
    """Disjoint per-task partition: each of the D columns has a single 1 in exactly one task row.

    Elements are evenly partitioned across tasks (counts differ by at most 1) then shuffled by
    `seed`. For task t, raw = P[t] marks that task's private slice; the per-task gates are disjoint
    and together cover every element (column sums are all 1).
    """
    g = torch.Generator().manual_seed(seed)
    assign = (torch.arange(D) % n_tasks)[torch.randperm(D, generator=g)]  # (D,) balanced task id/element
    P = torch.zeros(n_tasks, D)
    P[assign, torch.arange(D)] = 1.0
    return P


def build_shared_proj(n_tasks: int, D: int, shared_frac: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Shared backbone + private capacity: ~`shared_frac` of columns are all-ones (shared by every
    task), the rest are disjointly assigned to one task each (as in build_disjoint_proj)."""
    g = torch.Generator().manual_seed(seed)
    P = torch.zeros(n_tasks, D)
    n_shared = int(round(D * shared_frac))
    perm = torch.randperm(D, generator=g)
    shared_cols, private_cols = perm[:n_shared], perm[n_shared:]
    P[:, shared_cols] = 1.0
    if len(private_cols) > 0:
        assign = (torch.arange(len(private_cols)) % n_tasks)[torch.randperm(len(private_cols), generator=g)]
        P[assign, private_cols] = 1.0
    return P


def build_fixed_proj(projection: str, n_tasks: int, D: int, shared_frac: float, seed: int) -> torch.Tensor:
    """Dispatch the FIXED binary projection builder (disjoint/shared). `learned` is not a fixed
    projection, so it raises here; use build_proj for the general (fixed-or-learned) case."""
    if projection == "disjoint":
        return build_disjoint_proj(n_tasks, D, seed)
    if projection == "shared":
        return build_shared_proj(n_tasks, D, shared_frac, seed)
    if projection == "learned":
        raise NotImplementedError("learned is not a FIXED projection; use build_proj / register_proj")
    raise ValueError(f"unknown projection {projection!r}; known: {' | '.join(PROJECTIONS)}")


def build_proj(
    projection: str, n_tasks: int, D: int, shared_frac: float, seed: int
) -> tuple[torch.Tensor, bool]:
    """General pt5 projection builder: return (P, learned) for any granularity/bias variant.

    Fixed projections (disjoint/shared) return a binary {0,1} tensor to be registered as a buffer
    (learned=False, nothing to train). The learned projection (pt5 Iteration 3) returns a ZERO-init
    real tensor to be registered as an nn.Parameter (learned=True). Zero-init means raw = e_t @ P = 0
    for every task at the start, so the gate begins at its neutral point per target (unbounded gain
    -> 1.0 = parity; bounded01 gain / plasticity / mask -> sigmoid(0) = 0.5); and because the one-hot
    e_t selects row t, only P[t] ever receives a gradient during task t, so the per-task rows
    specialise independently. The tensor is agnostic to WHICH loss trains it: any loss that places a
    gradient on P (a modulator-only meta-loss, or the ordinary main loss) trains it, since every
    target's gate is a differentiable function of P.

    CAUTION (the double-zero-init saddle, see neurocore.controls.assert_dead_gate): a zero-init P
    stacked BEHIND another zero-init module gives neither one a gradient (dL/dP is proportional to
    the module output = 0, and dL/dmodule is proportional to P = 0), so the gate is pinned at parity
    for the whole run and trivially reproduces the baseline.
    """
    if projection == "learned":
        return torch.zeros(n_tasks, D), True
    return build_fixed_proj(projection, n_tasks, D, shared_frac, seed), False


def register_proj(
    module: nn.Module, name: str, projection: str, n_tasks: int, D: int, shared_frac: float, seed: int
) -> None:
    """Register projection `name` on `module`: a buffer for fixed projections, an nn.Parameter for
    the learned one (so it lands in module.parameters() and any loss can train it). See build_proj.
    Used by every pt5 driver modulator (per-neuron and per-synapse, weights and biases) so the
    fixed-vs-learned split lives in exactly one place."""
    P, learned = build_proj(projection, n_tasks, D, shared_frac, seed)
    if learned:
        module.register_parameter(name, nn.Parameter(P))
    else:
        module.register_buffer(name, P)


def requires_task_id_at_eval(projection: str) -> bool:
    """True if this projection needs a task id at EVAL to select its gate row. In class-IL that is
    an oracle and must be labelled as such; in task-IL it is the standard XdG convention."""
    return projection in REQUIRES_TASK_ID_AT_EVAL


def _gate_logit_bias(init_gate: float) -> float:
    """Logit offset b such that sigmoid(b) == init_gate (0.5 -> 0.0). Sets the initial LEARNED
    plasticity gate (pt5 iter3 init-bias sweep); clamped away from {0,1} to keep b finite."""
    p = min(max(float(init_gate), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))
