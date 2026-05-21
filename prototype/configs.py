from dataclasses import dataclass, field


@dataclass
class StandardConfig:
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    seed: int = 42
    val_size: int = 10_000
    use_neuromod: bool = False
    neuromod_variant: str = "gain"
    neuromod_target: str = "hidden"
    neuromod_learned_projection: bool = False


@dataclass
class CLConfig:
    lr: float = 1e-3
    epochs_per_task: int = 5
    batch_size: int = 64
    seed: int = 42
    ewc_lambda: float = 1e5
    ewc_samples: int = 200
    er_buffer_size: int = 200
    use_neuromod: bool = False
    neuromod_variant: str = "gain"
    neuromod_target: str = "hidden"
    neuromod_learned_projection: bool = False


# ---------------------------------------------------------------------------
# Phase 6 best configs — selected on validation data only (never the test set)
# ---------------------------------------------------------------------------
# (a) Standard-MNIST sweep: lr ∈ {3e-4, 1e-3} × epochs ∈ {10, 20}, 1 seed
#     Selected by val_acc on held-out 10k of MNIST training set.
BEST_STANDARD_VANILLA = StandardConfig(lr=3e-4, epochs=20)
BEST_STANDARD_NEUROMOD = StandardConfig(lr=3e-4, epochs=20, use_neuromod=True)

# (b) CL sweep: lr ∈ {3e-4, 1e-3} × epochs_per_task ∈ {5, 10},
#     plus method-specific knobs (EWC λ, ER buffer, neuromod learned_projection).
#     Validation sequence: make_sequence(7).  Selected by avg_final_acc.
BEST_CL_NAIVE = CLConfig(lr=1e-3, epochs_per_task=5)
BEST_CL_JOINT = CLConfig(lr=1e-3, epochs_per_task=10)
BEST_CL_EWC = CLConfig(lr=3e-4, epochs_per_task=5, ewc_lambda=1e4)
BEST_CL_ER = CLConfig(lr=3e-4, epochs_per_task=5, er_buffer_size=1000)
BEST_CL_NEUROMOD = CLConfig(lr=1e-3, epochs_per_task=10, use_neuromod=True)
