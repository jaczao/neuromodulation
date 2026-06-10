from dataclasses import dataclass, field


@dataclass
class StandardConfig:
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    seed: int = 42
    val_size: int = 10_000
    use_neuromod: bool = False
    neuromod_variant: str = "feedforward"   # feedforward | stateful (Iter 4)
    neuromod_target: str = "activation"     # activation | plasticity | weight_mask
    neuromod_driver: str = "none"           # none | surprise | uncertainty | activation_stats (Iter 3)
    neuromod_learned_projection: bool = False
    neuromod_lr: float = 1e-3               # modulator optimizer LR (own LR, see Iter1 checklist item 4)
    neuromod_alpha_init: float = 0.95       # plasticity: initial gate α (≈ full plasticity)
    neuromod_mask_layer: int = 2            # weight_mask: which net.<idx> linear to mask (default 2nd: 400×400)
    neuromod_mask_rank: int = 0             # weight_mask: 0 = full-rank head; r>0 = low-rank sigmoid(A·diag(g)·Bᵀ)
    neuromod_mask_init: float = 0.99        # weight_mask: initial mask value M (≈1 → near-vanilla at init)


@dataclass
class CLConfig:
    lr: float = 1e-3
    epochs_per_task: int = 5
    batch_size: int = 64
    seed: int = 42
    ewc_lambda: float = 1e5
    ewc_samples: int = 200
    er_buffer_size: int = 200
    optimizer: str = "adam"                 # adam (baselines) | sgd (matched plasticity control)
    use_neuromod: bool = False
    neuromod_variant: str = "feedforward"   # feedforward | stateful (Iter 4)
    neuromod_target: str = "activation"     # activation | plasticity | weight_mask
    neuromod_driver: str = "none"           # none | surprise | uncertainty | activation_stats (Iter 3)
    neuromod_learned_projection: bool = False
    neuromod_lr: float = 1e-3               # modulator optimizer LR (own LR, see Iter1 checklist item 4)
    neuromod_alpha_init: float = 0.95       # plasticity: initial gate α (≈ full plasticity)
    neuromod_mask_layer: int = 2            # weight_mask: which net.<idx> linear to mask (default 2nd: 400×400)
    neuromod_mask_rank: int = 0             # weight_mask: 0 = full-rank head; r>0 = low-rank sigmoid(A·diag(g)·Bᵀ)
    neuromod_mask_init: float = 0.99        # weight_mask: initial mask value M (≈1 → near-vanilla at init)


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
