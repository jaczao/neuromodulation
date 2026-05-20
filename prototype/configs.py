from dataclasses import dataclass


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
