from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Modulator(ABC, nn.Module):
    """Base interface for neuromodulation variants."""

    @abstractmethod
    def modulate(
        self, h: torch.Tensor, context: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Apply modulation to post-ReLU hidden activations.

        Args:
            h: (B, hidden_dim) post-ReLU activations to modulate
            context: (B, 784) flattened input image (modulation signal source)
            layer_idx: 0 or 1 (first or second hidden layer)

        Returns:
            Modulated activations, same shape as h
        """
        ...


class GainModulator(Modulator):
    """FiLM-style multiplicative gain: h_l ← (1 + mod_l) ⊙ h_l.

    Signal net: Linear(784→64) → ReLU → Linear(64→k) [zero-init → output=0 at init].
    Broadcast: mod_l = signal @ P_l, P_l ∈ ℝ^(k×hidden_dim), fixed random (randn/√k).
    Zero-init of the final linear ensures mod=0 at init → gain=1.0 → identical to vanilla
    at the start of training.
    """

    def __init__(
        self,
        k: int = 8,
        hidden_dim: int = 400,
        learned_projection: bool = False,
    ) -> None:
        super().__init__()
        self.signal_net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, k),
        )
        # Zero-init final linear so gain starts at exactly 1.0
        nn.init.zeros_(self.signal_net[2].weight)
        nn.init.zeros_(self.signal_net[2].bias)

        # Per-layer random projection P_l: (k, hidden_dim), one per hidden layer
        for l_idx in range(2):
            P = torch.randn(k, hidden_dim) / (k ** 0.5)
            if learned_projection:
                self.register_parameter(f"P_{l_idx}", nn.Parameter(P))
            else:
                self.register_buffer(f"P_{l_idx}", P)

    def modulate(
        self, h: torch.Tensor, context: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        sig = self.signal_net(context)       # (B, k)
        P = getattr(self, f"P_{layer_idx}")  # (k, hidden_dim)
        mod = sig @ P                        # (B, hidden_dim)
        return (1.0 + mod) * h


# Registry: variant name → class.  None = planned but not implemented this sprint.
_REGISTRY: dict[str, type[Modulator] | None] = {
    "gain": GainModulator,
    "gating": None,
    "lr_modulation": None,
}

# Valid targets (dispatch to the correct hook lives in ModulatedMLP / train.py)
_TARGETS: frozenset[str] = frozenset({"hidden"})


def make_modulator(
    variant: str,
    *,
    target: str = "hidden",
    learned_projection: bool = False,
) -> Modulator:
    """Instantiate a modulator by variant name."""
    if variant not in _REGISTRY:
        raise ValueError(
            f"Unknown neuromod variant {variant!r}. Known: {sorted(_REGISTRY)}"
        )
    if target not in _TARGETS:
        raise ValueError(
            f"Unknown neuromod target {target!r}. Known: {sorted(_TARGETS)}"
        )
    cls = _REGISTRY[variant]
    if cls is None:
        raise NotImplementedError(
            f"Neuromod variant {variant!r} is registered but not implemented this sprint."
        )
    if variant == "gain":
        return cls(learned_projection=learned_projection)
    return cls()


class ModulatedMLP(nn.Module):
    """Sidecar wrapper: runs base MLP layer-by-layer, applying Modulator post-ReLU.

    Assumes base_mlp.net is an nn.Sequential with layout:
        [0] Linear(784, 400)
        [1] ReLU
        [2] Linear(400, 400)
        [3] ReLU
        [4] Linear(400, 10)
    """

    def __init__(self, base_mlp: nn.Module, modulator: Modulator) -> None:
        super().__init__()
        self.base = base_mlp
        self.modulator = modulator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)                   # (B, 784)
        h1 = self.base.net[1](self.base.net[0](x_flat))  # Linear → ReLU → (B, 400)
        h1 = self.modulator.modulate(h1, x_flat, layer_idx=0)
        h2 = self.base.net[3](self.base.net[2](h1))      # Linear → ReLU → (B, 400)
        h2 = self.modulator.modulate(h2, x_flat, layer_idx=1)
        return self.base.net[4](h2)                       # (B, 10)
