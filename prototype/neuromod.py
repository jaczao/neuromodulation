from abc import ABC

import torch
import torch.nn as nn


class Modulator(ABC, nn.Module):
    """Base interface for neuromodulation variants.

    Two orthogonal hooks, one per target axis. A subclass overrides only the
    hook for the target it implements; the other stays a no-op.
      - modulate            -> activation target (forward-pass gain/gating)
      - modulate_gradients  -> plasticity target (backward-pass LR gating)
    """

    def modulate(
        self, h: torch.Tensor, context: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Activation-target hook: modulate post-ReLU hidden activations.

        Args:
            h: (B, hidden_dim) post-ReLU activations to modulate
            context: (B, 784) flattened input image (modulation signal source)
            layer_idx: 0 or 1 (first or second hidden layer)

        Returns:
            Modulated activations, same shape as h. Default: identity.
        """
        return h

    def modulate_gradients(
        self, named_params, context: torch.Tensor
    ) -> None:
        """Plasticity-target hook: scale param gradients in place. Default: no-op.

        See PlasticityModulator for the trainable lookahead path; this in-place
        variant is the SPEC interface and does NOT train the modulator on its own.
        """
        return None


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


class PlasticityModulator(Modulator):
    """Plasticity target: per-neuron learning-rate gate α ∈ [0,1] (Iteration 1).

    Same modulator architecture as GainModulator (signal net 784→64→k, per-layer
    fixed random projection P_l: k→hidden_dim), but the output gates *gradients*
    rather than activations. One α per hidden unit (per-neuron granularity),
    broadcast to that unit's incoming and outgoing weights.

    Init: signal-net final layer is zero-init and a constant logit bias is added
    before the sigmoid, so α ≈ alpha_init (default 0.95, ~full plasticity) for
    every unit at the start of training, regardless of the random P_l.

    NOTE: the forward pass of the main net is untouched, so the *task* loss does
    not depend on α. This modulator is therefore trained by a lookahead /
    first-order meta-gradient in train.py (differentiate the gated one-step
    update), NOT by the in-place modulate_gradients hook. compute_alphas and
    param_factors expose the differentiable pieces that loop needs.
    """

    def __init__(
        self,
        k: int = 8,
        hidden_dim: int = 400,
        n_hidden_layers: int = 2,
        alpha_init: float = 0.95,
        learned_projection: bool = False,
    ) -> None:
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        self.signal_net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, k),
        )
        # Zero-init final linear → signal=0 at init → α = sigmoid(logit_bias) for all units.
        nn.init.zeros_(self.signal_net[2].weight)
        nn.init.zeros_(self.signal_net[2].bias)

        # Constant logit offset so the initial gate equals alpha_init (≈ full plasticity).
        alpha_init = min(max(alpha_init, 1e-4), 1 - 1e-4)
        logit = torch.logit(torch.tensor(alpha_init))
        self.register_buffer("alpha_logit_bias", logit)

        for l_idx in range(n_hidden_layers):
            P = torch.randn(k, hidden_dim) / (k ** 0.5)
            if learned_projection:
                self.register_parameter(f"P_{l_idx}", nn.Parameter(P))
            else:
                self.register_buffer(f"P_{l_idx}", P)

    def compute_alphas(self, context: torch.Tensor) -> dict[int, torch.Tensor]:
        """Per-hidden-layer gates {layer_idx: α (hidden_dim,)}, differentiable in modulator params.

        Gradients are gated per parameter (batch-aggregated), so we drive the gate
        from the batch-mean input rather than per-sample: one α vector per layer.
        """
        ctx = context.view(context.size(0), -1).mean(dim=0, keepdim=True)  # (1, 784)
        sig = self.signal_net(ctx)                                         # (1, k)
        alphas: dict[int, torch.Tensor] = {}
        for l_idx in range(self.n_hidden_layers):
            P = getattr(self, f"P_{l_idx}")          # (k, hidden_dim)
            raw = (sig @ P).squeeze(0)               # (hidden_dim,)
            alphas[l_idx] = torch.sigmoid(self.alpha_logit_bias + raw)
        return alphas

    def param_factors(self, alphas: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Per-parameter gradient multipliers for the [784,400,400,10] MLP.

        Per-neuron α broadcast to incoming and outgoing weights:
          net.0.weight[i,:] *= α0[i]              (rows = hidden-0 units)
          net.2.weight[i,j] *= α1[i] * α0[j]      (rows = hidden-1, cols = hidden-0)
          net.4.weight[:,j] *= α1[j]              (cols = hidden-1 units)
        Output bias (net.4.bias) is left unmodulated.
        """
        a0, a1 = alphas[0], alphas[1]
        return {
            "net.0.weight": a0.unsqueeze(1),                 # (400, 1)
            "net.0.bias": a0,                                # (400,)
            "net.2.weight": a1.unsqueeze(1) * a0.unsqueeze(0),  # (400, 400)
            "net.2.bias": a1,                                # (400,)
            "net.4.weight": a1.unsqueeze(0),                 # (1, 400) → (10, 400)
        }

    def modulate_gradients(self, named_params, context: torch.Tensor) -> None:
        """In-place SPEC hook: scale grads by α (no modulator training). Unused by the
        lookahead loop, kept for interface completeness / non-meta experiments."""
        with torch.no_grad():
            factors = self.param_factors(self.compute_alphas(context))
        for name, p in named_params:
            if p.grad is not None and name in factors:
                p.grad.mul_(factors[name])


# Registry: target name → modulator class.  None = planned but not yet implemented.
_REGISTRY: dict[str, type[Modulator] | None] = {
    "activation": GainModulator,
    "plasticity": PlasticityModulator,
    "weight_mask": None,   # Iteration 2
}

# Accepted modulator-architecture variants (only feedforward is wired pre-Iteration 4).
_VARIANTS: frozenset[str] = frozenset({"feedforward", "stateful", "gain"})


def make_modulator(
    target: str,
    *,
    variant: str = "feedforward",
    learned_projection: bool = False,
    alpha_init: float = 0.95,
) -> Modulator:
    """Instantiate a modulator by target. `variant` selects architecture (only
    feedforward wired now; 'gain' is a legacy alias for feedforward)."""
    # Legacy alias: the sprint used target='hidden' for activation gain modulation.
    if target == "hidden":
        target = "activation"
    if target not in _REGISTRY:
        raise ValueError(
            f"Unknown neuromod target {target!r}. Known: {sorted(_REGISTRY)}"
        )
    if variant not in _VARIANTS:
        raise ValueError(
            f"Unknown neuromod variant {variant!r}. Known: {sorted(_VARIANTS)}"
        )
    if variant == "stateful":
        raise NotImplementedError("Stateful modulator lands in Iteration 4.")
    cls = _REGISTRY[target]
    if cls is None:
        raise NotImplementedError(
            f"Neuromod target {target!r} is registered but not implemented yet."
        )
    if cls is GainModulator:
        return cls(learned_projection=learned_projection)
    if cls is PlasticityModulator:
        return cls(learned_projection=learned_projection, alpha_init=alpha_init)
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
