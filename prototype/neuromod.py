from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Drivers (Iteration 3): detached control signals fed into the modulator's input.
# A driver is computed from the PREVIOUS step's loss/logits/activations (lag-1) and
# used to drive the NEXT step's mask, so it never sits on the main-loss backprop path.
# ---------------------------------------------------------------------------
_DRIVER_DIMS: dict[str, int] = {
    "none": 0,
    "surprise": 1,          # (loss - ema_loss)
    "uncertainty": 1,       # mean predictive entropy
    "activation_stats": 8,  # per hidden layer (×2): [L2 norm, mean, var, sparsity]
}


def driver_dim(name: str) -> int:
    if name not in _DRIVER_DIMS:
        raise ValueError(f"Unknown driver {name!r}. Known: {sorted(_DRIVER_DIMS)}")
    return _DRIVER_DIMS[name]


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean softmax entropy H(p) = -Σ p log p over the batch, detached. Shape (1,)."""
    logp = F.log_softmax(logits, dim=1)
    H = -(logp.exp() * logp).sum(dim=1).mean()
    return H.detach().view(1)


def activation_stats(acts: list[torch.Tensor]) -> torch.Tensor:
    """Per-layer [mean-L2-norm, mean, variance, sparsity] for post-ReLU activations.

    acts: list of (B, d) hidden activations. Returns a detached (4·len(acts),) vector.
    """
    feats: list[torch.Tensor] = []
    for a in acts:
        a = a.detach()
        feats.append(a.norm(dim=1).mean())              # mean L2 norm across batch
        feats.append(a.mean())                          # mean activation
        feats.append(a.var(unbiased=False))             # variance
        feats.append((a <= 1e-6).float().mean())        # sparsity (fraction near zero)
    return torch.stack(feats).detach()


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

    def compute_mask(self, context: torch.Tensor) -> torch.Tensor | None:
        """Weight-mask-target hook: per-synapse mask M ∈ [0,1]^(d_out×d_in). Default: None."""
        return None

    def modulate_logits(self, logits: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Logit-target hook: calibrate the output logits per sample. Default: identity."""
        return logits


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


class WeightMaskModulator(Modulator):
    """Weight-mask target: context-driven per-synapse mask M ∈ [0,1]^(d_out×d_in) (Iteration 2).

    Applied as y = (M ⊙ W) x + b on one targeted weight matrix (default: the second
    linear layer, 400×400). A single mask gates both the forward pass AND the gradient
    at W (∂L/∂W = M ⊙ (∂L/∂y ⊗ x)), coupling activation- and plasticity-like modulation
    through one mask. The mask is in the forward graph, so the modulator trains by
    ordinary backprop (no lookahead, unlike the plasticity target).

    Same input as the other modulators (batch-mean image → signal net 784→64→k). The
    mask head is zero-init, so M ≈ mask_init (≈1, near-vanilla) for every synapse at the
    start and the per-synapse structure emerges as the head trains.

    rank=0  → full-rank head: Linear(k → d_out·d_in), the modulator outputs all d_out·d_in
              mask logits directly (SPEC: "try full-rank first").
    rank=r>0 → low-rank fallback M = sigmoid(bias + A·diag(g(s))·Bᵀ), A∈(d_out,r), B∈(d_in,r),
              context-dependent coefficients g(s)∈ℝ^r (SPEC memory-note fallback).

    learned_projection (rank=0 only): True (default) = the k→mask map is a learned Linear head
    (the established Iteration 2/3 form). False = a FIXED RANDOM projection R∈ℝ^(k×d_out·d_in)
    (mirrors gain/plasticity's random P_l): only the 784→64→k signal net is learned, R is a buffer.
    In the random case the signal net's final layer is zero-init so the k-code is 0 at init →
    M = mask_init (vanilla parity), exactly as gain/plasticity get parity from their zero-init.
    """

    def __init__(
        self,
        d_out: int = 400,
        d_in: int = 400,
        k: int = 8,
        rank: int = 0,
        mask_init: float = 0.99,
        driver_dim: int = 0,
        learned_projection: bool = True,
    ) -> None:
        super().__init__()
        self.d_out, self.d_in, self.rank = d_out, d_in, rank
        self.driver_dim = driver_dim
        self.learned_projection = learned_projection
        # Iteration 3: a detached driver vector is concatenated onto the image context.
        # current_driver is set by the trainer each step (lag-1); zeros → driver=none behaviour.
        if driver_dim > 0:
            self.register_buffer("current_driver", torch.zeros(driver_dim))
        self.signal_net = nn.Sequential(
            nn.Linear(784 + driver_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k),
        )
        mask_init = min(max(mask_init, 1e-4), 1 - 1e-4)
        self.register_buffer("mask_logit_bias", torch.logit(torch.tensor(mask_init)))

        if rank and rank > 0:
            self.coef_head = nn.Linear(k, rank)
            nn.init.zeros_(self.coef_head.weight)   # g=0 at init → M = mask_init everywhere
            nn.init.zeros_(self.coef_head.bias)
            self.A = nn.Parameter(torch.randn(d_out, rank) / (rank ** 0.5))
            self.B = nn.Parameter(torch.randn(d_in, rank) / (rank ** 0.5))
        elif learned_projection:
            self.mask_head = nn.Linear(k, d_out * d_in)
            nn.init.zeros_(self.mask_head.weight)   # logits=0 at init → M = mask_init everywhere
            nn.init.zeros_(self.mask_head.bias)
        else:
            # Fixed random projection: only the signal net is learned. Zero-init its final layer
            # so the k-code is 0 at init → logits 0 → M = mask_init (parity), as in gain/plasticity.
            nn.init.zeros_(self.signal_net[2].weight)
            nn.init.zeros_(self.signal_net[2].bias)
            self.register_buffer("mask_proj", torch.randn(k, d_out * d_in) / (k ** 0.5))

    def set_driver(self, driver: torch.Tensor) -> None:
        """Store the (detached) driver vector for the next forward. No grad path."""
        if self.driver_dim > 0:
            with torch.no_grad():
                self.current_driver.copy_(driver.detach().view(-1))

    def compute_mask(self, context: torch.Tensor) -> torch.Tensor:
        ctx = context.view(context.size(0), -1).mean(dim=0, keepdim=True)  # (1, 784)
        if self.driver_dim > 0:
            ctx = torch.cat([ctx, self.current_driver.view(1, -1)], dim=1)  # (1, 784+dd)
        s = self.signal_net(ctx)                                            # (1, k)
        if self.rank and self.rank > 0:
            g = self.coef_head(s).squeeze(0)            # (rank,)
            logits = (self.A * g) @ self.B.t()          # (d_out, d_in)
        elif self.learned_projection:
            logits = self.mask_head(s).view(self.d_out, self.d_in)
        else:
            logits = (s @ self.mask_proj).view(self.d_out, self.d_in)  # fixed random projection
        return torch.sigmoid(self.mask_logit_bias + logits)


class StatefulModulator(Modulator):
    """Stateful (GRU) modulator for the weight_mask target (Iteration 4).

    Replaces the feedforward signal path of WeightMaskModulator with a GRU cell that
    maintains hidden state across training steps AND across task boundaries (never reset),
    so it can track "what has been learned so far / how much things are shifting" that a
    stateless modulator cannot. Pipeline each step:
        x = [batch-mean image (784), driver (dd)]   (driver as in Iteration 3, default surprise)
        h ← GRUCell(x, h_prev)                       (h_prev detached: truncated BPTT length 1)
        s = Linear(h → k);  logits = mask_head(s) → (d_out, d_in);  M = sigmoid(bias + logits)
    The hidden state is detached each step so the graph stays bounded; the state still carries
    information forward numerically. mask_head is zero-init so M ≈ mask_init (near-vanilla) at
    the start. Same per-synapse mask interface as WeightMaskModulator (used by WeightMaskMLP).
    """

    def __init__(
        self,
        d_out: int = 400,
        d_in: int = 400,
        k: int = 8,
        hidden_size: int = 64,
        rank: int = 0,
        mask_init: float = 0.99,
        driver_dim: int = 0,
    ) -> None:
        super().__init__()
        self.d_out, self.d_in, self.rank = d_out, d_in, rank
        self.hidden_size = hidden_size
        self.driver_dim = driver_dim
        if driver_dim > 0:
            self.register_buffer("current_driver", torch.zeros(driver_dim))
        self.gru = nn.GRUCell(784 + driver_dim, hidden_size)
        self.register_buffer("h", torch.zeros(1, hidden_size))  # persistent state (never reset between tasks)
        self.to_k = nn.Linear(hidden_size, k)

        mask_init = min(max(mask_init, 1e-4), 1 - 1e-4)
        self.register_buffer("mask_logit_bias", torch.logit(torch.tensor(mask_init)))
        if rank and rank > 0:
            self.coef_head = nn.Linear(k, rank)
            nn.init.zeros_(self.coef_head.weight)
            nn.init.zeros_(self.coef_head.bias)
            self.A = nn.Parameter(torch.randn(d_out, rank) / (rank ** 0.5))
            self.B = nn.Parameter(torch.randn(d_in, rank) / (rank ** 0.5))
        else:
            self.mask_head = nn.Linear(k, d_out * d_in)
            nn.init.zeros_(self.mask_head.weight)   # logits=0 at init → M = mask_init everywhere
            nn.init.zeros_(self.mask_head.bias)

    def set_driver(self, driver: torch.Tensor) -> None:
        if self.driver_dim > 0:
            with torch.no_grad():
                self.current_driver.copy_(driver.detach().view(-1))

    def reset_state(self) -> None:
        """Zero the hidden state. Called ONCE at start of training, never on a task boundary."""
        with torch.no_grad():
            self.h.zero_()

    def compute_mask(self, context: torch.Tensor) -> torch.Tensor:
        ctx = context.view(context.size(0), -1).mean(dim=0, keepdim=True)  # (1, 784)
        if self.driver_dim > 0:
            ctx = torch.cat([ctx, self.current_driver.view(1, -1)], dim=1)
        # Feed a clone of the persisted state (truncated BPTT, length 1): the GRU's saved
        # input is the clone, so updating the buffer in place below does not corrupt autograd.
        h_new = self.gru(ctx, self.h.clone())
        s = self.to_k(h_new)                     # (1, k)
        if self.rank and self.rank > 0:
            g = self.coef_head(s).squeeze(0)
            logits = (self.A * g) @ self.B.t()
        else:
            logits = self.mask_head(s).view(self.d_out, self.d_in)
        with torch.no_grad():
            self.h.copy_(h_new.detach())         # persist state for next step (never reset between tasks)
        return torch.sigmoid(self.mask_logit_bias + logits)


class LogitModulator(Modulator):
    """Logit target: context-driven per-sample FiLM on the output logits (pt3 Iteration 6).

    logits' = (1 + γ(x)) ⊙ logits + β(x), with γ, β produced per sample from the input image
    by a small signal net (784→64→k) plus a head k→2·n_classes. The head is zero-init, so
    γ=β=0 at the start and the modulator is identical to vanilla (parity). Reaches the output
    head directly (unlike the pt2 hidden-layer mechanisms): a learned per-input logit
    calibration meant to counteract the class-IL recency bias. Trained on the current task
    alone it just favors current classes, so it is paired with a retention term (output_masking
    'loss', or ER) in the experiments.
    """

    def __init__(self, n_classes: int = 10, k: int = 8, driver_dim: int = 0) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.driver_dim = driver_dim
        # Iteration 9: an optional retention driver (e.g. per-class recency/presence EMA) is
        # concatenated onto the image features so the calibrator can learn to boost stale classes.
        if driver_dim > 0:
            self.register_buffer("current_driver", torch.zeros(driver_dim))
        self.signal_net = nn.Sequential(
            nn.Linear(784 + driver_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k),
            nn.ReLU(),
        )
        self.head = nn.Linear(k, 2 * n_classes)
        nn.init.zeros_(self.head.weight)   # γ=β=0 at init → logits' = logits (vanilla parity)
        nn.init.zeros_(self.head.bias)

    def set_driver(self, driver: torch.Tensor) -> None:
        if self.driver_dim > 0:
            with torch.no_grad():
                self.current_driver.copy_(driver.detach().view(-1))

    def modulate_logits(self, logits: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        feat = context.view(context.size(0), -1)
        if self.driver_dim > 0:
            feat = torch.cat([feat, self.current_driver.view(1, -1).expand(feat.size(0), -1)], dim=1)
        gb = self.head(self.signal_net(feat))  # (B, 2C)
        gamma, beta = gb[:, : self.n_classes], gb[:, self.n_classes :]
        return (1.0 + gamma) * logits + beta


class DirectGainModulator(Modulator):
    """Direct per-neuron gain: gain vectors produced straight from the input, no bottleneck (pt4/5).

    Variant of the pt1 GainModulator. The pt1 version maps the image to a low-dim signal s (k=8)
    then broadcasts it through a fixed/learned projection P_l (k -> hidden) to get the per-neuron
    gain. This version drops the bottleneck and the projection: one head per gated layer maps the
    image directly to that layer's full gain vector, so the neuromod net's weight has shape
    (in=784, out=layer_width). Modulation is the same FiLM gain g_l = (1 + m_l(x)) applied as
    g_l ⊙ h_l (and ⊙ logits for the output layer). Each head is zero-init, so g=1 everywhere at
    init and the model is identical to vanilla (parity).

    gate_hidden: which hidden layers to gate (subset of {0,1}; 1 = last hidden layer).
    gate_output: also gate the 10 output logits (the head pt3 found to be the class-IL bottleneck).
    """

    def __init__(
        self,
        gate_hidden: tuple[int, ...] = (0, 1),
        gate_output: bool = False,
        hidden_dim: int = 400,
        n_classes: int = 10,
        bounded: bool = False,
    ) -> None:
        super().__init__()
        self.gate_hidden = tuple(sorted(set(gate_hidden)))
        self.gate_output = bool(gate_output)
        # bounded=True: gain = 1 + tanh(m) in [0,2] (delta in [-1,1]); else unbounded gain = 1 + m.
        # tanh(0)=0 so zero-init heads still give gain=1 (vanilla parity) either way.
        self.bounded = bool(bounded)
        self.heads = nn.ModuleDict()
        for l_idx in self.gate_hidden:
            lin = nn.Linear(784, hidden_dim)        # direct image -> per-neuron gain (784 x hidden)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)                # m=0 at init -> gain=1 -> vanilla parity
            self.heads[f"h{l_idx}"] = lin
        if self.gate_output:
            lin = nn.Linear(784, n_classes)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.heads["out"] = lin

    def _gain(self, m: torch.Tensor) -> torch.Tensor:
        return 1.0 + (torch.tanh(m) if self.bounded else m)

    def modulate(self, h: torch.Tensor, context: torch.Tensor, layer_idx: int) -> torch.Tensor:
        key = f"h{layer_idx}"
        if key in self.heads:
            m = self.heads[key](context.view(context.size(0), -1))   # (B, hidden)
            return self._gain(m) * h
        return h

    def modulate_logits(self, logits: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if "out" in self.heads:
            m = self.heads["out"](context.view(context.size(0), -1))  # (B, n_classes)
            return self._gain(m) * logits
        return logits


class DirectPlasticityModulator(Modulator):
    """Non-bottleneck plasticity: per-neuron LR gate α∈[0,1] straight from the image.

    The bottlenecked PlasticityModulator maps the image to a k-dim signal then broadcasts it
    through a (fixed-random or learned) projection P_l: k→hidden to get the per-neuron gate.
    This variant drops the bottleneck and the projection: one head Linear(784→hidden_dim) per
    hidden layer emits that layer's full α vector directly (head weight shape 784×hidden). Heads
    are zero-init and a constant logit bias is added before the sigmoid, so α = alpha_init
    (≈ full plasticity) at init regardless of the input. Same gradient-gating interface
    (compute_alphas / param_factors) as PlasticityModulator, so the same lookahead loop trains it.
    """

    def __init__(
        self,
        hidden_dim: int = 400,
        n_hidden_layers: int = 2,
        alpha_init: float = 0.95,
    ) -> None:
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        self.heads = nn.ModuleList()
        for _ in range(n_hidden_layers):
            lin = nn.Linear(784, hidden_dim)        # direct image → per-neuron α logit
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.heads.append(lin)
        alpha_init = min(max(alpha_init, 1e-4), 1 - 1e-4)
        self.register_buffer("alpha_logit_bias", torch.logit(torch.tensor(alpha_init)))

    def compute_alphas(self, context: torch.Tensor) -> dict[int, torch.Tensor]:
        ctx = context.view(context.size(0), -1).mean(dim=0, keepdim=True)  # (1, 784)
        alphas: dict[int, torch.Tensor] = {}
        for l_idx in range(self.n_hidden_layers):
            raw = self.heads[l_idx](ctx).squeeze(0)                        # (hidden_dim,)
            alphas[l_idx] = torch.sigmoid(self.alpha_logit_bias + raw)
        return alphas

    # Per-neuron α → per-parameter multipliers and the in-place hook are identical to the
    # bottlenecked version (they only depend on the α dict), so reuse them directly.
    param_factors = PlasticityModulator.param_factors
    modulate_gradients = PlasticityModulator.modulate_gradients


class DirectWeightMaskModulator(Modulator):
    """Non-bottleneck per-synapse weight mask: M produced straight from the batch-mean image.

    The bottlenecked WeightMaskModulator maps the image to a k-dim signal then a head k→d_out·d_in
    emits the mask logits. This variant drops the k bottleneck: a single head Linear(784→d_out·d_in)
    maps the image directly to all mask logits (head weight shape 784×d_out·d_in — note this is far
    larger than the bottlenecked head, 784× the input width vs k). Zero-init head + logit bias →
    M ≈ mask_init (near-vanilla) at init. Same compute_mask interface (used by WeightMaskMLP) and the
    same optional lag-1 detached driver concatenation as Iteration 3.
    """

    def __init__(
        self,
        d_out: int = 400,
        d_in: int = 400,
        mask_init: float = 0.99,
        driver_dim: int = 0,
    ) -> None:
        super().__init__()
        self.d_out, self.d_in, self.driver_dim = d_out, d_in, driver_dim
        if driver_dim > 0:
            self.register_buffer("current_driver", torch.zeros(driver_dim))
        self.mask_head = nn.Linear(784 + driver_dim, d_out * d_in)
        nn.init.zeros_(self.mask_head.weight)   # logits=0 at init → M = mask_init everywhere
        nn.init.zeros_(self.mask_head.bias)
        mask_init = min(max(mask_init, 1e-4), 1 - 1e-4)
        self.register_buffer("mask_logit_bias", torch.logit(torch.tensor(mask_init)))

    def set_driver(self, driver: torch.Tensor) -> None:
        if self.driver_dim > 0:
            with torch.no_grad():
                self.current_driver.copy_(driver.detach().view(-1))

    def compute_mask(self, context: torch.Tensor) -> torch.Tensor:
        ctx = context.view(context.size(0), -1).mean(dim=0, keepdim=True)  # (1, 784)
        if self.driver_dim > 0:
            ctx = torch.cat([ctx, self.current_driver.view(1, -1)], dim=1)
        logits = self.mask_head(ctx).view(self.d_out, self.d_in)
        return torch.sigmoid(self.mask_logit_bias + logits)


# Direct-gain gate specs (pt4/5): which layers the direct-gain modulator gates.
_GAIN_GATE: dict[str, tuple[tuple[int, ...], bool]] = {
    "last_hidden": ((1,), False),
    "two_hidden": ((0, 1), False),
    "last_hidden_output": ((1,), True),
    "two_hidden_output": ((0, 1), True),
}


# Registry: target name → modulator class.  None = planned but not yet implemented.
_REGISTRY: dict[str, type[Modulator] | None] = {
    "activation": GainModulator,
    "plasticity": PlasticityModulator,
    "weight_mask": WeightMaskModulator,
    "logit": LogitModulator,
    "direct_gain": DirectGainModulator,
    "direct_plasticity": DirectPlasticityModulator,   # non-bottleneck plasticity (convergence study)
    "direct_weight_mask": DirectWeightMaskModulator,   # non-bottleneck weight mask (convergence study)
}

# Accepted modulator-architecture variants (only feedforward is wired pre-Iteration 4).
_VARIANTS: frozenset[str] = frozenset({"feedforward", "stateful", "gain"})


def make_modulator(
    target: str,
    *,
    variant: str = "feedforward",
    learned_projection: bool = False,
    alpha_init: float = 0.95,
    mask_dims: tuple[int, int] | None = None,
    mask_rank: int = 0,
    mask_init: float = 0.99,
    driver: str = "none",
    stateful_hidden: int = 64,
    gain_gate: str = "two_hidden",
    gain_bounded: bool = False,
) -> Modulator:
    """Instantiate a modulator by target. `variant` selects architecture
    (feedforward, or stateful=GRU for the weight_mask target; 'gain' is a legacy
    alias for feedforward). `gain_gate` selects which layers the direct_gain target gates."""
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
    cls = _REGISTRY[target]
    if cls is None:
        raise NotImplementedError(
            f"Neuromod target {target!r} is registered but not implemented yet."
        )
    if driver != "none" and cls is WeightMaskModulator:
        pass  # Iteration 3 drivers (surprise/uncertainty/activation_stats)
    elif driver in ("none", "recency") and cls is LogitModulator:
        pass  # Iteration 9 retention driver on the logit calibrator
    elif driver != "none":
        raise NotImplementedError(
            f"driver {driver!r} is not wired for target {target!r}"
        )
    if variant == "stateful":
        # Iteration 4: stateful (GRU) modulator, wired for the weight_mask target only.
        if cls is not WeightMaskModulator:
            raise NotImplementedError(
                f"stateful variant is wired for the weight_mask target only, not {target!r}"
            )
        if mask_dims is None:
            raise ValueError("weight_mask target requires mask_dims=(d_out, d_in)")
        d_out, d_in = mask_dims
        return StatefulModulator(
            d_out=d_out, d_in=d_in, hidden_size=stateful_hidden, rank=mask_rank,
            mask_init=mask_init, driver_dim=driver_dim(driver),
        )
    if cls is DirectGainModulator:
        if gain_gate not in _GAIN_GATE:
            raise ValueError(f"Unknown gain_gate {gain_gate!r}. Known: {sorted(_GAIN_GATE)}")
        gate_hidden, gate_output = _GAIN_GATE[gain_gate]
        return cls(gate_hidden=gate_hidden, gate_output=gate_output, bounded=gain_bounded)
    if cls is GainModulator:
        return cls(learned_projection=learned_projection)
    if cls is LogitModulator:
        return cls(driver_dim=10 if driver == "recency" else 0)
    if cls is PlasticityModulator:
        return cls(learned_projection=learned_projection, alpha_init=alpha_init)
    if cls is DirectPlasticityModulator:
        return cls(alpha_init=alpha_init)
    if cls is DirectWeightMaskModulator:
        if mask_dims is None:
            raise ValueError("direct_weight_mask target requires mask_dims=(d_out, d_in)")
        d_out, d_in = mask_dims
        return cls(d_out=d_out, d_in=d_in, mask_init=mask_init, driver_dim=driver_dim(driver))
    if cls is WeightMaskModulator:
        if mask_dims is None:
            raise ValueError("weight_mask target requires mask_dims=(d_out, d_in)")
        d_out, d_in = mask_dims
        return cls(
            d_out=d_out, d_in=d_in, rank=mask_rank, mask_init=mask_init,
            driver_dim=driver_dim(driver),
        )
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
        logits = self.base.net[4](h2)                     # (B, 10)
        # Logit-target hook (no-op for GainModulator; used by direct_gain output gating).
        return self.modulator.modulate_logits(logits, x_flat)


class WeightMaskMLP(nn.Module):
    """Sidecar wrapper: applies a per-synapse mask M⊙W at one linear layer (Iteration 2).

    The targeted nn.Linear in base_mlp.net is replaced (in place) by a ModulatedLinear
    carrying the same weights, so the base net is numerically unchanged at init. Each
    forward computes M = modulator.compute_mask(input) and threads it into that layer;
    all other layers run unchanged. The modulator's params are submodules here, so a
    single optimizer over WeightMaskMLP.parameters() trains net and modulator together.
    """

    def __init__(self, base_mlp: nn.Module, modulator: Modulator, layer_idx: int = 2) -> None:
        super().__init__()
        from prototype.model import ModulatedLinear

        old = base_mlp.net[layer_idx]
        if not isinstance(old, nn.Linear):
            raise ValueError(f"layer {layer_idx} is {type(old).__name__}, expected nn.Linear")
        ml = ModulatedLinear(old.in_features, old.out_features, bias=old.bias is not None)
        with torch.no_grad():
            ml.weight.copy_(old.weight)
            if old.bias is not None:
                ml.bias.copy_(old.bias)
        base_mlp.net[layer_idx] = ml

        self.base = base_mlp
        self.modulator = modulator
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        net = self.base.net
        h = x_flat
        for i in range(self.layer_idx):
            h = net[i](h)
        mask = self.modulator.compute_mask(x_flat)        # (d_out, d_in)
        h = net[self.layer_idx](h, mask)                  # ModulatedLinear with mask
        for i in range(self.layer_idx + 1, len(net)):
            h = net[i](h)
        return h


def parse_layer_list(s: str) -> list[int]:
    """Parse a comma-separated ``net`` linear-index list, e.g. "0,2,4" -> [0, 2, 4].

    Empty / whitespace -> []. Duplicates removed, result sorted. Used by the multi-layer
    weight_mask target to pick which linears to mask (see MultiWeightMaskMLP).
    """
    idxs = {int(tok) for tok in s.replace(" ", "").split(",") if tok != ""}
    return sorted(idxs)


class WeightMaskHead(nn.Module):
    """Per-layer up-projection: shared k-dim signal s -> per-synapse mask M ∈ [0,1]^(d_out×d_in).

    This is exactly the mask-head half of WeightMaskModulator, split out so several heads can
    share ONE signal net (see MultiWeightMaskMLP). The head is zero-init, so M = mask_init (near
    vanilla) at the start regardless of s, and each layer's own projection emerges as it trains.

    rank=0  -> full-rank head Linear(k -> d_out·d_in).
    rank=r>0 -> low-rank M = sigmoid(bias + A·diag(g(s))·Bᵀ), A∈(d_out,r), B∈(d_in,r).
    """

    def __init__(self, d_out: int, d_in: int, k: int = 8, rank: int = 0, mask_init: float = 0.99) -> None:
        super().__init__()
        self.d_out, self.d_in, self.rank = d_out, d_in, rank
        mask_init = min(max(mask_init, 1e-4), 1 - 1e-4)
        self.register_buffer("mask_logit_bias", torch.logit(torch.tensor(mask_init)))
        if rank and rank > 0:
            self.coef_head = nn.Linear(k, rank)
            nn.init.zeros_(self.coef_head.weight)   # g=0 at init -> M = mask_init everywhere
            nn.init.zeros_(self.coef_head.bias)
            self.A = nn.Parameter(torch.randn(d_out, rank) / (rank ** 0.5))
            self.B = nn.Parameter(torch.randn(d_in, rank) / (rank ** 0.5))
        else:
            self.mask_head = nn.Linear(k, d_out * d_in)
            nn.init.zeros_(self.mask_head.weight)   # logits=0 at init -> M = mask_init everywhere
            nn.init.zeros_(self.mask_head.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if self.rank and self.rank > 0:
            g = self.coef_head(s).squeeze(0)        # (rank,)
            logits = (self.A * g) @ self.B.t()      # (d_out, d_in)
        else:
            logits = self.mask_head(s).view(self.d_out, self.d_in)
        return torch.sigmoid(self.mask_logit_bias + logits)


class MultiWeightMaskMLP(nn.Module):
    """Sidecar wrapper: applies a per-synapse mask M⊙W at MULTIPLE linear layers at once.

    Generalizes WeightMaskMLP (which masks a single layer) to any subset of the MLP's linears,
    including the output head (net.4). Each targeted nn.Linear is replaced in place by a
    ModulatedLinear carrying the same weights (so the base net is numerically unchanged at init).

    ONE shared signal net (784 -> 64 -> k) computes the k-dim code s from the batch-mean image;
    each masked layer has its OWN WeightMaskHead (up-projection) that maps that single s to its own
    per-synapse mask. So the signal is shared across layers and only the projection differs. All
    params are submodules here, so one optimizer over parameters() trains the net, the signal net,
    and every head together. With one layer this is behaviourally equivalent to WeightMaskMLP
    (learned_projection form).
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        layer_dims: dict[int, tuple[int, int]],
        k: int = 8,
        rank: int = 0,
        mask_init: float = 0.99,
    ) -> None:
        super().__init__()
        from prototype.model import ModulatedLinear

        if not layer_dims:
            raise ValueError("MultiWeightMaskMLP needs at least one layer to mask")
        self.base = base_mlp
        self.layer_indices = sorted(layer_dims)
        # Shared bottleneck: one signal net for all layers (normal init; heads carry the zero-init).
        self.signal_net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, k),
        )
        # Per-layer up-projections (nn.ModuleDict keys must be strings; key by net.<idx>).
        self.heads = nn.ModuleDict({
            str(i): WeightMaskHead(layer_dims[i][0], layer_dims[i][1], k=k, rank=rank, mask_init=mask_init)
            for i in self.layer_indices
        })

        for idx in self.layer_indices:
            old = base_mlp.net[idx]
            if not isinstance(old, nn.Linear):
                raise ValueError(f"layer {idx} is {type(old).__name__}, expected nn.Linear")
            ml = ModulatedLinear(old.in_features, old.out_features, bias=old.bias is not None)
            with torch.no_grad():
                ml.weight.copy_(old.weight)
                if old.bias is not None:
                    ml.bias.copy_(old.bias)
            base_mlp.net[idx] = ml

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        ctx = x_flat.mean(dim=0, keepdim=True)   # (1, 784)
        s = self.signal_net(ctx)                 # (1, k) shared across all masked layers
        h = x_flat
        for i, layer in enumerate(self.base.net):
            key = str(i)
            if key in self.heads:
                mask = self.heads[key](s)        # (d_out, d_in) from the shared signal
                h = layer(h, mask)               # ModulatedLinear with mask
            else:
                h = layer(h)
        return h


class TaskInferenceNet(nn.Module):
    """Small classifier g(x) -> task id, for class-IL task-inferred routing (pt3 Iteration 8).

    Trained sequentially on the current task's index (no replay, no task ID at test). At eval the
    inferred task selects which output classes are active (lever C: the legal class-IL substitute
    for HAT's task input). Its routing accuracy is the binding constraint and is measured directly.
    """

    def __init__(self, n_tasks: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class LogitModulatedMLP(nn.Module):
    """Sidecar wrapper: applies a per-sample logit calibration after the base MLP (Iteration 6)."""

    def __init__(self, base_mlp: nn.Module, modulator: Modulator) -> None:
        super().__init__()
        self.base = base_mlp
        self.modulator = modulator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        logits = self.base(x_flat)
        return self.modulator.modulate_logits(logits, x_flat)
