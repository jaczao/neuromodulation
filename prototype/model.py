import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Vanilla MLP: 784 → 400 → 400 → 10, ReLU activations, no batchnorm, no dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class ModulatedLinear(nn.Module):
    """nn.Linear with an optional externally-supplied per-synapse mask M (Iteration 2).

    forward(x, mask=None, bias_mask=None):
        y = (M ⊙ W) x + (β ⊙ b)   with mask M and per-neuron bias_mask β
        y = (M ⊙ W) x + b          if only mask is given
        y =  W x + b               otherwise  (exactly nn.Linear, parity)

    The masks are supplied per forward call (computed by a modulator), not owned here.
    `bias_mask` is per output neuron (shape (out_features,)); it gates the bias in the
    forward AND, symmetrically with M, its gradient (β_j=0 → bias_j unused and frozen).
    Weights are initialised exactly like nn.Linear so that, with no masks, a
    ModulatedLinear is numerically identical to the nn.Linear it replaces.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.weight if mask is None else mask * self.weight
        bias = self.bias
        if bias is not None and bias_mask is not None:
            bias = bias_mask * bias
        return F.linear(x, weight, bias)
