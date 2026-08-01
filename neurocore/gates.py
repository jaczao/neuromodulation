"""Gate primitives: the rank-K linear gain gate, the gain forms, and the modulator head.

Extracted verbatim from results/pt7_neuromodulators.py (NeuronGate / SynapseGate / Heads) and
prototype/neuromod.py (the gain forms and their init-parity semantics).

THE RANK-K LINEAR GATE.  Gamma_i = 1 + sum_k m_ik P_k, with m:(B,K) the per-sample driver values and
P the projection. Because Gamma is LINEAR in m, the per-synapse form

    (Gamma_i . W) x_i + b  =  W x_i + b + sum_k m_ik (P_k . W) x_i

costs K+1 matmuls per layer and needs NO (B, d_out, d_in) per-sample expansion — which is what makes
per-synapse granularity tractable at all (P is (K, n_syn) ~ 1.9M, not 374M). Linearity is also why a
soft blend over per-task gates is exact in T matmuls: Gamma_i = sum_t p_it Gamma_t and the bias rides
along because sum_t p_it = 1.

INIT PARITY (read this before choosing a form):
  - `unbounded` (1 + raw) and `positive` (softplus(raw + ln(e-1))) are EXACTLY 1.0 at raw = 0, so a
    zero-init P starts at parity with the unmodulated network.
  - `bounded01` (sigmoid(raw)) is 0.5 at raw = 0 -- NOT parity. It halves every gated activation from
    step one and is capped at 1, so it can never recover the scale. Under SGD that collapses the run;
    Adam absorbs the uniform rescale. Low accuracy AT low forgetting is the over-suppression
    signature (under-learning, not retention) -- never read it as a retention win.

PER-LAYER REPORTING. `per_layer_mag` returns |gate deviation| per layer, never a single mean: a scalar
mean over 4050 entries hid the fact that the ER-arm gate is a pure per-task OUT-layer logit adjustment
(h0 0.001 / h1 0.002 / out 0.107). Which layer the gate settles in is set by the ARM, not the driver:
with replay it goes to the out layer, standalone it goes to the hidden layers.
"""
import math
from typing import NamedTuple, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DEV


class GateDims(NamedTuple):
    """Widths the gate spans, passed at CONSTRUCTION.

    These used to be module globals that the classes read at call time, which meant the only way to
    run a different width was to rebind them from outside (`pt7_capacity.width(H)` mutating
    p7.H0/H1/GATEDIM). That does not survive several problem packages running side by side, so the
    dims now travel with the gate instance. Defaults are the Split-MNIST MLP, so existing behaviour
    is unchanged and the frozen anchors still reproduce bit-exact.
    """
    in_dim: int = 784
    h0: int = 400
    h1: int = 400
    out: int = 10

    @property
    def total(self) -> int:
        """Per-neuron gate width: one gain per gated unit across h0, h1 and the logits."""
        return self.h0 + self.h1 + self.out


DEFAULT_DIMS = GateDims()


# ------------------------------- gain forms -------------------------------
GAIN_FORMS = ("unbounded", "bounded01", "positive")

# softplus(x + SOFTPLUS_PARITY_BIAS) == 1.0 at x == 0, so `positive` shares `unbounded`'s neutral
# init under a zero-init learned P: ln(1 + e^b) = 1  <=>  b = ln(e - 1).
SOFTPLUS_PARITY_BIAS = math.log(math.e - 1.0)


def check_gain_form(form: str) -> str:
    """Validate a gain form at construction. gain_gamma short-circuits on fixed projections (never
    reaching its own form dispatch), so a typo would otherwise pass silently under disjoint/shared."""
    if form not in GAIN_FORMS:
        raise ValueError(f"unknown gain form {form!r}; known: {' | '.join(GAIN_FORMS)}")
    return form


def gain_gamma(raw: torch.Tensor, *, fixed: bool, form: str) -> torch.Tensor:
    """Gain gamma from raw = z @ P. The one place the gain form is applied, for BOTH granularities.

    Fixed projections: raw is binary {0,1}, used DIRECTLY (no squashing), so ALL forms collapse to
    the same suppress-only {0,1} gamma -- an exact 0 hard-freezes a unit, which is the pt5 iter-1
    lever (a hard freeze kills the gradient and is therefore un-absorbable; a soft gate is absorbed).
    Learned projections:
      - bounded01: sigmoid(raw) in (0,1)         -- suppress-only; this is the weight_mask gamma.
      - unbounded: 1 + raw in (-inf,+inf)        -- init 1.0; amplifies above 1, INVERTS below 0.
      - positive:  softplus(raw + b) in (0,+inf) -- init 1.0; amplifies, never inverts.
    `positive` cannot hard-freeze (softplus is 0 only asymptotically), and its L1 pull vanishes as
    raw -> -inf (dgamma/draw = sigmoid(raw + b)) whereas `unbounded`'s is constant |1|, so it is not
    expected to reproduce unbounded's sparsity result -- it ablates whether sign inversion matters.
    """
    if fixed:
        return raw
    if form == "bounded01":
        return torch.sigmoid(raw)
    if form == "unbounded":
        return 1.0 + raw
    if form == "positive":
        return F.softplus(raw + SOFTPLUS_PARITY_BIAS)
    raise ValueError(f"unknown gain form {form!r}; known: {' | '.join(GAIN_FORMS)}")


def gate_l1(gamma: torch.Tensor) -> torch.Tensor:
    """L1 sparsity term on an APPLIED gate: mean |gamma|.

    CAVEAT (pt6 followup E): against a JOINTLY-TRAINED backbone this is a scale degeneracy -- the
    backbone simply rescales W to absorb the gate's magnitude, so the penalty shifts scale between
    gamma and W without changing the function (mean|P| moved 0.06 -> 1.0 with accuracy flat). It bit
    only where the gate is trained by a SEPARATE loss from the weights it multiplies (the pt5 iter-3
    learned-projection standalone arms, where it pushed the gate toward the disjoint {0,1}).
    """
    return gamma.abs().mean()


def apply_neuron_gain(net, raw, x, dims: "GateDims" = None):
    """Apply an ALREADY-COMPUTED per-sample per-neuron gain `raw`:(B, dims.total) to a 3-layer MLP.

    Split out from NeuronGate.forward because the gain does not have to come from a rank-K driver
    gate: a task-SELECTOR (neurocore.task_selection) produces the same (B, total) vector by indexing or
    blending a per-task table. One place applies it, whatever produced it.
    """
    d = dims or DEFAULT_DIMS
    x = x.view(x.size(0), -1)
    z0 = F.relu(net.l0(x)) * (1 + raw[:, :d.h0])
    z1 = F.relu(net.l1(z0)) * (1 + raw[:, d.h0:d.h0 + d.h1])
    return net.l2(z1) * (1 + raw[:, d.h0 + d.h1:d.total])


# ------------------------------- rank-K gates -------------------------------
class NeuronGate(nn.Module):
    """Gamma = 1 + m @ P over the per-neuron gains; P:(K, dims.total)  (810 by default)."""
    def __init__(self, K, layers, dims: GateDims = DEFAULT_DIMS):
        super().__init__()
        self.dims = dims
        self.P = nn.Parameter(torch.zeros(K, dims.total)); self.set_layers(layers)

    def _slices(self):
        d = self.dims
        return slice(0, d.h0), slice(d.h0, d.h0 + d.h1), slice(d.h0 + d.h1, d.total)

    def set_layers(self, layers):                  # zero (and freeze grad on) disallowed columns
        s0, s1, so = self._slices()
        m = torch.zeros(self.dims.total)
        if layers is None or "h0" in layers:  m[s0] = 1
        if layers is None or "h1" in layers:  m[s1] = 1
        if layers is None or "out" in layers: m[so] = 1
        self.register_buffer("lm", m)

    def raw(self, m):                              # m:(B,K) -> (B, dims.total)
        return (m @ self.P) * self.lm

    def forward(self, net, m, x, detach_P=False):
        P = self.P.detach() if detach_P else self.P
        return apply_neuron_gain(net, (m @ P) * self.lm, x, self.dims)

    def params(self):
        return [self.P]

    @torch.no_grad()
    def per_layer_mag(self, m):
        raw = (m @ self.P) * self.lm
        s0, s1, so = self._slices()
        return {"h0": raw[:, s0].abs().mean().item(),
                "h1": raw[:, s1].abs().mean().item(),
                "out": raw[:, so].abs().mean().item()}


class SynapseGate(nn.Module):
    """(Gamma . W)x = Wx + sum_k m_k (P_k . W)x ; P per layer (K, d_out, d_in). Layers 0,2,4 map l0,l1,l2."""
    def __init__(self, K, layers, dims: GateDims = DEFAULT_DIMS):
        super().__init__()
        self.dims = dims
        self.on = {"h0": layers is None or "h0" in layers, "h1": layers is None or "h1" in layers,
                   "out": layers is None or "out" in layers}
        self.P0 = nn.Parameter(torch.zeros(K, dims.h0, dims.in_dim)) if self.on["h0"] else None
        self.P1 = nn.Parameter(torch.zeros(K, dims.h1, dims.h0)) if self.on["h1"] else None
        self.P2 = nn.Parameter(torch.zeros(K, dims.out, dims.h1)) if self.on["out"] else None

    @staticmethod
    def _layer(inp, lin, P, m, detach_P):
        base = F.linear(inp, lin.weight, lin.bias)
        if P is None:
            return base
        Pw = (P.detach() if detach_P else P) * lin.weight.unsqueeze(0)      # (K,do,di)
        mod = torch.einsum("kod,bd->bko", Pw, inp)                          # (B,K,do)
        return base + (m.unsqueeze(-1) * mod).sum(1)

    def forward(self, net, m, x, detach_P=False):
        x = x.view(x.size(0), -1)
        z0 = F.relu(self._layer(x, net.l0, self.P0, m, detach_P))
        z1 = F.relu(self._layer(z0, net.l1, self.P1, m, detach_P))
        return self._layer(z1, net.l2, self.P2, m, detach_P)

    def params(self):
        return [p for p in (self.P0, self.P1, self.P2) if p is not None]

    @torch.no_grad()
    def per_layer_mag(self, m, net):               # cheap proxy: mean_k |m_k| * |P_k . W|  (no BxKxdoxdi tensor)
        out = {}
        mk = m.abs().mean(0)                        # (K,)
        for name, lin, P in (("h0", net.l0, self.P0), ("h1", net.l1, self.P1), ("out", net.l2, self.P2)):
            if P is None:
                out[name] = 0.0; continue
            pw = (P * lin.weight.unsqueeze(0)).abs().mean(dim=(1, 2))   # (K,)
            out[name] = float((mk * pw).sum().item())
        return out


def make_gate(gran, K, layers, dims: GateDims = DEFAULT_DIMS):
    return (NeuronGate(K, layers, dims) if gran == "neuron"
            else SynapseGate(K, layers, dims)).to(DEV)


def gate_K(gate, gran):
    """Rank K of a gate, whichever granularity it is."""
    return gate.P.size(0) if gran == "neuron" else next(p for p in (gate.P0, gate.P1, gate.P2)
                                                        if p is not None).size(0)


# ------------------------------- modulator head -------------------------------
@runtime_checkable
class ModulatorHead(Protocol):
    """The contract a modulator net satisfies: x -> m of shape (B, K).

    The gate only ever consumes m, so ANY module meeting this signature drives it — the MLP head
    below, or a recurrent head that carries state across steps (pt7's stateful GRU predictors are the
    worked example, frozen in results/pt7_stateful.py; port one here when a second problem calls for
    it rather than re-implementing it speculatively).

    Two invariants a replacement must keep:
      - zero-init the OUTPUT layer, so the gate starts at parity (gamma = 1);
      - be aware that doing so puts it in the double-zero-init saddle unless something else forces
        m != 0 (an MSE target on a biological tau, or a non-zero P). See neurocore.controls.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class Heads(nn.Module):
    """m_k(x): in_dim -> hid -> K. K per-sample scalars; drives the gate (train & eval).

    Zero-init on the OUTPUT layer so the gate starts at parity. NOTE this is exactly one half of the
    double-zero-init saddle (see neurocore.controls): zero-init heads feeding a zero-init P give
    neither a gradient, so a head with no external target (the `free` control) never leaves |g| = 0.
    A head pinned by an MSE target to a biological tau escapes the saddle because the target forces
    m != 0 regardless of P.
    """
    def __init__(self, K, hid=32, in_dim=784):
        super().__init__(); self.f1 = nn.Linear(in_dim, hid); self.f2 = nn.Linear(hid, K)
        nn.init.zeros_(self.f2.weight); nn.init.zeros_(self.f2.bias)   # start ~0 -> gate ~parity

    def forward(self, x):
        return self.f2(F.relu(self.f1(x.view(x.size(0), -1))))


MLPHead = Heads          # explicit name for when a second head type exists alongside it
