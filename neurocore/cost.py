"""First-class cost accounting — memory, parameters, and compute at train AND at inference.

Every direction is to be reported in three memory regimes, so what a method COSTS is part of the
result, not a footnote. These become ledger columns (see ledger.COST_METRICS).

  regime            normal | rehearsal-free | memory-budgeted. A KEY column, not a metric: the same
                    mechanism reported under a different regime is a different cell.
  buffer_bytes      resident bytes of stored samples. 0 is what makes a rehearsal-free method's claim
                    legible — DGR's ~91% bar is reached at zero stored bytes, which no ER row can say.
  backbone_params   parameters of the network being modulated.
  extra_params      parameters the mechanism ADDS (modulator head + gate projection + any auxiliary
                    net). Reported alongside, never folded into a single total.
  param_ratio       extra_params / backbone_params. THE capacity-confound guard: pt7's capacity
                    ablation found the one arm that helped under scarcity was the content-free one,
                    and it helped because its 25,252-param head was 6.3x the 4,015-param backbone it
                    "modulated". A win from a modulator comparable in size to what it modulates is a
                    capacity result, not a gain-control result. `capacity_confound` flags it.
  fwd_train/bwd_train    forward / backward passes per optimisation step.
  fwd_infer/bwd_infer    forward / backward passes per inference sample. This is the column the TTA
                    work is expected to win outright, and the one that prices honestly: pt7's Signals
                    costs an extra plain forward per step, a two-pass `true` eval costs two at
                    inference, and TENT-style adaptation costs a backward at inference where a frozen
                    source model costs none.

Counts are DECLARED by the mechanism (they are structural, not measurable without ambiguity) and can
be cross-checked at runtime with `ForwardCounter`, which catches declarations that drift away from
the loop they describe.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch.nn as nn

REGIMES = ("normal", "rehearsal-free", "memory-budgeted")

# param_ratio at or above which a positive result must be read as a capacity confound first.
CONFOUND_RATIO = 1.0


@dataclass(frozen=True)
class Cost:
    """Declared cost of one cell. `as_row()` feeds the ledger."""
    backbone_params: int
    extra_params: int = 0
    buffer_bytes: int = 0
    fwd_train: float = 1.0
    bwd_train: float = 1.0
    fwd_infer: float = 1.0
    bwd_infer: float = 0.0

    @property
    def param_ratio(self) -> float:
        return self.extra_params / self.backbone_params if self.backbone_params else float("inf")

    def capacity_confound(self, threshold: float = CONFOUND_RATIO) -> bool:
        """True when the modulator is comparable to or larger than what it modulates."""
        return self.param_ratio >= threshold

    def as_row(self) -> dict:
        d = asdict(self)
        d["param_ratio"] = round(self.param_ratio, 4)
        return d

    def warn_if_confounded(self, label="cell", threshold: float = CONFOUND_RATIO) -> str | None:
        if not self.capacity_confound(threshold):
            return None
        return (f"{label}: extra_params={self.extra_params:,} is {self.param_ratio:.2f}x "
                f"backbone_params={self.backbone_params:,} — any positive result here is a capacity "
                f"confound until shown otherwise (pt7 capacity ablation).")


def count_params(*modules, trainable_only=False) -> int:
    """Total parameters across modules. None entries are skipped so optional parts (heads, gate,
    auxiliary nets) can be passed unconditionally."""
    n = 0
    for m in modules:
        if m is None:
            continue
        ps = m.parameters() if isinstance(m, nn.Module) else m
        n += sum(p.numel() for p in ps if not trainable_only or p.requires_grad)
    return n


def buffer_bytes(buf) -> int:
    """Resident bytes of a replay buffer. 0 for a rehearsal-free method (pass None)."""
    if buf is None:
        return 0
    if hasattr(buf, "nbytes"):
        return buf.nbytes()
    raise TypeError(f"cannot size buffer of type {type(buf).__name__}; give it an nbytes() method")


def _entry_leaf(module: nn.Module) -> nn.Module:
    """The first leaf submodule — the one a forward pass enters through.

    Hooks are attached HERE, not on the tracked module, because a forward hook only fires on
    __call__: the backbone contract exposes `plain(x)`, so `net.plain(x)` would never trigger a hook
    on `net` itself. The entry leaf fires once per forward however the forward is invoked.
    """
    for m in module.modules():
        if not list(m.children()):
            return m
    return module


class ForwardCounter:
    """Cross-check a declared fwd count by counting real forward calls on the modules involved.

    Declarations drift when a loop grows an extra pass; this catches that. Wrap ONE representative
    step (or one inference batch) and compare against the declared number.

        with ForwardCounter(net, heads) as c:
            ...one training step...
        assert c.count(net) == cost.fwd_train

    Counts passes through each module's ENTRY LEAF, so it is accurate for feedforward stacks. A
    module that re-enters its own first leaf several times per forward (a recurrent unrolled loop)
    counts once per re-entry — which is usually what you want for a compute column, but state it
    when reporting.
    """
    def __init__(self, *modules):
        self.modules = [m for m in modules if m is not None]
        self.counts = {}
        self._handles = []

    def __enter__(self):
        for m in self.modules:
            self.counts[id(m)] = 0
            self._handles.append(_entry_leaf(m).register_forward_hook(self._make_hook(m)))
        return self

    def _make_hook(self, m):
        def hook(_mod, _inp, _out):
            self.counts[id(m)] += 1
        return hook

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def count(self, module) -> int:
        return self.counts.get(id(module), 0)


@contextmanager
def counted(*modules):
    with ForwardCounter(*modules) as c:
        yield c
