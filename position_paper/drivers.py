"""Driver providers for the position-paper mechanisms (THESIS-PLAN direction B).

COPY-FORWARD of `pt5_taskil/plast_drivers.py`'s driver adapters, not a cut (CLAUDE.md's extraction
rule): that study's numbers depend on module-construction ORDER for their RNG stream, so its file
stays byte-identical and keeps reproducing. This is their second use; a third should promote them
into `neurocore/`.

ADDED HERE: `taskid`, the task one-hot. Every driver in `plast_drivers` was content-based and
oracle-free; the position-paper mechanisms are asked to run with the task id as well, so the two
families sit side by side under one interface.

  WHERE THE ORACLE IS, PRECISELY. `taskid` uses the task boundary at TRAINING time, which both
  protocols here already hand over (the task sequence is given). It is not an EVAL-time oracle for
  these mechanisms, because neither weight decay nor plasticity puts its gate in the forward — at
  test the net is the plain unmodulated net and no driver value reaches a prediction. That is a
  weaker claim than pt5's forward-gain studies could make, where selecting P[t] at eval WAS an
  oracle and capped everything at `pred ~= oracle x infer`. Under a FORWARD target, `taskid` would
  re-acquire the eval oracle and must be labelled as such.

STANDARDIZATION is per driver, by CLAUDE.md's rule (standardize a per-sample driver, NEVER a tonic
one — a tonic driver's within-batch variance is ~0, so standardizing divides by ~EPS and pt7
measured the resulting gate blow-up at 0.098 = chance). `ach` and `ach_ema` are the same signal on
opposite sides of that rule and act as a built-in check on it. `taskid` is a one-hot: already O(1),
exactly constant within a batch, and standardizing it would divide by ~0 — so it is RAW, for the
same reason a tonic driver is.
"""
import copy
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

import pt7_neuromodulators as p7                                   # noqa: E402  (frozen, read-only)
import pt7_plast_tempslope as pts                                  # noqa: E402  (frozen, read-only)
from pt7_variants import NEDriver                                  # noqa: E402

DEV, EPS = p7.DEV, p7.EPS
N_TASKS = 5

# The seven drivers this direction runs. `all5` is the rank-5 composite of the five content drivers.
SINGLE = ("taskid", "ach", "ach_ema", "nerisez", "vec_x", "vecproj")
CONTROLS = ("const",)          # content-free; see ConstDriver
ALL5 = ("ach", "ach_ema", "nerisez", "vec_x_norm", "vecproj_norm")
COMPOSITE = "all5"
DRIVERS = SINGLE + (COMPOSITE,)
NORM_DRIVERS = ("vec_x_norm", "vecproj_norm")

# vec_x is K=784, so a per-SYNAPSE P is (784, 313600)+(784, 160000)+(784, 4000) = 3.7e8 params, ~1.5 GB
# before Adam's moments and ~780x the 478k backbone. Skipped deliberately, and not for memory:
# pt7_capacity's rule is that a modulator comparable to or larger than its backbone makes any result
# a capacity confound, so the cell would be uninterpretable even if it fit. `vecproj` is the
# synapse-tractable form of exactly this driver — that is why pt7 introduced it — and it is run.
SKIP = {("vec_x", "synapse")}

# SINGLE-driver cells run UNSTANDARDISED (user-directed). This is a deliberate deviation from
# CLAUDE.md's "standardize per-sample drivers, NEVER a tonic one", and it is worth stating what it
# buys and what it costs, because the rule was not free to learn:
#   BUYS  the raw tau is what the position paper's f_w(s_t) actually names — a signal, not a
#         z-score. Standardization is a linear rescale the gate could in principle absorb into P,
#         so running raw asks whether the rule is about the DRIVER or about the OPTIMIZER's ability
#         to find the right P scale.
#   COSTS the raw drivers have wildly different scales (pt7_driver_traces measured NE_emb/vec_* at
#         15-35, ACh at 0.5 +- 0.32, NE spiking to 1.2e6), and this mechanism's gate is exp() and
#         COMPOUNDS over ~4750 steps. A raw driver of magnitude 30 makes the usable neuro_lr range
#         far narrower than a standardized one, and it will differ per driver — so the stable range
#         has to be measured per driver rather than assumed from `taskid`.
# The tonic/per-sample distinction still shows up, just not through standardization: `ach_ema` has
# ~zero within-batch variance either way, so it drives a near-constant gate.
STANDARDIZE = {"taskid": False, "const": False, "ach": False, "ach_ema": False, "nerisez": False,
               "vec_x": False, "vecproj": False, "vec_x_norm": False, "vecproj_norm": False}

# The COMPOSITE keeps per-column standardization (user-directed: `all5` is excluded from the
# raw-driver change). This is not an inconsistency — it is forced by two findings that point in
# opposite directions. A TONIC column collapses WITH standardization (within-batch variance ~0 ->
# divides by ~EPS -> pt7 measured 0.098 = chance), while a MIXED-SCALE composite collapses WITHOUT
# it (pt7's all4 at std0 under SGD went to NaN). A single driver has no mixing problem, so it can
# run raw; a rank-5 gate summing five columns of different scale cannot, or the largest column
# swamps the rest until P shrinks its weight.
STANDARDIZE_COMPOSITE = {"ach": True, "ach_ema": False, "nerisez": False,
                         "vec_x_norm": True, "vecproj_norm": True}


# ============================================================================== the new driver
class TaskIdDriver:
    """The task one-hot, m(x) = e_t broadcast over the batch. K = N_TASKS.

    Constant within a batch by construction, so it consumes NO construction RNG and has no state.
    That makes it the cleanest possible contrast with the content drivers: same gate, same loop,
    same meta-loss, and the maximally task-informative signal. `plast_drivers` found that the MOST
    task-decodable content driver (`vec_x`, probe 0.934) was also its WORST cell, so this is the
    matched high-information control for that finding rather than an expected win.
    """

    def __init__(self, *_, **__):
        self.name = "taskid"
        self.kind = "taskid"
        self.K = N_TASKS
        self.t = 0

    def set_task(self, t):
        self.t = t

    def value(self, net, X, update=True):
        v = torch.zeros(X.size(0), self.K, device=DEV)
        v[:, self.t] = 1.0
        return v

    def train_head(self, net, X, Y):
        return

    def live_update(self, net, X):
        return

    def state(self):
        return self.t

    def restore(self, st):
        self.t = st


# ============================================================================== copy-forwarded
class NormNovelty:
    """L2 norm of a headless novelty vector -> a 1-D per-sample driver, standardized as a scalar.

    ORDER MATTERS AND IS THE OPPOSITE OF THE OBVIOUS ONE: the norm is taken on the RAW diff and the
    resulting SCALAR is standardized. Standardizing per-dimension first and then taking the norm
    concentrates the value at sqrt(K) (pt7_driver_traces' methodology note) — it would manufacture a
    near-constant driver rather than measure novelty. This order also sidesteps the
    constant-dimension blow-up: 212 of vec_x's 784 dims have zero variance, but ||x - ema(x)|| does not.
    """

    def __init__(self, kind, standardize=True):
        self.inner = NEDriver(kind, standardize=False)     # raw vector; we standardize the norm
        self.standardize = standardize
        self.rm = None; self.rv = None; self.inited = False

    def K(self):
        return 1

    @torch.no_grad()
    def value(self, net, x, update=True):
        v = self.inner.value(net, x, update=update).norm(dim=1, keepdim=True)
        if update:
            bm, bv = v.mean(0), v.var(0, unbiased=False)
            if not self.inited:
                self.rm, self.rv, self.inited = bm.clone(), bv.clone(), True
            else:
                self.rm = 0.99 * self.rm + 0.01 * bm
                self.rv = 0.99 * self.rv + 0.01 * bv
        if not self.inited or not self.standardize:
            return v
        return (v - self.rm) / (self.rv.sqrt() + EPS)

    def state(self):
        d = self.inner
        return copy.deepcopy((self.rm, self.rv, self.inited,
                              d.mh1, d.mx, d.ch1, d.cx, d.run_mean, d.run_var, d.inited))

    def restore(self, st):
        d = self.inner
        (self.rm, self.rv, self.inited,
         d.mh1, d.mx, d.ch1, d.cx, d.run_mean, d.run_var, d.inited) = copy.deepcopy(st)


class Driver:
    """m(x) provider + its own optimizer. Every driver but `taskid` is oracle-free."""

    HEAD_KEY = {"ach": "ACh", "ach_ema": "ACh_ema"}

    def __init__(self, name, lr, standardize=None):
        """`standardize=None` takes the single-driver default (raw); `CompositeDriver` passes its
        own per-column rule explicitly, so the two policies cannot drift apart silently."""
        self.name = name
        std = STANDARDIZE[name] if standardize is None else standardize
        self.standardize = std
        if name in self.HEAD_KEY:                                  # Signals head: 784->32->1 regresses tau
            self.kind = "head"; self.K = 1
            self.heads = p7.Heads(1).to(DEV)
            self.sig = p7.Signals([self.HEAD_KEY[name]], standardize=std)
            self.opt = torch.optim.Adam(self.heads.parameters(), lr)
        elif name in ("vec_x", "vecproj"):                         # head-free input novelty
            self.kind = "ne"
            self.drv = NEDriver(name, std)
            self.K = self.drv.K()
        elif name in NORM_DRIVERS:                                 # 1-D norm of the same novelty
            self.kind = "ne"
            self.drv = NormNovelty(name.replace("_norm", ""), standardize=std)
            self.K = self.drv.K()
        elif name == "nerisez":                                    # stateful entropy z-score (MLP)
            self.kind = "stateful"; self.K = 1
            self.drv = pts.StatefulStd("nerisez", gru=False, standardize=std).to(DEV)
            self.opt = torch.optim.Adam(self.drv.parameters(), lr)
        else:
            raise ValueError(name)

    def set_task(self, t):
        return                                                     # only `taskid` cares

    def value(self, net, X, update=True):
        """(B,K) driver, DETACHED — the gate path never grads back into the driver."""
        if self.kind == "head":
            return self.heads(X).detach()
        if self.kind == "ne":
            return self.drv.value(net, X, update=update).detach()
        return self.drv.driver(X, update_state=update, update_stats=False).detach()

    def train_head(self, net, X, Y):
        if self.kind == "ne":
            return                                                 # deterministic, nothing to train
        if self.kind == "head":
            hloss = F.mse_loss(self.heads(X), self.sig.targets(net, X, Y))
        else:
            with torch.no_grad():
                Hact = p7.entropy(net.plain(X)[0]).unsqueeze(1)
            self.drv.upd_actual(Hact)
            hloss = F.mse_loss(self.drv.predictH(X, update_state=False), Hact)
        self.opt.zero_grad(); hloss.backward(); self.opt.step()

    def live_update(self, net, X):
        """The 'no freezes at inference' update for state the gate path does NOT already advance.

        Only `nerisez` needs it: its z-score divides by ACTUAL-entropy stats advanced by
        `train_head`, not by reading the driver. HEAD drivers have nothing to advance (the gate
        reads m(x) = heads(x), a pure function of the image), and NEDriver advances its own stats
        inside `value(update=True)`.
        """
        if self.kind != "stateful":
            return
        with torch.no_grad():
            self.drv.upd_actual(p7.entropy(net.plain(X)[0]).unsqueeze(1))

    def state(self):
        """Snapshot of every mutable running statistic (frozen-vs-live eval without cross-talk)."""
        if hasattr(self.drv if self.kind != "head" else None, "state"):
            return self.drv.state()                        # wrapper owns its own stats (NormNovelty)
        if self.kind == "head":
            s = self.sig
            return copy.deepcopy((s.ef, s.es, s.esq, s.er, s.prev, s.emaH, s.mh1,
                                  s.run_mean, s.run_var, s.inited))
        if self.kind == "ne":
            d = self.drv
            return copy.deepcopy((d.mh1, d.mx, d.ch1, d.cx, d.run_mean, d.run_var, d.inited))
        d = self.drv
        return copy.deepcopy((d.emaH, d.varH, d.rm, d.rv,
                              d.hidden if getattr(d, "gru", False) else None))

    def restore(self, st):
        if hasattr(self.drv if self.kind != "head" else None, "restore"):
            self.drv.restore(st)
            return
        if self.kind == "head":
            s = self.sig
            (s.ef, s.es, s.esq, s.er, s.prev, s.emaH, s.mh1,
             s.run_mean, s.run_var, s.inited) = copy.deepcopy(st)
        elif self.kind == "ne":
            d = self.drv
            (d.mh1, d.mx, d.ch1, d.cx, d.run_mean, d.run_var, d.inited) = copy.deepcopy(st)
        else:
            d = self.drv
            d.emaH, d.varH, d.rm, d.rv, hidden = copy.deepcopy(st)
            if hidden is not None:
                d.hidden = hidden


class CompositeDriver:
    """all5: five single drivers concatenated into one (B,5) code driving a rank-5 gate.

    EACH COLUMN KEEPS ITS OWN STANDARDIZATION RULE rather than standardizing the stacked vector,
    forced by two findings that point in opposite directions (CLAUDE.md): a TONIC driver collapses
    WITH standardization (within-batch variance ~0 -> divides by ~EPS -> pt7 measured 0.098 =
    chance), while a MIXED-SCALE composite collapses WITHOUT it (pt7's all4 at std0 under SGD went
    to NaN). Per-column, by each driver's own rule, is the only arrangement satisfying both.

    Uses the NORM forms of the two novelty drivers so every column is 1-D and the gate is rank-5,
    which keeps it synapse-tractable — the vector forms would make K = 816.

    Consequence for the dead control: this builds 2 Signals heads + nerisez's MLP, so it consumes
    different construction RNG from any single driver and needs its OWN neuro_lr=0 control.
    """

    def __init__(self, lr):
        self.name = COMPOSITE
        self.kind = "composite"
        self.subs = [Driver(n, lr, standardize=STANDARDIZE_COMPOSITE[n]) for n in ALL5]
        self.K = sum(s.K for s in self.subs)

    def set_task(self, t):
        return

    def value(self, net, X, update=True):
        return torch.cat([s.value(net, X, update=update) for s in self.subs], dim=1)

    def train_head(self, net, X, Y):
        for s in self.subs:
            s.train_head(net, X, Y)

    def live_update(self, net, X):
        for s in self.subs:
            s.live_update(net, X)

    def state(self):
        return [s.state() for s in self.subs]

    def restore(self, st):
        for s, v in zip(self.subs, st):
            s.restore(v)


class ConstDriver:
    """m(x) = 1, constant. The CONTENT-FREE control — pt7's `5ht-const` in this study's harness.

    NOT a dead gate. `P` is still learned, still per-neuron / per-synapse, still trained by the same
    meta-loss, and the gate it produces is still applied — the ONLY thing removed is any dependence
    on the input, the loss, the task, or anything else. So f = exp(P) is a learned structured decay
    pattern with no neuromodulator signal in it at all.

    IT EXISTS TO SPLIT ONE RESULT IN TWO. If a mechanism cell beats its dead control, that gain is
    either (a) the driver telling the gate something, or (b) the gate having per-parameter freedom
    trained on replay, which any constant driver also has. The dead control cannot separate those —
    it has no learned gate at all. This one can, and CLAUDE.md's whole arc says to expect (b): pt7's
    content-free `free` and `5ht-const` arms repeatedly matched or beat the real bio drivers.

    A near-identical delta across four unrelated drivers is exactly the signature that predicts (b),
    which is why this control is run as soon as that pattern appears rather than after the writeup.
    """

    def __init__(self, *_, **__):
        self.name = "const"
        self.kind = "const"
        self.K = 1

    def set_task(self, t):
        return

    def value(self, net, X, update=True):
        return torch.ones(X.size(0), 1, device=DEV)

    def train_head(self, net, X, Y):
        return

    def live_update(self, net, X):
        return

    def state(self):
        return None

    def restore(self, st):
        return


def make_driver(name, lr):
    """The one constructor. Order of construction is RNG-relevant — see the dead-control note."""
    if name == "taskid":
        return TaskIdDriver()
    if name == "const":
        return ConstDriver()
    if name == COMPOSITE:
        return CompositeDriver(lr)
    return Driver(name, lr)


# ============================================================================== RNG discipline
@contextmanager
def rng_frozen():
    """Run a block without advancing any RNG stream.

    REQUIRED around mid-training evals, and NOT for an obvious reason: iterating a DataLoader
    consumes a draw from the global torch generator EVEN WITH shuffle=False, because
    `_BaseDataLoaderIter.__init__` draws a `_base_seed` whenever `loader.generator is None`. Adding
    a read-only eval pass to a training loop therefore moves the run off its reference trajectory —
    this cost `plast_drivers` an anchor failure (0.9019 -> 0.8955) and ~40 minutes to isolate.
    """
    st = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    try:
        yield
    finally:
        torch.set_rng_state(st[0]); np.random.set_state(st[1]); random.setstate(st[2])
