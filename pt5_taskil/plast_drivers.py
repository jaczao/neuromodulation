"""er-own PLASTICITY driven by pt7 NEUROMODULATOR signals (not the task id), task-IL, SGD, tuned.

User-requested. Every plasticity cell in this package so far was driven by the task-id one-hot, and
`gate_stats.py` / `meta_schedule.py` showed why they all tie their dead gate: the learned gate collapses
to a uniform global LR knob (cos(dev) 0.97-0.999, per-task mean alpha identical across tasks). A task
one-hot is the most task-informative driver available, so the natural next question is whether a
CONTENT driver — one that varies per SAMPLE rather than per task — changes that. pt7 answered it for
FORWARD gain targets (a controlled negative: difficulty/novelty is not task identity); this asks it for
the PLASTICITY target, in task-IL, at a tuned operating point, with the dead-gate discipline this
package uses and pt7's 1-seed studies did not.

MECHANISM = pt7's er-own plasticity: main net + gate trained jointly on the ER batch, no task ids
anywhere. Per batch: mbar = mean over the (current + replay) batch of the driver m(x); the gate is
alpha = exp(mbar @ P) (P zero-init => alpha = 1 parity, exp keeps alpha > 0 and can amplify OR suppress
the LR); P is trained by the pt5 LOOKAHEAD meta-loss (the gate multiplies gradients in place, so the
main loss gives P no gradient) — W_fast = W.detach() - lr*(alpha (x) g) with g detached, meta-CE on the
SAME ER batch (replay in the batch = a retention signal) trains ONLY P via Adam, then the real gated
SGD step commits with the detached alpha.

GRANULARITIES (the parameter axis, all three requested)
  global  : one scalar alpha scaling every learning rate equally     P: (K,)
  neuron  : per-neuron alpha, gates each layer's incoming weight+bias grads   P: (K, 810)
  synapse : per-synapse alpha per layer, WEIGHT grads only (biases stay plastic)  P: (K, n_syn)

  NOTE the driver is MEAN-POOLED over the batch (`mbar`), because a plasticity gate multiplies a
  gradient that is already a batch aggregate. So for THIS target "per-neuron"/"per-synapse" name the
  PARAMETER axis only — along the SAMPLE axis every granularity is equally constant within a batch.
  That is not a bug in the port, it is intrinsic to gating gradients, and it is the same structural
  point `driver_traces/signalnet_traces.md` made from the other direction (a gate with no per-sample
  variation is a global gain however rich its input).

!! SUPERSEDED FOR FUTURE RUNS (user-set convention, see CLAUDE.md): `ach`, `ach_ema` and `nerisez`
   below are HEAD PREDICTIONS (m(x) = heads(x), the head regressing the true value by MSE). The
   standing convention is now that these three use ACTUAL values instead — H from one extra
   unmodulated forward, ema/var of the ACTUAL H — because entropy is label-free, so the head buys
   only a saved forward while measurably distorting the signal, and a head regressing the TONIC
   `ach_ema` target degenerates to emitting a constant (measured per-sample sd 0.01 here). Every
   number already in `plast_drivers_results.tsv` was produced with the PREDICTED versions; they are
   not retro-fitted. Re-run under the new convention before comparing against future tables.

DRIVERS (the five requested), each with the standardization the project's own rule prescribes:
  ach      per-sample entropy H(x)             head-regressed, K=1    STANDARDIZED (per-sample)
  ach_ema  tonic ema(entropy)                  head-regressed, K=1    RAW (tonic!)
  nerisez  relu((Hpred - ema_H)/sqrt(var_H))   MLP predictor,  K=1    intrinsic z-score
  vec_x    x - ema(x), input novelty           head-free,      K=784  STANDARDIZED (per-sample)
  vecproj  (x - ema(x)) @ R, R random 784->32  head-free,      K=32   STANDARDIZED (per-sample)
  Standardization is NOT a free knob here: CLAUDE.md's rule is "standardize per-sample drivers, NEVER a
  tonic one" — a tonic driver's within-batch variance is ~0, so standardizing divides by ~0 and the
  exp() gate blows up (pt7 measured 0.098 = chance). `ach` and `ach_ema` are the same signal on
  opposite sides of that rule, which makes them a built-in check on it.

"NO FREEZES AT INFERENCE" (as requested) — and what it can and cannot do here. The driver's running
stats / EMAs / predictor state keep updating during the test pass (`driver_traces/live_traces.py`
protocol) instead of being frozen. STATED PLAINLY: for a PLASTICITY target this cannot change accuracy
by construction — the gate multiplies gradients, never the forward, so eval is the plain unmodulated
net and no driver value reaches a prediction. It is implemented and reported anyway because it does
change the DIAGNOSTICS (|alpha-1|, probe), and the frozen-vs-live `pred` equality is printed as a
check rather than asserted. The A-matrix evals after each task never call the driver at all, so live
stats cannot leak test statistics into later training.

DEAD-GATE CONTROL (rule #10), per DRIVER: the same config with neuro_lr = 0, so P stays at zero,
alpha == 1 exactly, and the run is plain ER-SGD plus whatever RNG the modulator's construction and the
driver's own forward consumed. It is per-driver rather than per-granularity because `Heads(K)` init
consumes RNG proportional to K while `PlastGate` is all zeros (no RNG) — a claim the run checks
empirically instead of assuming (`--part deadcheck`).

OPERATING POINT: task-IL, SGD, main lr 0.1 / ep 5 — `plast_taskil.py`'s val-tuned ER-SGD point (its
grid was extended upward to {3e-1, 1.0} to confirm 1e-1 is a genuine interior max, not truncation).
neuro_lr is FIXED at 1e-3 for every cell (pt7_tuned_neuro's DEFAULT_NEURO_LR, the same value
`results/pt7_plast_tempslope.py` used for its tuned variant): identical budget for every arm (rule #3),
and four consecutive neuro_lr sweeps in this package have come back unresolved within the noise floor.
Stated as a limit, not hidden.

ANCHOR: `--part anchor` reproduces three cells of the FROZEN `results/pt7_plast_tempslope_results.tsv`
(ach_ema x {neuron, synapse, global}, class-IL, sgd-tuned) through this copy-forward. `results/` is
frozen (rule #9) so its primitives are imported READ-ONLY and the loop is copy-forwarded, exactly as
`driver_traces/` does.

Ledger pt5_taskil/plast_drivers_results.tsv; `--resume` skips done rows; `--part` chunks the run.

Run: uv run python pt5_taskil/plast_drivers.py --part all --resume  (redirect to plast_drivers.log)
"""
import argparse
import copy
import io
import math
import random
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

import pt7_neuromodulators as p7                                   # noqa: E402  (frozen, read-only)
import pt7_plast_tempslope as pts                                  # noqa: E402  (frozen, read-only)
from pt7_stateful import StatefulDriver                            # noqa: E402
from pt7_variants import NEDriver                                  # noqa: E402
from prototype.data import SplitMNIST                              # noqa: E402

DEV, EPS = p7.DEV, p7.EPS
TSV = Path(__file__).resolve().parent / "plast_drivers_results.tsv"
COLS = ["stage", "driver", "gran", "nlr", "seed", "acc", "forget", "probe", "a_h0", "a_h1", "a_out",
        "acc_live"]

DRIVERS = ("ach", "ach_ema", "nerisez", "vec_x", "vecproj")
GRANS = ("global", "neuron", "synapse")
# 1-D NORM forms of the two headless novelty drivers (user-requested): m(x) = ||x - ema(x)|| and
# ||(x - ema(x)) @ R||, i.e. "how novel is this sample" as ONE number instead of a 784-/32-dim vector.
# Three things change, and they are the reasons to run it:
#   (a) K = 1 makes vec_x per-SYNAPSE runnable — P drops from 3.7e8 params (~500x the backbone, the
#       reason that cell was skipped) to (1, 477600) ~ 477k, so the granularity axis completes.
#   (b) it removes the constant-DIMENSION standardization blow-up: the norm over 784 dims has real
#       variance even though 212 individual dims do not, so nothing divides by ~EPS.
#   (c) it is the input-space analogue of pt7's `emb_all` (= the norm form of h1 novelty), so the
#       vector-vs-norm contrast is exactly pt7's emb_all-vs-vec_h1 contrast on a plasticity target.
NORM_DRIVERS = ("vec_x_norm", "vecproj_norm")
# all5: the rank-5 composite of the five single drivers, i.e. this package's analogue of pt7's all4.
# Uses the NORM forms of the two novelty drivers so every column is 1-D and the gate is rank-5
# (P: (5, D)), which keeps it synapse-tractable — the vector forms would make K = 816.
ALL5 = ("ach", "ach_ema", "nerisez", "vec_x_norm", "vecproj_norm")
COMPOSITE = "all5"
# vec_x is K=784, so a per-SYNAPSE P is (784, 313600)+(784,160000)+(784,4000) = 3.7e8 params ~ 1.5 GB
# (plus 2x for Adam moments) and ~500x the 478k backbone. Skipped deliberately: CLAUDE.md's rule from
# pt7_capacity is that a modulator comparable to or larger than its backbone makes any result a capacity
# confound, so the cell would be uninterpretable even if it fit in memory. `vecproj` IS the
# synapse-tractable form of exactly this driver (that is why pt7 introduced it), and it is run.
SKIP = {("vec_x", "synapse")}
# per-driver standardization, per CLAUDE.md's tonic-vs-per-sample rule (see docstring)
STANDARDIZE = {"ach": True, "ach_ema": False, "nerisez": False, "vec_x": True, "vecproj": True,
               "vec_x_norm": True, "vecproj_norm": True}


class NormNovelty:
    """L2 norm of a headless novelty vector -> a 1-D per-sample driver, standardized as a scalar.

    ORDER MATTERS AND IS THE OPPOSITE OF THE OBVIOUS ONE: the norm is taken on the RAW diff and the
    resulting SCALAR is standardized. Standardizing per-dimension first and then taking the norm
    concentrates the value at sqrt(K) (CLAUDE.md's pt7_driver_traces methodology note) — it would
    manufacture a near-constant driver rather than measure novelty. Doing it in this order also
    sidesteps the constant-dimension blow-up entirely: 212 of vec_x's 784 dims have zero variance,
    but ||x - ema(x)|| does not.
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
        if not self.standardize or not self.inited:
            return v
        return (v - self.rm) / (self.rv.sqrt() + EPS)

    def state(self):
        """Own snapshot: the wrapper holds scalar stats AND the inner vector driver's running mean,
        so both have to travel together for the frozen-vs-live eval to be side-effect free."""
        d = self.inner
        return copy.deepcopy((self.rm, self.rv, self.inited,
                              d.mh1, d.mx, d.ch1, d.cx, d.run_mean, d.run_var, d.inited))

    def restore(self, st):
        d = self.inner
        (self.rm, self.rv, self.inited,
         d.mh1, d.mx, d.ch1, d.cx, d.run_mean, d.run_var, d.inited) = copy.deepcopy(st)

MAIN_LR = 0.1          # plast_taskil val-tuned task-IL ER-SGD (interior max; grid extended to verify)
EPOCHS = 5
BUFFER = 1000
NEURO_LR = 1e-3        # pt7_tuned_neuro DEFAULT_NEURO_LR; fixed for every cell (rule #3)
DEAD_NLR = 0.0
SEEDS = (42, 43, 44)
NOISE_FLOOR = 0.007

# frozen pt7_plast_tempslope cells this copy-forward must reproduce (class-IL, sgd-tuned lr .03/nlr 1e-3)
ANCHORS = {"neuron": 0.9017, "synapse": 0.9010, "global": 0.9019}


# ============================================================ driver provider (copy-forward + extended)
class ActualEntropy:
    """The entropy family from ACTUAL values (the standing convention), not head predictions.

    ONE extra unmodulated forward per call yields H, and all three columns are derived from it — so
    a composite pays for one forward, not three. No head, nothing to train, and still ORACLE-FREE:
    entropy needs no labels.

      ach      = H                                   (standardized by running stats)
      ach_ema  = broadcast ema(H)                    (tonic => RAW, per the standardization rule)
      nerisez  = relu((H - ema_H)/sqrt(var_H + eps)) (now self-consistent: actual H against actual
                                                      statistics, where the predicted form scored a
                                                      smoothed Hhat against actual stats)

    LAG-1 IS PRESERVED EXACTLY as in the head versions: ach_ema and nerisez read ema_H/var_H BEFORE
    this batch is folded in (Signals appended emaH before its update; nerisez's stats advanced later,
    in train_head). `ach`'s standardization mirrors Signals the other way — it updates the running
    mean/var first and then standardizes — so each column keeps the timing it already had and only
    the SOURCE of H changes.
    """

    def __init__(self, cols, standardize_ach=True):
        self.cols = tuple(cols)
        self.standardize_ach = standardize_ach
        self.emaH = None; self.varH = None
        self.rm = None; self.rv = None; self.inited = False

    def K(self):
        return len(self.cols)

    @torch.no_grad()
    def value(self, net, x, update=True):
        H = p7.entropy(net.plain(x)[0]).unsqueeze(1)              # (B,1) ACTUAL, one extra forward
        if self.emaH is None:
            self.emaH = H.mean().item(); self.varH = float(H.var(unbiased=False))
        out = []
        for c in self.cols:
            if c == "ach_ema":
                out.append(torch.full_like(H, self.emaH))          # lag-1 tonic scalar
            elif c == "nerisez":
                out.append(F.relu((H - self.emaH) / math.sqrt(self.varH + EPS)))
            else:                                                  # ach
                if self.standardize_ach:
                    if update:
                        bm, bv = H.mean(0), H.var(0, unbiased=False)
                        if not self.inited:
                            self.rm, self.rv, self.inited = bm.clone(), bv.clone(), True
                        else:
                            self.rm = 0.99 * self.rm + 0.01 * bm
                            self.rv = 0.99 * self.rv + 0.01 * bv
                    out.append((H - self.rm) / (self.rv.sqrt() + EPS) if self.inited else H)
                else:
                    out.append(H)
        if update:                                                 # fold in AFTER reading (lag-1)
            self.varH = (1 - p7.BS) * self.varH + p7.BS * float(((H - self.emaH) ** 2).mean())
            self.emaH = (1 - p7.BS) * self.emaH + p7.BS * H.mean().item()
        return torch.cat(out, dim=1)

    def train_head(self, net, X, Y):
        return                                                     # no head exists

    def live_update(self, net, X):
        return                                                     # value(update=True) does it all

    def state(self):
        return copy.deepcopy((self.emaH, self.varH, self.rm, self.rv, self.inited))

    def restore(self, st):
        self.emaH, self.varH, self.rm, self.rv, self.inited = copy.deepcopy(st)


ENTROPY_FAMILY = ("ach", "ach_ema", "nerisez")


class CompositeDriver:
    """all5: the five single drivers concatenated into one (B,5) code driving a rank-5 gate.

    EACH COLUMN KEEPS ITS OWN STANDARDIZATION RULE rather than standardizing the stacked vector,
    and that is forced by two findings that point in opposite directions (CLAUDE.md): a TONIC driver
    collapses WITH standardization (its within-batch variance is ~0, so it divides by ~EPS — pt7
    measured 0.098 = chance), while a MIXED-SCALE composite collapses WITHOUT it (pt7's all4 at
    std0 under SGD went to NaN). Standardizing per column, by each driver's own rule, is the only
    arrangement that satisfies both: `ach_ema` stays raw, the rest arrive at O(1), and no column can
    swamp the others in the linear gate.

    Consequence for the dead control: this builds 2 Signals heads + nerisez's MLP, so it consumes
    different construction RNG from any single driver and needs its OWN neuro_lr=0 control.
    """

    def __init__(self, lr, std=None, actual=False):
        self.kind = "composite"
        if actual:
            # ONE ActualEntropy covers ach/ach_ema/nerisez, so the composite pays for a single extra
            # forward per batch rather than three. Column ORDER is preserved (ALL5) so this table
            # stays comparable to the head-based one.
            ach_std = STANDARDIZE["ach"] if std is None else std
            self.subs = [ActualEntropy(ENTROPY_FAMILY, standardize_ach=ach_std)]
            self.subs += [Driver(n, lr, std=std) for n in ALL5 if n not in ENTROPY_FAMILY]
        else:
            self.subs = [Driver(n, lr, std=std) for n in ALL5]
        self.K = sum(s.K() if callable(getattr(s, "K", None)) else s.K for s in self.subs)

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


class Driver:
    """m(x) provider + its own optimizer. Oracle-free: no driver here sees a task id.

    Copy-forward of `pt7_plast_tempslope.Driver`, extended with `ach` (per-sample entropy, absent
    there) and `vec_x`, and with standardization set PER DRIVER instead of one flag for the run.
    """
    HEAD_KEY = {"ach": "ACh", "ach_ema": "ACh_ema"}

    def __init__(self, name, lr, std=None, actual=False):
        self.name = name
        # std=None -> the project's per-driver rule; std=False -> the un-standardised ablation.
        # NOTE for `nerisez` this flag is inert: StatefulStd only consults it for mech=="ach", and
        # nerisez's z-score IS the driver, so "un-standardised nerisez" is not expressible.
        std = STANDARDIZE[name] if std is None else std
        if actual and name in ENTROPY_FAMILY:                      # standing convention: no heads
            self.kind = "actual"; self.K = 1
            self.drv = ActualEntropy((name,), standardize_ach=std)
        elif name in self.HEAD_KEY:                                # Signals head: 784->32->1 regresses tau
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
            self.drv = NormNovelty(name.replace("_norm", ""), std)
            self.K = self.drv.K()
        elif name == "nerisez" and not actual:                     # stateful entropy z-score (MLP)
            self.kind = "stateful"; self.K = 1
            self.drv = pts.StatefulStd("nerisez", gru=False, standardize=std).to(DEV)
            self.opt = torch.optim.Adam(self.drv.parameters(), lr)
        else:
            raise ValueError(name)

    def value(self, net, X, update=True):
        """(B,K) driver, DETACHED — the plasticity path never grads back into the driver."""
        if self.kind == "actual":
            return self.drv.value(net, X, update=update).detach()
        if self.kind == "head":
            return self.heads(X).detach()
        if self.kind == "ne":
            return self.drv.value(net, X, update=update).detach()
        return self.drv.driver(X, update_state=update, update_stats=False).detach()

    def train_head(self, net, X, Y):
        if self.kind in ("ne", "actual"):
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
        """The 'no freezes at inference' update for state that the gate path does NOT already advance.

        Only `nerisez` needs this: its z-score divides by the ACTUAL-entropy stats (emaH/varH), which
        during training are advanced by `train_head`, not by reading the driver. Entropy needs no
        labels, so it is computable at test from one unmodulated forward — the `live_traces.py`
        protocol, which is also where a live emaH was shown to make a rectified surprise driver
        collapse to ~0. NEDriver (vec_x/vecproj) advances its own mx/run_* inside `value(update=True)`.
        HEAD drivers have NOTHING to advance: the gate reads m(x) = heads(x), a pure function of the
        image, and the Signals EMAs only ever built the head's training targets — so for `ach` and
        `ach_ema` frozen and live are identical BY CONSTRUCTION, not by measurement.
        """
        if self.kind != "stateful":
            return                                                 # 'actual' updates inside value()
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


# ================================================================================ task-IL helpers
@contextmanager
def rng_frozen():
    """Run a block without advancing any RNG stream.

    REQUIRED around the mid-training evals, and NOT for an obvious reason: iterating a DataLoader
    consumes a draw from the global torch generator EVEN WITH shuffle=False, because
    `_BaseDataLoaderIter` draws a `_base_seed` per iterator whenever `loader.generator is None`. The
    A-matrix costs 15 extra iterators over a run, which shifted every later reservoir draw and train
    shuffle and moved the net off its reference trajectory — measured as a 0.9019 -> 0.8955 anchor
    failure against the frozen ledger. Same lesson as `driver_traces/live_traces.py`'s rng_frozen()
    around module construction; the surprise here is that a read-only, no-shuffle eval pass is not
    RNG-free either.
    """
    ts, ns, ps = torch.get_rng_state(), np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(ts); np.random.set_state(ns); random.setstate(ps)


def _masked_logits(logits, allowed):
    add = torch.full_like(logits, float("-inf"))
    add[:, list(allowed)] = 0.0
    return logits + add


@torch.no_grad()
def _acc(net, loader, allowed=None):
    """Plain (unmodulated) accuracy; task-IL masks to the task's 2 classes. The plasticity gate never
    enters the forward, so this is the whole of `pred` — no driver value is consulted."""
    net.eval(); c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        logits = net.plain(x)[0]
        if allowed is not None:
            logits = _masked_logits(logits, allowed)
        c += (logits.argmax(1) == y).sum().item(); tot += len(y)
    return c / tot


# ==================================================================================== training loop
def run(driver_name, gran, seed, neuro_lr=NEURO_LR, main_lr=MAIN_LR, epochs=EPOCHS,
        buffer=BUFFER, taskil=True, std=None, actual=False):
    """er-own plasticity with a pt7 driver. Copy-forward of pt7_plast_tempslope.run_plast, extended
    with task-IL masking, an A-matrix (so forgetting is reported), and frozen-vs-live eval."""
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV)
    # driver "er" = the plain ER-SGD baseline IN THIS HARNESS: no driver, no gate, nothing constructed.
    # Not a substitute for the dead control (it is not RNG-matched — that is the whole point of rule
    # #10) but it ties this harness's numbers to plast_taskil's ER 0.9946 from the prototype harness.
    plain = driver_name == "er"
    drv = (None if plain else
           CompositeDriver(neuro_lr, std=std, actual=actual) if driver_name == COMPOSITE
           else Driver(driver_name, lr=neuro_lr, std=std, actual=actual))
    gate = None if plain else pts.PlastGate(gran, drv.K, neuro_lr)
    buf = p7.Reservoir(buffer)
    loss_fn = p7.masked_ce if taskil else (lambda lo, yy: p7.CE(lo, yy))
    A = np.full((5, 5), np.nan)

    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                params = pts._net_params(net)
                if plain:                                          # ungated ER-SGD step
                    g = torch.autograd.grad(loss_fn(net.plain(Xm)[0], Ym), params)
                    with torch.no_grad():
                        for i in range(pts.NPARAMS):
                            params[i].add_(g[i], alpha=-main_lr)
                    buf.add(x, y)
                    continue
                mbar = drv.value(net, Xm).mean(0)                  # (K,) detached, batch-pooled
                g = torch.autograd.grad(loss_fn(net.plain(Xm)[0], Ym), params)
                mult, _ = gate.mult(mbar)                          # alpha differentiable in P
                Wf = [params[i].detach() - main_lr * (mult[i] * g[i]) for i in range(pts.NPARAMS)]
                meta = loss_fn(pts._fwd_fast(Wf, Xm), Ym)          # retention meta-loss trains ONLY P
                gate.opt.zero_grad(); meta.backward(); gate.opt.step()
                with torch.no_grad():                              # real gated step, detached alpha
                    for i in range(pts.NPARAMS):
                        params[i].add_(mult[i].detach() * g[i], alpha=-main_lr)
                buf.add(x, y)
                drv.train_head(net, Xm, Ym)
        with rng_frozen():                    # else the eval's DataLoader iterators shift training
            for i in range(t + 1):            # A-matrix: NO driver call => live stats cannot leak
                A[t, i] = _acc(net, loaders[i][1], allowed=p7.SEQ[i] if taskil else None)

    acc = float(np.nanmean(A[4, :]))
    forget = float(np.mean([max([A[k, i] for k in range(i, 5)]) - A[4, i] for i in range(5)]))
    if plain:
        z = dict(pred=acc, probe=float("nan"), per_layer={k: 0.0 for k in ("h0", "h1", "out")})
        return dict(acc=acc, forget=forget, A=A, frozen=z, live=z)
    # Diagnostics: FROZEN first (leaves the driver state untouched), then LIVE from the same snapshot.
    st = drv.state()
    diag_frozen = _diag(net, drv, gate, loaders, update=False, taskil=taskil)
    drv.restore(st)
    diag_live = _diag(net, drv, gate, loaders, update=True, taskil=taskil)
    drv.restore(st)
    return dict(acc=acc, forget=forget, A=A, frozen=diag_frozen, live=diag_live)


@torch.no_grad()
def _diag(net, drv, gate, loaders, update, taskil):
    """|alpha-1| per layer + the task-decodability probe, over the test set. `update` = the requested
    'no freezes at inference' (live running stats / predictor state) vs the frozen protocol. `pred` is
    recomputed here purely to CHECK that it does not depend on this — the gate is not in the forward."""
    net.eval()
    mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}; tot = 0; Ms, Ts = [], []
    for i in range(5):
        for x, y in loaders[i][1]:
            x = x.to(DEV); b = x.size(0)
            if update:
                drv.live_update(net, x)                    # advance state the gate path does not
            m = drv.value(net, x, update=update)
            _, structs = gate.mult(m.mean(0))
            if gate.mech == "global":
                dev = (structs[0] - 1).abs().item()
                for k in mags:
                    mags[k] += dev * b
            else:
                a0, a1, a2 = structs
                mags["h0"] += (a0 - 1).abs().mean().item() * b
                mags["h1"] += (a1 - 1).abs().mean().item() * b
                mags["out"] += (a2 - 1).abs().mean().item() * b
            Ms.append(m.cpu()); Ts.append(torch.full((b,), i)); tot += b
    pred = float(np.mean([_acc(net, loaders[i][1], allowed=p7.SEQ[i] if taskil else None)
                          for i in range(5)]))
    M = torch.cat(Ms)
    return dict(pred=pred, probe=p7._probe(M, torch.cat(Ts), drv.K),
                per_layer={k: mags[k] / tot for k in mags},
                # per-COLUMN scale, for a composite: a linear gate sums m_k P_k, so a column entering
                # 80x larger than its neighbours dominates until P shrinks its weight. Reported
                # because pt7's UNIFY-12 found that piling columns on DILUTES a composite gate.
                col_mean=M.mean(0).tolist(), col_sd=M.std(0).tolist())


# ========================================================================================== ledger
def load_ledger():
    if not TSV.exists():
        return {}
    rows = {}
    for line in TSV.read_text().splitlines()[1:]:
        if line.strip():
            f = line.split("\t")
            rows[tuple(f[:5])] = tuple(float(v) for v in f[5:])
    return rows


def append(key, vals):
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with TSV.open("a") as fh:
        fh.write("\t".join(list(key) + [f"{v:.6f}" for v in vals]) + "\n")


def key_of(stage, driver, gran, nlr, seed):
    return (stage, driver, gran, f"{nlr:g}", str(seed))


def run_cell(stage, driver, gran, nlr, seed, ledger, taskil=True, main_lr=MAIN_LR, std=None,
             actual=False):
    key = key_of(stage, driver, gran, nlr, seed)
    if key in ledger:
        print(f"[skip] {'|'.join(key)} acc={ledger[key][0]:.4f}", flush=True)
        return ledger[key]
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = run(driver, gran, seed, neuro_lr=nlr, taskil=taskil, main_lr=main_lr, std=std,
                actual=actual)
    pl = r["frozen"]["per_layer"]
    vals = (r["acc"], r["forget"], r["frozen"]["probe"], pl["h0"], pl["h1"], pl["out"],
            r["live"]["pred"])
    append(key, vals)
    ledger[key] = vals
    print(f"[run ] {'|'.join(key)} acc={r['acc']:.4f} forget={r['forget']:.4f} "
          f"probe={r['frozen']['probe']:.3f} |a-1|={pl['h0']:.4f}/{pl['h1']:.4f}/{pl['out']:.4f} "
          f"live_pred={r['live']['pred']:.4f} (frozen {r['frozen']['pred']:.4f})", flush=True)
    if driver == COMPOSITE:
        cm, cs = r["frozen"]["col_mean"], r["frozen"]["col_sd"]
        print("        per-column m at eval: " + "  ".join(
            f"{n}={a:+.2f}+/-{b:.2f}" for n, a, b in zip(ALL5, cm, cs)), flush=True)
    return vals


# ========================================================================================== stages
def anchor(ledger):
    """Reproduce frozen pt7_plast_tempslope cells (class-IL, sgd-tuned) through this copy-forward."""
    print("\n" + "=" * 96)
    print("ANCHOR — copy-forward vs frozen results/pt7_plast_tempslope_results.tsv "
          "(ach_ema, class-IL, sgd-tuned)")
    print("=" * 96)
    ok = True
    for gran in GRANS:
        vals = run_cell("anchor", "ach_ema", gran, NEURO_LR, 42, ledger, taskil=False, main_lr=0.03)
        ref = ANCHORS[gran]
        match = abs(vals[0] - ref) < 1e-4          # frozen ledger stores 4 dp
        ok &= match
        print(f"  ach_ema {gran:8s}: {vals[0]:.4f} vs {ref:.4f}  "
              f"{'[MATCHES]' if match else '!! MISMATCH'}", flush=True)
    print(f"\n  copy-forward parity: {'CONFIRMED' if ok else 'FAILED — do not trust the rest'}")
    return ok


def deadcheck(ledger):
    """Is the dead control really granularity-independent? P is all zeros (no RNG) while Heads(K) init
    consumes RNG with K, so the claim is 'dead depends on the DRIVER, not the gran' — checked, not
    assumed (the same reasoning that made multi5's dead control byte-identical to base's)."""
    print("\n" + "=" * 96)
    print("DEAD-CONTROL GRANULARITY CHECK (expect byte-identical across gran at fixed driver)")
    print("=" * 96)
    for driver in ("ach", "vecproj"):
        accs = [run_cell("deadchk", driver, g, DEAD_NLR, 42, ledger)[0] for g in GRANS]
        same = max(accs) - min(accs) < 1e-9
        print(f"  {driver:8s} " + "  ".join(f"{g}={a:.6f}" for g, a in zip(GRANS, accs))
              + f"   {'[IDENTICAL]' if same else '!! DIFFER'}", flush=True)


def baseline(ledger):
    """Plain ER-SGD task-IL at the same point, and the per-driver dead gates (RNG-matched controls)."""
    for s in SEEDS:
        run_cell("base", "er", "-", 0.0, s, ledger)
    for driver in DRIVERS:
        for s in SEEDS:
            run_cell("dead", driver, "global", DEAD_NLR, s, ledger)


def test(ledger):
    for driver in DRIVERS:
        for gran in GRANS:
            if (driver, gran) in SKIP:
                continue
            for s in SEEDS:
                run_cell("test", driver, gran, NEURO_LR, s, ledger)


def norm(ledger):
    """The 1-D norm forms, 1 seed, NO tuning (neuro_lr stays 1e-3, as for every other cell).

    All three granularities run — with K=1 the vec_x synapse cell that was skipped as a capacity
    confound (3.7e8 params) is now a 477k-param gate, so the axis completes.

    The DEAD control is reused rather than re-run: these drivers are HEAD-FREE, so `Driver.__init__`
    builds no `Heads(K)` and NEDriver's projection uses its own generator — no global RNG is consumed
    at construction, and `PlastGate`'s P is all zeros whatever K is. So the neuro_lr=0 control is
    independent of K and identical to the existing vec_x/vecproj dead rows (which the report checks
    against plain ER rather than asserting).
    """
    print("\n" + "=" * 96)
    print("1-D NORM forms of the headless novelty drivers — 1 seed, no tuning")
    print("=" * 96)
    for driver in NORM_DRIVERS:
        for gran in GRANS:
            run_cell("norm", driver, gran, NEURO_LR, 42, ledger)


def all5(ledger):
    """The rank-5 composite + its OWN dead control (it builds 3 nets, so it is not RNG-matched to
    the head-free control the norm cells reuse). 1 seed, no tuning, all three granularities."""
    print("\n" + "=" * 96)
    print(f"ALL5 composite {ALL5} — 1 seed, no tuning")
    print("=" * 96)
    run_cell("dead", COMPOSITE, "global", DEAD_NLR, 42, ledger)
    for gran in GRANS:
        run_cell("norm", COMPOSITE, gran, NEURO_LR, 42, ledger)


def actual_stage(ledger):
    """Re-run every PREDICTION-based driver under the standing actual-value convention.

    Seeds are matched to what each cell had before, so the comparison is like-for-like: the three
    entropy drivers were 3 seeds, all5 was 1 (comparing a 1-seed rerun against a 3-seed mean is the
    trap that already bit me once here).

    DEAD CONTROL: with actual values nothing is constructed — no Heads, no predictor MLP — so these
    drivers should now share the head-free control (= plain ER). Checked with one run rather than
    assumed, since that is exactly the RNG-matching claim rule #10 exists for.
    """
    print("\n" + "=" * 96)
    print("ACTUAL-VALUE rerun of the prediction-based drivers (ach, ach_ema, nerisez, all5)")
    print("=" * 96)
    a = run_cell("actual", "ach", "global", DEAD_NLR, 42, ledger, actual=True)
    hf = ledger.get(key_of("dead", "vec_x", "global", DEAD_NLR, 42))
    # 1e-6, NOT 1e-9: `hf` was round-tripped through the ledger's "%.6f" while `a` may be full
    # precision, so a tighter tolerance flags identical runs as mismatched. Third time this has
    # bitten me — match the tolerance to how the value was PRODUCED (CLAUDE.md).
    print(f"  dead(actual) {a[0]:.6f} vs head-free dead {hf[0]:.6f}  "
          f"{'[IDENTICAL — no construction RNG]' if abs(a[0]-hf[0]) < 1e-6 else '!! DIFFERS'}",
          flush=True)
    for driver in ENTROPY_FAMILY:
        for gran in GRANS:
            for seed in SEEDS:
                run_cell("actual", driver, gran, NEURO_LR, seed, ledger, actual=True)
    for gran in GRANS:
        run_cell("actual", COMPOSITE, gran, NEURO_LR, 42, ledger, actual=True)


def actual_report(ledger):
    hf = ledger.get(key_of("dead", "vec_x", "global", DEAD_NLR, 42))
    print("\n" + "=" * 116)
    print("PREDICTED vs ACTUAL entropy-family drivers — er-own plasticity, task-IL, neuro_lr 1e-3")
    print("=" * 116)
    print(f"{'driver':10s} {'gran':8s} {'seeds':>5s} {'predicted':>17s} {'actual':>17s} "
          f"{'d(actual)':>10s} {'d-dead':>8s} {'probe':>6s}")
    for driver in ENTROPY_FAMILY + (COMPOSITE,):
        seeds = SEEDS if driver in ENTROPY_FAMILY else (42,)
        old_stage = "test" if driver in ENTROPY_FAMILY else "norm"
        old_dead = ledger.get(key_of("dead", driver, "global", DEAD_NLR, 42))
        for gran in GRANS:
            o = [ledger.get(key_of(old_stage, driver, gran, NEURO_LR, s)) for s in seeds]
            n = [ledger.get(key_of("actual", driver, gran, NEURO_LR, s)) for s in seeds]
            if any(v is None for v in n):
                continue
            om = np.mean([v[0] for v in o]) if all(v is not None for v in o) else float("nan")
            nm = np.mean([v[0] for v in n]); nsd = np.std([v[0] for v in n])
            osd = np.std([v[0] for v in o]) if all(v is not None for v in o) else float("nan")
            dd = nm - hf[0] if hf else float("nan")
            print(f"{driver:10s} {gran:8s} {len(seeds):>5d} {om:>10.4f}+/-{osd:.4f} "
                  f"{nm:>10.4f}+/-{nsd:.4f} {nm - om:>+10.4f} {dd:>+8.4f} "
                  f"{np.mean([v[2] for v in n]):>6.3f}")
        print()
    print("  d(actual) = actual - predicted, SEED-MATCHED. d-dead vs the head-free control")
    print(f"  ({hf[0]:.4f}), which the actual drivers now share since they construct nothing.")
    print("  NOTE the predicted arms' own dead controls differed (heads consumed RNG); with actual")
    print("  values that difference disappears, which is itself a simplification of the design.")


NOSTD_DRIVERS = ("ach", "vec_x", "vecproj", "vec_x_norm", "vecproj_norm", COMPOSITE)


def nostd(ledger):
    """The un-standardised ablation (user-requested), 1 seed, no tuning.

    Removes the per-driver standardization from every driver that has one. `ach_ema` is untouched
    (already raw by rule) and `nerisez` is inert to the flag (its z-score IS the driver), so within
    all5 those two columns are unchanged and only the other three go raw.

    DEAD CONTROLS ARE REUSED, not re-run: with neuro_lr = 0 the gate is alpha == 1, so no driver
    value can reach the main net whatever its scale, and construction RNG does not depend on the
    standardize flag. The existing per-arm dead rows therefore still apply exactly.

    EXPECT COLLAPSES, not just nulls: the main lr is 0.1 and the gate is exp(mbar @ P), so an
    un-standardised driver with a large mean can blow the gate up — pt7 measured exactly this for
    its un-standardised all4 (NaN -> 0.098 = chance under SGD). A collapse here is a RESULT (it
    re-derives the standardization rule at a new operating point), not a failed run.
    """
    print("\n" + "=" * 96)
    print("UN-STANDARDISED ablation — 1 seed, no tuning, dead controls reused")
    print("=" * 96)
    for driver in NOSTD_DRIVERS:
        for gran in GRANS:
            if (driver, gran) in SKIP:
                print(f"{driver} {gran}: skipped (K=784 synapse gate, 3.7e8 params)", flush=True)
                continue
            run_cell("nostd", driver, gran, NEURO_LR, 42, ledger, std=False)


def nostd_report(ledger):
    dead = {"ach": ledger.get(key_of("dead", "ach", "global", DEAD_NLR, 42)),
            COMPOSITE: ledger.get(key_of("dead", COMPOSITE, "global", DEAD_NLR, 42))}
    hf = ledger.get(key_of("dead", "vec_x", "global", DEAD_NLR, 42))
    print("\n" + "=" * 112)
    print("UN-STANDARDISED vs STANDARDISED — er-own plasticity, task-IL, seed 42, neuro_lr 1e-3")
    print("=" * 112)
    print(f"{'driver':14s} {'gran':8s} {'std acc':>9s} {'NOstd acc':>10s} {'d(no-std)':>10s} "
          f"{'d-dead':>8s} {'probe':>6s} {'|a-1| h0/h1/out':>24s}")
    for driver in NOSTD_DRIVERS:
        d = dead.get(driver, hf)
        for gran in GRANS:
            if (driver, gran) in SKIP:
                continue
            stage_std = "norm" if (driver in NORM_DRIVERS or driver == COMPOSITE) else "test"
            a = ledger.get(key_of(stage_std, driver, gran, NEURO_LR, 42))
            b = ledger.get(key_of("nostd", driver, gran, NEURO_LR, 42))
            if b is None:
                continue
            sa = f"{a[0]:.4f}" if a else "-"
            dd = f"{b[0] - d[0]:+.4f}" if d else "-"
            ds = f"{b[0] - a[0]:+.4f}" if a else "-"
            print(f"{driver:14s} {gran:8s} {sa:>9s} {b[0]:>10.4f} {ds:>10s} {dd:>8s} "
                  f"{b[2]:>6.3f} {b[3]:>7.4f}/{b[4]:.4f}/{b[5]:.4f}")
        print()
    print("  0.098 = chance (1/10) under the un-masked read; ~0.50 = the 2-way task-IL chance floor.")
    print("  d-dead vs each arm's own neuro_lr=0 control (reused; alpha==1 makes it std-independent).")


def norm_report(ledger):
    def one(stage, driver, gran, nlr, seed=42):
        return ledger.get(key_of(stage, driver, gran, nlr, seed))   # key_of does the :g formatting
    print("\n" + "=" * 112)
    print("NORM (1-D) vs VECTOR (multi-D) novelty drivers — er-own plasticity, task-IL, seed 42")
    print("=" * 112)
    er = one("base", "er", "-", 0.0)
    print(f"{'driver':14s} {'K':>5s} {'gran':8s} {'acc':>9s} {'d-dead':>8s} {'forget':>8s} "
          f"{'probe':>6s} {'|a-1| h0/h1/out':>26s}")
    if er:
        print(f"{'ER (plain)':14s} {'-':>5s} {'-':8s} {er[0]:>9.4f}")
    for norm_d, base_d, kv in (("vec_x_norm", "vec_x", (1, 784)),
                               ("vecproj_norm", "vecproj", (1, 32))):
        dead = one("dead", base_d, "global", DEAD_NLR)
        for label, drv, K, stage in ((norm_d, norm_d, kv[0], "norm"),
                                     (base_d, base_d, kv[1], "test")):
            for gran in GRANS:
                r = one(stage, drv, gran, NEURO_LR)
                if r is None:
                    print(f"{label:14s} {K:>5d} {gran:8s} {'— not run (K=784 synapse gate)':>9s}")
                    continue
                dd = r[0] - dead[0] if dead else float("nan")
                print(f"{label:14s} {K:>5d} {gran:8s} {r[0]:>9.4f} {dd:>+8.4f} {r[1]:>8.4f} "
                      f"{r[2]:>6.3f} {r[3]:>8.4f}/{r[4]:.4f}/{r[5]:.4f}")
        print()
    dead5 = one("dead", COMPOSITE, "global", DEAD_NLR)
    for gran in GRANS:
        r = one("norm", COMPOSITE, gran, NEURO_LR)
        if r is None:
            continue
        dd = r[0] - dead5[0] if dead5 else float("nan")
        print(f"{COMPOSITE:14s} {5:>5d} {gran:8s} {r[0]:>9.4f} {dd:>+8.4f} {r[1]:>8.4f} "
              f"{r[2]:>6.3f} {r[3]:>8.4f}/{r[4]:.4f}/{r[5]:.4f}")
    if dead5:
        print(f"{'all5 DEAD':14s} {5:>5d} {'-':8s} {dead5[0]:>9.4f}   (its own RNG-matched control)")
    print("\n  d-dead vs each arm's own neuro_lr=0 control; 1 seed, so read >|0.002| only.")


def report(ledger):
    def agg(stage, driver, gran, nlr):
        rows = [ledger.get(key_of(stage, driver, gran, nlr, s)) for s in SEEDS]
        rows = [r for r in rows if r is not None]
        if not rows:
            return None
        return {k: np.array([r[i] for r in rows]) for i, k in enumerate(
            ["acc", "forget", "probe", "h0", "h1", "out", "live"])}

    print("\n" + "=" * 118)
    print("er-own PLASTICITY with pt7 NEUROMODULATOR drivers — task-IL, SGD, main lr 0.1 ep 5, "
          "neuro_lr 1e-3, 3 seeds, TEST")
    print("=" * 118)
    print(f"{'driver':9s} {'gran':8s} {'acc':>16s} {'d-dead':>8s} {'per seed':>28s} {'pos':>4s} "
          f"{'forget':>7s} {'probe':>6s} {'|a-1| h0/h1/out':>22s}")
    E = agg("base", "er", "-", 0.0)
    if E is not None:
        print(f"{'ER':9s} {'(plain)':8s} {E['acc'].mean():>10.4f}±{E['acc'].std():.4f} "
              f"{'—':>8s} {'':>28s} {'':>4s} {E['forget'].mean():>7.4f}")
    for driver in DRIVERS:
        D = agg("dead", driver, "global", DEAD_NLR)
        if D is not None:
            print(f"{driver:9s} {'DEAD':8s} {D['acc'].mean():>10.4f}±{D['acc'].std():.4f} "
                  f"{'—':>8s} {'':>28s} {'':>4s} {D['forget'].mean():>7.4f}")
        for gran in GRANS:
            if (driver, gran) in SKIP:
                print(f"{driver:9s} {gran:8s} {'NOT RUN — K=784 => P is 3.7e8 params (~500x the backbone)':>60s}")
                continue
            L = agg("test", driver, gran, NEURO_LR)
            if L is None or D is None:
                continue
            ps = L["acc"] - D["acc"]
            print(f"{driver:9s} {gran:8s} {L['acc'].mean():>10.4f}±{L['acc'].std():.4f} "
                  f"{ps.mean():>+8.4f} {', '.join(f'{v:+.4f}' for v in ps):>28s} "
                  f"{int((ps > 0).sum())}/3 {L['forget'].mean():>7.4f} {L['probe'].mean():>6.3f} "
                  f"{L['h0'].mean():>7.4f}/{L['h1'].mean():.4f}/{L['out'].mean():.4f}")
    print("\n  reference (plast_taskil, task-IL SGD tuned): naive 0.9784  EWC 0.9821  ER 0.9946")
    print("  d-dead = vs the SAME driver's neuro_lr=0 control (alpha == 1, RNG-matched).")
    print("  'live' eval (no frozen stats) is in the ledger's acc_live column; for a plasticity target")
    print("  the gate is not in the forward, so it CANNOT move pred — the column is the check of that.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "anchor", "deadcheck", "baseline", "test", "norm", "all5", "nostd", "actual", "report"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--drivers", default=None, help="comma filter")
    args = ap.parse_args()
    global DRIVERS
    if args.drivers:
        DRIVERS = tuple(d for d in DRIVERS if d in args.drivers.split(","))
    print(f"er-own plasticity | drivers {DRIVERS} x grans {GRANS} | task-IL sgd "
          f"main_lr {MAIN_LR:g} ep {EPOCHS} neuro_lr {NEURO_LR:g} buffer {BUFFER}\n", flush=True)
    ledger = load_ledger() if args.resume else {}
    if args.part in ("all", "anchor"):
        anchor(ledger)
        if args.part == "anchor":
            return
    if args.part in ("all", "deadcheck"):
        deadcheck(ledger)
        if args.part == "deadcheck":
            return
    if args.part in ("all", "baseline"):
        baseline(ledger)
        if args.part == "baseline":
            return
    if args.part == "norm":
        norm(ledger)
        all5(ledger)
        norm_report(ledger)
        return
    if args.part == "actual":
        actual_stage(ledger)
        actual_report(ledger)
        return
    if args.part == "nostd":
        nostd(ledger)
        nostd_report(ledger)
        return
    if args.part == "all5":
        all5(ledger)
        norm_report(ledger)
        return
    if args.part in ("all", "test"):
        test(ledger)
    report(ledger)


if __name__ == "__main__":
    main()
