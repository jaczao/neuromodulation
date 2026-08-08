"""LEARNING-STATE drivers: the network's own internal state as the neuromodulator signal.

Every driver run so far reads the TASK (`taskid`, `soft_mlp`, `embedding`) or the network's
PERFORMANCE on an input (loss, entropy, novelty). This bank reads the third thing the position
paper's "learning state" names — parameter values, optimisation status, and neuronal activity —
i.e. the internal environment rather than the external one.

    ACTIVITY (per-sample)      act_frac  act_norm  act_entropy  act_pr        + state01
    PARAMETERS (tonic)         w_fro  w_l1  w_absmean  w_absmax
    OPTIMISATION (tonic)       grad_norm  grad_norm_layer  grad_weight_ratio  step_norm

TWO FILTERS DECIDE WHICH OF THE THREE IS ACTUALLY USABLE, and they cut the bank in half.

  PER-SAMPLE VARIANCE. A driver must vary WITHIN a batch or it carries no per-sample information.
  Only the activity drivers do: a weight norm, a gradient norm and a step norm are one number per
  STEP, broadcast across the batch, exactly like pt7's `ACh_ema` / `NE_rise` / `5HT_ema`. That is
  not a small caveat — `signalnet_traces.md` measured a 23-dim feature vector whose 13 broadcast
  scalar columns swamped its 4 live ones, and the whole apparatus emitted a CONSTANT at inference
  (= pt7's `5ht-const` control, reached expensively). Every tonic driver here is marked `tonic=True`
  and is a candidate for that failure by construction.

  AVAILABILITY AT INFERENCE. Optimisation has stopped at test time, so `grad_*` and `step_norm` are
  undefined there and `w_*` is frozen. Under a FORWARD target (gain) they are therefore a fixed gain
  vector at eval and cannot be anything else. They are legitimate only under a TRAINING-TIME-ONLY
  target — plasticity (gates gradients) or the boundary weight decay of position_paper — where the
  gate never enters the forward and `pred` is the plain unmodulated net. Under a forward target,
  say so in the writeup rather than letting the reader assume a live signal.

  So: `state01` is the deployable driver of this bank, and it is the composite of exactly the
  activity statistics that survive both filters.

NORMALISATION IS ANALYTIC, NOT STATISTICAL — the deliberate difference from `neurocore.signals`.
Every divisor here is a constant known from the architecture and the config before a single batch
runs (a layer width `d_l`, a parameter count `n_l`, the learning rate, `ln d_l`), never a measured
running mean/variance. That buys four things a running estimator cannot:

  - correct at step 0, with no warmup over garbage statistics;
  - NO divide-by-~0, which is the only reason the tonic rule exists — that failure comes from an
    ESTIMATED variance collapsing, and a fixed `sqrt(n_l)` cannot collapse. So the tonic drivers in
    this bank are merely uninformative, never catastrophic, provided nothing standardises them;
  - no frozen-vs-live decision at inference: there is no statistic to freeze or to keep updating, so
    the whole axis that `driver_traces/live_traces.md` had to open simply does not exist here;
  - deterministic and batch-order independent, so a driver read cannot drift the run.

`standardize=True` is still available for the per-sample drivers and RAISES for the tonic ones,
rather than silently reproducing pt7's 0.098 collapse.

RANGES after normalisation (`normalize=True`, the default):

    act_frac            [0, 1]        exact, no divisor needed
    act_entropy         [0, 1]        H(h/sum h) / ln(d_l)
    act_pr              [0, 1]        ((sum h)^2/sum h^2 - 1) / (d_l - 1)
    state01             [0, 1]        the three above, per hidden layer
    step_norm           [0, ~1]       ||dW|| / (lr * sqrt(n_l)) -- ADAM ONLY (see StepDriver)
    act_norm            [0, inf)      ||h||_2 / sqrt(d_l)
    w_fro               [0, inf)      ||W||_F / sqrt(n_l); starts at exactly sqrt(2/fan_in)
    w_l1, w_absmean     [0, inf)      mean |w|  (identical -- see WeightDriver)
    w_absmax            [0, inf)      max |w|
    grad_norm*          [0, inf)      ||g|| / sqrt(n)
    grad_weight_ratio   [0, inf)      ||g_l|| / ||W_l||, dimension-free already

NO DRIVER HERE NEEDS A HEAD. The activity drivers are functions of the image through the current
weights (one extra unmodulated forward, no labels), and the rest read parameters or gradients
directly. So the whole bank is oracle-free and label-free, and none of it can suffer the head
distortion `driver_traces/live_traces.md` measured (Pearson r as low as -0.30 between a driver and
the head deployed to predict it). It also builds NO parameters, which matters for rule #10: these
drivers consume the same construction RNG as `position_paper.drivers.ConstDriver`, so a dead control
run against them is RNG-matched to the content-free control without further argument. VERIFY it per
comparison anyway -- `position_paper/wd_modulation.md` records the grouping changing between memory
regimes for reasons that did not follow from parameter counts.

INTERFACE matches position_paper/drivers.py: `.name`, `.kind`, `.K` (int attribute), and
`set_task / value / train_head / live_update / state / restore`. `columns()` is added here and
returns the K column names, so gate magnitude can be reported PER LAYER -- never as a single mean,
which is the trap that hid pt6's out-layer gate inside a 4050-entry average.
"""
import copy
import math

import torch
import torch.nn as nn

from neurocore.utils import DEV, EPS

# ------------------------------------------------------------------ the bank
ACTIVITY = ("act_frac", "act_norm", "act_entropy", "act_pr")
WEIGHT = ("w_fro", "w_l1", "w_absmean", "w_absmax")
OPTIM = ("grad_norm", "grad_norm_layer", "grad_weight_ratio", "step_norm")
COMPOSITE = "state01"
# `const` is a CONTROL, not a driver: content-free, so it isolates "the driver said something"
# from "the gate had per-parameter freedom". Kept OUT of DRIVERS so the grid does not treat it as
# a 14th driver, and run explicitly wherever a result needs it.
CONTROLS = ("const",)
DRIVERS = ACTIVITY + WEIGHT + OPTIM + (COMPOSITE,)

# One value per STEP, broadcast across the batch: within-batch sd is exactly 0.
# Standardising any of these divides by ~0. Enforced in _StateDriver, not just documented.
TONIC = frozenset(WEIGHT + OPTIM)

# The three activity statistics that are bounded, or map exactly onto [0, 1] by an analytic divisor.
BOUNDED_STATS = ("frac", "entropy", "pr")
STAT_OF = {"act_frac": "frac", "act_norm": "norm", "act_entropy": "entropy", "act_pr": "pr"}


def linear_leaves(net):
    """The nn.Linear leaves of `net`, in registration order (l0, l1, l2 for the Split-MNIST MLP)."""
    return [m for m in net.modules() if isinstance(m, nn.Linear)]


# ============================================================================== control
class ConstDriver:
    """m(x) == 1. THE CONTENT-FREE CONTROL -- no input, no loss, no task, no learning state at all.

    Not a dead gate: P is still learned, still per-neuron, still trained by the same meta-loss, and
    the gate it produces is still applied. The ONLY thing removed is any dependence on anything.

    IT EXISTS TO SPLIT ONE RESULT IN TWO. A dead control cannot separate "the driver told the gate
    something" from "the gate had per-parameter freedom trained on replay", because it has no
    learned gate at all. This one can. `position_paper/wd_modulation` ran it for exactly that reason
    and found it captured ~85% of the boundary-decay effect at buffer 1000, with every real driver's
    margin over it inside the noise floor -- so for THIS mechanism the dead control was measuring
    the wrong thing and only the content-free one settled it.

    RUN IT THE MOMENT A NEAR-IDENTICAL DELTA APPEARS ACROSS UNRELATED DRIVERS: that pattern is the
    signature of (b). It is also why this is in the bank rather than borrowed cross-study -- a
    comparison against another study's control shares no harness, no metric convention and no seed.
    """

    tonic = True                                # constant within a batch AND across batches

    def __init__(self, K=1, **_):
        self.name = "const" if K == 1 else f"const{K}"
        self.kind = "control"
        self.K = K
        self.n_missing = 0

    def set_task(self, t):
        return

    def columns(self):
        return [f"const:{i}" for i in range(self.K)]

    @torch.no_grad()
    def value(self, net, X, update=True):
        return torch.ones(X.size(0), self.K, device=X.device)

    def train_head(self, net, X, Y):
        return

    def live_update(self, net, X):
        return

    def state(self):
        return None

    def restore(self, st):
        return


# ============================================================================== base
class _StateDriver:
    """Shared interface + the standardisation guard. Subclasses implement `_raw`."""

    tonic = False

    def __init__(self, name, K, standardize=False, normalize=True):
        if standardize and name in TONIC:
            raise ValueError(
                f"{name!r} is TONIC (one value per step, within-batch sd == 0), so standardising it "
                f"divides by ~0 -- pt7 measured the resulting gate blow-up at 0.098 = chance. Leave "
                f"it raw; its analytic normaliser is a constant and cannot collapse.")
        self.name = name
        self.kind = "state"
        self.K = K
        self.standardize = bool(standardize)
        self.normalize = bool(normalize)
        self.run_mean = None
        self.run_var = None
        self.inited = False

    # ---- interface ----
    def set_task(self, t):
        return                                          # no driver here is task-conditioned

    def train_head(self, net, X, Y):
        return                                          # nothing to train -- there is no head

    def live_update(self, net, X):
        return                                          # value() already advances what state exists

    def columns(self):
        raise NotImplementedError

    @torch.no_grad()
    def value(self, net, X, update=True):
        """(B, K) driver, DETACHED. See each subclass for what is read and WHEN it must be read."""
        v = self._raw(net, X)
        if not self.standardize:
            return v
        if update:
            bm, bv = v.mean(0), v.var(0, unbiased=False)
            if not self.inited:
                self.run_mean, self.run_var, self.inited = bm.clone(), bv.clone(), True
            else:
                self.run_mean = 0.99 * self.run_mean + 0.01 * bm
                self.run_var = 0.99 * self.run_var + 0.01 * bv
        if not self.inited:
            return v
        return (v - self.run_mean) / (self.run_var.sqrt() + EPS)

    def _broadcast(self, per_layer, B):
        """A tonic (B, K) block from K per-step scalars, kept ON DEVICE.

        `per_layer` is a list of 0-dim TENSORS, never floats: a `.item()` per layer per step is a
        device->host sync, and `results/pt7_driver_traces.md` measured exactly that pattern as ~62%
        of a traced step's cost (68 syncs/step). Stacking on device costs one kernel and no sync.
        """
        return torch.stack(per_layer).unsqueeze(0).expand(B, -1).contiguous()

    def state(self):
        return copy.deepcopy((self.run_mean, self.run_var, self.inited))

    def restore(self, st):
        self.run_mean, self.run_var, self.inited = copy.deepcopy(st)


# ============================================================================== activity
class ActivationDriver(_StateDriver):
    """Per-sample statistics of the HIDDEN activations -- "neuronal activity", the one component of
    the learning state that is both per-sample and live at inference.

    STATS (each computed per hidden layer, so K = n_stats * n_hidden):
      frac     fraction of units with h > 0.                          [0, 1] exactly, no divisor.
      norm     ||h||_2, normalised by sqrt(d_l) so widths compare.    [0, inf).
      entropy  H(p), p = h / sum h -- how spread the code is over
               units. Post-ReLU h >= 0, so p is a valid distribution
               without a softmax. Raw range [0, ln d_l]; /ln d_l.     [0, 1].
      pr       participation ratio (sum h)^2 / sum h^2 -- the
               effective NUMBER of units carrying the code. Raw range
               [1, d_l]; mapped by (pr - 1)/(d_l - 1).                [0, 1].
    `entropy` and `pr` measure the same thing (concentration of the code) through different lenses
    and will correlate; they are separate columns because the gate is free to weight them
    differently, not because they are independent signals.

    HOW THE ACTIVATIONS ARE OBTAINED, and the trap that shapes it. The backbone contract is
    `net.plain(x) -> (logits, features)` and `features` is only the PENULTIMATE layer, so h0 is not
    exposed. It is captured with forward hooks -- but note CLAUDE.md's `neurocore/cost.py` gotcha:
    `register_forward_hook` fires on `__call__` and NOT on a custom method, so a hook on `net` would
    never fire for `net.plain(x)` and would silently read nothing. The hooks here go on the nn.Linear
    LEAVES, which `plain` does invoke via `__call__`, so they fire however the forward is spelled.
    They are attached only for the duration of this driver's own forward and removed immediately, so
    they cannot pick up the main training forward or the gated one.

    The hooks see PRE-activations; `act` (default relu) is applied here. If a backbone uses a
    different nonlinearity, pass it -- with the wrong `act`, `frac` and `pr` are quietly meaningless
    rather than wrong-looking. A dead sample (all h == 0) yields frac 0, entropy 0, pr 0 by the
    clamps below, i.e. "maximally concentrated", which is the sane reading of a silent layer.

    COST: one extra unmodulated forward per read, no labels, no head -- the same trade as the
    head-free novelty drivers (`emb_all` vs `NE_emb`).
    """

    def __init__(self, name, stats=None, standardize=False, normalize=True,
                 act=torch.relu, hidden_dims=None):
        self.stats = tuple(stats) if stats is not None else (STAT_OF[name],)
        for s in self.stats:
            if s not in ("frac", "norm", "entropy", "pr"):
                raise ValueError(f"unknown activation stat {s!r}")
        self.act = act
        self.dims = list(hidden_dims) if hidden_dims else None      # filled on first read
        K = len(self.stats) * len(self.dims) if self.dims else 0
        super().__init__(name, K, standardize=standardize, normalize=normalize)

    def columns(self):
        if self.dims is None:
            raise RuntimeError(f"{self.name}: K is known only after the first read (it depends on "
                               f"the backbone's hidden widths); pass hidden_dims= to fix it upfront.")
        return [f"{s}:h{i}" for s in self.stats for i in range(len(self.dims))]

    def _hidden(self, net, X):
        """Post-activation hidden layers, via temporary hooks on the Linear leaves."""
        lins = linear_leaves(net)
        if len(lins) < 2:
            raise ValueError(f"{self.name}: backbone has {len(lins)} Linear leaves; need >= 2 for a "
                             f"hidden layer to exist")
        caught = {}
        handles = [m.register_forward_hook(
            lambda mod, _in, out, key=i: caught.__setitem__(key, out)) for i, m in enumerate(lins)]
        try:
            net.plain(X)
        finally:
            for h in handles:
                h.remove()
        missing = [i for i in range(len(lins) - 1) if i not in caught]
        if missing:
            raise RuntimeError(f"{self.name}: no activation captured for Linear leaf/leaves {missing} "
                               f"-- the forward did not invoke them via __call__.")
        return [self.act(caught[i]) for i in range(len(lins) - 1)]   # drop the output layer

    def _stat(self, h, stat, d):
        if stat == "frac":
            return (h > 0).float().mean(dim=1)                       # [0, 1] by construction
        if stat == "norm":
            v = h.norm(dim=1)
            return v / math.sqrt(d) if self.normalize else v
        if stat == "entropy":
            p = h / (h.sum(dim=1, keepdim=True) + EPS)
            H = -(p * torch.log(p + EPS)).sum(dim=1)
            return H / math.log(d) if self.normalize else H
        s1 = h.sum(dim=1)
        pr = (s1 * s1) / (h.pow(2).sum(dim=1) + EPS)                 # effective number of units
        pr = pr.clamp(1.0, float(d))                                 # all-zero h -> 1 (fully concentrated)
        return (pr - 1.0) / (d - 1.0) if self.normalize else pr

    def _raw(self, net, X):
        hs = self._hidden(net, X)
        if self.dims is None:
            self.dims = [h.size(1) for h in hs]
            self.K = len(self.stats) * len(self.dims)
        cols = [self._stat(h, s, h.size(1)) for s in self.stats for h in hs]
        return torch.stack(cols, dim=1)


class StateVector(ActivationDriver):
    """`state01` -- the composite of every learning-state statistic that lives in [0, 1].

    THE ONE DRIVER IN THIS BANK BUILT TO BE PROJECTED AS A VECTOR. `Gamma = 1 + sum_k m_k P_k` is
    scale-covariant in principle (a 100x larger column can be met by a 100x smaller P row), but all
    columns of P share one learning rate, so mixed-scale columns mean the large ones dominate the
    gradient and the small ones train ~100x slower -- badly conditioned rather than impossible. Plus
    an unbounded column has a tail, and one outlier through an unbounded gain form is how `all4`
    unstandardised at a tuned SGD lr went to NaN -> 0.0980.

    What a projection needs is COMPARABLE VARIANCE across columns, not a common range. These three
    statistics get it from the analytic divisors alone: all K columns land in [0, 1] with no running
    statistics, no standardiser, and therefore none of the failure modes standardisation carries.

    It deliberately EXCLUDES the tonic weight / gradient / step channels even though they normalise
    perfectly well. Mixing tonic and per-sample columns in one projected vector is precisely the
    signal net: its DC offset swamped the live columns and the code came out constant at inference.
    If those channels are wanted, give them their own path -- a separate driver instance and a
    separate gate -- rather than the same vector.

    K = 3 * n_hidden (6 for the Split-MNIST MLP). Column order is stat-major:
    [frac:h0, frac:h1, entropy:h0, entropy:h1, pr:h0, pr:h1] -- see `columns()`.
    """

    def __init__(self, standardize=False, act=torch.relu, hidden_dims=None):
        super().__init__(COMPOSITE, stats=BOUNDED_STATS, standardize=standardize,
                         normalize=True, act=act, hidden_dims=hidden_dims)   # normalize is the point


# ============================================================================== parameters
class WeightDriver(_StateDriver):
    """Magnitude statistics of each weight MATRIX. TONIC: one scalar per layer per step.

      w_fro       ||W||_F / sqrt(n_l)  = RMS weight. This is the one weight channel with a known
                  reference VALUE and not merely a known scale -- but the value depends on the init:
                  sqrt(2/fan_in) under explicit He init (0.0505 / 0.0707 / 0.0707 for this MLP), and
                  1/sqrt(3*fan_in) under PyTorch's DEFAULT nn.Linear init, which is uniform
                  kaiming with a=sqrt(5) (measured: 0.0206 / 0.0289 / 0.0290 -- what the frozen
                  Split-MNIST backbones actually start at, since none of them re-initialises).
      w_l1        ||W||_1 (entrywise) / n_l
      w_absmean   mean |w|
      w_absmax    max |w|, NOT normalised -- a max has no meaningful per-element divisor. It does
                  drift weakly with n (~sqrt(2 ln n) * sigma for gaussian weights), so it is not
                  strictly comparable across layers of very different size.

    w_l1 AND w_absmean ARE THE SAME NUMBER once normalised: ||W||_1 / n IS mean|w|, exactly (measured
    max|diff| = 0.0). They are kept as separate names because they were asked for separately, and the
    identity is then a free correctness check rather than two mechanisms -- if they ever disagree,
    the normaliser is wrong. `w_fro` is genuinely distinct: RMS >= mean-abs, by 2/sqrt(3) ~= 1.1547
    for uniformly-initialised weights (measured 1.154) and sqrt(pi/2) ~= 1.253 for gaussian ones.

    BIASES ARE EXCLUDED, and this is a deliberate choice with a measured reason. Folding them into a
    per-layer statistic buries them: the head has 4,000 weight entries against 10 bias entries, so
    b_4 could double and a combined norm would barely move. That matters because the head bias is the
    one shared parameter a multiplicative activation gate structurally cannot freeze (a bias has no
    input activation to zero), and it was the entire residual leak in pt5 iter-1's otherwise perfect
    disjoint freeze. If bias drift is the quantity of interest it needs its OWN channel, at which
    point it is a different driver -- the same arithmetic as pt6's `mean|P|` = 0.003 hiding an
    out-layer gate of 0.107.
    """

    tonic = True

    def __init__(self, name, standardize=False, normalize=True, n_layers=None):
        self.stat = name
        super().__init__(name, n_layers or 0, standardize=standardize, normalize=normalize)

    def columns(self):
        return [f"{self.stat}:l{i}" for i in range(self.K)]

    def _raw(self, net, X):
        ws = [m.weight for m in linear_leaves(net)]
        self.K = len(ws)
        vals = []
        for w in ws:
            n = w.numel()
            if self.stat == "w_fro":
                vals.append(w.norm() / math.sqrt(n) if self.normalize else w.norm())
            elif self.stat == "w_l1":
                vals.append(w.abs().sum() / n if self.normalize else w.abs().sum())
            elif self.stat == "w_absmean":
                vals.append(w.abs().mean())
            else:                                                    # w_absmax -- no divisor
                vals.append(w.abs().max())
        return self._broadcast(vals, X.size(0))


# ============================================================================== optimisation
class GradDriver(_StateDriver):
    """Gradient magnitude. TONIC, and READ ORDER IS PART OF THE CONTRACT.

    `.grad` is only populated between `loss.backward()` and `optimizer.zero_grad()`, so this driver
    must be read AFTER the backward and BEFORE the zero -- and, if the gate it feeds is meant to
    shape the step being taken, before `optimizer.step()` too. Read anywhere else it returns zeros,
    which is the worst kind of failure because a zero driver is numerically indistinguishable from a
    working driver whose gate learned nothing. So the misses are COUNTED: assert `drv.n_missing <= 1`
    at the end of a run (one legitimate miss, the first read before any backward has happened).

      grad_norm          ||g|| over ALL weight matrices, K = 1, / sqrt(N_total).
      grad_norm_layer    ||g_l|| per matrix, / sqrt(n_l). Layers differ by ~80x in parameter count
                         here (313,600 vs 4,000), so the un-normalised version is not comparable
                         across layers and must not be pooled into one statistic.
      grad_weight_ratio  ||g_l|| / ||W_l||, dimension-free by construction and therefore the sanest
                         optimiser-side channel: it is the scale-free version, and the one that
                         actually says whether a step is large RELATIVE to the weights it moves.
                         Typically 1e-3 to 1e-1. No divisor applied.

    UNDEFINED AT INFERENCE -- there is no backward pass at test time. Under a forward gate this
    driver is a constant at eval; use it only under a training-time-only target, or state the
    constancy.
    """

    tonic = True

    def __init__(self, name, standardize=False, normalize=True, n_layers=None):
        self.stat = name
        K = 1 if name == "grad_norm" else (n_layers or 0)
        super().__init__(name, K, standardize=standardize, normalize=normalize)
        self.n_missing = 0
        self.seen_grad = False

    def columns(self):
        if self.stat == "grad_norm":
            return ["grad_norm:all"]
        return [f"{self.stat}:l{i}" for i in range(self.K)]

    def _raw(self, net, X):
        ws = [m.weight for m in linear_leaves(net)]
        grads = [w.grad for w in ws]
        if any(g is None for g in grads):
            self.n_missing += 1
            K = 1 if self.stat == "grad_norm" else len(ws)
            self.K = K
            return torch.zeros(X.size(0), K, device=ws[0].device)
        self.seen_grad = True
        if self.stat == "grad_norm":
            self.K = 1
            tot = torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt()
            n = sum(w.numel() for w in ws)
            return self._broadcast([tot / math.sqrt(n) if self.normalize else tot], X.size(0))
        self.K = len(ws)
        vals = []
        for w, g in zip(ws, grads):
            gn = g.norm()
            if self.stat == "grad_weight_ratio":
                vals.append(gn / (w.norm() + EPS))                   # already dimension-free
            else:
                vals.append(gn / math.sqrt(w.numel()) if self.normalize else gn)
        return self._broadcast(vals, X.size(0))


class StepDriver(_StateDriver):
    """`step_norm` -- ||dW_l|| actually applied, per layer. TONIC, and LAG-1 by construction.

    Measured by DIFFING the weights against the previous read rather than by instrumenting the
    optimiser, so nothing has to be threaded through the training loop and the driver works
    identically under SGD, Adam, or anything else. That lag-1 shape is the same convention every
    stateful driver in this project uses (pt2's drivers compute from step t and gate step t+1) and is
    unavoidable anyway: the update cannot be known before it is applied.

    THE READ CADENCE IS PART OF THE MEASUREMENT, and getting it wrong is silent. What this returns is
    the movement SINCE THE PREVIOUS READ, so it equals one step's update only if it is read exactly
    once per step: read it twice in a step and the second read sees ~0, read it every other step and
    it reports two steps' movement as one. Measured on a fresh Adam run at lr 3e-4, one read per
    step gives ~0.88-0.92 (the ~[0,1] bound holding), and a read spanning two steps gives ~1.9 --
    which looks exactly like the bias-correction overshoot the bound allows, but is not.

    THE NORMALISER IS THE ONLY ANALYTIC BOUND IN THE BANK, AND IT IS ADAM-SPECIFIC. Adam's
    per-parameter update is lr * m_hat / (sqrt(v_hat) + eps) and |m_hat / sqrt(v_hat)| <~ 1 in steady
    state, so ||dW_l|| <~ lr * sqrt(n_l) and the normalised driver lands in ~[0, 1]. It can exceed 1
    in the first steps (bias correction) and just after a gradient sign flip -- it is a steady-state
    bound, not a hard one. Under SGD the same quantity is exactly lr * ||g_l||, so dividing by
    lr * sqrt(n_l) reduces it to `grad_norm_layer` and the bound is GONE. Pass the lr you are
    actually using; `lr=None` disables the normalisation rather than guessing.

    The first read has no previous snapshot and returns zeros (counted in `n_missing`, same contract
    as GradDriver). The snapshot is a full copy of the weight matrices -- ~1.9 MB for the
    Split-MNIST MLP, negligible, but it scales with the backbone.
    """

    tonic = True

    def __init__(self, lr=None, standardize=False, normalize=True, n_layers=None):
        self.lr = lr
        if normalize and lr is None:
            normalize = False                                        # no lr, no analytic divisor
        super().__init__("step_norm", n_layers or 0, standardize=standardize, normalize=normalize)
        self.prev = None
        self.n_missing = 0

    def columns(self):
        return [f"step_norm:l{i}" for i in range(self.K)]

    def _raw(self, net, X):
        ws = [m.weight for m in linear_leaves(net)]
        self.K = len(ws)
        if self.prev is None:
            self.prev = [w.detach().clone() for w in ws]
            self.n_missing += 1
            return torch.zeros(X.size(0), self.K, device=ws[0].device)
        vals = []
        for w, p in zip(ws, self.prev):
            d = (w.detach() - p).norm()
            vals.append(d / (self.lr * math.sqrt(w.numel())) if self.normalize else d)
        self.prev = [w.detach().clone() for w in ws]
        return self._broadcast(vals, X.size(0))

    def state(self):
        return copy.deepcopy((self.prev, self.n_missing, self.run_mean, self.run_var, self.inited))

    def restore(self, st):
        self.prev, self.n_missing, self.run_mean, self.run_var, self.inited = copy.deepcopy(st)


# ============================================================================== factory
def make_driver(name, lr=None, standardize=False, normalize=True, act=torch.relu,
                hidden_dims=None, n_layers=None):
    """The one constructor. `lr` is read ONLY by `step_norm` (its analytic divisor).

    Construction order is RNG-relevant across a study -- but nothing here builds a parameter or draws
    from any generator, so every driver in this bank is RNG-neutral and a dead control run against
    one is matched to a control run against any other. That is a property to CHECK per comparison,
    not to assume (position_paper/wd_modulation.md records the control grouping changing between
    memory regimes for reasons that did not follow from parameter counts).
    """
    if name in CONTROLS:
        return ConstDriver(1)
    if name == COMPOSITE:
        return StateVector(standardize=standardize, act=act, hidden_dims=hidden_dims)
    if name in ACTIVITY:
        return ActivationDriver(name, standardize=standardize, normalize=normalize,
                                act=act, hidden_dims=hidden_dims)
    if name in WEIGHT:
        return WeightDriver(name, standardize=standardize, normalize=normalize, n_layers=n_layers)
    if name == "step_norm":
        return StepDriver(lr=lr, standardize=standardize, normalize=normalize, n_layers=n_layers)
    if name in OPTIM:
        return GradDriver(name, standardize=standardize, normalize=normalize, n_layers=n_layers)
    raise ValueError(f"unknown learning-state driver {name!r}; known: {' '.join(DRIVERS)}")
