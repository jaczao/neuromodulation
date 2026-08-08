"""LEARNING-STATE DRIVERS across seven mechanisms: does the internal environment drive anything?

`state_drivers/drivers.py` builds 13 drivers out of the three things the position paper's "learning
state" names -- neuronal activity, parameter values, optimisation status. This runs all of them
through every mechanism family the project has built, at operating points that are ALREADY
val-tuned (`neurocore.tuned`; rule #11 -- an untuned main lr is what manufactured pt7's fake +0.11
global-LR "win", and tuning ER is what dissolved pt7's +0.14).

  REGIME (8)                        GATE LANDS ON            BASE   TUNED POINT
  gain      classil adam            forward activations      er     lr 3e-4  ep 5
  gain      taskil  adam            forward activations      er     lr 3e-4  ep 10
  slope     classil adam            forward hidden ReLUs     er     lr 3e-4  ep 5
  temp      classil adam            forward out logits       er     lr 3e-4  ep 5
  plast     taskil  sgd             gradients (lookahead)    er     lr 1e-1  ep 5
  selplast  taskil  sgd             gradients, binary STE    naive  lr 3e-3  ep 5
  lossmod   classil adam            the training loss        er     lr 3e-4  ep 5
  wdecay    classil sgd             weights, at boundaries   er     lr 3e-2  ep 5

13 drivers x 8 regimes = 104 live cells, granularity NEURON where the mechanism has one (`lossmod`
weights per-TASK coefficients, so it does not), learned projection, 1 seed, live only -- plus 8
dead-gate controls on the composite `state01`, one per regime. See CONTROLS below for why those 8.

THE SPLIT THAT DECIDES HOW A CELL CAN BE READ, and it is not a caveat but the study's main axis:

  52 cells are TRAINING-TIME-ONLY (plast, selplast, lossmod, wdecay). The gate never enters the
  forward, `pred` is the plain unmodulated net, and every driver is legitimate -- including the
  tonic ones, which is exactly why those four regimes are here. This is the only setting in which a
  parameter-norm or gradient-norm driver can be honestly tested.

  20 cells are LIVE FORWARD GATES: the 5 per-sample activity drivers (act_frac, act_norm,
  act_entropy, act_pr, state01) under the 4 forward regimes.

  32 cells are EVAL-DEGENERATE BY CONSTRUCTION: the 8 tonic drivers under the 4 forward regimes.
  Optimisation has stopped at test time, so there is no gradient and no weight movement to read.
  The convention here is explicit rather than implicit -- gradients are ZEROED before evaluation, so
  `grad_*` and `step_norm` read exactly 0 at eval, the gate returns to Gamma = 1, and the arm is
  numerically its own dead control at inference whatever it did during training. `w_*` is the one
  tonic family that survives eval, as a CONSTANT, i.e. pt7's `5ht-const` scale degeneracy. Both are
  reported, neither is a mechanism test. The alternative convention (freeze the last training value
  and keep gating with it) was rejected because it hides the degeneracy behind a plausible number.
  `n_miss` counts the reads that found no gradient, so the degeneracy is visible in the ledger.

CONTROLS. Live-only was the requested scope, so there is no per-driver `d-dead`. The 8 `state01`
dead cells exist to answer the one question that live-only cannot: whether a dead gate over these
drivers is numerically the plain baseline. It should be -- this bank builds NO parameters and draws
NO RNG (unlike `p7.Heads`, whose construction is worth ~0.002 at width 400 and ~0.06 at width 5) --
but `pt5_taskil/plast_taskil.py` only KNEW its dead control was bit-exact because it ran it, and
`position_paper/wd_modulation.md` found the control grouping changing between memory regimes for
reasons that did not follow from parameter counts. If dead == baseline in all 8, every live number
here is readable against the plain baseline and the missing controls cost nothing. If it does not,
the live-only deltas are not deltas and the study needs the other 104 controls.

THE DIAGNOSTIC THAT MAKES THIS FALSIFIABLE. `msd` is the driver's mean WITHIN-BATCH standard
deviation at eval. It is the tonic test, measured rather than assumed: a per-sample driver should
read ~1e-2, a tonic one exactly 0. It is reported for every cell because the whole design rests on
that split, and because `driver_traces/signalnet_traces.md` found a "rich 23-dim signal vector"
whose within-batch sd was 3-5 orders below its magnitude -- a constant wearing a feature vector's
clothes. `probe` (linear task-decodability of m) is reported beside it: `pt5_taskil/plast_drivers`
measured the MOST task-decodable driver (vecproj, 0.934) as the WORST cell, so a high probe is not
a prediction of benefit -- it is there so the two can be checked against each other again.

METRIC: class-IL accuracy is the MACRO mean of the five per-task accuracies, for mechanism and
baseline alike (the frozen pt7 eval pools instead, which biases a gated cell ~+0.0015 against a
macro-averaged baseline -- the same size as several "small consistent positives" in this project).
task-IL masks eval logits to the task's 2 classes.

NO NEW TUNING, as scoped: every main lr / epoch count comes from `neurocore.tuned`, and neuro_lr
from its optimizer-scale default (adam 1e-3, sgd 3e-3). That default is a REUSED value, not a tune,
so a null here is "null at the inherited operating point", not "null at this mechanism's optimum".

Run:  uv run python state_drivers/run_state.py --part anchor
      uv run python -m neurocore.shard --script state_drivers/run_state.py \
          --ledger state_drivers/state_drivers_results.tsv \
          --split mechs=gain,slope,temp,plast,selplast,lossmod,wdecay \
          --args "--part grid --resume" --workers 6 --device mps
"""
import argparse
import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

import pt7_neuromodulators as p7                                   # noqa: E402  (frozen, read-only)
import pt7_plast_tempslope as pts                                  # noqa: E402  (frozen, read-only)
from prototype.data import SplitMNIST                              # noqa: E402
from neurocore import shard                                        # noqa: E402
from neurocore.cost import Cost, count_params                      # noqa: E402
from neurocore.ledger import NOISE_FLOOR                           # noqa: E402
from neurocore.tuned import tuned_main, tuned_neuro_lr             # noqa: E402
from neurocore.utils import DEV, EPS, rng_frozen                   # noqa: E402
from state_drivers.drivers import DRIVERS, TONIC, make_driver      # noqa: E402

SEED = 42
BUFFER = 1000
# MEMORY REGIMES (rule #12). `regime` is a KEY column, not a metric: the same mechanism under a
# different buffer is a different cell.
#
# `rfree` IS DELIBERATELY ABSENT, and that is the rule-#12-compliant choice rather than a gap.
# Rehearsal-free means reporting methods that do NOT REQUIRE rehearsal; setting a replay method's
# buffer to 0 just makes it naive, which is a DEGENERACY CHECK and must be labelled as one. For
# `wdecay` the degeneracy is structural and provable without running it: the boundary meta-loss
# trains P on BUFFER samples, so at buffer 0 P never leaves zero, f == exp(0) == 1 exactly, and the
# mechanism does not underperform -- it does not EXIST. `position_paper/wd_modulation` measured
# exactly that (every cell 0.1976 = naive, |f-1| = 0.0000). The same argument covers `plast` /
# `selplast` (their lookahead meta-loss is fed by the replay-augmented batch) and `lossmod` (with
# one task per batch the per-class adjustment collapses to a scalar rescale). See `--part rfree`.
BUFFERS = {"normal": 1000, "budget": 200}
GRAN = "neuron"
PROBLEM = "splitmnist"
N_TASKS = 5
META_STEPS = 50                          # wdecay boundary meta-steps (position_paper/wd_modulation)
MASK_SPREAD = 0.5                        # selplast STE offset init; see build_gate

# v2 schema: adds the `regime` KEY column and the neurocore.cost accounting block. A new file
# rather than an in-place widening, because a mixed-schema ledger silently breaks --resume; the v1
# rows are carried over exactly by `--part migrate` (cost is DECLARED and deterministic from the
# driver / gate / buffer, so nothing has to be re-run to fill the new columns).
TSV = shard.ledger_path(Path(__file__).resolve().parent / "state_drivers_regimes_results.tsv")
TSV_V1 = Path(__file__).resolve().parent / "state_drivers_results.tsv"
KEYS = ["regime", "mech", "metric", "opt", "driver", "arm", "seed"]
METRICS = ["acc", "forget", "probe", "g_h0", "g_h1", "g_out", "msd", "n_miss"]
COST = ["backbone_params", "extra_params", "param_ratio", "buffer_bytes",
        "fwd_train", "bwd_train", "fwd_infer", "bwd_infer"]
COLS = KEYS + METRICS + COST

# mechanism -> where the gate lands. FORWARD mechanisms make a driver that is undefined at
# inference degenerate; the other four never put the gate in the forward.
FORWARD = ("gain", "slope", "temp")
GRADIENT = ("plast", "selplast")

REGIMES = (
    dict(mech="gain",     metric="classil", opt="adam", base="er"),
    dict(mech="gain",     metric="taskil",  opt="adam", base="er"),
    dict(mech="slope",    metric="classil", opt="adam", base="er"),
    dict(mech="temp",     metric="classil", opt="adam", base="er"),
    dict(mech="plast",    metric="taskil",  opt="sgd",  base="er"),
    dict(mech="selplast", metric="taskil",  opt="sgd",  base="naive"),
    dict(mech="lossmod",  metric="classil", opt="adam", base="er"),
    dict(mech="wdecay",   metric="classil", opt="sgd",  base="er"),
)
MECHS = tuple(dict.fromkeys(r["mech"] for r in REGIMES))
L2T = torch.tensor([c // 2 for c in range(10)], device=DEV)        # class -> task, for lossmod


# Drivers that need their OWN forward through the backbone. The rest read parameters or gradients,
# which costs no pass at all -- so the compute column separates the activity family from the other
# two exactly as the availability filter does.
ACTIVITY_DRIVERS = frozenset(("act_frac", "act_norm", "act_entropy", "act_pr", "state01"))


def cell_cost(mech, driver, gate, net, cap):
    """DECLARED cost of one cell (neurocore.cost). Counts are structural, not measured.

    `fwd_infer` / `bwd_infer` are the DEPLOYED inference cost, which for the four training-time-only
    mechanisms is the plain net (1 forward, 0 backward) no matter how expensive the gate was to
    train -- the gate never enters the forward, so a deployed system would not compute the driver at
    all. This study DOES compute it at eval, but only to log `probe` and `msd`; pricing that as an
    inference cost would misreport the mechanism.
    """
    cap = int(cap)
    drv_fwd = 1.0 if driver in ACTIVITY_DRIVERS else 0.0
    if mech in FORWARD:                      # gate is in the forward at train AND at inference
        f_tr, b_tr, f_in = 1.0 + drv_fwd, 1.0, 1.0 + drv_fwd
    elif mech in GRADIENT:                   # + the lookahead forward and the meta backward
        f_tr, b_tr, f_in = 2.0 + drv_fwd, 2.0, 1.0
    else:                                    # lossmod / wdecay: ordinary step, plain net at eval
        f_tr, b_tr, f_in = 1.0 + drv_fwd, 1.0, 1.0
    return Cost(backbone_params=count_params(net), extra_params=count_params(gate),
                buffer_bytes=cap * 784 * 4 + cap * 8,      # Reservoir: X float32 (cap,784), Y int64
                fwd_train=f_tr, bwd_train=b_tr, fwd_infer=f_in, bwd_infer=0.0)


def point(r):
    """Main-net operating point + the neuromod lr. Both come from neurocore.tuned; nothing new."""
    tp = tuned_main(PROBLEM, r["metric"], r["base"], r["opt"])
    nlr = tuned_neuro_lr(PROBLEM, r["metric"], r["base"], r["opt"], r["mech"], GRAN)
    return tp["lr"], tp["epochs_per_task"], nlr


# ============================================================================== gates
class BinaryPlastGate(pts.PlastGate):
    """alpha in {0,1} via the clipped STE, over the same P as pt7's plasticity gate.

    Copy-forward of `pt5_taskil/plast_binary`'s estimator onto this study's gate (THESIS-PLAN B's
    indicator form). It is the one gate here that CANNOT express a graded global LR rescale -- a
    uniform binary gate is either alpha == 1 (exactly vanilla) or alpha == 0 (no learning) -- which
    is why it is worth running: the global-LR artifact that explained pt7's `ach_ema` and
    `meta_schedule`'s alpha -> 0.99 is unavailable by construction here.

    OFFSET INIT IS `spread`, NOT PARITY, and that is a measured requirement rather than a taste.
    Initialised exactly on the threshold (b = 0), the meta-loss pushes z uniformly UP -- a one-step
    lookahead always prefers a larger step -- until every element leaves the |z| <= 1 window, at
    which point P.grad is EXACTLY 0 and the gate welds at alpha == 1 forever. `plast_binary`
    measured that: stuck fraction 1.00 and runs byte-identical to naive, with a val sweep returning
    the same number to 4 dp across five decades of neuro_lr. The clipping is what makes it a one-way
    door. b ~ N(0, 0.5^2) starts ~50% frozen with ~95% of elements inside the window instead.
    """

    def __init__(self, mech, K, lr, seed=SEED, spread=MASK_SPREAD):
        super().__init__(mech, K, lr)
        g = torch.Generator().manual_seed(seed + 991)
        for name in ("P", "P0", "P1", "P2"):
            p = getattr(self, name, None)
            if p is None:
                continue
            b = (torch.randn(p.size(1) if p.dim() > 1 else 1, generator=g) * spread).to(DEV)
            self.register_buffer(f"b_{name}", b)

    def _off(self, name):
        return getattr(self, f"b_{name}")

    def mult(self, mbar):
        one = lambda n: torch.ones(n, device=DEV)                  # noqa: E731
        if self.mech == "neuron":
            a = binary_ste(mbar @ self.P + self._off("P"))
            ah0, ah1, ao = a[:p7.H0], a[p7.H0:p7.H0 + p7.H1], a[p7.H0 + p7.H1:]
            return ({0: ah0[:, None], 1: ah0, 2: ah1[:, None], 3: ah1, 4: ao[:, None], 5: ao},
                    (ah0, ah1, ao))
        if self.mech == "synapse":
            a0 = binary_ste(mbar @ self.P0 + self._off("P0")).view(p7.H0, 784)
            a1 = binary_ste(mbar @ self.P1 + self._off("P1")).view(p7.H1, p7.H0)
            a2 = binary_ste(mbar @ self.P2 + self._off("P2")).view(p7.OUT, p7.H1)
            return ({0: a0, 1: one(p7.H0), 2: a1, 3: one(p7.H1), 4: a2, 5: one(p7.OUT)}, (a0, a1, a2))
        a = binary_ste(mbar @ self.P + self._off("P"))
        return ({i: a for i in range(pts.NPARAMS)}, (a,))


class _BinarySTE(torch.autograd.Function):
    """alpha = 1[z >= 0] forward; d alpha/dz := 1[|z| <= 1] backward. Copy-forward of plast_binary."""

    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return (z >= 0).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (z,) = ctx.saved_tensors
        return grad_out * (z.abs() <= 1).to(grad_out.dtype)


def binary_ste(z):
    return _BinarySTE.apply(z)


class CoefGate(nn.Module):
    """lossmod: c = softmax(mbar @ Pc), Pc:(K, T) zero-init => c UNIFORM at parity.

    Uniform is the exact parity control in the LOGIT formulation (a constant c cancels inside the
    softmax), so zero-init is a genuine no-op start, exactly as a zero-init P is for the gain gate.
    The v2 per-class formulation is the one used here: `position_paper/loss_modulation` measured the
    v1 per-SAMPLE form to be capped at the baseline BY ALGEBRA (plain batch-mean CE already IS
    sum_T (n_T/N) L_T, so a posterior estimating that vector can only approach ER from below),
    while the per-class form -- z_c += log c_task(c) -- moved +0.0232 and, uniquely in phase B,
    beat its content-free control. It is per-BATCH by construction, which is what a free constant
    vector could not be and is the measured reason that control fails there.
    """

    def __init__(self, K, lr, T=N_TASKS):
        super().__init__()
        self.Pc = nn.Parameter(torch.zeros(K, T))
        self.to(DEV)
        self.opt = torch.optim.Adam(self.parameters(), lr)

    def coef(self, mbar):
        return F.softmax(mbar @ self.Pc, dim=0)

    @torch.no_grad()
    def mag(self, m):
        c = self.coef(m.mean(0))
        return {"h0": 0.0, "h1": 0.0, "out": (c - 1.0 / c.numel()).abs().mean().item()}


def build_gate(mech, K, nlr, seed=SEED):
    """The gate + whether it owns a separate neuromod optimizer."""
    if mech == "gain":
        g = p7.NeuronGate(K, None).to(DEV)
        return g, torch.optim.Adam(g.parameters(), nlr)
    if mech in ("slope", "temp"):
        g = pts.GateForm(mech, K).to(DEV)
        return g, torch.optim.Adam(g.parameters(), nlr)
    if mech == "plast":
        g = pts.PlastGate(GRAN, K, nlr)                            # owns g.opt
        return g, g.opt
    if mech == "selplast":
        g = BinaryPlastGate(GRAN, K, nlr, seed=seed)
        return g, g.opt
    if mech == "lossmod":
        g = CoefGate(K, nlr)
        return g, g.opt
    if mech == "wdecay":
        g = pts.PlastGate(GRAN, K, nlr)                            # DecayGate: same object, weight-side
        return g, g.opt
    raise ValueError(mech)


# ============================================================================== eval
@torch.no_grad()
def gate_mag(gate, m):
    """Per-layer DEVIATION FROM PARITY of the gate, always as h0/h1/out.

    Reported per layer and never as a single mean: pt6's follow-up found `mean|P|` = 0.003 reading
    as "gate ~ parity" while the same gate was h0 0.001 / h1 0.002 / out 0.107, i.e. a pure
    per-task logit adjustment that the average hid entirely.

    NOTE this is the gate recomputed AT EVAL. For a training-time-only mechanism that is a PROXY for
    the gate that was actually applied, and `position_paper/wd_modulation` measured the proxy off by
    eleven orders of magnitude for one raw headless driver reading out of distribution on the test
    stream. Every driver in this bank is bounded or analytically normalised, so the gap should be
    small -- but for plast / selplast / wdecay this column is a diagnostic, not the applied gate.
    """
    if gate is None:
        return {"h0": 0.0, "h1": 0.0, "out": 0.0}
    if hasattr(gate, "mag"):                                       # GateForm (slope/temp), CoefGate
        return gate.mag(m)
    if hasattr(gate, "per_layer_mag"):                             # NeuronGate (gain)
        return gate.per_layer_mag(m)
    _, alphas = gate.mult(m.mean(0))                               # PlastGate / BinaryPlastGate
    if len(alphas) == 1:                                           # global: one scalar, all layers
        v = (alphas[0] - 1).abs().mean().item()
        return {"h0": v, "h1": v, "out": v}
    return {k: (a - 1).abs().mean().item() for k, a in zip(("h0", "h1", "out"), alphas)}


def _masked(logits, allowed):
    add = torch.full_like(logits, float("-inf"))
    add[:, list(allowed)] = 0.0
    return logits + add


@torch.no_grad()
def evaluate(net, drv, gate, mech, loaders, metric):
    """MACRO mean of the per-task accuracies (both arms, same convention -- see the module docstring).

    Gradients are ZEROED first, so a driver that reads `.grad` returns exactly 0 at eval and the
    forward gate falls back to parity. That is the honest reading of "optimisation has stopped", and
    it is what makes the 32 eval-degenerate cells legible rather than plausible-looking.
    """
    for p in net.parameters():
        p.grad = None
    accs, Ms, Ts, mags, nb = [], [], [], {"h0": 0.0, "h1": 0.0, "out": 0.0}, 0
    sd_tot, sd_n = 0.0, 0
    for i, (_, te) in enumerate(loaders):
        c = tot = 0
        for x, y in te:
            x, y = x.to(DEV), y.to(DEV)
            xf = x.view(x.size(0), -1)
            m = drv.value(net, xf, update=False)
            if mech in FORWARD and gate is not None:
                logits = gate(net, m, xf)
            else:
                logits = net.plain(xf)[0]
            if metric == "taskil":
                logits = _masked(logits, p7.SEQ[i])
            c += (logits.argmax(1) == y).sum().item(); tot += y.numel()
            g = gate_mag(gate, m)
            for k in mags:
                mags[k] += g[k]
            nb += 1
            sd_tot += m.std(0).mean().item() if m.size(0) > 1 else 0.0; sd_n += 1
            Ms.append(m.cpu()); Ts.append(torch.full((y.numel(),), i))
        accs.append(c / tot)
    M, T = torch.cat(Ms), torch.cat(Ts)
    probe = p7._probe(M, T, M.size(1))
    return (float(np.mean(accs)), probe, {k: mags[k] / max(nb, 1) for k in mags},
            sd_tot / max(sd_n, 1))


# ============================================================================== the loops
def run(mech, metric, opt_kind, driver_name, dead=False, seed=SEED, regime="normal"):
    r = dict(mech=mech, metric=metric, opt=opt_kind,
             base=next(x["base"] for x in REGIMES if x["mech"] == mech and x["metric"] == metric))
    main_lr, epochs, nlr = point(r)
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(N_TASKS)]
    net = p7.Net().to(DEV)
    drv = make_driver(driver_name, lr=main_lr)
    # K is known only after a read for the activity drivers (it depends on the backbone's widths),
    # so probe it once RNG-neutrally before the gate is sized.
    with rng_frozen(), torch.no_grad():
        x0 = next(iter(loaders[0][0]))[0].to(DEV).view(-1, 784)
        drv.value(net, x0, update=False)
    gate, gopt = build_gate(mech, drv.K, nlr, seed=seed)
    cap = BUFFERS[regime]
    buf = p7.Reservoir(cap)
    loss_fn = p7.masked_ce if metric == "taskil" else p7.CE
    A = np.full((N_TASKS, N_TASKS), np.nan)
    main_opt = (torch.optim.Adam(net.parameters(), main_lr) if opt_kind == "adam"
                else torch.optim.SGD(net.parameters(), main_lr))

    for t in range(N_TASKS):
        mbar_sum, mbar_n = None, 0                                 # wdecay: this task's mean driver
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                rp = buf.sample_any(64)
                if rp is not None and mech != "selplast":
                    Xs.append(rp[0].to(DEV)); Ys.append(rp[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)

                if mech in FORWARD:
                    # Driver FIRST: `.grad` still holds the previous step's gradient, so a grad
                    # driver is a lag-1 control signal (the pt2 convention) rather than empty.
                    m = drv.value(net, Xm)
                    main_opt.zero_grad(); gopt.zero_grad()
                    loss_fn(gate(net, m, Xm), Ym).backward()
                    main_opt.step()
                    if not dead:
                        gopt.step()

                elif mech == "lossmod":
                    m = drv.value(net, Xm)
                    main_opt.zero_grad(); gopt.zero_grad()
                    c = gate.coef(m.mean(0))
                    adj = torch.log(c.clamp_min(1e-12))[L2T]       # per-class, from its task's coef
                    logits = net.plain(Xm)[0] + adj.unsqueeze(0)
                    (loss_fn(logits, Ym) if metric == "taskil"
                     else F.cross_entropy(logits, Ym)).backward()
                    main_opt.step()
                    if not dead:
                        gopt.step()

                elif mech in GRADIENT:
                    params = pts._net_params(net)
                    # selplast is the STANDALONE buf-cur arm: the main step sees the CURRENT task
                    # only, the meta-loss sees the replay-augmented batch.
                    Xg, Yg = (Xs[0], Ys[0]) if mech == "selplast" else (Xm, Ym)
                    g = torch.autograd.grad(loss_fn(net.plain(Xg)[0], Yg), params)
                    for p_, gi in zip(params, g):                  # so a grad driver can read them
                        p_.grad = gi.detach()
                    Xmeta, Ymeta = Xm, Ym
                    if mech == "selplast" and rp is not None:
                        Xmeta = torch.cat([Xs[0], rp[0].to(DEV)]); Ymeta = torch.cat([Ys[0], rp[1].to(DEV)])
                    mbar = drv.value(net, Xmeta).mean(0)
                    mult, _ = gate.mult(mbar)
                    if not dead:
                        Wf = [params[i].detach() - main_lr * (mult[i] * g[i])
                              for i in range(pts.NPARAMS)]
                        meta = loss_fn(pts._fwd_fast(Wf, Xmeta), Ymeta)
                        gopt.zero_grad(); meta.backward(); gopt.step()
                        mult, _ = gate.mult(mbar)
                    with torch.no_grad():
                        for i in range(pts.NPARAMS):
                            params[i].add_(mult[i].detach() * g[i], alpha=-main_lr)

                else:                                              # wdecay: plain ER step in-task
                    params = pts._net_params(net)
                    g = torch.autograd.grad(loss_fn(net.plain(Xm)[0], Ym), params)
                    for p_, gi in zip(params, g):
                        p_.grad = gi.detach()
                    m = drv.value(net, Xm).mean(0)
                    mbar_sum = m.clone() if mbar_sum is None else mbar_sum + m
                    mbar_n += 1
                    with torch.no_grad():
                        for i in range(pts.NPARAMS):
                            params[i].add_(g[i], alpha=-main_lr)

                buf.add(x, y)
                drv.train_head(net, Xm, Ym)

        if mech == "wdecay" and mbar_n and not dead:
            _boundary_decay(net, gate, gopt, buf, mbar_sum / mbar_n, loss_fn)

        with rng_frozen():                    # an eval's DataLoader iterators would shift training
            for i in range(t + 1):
                A[t, i] = _acc(net, drv, gate, mech, loaders[i][1],
                               p7.SEQ[i] if metric == "taskil" else None)

    acc, probe, mags, msd = evaluate(net, drv, gate, mech, loaders, metric)
    forget = float(np.mean([max(A[k, i] for k in range(i, N_TASKS)) - A[N_TASKS - 1, i]
                            for i in range(N_TASKS)]))
    return dict(acc=acc, forget=forget, probe=probe, mags=mags, msd=msd,
                n_miss=getattr(drv, "n_missing", 0),
                cost=cell_cost(mech, driver_name, gate, net, cap))


def _boundary_decay(net, gate, gopt, buf, mbar, loss_fn):
    """w <- f . w ONCE per task, f = exp(mbar @ P) meta-trained on the buffer for META_STEPS steps.

    The project's variant of the position paper's per-STEP form, and the reason it exists is
    measured: applied every step, f reaches the weight as f^4750 while the meta-loss is ONE step
    deep, so the gate is chosen on a criterion three orders of magnitude away from what it does --
    `wd_modulation` found the per-step form null-or-divergent and NON-MONOTONE in neuro_lr (a
    positive feedback loop, so there is no stability threshold to normalise against), while the
    boundary form was stable to 1e-1 and beat tuned ER by ~2pts.
    """
    params = pts._net_params(net)
    for _ in range(META_STEPS):
        s = buf.sample_any(64)
        if s is None:
            break
        xb, yb = s[0].to(DEV).view(-1, 784), s[1].to(DEV)
        mult, _ = gate.mult(mbar)
        Wf = [params[i] * mult[i] for i in range(pts.NPARAMS)]
        gopt.zero_grad(); loss_fn(pts._fwd_fast(Wf, xb), yb).backward(); gopt.step()
    with torch.no_grad():
        mult, _ = gate.mult(mbar)
        for i in range(pts.NPARAMS):
            params[i].mul_(mult[i].detach())


@torch.no_grad()
def _acc(net, drv, gate, mech, loader, allowed):
    """A-matrix cell. Reads the driver with update=False so the intermediate evals cannot advance
    any driver state into the training run (the live-vs-frozen axis is not open here)."""
    c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        xf = x.view(x.size(0), -1)
        logits = (gate(net, drv.value(net, xf, update=False), xf)
                  if mech in FORWARD and gate is not None else net.plain(xf)[0])
        if allowed is not None:
            logits = _masked(logits, allowed)
        c += (logits.argmax(1) == y).sum().item(); tot += y.numel()
    return c / tot


def run_baseline(metric, opt_kind, base, seed=SEED, regime="normal", mech="gain"):
    """The plain ungated baseline in THIS harness -- no driver, no gate, nothing constructed."""
    main_lr, epochs, _ = point(dict(metric=metric, base=base, opt=opt_kind, mech="gain"))
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(N_TASKS)]
    net = p7.Net().to(DEV)
    cap = BUFFERS[regime] if base == "er" else 0
    buf = p7.Reservoir(cap)
    loss_fn = p7.masked_ce if metric == "taskil" else p7.CE
    opt = (torch.optim.Adam(net.parameters(), main_lr) if opt_kind == "adam"
           else torch.optim.SGD(net.parameters(), main_lr))
    A = np.full((N_TASKS, N_TASKS), np.nan)
    for t in range(N_TASKS):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                if base == "er":
                    rp = buf.sample_any(64)
                    if rp is not None:
                        Xs.append(rp[0].to(DEV)); Ys.append(rp[1].to(DEV))
                opt.zero_grad(); loss_fn(net.plain(torch.cat(Xs))[0], torch.cat(Ys)).backward()
                opt.step(); buf.add(x, y)
        with rng_frozen():
            for i in range(t + 1):
                A[t, i] = _acc(net, None, None, "none", loaders[i][1],
                               p7.SEQ[i] if metric == "taskil" else None)
    acc = float(np.nanmean(A[N_TASKS - 1, :]))
    forget = float(np.mean([max(A[k, i] for k in range(i, N_TASKS)) - A[N_TASKS - 1, i]
                            for i in range(N_TASKS)]))
    return dict(acc=acc, forget=forget, probe=float("nan"),
                mags={"h0": 0.0, "h1": 0.0, "out": 0.0}, msd=0.0, n_miss=0,
                cost=cell_cost(mech, base, None, net, cap))




# ============================================================================== ledger (v2 schema)
NK = len(KEYS)


def load_ledger(path=None):
    path = Path(path) if path else TSV
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines()[1:]:
        if line.strip():
            f = line.split("\t")
            rows[tuple(f[:NK])] = tuple(float(v) for v in f[NK:])
    return rows


def append(key, vals, path=None):
    path = Path(path) if path else TSV
    if not path.exists():
        path.write_text("\t".join(COLS) + "\n")
    with path.open("a") as fh:
        fh.write("\t".join(list(key) + [f"{v:.6f}" for v in vals]) + "\n")


def key_of(regime, mech, metric, opt_kind, driver, arm, seed):
    return (regime, mech, metric, opt_kind, driver, arm, str(seed))


def _vals(r):
    c = r["cost"].as_row()
    return ((r["acc"], r["forget"], r["probe"], r["mags"]["h0"], r["mags"]["h1"], r["mags"]["out"],
             r["msd"], float(r["n_miss"])) + tuple(float(c[k]) for k in COST))


def run_cell(regime, mech, metric, opt_kind, driver, arm, ledger, seed=SEED):
    key = key_of(regime, mech, metric, opt_kind, driver, arm, seed)
    if key in ledger:
        print(f"[skip] {'|'.join(key)} acc={ledger[key][0]:.4f}", flush=True)
        return ledger[key]
    buf = io.StringIO()
    with redirect_stdout(buf):
        if arm == "base":
            r = run_baseline(metric, opt_kind, driver, regime=regime, mech=mech)
        else:
            r = run(mech, metric, opt_kind, driver, dead=(arm == "dead"), seed=seed, regime=regime)
    vals = _vals(r)
    append(key, vals); ledger[key] = vals
    warn = r["cost"].warn_if_confounded(label="|".join(key))
    print(f"[run ] {'|'.join(key)} acc={r['acc']:.4f} forget={r['forget']:.4f} "
          f"probe={r['probe']:.3f} |g|={r['mags']['h0']:.4f}/{r['mags']['h1']:.4f}/"
          f"{r['mags']['out']:.4f} msd={r['msd']:.2e} n_miss={r['n_miss']} "
          f"ratio={r['cost'].param_ratio:.4f}", flush=True)
    if warn:
        print(f"       WARN {warn}", flush=True)
    return vals


# ============================================================================== stages
def cells(part, mechs, metrics=("classil", "taskil"), regimes=tuple(BUFFERS)):
    """Cells for this shard. `metrics` exists purely to BALANCE the split: `gain` is the only
    mechanism with two regimes, so sharding on mech alone makes it a 30-cell shard against 15
    elsewhere and it becomes the critical path."""
    out = []
    regs = [r for r in REGIMES if r["mech"] in mechs and r["metric"] in metrics]
    for g in regimes:
        if part in ("base", "grid", "all"):
            for r in regs:
                out.append((g, r["mech"], r["metric"], r["opt"], r["base"], "base"))
        if part in ("grid", "all"):
            for r in regs:
                for d in DRIVERS:
                    out.append((g, r["mech"], r["metric"], r["opt"], d, "live"))
        if part in ("dead", "grid", "all"):
            for r in regs:
                out.append((g, r["mech"], r["metric"], r["opt"], "state01", "dead"))
    return out


def migrate(ledger):
    """Carry the v1 (normal-regime, no-cost) rows into the v2 schema WITHOUT re-running them.

    Cost is DECLARED, not measured -- it is a function of the mechanism, the driver, the gate shape
    and the buffer capacity, none of which depend on the run. So the 120 v1 rows can be completed
    exactly rather than recomputed, which is the whole reason a schema widening does not cost 80
    minutes. The gate is built (cheaply, on CPU-side param counts only) purely to size it.
    """
    # v1 had SIX key columns (no `regime`), so it must be parsed at its own width -- load_ledger()
    # splits at the v2 width and would silently fold a metric into the key.
    if not TSV_V1.exists():
        print(f"nothing to migrate: {TSV_V1} not found"); return
    old = {}
    for line in TSV_V1.read_text().splitlines()[1:]:
        if line.strip():
            f = line.split("\t")
            old[tuple(f[:6])] = tuple(float(v) for v in f[6:])
    n = 0
    for k, v in old.items():
        mech, metric, op, driver, arm, seed = k
        nk = key_of("normal", mech, metric, op, driver, arm, seed)
        if nk in ledger:
            continue
        net = p7.Net()
        if arm == "base":
            gate, cap = None, (BUFFERS["normal"] if driver == "er" else 0)
        else:
            drv = make_driver(driver, lr=1.0)
            with rng_frozen(), torch.no_grad():
                drv.value(net.to(DEV), torch.zeros(8, 784, device=DEV), update=False)
            gate, _ = build_gate(mech, drv.K, 1e-3)
            cap = BUFFERS["normal"]
        c = cell_cost(mech, driver, gate, net, cap)
        vals = tuple(v) + tuple(float(c.as_row()[x]) for x in COST)
        append(nk, vals); ledger[nk] = vals; n += 1
    print(f"migrated {n} v1 rows -> {TSV.name} (regime=normal, cost columns filled)")


def rfree_note():
    """Rule #12's rehearsal-free REPORT for this direction: a labelled degeneracy, not a number."""
    print("""
================================================================================================
REHEARSAL-FREE (rule #12): reported as a STRUCTURAL DEGENERACY, deliberately not run as cells
================================================================================================
Rule #12 is explicit that `rfree` is a CATEGORY, not an ablation: it means reporting methods that
do NOT REQUIRE rehearsal. Setting a replay method's buffer to 0 makes it naive, and setting a
replay-trained gate's buffer to 0 makes it structurally inert -- neither is a rehearsal-free
RESULT, and a chance-level number from one would be reported as if the mechanism had failed when
in fact it cannot exist.

For all four training-time-only mechanisms here the degeneracy is provable without running it:

  wdecay    the boundary meta-loss draws its batch from the buffer. At cap 0, `buf.sample_any`
            returns None, `_boundary_decay` breaks immediately, P never leaves zero, and
            f = exp(mbar @ 0) = 1 EXACTLY. The decay is the identity.
            (position_paper/wd_modulation measured this: every cell 0.1976 = naive, |f-1| = 0.0000.)
  plast /   the lookahead meta-loss is computed on the replay-augmented batch. With no replay it
  selplast  sees only the current task, so it carries NO retention signal -- pt5's iter-3
            follow-up 2 is the measurement of exactly that gap.
  lossmod   with one task per batch, exactly one c_T is nonzero and the per-class logit adjustment
            collapses to a scalar rescale of the loss: an lr knob with nothing task-differentiated
            in it. (position_paper/loss_modulation reports the same degeneracy.)

The four FORWARD mechanisms are not rehearsal-free either -- their baseline is ER, and at cap 0
that baseline is naive, so the comparison is against a different method rather than a different
budget.

A genuine rehearsal-free row for this direction would need a rehearsal-free BASE (EWC/SI/MAS/LwF,
DGR at ~91%), which is a different study, not a buffer setting on this one.
""")


def report(ledger, regimes=tuple(BUFFERS)):
    print("\n" + "=" * 128)
    print("LEARNING-STATE DRIVERS | 1 seed | neuron | learned P | live (+ state01 dead) | "
          "no new tuning | regimes: " + ", ".join(f"{g} (buffer {BUFFERS[g]})" for g in regimes))
    print("=" * 128)
    for g in regimes:
        for r in REGIMES:
            mech, metric, op = r["mech"], r["metric"], r["opt"]
            lr, ep, nlr = point(r)
            base = ledger.get(key_of(g, mech, metric, op, r["base"], "base", SEED))
            dead = ledger.get(key_of(g, mech, metric, op, "state01", "dead", SEED))
            if not base:
                continue
            ref, dref = base[0], (dead[0] if dead else base[0])
            print(f"\n--- [{g}] {mech} | {metric} | {op} | base={r['base']} lr={lr:g} ep={ep} "
                  f"nlr={nlr:g} buffer={BUFFERS[g]}")
            print(f"    baseline acc={ref:.4f}" +
                  (f"   state01-dead={dead[0]:.4f} (d={dead[0] - ref:+.4f})" if dead else ""))
            print(f"    {'driver':18s} {'acc':>8s} {'d-base':>8s} {'d-dead':>8s} {'forget':>7s} "
                  f"{'probe':>6s} {'|g| h0/h1/out':>22s} {'msd':>9s} {'miss':>5s} {'p_ratio':>8s}"
                  f"  class")
            for d in DRIVERS:
                v = ledger.get(key_of(g, mech, metric, op, d, "live", SEED))
                if not v:
                    continue
                cls = "tonic" if d in TONIC else "per-sample"
                if mech in FORWARD and d in TONIC:
                    cls += ", EVAL-DEGEN"
                pr = v[len(METRICS) + COST.index("param_ratio")]
                print(f"    {d:18s} {v[0]:8.4f} {v[0] - ref:+8.4f} {v[0] - dref:+8.4f} {v[1]:7.4f} "
                      f"{v[2]:6.3f} {v[3]:6.4f}/{v[4]:6.4f}/{v[5]:6.4f} {v[6]:9.2e} "
                      f"{int(v[7]):5d} {pr:8.4f}  {cls}")
    print(f"\nNOISE FLOOR {NOISE_FLOOR} at 1 seed (MPS): treat |d| below it as null.")


def cross_regime(ledger):
    """normal -> budget, per cell. THE question the budget regime exists to answer.

    `position_paper/wd_modulation` found the content-free control explains the whole effect at
    buffer 1000 and STOPS explaining it at buffer 200 (taskid +0.0233, vecproj +0.0319 over `const`,
    while the entropy/difficulty drivers stayed null) -- i.e. the drivers that carry TASK information
    separated from a constant only under memory pressure. A null at buffer 1000 is therefore not
    evidence about the budget regime, which is why this table exists.
    """
    print("\n" + "=" * 112)
    print("NORMAL -> BUDGET   (d = driver - its own regime's reference; ref = dead for selplast)")
    print("=" * 112)
    for r in REGIMES:
        mech, metric, op = r["mech"], r["metric"], r["opt"]
        rows = {}
        for g in BUFFERS:
            b = ledger.get(key_of(g, mech, metric, op, r["base"], "base", SEED))
            dd = ledger.get(key_of(g, mech, metric, op, "state01", "dead", SEED))
            if b:
                rows[g] = (b[0], (dd[0] if (dd and mech == "selplast") else b[0]))
        if len(rows) < 2:
            continue
        print(f"\n--- {mech} | {metric} | {op}    base normal={rows['normal'][0]:.4f} "
              f"budget={rows['budget'][0]:.4f}  (ER falls {rows['budget'][0] - rows['normal'][0]:+.4f})")
        print(f"    {'driver':18s} {'d normal':>10s} {'d budget':>10s} {'change':>9s}")
        for d in DRIVERS:
            vn = ledger.get(key_of("normal", mech, metric, op, d, "live", SEED))
            vb = ledger.get(key_of("budget", mech, metric, op, d, "live", SEED))
            if not (vn and vb):
                continue
            dn, db = vn[0] - rows["normal"][1], vb[0] - rows["budget"][1]
            flag = ""
            if vn[0] < 0.2 or vb[0] < 0.2:
                flag = "  DIVERGED"
            elif abs(db) > NOISE_FLOOR and abs(dn) <= NOISE_FLOOR:
                flag = "  <- emerges under pressure"
            print(f"    {d:18s} {dn:+10.4f} {db:+10.4f} {db - dn:+9.4f}{flag}")


def anchor(ledger):
    """The two claims this harness must make about itself before any cell is readable.

    1. Each regime's plain baseline lands where the project's frozen work puts it. These are NOT
       bit-exact anchors -- this is a new loop, not a copy-forward of one study's inner loop -- so
       they are checked as a BAND, at the 0.007 1-seed noise floor.
    2. `state01`'s DEAD gate equals that baseline. This bank builds no parameters and draws no RNG,
       so rule #10's shift should be absent; if it is, live-only cells are readable against the
       plain baseline, which is the whole reason the 8 dead cells are in scope.
    """
    want = {("gain", "classil", "adam"): ("er", 0.8975, "pt7_tuned_syn tuned ER-adam seed42 0.8975"),
            ("plast", "taskil", "sgd"): ("er", 0.994133, "plast_drivers_results.tsv seed42 0.994133"),
            ("wdecay", "classil", "sgd"): ("er", 0.9034, "pt7_tuned_syn tuned ER-SGD 0.9034")}
    print("\n" + "=" * 96); print("ANCHOR (normal regime)"); print("=" * 96)
    for (mech, metric, op), (base, ref, note) in want.items():
        v = run_cell("normal", mech, metric, op, base, "base", ledger)
        d = v[0] - ref
        print(f"  {mech:9s} {metric:8s} {op:5s} base={v[0]:.6f} vs {ref:.6f} ({d:+.6f}) "
              f"{'OK' if abs(d) < 0.02 else 'CHECK'}   [{note}]")
    print("\n  dead-gate check (state01)   [selplast is EXPECTED to differ -- see below]")
    for g in BUFFERS:
        for r in REGIMES:
            b = run_cell(g, r["mech"], r["metric"], r["opt"], r["base"], "base", ledger)
            dd = run_cell(g, r["mech"], r["metric"], r["opt"], "state01", "dead", ledger)
            d = dd[0] - b[0]
            if r["mech"] == "selplast":
                verdict = "HANDICAPPED (by design)"
            elif abs(d) < 1e-6:
                verdict = "RNG-MATCHED (bit-exact)"
            elif abs(d) < NOISE_FLOOR:
                verdict = f"fp drift only, {NOISE_FLOOR / abs(d):.0f}x below the noise floor"
            else:
                verdict = "SHIFTED -- live-only cells NOT readable"
            print(f"  [{g:6s}] {r['mech']:9s} {r['metric']:8s} {r['opt']:5s} dead={dd[0]:.6f} "
                  f"base={b[0]:.6f} d={d:+.6f} {verdict}")
    print("""
  selplast's dead gate is NOT the plain baseline and must not be read as one. Its STE offset
  b ~ N(0, 0.5^2) freezes ~half the elements at alpha = 0 from step one, and that offset is present
  whether or not P learns -- so the control carries the mechanism's own initialisation handicap.
  That is the RIGHT control for `d-dead` (matched handicap, matched RNG) and the WRONG one for
  reading absolute accuracy. `plast_binary` measured this exactly: its neuron `d-dead` of +0.0119,
  positive in all 3 seeds, was the live gate climbing back TO naive from a handicapped control,
  not past it -- vs the plain baseline the same cell was -0.0008. Always read both columns.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="grid",
                    choices=["anchor", "base", "dead", "grid", "all", "report", "migrate",
                             "rfree", "cross"])
    ap.add_argument("--mechs", default=",".join(MECHS))
    ap.add_argument("--metrics", default="classil,taskil")
    ap.add_argument("--regimes", default=",".join(BUFFERS))
    ap.add_argument("--drivers", default=",".join(DRIVERS))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    ledger = load_ledger()
    regimes = tuple(a.regimes.split(","))
    if a.part == "report":
        return report(ledger, regimes)
    if a.part == "cross":
        return cross_regime(ledger)
    if a.part == "rfree":
        return rfree_note()
    if a.part == "migrate":
        return migrate(ledger)
    if a.part == "anchor":
        return anchor(ledger)
    keep = set(a.drivers.split(","))
    todo = [c for c in cells(a.part, a.mechs.split(","), tuple(a.metrics.split(",")), regimes)
            if c[5] != "live" or c[4] in keep]
    print(f"learning-state drivers | {len(todo)} cells | seed {SEED} | gran {GRAN} | dev {DEV} | "
          f"regimes {regimes}", flush=True)
    for g, mech, metric, op, drv, arm in todo:
        run_cell(g, mech, metric, op, drv, arm, ledger)
    report(ledger, regimes)


if __name__ == "__main__":
    main()
