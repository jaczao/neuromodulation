"""MODULATED WEIGHT DECAY — THESIS-PLAN direction B, position-paper mechanism 2.

    w_{t+1} = f_w(s_t) . (w_t - sigma . grad_w L)

A neuromodulator-driven multiplicative factor applied to the weight AFTER the SGD step, rather than
to the gradient before it (that is mechanism 3, selective plasticity, already run and rejected in
`pt5_taskil/`). Both metrics (class-IL and task-IL), both arms, three granularities, four drivers,
and TWO SCHEDULES — the paper's per-step form and a per-task-boundary variant of our own.

WHY THIS IS THE RIGHT NEXT CELL, AND WHAT WOULD MAKE IT DIFFERENT FROM PLASTICITY. Everything this
project has rejected shares one shape: a MULTIPLICATIVE gate that the thing it multiplies can
absorb. A jointly-trained forward gain is reabsorbed by the weights (the h1-gate sat at mean 0.281
and landed exactly on ER); a gradient gate collapses to a global LR knob and dissolves once the main
lr is tuned. Decay is the one member of the family that is NOT obviously absorbable: it is applied
OUTSIDE the loss, so no weight update can cancel it, and repeated application COMPOUNDS (a weight
touched by T tasks is multiplied T times). Whether that makes it a retention lever or just a
learned-away nuisance is the question. The prior is unkind — see the pre-registered predictions
below — but the mechanism is structurally distinct from the ones already closed, which is the bar
for running it at all.

THE GATE IS DELIBERATELY IDENTICAL TO pt7's PLASTICITY GATE. `DecayGate` subclasses
`pt7_plast_tempslope.PlastGate` and changes NOTHING: f_w = exp(mbar @ P), P zero-init so f = 1 is
parity, exp keeping f > 0 and able to grow OR shrink a weight. Same projection shapes, same
granularities, same meta-optimizer. ONE LEVER CHANGES between this study and the plasticity studies
— where the multiplier lands — so any difference is attributable. Reusing the gate is the point,
not a shortcut.

TRAINING THE GATE. The decay multiplies weights in place under `no_grad`, so the main loss gives P
no gradient (exactly the situation CLAUDE.md records for the plasticity target). P is trained by the
pt5 LOOKAHEAD meta-loss: W_fast = f . (W.detach() - lr . g) with g detached, meta-CE on a batch that
CONTAINS REPLAY (so the meta-loss carries a retention signal), Adam on P only, then the real step
commits with the detached f. This is a copy-forward of the plasticity loop with one line moved.

ARMS (each at its OWN val-tuned main lr — rule: arms sit at different operating points, never
compare arm A to arm B, only mechanism to its own control):
  erown   main net AND gate trained on the ER batch (current + replay). No task ids in the loop.
  bufcur  standalone: the MAIN net steps on the current task only (naive), while the meta-loss gets
          the replay buffer — pt5's "modulator-only replay", i.e. replay reaches the GATE, never the
          backbone. This is the arm where a retention lever would have room to show.

REGIMES (rule #12). The full grid runs at `normal` (buffer 1000). `--part regimes` re-runs the
headline cells at `budget` (buffer 200) and `rfree` (buffer 0 — no buffer anywhere, which for
`bufcur` also removes the meta-replay and leaves the meta-loss with a current-batch-only signal).
Cost columns are recorded for every cell, so "reaches X at zero stored bytes" is a statement the
ledger can make.

SCHEDULES. `step` is the paper's literal form, f applied after every update. `boundary` is this
project's variant, f applied once per task over that task's mean driver — see `_apply_boundary` for
why it exists and what it deviates from. They are reported as separate arms, never merged.

DRIVERS (scoped to four): taskid, ach, nerisez, vecproj. `drivers.py` documents the per-driver
standardization rule, exactly where the task-id oracle is and is not, and what dropping ach_ema /
vec_x / all5 costs. NOTE `vecproj` x synapse is a (32, 477600) ~ 15.3M-param projection, ~32x the
478k backbone — pt7_capacity's rule makes any positive result there a capacity confound, so
`--part report` prints the ratio rather than leaving it to be noticed.

TUNING. neuro_lr is swept on VALIDATION per driver, over a grid that depends on the schedule. The
study was specified with "tune taskid/ach/all5, reuse for the rest", and that became unserviceable
once the single drivers went RAW: `ach` is K=1 and `vecproj` is K=32, so one neuro_lr does not
transfer, and the stability probe found the usable range differs by decades across drivers and is
NON-MONOTONE within one (`nerisez` diverged at 1e-5 but not at 1e-3). There is no stability
threshold to normalise against, hence a per-driver sweep. Selection excludes diverged cells.
Main lr is NOT swept: every (metric, arm) point is already val-selected in `neurocore.tuned`
(class-IL from pt7_tuned_syn / pt3_retry, task-IL from pt5_taskil/plast_taskil). Rule #11 is what
makes that non-negotiable — an untuned main lr is what manufactured pt7's fake +0.11 global-LR "win".

DEAD-GATE CONTROL (rule #10), per DRIVER: the same config at neuro_lr = 0, so P stays at zero, f == 1
exactly, and the run is the plain baseline plus whatever RNG the modulator's construction consumed.
It is per-driver rather than per-granularity because `Heads(K)` init consumes RNG proportional to K
while `DecayGate` is all zeros — a claim `--part deadcheck` verifies empirically rather than assumes.

PRE-REGISTERED PREDICTIONS (written before running, so the writeup cannot be retrofitted):
  1. `global` will match `neuron` will match `synapse` for any driver with a nonzero-mean signal —
     if it does, the mechanism is a global weight-norm knob, not allocation (the pt7 SET-1 result).
  2. (NOW UNTESTABLE — ach_ema was dropped for budget.) A tonic driver would produce the LARGEST
     |f-1| and no accuracy gain, because a sustained-positive driver is what a compounding decay
     integrates fastest. Recorded so a follow-up knows the prediction was made, not retrofitted.
  3. The probe will not predict benefit: `plast_drivers` found the most task-decodable driver
     (vec_x, probe 0.934) was its worst cell. If a probe/benefit relation appears HERE it is a real
     difference between gating a weight and gating a gradient, and is the finding.
  4. `bufcur` is where a decay lever could beat its control; `erown` has ~no headroom in task-IL
     (ER is already 0.9941 in this harness, and `trajectories.py` showed perfect retention tops out
     at 0.9845 vs ER 0.9946 — state the ceiling rather than claiming a strong null).

ANCHORS (`--part anchor`): the ungated baselines of THIS harness must reproduce the frozen ledgers —
task-IL ER-SGD 0.994133 (pt5_taskil/plast_drivers_results.tsv, seed 42) and class-IL ER-SGD 0.9034
(pt7_tuned_syn). They are not a substitute for the dead control (they are not RNG-matched — that is
rule #10's whole point) but they tie this loop to known numbers before any mechanism is read.

Run (see CLAUDE.md for the parallel form):
    uv run python -m neurocore.shard --script position_paper/wd_modulation.py \
        --ledger position_paper/wd_modulation_results.tsv \
        --split driver=taskid,ach,nerisez,vecproj --split gran=global,neuron,synapse \
        --args "--part tune --resume" --workers 8 --device cpu

Shard on gran as well as driver: the synapse cells carry a far bigger gate, so spreading them
across workers beats letting one shard own all of them.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

import pt7_neuromodulators as p7                                   # noqa: E402  (frozen, read-only)
import pt7_plast_tempslope as pts                                  # noqa: E402  (frozen, read-only)
from neurocore import shard                                        # noqa: E402
from neurocore.cost import Cost, count_params        # noqa: E402
from neurocore.ledger import NOISE_FLOOR, Ledger, delta_table, where  # noqa: E402
from neurocore.tuned import tuned_main                             # noqa: E402
from position_paper import drivers as D                            # noqa: E402
from prototype.data import SplitMNIST, make_sequence               # noqa: E402

DEV = p7.DEV
PROBLEM, OPT = "splitmnist", "sgd"

TSV = shard.ledger_path(Path(__file__).resolve().parent / "wd_modulation_results.tsv")
KEYS = ["regime", "schedule", "metric", "arm", "driver", "gran", "nlr", "seed", "split"]
METRICS = ["acc", "forget", "probe", "g_h0", "g_h1", "g_out", "acc_live"]

# The four drivers this study runs (scoped down from seven for budget). What the dropped three
# would have added, so a follow-up knows what is missing rather than rediscovering it:
#   ach_ema  the only TONIC driver (within-batch variance ~0 => a near-constant gate). It is the
#            sharpest test of prediction 2, since a COMPOUNDING decay integrates a sustained-positive
#            signal fastest. NOTE it is no longer a check on the standardization rule — that check
#            was `ach` (standardized) vs `ach_ema` (raw), and it disappeared when the single drivers
#            all went raw, not when `ach_ema` was dropped.
#   vec_x    the un-projected 784-d novelty vector; `vecproj` is its tractable form.
#   all5     the only composite, so "does stacking drivers help" is unanswered here.
DRIVERS = ("taskid", "ach", "nerisez", "vecproj")
# `const` is a CONTROL, not a driver: it is content-free, so it isolates "the driver said
# something" from "the gate had per-parameter freedom trained on replay". Run explicitly via
# --driver const, and read it beside any cell that beats its dead control.
CONTROLS = ("const",)

GRANS = ("global", "neuron", "synapse")
ARMS = ("erown", "bufcur")
EVAL_METRICS = ("classil", "taskil")
SEEDS = (42, 43, 44)
TUNE_SEED = 42
BUFFERS = {"normal": 1000, "budget": 200, "rfree": 0}

# WHEN the decay fires. `step` is the position paper's literal form; `boundary` is this project's
# variant. See `_apply_boundary` for why the second exists — in one line, the meta-loss is one step
# deep while `step` applies f ~4750 times, so the gate is chosen on a criterion three orders of
# magnitude away from what it does.
SCHEDULES = ("step", "boundary")
META_STEPS = 50          # meta-updates per boundary; `step` gets one per batch and needs no analogue

# neuro_lr grid, WIDE and per driver. The original 4-point grid at 1e-4..1e-1 was measured to be
# entirely inside the divergent region, and the stability probe found the usable range differs by
# decades ACROSS drivers (raw drivers, so a K=784 driver is ~350x more sensitive per unit P than a
# K=1 one) and is NON-MONOTONE within a driver (`nerisez` diverged at 1e-5 but not at 1e-3). There is
# therefore no stability threshold to normalise against, which is why every driver gets its own
# sweep rather than inheriting one.
NEURO_GRID = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
# Above this, a reported |f-1| is not "engagement" and must not win a tie-break. Two reasons, and
# the second is the one that bit: (a) a decay factor 10x from parity is already extreme; (b) the
# g_* columns are the gate RECOMPUTED AT EVAL, not the gate that was applied during training, and
# the raw novelty drivers take much larger values on the test stream than in training — so
# `vecproj`/boundary logged |f-1| up to 3.8e12 while its accuracy sat at baseline. That number is
# real but it describes an out-of-distribution driver reading, not the training-time operation.
# For a DECAY target the gate never enters the forward, so an eval-time gate is diagnostic only.
ENGAGE_CAP = 10.0
# ...and the two schedules need DIFFERENT grids, which is itself evidence for the compounding
# diagnosis. `step` applies f ~4750 times and is already diverging at 1e-4, so its grid runs low.
# `boundary` applies it 5 times, is stable out to 1e-1, and is INERT below 1e-3 — measured:
# |f-1| at nlr 1e-3 / 1e-1 = taskid 0.021 / 1.276, ach 0.012 / 0.283, vecproj 0.081 / 14.3.
# Sharing one grid would test the two arms at incomparable engagement levels.
GRID_BY_SCHEDULE = {"step": NEURO_GRID, "boundary": (1e-4, 1e-3, 1e-2, 1e-1)}

VAL_SEQ = make_sequence(7)          # rule #1: tuning order, never the reporting order
VAL_FRAC = 0.1

# The base whose tuned main-lr point each arm sits at (rule: never reuse another arm's grid).
BASE_OF = {("classil", "erown"): "er", ("classil", "bufcur"): "naive_masked",
           ("taskil", "erown"): "er", ("taskil", "bufcur"): "naive"}

# Ungated baselines of THIS harness vs the frozen ledgers they must reproduce (seed 42).
ANCHORS = {("taskil", "er"): 0.994133,      # pt5_taskil/plast_drivers_results.tsv
           ("classil", "er"): 0.9034}       # pt7_tuned_syn (3-seed mean 0.9029 +- 0.0042)


class DecayGate(pts.PlastGate):
    """f_w = exp(mbar @ P), applied to the WEIGHT after the step instead of to the gradient before.

    Subclasses pt7's plasticity gate and overrides nothing: identical parameterisation, identical
    zero-init parity (f = 1), identical granularity shapes. Named separately so a traceback says
    which lever is under test, and so the semantics are documented where they differ:

      PLASTICITY  w <- w - lr . (alpha . g)    alpha rescales a STEP; a uniform alpha is an lr knob
      DECAY       w <- f . (w - lr . g)        f rescales the WEIGHT; a uniform f is a norm knob and
                                               COMPOUNDS over steps (f^n after n applications)

    The compounding is why this is not merely plasticity in disguise, and also why it is dangerous:
    a driver with a small sustained negative mean shrinks every weight toward zero over ~4750 steps.
    `mult()` returns multiplier 1 for biases at `synapse` granularity (biases undecayed, matching the
    plasticity convention) and the per-neuron factor for biases at `neuron`.
    """


def _params(net):
    return pts._net_params(net)


def _acc(net, loader, allowed=None):
    net.eval()
    c = tot = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            logits = net.plain(x)[0]
            if allowed is not None:
                logits = _mask(logits, allowed)
            c += (logits.argmax(1) == y).sum().item(); tot += len(y)
    net.train()
    return c / tot


def _mask(logits, allowed):
    m = torch.full_like(logits, float("-inf"))
    m[:, list(allowed)] = 0.0
    return logits + m


# ==================================================================================== training loop
def run(driver_name, gran, seed, arm, metric, neuro_lr, main_lr, epochs, buffer, split="test",
        tasks=5, schedule="step"):
    """One cell. `driver_name` in {"er", "naive"} runs the ungated baseline of this harness.

    `tasks < 5` truncates the sequence — used ONLY by the stability probe, which needs to know
    whether the compounding gate diverges, not what the final accuracy is. A truncated run is not a
    result and is never ledgered.

    `schedule` selects WHEN the decay fires: `step` (the paper's per-update form) or `boundary`
    (once per task, over the task's mean driver). See `_apply_boundary`.
    """
    p7.seed_all(seed)
    seq = VAL_SEQ if split == "val" else p7.SEQ
    ds = SplitMNIST(sequence=seq, val_frac=VAL_FRAC if split == "val" else 0.0)
    loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    evals = [ds.get_task_val_loader(t, 64) if split == "val" else loaders[t][1] for t in range(5)]

    net = p7.Net().to(DEV)
    plain = driver_name in ("er", "naive")
    drv = None if plain else D.make_driver(driver_name, neuro_lr)
    gate = None if plain else DecayGate(gran, drv.K, neuro_lr)
    buf = p7.Reservoir(buffer) if buffer > 0 else None

    taskil = metric == "taskil"
    # class-IL ER trains on the plain 10-way CE; the standalone arm uses the masked loss, which is
    # the convention its tuned point (`naive_masked`) was selected under. task-IL masks in both.
    masked = taskil or arm == "bufcur" or driver_name == "naive"
    loss_fn = p7.masked_ce if masked else (lambda lo, yy: p7.CE(lo, yy))
    mrun = _RunMean()                     # task-pooled driver, used by the `boundary` schedule only
    applied = []                          # per-boundary |f-1| ACTUALLY applied (diagnostic only)
    A = np.full((5, 5), np.nan)

    for t in range(tasks):
        if drv is not None:
            drv.set_task(t)
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                xf = x.view(x.size(0), -1)
                rep = buf.sample_any(64) if buf is not None else None
                if rep is not None:
                    Xr, Yr = rep[0].to(DEV), rep[1].to(DEV)
                    Xmix, Ymix = torch.cat([xf, Xr]), torch.cat([y, Yr])
                else:
                    Xmix, Ymix = xf, y
                # erown: the main net sees replay too. bufcur: only the META-loss does.
                Xmain, Ymain = (Xmix, Ymix) if arm == "erown" else (xf, y)
                params = _params(net)

                if plain:
                    g = torch.autograd.grad(loss_fn(net.plain(Xmain)[0], Ymain), params)
                    with torch.no_grad():
                        for i in range(pts.NPARAMS):
                            params[i].add_(g[i], alpha=-main_lr)
                    if buf is not None:
                        buf.add(x, y)
                    continue

                mbar = drv.value(net, Xmain).mean(0)                # (K,) detached, batch-pooled
                g = torch.autograd.grad(loss_fn(net.plain(Xmain)[0], Ymain), params)
                if schedule == "step":
                    f, _ = gate.mult(mbar)                          # f differentiable in P
                    # THE MECHANISM: decay lands on the post-step weight, not on the gradient.
                    Wf = [f[i] * (params[i].detach() - main_lr * g[i]) for i in range(pts.NPARAMS)]
                    meta = loss_fn(pts._fwd_fast(Wf, Xmix), Ymix)   # replay in Xmix = retention signal
                    gate.opt.zero_grad(); meta.backward(); gate.opt.step()
                    with torch.no_grad():                           # real step, detached f
                        for i in range(pts.NPARAMS):
                            params[i].add_(g[i], alpha=-main_lr)
                            params[i].mul_(f[i].detach())
                else:                                               # boundary: ordinary step now
                    with torch.no_grad():
                        for i in range(pts.NPARAMS):
                            params[i].add_(g[i], alpha=-main_lr)
                    mrun.update(mbar)                               # pool the driver over the task
                if buf is not None:
                    buf.add(x, y)
                drv.train_head(net, Xmain, Ymain)

        if not plain and schedule == "boundary":
            applied.append(_apply_boundary(net, gate, buf, mrun.value(), loss_fn))
            mrun.reset()

        with D.rng_frozen():                  # else the eval's DataLoader iterators shift training
            for i in range(t + 1):            # A-matrix: no driver call => live stats cannot leak
                A[t, i] = _acc(net, evals[i], allowed=seq[i] if taskil else None)

    last = tasks - 1
    acc = float(np.nanmean(A[last, :last + 1]))
    forget = float(np.mean([max([A[k, i] for k in range(i, tasks)]) - A[last, i]
                            for i in range(tasks)]))
    cost = _cost(net, drv, gate, buf, plain)
    app = ({k: float(np.mean([a[k] for a in applied])) for k in ("h0", "h1", "out")}
           if applied else {k: float("nan") for k in ("h0", "h1", "out")})
    if plain:
        return dict(acc=acc, forget=forget, A=A, probe=float("nan"),
                    per_layer={k: 0.0 for k in ("h0", "h1", "out")}, acc_live=acc, cost=cost,
                    applied=app)
    st = drv.state()                          # FROZEN first (leaves driver state untouched), then LIVE
    dg = _diag(net, drv, gate, evals[:tasks], seq, update=False, taskil=taskil)
    drv.restore(st)
    dl = _diag(net, drv, gate, evals[:tasks], seq, update=True, taskil=taskil)
    drv.restore(st)
    return dict(acc=acc, forget=forget, A=A, probe=dg["probe"], per_layer=dg["per_layer"],
                acc_live=dl["pred"], cost=cost, applied=app)


class _RunMean:
    """Running mean of the driver over a task. Reset at each boundary."""

    def __init__(self):
        self.s = None; self.n = 0

    def update(self, v):
        self.s = v.clone() if self.s is None else self.s + v
        self.n += 1

    def value(self):
        return self.s / max(self.n, 1)

    def reset(self):
        self.s = None; self.n = 0


def _apply_boundary(net, gate, buf, mbar, loss_fn, meta_steps=META_STEPS):
    """Fire the decay ONCE per task, and train P on exactly the operation it performs.

    WHY THIS ARM EXISTS. `step` applies f after every update — ~4750 times — so what actually hits a
    weight is f^4750, while the lookahead meta-loss is ONE step deep and cannot see the other 4749
    applications. A per-step f = 1.001 is a 115x blow-up. That mismatch (not a step-size problem) is
    what the stability probe measured as a NON-MONOTONE divergence: a positive feedback loop where
    weights grow, the driver moves, and the gate pushes further. Firing once per boundary makes f
    itself the operative factor, so the meta-loss optimises the quantity that acts.

    It also fits the neuromodulatory story better: a PHASIC signal at a context switch that
    consolidates, rather than a tonic factor bleeding into every update.

    THE HONEST LIMITATION, stated because it is structural and not a tuning matter. The meta-loss
    here is retention only — CE of (f . W) on the replay buffer — and decay can only ever REDUCE fit
    to what is already learned, so its gradient pushes f toward 1 and f = 1 is close to a fixed
    point. A gate that shrinks only unimportant weights would pay almost nothing, but the benefit of
    having done so accrues to the NEXT task, which this meta-loss never sees. So an inert result
    here is `f -> 1 by construction`, NOT evidence that selective decay cannot retain. Making it a
    real objective needs a decay PRESSURE term (an explicit weight-norm penalty on f.W, so the gate
    must choose WHERE to shrink) — the same gap `plast_binary` recorded for selective plasticity,
    whose meta-loss only ever rewarded learning the current batch faster. Left unimplemented rather
    than added silently: it is a new axis, and the `f -> 1` check below tells us whether it is needed.

    NOTE this is a deviation from the position paper, which writes the per-step form. It is reported
    as a separate `schedule` arm, never folded into the paper's mechanism.
    """
    params = _params(net)
    if buf is not None:
        for _ in range(meta_steps):
            s = buf.sample_any(64)
            if s is None:
                break
            Xb, Yb = s[0].to(DEV), s[1].to(DEV)
            f, _ = gate.mult(mbar)
            Wf = [f[i] * params[i].detach() for i in range(pts.NPARAMS)]
            meta = loss_fn(pts._fwd_fast(Wf, Xb), Yb)
            gate.opt.zero_grad(); meta.backward(); gate.opt.step()
    with torch.no_grad():
        f, structs = gate.mult(mbar)
        for i in range(pts.NPARAMS):
            params[i].mul_(f[i].detach())
        # what was ACTUALLY applied, for the applied-vs-eval diagnostic. Read-only: no RNG, no
        # optimizer state, so collecting it cannot move the run off its trajectory.
        if gate.mech == "global":
            d = (structs[0] - 1).abs().item()
            return {"h0": d, "h1": d, "out": d}
        return {k: (a - 1).abs().mean().item()
                for k, a in zip(("h0", "h1", "out"), structs)}


@torch.no_grad()
def _diag(net, drv, gate, evals, seq, update, taskil):
    """|f-1| per layer + the task-decodability probe over the eval set.

    `update` is the requested "no freezes at inference": the driver's running stats / predictor state
    keep advancing on the test stream. For a DECAY target this cannot move accuracy by construction —
    the gate never enters the forward, so eval is the plain net — and `acc_live` is recorded as the
    CHECK of that rather than asserted.
    """
    net.eval()
    mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}; tot = 0; Ms, Ts = [], []
    for i in range(len(evals)):
        for x, y in evals[i]:
            x = x.to(DEV); b = x.size(0)
            if update:
                drv.live_update(net, x)
            m = drv.value(net, x.view(b, -1), update=update)
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
    pred = float(np.mean([_acc(net, evals[i], allowed=seq[i] if taskil else None)
                          for i in range(len(evals))]))
    net.train()
    return dict(pred=pred, probe=p7._probe(torch.cat(Ms), torch.cat(Ts), drv.K),
                per_layer={k: mags[k] / tot for k in mags})


def _cost(net, drv, gate, buf, plain):
    """Declared cost. Note `bwd_infer = 0` and `fwd_infer = 1`: the decay gate is a TRAINING-time
    mechanism, so inference is the plain backbone — the column where a TTA method would pay."""
    extra = 0 if plain else count_params(gate) + _driver_params(drv)
    return Cost(backbone_params=count_params(net), extra_params=extra,
                buffer_bytes=_buf_bytes(buf),
                # per batch: 1 main forward + 1 meta forward (+1 small head-target forward for head
                # drivers); 1 main grad + 1 meta backward. Baselines are 1/1.
                fwd_train=1.0 if plain else 3.0, bwd_train=1.0 if plain else 2.0,
                fwd_infer=1.0, bwd_infer=0.0)


def _buf_bytes(buf):
    """`neurocore.cost.buffer_bytes` wants an `nbytes()`, and pt7's `Reservoir` is FROZEN code that
    cannot grow one (rule #9). Sized here instead, from its RESIDENT allocation (cap-sized, not
    fill-sized) — that is what a memory-budgeted regime actually has to pay for."""
    if buf is None:
        return 0
    return buf.X.element_size() * buf.X.nelement() + buf.Y.element_size() * buf.Y.nelement()


def _driver_params(drv):
    mods = []
    for obj in (getattr(drv, "subs", None) or [drv]):
        for attr in ("heads", "drv"):
            m = getattr(obj, attr, None)
            if isinstance(m, torch.nn.Module):
                mods.append(m)
    return count_params(*mods) if mods else 0


# ========================================================================================== driving
def ledger():
    return Ledger(TSV, keys=KEYS, metrics=METRICS, with_cost=True)


def main_point(metric, arm):
    p = tuned_main(PROBLEM, metric, BASE_OF[(metric, arm)], OPT)
    return p["lr"], p["epochs_per_task"]


def run_cell(led, regime, metric, arm, driver, gran, nlr, seed, split="test", schedule="step"):
    key = dict(regime=regime, schedule=schedule, metric=metric, arm=arm, driver=driver, gran=gran,
               nlr=f"{nlr:g}", seed=seed, split=split)
    if led.is_done(**key):
        return led_get(led, key)
    lr, ep = main_point(metric, arm)
    r = run(driver, gran, seed, arm, metric, nlr, lr, ep, BUFFERS[regime], split=split,
            schedule=schedule)
    led.append(key, dict(acc=r["acc"], forget=r["forget"], probe=r["probe"],
                         g_h0=r["per_layer"]["h0"], g_h1=r["per_layer"]["h1"],
                         g_out=r["per_layer"]["out"], acc_live=r["acc_live"]), cost=r["cost"])
    print(f"  {regime:6s} {schedule:8s} {metric:7s} {arm:6s} {driver:9s} {gran:7s} "
          f"nlr={nlr:<7g} s{seed} "
          f"acc={r['acc']:.4f} forget={r['forget']:.4f} probe={r['probe']:.3f} "
          f"|f-1|={r['per_layer']['h0']:.4f}/{r['per_layer']['h1']:.4f}/{r['per_layer']['out']:.4f}",
          flush=True)
    return r["acc"]


def led_get(led, key):
    rows = where(led.rows(), **key)
    return float(rows[0]["acc"]) if rows else None


def cells_test(drivers, grans, metrics, arms, regime="normal"):
    out = []
    for metric in metrics:
        for arm in arms:
            for drv in drivers:
                for gran in grans:
                    if (drv, gran) in D.SKIP:
                        continue
                    out.append((regime, metric, arm, drv, gran))
    return out


def diverged(acc, g_h0, metric):
    """Did the compounding gate run away? Two tells, because it shows up both ways: a NON-FINITE
    gate magnitude, and an accuracy at or below chance (0.5 for the 2-way task-IL eval, 0.1 for the
    10-way class-IL one). A diverged cell is not a tuning candidate — it is a failure to report."""
    if not np.isfinite(g_h0) or not np.isfinite(acc):
        return True
    return acc <= (0.55 if metric == "taskil" else 0.15)


# ------------------------------------------------------------------ parts
def part_anchor(led):
    print("ANCHOR — ungated baselines of this harness vs the frozen ledgers\n", flush=True)
    for metric in EVAL_METRICS:
        acc = run_cell(led, "normal", metric, "erown", "er", "-", 0.0, 42)
        ref = ANCHORS[(metric, "er")]
        d = acc - ref
        flag = "OK" if abs(d) < 1e-6 else ("~noise" if abs(d) < NOISE_FLOOR else "MISMATCH")
        print(f"    {metric:8s} er  {acc:.6f} vs frozen {ref:.6f}  d={d:+.6f}  {flag}", flush=True)
    print("\n  class-IL's reference is a 3-seed mean from another harness, so ~noise is the pass "
          "condition there; task-IL's is a seed-42 cell and should match to 1e-6.", flush=True)


def part_baseline(led, metrics, arms):
    """Ungated baselines. Schedule-independent (no gate fires), so recorded once under `step`."""
    print("BASELINES — ungated, per (metric, arm), 3 seeds\n", flush=True)
    for metric in metrics:
        for arm in arms:
            base = "er" if arm == "erown" else "naive"
            for s in SEEDS:
                run_cell(led, "normal", metric, arm, base, "-", 0.0, s)


def part_deadcheck(led, drivers, schedules=("step",)):
    """Verify the dead control is GRANULARITY-independent instead of assuming it (plast_drivers'
    argument: DecayGate is all zeros and draws no RNG, while Heads(K) draws RNG proportional to K).

    It must also be SCHEDULE-independent, which is a stronger claim and worth checking: at
    neuro_lr = 0 the gate is exactly 1, so `boundary`'s extra meta-steps consume buffer samples via
    python's `random` — a stream disjoint from the torch generator the DataLoader shuffles on — and
    must not move the main net at all.
    """
    print("DEADCHECK — is the neuro_lr=0 control identical across granularities and schedules?\n",
          flush=True)
    for sched in schedules:
        for drv in drivers:
            accs = {}
            for gran in GRANS:
                if (drv, gran) in D.SKIP:
                    continue
                accs[gran] = run_cell(led, "normal", "taskil", "erown", drv, gran, 0.0, 42,
                                      schedule=sched)
            vals = [v for v in accs.values() if v is not None]
            spread = max(vals) - min(vals) if len(vals) > 1 else 0.0
            verdict = "gran-INDEPENDENT" if spread < 1e-9 else f"DIFFERS by {spread:.2e}"
            print(f"    {sched:9s} {drv:9s} {verdict}  "
                  f"{' '.join(f'{k}={v:.6f}' for k, v in accs.items())}", flush=True)


def part_tune(led, drivers, metrics, arms, schedules=("step",)):
    """neuro_lr on VALIDATION (make_sequence(7), val_frac 0.1), 1 seed. Never the test set.

    PER DRIVER, over a wide grid, because the reuse rule this study started with is unserviceable
    with raw drivers: `ach` is K=1 and `vecproj` is K=32, so one neuro_lr does not transfer. And
    selection is STABILITY-AWARE — a diverged cell is excluded from the argmax rather than being
    allowed to win or to drag the span, since a run that hit chance is a failure to report, not a
    tuning candidate.
    """
    print(f"TUNE neuro_lr on VAL — drivers {drivers}, schedules {schedules}, grids "
          f"{ {k: GRID_BY_SCHEDULE[k] for k in schedules} }\n", flush=True)
    for sched in schedules:
        for metric in metrics:
            for arm in arms:
                for drv in drivers:
                    for gran in GRANS:
                        if (drv, gran) in D.SKIP:
                            continue
                        _tune_one(led, sched, metric, arm, drv, gran)


def _tune_one(led, sched, metric, arm, drv, gran):
    scores, bad = {}, []
    grid = GRID_BY_SCHEDULE[sched]
    for nlr in grid:
        run_cell(led, "normal", metric, arm, drv, gran, nlr, TUNE_SEED, split="val",
                 schedule=sched)
        r = _val_row(led, sched, metric, arm, drv, gran, nlr)
        if r is None:
            continue
        if diverged(float(r["acc"]), float(r["g_h0"]), metric):
            bad.append(nlr)
        else:
            scores[nlr] = (float(r["acc"]), float(r["g_h0"]))
    if not scores:
        print(f"  >>> {sched:8s} {metric:7s} {arm:6s} {drv:9s} {gran:7s} ALL {len(bad)} CELLS "
              f"DIVERGED — no usable operating point", flush=True)
        return None
    best_acc = max(v[0] for v in scores.values())
    # Ties inside the noise floor go to MORE engagement, never less: neuro_lr = 0 is the dead gate,
    # so a "prefer the smaller value" tie-break walks the selection into the mechanism's own
    # off-switch and produces d-dead ~= 0 by construction rather than by measurement
    # (pt5_taskil/meta_schedule.md). That rule is right for a COST axis, wrong for this one.
    tied = {k: v for k, v in scores.items() if best_acc - v[0] <= NOISE_FLOOR}
    best = max(tied, key=lambda k: _engagement(tied[k][1]))
    span = best_acc - min(v[0] for v in scores.values())
    note = "  !! span < noise floor — axis UNRESOLVED at 1 seed" if span < NOISE_FLOOR else ""
    edge = "  !! at a GRID EDGE" if best in (grid[0], grid[-1]) else ""
    div = f"  [diverged: {','.join(f'{b:g}' for b in bad)}]" if bad else ""
    print(f"  >>> {sched:8s} {metric:7s} {arm:6s} {drv:9s} {gran:7s} neuro_lr={best:g} "
          f"(val {tied[best][0]:.4f}, |f-1| {tied[best][1]:.4f}, span {span:.4f}){note}{edge}{div}",
          flush=True)
    return best


def _val_row(led, sched, metric, arm, drv, gran, nlr):
    rows = where(led.rows(), schedule=sched, metric=metric, arm=arm, driver=drv, gran=gran,
                 nlr=f"{nlr:g}", seed=TUNE_SEED, split="val", regime="normal")
    return rows[0] if rows else None


def tuned_nlr(led, sched, metric, arm, driver, gran):
    """The tune stage's pick, re-derived from the ledger so it survives a sharded/resumed run."""
    rows = [r for r in where(led.rows(), schedule=sched, metric=metric, arm=arm, driver=driver,
                             gran=gran, split="val", seed=TUNE_SEED, regime="normal")
            if not diverged(float(r["acc"]), float(r["g_h0"]), metric)]
    if not rows:
        raise KeyError(f"no usable val rows for ({sched}, {metric}, {arm}, {driver}, {gran}) — "
                       f"run --part tune first, or every cell diverged")
    best_acc = max(float(r["acc"]) for r in rows)
    tied = [r for r in rows if best_acc - float(r["acc"]) <= NOISE_FLOOR]
    return float(max(tied, key=lambda r: _engagement(float(r["g_h0"])))["nlr"])


def _engagement(g):
    """Tie-break score for gate magnitude, capped. An absurd |f-1| is an eval-time driver blow-up
    (see ENGAGE_CAP), so it must not out-rank a genuinely engaged gate — but it must not rank BELOW
    an inert one either, since the cell is not inert. Capped, not discarded."""
    if not np.isfinite(g):
        return -1.0
    return min(g, ENGAGE_CAP)


def part_test(led, drivers, metrics, arms, schedules=("step",), regime="normal"):
    print("TEST — 3 seeds at the tuned neuro_lr, plus the per-driver dead control\n", flush=True)
    for sched in schedules:
        for regime_, metric, arm, drv, gran in cells_test(drivers, GRANS, metrics, arms, regime):
            try:
                nlr = tuned_nlr(led, sched, metric, arm, drv, gran)
            except KeyError as e:
                print(f"  SKIP {sched} {metric} {arm} {drv} {gran}: {e}", flush=True)
                continue
            for s in SEEDS:
                run_cell(led, regime_, metric, arm, drv, gran, nlr, s, schedule=sched)
                run_cell(led, regime_, metric, arm, drv, gran, 0.0, s, schedule=sched)


APPLIED_TSV = Path(__file__).resolve().parent / "wd_applied_gate_results.tsv"
APPLIED_KEYS = ["schedule", "metric", "arm", "driver", "gran", "nlr", "seed"]
APPLIED_METRICS = ["acc", "app_h0", "app_h1", "app_out", "eval_h0", "eval_h1", "eval_out"]


def part_applied(led, drivers, regime="normal", seed=42):
    """APPLIED-vs-EVAL gate magnitude — closes the diagnostic gap the main study flagged.

    The `g_*` columns of the main ledger are the gate RECOMPUTED AT EVAL. For a decay target that is
    diagnostic only (the gate never enters the forward), but it is NOT the operative factor, and the
    raw un-standardised novelty drivers read far larger on the test stream than in training —
    `vecproj`/boundary logged |f-1| up to 3.8e12 with accuracy at baseline. This part records what
    was ACTUALLY applied at each task boundary, alongside the eval recomputation, so the two can be
    told apart instead of the reader being asked to discount one of them.

    Its own ledger: adding a column to the main one is schema drift against 1236 finished rows, and
    `Ledger` raises on that by design.
    """
    ap_led = Ledger(APPLIED_TSV, keys=APPLIED_KEYS, metrics=APPLIED_METRICS)
    print("APPLIED vs EVAL gate magnitude — boundary schedule, class-IL, er-own\n", flush=True)
    print(f"  {'driver':9s} {'gran':8s} {'acc':>8s} {'applied |f-1|':>26s} {'eval |f-1|':>26s} "
          f"{'ratio':>10s}")
    for drv in drivers:
        for gran in GRANS:
            if (drv, gran) in D.SKIP:
                continue
            try:
                nlr = tuned_nlr(led, "boundary", "classil", "erown", drv, gran)
            except KeyError:
                continue
            key = dict(schedule="boundary", metric="classil", arm="erown", driver=drv, gran=gran,
                       nlr=f"{nlr:g}", seed=seed)
            if ap_led.is_done(**key):
                r = where(ap_led.rows(), **key)[0]
                a = [float(r[f"app_{k}"]) for k in ("h0", "h1", "out")]
                e = [float(r[f"eval_{k}"]) for k in ("h0", "h1", "out")]
                acc = float(r["acc"])
            else:
                lr, ep = main_point("classil", "erown")
                res = run(drv, gran, seed, "erown", "classil", nlr, lr, ep, BUFFERS[regime],
                          schedule="boundary")
                a = [res["applied"][k] for k in ("h0", "h1", "out")]
                e = [res["per_layer"][k] for k in ("h0", "h1", "out")]
                acc = res["acc"]
                ap_led.append(key, dict(acc=acc, app_h0=a[0], app_h1=a[1], app_out=a[2],
                                        eval_h0=e[0], eval_h1=e[1], eval_out=e[2]))
            ratio = (np.mean(e) / np.mean(a)) if np.mean(a) > 0 else float("nan")
            print(f"  {drv:9s} {gran:8s} {acc:>8.4f} "
                  f"{a[0]:>8.4f}/{a[1]:.4f}/{a[2]:.4f} {e[0]:>10.4g}/{e[1]:.4g}/{e[2]:.4g} "
                  f"{ratio:>10.3g}", flush=True)
    print("\n  ratio = eval / applied. ~1 means the eval recomputation is a fair proxy for what the")
    print("  mechanism did; >>1 means the driver reads out of distribution at test and the eval")
    print("  number describes the DRIVER's test-time behaviour, not the operation that was applied.")


def part_regimes(led, drivers, metrics, arms, schedules=("step",)):
    """Rule #12: the same cells at budget (200) and rehearsal-free (0) buffers."""
    print("REGIMES — budget (buffer 200) and rehearsal-free (buffer 0)\n", flush=True)
    for regime in ("budget", "rfree"):
        for sched in schedules:
            for metric in metrics:
                for arm in arms:
                    base = "er" if arm == "erown" else "naive"
                    for s in SEEDS:
                        run_cell(led, regime, metric, arm, base, "-", 0.0, s, schedule=sched)
                    for drv in drivers:
                        for gran in GRANS:
                            if (drv, gran) in D.SKIP:
                                continue
                            try:
                                nlr = tuned_nlr(led, sched, metric, arm, drv, gran)
                            except KeyError:
                                continue
                            for s in SEEDS:
                                run_cell(led, regime, metric, arm, drv, gran, nlr, s,
                                         schedule=sched)
                                run_cell(led, regime, metric, arm, drv, gran, 0.0, s,
                                         schedule=sched)


# ------------------------------------------------------------------ report
def part_report(led, metrics, arms, schedules=SCHEDULES, regime="normal"):
    rows = led.rows()
    print("\n" + "=" * 108)
    print("MODULATED WEIGHT DECAY   w <- f(s).(w - lr.grad)   |   d-dead = vs the same cell's "
          "neuro_lr=0 control (rule #10)")
    print("=" * 108)
    for sched in schedules:
        for metric in metrics:
            for arm in arms:
                lr, ep = main_point(metric, arm)
                base = "er" if arm == "erown" else "naive"
                bl = [float(r["acc"]) for r in where(rows, metric=metric, arm=arm, driver=base,
                                                     split="test", regime=regime)]
                shown = f"{base} {np.mean(bl):.4f}" if bl else f"{base} not run"
                print(f"\n--- schedule={sched}  {metric} / {arm}   (main lr {lr:g}, ep {ep}; "
                      f"ungated {shown}) ---")
                print(f"  {'driver':9s} {'gran':8s} {'acc':>9s} {'sd':>8s} {'d-dead':>10s} "
                      f"{'pos':>5s} {'forget':>8s} {'probe':>7s} {'|f-1| h0/h1/out':>26s}")
                for drv in DRIVERS:
                    for gran in GRANS:
                        if (drv, gran) in D.SKIP:
                            continue
                        live = _seeded(rows, sched, metric, arm, drv, gran, regime, dead=False)
                        dead = _seeded(rows, sched, metric, arm, drv, gran, regime, dead=True)
                        both = sorted(set(live) & set(dead))
                        if not both:
                            continue
                        d = [live[s] - dead[s] for s in both]
                        a = [live[s] for s in sorted(live)]
                        ex = _one(rows, sched, metric, arm, drv, gran, regime)
                        flag = " " if abs(np.mean(d)) >= NOISE_FLOOR else "~"
                        print(f"  {drv:9s} {gran:8s} {np.mean(a):>9.4f} {np.std(a):>8.4f} "
                              f"{np.mean(d):>+10.4f}{flag}"
                              f"{sum(x > 0 for x in d)}/{len(d):<3d} "
                              f"{float(ex['forget']):>8.4f} {float(ex['probe']):>7.3f} "
                              f"{float(ex['g_h0']):>8.4f}/{float(ex['g_h1']):.4f}/"
                              f"{float(ex['g_out']):.4f}")
    print(f"\n  ~ = |d-dead| < {NOISE_FLOOR} (1-seed noise floor); 'pos' = seeds with d-dead > 0.")
    print("  PREDICTION 1: if global ~= neuron ~= synapse within a driver, this is a global "
          "weight-NORM knob,")
    print("               not allocation — pt7 SET-1's result for the gradient lever, reproduced "
          "on the weight one.")
    print("  PREDICTION 3: check whether `probe` orders `d-dead`. plast_drivers found it did NOT "
          "(its most")
    print("               task-decodable driver was its worst cell); a relation HERE would be a "
          "real difference")
    print("               between gating a weight and gating a gradient.")
    _report_capacity(rows)


def _report_capacity(rows):
    """pt7_capacity's rule: when the modulator approaches the backbone in size, a positive result is
    a capacity confound before it is a mechanism. `vecproj` x synapse is (32, 477600) ~ 15.3M, ~32x
    the 478k backbone, so it is printed rather than left for a reader to notice."""
    seen = {}
    for r in rows:
        if r.get("extra_params") in (None, "", 0):
            continue
        seen[(r["driver"], r["gran"])] = (int(r["extra_params"]), float(r["param_ratio"]))
    bad = {k: v for k, v in seen.items() if v[1] >= 1.0}
    if not bad:
        return
    print("\n  CAPACITY CONFOUND — modulator size vs backbone (any positive result here is "
          "confounded):")
    for (drv, gran), (n, ratio) in sorted(bad.items(), key=lambda kv: -kv[1][1]):
        print(f"    {drv:9s} {gran:8s} extra_params {n:>12,}  = {ratio:>6.1f}x backbone")


def _seeded(rows, sched, metric, arm, drv, gran, regime, dead):
    out = {}
    for r in where(rows, schedule=sched, metric=metric, arm=arm, driver=drv, gran=gran,
                   split="test", regime=regime):
        if (float(r["nlr"]) == 0.0) == dead:
            out[int(r["seed"])] = float(r["acc"])
    return out


def _one(rows, sched, metric, arm, drv, gran, regime):
    sel = [r for r in where(rows, schedule=sched, metric=metric, arm=arm, driver=drv, gran=gran,
                            split="test", regime=regime) if float(r["nlr"]) != 0.0]
    return sel[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "anchor", "baseline", "deadcheck", "tune", "test", "regimes",
                             "applied", "report"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--driver", default=None, help="comma filter (a good shard axis)")
    ap.add_argument("--gran", default=None, help="comma filter; shard on this too — the synapse "
                                                 "cells are much heavier, so spreading them across "
                                                 "workers beats piling them into one")
    ap.add_argument("--metric", default=None, help="comma filter: classil,taskil")
    ap.add_argument("--arm", default=None, help="comma filter: erown,bufcur")
    ap.add_argument("--schedule", default="step", help="comma filter: step,boundary")
    ap.add_argument("--regime", default="normal", help="report/read regime: normal,budget,rfree")
    a = ap.parse_args()

    drivers = tuple(a.driver.split(",")) if a.driver else DRIVERS
    metrics = tuple(a.metric.split(",")) if a.metric else EVAL_METRICS
    arms = tuple(a.arm.split(",")) if a.arm else ARMS
    scheds = tuple(a.schedule.split(",")) if a.schedule else SCHEDULES
    global GRANS
    if a.gran:
        GRANS = tuple(g for g in GRANS if g in a.gran.split(","))
    led = ledger()
    tag = shard.shard_tag()
    print(f"modulated weight decay | device {DEV} | drivers {drivers} | grans {GRANS} | "
          f"metrics {metrics} | arms {arms} | schedules {scheds}"
          f"{' | shard ' + tag if tag else ''}\nledger {TSV}\n", flush=True)

    if a.part in ("all", "anchor"):
        part_anchor(led)
    if a.part in ("all", "baseline"):
        part_baseline(led, metrics, arms)
    if a.part == "deadcheck":
        part_deadcheck(led, drivers, scheds)
    if a.part in ("all", "tune"):
        part_tune(led, drivers, metrics, arms, scheds)
    if a.part in ("all", "test"):
        part_test(led, drivers, metrics, arms, scheds)
    if a.part == "regimes":
        part_regimes(led, drivers, metrics, arms, scheds)
    if a.part == "applied":
        part_applied(led, drivers)
    if a.part in ("all", "report"):
        part_report(led, metrics, arms, scheds, regime=a.regime)


if __name__ == "__main__":
    main()
