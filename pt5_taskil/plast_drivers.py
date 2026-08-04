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
# vec_x is K=784, so a per-SYNAPSE P is (784, 313600)+(784,160000)+(784,4000) = 3.7e8 params ~ 1.5 GB
# (plus 2x for Adam moments) and ~500x the 478k backbone. Skipped deliberately: CLAUDE.md's rule from
# pt7_capacity is that a modulator comparable to or larger than its backbone makes any result a capacity
# confound, so the cell would be uninterpretable even if it fit in memory. `vecproj` IS the
# synapse-tractable form of exactly this driver (that is why pt7 introduced it), and it is run.
SKIP = {("vec_x", "synapse")}
# per-driver standardization, per CLAUDE.md's tonic-vs-per-sample rule (see docstring)
STANDARDIZE = {"ach": True, "ach_ema": False, "nerisez": False, "vec_x": True, "vecproj": True}

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
class Driver:
    """m(x) provider + its own optimizer. Oracle-free: no driver here sees a task id.

    Copy-forward of `pt7_plast_tempslope.Driver`, extended with `ach` (per-sample entropy, absent
    there) and `vec_x`, and with standardization set PER DRIVER instead of one flag for the run.
    """
    HEAD_KEY = {"ach": "ACh", "ach_ema": "ACh_ema"}

    def __init__(self, name, lr):
        self.name = name
        std = STANDARDIZE[name]
        if name in self.HEAD_KEY:                                  # Signals head: 784->32->1 regresses tau
            self.kind = "head"; self.K = 1
            self.heads = p7.Heads(1).to(DEV)
            self.sig = p7.Signals([self.HEAD_KEY[name]], standardize=std)
            self.opt = torch.optim.Adam(self.heads.parameters(), lr)
        elif name in ("vec_x", "vecproj"):                         # head-free input novelty
            self.kind = "ne"
            self.drv = NEDriver(name, std)
            self.K = self.drv.K()
        elif name == "nerisez":                                    # stateful entropy z-score (MLP)
            self.kind = "stateful"; self.K = 1
            self.drv = pts.StatefulStd("nerisez", gru=False, standardize=std).to(DEV)
            self.opt = torch.optim.Adam(self.drv.parameters(), lr)
        else:
            raise ValueError(name)

    def value(self, net, X, update=True):
        """(B,K) driver, DETACHED — the plasticity path never grads back into the driver."""
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
            return
        with torch.no_grad():
            self.drv.upd_actual(p7.entropy(net.plain(X)[0]).unsqueeze(1))

    def state(self):
        """Snapshot of every mutable running statistic (frozen-vs-live eval without cross-talk)."""
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
        buffer=BUFFER, taskil=True):
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
    drv = None if plain else Driver(driver_name, lr=neuro_lr)
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
    return dict(pred=pred, probe=p7._probe(torch.cat(Ms), torch.cat(Ts), drv.K),
                per_layer={k: mags[k] / tot for k in mags})


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


def run_cell(stage, driver, gran, nlr, seed, ledger, taskil=True, main_lr=MAIN_LR):
    key = key_of(stage, driver, gran, nlr, seed)
    if key in ledger:
        print(f"[skip] {'|'.join(key)} acc={ledger[key][0]:.4f}", flush=True)
        return ledger[key]
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = run(driver, gran, seed, neuro_lr=nlr, taskil=taskil, main_lr=main_lr)
    pl = r["frozen"]["per_layer"]
    vals = (r["acc"], r["forget"], r["frozen"]["probe"], pl["h0"], pl["h1"], pl["out"],
            r["live"]["pred"])
    append(key, vals)
    ledger[key] = vals
    print(f"[run ] {'|'.join(key)} acc={r['acc']:.4f} forget={r['forget']:.4f} "
          f"probe={r['frozen']['probe']:.3f} |a-1|={pl['h0']:.4f}/{pl['h1']:.4f}/{pl['out']:.4f} "
          f"live_pred={r['live']['pred']:.4f} (frozen {r['frozen']['pred']:.4f})", flush=True)
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
                    choices=["all", "anchor", "deadcheck", "baseline", "test", "report"])
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
    if args.part in ("all", "test"):
        test(ledger)
    report(ledger)


if __name__ == "__main__":
    main()
