"""FIXED-RANDOM PROJECTION SCALE + OFFSET — does a BIGGER gate change pt7's negative? (user-requested)

pt7's `run_all4_fixedproj` (results/pt7_signalnet.py) froze the rank-K projection P to a zero-mean
gaussian and found it landed on ER, i.e. the projection was never the lever. But it only ever swept
sigma in {0.1, 0.3}, always zero-mean, always on the `all4` composite. The open question this study
closes: if the drivers were given a MUCH bigger say in the main net's activations — a wider P, or a
P with a non-zero MEAN — would the mechanism stop being absorbed?

    Gamma_i = 1 + sum_k m_ik P_k,     P_kj ~ N(pmean, pstd^2),  FROZEN (in no optimizer)

WHAT THE TWO AXES DO, AND WHY THEY ARE NOT THE SAME KNOB.
  pstd  scales the TASK/SAMPLE-SPECIFIC part of the gate: each gated unit gets its own signed
        coefficient, so a larger pstd differentiates units more strongly per sample.
  pmean adds a COMMON-MODE component. With P_kj = mu + eps_kj, the gate splits as
            Gamma_i = 1 + mu * (sum_k m_ik)  +  sum_k m_ik eps_kj
        and the first term is IDENTICAL for every gated unit — a per-sample GLOBAL gain. On a ReLU
        hidden layer that is positively homogeneous (it propagates as a scalar on the logits, modulo
        biases) and on the out layer a uniform positive scale is ARGMAX-INVARIANT at eval. So the
        pre-registered expectation is that pmean is the WEAKER axis per unit of |g|, because it
        spends gate magnitude on the most absorbable / least prediction-relevant direction — the
        `temp`-vs-`slope` asymmetry results/pt7_plast_tempslope.md measured from the other side.
  pmean = 0 is run at BOTH pstd values as the reference that separates the two axes; without it
  "mean 1 vs mean 10" has no zero-mean control at the same width.

PRE-REGISTERED PREDICTION (written before running): every cell either ties ER or falls below it, and
accuracy is NON-MONOTONE-DOWN in |g| — because the magnitude axis is already mapped end to end by
prior studies (learned all4 |g| 0.006-0.08 = ER; fixed-random |g| 0.33-0.43 = ER; signalnet |g|
0.9-2.5 = ER at H=400 and chance at H<=10; unstandardised/tonic |g| 3-17 = collapse). This study is
worth running because it puts MEASURED numbers on the far end of that curve for the drivers the user
cares about, and because the pmean axis has never been run at all.

DRIVERS (three, all ORACLE-FREE, all UNSTANDARDISED as requested):
  ach      per-sample entropy H(x)                        gain-SYNAPSE   K=1
  nerisez  relu((H - ema_H)/sqrt(var_H + eps))            gain-SYNAPSE   K=1
  vecproj  (x - ema(x)) @ R,  R random 784->32            gain-NEURON    K=32

  `ach` and `nerisez` use the ACTUAL-VALUE convention (CLAUDE.md: the entropy family must not be
  head-predicted) — H comes from ONE extra unmodulated forward and nerisez's z-score is actual H
  against actual statistics. That is what makes this study head-free: `ActualEntropy` builds no
  modules and `NEDriver` draws its R from a PRIVATE generator, so the modulator consumes NO global
  torch RNG and the plain ER baseline is (predicted to be, and checked to be) already RNG-matched.
  UNSTANDARDISED is the user's choice and is a deliberate deviation from CLAUDE.md's rule; note it
  is inert for `nerisez` (its z-score IS the driver) and that raw `ach` is bounded in [0, ln 10], so
  the only driver whose raw scale can itself blow the gate up is `vecproj`.

ARM / OPERATING POINT: er-own (main net on the ER batch, current + replay, plain CE), class-IL,
buffer 1000, seed 42, 1 seed. lr/epochs are NOT retuned — they are read from `neurocore.tuned`
(class-IL ER: sgd 3e-2/ep5, adam 3e-4/ep5, the pt7_tuned_syn val-selected points). P is FROZEN, so
the main optimizer holds the backbone only and there is no neuro_lr to tune.

CONTROLS:
  dead  pstd = pmean = 0 => P == 0 => Gamma == 1 exactly. Same construction, same driver forwards,
        same buffer draws: the rule-#10 RNG-matched baseline for each (driver, opt).
  er    the plain ungated ER baseline, which `--part anchor` checks against the FROZEN
        pt7_tuned_syn ledger (sgd 0.9034, adam 0.8975) AND against `dead`. The dead-vs-plain
        equality is a CLAIM THIS RUN CHECKS, not an assumption — if the two differ, the modulator is
        consuming RNG somewhere and only `dead` is a valid reference. It came out EXACTLY equal
        (both optimizers, all three drivers), which is the measured version of the RNG argument
        above and is why `d-er` and `d-dead` are the same number in this ledger. Do NOT carry that
        over to a head-based driver: `Heads(K)` init does consume RNG (rule #10) and there the two
        references genuinely differ.

METRIC: accuracy is the MEAN OF THE FIVE PER-TASK ACCURACIES for every arm (see `evaluate` — the
frozen pt7 eval pools instead, which biases a gated cell ~+0.0015 against a macro-averaged
baseline; this ledger uses one convention for both arms).

`results/` is frozen (rule #9), so its primitives are imported READ-ONLY and the loop is
copy-forwarded, exactly as driver_traces/ and pt5_taskil/ do.

Run:  uv run python fixedproj_scale/fixedproj_scale.py --part all --resume
      uv run python -m neurocore.shard --script fixedproj_scale/fixedproj_scale.py \
          --ledger fixedproj_scale/fixedproj_scale_results.tsv \
          --split drivers=ach,nerisez,vecproj --split opts=sgd,adam \
          --args "--part grid --resume" --workers 5 --device mps
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

import pt7_neuromodulators as p7                                   # noqa: E402  (frozen, read-only)
from pt7_variants import NEDriver                                  # noqa: E402  (frozen, read-only)
from prototype.data import SplitMNIST                              # noqa: E402
from neurocore import shard                                        # noqa: E402
from neurocore.controls import probe as task_probe                 # noqa: E402
from neurocore.ledger import Ledger                                # noqa: E402
from neurocore.tuned import tuned_main                             # noqa: E402

DEV, EPS = p7.DEV, p7.EPS
CE = nn.CrossEntropyLoss()
SEED = 42
BUFFER = 1000
PROJ_SEED_BASE = 7000            # same private-generator namespace pt7_signalpnet's fixedproj used

TSV = shard.ledger_path(Path(__file__).resolve().parent / "fixedproj_scale_results.tsv")
KEYS = ["driver", "gran", "opt", "pmean", "pstd", "seed"]
METRICS = ["acc", "probe", "mabs", "g_h0", "g_h1", "g_out"]

# driver -> granularity (user-set: entropy family per-synapse, vecproj per-neuron)
GRAN = {"ach": "synapse", "nerisez": "synapse", "vecproj": "neuron"}
DRIVERS = ("ach", "nerisez", "vecproj")
# (pmean, pstd) grid. (0,0) is the dead control; pmean=0 rows separate the width axis from the
# offset axis at each width.
PSTDS = (1.0, 10.0)
PMEANS = (0.0, 1.0, 10.0)
DEAD = (0.0, 0.0)

# frozen pt7_tuned_syn `report|er|...` cells this copy-forward must reproduce (class-IL, tuned, MPS)
ANCHORS = {"sgd": 0.9034, "adam": 0.8975}


# ============================================================ driver providers
class ActualEntropy:
    """The entropy family from ACTUAL values (the standing convention), not head predictions.

    COPY-FORWARD of pt5_taskil/plast_drivers.py's class, trimmed to the two columns this study uses.
    ONE extra unmodulated forward per call yields H and both columns derive from it. No head,
    nothing to train, and still oracle-free: entropy needs no labels.

      ach      = H                                   (raw here — the study runs unstandardised)
      nerisez  = relu((H - ema_H)/sqrt(var_H + eps)) (actual H against actual statistics)

    LAG-1 IS PRESERVED as in the head versions: nerisez reads ema_H/var_H BEFORE this batch is
    folded in. At eval `update=False` freezes the statistics (the pt7 `frozen` convention).
    """

    def __init__(self, col, standardize=False):
        self.col = col
        self.standardize = standardize
        self.emaH = None; self.varH = None
        self.rm = None; self.rv = None; self.inited = False

    def K(self):
        return 1

    @torch.no_grad()
    def value(self, net, x, update=True):
        H = p7.entropy(net.plain(x)[0]).unsqueeze(1)              # (B,1) ACTUAL, one extra forward
        if self.emaH is None:
            self.emaH = H.mean().item(); self.varH = float(H.var(unbiased=False))
        if self.col == "nerisez":
            out = F.relu((H - self.emaH) / math.sqrt(self.varH + EPS))
        elif self.standardize:
            if update:
                bm, bv = H.mean(0), H.var(0, unbiased=False)
                if not self.inited:
                    self.rm, self.rv, self.inited = bm.clone(), bv.clone(), True
                else:
                    self.rm = 0.99 * self.rm + 0.01 * bm
                    self.rv = 0.99 * self.rv + 0.01 * bv
            out = (H - self.rm) / (self.rv.sqrt() + EPS) if self.inited else H
        else:
            out = H
        if update:                                                 # fold in AFTER reading (lag-1)
            self.varH = (1 - p7.BS) * self.varH + p7.BS * float(((H - self.emaH) ** 2).mean())
            self.emaH = (1 - p7.BS) * self.emaH + p7.BS * H.mean().item()
        return out


def make_driver(name):
    """The one constructor. NOTHING here draws from the global torch RNG: ActualEntropy builds no
    modules, and NEDriver's random projection R comes from its own seeded Generator. That is what
    makes the plain ER baseline RNG-matched to the gated cells — a claim `--part anchor` checks."""
    if name in ("ach", "nerisez"):
        return ActualEntropy(name, standardize=False)
    if name == "vecproj":
        return NEDriver("vecproj", standardize=False)
    raise ValueError(name)


# ============================================================ the frozen projection
def freeze_proj(gate, pmean, pstd, seed):
    """Overwrite the gate's zero-init P with a FIXED gaussian N(pmean, pstd^2) and freeze it.

    Generalises `pt7_signalnet._freeze_random_proj` (gaussian/rademacher, zero-mean only) with the
    location parameter. Drawn from a PRIVATE generator so the projection costs no global RNG and the
    dead control stays byte-comparable to the plain baseline. P is left out of every optimizer, so
    the only thing that adapts to the modulation is the backbone.
    """
    g = torch.Generator().manual_seed(PROJ_SEED_BASE + seed)
    for P in gate.params():
        r = torch.randn(P.shape, generator=g) * pstd + pmean if pstd > 0 else torch.full(P.shape, pmean)
        P.data = r.to(DEV)
        P.requires_grad_(False)


# ============================================================ the loop (copy-forward)
def run_cell(driver, opt_kind, pmean, pstd, lr, epochs, loaders, seed=SEED):
    """pt7 er-own gain, but the rank-K projection is FIXED at N(pmean, pstd^2) and the driver is
    head-free. Copy-forward of `pt7_tuned_syn.run_headless` with the projection frozen and the
    driver provider swapped; P is in NO optimizer, so `main_opt` holds the backbone alone."""
    p7.seed_all(seed)
    net = p7.Net().to(DEV)
    drv = make_driver(driver)
    gran = GRAN[driver]
    gate = p7.make_gate(gran, drv.K(), None)
    freeze_proj(gate, pmean, pstd, seed)
    opt = p7._opt(opt_kind, net.parameters(), lr)
    buf = p7.Reservoir(BUFFER)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                m = drv.value(net, Xm).detach()
                loss = CE(gate(net, m, Xm), Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                buf.add(x, y)
    return evaluate(net, gate, drv, gran, loaders)


def run_baseline(opt_kind, lr, epochs, loaders, seed=SEED):
    """Plain ungated ER — verbatim `pt7_tuned_syn.run_baseline(method='er')`."""
    p7.seed_all(seed)
    net = p7.Net().to(DEV)
    opt = p7._opt(opt_kind, net.parameters(), lr)
    buf = p7.Reservoir(BUFFER)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                loss = CE(net.plain(torch.cat(Xs))[0], torch.cat(Ys))
                opt.zero_grad(); loss.backward(); opt.step()
                buf.add(x, y)
    acc = float(np.mean([p7._acc_plain(net, loaders[i][1]) for i in range(5)]))
    return dict(acc=acc, probe=float("nan"), mabs=0.0, g_h0=0.0, g_h1=0.0, g_out=0.0)


@torch.no_grad()
def evaluate(net, gate, drv, gran, loaders):
    """Class-IL accuracy under the gate, plus |g| per layer, mean |m| and the task probe.

    ACCURACY IS THE MEAN OF THE FIVE PER-TASK ACCURACIES, not the pooled count over all test
    samples. The two differ because Split MNIST's test tasks have different sizes, and it is a
    SYSTEMATIC difference, not noise: pooling reads +0.0015 higher here at every driver. The macro
    average is the project's primary CL metric and is what `run_baseline` / `p7._acc_plain` compute,
    so pooling would have compared a gated cell to a baseline on a different statistic.

      NOTE this is a real inconsistency inside the FROZEN pt7 code, which is left alone (rule #9):
      `p7.eval_cell` and `pt7_tuned_syn.run_headless` POOL (`c/tot`) while `p7.train_baseline` /
      `run_baseline` MACRO-AVERAGE, so every frozen pt7 mechanism-vs-baseline delta carries a
      ~+0.0015 bias in the mechanism's favour — about a fifth of the 1-seed noise floor, so it
      changes no published conclusion, but it is exactly the size of several "small consistent
      positives". This ledger uses the macro average for BOTH arms; the consequence is that its
      gated cells are not directly comparable to a frozen pt7 gated number, only to the controls
      inside it. That is what rule #10 asks for anyway.

    Driver statistics are FROZEN at test (`update=False`). The diagnostics are pooled over all test
    samples (they are not the reported metric) and guarded against a diverged run: at large pstd the
    gate is expected to blow the net up, and a NaN probe input would otherwise raise instead of
    recording the collapse.
    """
    net.eval()
    accs = []
    mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    msum = 0.0
    tot = 0
    Ms, Ts = [], []
    for i in range(5):
        c = n = 0
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            m = drv.value(net, x, update=False)
            c += (gate(net, m, x).argmax(1) == y).sum().item(); n += b
            pl = gate.per_layer_mag(m) if gran == "neuron" else gate.per_layer_mag(m, net)
            for k in mags:
                mags[k] += pl[k] * b
            msum += float(m.abs().mean().item()) * b
            Ms.append(m.cpu()); Ts.append(torch.full((b,), i))
            tot += b
        accs.append(c / n)
    M = torch.cat(Ms); T = torch.cat(Ts)
    pr = float(task_probe(M, T, M.size(1))) if torch.isfinite(M).all() else float("nan")
    return dict(acc=float(np.mean(accs)), probe=pr, mabs=msum / tot,
                g_h0=mags["h0"] / tot, g_h1=mags["h1"] / tot, g_out=mags["out"] / tot)


# ============================================================ grid / ledger
def build_loaders():
    ds = SplitMNIST(sequence=p7.SEQ)
    return [ds.get_task_loaders(t, 64) for t in range(5)]


def point(opt_kind):
    """The val-tuned class-IL ER operating point. NOT retuned here (user-directed); a missing key
    raises by design, which is `neurocore.tuned`'s way of saying 'tune this first'."""
    tp = tuned_main("splitmnist", "classil", "er", opt_kind)
    return tp["lr"], tp["epochs_per_task"]


def build_cells(part):
    """[(kind, driver, opt, pmean, pstd)] for a --part selection."""
    cells = []
    if part in ("all", "anchor"):
        cells += [("base", "er", o, "-", "-") for o in ("sgd", "adam")]
        cells += [("cell", d, o, *DEAD) for d in DRIVERS for o in ("sgd", "adam")]
    if part in ("all", "grid"):
        cells += [("cell", d, o, mu, sd) for d in DRIVERS for o in ("sgd", "adam")
                  for sd in PSTDS for mu in PMEANS]
    return cells


def fmt(r):
    return (f"acc={r['acc']:.4f}  probe={r['probe']:.3f}  |m|={r['mabs']:.3g}  "
            f"|g|(h0/h1/out)={r['g_h0']:.3g}/{r['g_h1']:.3g}/{r['g_out']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "anchor", "grid", "table"])
    ap.add_argument("--drivers", default=None, help="comma filter on driver")
    ap.add_argument("--opts", default=None, help="comma filter on optimizer")
    ap.add_argument("--resume", action="store_true", help="skip cells already in the ledger")
    args = ap.parse_args()

    led = Ledger(TSV, keys=KEYS, metrics=METRICS)
    if args.part == "table":
        print(table(led.rows()))
        return

    print(f"device={DEV}  class-IL er-own gain, FIXED random P ~ N(pmean, pstd^2), 1 seed\n"
          f"tuned (not retuned): sgd {point('sgd')}  adam {point('adam')}\n", flush=True)
    loaders = build_loaders()
    dfil = set(args.drivers.split(",")) if args.drivers else None
    ofil = set(args.opts.split(",")) if args.opts else None

    for kind, driver, opt_kind, pmean, pstd in build_cells(args.part):
        if kind == "cell" and dfil and driver not in dfil:
            continue
        if ofil and opt_kind not in ofil:
            continue
        key = dict(driver=driver, gran=GRAN.get(driver, "-"), opt=opt_kind,
                   pmean=pmean, pstd=pstd, seed=SEED)
        if args.resume and led.is_done(**key):
            continue
        lr, ep = point(opt_kind)
        if kind == "base":
            r = run_baseline(opt_kind, lr, ep, loaders)
            note = f"   [anchor {ANCHORS[opt_kind]:.4f}, d={r['acc'] - ANCHORS[opt_kind]:+.4f}]"
        else:
            r = run_cell(driver, opt_kind, float(pmean), float(pstd), lr, ep, loaders)
            note = ""
        print(f"  {driver:8s} {GRAN.get(driver, '-'):7s} {opt_kind:4s} "
              f"mu={str(pmean):>4s} sd={str(pstd):>4s} | {fmt(r)}{note}", flush=True)
        led.append(key, r)
    print("ALL SELECTED CELLS DONE", flush=True)


def table(rows):
    """One block per (driver, optimizer): the (pmean, pstd) grid against its own dead control."""
    out = []
    base = {(r["opt"]): float(r["acc"]) for r in rows if r["driver"] == "er"}
    for d in DRIVERS:
        for o in ("sgd", "adam"):
            sel = [r for r in rows if r["driver"] == d and r["opt"] == o]
            if not sel:
                continue
            dead = next((float(r["acc"]) for r in sel
                         if float(r["pmean"]) == 0 and float(r["pstd"]) == 0), None)
            out.append(f"\n{d} / {GRAN[d]} / {o}   (dead {dead if dead is None else f'{dead:.4f}'}, "
                       f"er {base.get(o, float('nan')):.4f})")
            out.append(f"  {'pmean':>6s}{'pstd':>6s}{'acc':>9s}{'d-dead':>9s}{'probe':>7s}"
                       f"{'|m|':>10s}{'|g|h0':>10s}{'|g|h1':>10s}{'|g|out':>10s}")
            for r in sorted(sel, key=lambda r: (float(r["pstd"]), float(r["pmean"]))):
                dd = "-" if dead is None else f"{float(r['acc']) - dead:+.4f}"
                out.append(f"  {float(r['pmean']):>6g}{float(r['pstd']):>6g}{float(r['acc']):>9.4f}"
                           f"{dd:>9s}{float(r['probe']):>7.3f}{float(r['mabs']):>10.3g}"
                           f"{float(r['g_h0']):>10.3g}{float(r['g_h1']):>10.3g}"
                           f"{float(r['g_out']):>10.3g}")
    return "\n".join(out)


if __name__ == "__main__":
    main()
