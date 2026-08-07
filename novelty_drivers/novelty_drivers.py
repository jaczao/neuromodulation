"""NOVELTY-DRIVER FORM: the difference VECTOR vs its NORM, and what "normal" is measured against.

The head-free input-novelty drivers are `reduce(x - reference)`. Everything run so far in pt5-pt7
fixed both halves of that expression by accident rather than by choice: the reduction was always the
raw difference vector (or, for `emb_all`, always its norm), and the reference was always an EMA
(with one `cumulative` ablation in results/pt7_variants.py). This study varies both, on the two
drivers the user named, in both regimes the thesis reports.

    kind        vec_x    = x - reference                     (784-d)
                vecproj  = R(x - reference), R random 784->32 (32-d)
    norm        0 = the difference VECTOR is the driver, K = its width
                1 = its L2 NORM is the driver, K = 1
    mean_mode   ema         recent-weighted streaming mean (the historical default)
                cumulative  exact running arithmetic mean over every sample seen so far
                trueavg     exact arithmetic mean of the data, from a full pass (both phases)
                ema+trueavg the ema while TRAINING, the exact mean at INFERENCE
    proj        learned  rank-K gate projection P trained by the main loss (zero-init => parity)
                random   P frozen at N(0, 0.1^2), in no optimizer (pt7_signalnet's fixedproj)
                dead     P frozen at ZERO => Gamma == 1 exactly: the RNG-matched control
    std         1 = per-dimension running standardisation (CLAUDE.md's per-sample-driver rule)
                0 = the RAW driver (the convention the recent single-driver work runs under)

WHY THE TWO AXES ARE NOT COSMETIC.

  NORM IS ALSO A CAPACITY AXIS, and a decisive one here. The gate projection is (K, 810), so
  vec_x/neuron with a learned P carries 635,040 parameters against a 478,410-parameter backbone —
  a ratio of 1.33, which neurocore.cost flags as a capacity confound outright (pt7's capacity
  ablation: the one arm that helped under scarcity helped because its modulator dwarfed the
  backbone). The same driver reduced to its norm carries 810 parameters, a ratio of 0.0017. If the
  vector form wins and the norm form does not, the first question is not "direction matters" but
  "did 635k extra trainable parameters matter". The two forms are also semantically different:
  the vector says WHERE a sample is unusual, the norm says only HOW unusual.

  THE REFERENCE DECIDES WHAT "NOVEL" MEANS. An EMA reference is recency-relative: after a task
  switch it chases the new task, so within a few hundred steps every sample of the new task is
  "normal" again. An exact dataset mean is absolute and never chases. In class-IL that is exactly
  the difference between a driver that tracks the task boundary and one that does not — so the two
  modes are testing different mechanisms, not different estimators of one. `ema+trueavg` is the
  deployable middle: cheap streaming statistics while training, an exact reference at inference,
  where the test stream can be averaged on the way through.

WHAT THIS RUNS. gain modulation, per-NEURON gate over (h0, h1, out), ADAM, BOTH standardisation
arms (the norm, when asked for, is always taken BEFORE standardising). NO NEW TUNING — every operating point is read from `neurocore.tuned`:
  class-IL  er-own, lr 3e-4, 5 ep/task, buffer 1000       (the pt7_tuned_syn val-selected ER point)
  standard  full MNIST, lr 1e-3, <=6 epochs early-stopped on the held-out VAL split, never test
            (results/pt7_std_tuned's protocol; that study swept epochs but NOT lr — recorded as a
            partial tune in neurocore.tuned, not dressed up as a full one)

CONTROLS AND THE ONE THING THIS STUDY HAS TO PROVE ABOUT ITSELF.
  `dead` is the rule-#10 RNG-matched baseline: same driver, same gate object, same buffer draws, but
  P frozen at zero so the gate is provably inert. Nothing in the modulator draws from the GLOBAL
  torch RNG (NEDriver's R and the frozen P both come from private generators, and a zero-init P is
  a `torch.zeros`), so `dead` is PREDICTED to equal the plain ungated baseline exactly — a claim
  `--part anchor` checks rather than assumes.
  The `trueavg` modes add a full pass over the images to compute the exact mean, and ITERATING A
  DATALOADER DRAWS FROM THE GLOBAL TORCH RNG even with shuffle=False, which would shift the data
  order and move those cells off the trajectory the `ema` cells run on. `neurocore.signals.
  dataset_mean` is RNG-neutral for exactly that reason, so a mean_mode cell differs from its `ema`
  twin ONLY in the reference vector. `--part anchor` also checks that directly: a DEAD gate under
  `trueavg` must be bit-identical to a DEAD gate under `ema`.

METRIC: class-IL accuracy is the MEAN OF THE FIVE PER-TASK ACCURACIES (macro), for the gated arms
and the baselines alike — the frozen pt7 eval pools instead, which biases a gated cell ~+0.0015
against a macro-averaged baseline. Standard accuracy is the single pooled test accuracy.

Run:  uv run python novelty_drivers/novelty_drivers.py --part anchor
      uv run python -m neurocore.shard --script novelty_drivers/novelty_drivers.py \
          --ledger novelty_drivers/novelty_drivers_results.tsv \
          --split metrics=classil,standard --split kinds=vec_x,vecproj \
          --args "--part grid --resume" --workers 4 --device mps
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

import pt7_neuromodulators as p7                                   # noqa: E402  (frozen, read-only)
from prototype.data import SplitMNIST, get_standard_loaders        # noqa: E402
from neurocore import shard                                        # noqa: E402
from neurocore.buffers import Reservoir                            # noqa: E402
from neurocore.controls import probe as task_probe                 # noqa: E402
from neurocore.cost import Cost, count_params                      # noqa: E402
from neurocore.gates import make_gate                              # noqa: E402
from neurocore.ledger import Ledger                                # noqa: E402
from neurocore.signals import NEDriver, dataset_mean               # noqa: E402
from neurocore.tuned import tuned_main                             # noqa: E402
from neurocore.utils import DEV, rng_frozen, seed_all              # noqa: E402

CE = nn.CrossEntropyLoss()
SEED = 42
BUFFER = 1000
GRAN = "neuron"                       # gain modulation; vec_x per-SYNAPSE would be 784 x 400 x 784
PROJ_STD = 0.1                        # frozen-random P scale (pt7_signalnet.run_all4_fixedproj)
PROJ_SEED_BASE = 7000                 # its private-generator namespace
STD_MAX_EPOCHS = 6                    # standard regime: val-selected epoch ceiling (pt7_std_tuned)

TSV = shard.ledger_path(Path(__file__).resolve().parent / "novelty_drivers_results.tsv")
KEYS = ["metric", "kind", "std", "norm", "mean_mode", "proj", "gran", "opt", "regime", "seed"]
METRICS = ["acc", "val", "epoch", "probe", "mabs", "g_h0", "g_h1", "g_out"]

KINDS = ("vec_x", "vecproj")
MEAN_MODES = ("ema", "cumulative", "trueavg", "ema+trueavg")
PROJS = ("learned", "random")

# Frozen reference points. The class-IL one is a BIT-EXACT anchor: this loop is a copy-forward of
# pt7_tuned_syn.run_baseline at the same tuned point, on the same device, with the same macro metric.
# The standard one is a BAND, not an anchor — pt7_std_tuned evaluates val/test inside its training
# loop without an RNG guard (and its test pass fires only on a val improvement, so its RNG draw
# depends on the accuracy trajectory); this study wraps every eval in rng_frozen(), which is the
# correct thing to do and necessarily lands on a slightly different trajectory.
ANCHOR_CLASSIL = 0.8975               # results/pt7_tuned_syn: report|er|adam, macro, MPS
REF_STANDARD = 0.9802                 # results/pt7_std_tuned: vanilla|-|std1|adam (band, ~0.007)


# ============================================================ construction
def build_driver(kind, norm, mean_mode, std):
    """The driver. The norm, when asked for, is taken before standardising (see NEDriver).

    `std` IS A KEY COLUMN, not a constant. The first pass of this study ran std=1 throughout
    (CLAUDE.md's "standardize per-sample drivers") and found that `vec_x`'s vector form collapses to
    chance once the reference mean is exact — a failure whose whole mechanism is division by the
    ~zero running variance of MNIST's constant border pixels. That makes the std=0 arm the control
    the claim needs rather than an optional extra: without standardisation there is no division, so
    if the collapse survives it was never about conditioning. It is also the convention the recent
    single-driver work runs under (position_paper/drivers.py's STANDARDIZE table is all-False,
    user-directed; fixedproj_scale likewise), so the two arms together cover both conventions.
    """
    return NEDriver(kind, standardize=bool(std), norm=norm, mean_mode=mean_mode)


def build_gate(drv, proj, seed=SEED):
    """(gate, trainable) for a projection mode. `learned` is the only one the optimizer sees.

    Neither `random` nor `dead` touches the global torch RNG — the frozen draw comes from a private
    generator and the dead P is `torch.zeros` — which is what makes `dead` an RNG-matched baseline
    and `random` comparable to it.
    """
    gate = make_gate(GRAN, drv.K(), None)
    if proj == "learned":
        return gate, True
    g = torch.Generator().manual_seed(PROJ_SEED_BASE + seed)
    for P in gate.params():
        r = torch.zeros(P.shape) if proj == "dead" else torch.randn(P.shape, generator=g) * PROJ_STD
        P.data = r.to(DEV)
        P.requires_grad_(False)
    return gate, False


def cell_cost(gate, buf):
    net_params = 784 * 400 + 400 + 400 * 400 + 400 + 400 * 10 + 10
    return Cost(backbone_params=net_params,
                extra_params=count_params(gate),
                buffer_bytes=buf.nbytes() if buf is not None else 0,
                # vec_x / vecproj are PRE-forward: the driver needs no pass through the net, so the
                # gate costs no extra forward at train OR at inference. (`trueavg` additionally pays
                # one amortised pass over the data per reference update — not a per-step cost.)
                fwd_train=1.0, bwd_train=1.0, fwd_infer=1.0, bwd_infer=0.0)


# ============================================================ class-IL (er-own)
def cl_true_mean(drv, loaders, upto):
    """Exact mean over the TRAINING images of every task seen so far — the causal, exact analogue of
    `cumulative` (which builds the same quantity online and therefore lags it)."""
    drv.set_true_mean(dataset_mean([loaders[j][0] for j in range(upto + 1)], space="x"))


def run_cl(kind, norm, mean_mode, std, proj, lr, epochs, loaders, opt_kind="adam", seed=SEED):
    """pt7's er-own gain arm: main net (and, when learned, P) trained jointly on the ER batch under
    plain CE. Copy-forward of pt7_tuned_syn.run_headless with the driver and projection swapped."""
    seed_all(seed)
    net = p7.Net().to(DEV)
    drv = build_driver(kind, norm, mean_mode, std)
    gate, trainable = build_gate(drv, proj, seed)
    params = list(net.parameters()) + (gate.params() if trainable else [])
    opt = p7._opt(opt_kind, params, lr)
    buf = Reservoir(BUFFER)
    for t in range(5):
        if drv.mean_mode == "trueavg":          # exact reference over the tasks seen so far
            cl_true_mean(drv, loaders, t)
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                m = drv.value(net, Xm, inference=False).detach()
                loss = CE(gate(net, m, Xm), Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                buf.add(x, y)
    if drv.uses_true_mean(inference=True):      # the evaluation stream's own exact mean
        drv.set_true_mean(dataset_mean([loaders[i][1] for i in range(5)], space="x"))
    res = eval_cl(net, gate, drv, loaders)
    res["cost"] = cell_cost(gate, buf)
    return res


def run_cl_plain(lr, epochs, loaders, opt_kind="adam", seed=SEED):
    """Plain ungated ER — verbatim pt7_tuned_syn.run_baseline('er'). The bit-exact anchor."""
    seed_all(seed)
    net = p7.Net().to(DEV)
    opt = p7._opt(opt_kind, net.parameters(), lr)
    buf = Reservoir(BUFFER)
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
    return dict(acc=acc, val=float("nan"), epoch=epochs, probe=float("nan"), mabs=0.0,
                g_h0=0.0, g_h1=0.0, g_out=0.0, cost=cell_cost(None, buf))


@torch.no_grad()
def eval_cl(net, gate, drv, loaders):
    """Macro class-IL accuracy under the gate, plus |g| per layer, mean |m| and the task probe.

    Driver statistics are frozen at test (`update=False`), which also selects the inference reference
    for `ema+trueavg`. Diagnostics are pooled (they are not the reported metric) and guarded against
    a diverged run, since a NaN probe input would raise instead of recording the collapse.
    """
    net.eval()
    accs, Ms, Ts = [], [], []
    mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    msum = 0.0
    tot = 0
    for i in range(5):
        c = n = 0
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            m = drv.value(net, x, update=False)
            c += (gate(net, m, x).argmax(1) == y).sum().item(); n += b
            pl = gate.per_layer_mag(m)
            for k in mags:
                mags[k] += pl[k] * b
            msum += float(m.abs().mean().item()) * b
            Ms.append(m.cpu()); Ts.append(torch.full((b,), i))
            tot += b
        accs.append(c / n)
    M = torch.cat(Ms); T = torch.cat(Ts)
    pr = float(task_probe(M, T, M.size(1))) if torch.isfinite(M).all() else float("nan")
    return dict(acc=float(np.mean(accs)), val=float("nan"), epoch=float("nan"), probe=pr,
                mabs=msum / tot, g_h0=mags["h0"] / tot, g_h1=mags["h1"] / tot,
                g_out=mags["out"] / tot)


# ============================================================ standard regime (full MNIST)
@torch.no_grad()
def _std_acc(fwd, loader):
    c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        c += (fwd(x).argmax(1) == y).sum().item(); tot += len(y)
    return c / tot


def run_standard(kind, norm, mean_mode, std, proj, lr, max_epochs, opt_kind="adam", seed=SEED):
    """Full MNIST, single task, 10-way CE. Epochs selected on the held-out VAL split (never test),
    which is results/pt7_std_tuned's protocol; the test number reported is the one at the best-val
    epoch. `None` for `kind` runs the ungated vanilla reference.

    Every eval pass is wrapped in rng_frozen(): iterating val/test inside the training loop would
    otherwise consume global torch RNG and make the data order depend on the accuracy trajectory.
    """
    seed_all(seed)
    tr, va, te = get_standard_loaders(batch_size=64)
    net = p7.Net().to(DEV)
    if kind is None:
        opt = p7._opt(opt_kind, net.parameters(), lr)
        drv = gate = None

        def step(x, y):
            loss = CE(net.plain(x)[0], y)
            opt.zero_grad(); loss.backward(); opt.step()

        def fwd(x):
            return net.plain(x)[0]
    else:
        drv = build_driver(kind, norm, mean_mode, std)
        gate, trainable = build_gate(drv, proj, seed)
        opt = p7._opt(opt_kind, list(net.parameters()) + (gate.params() if trainable else []), lr)
        if drv.mean_mode == "trueavg":                     # exact mean of the training images
            drv.set_true_mean(dataset_mean(tr, space="x"))

        def step(x, y):
            loss = CE(gate(net, drv.value(net, x, inference=False).detach(), x), y)
            opt.zero_grad(); loss.backward(); opt.step()

        def fwd(x):
            return gate(net, drv.value(net, x, update=False), x)

    best = dict(val=-1.0, acc=0.0, epoch=0)
    for ep in range(max_epochs):
        net.train()
        for x, y in tr:
            step(x.to(DEV), y.to(DEV))
        net.eval()
        with rng_frozen():
            # the eval reference is the evaluated stream's OWN exact mean, so val never sees test
            if drv is not None and drv.uses_true_mean(inference=True):
                drv.set_true_mean(dataset_mean(va, space="x"))
            v = _std_acc(fwd, va)
            if v > best["val"]:
                if drv is not None and drv.uses_true_mean(inference=True):
                    drv.set_true_mean(dataset_mean(te, space="x"))
                best = dict(val=v, acc=_std_acc(fwd, te), epoch=ep + 1)
            if drv is not None and drv.mean_mode == "trueavg":       # restore the training reference
                drv.set_true_mean(dataset_mean(tr, space="x"))

    out = dict(acc=best["acc"], val=best["val"], epoch=best["epoch"], probe=float("nan"))
    if gate is None:
        out.update(mabs=0.0, g_h0=0.0, g_h1=0.0, g_out=0.0, cost=cell_cost(None, None))
        return out
    with torch.no_grad(), rng_frozen():
        xb = next(iter(te))[0].to(DEV)
        m = drv.value(net, xb, update=False)
        pl = gate.per_layer_mag(m)
    out.update(mabs=float(m.abs().mean().item()), g_h0=pl["h0"], g_h1=pl["h1"], g_out=pl["out"],
               cost=cell_cost(gate, None))
    return out


# ============================================================ grid / ledger
def point(metric, opt_kind="adam"):
    """The val-tuned operating point. NOT retuned here (user-directed); a missing key raises, which
    is `neurocore.tuned`'s way of saying 'tune this first'."""
    base = "er" if metric == "classil" else "vanilla"
    tp = tuned_main("splitmnist", metric, base, opt_kind)
    return tp["lr"], tp["epochs_per_task"]


def build_cells(part, stds):
    """[(metric, kind, std, norm, mean_mode, proj)]; kind None = the ungated plain baseline.

    The plain baseline carries std="-" rather than a value: it builds no driver, so there is exactly
    ONE plain row per regime and the two standardisation arms share it instead of recording the same
    number twice under different keys.
    """
    cells = []
    if part in ("all", "anchor"):
        for metric in ("classil", "standard"):
            cells.append((metric, None, "-", 0, "-", "-"))                 # plain, ungated
            for std in stds:
                for kind in KINDS:                                         # dead gate, both shapes
                    for norm in (0, 1):
                        # `trueavg` alongside `ema` at a DEAD gate: the two must be bit-identical,
                        # which is the check that the extra mean pass left the RNG stream alone.
                        for mm in ("ema", "trueavg"):
                            cells.append((metric, kind, std, norm, mm, "dead"))
    if part in ("all", "grid"):
        for metric in ("classil", "standard"):
            for std in stds:
                for kind in KINDS:
                    for norm in (0, 1):
                        for mm in MEAN_MODES:
                            for proj in PROJS:
                                cells.append((metric, kind, std, norm, mm, proj))
    return cells


def fmt(r):
    val = "" if not np.isfinite(r["val"]) else f"  val={r['val']:.4f}@ep{int(r['epoch'])}"
    pr = "  probe=nan" if not np.isfinite(r["probe"]) else f"  probe={r['probe']:.3f}"
    return (f"acc={r['acc']:.4f}{val}{pr}  |m|={r['mabs']:.3g}  "
            f"|g|(h0/h1/out)={r['g_h0']:.3g}/{r['g_h1']:.3g}/{r['g_out']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "anchor", "grid", "table"])
    ap.add_argument("--metrics", default=None, help="comma filter: classil,standard")
    ap.add_argument("--kinds", default=None, help="comma filter: vec_x,vecproj")
    ap.add_argument("--norms", default=None, help="comma filter: 0,1")
    ap.add_argument("--std", default="0,1", help="standardisation arm(s) to run: 0, 1 or 0,1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true", help="skip cells already in the ledger")
    args = ap.parse_args()

    led = Ledger(TSV, keys=KEYS, metrics=METRICS, with_cost=True)
    if args.part == "table":
        print(table(led.rows()))
        return

    print(f"device={DEV}  novelty-driver form study (gain-{GRAN}, adam, 1 seed, NOT retuned)\n"
          f"  class-IL er-own : lr {point('classil')[0]:g}, {point('classil')[1]} ep/task, "
          f"buffer {BUFFER}   [anchor {ANCHOR_CLASSIL:.4f}]\n"
          f"  standard        : lr {point('standard')[0]:g}, <={point('standard')[1]} epochs "
          f"val-selected   [ref band {REF_STANDARD:.4f}]\n", flush=True)
    mfil = set(args.metrics.split(",")) if args.metrics else None
    kfil = set(args.kinds.split(",")) if args.kinds else None
    nfil = {int(v) for v in args.norms.split(",")} if args.norms else None
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = None

    stds = [int(v) for v in args.std.split(",")]
    for metric, kind, std, norm, mm, proj in build_cells(args.part, stds):
        if mfil and metric not in mfil:
            continue
        if kfil and kind is not None and kind not in kfil:
            continue
        if nfil and kind is not None and norm not in nfil:
            continue
        key = dict(metric=metric, kind=kind or "none", std=std, norm=norm, mean_mode=mm,
                   proj=proj, gran=GRAN if kind else "-", opt="adam",
                   regime="normal" if metric == "classil" else "single-task", seed=args.seed)
        if args.resume and led.is_done(**key):
            continue
        lr, ep = point(metric)
        if metric == "classil":
            if loaders is None:
                loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
            r = (run_cl_plain(lr, ep, loaders, seed=args.seed) if kind is None else
                 run_cl(kind, bool(norm), mm, std, proj, lr, ep, loaders, seed=args.seed))
            ref = ANCHOR_CLASSIL if kind is None else None
        else:
            r = run_standard(kind, bool(norm), mm, std, proj, lr, ep, seed=args.seed)
            ref = REF_STANDARD if kind is None else None
        note = "" if ref is None else f"   [ref {ref:.4f}, d={r['acc'] - ref:+.4f}]"
        print(f"  {metric:8s} {kind or 'plain':8s} std{std} norm{norm} {mm:12s} {proj:7s} | "
              f"{fmt(r)}{note}", flush=True)
        led.append(key, {k: r[k] for k in METRICS}, cost=r["cost"])
    print("ALL SELECTED CELLS DONE", flush=True)


# ============================================================ table
def table(rows):
    """One block per (metric, kind, norm): the two standardisation arms side by side.

    The dead gate is Gamma == 1 whatever the driver does, so `std` cannot reach it — all dead cells
    of a regime are one control, and the header reports their spread as the check that this is so.
    """
    out = []
    for metric in ("classil", "standard"):
        sel_m = [r for r in rows if r["metric"] == metric]
        if not sel_m:
            continue
        plain = next((float(r["acc"]) for r in sel_m if r["kind"] == "none"), float("nan"))
        deads = [r for r in sel_m if r["proj"] == "dead"]
        dead = float(np.mean([float(r["acc"]) for r in deads])) if deads else None
        head = f"\n=== {metric}   plain {plain:.4f}"
        if dead is not None:
            spread = max(float(r["acc"]) for r in deads) - min(float(r["acc"]) for r in deads)
            head += (f"   dead {dead:.4f}   (n={len(deads)}, spread {spread:.6f}, "
                     f"d-plain {dead - plain:+.6f})")
        out.append(head)
        for kind in KINDS:
            for norm in (0, 1):
                sel = [r for r in sel_m if r["kind"] == kind and int(r["norm"]) == norm
                       and r["proj"] in PROJS]
                if not sel:
                    continue
                ex = sel[0]
                out.append(f"\n  {kind} norm{norm}   (extra params {int(ex['extra_params']):,}, "
                           f"{float(ex['param_ratio']):.4f}x backbone)")
                out.append(f"    {'':<22s}{'--- std=0 (raw) ---':^36s}  {'--- std=1 ---':^36s}")
                out.append(f"    {'mean_mode':<13s}{'proj':<9s}" +
                           2 * f"{'acc':>9s}{'d-dead':>9s}{'probe':>8s}{'|m|':>10s}")
                for mm in MEAN_MODES:
                    for proj in PROJS:
                        line = f"    {mm:<13s}{proj:<9s}"
                        any_row = False
                        for std in (0, 1):
                            r = next((r for r in sel if r["mean_mode"] == mm and r["proj"] == proj
                                      and int(r["std"]) == std), None)
                            if r is None:
                                line += f"{'-':>9s}{'-':>9s}{'-':>8s}{'-':>10s}"
                                continue
                            any_row = True
                            dd = "-" if dead is None else f"{float(r['acc']) - dead:+.4f}"
                            pr = float(r["probe"])
                            line += (f"{float(r['acc']):>9.4f}{dd:>9s}"
                                     f"{(f'{pr:.3f}' if np.isfinite(pr) else '-'):>8s}"
                                     f"{float(r['mabs']):>10.3g}")
                        if any_row:
                            out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    main()
