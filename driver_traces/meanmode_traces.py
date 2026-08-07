"""REFERENCE-MEAN DRIVER TRACES — what the input-novelty drivers actually look like under `trueavg`
and `ema+trueavg`, next to the `ema` and `cumulative` references they replace.

Third study in the observer-only line (results/pt7_driver_traces* -> live_traces.py -> here). Same
contract: a PLAIN net trains, every driver is a passive read-out, NOTHING the observer computes
touches the loss, the parameters or the RNG. No gate is applied anywhere, so nothing here is an
accuracy claim — it is a claim about the SIGNAL.

WHAT IS TRACED. The two input-space novelty drivers, in both reductions, against all four reference
means — 16 series per arm:

    kind       vec_x  (784-d difference)      vecproj (32-d random projection of it)
    norm       0 = the difference vector      1 = its L2 norm (K=1)
    mean_mode  ema | cumulative | trueavg | ema+trueavg

Only the INPUT-space kinds are traced. An h1 reference mean is exact only w.r.t. the weights at the
moment it was computed, so `trueavg` in embedding space is a different (and much more delicate)
object than in input space — out of scope here rather than quietly conflated with it. `emb_all` /
`vec_h1*` keep their existing panels in live_traces.md.

WHAT THE REFERENCE POLICY IS (the thing a reader has to know to read the figures).
  training  `trueavg` installs the EXACT mean of the training images of every task seen so far,
            recomputed at each task boundary. That is the causal, exact counterpart of `cumulative`,
            which builds the same quantity online and therefore lags it. `ema+trueavg` uses the ema
            here, so its training panel must lie exactly on the `ema` line — a free correctness check
            visible in every figure.
  test      both true-mean modes install the exact mean of the EVALUATION STREAM (the pooled 5-task
            test set). Computing it costs one pass, which at inference the stream pays anyway; it is
            label-free, so it is deployable rather than an oracle.
  Statistics are FROZEN at test (`update=False`), the convention of the frozen study — the live-stat
  protocol is live_traces.py's premise and is deliberately not re-litigated here.

THE TWO THINGS THE FIGURES ARE FOR.
  1. An EMA reference is RECENCY-RELATIVE and an exact mean is ABSOLUTE. After a task switch the ema
     chases the new task, so within a few hundred steps the new digits read as "normal" again; the
     exact mean cannot chase. Whether the driver carries a task boundary at all is decided here, not
     by the formula.
  2. The standardised NORM-0 panels are the measured form of a trap this project has recorded but
     never plotted: standardising a K-dim vector per dimension and THEN taking its norm concentrates
     it at sqrt(K) (~28 for vec_x, ~5.7 for vecproj), so the standardised vector-form trace is a flat
     line near sqrt(K) carrying almost no per-sample information, while the norm-form trace (norm
     taken BEFORE standardising) is a genuine z-score straddling zero.

ANCHOR: the observer builds no global-RNG-consuming state (NEDriver's projection comes from a
private generator) and `dataset_mean`'s extra pass is RNG-neutral, so both arms must reproduce the
frozen live_traces numbers exactly: naive 0.5514, er 0.8988 (tuned Adam, lr 3e-4, 5 ep/task,
buffer 1000, seed 42). If they move, the observer has started perturbing training.

Outputs: meanmode_traces.npz, figs_meanmode/*.png.
Usage:
  uv run python driver_traces/meanmode_traces.py                 # train both arms + plot
  uv run python driver_traces/meanmode_traces.py --plot-only     # re-plot from the npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results"))

import pt7_neuromodulators as p7                                    # noqa: E402  (frozen, read-only)
from live_traces import Trace, test_loaders                         # noqa: E402  (live package)
from prototype.data import SplitMNIST                               # noqa: E402
from neurocore.buffers import Reservoir                             # noqa: E402
from neurocore.signals import NEDriver, dataset_mean                # noqa: E402
from neurocore.tuned import tuned_main                              # noqa: E402
from neurocore.utils import DEV, rng_frozen, seed_all               # noqa: E402

CE = nn.CrossEntropyLoss()
KINDS = ("vec_x", "vecproj")
NORMS = (0, 1)
MEAN_MODES = ("ema", "cumulative", "trueavg", "ema+trueavg")
SERIES = [(k, n, m) for k in KINDS for n in NORMS for m in MEAN_MODES]

NPZ = HERE / "meanmode_traces.npz"
FIGDIR = HERE / "figs_meanmode"
REF = {"naive": 0.5514, "er": 0.8988}                # live_traces.py, tuned Adam, seed 42
BUFFER = 1000


def key(kind, norm, mm):
    return f"{kind}|norm{norm}|{mm}"


# ------------------------------- the observer -------------------------------
class MeanModeObserver:
    """16 passive NEDrivers over one batch. Builds no parameters and consumes no global torch RNG
    (each projection comes from a private generator), so it is RNG-neutral by construction; the
    rng_frozen() wrap is belt-and-braces for anyone who adds state to it later."""

    def __init__(self):
        with rng_frozen():
            self.d = {key(k, n, m): NEDriver(k, standardize=True, norm=bool(n), mean_mode=m)
                      for k, n, m in SERIES}
        # a second copy with standardisation OFF: the raw panel must be the raw value, and a driver
        # cannot report both at once (its running stats are part of its state)
        with rng_frozen():
            self.raw = {key(k, n, m): NEDriver(k, standardize=False, norm=bool(n), mean_mode=m)
                        for k, n, m in SERIES}

    def set_true_mean(self, mu):
        for d in list(self.d.values()) + list(self.raw.values()):
            if d.mean_mode in ("trueavg", "ema+trueavg"):
                d.set_true_mean(mu)

    @torch.no_grad()
    def __call__(self, X, update=True, inference=False):
        """{series: (raw scalar, standardised scalar)} per sample. K>1 series are reduced to the L2
        norm for plotting, which is exactly why the norm-0 standardised panels show the sqrt(K)
        concentration: there the standardisation happened per DIMENSION, before this reduction."""
        out = {}
        for s in self.d:
            v = self.raw[s].value(None, X, update=update, inference=inference)
            z = self.d[s].value(None, X, update=update, inference=inference)
            out[s] = (_scalarise(v), _scalarise(z))
        return out


def _scalarise(v):
    return v.squeeze(1) if v.size(1) == 1 else v.norm(dim=1)


# ------------------------------- run -------------------------------
def run(arm, lr, epochs, ds, loaders, seed=42, shuffle_test=True, log=print):
    """Train a PLAIN net under `arm` (naive+masked CE, or ER with plain CE over current+replay) while
    reading out all 16 driver variants. Returns (train trace, test trace, accuracy)."""
    seed_all(seed)
    net = p7.Net().to(DEV)
    opt = p7._opt("adam", net.parameters(), lr)
    buf = Reservoir(BUFFER)
    obs = MeanModeObserver()
    tr = Trace([key(*s) for s in SERIES])

    for t in range(5):
        obs.set_true_mean(dataset_mean([loaders[j][0] for j in range(t + 1)], space="x"))
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if arm == "naive":
                    Xm, Ym = x.view(x.size(0), -1), y
                    loss = p7.masked_ce(net.plain(Xm)[0], Ym)
                else:
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                    loss = CE(net.plain(Xm)[0], Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                tr.add(obs(Xm, update=True, inference=False))     # observe AFTER the step, as pt7 does
                if arm == "er":
                    buf.add(x, y)
        tr.mark()
        log(f"    [{arm}] task {t} done ({tr.bounds[-1]} steps)")

    # test: the evaluation stream's own exact mean, statistics frozen
    obs.set_true_mean(dataset_mean([loaders[i][1] for i in range(5)], space="x"))
    net.eval()
    te = Trace([key(*s) for s in SERIES])
    c = tot = 0
    with torch.no_grad(), rng_frozen():
        for tl in test_loaders(ds, shuffle=shuffle_test):
            for x, y in tl:
                x, y = x.to(DEV), y.to(DEV)
                Xm = x.view(x.size(0), -1)
                c += (net.plain(Xm)[0].argmax(1) == y).sum().item(); tot += len(y)
                te.add(obs(Xm, update=False, inference=True))
            te.mark()
    return tr, te, c / tot


# ------------------------------- plotting -------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
# one categorical slot per reference mean, kept identical across every figure
COLOR = {"ema": "#2a78d6", "cumulative": "#eb6834", "trueavg": "#1f9d76", "ema+trueavg": "#9455c9"}
DASH = {"ema": (None, None), "cumulative": (None, None), "trueavg": (None, None),
        "ema+trueavg": (4, 2)}          # dashed: must lie ON the ema line while training
LABEL = {"ema": "ema (streaming)", "cumulative": "cumulative (exact, online)",
         "trueavg": "trueavg (exact, full pass)", "ema+trueavg": "ema train / trueavg inference"}
FORM = {0: r"$x_i-\overline{x}$", 1: r"$\|x_i-\overline{x}\|$"}


def _smooth(v, w):
    if w <= 1 or len(v) < w:
        return v
    pad = w // 2
    return np.convolve(np.pad(v, (pad, w - 1 - pad), mode="edge"), np.ones(w) / w, mode="valid")


def _panel(ax, data, arm, phase, kind, norm, basis, title, sqrtk=None):
    used = []
    for mm in MEAN_MODES:
        mk = f"{arm}|{phase}|{key(kind, norm, mm)}|{basis}|mean"
        if mk not in data or not len(data[mk]):
            continue
        v = data[mk]
        w = max(1, len(v) // 180)
        ax.plot(v, color=COLOR[mm], lw=0.7, alpha=0.15, zorder=1)
        ax.plot(_smooth(v, w), color=COLOR[mm], lw=1.6, label=LABEL[mm], zorder=3,
                dashes=DASH[mm] if DASH[mm][0] else (None, None))
        used.append(v)
    for b in data[f"{arm}|{phase}|bounds"][:-1]:
        ax.axvline(b, color=GRID, lw=0.8, ls="--", zorder=0)
    if sqrtk is not None:
        ax.axhline(sqrtk, color=INK2, lw=0.9, ls=":", zorder=2)
        ax.annotate(rf"$\sqrt{{K}}={sqrtk:.1f}$", xy=(0.985, sqrtk), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color=INK2)
    sym = _pick_scale(ax, used)
    ax.set_title(title + (" — symlog" if sym else ""), fontsize=8.5, color=INK2, pad=4, loc="left")
    ax.set_xlabel("training step" if phase == "train" else "test batch (tasks 0→4)",
                  fontsize=7.5, color=INK2)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=7, colors=INK2, length=3)


def _pick_scale(ax, series):
    a = np.concatenate([np.abs(s[np.isfinite(s)]) for s in series if len(s)]) if series else np.array([])
    if not len(a):
        return False
    typ = np.median(a[a > 0]) if (a > 0).any() else 0.0
    if typ > 0 and a.max() > 50 * typ and a.max() > 10:
        ax.set_yscale("symlog", linthresh=max(typ, 1e-6))
        return True
    return False


def _setup(data, arm):
    return (f"plain net, NO modulation applied · {arm} arm · stats frozen at test · ADAM "
            f"lr {float(data[f'{arm}|lr']):g} · {int(data[f'{arm}|epochs'])} ep/task · buffer "
            f"{BUFFER} · seed 42 · final class-IL acc {float(data[f'{arm}|acc']):.4f}")


def plot_all(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(exist_ok=True)
    n = 0
    for arm in ("naive", "er"):
        for kind in KINDS:
            for norm in NORMS:
                K = (784 if kind == "vec_x" else 32) if norm == 0 else 1
                fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), facecolor=SURFACE, squeeze=False)
                fig.suptitle(f"{kind}   norm={norm}      {FORM[norm]}   (K={K})", fontsize=12,
                             color=INK, x=0.012, ha="left", y=0.988, va="top")
                note = ("batch-mean of the per-sample value" if K == 1 else
                        f"batch-mean of the per-sample L2 norm ({K}-d driver)")
                fig.text(0.012, 0.925,
                         f"{note} · reference mean varied, everything else identical\n{_setup(data, arm)}",
                         fontsize=7.5, color=INK2, ha="left", va="top")
                for r, basis in enumerate(("raw", "std")):
                    for c, phase in enumerate(("train", "test")):
                        lab = "non-standardised" if basis == "raw" else "standardised (running stats)"
                        # the sqrt(K) guide only means something on a standardised MULTI-dim series
                        sk = np.sqrt(K) if (basis == "std" and K > 1) else None
                        _panel(axes[r][c], data, arm, phase, kind, norm, basis,
                               f"{lab} — {'training' if phase == 'train' else 'test'}", sqrtk=sk)
                        if c == 0:
                            axes[r][c].set_ylabel(lab.split(" (")[0], fontsize=8, color=INK2)
                h, l = axes[0][0].get_legend_handles_labels()
                fig.legend(h, l, loc="upper right", frameon=False, fontsize=8,
                           labelcolor=INK2, ncol=2, bbox_to_anchor=(0.99, 0.997))
                fig.tight_layout(rect=[0, 0, 1, 0.90])
                fig.savefig(FIGDIR / f"{arm}_{kind}_norm{norm}.png", dpi=150, facecolor=SURFACE)
                plt.close(fig); n += 1
    n += _contact_sheet(plt, data)
    print(f"  wrote {n} figures to {FIGDIR}")


def _contact_sheet(plt, data):
    """One sheet: raw test traces, every (kind, norm) cell, all four reference means."""
    cells = [(k, n) for k in KINDS for n in NORMS]
    fig, axes = plt.subplots(2, len(cells), figsize=(4.6 * len(cells), 6.0),
                             facecolor=SURFACE, squeeze=False)
    fig.suptitle("input-novelty drivers — non-standardised value at test, by reference mean",
                 fontsize=13, color=INK, x=0.008, ha="left", y=0.996, va="top")
    fig.text(0.008, 0.955, "rows: naive arm / ER arm · stats frozen at test · no modulation applied",
             fontsize=9, color=INK2, ha="left", va="top")
    for r, arm in enumerate(("naive", "er")):
        for c, (kind, norm) in enumerate(cells):
            axes[r][c].set_facecolor(SURFACE)
            _panel(axes[r][c], data, arm, "test", kind, norm, "raw", f"{arm} · {kind} norm{norm}")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, fontsize=10, labelcolor=INK2, ncol=4)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(FIGDIR / "_contact_sheet_test_raw.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return 1


# ------------------------------- numeric summary -------------------------------
def summarise(data, log=print):
    """Per-series level and dispersion. The columns that matter:
      `sd`     across-batch spread of the batch mean — does the trace MOVE across the run at all
      `within` mean within-batch sd — the per-sample information the gate could actually use
    A series with a large mean and both spreads near zero is a constant, whatever its formula says."""
    log(f"\n  {'series':<34s}{'arm':<7s}{'phase':<7s}{'basis':<6s}{'mean':>11s}{'sd':>11s}"
        f"{'within':>11s}{'min':>11s}{'max':>11s}")
    log("  " + "-" * 110)
    for kind, norm, mm in SERIES:
        s = key(kind, norm, mm)
        for arm in ("naive", "er"):
            for phase in ("train", "test"):
                for basis in ("raw", "std"):
                    mk = f"{arm}|{phase}|{s}|{basis}|mean"
                    sk = f"{arm}|{phase}|{s}|{basis}|sd"
                    if mk not in data or not len(data[mk]):
                        continue
                    v, w = data[mk], data[sk]
                    log(f"  {s:<34s}{arm:<7s}{phase:<7s}{basis:<6s}{np.nanmean(v):>11.3g}"
                        f"{np.nanstd(v):>11.3g}{np.nanmean(w):>11.3g}"
                        f"{np.nanmin(v):>11.3g}{np.nanmax(v):>11.3g}")


def boundary_contrast(data, log=print):
    """The headline number: does the driver still see a task boundary?

    For each series, the ratio of the |jump| in the batch-mean trace ACROSS the four task switches to
    its typical within-task step-to-step change. A recency-relative (ema) reference is expected to
    show a transient and then re-normalise; an absolute reference cannot re-normalise and should hold
    a persistent per-task LEVEL. Reported on the raw traces, where the level is meaningful.

    `recovery` = the fraction of a boundary jump that has decayed away by the end of the new task
    (1.0 = fully re-normalised, 0.0 = the shift persists). It is a RATIO WITH THE JUMP IN THE
    DENOMINATOR, so it is meaningless where the jump is itself small — boundaries whose jump is under
    3x the typical step are excluded, and `n` reports how many of the four survived. Without that
    guard a near-zero jump produced readings like -77.9.
    """
    log(f"\n  task-boundary contrast (raw training trace)\n"
        f"  {'series':<34s}{'arm':<7s}{'jump/step':>11s}{'level range':>14s}{'recovery':>11s}{'n':>4s}")
    log("  " + "-" * 82)
    for kind, norm, mm in SERIES:
        s = key(kind, norm, mm)
        for arm in ("naive", "er"):
            mk = f"{arm}|train|{s}|raw|mean"
            if mk not in data or not len(data[mk]):
                continue
            v = np.asarray(data[mk], dtype=np.float64)
            b = [int(x) for x in data[f"{arm}|train|bounds"]][:-1]
            step = np.median(np.abs(np.diff(v))) + 1e-12
            jumps = [abs(v[i] - v[i - 1]) for i in b if 0 < i < len(v)]
            # per-task levels: the mean over the LAST 20% of each task's steps (post-transient)
            edges = [0] + b + [len(v)]
            levels = [float(np.mean(v[a + int(0.8 * (z - a)):z])) for a, z in zip(edges, edges[1:])
                      if z > a]
            # "recovery": how much of the boundary jump is gone by the end of the new task
            rec = []
            for i, e in zip(b, edges[2:]):          # each boundary paired with the END of its task
                if 0 < i < len(v) and e > i:
                    j = v[i] - v[i - 1]
                    if abs(j) > 3 * step:           # else the ratio divides by ~nothing
                        rec.append(1.0 - abs(float(np.mean(v[i + int(0.8 * (e - i)):e])) - v[i - 1])
                                   / abs(j))
            log(f"  {s:<34s}{arm:<7s}{np.mean(jumps) / step:>11.1f}"
                f"{max(levels) - min(levels):>14.4g}"
                f"{(np.mean(rec) if rec else float('nan')):>11.2f}{len(rec):>4d}")


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="override epochs/task (smoke tests)")
    ap.add_argument("--no-shuffle-test", dest="shuffle_test", action="store_false",
                    help="keep the MNIST test-file order (it is not neutral: mean |x| rises with "
                         "file index, rho ~ +0.64)")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        z = np.load(NPZ, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        plot_all(d); summarise(d); boundary_contrast(d)
        return
    if NPZ.exists() and not args.force:
        raise SystemExit(f"{NPZ.name} already exists — --plot-only to re-plot, --force to re-train.")

    print(f"device={DEV}  reference-mean driver traces (adam, tuned, seed 42, buffer {BUFFER})\n",
          flush=True)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    out = {}
    for arm in ("naive", "er"):
        tp = tuned_main("splitmnist", "classil", arm, "adam")
        lr, ep = tp["lr"], (args.epochs or tp["epochs_per_task"])
        print(f"  running {arm} (lr={lr:g}, {ep} ep/task) ...", flush=True)
        tr, te, acc = run(arm, lr, ep, ds, loaders, shuffle_test=args.shuffle_test,
                          log=lambda s: print(s, flush=True))
        d = acc - REF[arm]
        print(f"  {arm:5s} final avg class-IL acc = {acc:.4f}   "
              f"(live_traces {REF[arm]:.4f}, delta {d:+.4f})", flush=True)
        out[f"{arm}|lr"] = np.array(lr); out[f"{arm}|epochs"] = np.array(ep)
        out[f"{arm}|acc"] = np.asarray(acc, dtype=np.float32)
        for phase, t in (("train", tr), ("test", te)):
            for (k, b, s), v in t.d.items():
                out[f"{arm}|{phase}|{k}|{b}|{s}"] = np.asarray(v, dtype=np.float32)
            out[f"{arm}|{phase}|bounds"] = np.asarray(t.bounds, dtype=np.int32)
    np.savez_compressed(NPZ, **out)
    print(f"  wrote {NPZ}", flush=True)
    plot_all(out)
    summarise(out, log=lambda s: print(s, flush=True))
    boundary_contrast(out, log=lambda s: print(s, flush=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
