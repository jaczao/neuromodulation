"""pt7 DRIVER TRACES — the VALUE of each single-neuromodulator formula over training and at test.

OBSERVER-ONLY: no gate, no modulation, no head. A PLAIN Net is trained under two regimes and every pt7
single-neuromodulator formula tau_k is evaluated on each batch as a passive read-out. Nothing the observer
computes touches the loss, the parameters, or the RNG, so both runs reproduce the frozen pt7 baselines to
within the ~0.007-0.016 MPS 1-seed noise floor — that agreement is the sanity anchor.

OPERATING POINT: `--point tuned` (DEFAULT) takes lr/epochs per arm from configs.TUNED_MAIN, i.e. the
val-tuned class-IL point (Adam lr 3e-4, ep 5 for BOTH naive and er; references naive 0.5545 / er 0.8975 in
pt7_tuned_syn_results.tsv). `--point untuned` is the inherited pt7 point (lr 1e-3, ep 5; references naive
0.3900 / er 0.8946) that every pt7 driver number was actually recorded at. The distinction matters for the
loss-based drivers: naive goes 0.39 -> 0.55 when tuning LOWERS Adam's lr, so its loss trajectory — and hence
DA/5HT/ACh_vol — is materially different between the two. class-IL naive-SGD has no tuned entry.

REGIMES (x-axis of every "training" panel):
  naive : masked per-sample CE on the current task only, no buffer   (pt7 `nobuf`/naive baseline)
  er    : plain 10-way CE on cat([current, replay]), reservoir 1000  (pt7 `er` baseline, NO masked loss)

The loss-based formulas (DA*, ACh_vol*, 5HT*) contain an inner per-sample loss ell_i. pt7's Signals uses the
MASKED per-sample CE for it in BOTH arms — including er-own, where the net itself trains on plain CE — so
that is the default here (`--driver-loss pt7`). `--driver-loss arm` instead lets ell_i follow the arm's own
training loss (plain CE under er); use `--suffix` to write that variant beside the default.

DRIVERS (all single neuromodulators; the composites all4/UNIFY-12 and the controls free/5ht-const are
excluded by construction — they are not one formula). Multidimensional drivers are reported as the
per-sample L2 NORM, then averaged over the batch; scalar drivers keep their sign.

  DA         (ell - ema_slow)/std                 phasic reward-prediction error
  DA_step    (ell - prev_batch_loss)/std          one-step RPE
  DA_fast    (ell - ema_fast)/|ema_fast|          fast-baseline RPE
  ACh        H(logits)                            expected uncertainty (predictive entropy)
  ACh_ema    ema(H)                               TONIC (constant within a batch)
  ACh_vol    sqrt(ema((L - ema_fast)^2))          TONIC volatility
  ACh_vol_ps |ell - ema_fast|                     per-sample volatility
  NE         relu((|DA| - ACh_vol)/ACh_vol)       unexpected uncertainty
  NE_rise    max(ema_fast - ema_slow, 0)          TONIC rise
  NE_emb     ||h1 - mean_h1||                     embedding novelty (== pt7-variants `emb_all`)
  nerisez    relu((H - ema_H)/sqrt(var_H))        z-scored entropy surprise (ACTUAL H, no predictor)
  5HT        -ell                                 per-sample reward
  5HT_ema    -ema_slow                            TONIC average reward
  vec_h1     ||h1 - mean_h1||                     400-d embedding-novelty vector
  vec_h1proj ||R(h1 - mean_h1)||                  32-d random projection of it
  vec_x      ||x - mean_x||                       784-d input novelty
  vecproj    ||R(x - mean_x)||                    32-d random projection of it

STANDARDISED vs NOT: every driver is shown both raw and under pt7's running standardisation
(run_mean/run_var updated 0.99/0.01 on train batches, FROZEN at test) — the same transform Signals/NEDriver
apply, reproduced once here so raw and standardised come from a single pass.

Outputs: pt7_driver_traces.npz (all traces), pt7_driver_traces/*.png (one 2x2 figure per driver:
{raw, standardised} x {training, test} with both regimes overlaid), pt7_driver_traces.log.
seed 42, buffer 1000, batch 64, Adam by default (--opt sgd for the SGD point), 1 seed.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pt7_neuromodulators as p7                                    # noqa: E402
import pt7_variants as pv                                           # noqa: E402
sys.path.insert(0, str(HERE.parent / "prototype"))
from configs import TUNED_MAIN                                      # noqa: E402
from data import SplitMNIST                                         # noqa: E402

DEV, EPS, BS = p7.DEV, p7.EPS, p7.BS
CE = nn.CrossEntropyLoss()

HEAD_DRIVERS = ["DA", "DA_step", "DA_fast", "ACh", "ACh_ema", "ACh_vol", "ACh_vol_ps",
                "NE", "NE_rise", "NE_emb", "5HT", "5HT_ema"]
NE_KINDS = ["vec_h1", "vec_h1proj", "vec_x", "vecproj"]             # emb_all == NE_emb, not duplicated
ALL_DRIVERS = HEAD_DRIVERS + ["nerisez"] + NE_KINDS

# label -> (formula shown on the panel, neuromodulator family, dimensionality)
META = {
    "DA":         (r"$(\ell_i-\mathrm{ema}_{slow})/\mathrm{std}(\ell)$", "DA", 1),
    "DA_step":    (r"$(\ell_i-L_{t-1})/\mathrm{std}(\ell)$", "DA", 1),
    "DA_fast":    (r"$(\ell_i-\mathrm{ema}_{fast})/|\mathrm{ema}_{fast}|$", "DA", 1),
    "ACh":        (r"$H(\mathrm{logits}_i)$", "ACh", 1),
    "ACh_ema":    (r"$\mathrm{ema}(H)$  (tonic)", "ACh", 1),
    "ACh_vol":    (r"$\sqrt{\mathrm{ema}((L-\mathrm{ema}_{fast})^2)}$  (tonic)", "ACh", 1),
    "ACh_vol_ps": (r"$|\ell_i-\mathrm{ema}_{fast}|$", "ACh", 1),
    "NE":         (r"$\mathrm{relu}((|DA_i|-\mathrm{ACh}_{vol})/\mathrm{ACh}_{vol})$", "NE", 1),
    "NE_rise":    (r"$\max(\mathrm{ema}_{fast}-\mathrm{ema}_{slow},0)$  (tonic)", "NE", 1),
    "NE_emb":     (r"$\|h_1^{(i)}-\overline{h_1}\|$", "NE", 1),
    "nerisez":    (r"$\mathrm{relu}((H_i-\mathrm{ema}_H)/\sqrt{\mathrm{var}_H})$", "NE", 1),
    "5HT":        (r"$-\ell_i$", "5-HT", 1),
    "5HT_ema":    (r"$-\mathrm{ema}_{slow}$  (tonic)", "5-HT", 1),
    "vec_h1":     (r"$\|h_1^{(i)}-\overline{h_1}\|$  (400-d)", "NE", 400),
    "vec_h1proj": (r"$\|R(h_1^{(i)}-\overline{h_1})\|$  (32-d)", "NE", 32),
    "vec_x":      (r"$\|x_i-\overline{x}\|$  (784-d)", "NE", 784),
    "vecproj":    (r"$\|R(x_i-\overline{x})\|$  (32-d)", "NE", 32),
}

NPZ = HERE / "pt7_driver_traces.npz"
FIGDIR = HERE / "pt7_driver_traces"

# Reference accuracies for the sanity anchor. TUNED: pt7_tuned_syn_results.tsv `report|{naive,er}|...`
# (naive RISES 0.39->0.55 at the tuned Adam point because tuning LOWERS the lr 1e-3->3e-4 and a smaller lr
# forgets less). UNTUNED: pt7_results.tsv. classil-naive-sgd has no tuned entry (never swept).
REF = {("tuned", "naive", "adam"): 0.5545, ("tuned", "er", "adam"): 0.8975,
       ("tuned", "er", "sgd"): 0.9034, ("tuned", "naive", "sgd"): 0.5548,
       ("untuned", "naive", "adam"): 0.3900, ("untuned", "er", "adam"): 0.8946,
       ("untuned", "naive", "sgd"): 0.6287, ("untuned", "er", "sgd"): 0.7234}
UNTUNED = dict(lr=1e-3, epochs_per_task=5)          # the inherited pt7 point every driver number sits at


def hparams(point, arm, opt_kind):
    """(lr, epochs) for an arm. Tuned values come from configs.TUNED_MAIN (rule #7: no hardcoded
    hyperparameters); a missing key is a hard error meaning "that combination was never swept"."""
    if point == "untuned":
        return UNTUNED["lr"], UNTUNED["epochs_per_task"]
    try:
        c = TUNED_MAIN[("classil", arm, opt_kind)]
    except KeyError:
        raise SystemExit(f"no tuned entry for (classil, {arm}, {opt_kind}) — tune it first, "
                         f"or pass --point untuned")
    return c["lr"], c["epochs_per_task"]


# ------------------------------- standardisation (pt7's, reproduced once) -------------------------------
class Standardizer:
    """Running per-dim standardisation, identical to Signals/NEDriver: update 0.99/0.01 on train batches
    (init from the first batch), then (v - run_mean)/(sqrt(run_var) + eps). Frozen when update=False."""

    def __init__(self):
        self.rm = self.rv = None

    def __call__(self, v, update=True):
        if update:
            bm, bv = v.mean(0), v.var(0, unbiased=False)
            if self.rm is None:
                self.rm, self.rv = bm.clone(), bv.clone()
            else:
                self.rm = 0.99 * self.rm + 0.01 * bm
                self.rv = 0.99 * self.rv + 0.01 * bv
        if self.rm is None:
            return v
        return (v - self.rm) / (self.rv.sqrt() + EPS)


def _scalarise(v):
    """(B,K) -> per-sample scalar: the value itself if K==1 (sign kept), else the L2 norm."""
    return v.squeeze(1) if v.size(1) == 1 else v.norm(dim=1)


# ------------------------------- the observer -------------------------------
class Observer:
    """Evaluates every pt7 single-neuromodulator formula on a batch. Read-only: no grad, no RNG, no writes
    to the net. Holds the EMA state each formula needs (shared exactly as pt7 shares it inside Signals)."""

    def __init__(self, proj_seed=0, loss_fn=None):
        # loss_fn=None -> pt7's default (per-sample MASKED CE), which pt7 uses in the er-own arm too
        self.sig = p7.Signals(HEAD_DRIVERS, standardize=False, loss_fn=loss_fn)
        self.ne = {k: pv.NEDriver(k, standardize=False, seed=proj_seed) for k in NE_KINDS}
        self.emaH = self.varH = None                                # nerisez stats (ACTUAL entropy)
        self.std = {k: Standardizer() for k in ALL_DRIVERS}

    @torch.no_grad()
    def __call__(self, net, X, Y, update):
        vals = {}
        T = self.sig.targets(net, X, Y, update=update)              # (B, len(HEAD_DRIVERS)) raw
        for i, k in enumerate(HEAD_DRIVERS):
            vals[k] = T[:, i:i + 1]
        for k, d in self.ne.items():
            vals[k] = d.value(net, X, update=update)
        vals["nerisez"] = self._nerisez(net, X, update)

        out = {}
        for k, v in vals.items():
            s = self.std[k](v, update=update)                       # update-then-transform, as pt7 does
            out[k] = (_scalarise(v), _scalarise(s))
        return out

    def _nerisez(self, net, X, update):
        """relu((H - ema_H)/sqrt(var_H)) on the ACTUAL entropy (pt7_stateful predicts H with a head; the
        formula is the same, the predictor is a separate mechanism and is out of scope here)."""
        H = p7.entropy(net.plain(X)[0]).unsqueeze(1)
        if self.emaH is None:
            self.emaH = H.mean().item(); self.varH = H.var(unbiased=False).item()
            return F.relu(H - H.mean()) / (H.std() + EPS)           # bootstrap (batch z-score), as pt7
        v = F.relu((H - self.emaH) / math.sqrt(self.varH + EPS))
        if update:                                                  # var uses the OLD mean, then the mean
            self.varH = (1 - BS) * self.varH + BS * ((H - self.emaH) ** 2).mean().item()
            self.emaH = (1 - BS) * self.emaH + BS * H.mean().item()
        return v


class Trace:
    """Accumulates (batch mean, batch sd) of each driver's per-sample scalar, raw and standardised."""

    def __init__(self):
        self.d = {(k, b, s): [] for k in ALL_DRIVERS for b in ("raw", "std") for s in ("mean", "sd")}
        self.bounds = []

    def add(self, rec):
        for k, (raw, std) in rec.items():
            for b, v in (("raw", raw), ("std", std)):
                self.d[(k, b, "mean")].append(float(v.mean()))
                self.d[(k, b, "sd")].append(float(v.std()) if v.numel() > 1 else 0.0)

    def mark(self):
        self.bounds.append(len(self.d[(ALL_DRIVERS[0], "raw", "mean")]))


# ------------------------------- run -------------------------------
def run(arm, opt_kind="adam", lr=1e-3, epochs=5, buffer=1000, seed=42, driver_loss="pt7", log=print):
    """Train a PLAIN net under `arm` while passively tracing every driver. Returns (train, test, acc).

    driver_loss: "pt7" = the loss inside DA/5HT/... is the per-sample MASKED CE in BOTH arms (what pt7's
    Signals does, including in er-own); "arm" = it follows the arm's own training loss (plain CE under er).
    """
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    net = p7.Net().to(DEV)
    opt = p7._opt(opt_kind, net.parameters(), lr)
    buf = p7.Reservoir(buffer)
    obs = Observer(loss_fn=(pv.per_sample_ce_plain if (driver_loss == "arm" and arm == "er") else None))
    tr = Trace()

    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if arm == "naive":                                  # masked CE, current task only
                    Xm, Ym = x.view(x.size(0), -1), y
                    loss = p7.masked_ce(net.plain(Xm)[0], Ym)
                else:                                               # er: plain CE over current + replay
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                    loss = CE(net.plain(Xm)[0], Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                tr.add(obs(net, Xm, Ym, update=True))                # observe AFTER the step, as pt7 does
                if arm == "er":
                    buf.add(x, y)
        tr.mark()
        log(f"    [{arm}] task {t} done ({tr.bounds[-1]} steps)")

    net.eval()
    te = Trace()
    c = tot = 0
    with torch.no_grad():
        for i in range(5):
            for x, y in loaders[i][1]:
                x, y = x.to(DEV), y.to(DEV)
                Xm = x.view(x.size(0), -1)
                c += (net.plain(Xm)[0].argmax(1) == y).sum().item(); tot += len(y)
                te.add(obs(net, Xm, y, update=False))                # stats FROZEN at test
            te.mark()
    return tr, te, c / tot


# ------------------------------- plotting -------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
COLOR = {"naive": "#2a78d6", "er": "#eb6834"}                        # categorical slots 1 & 2
LABEL = {"naive": "naive + masked loss", "er": "ER (plain CE)"}


def _smooth(v, w):
    """Centred moving average with edge padding (mode='same' would droop at both ends)."""
    if w <= 1 or len(v) < w:
        return v
    pad = w // 2
    return np.convolve(np.pad(v, (pad, w - 1 - pad), mode="edge"), np.ones(w) / w, mode="valid")


def _pick_scale(ax, series):
    """Linear unless the dynamic range is extreme (the tonic drivers blow up once standardised), in which
    case symlog around the typical magnitude so both regimes stay legible in one panel."""
    a = np.concatenate([np.abs(s[np.isfinite(s)]) for s in series if len(s)])
    if not len(a):
        return False
    typ = np.median(a[a > 0]) if (a > 0).any() else 0.0
    if typ > 0 and a.max() > 50 * typ and a.max() > 10:
        ax.set_yscale("symlog", linthresh=max(typ, 1e-6))
        return True
    return False


def _panel(ax, data, key, basis, phase, title):
    used = []
    for arm in ("naive", "er"):
        m = data[f"{arm}|{phase}|{key}|{basis}|mean"]
        if not len(m):
            continue
        w = max(1, len(m) // 180)
        ax.plot(m, color=COLOR[arm], lw=0.7, alpha=0.16, zorder=1)
        ax.plot(_smooth(m, w), color=COLOR[arm], lw=1.6, label=LABEL[arm], zorder=3)
        used.append(m)
        for b in data[f"{arm}|{phase}|bounds"][:-1]:
            ax.axvline(b, color=GRID, lw=0.8, ls="--", zorder=0)
    sym = _pick_scale(ax, used)
    ax.set_title(title + (" — symlog" if sym else ""), fontsize=8.5, color=INK2, pad=4, loc="left")
    ax.set_xlabel("training step" if phase == "train" else "test batch (tasks 0→4)",
                  fontsize=7.5, color=INK2)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=7, colors=INK2, length=3)


def _setup_line(data):
    """Human-readable operating point, taken from the run itself (per arm, since a tuned point may differ)."""
    opt, point = str(data["opt"]), str(data["point"])
    hp = {a: (float(data[f"{a}|lr"]), int(data[f"{a}|epochs"])) for a in ("naive", "er")}
    same = hp["naive"] == hp["er"]
    hs = (f"lr {hp['naive'][0]:g} · {hp['naive'][1]} ep/task" if same else
          " · ".join(f"{a}: lr {l:g}/{e} ep" for a, (l, e) in hp.items()))
    return (f"plain net, NO modulation applied · {opt.upper()} ({point}) · {hs} · "
            f"buffer 1000 · seed 42")


def plot_all(data, opt_kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup = _setup_line(data)
    FIGDIR.mkdir(exist_ok=True)
    for key in ALL_DRIVERS:
        formula, fam, dim = META[key]
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0), facecolor=SURFACE)
        note = "batch-mean of the per-sample value" if dim == 1 else \
               f"batch-mean of the per-sample L2 norm ({dim}-d driver)"
        # tall LaTeX (norms, fractions) grows the title box downward, so keep a wide gap to the subtitle
        fig.suptitle(f"{key}   ({fam})      {formula}", fontsize=12, color=INK, x=0.012,
                     ha="left", y=0.988, va="top")
        fig.text(0.012, 0.918, f"{note} · {setup}", fontsize=8, color=INK2, ha="left", va="top")
        for r, basis in enumerate(("raw", "std")):
            for c, phase in enumerate(("train", "test")):
                ax = axes[r][c]
                ax.set_facecolor(SURFACE)
                lab = "non-standardised" if basis == "raw" else "standardised (running stats)"
                _panel(ax, data, key, basis, phase,
                       f"{lab} — {'training' if phase == 'train' else 'test'}")
                if c == 0:
                    ax.set_ylabel(lab.split(" (")[0], fontsize=8, color=INK2)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", frameon=False, fontsize=8.5,
                   labelcolor=INK2, ncol=2, bbox_to_anchor=(0.99, 0.995))
        fig.tight_layout(rect=[0, 0, 1, 0.895])
        fig.savefig(FIGDIR / f"{key}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # contact sheet: every driver, raw, over training
    n = len(ALL_DRIVERS)
    cols, rows = 4, math.ceil(n / 4)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.5 * rows), facecolor=SURFACE)
    fig.suptitle("pt7 neuromodulator drivers — non-standardised value over training",
                 fontsize=14, color=INK, x=0.008, ha="left", y=0.996, va="top")
    fig.text(0.008, 0.972, setup, fontsize=9, color=INK2, ha="left", va="top")
    flat = axes.ravel()
    for i, ax in enumerate(flat):
        ax.set_facecolor(SURFACE)
        if i >= n:
            ax.axis("off"); continue
        _panel(ax, data, ALL_DRIVERS[i], "raw", "train", ALL_DRIVERS[i])
    h, l = flat[0].get_legend_handles_labels()
    # park the legend in the first empty grid slot rather than over the title
    (flat[n] if n < len(flat) else fig).legend(h, l, loc="center", frameon=False, fontsize=12,
                                               labelcolor=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.958])
    fig.savefig(FIGDIR / "_contact_sheet_train_raw.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {n + 1} figures to {FIGDIR}")


# ------------------------------- numeric summary -------------------------------
def summarise(data, log=print):
    """Per-driver range table. `spread` = max/median of |value|: a large spread on the standardised row is
    the tonic-driver blow-up signature (standardising a near-constant divides by ~0)."""
    log("\n  driver      regime phase basis        mean        sd         min         max      spread")
    log("  " + "-" * 92)
    for k in ALL_DRIVERS:
        for arm in ("naive", "er"):
            for phase in ("train", "test"):
                for basis in ("raw", "std"):
                    v = data[f"{arm}|{phase}|{k}|{basis}|mean"]
                    a = np.abs(v[np.isfinite(v)])
                    med = np.median(a[a > 0]) if (a > 0).any() else 0.0
                    spread = (a.max() / med) if med > 0 else float("inf")
                    log(f"  {k:<11s} {arm:<6s} {phase:<5s} {basis:<4s} "
                        f"{np.nanmean(v):>11.3g} {np.nanstd(v):>10.3g} "
                        f"{np.nanmin(v):>11.3g} {np.nanmax(v):>11.3g} {spread:>10.3g}")


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--point", default="tuned", choices=["tuned", "untuned"],
                    help="'tuned' (default): per-arm lr/epochs from configs.TUNED_MAIN[('classil',arm,opt)]; "
                         "'untuned': the inherited pt7 point (lr 1e-3, ep 5) every pt7 driver number sits at")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs/task (smoke tests)")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from the saved npz")
    ap.add_argument("--driver-loss", default="pt7", choices=["pt7", "arm"],
                    help="'pt7' (default): the loss inside DA/5HT/... is masked CE in both arms, as pt7's "
                         "Signals does; 'arm': it follows the arm's training loss (plain CE under er)")
    ap.add_argument("--suffix", default="", help="suffix for the npz/figure outputs (side-by-side variants)")
    args = ap.parse_args()

    global NPZ, FIGDIR
    if args.suffix:
        NPZ = HERE / f"pt7_driver_traces{args.suffix}.npz"
        FIGDIR = HERE / f"pt7_driver_traces{args.suffix}"

    if args.plot_only:
        z = np.load(NPZ, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        plot_all(d, str(z["opt"]))
        summarise(d)
        return

    print(f"device={DEV}  observer-only driver traces  (opt={args.opt}, point={args.point}, seed 42, "
          f"buffer 1000, driver-loss={args.driver_loss})\n", flush=True)
    out = {"opt": np.array(args.opt), "point": np.array(args.point)}
    eps = []
    for arm in ("naive", "er"):
        lr, ep = hparams(args.point, arm, args.opt)
        if args.epochs is not None:
            ep = args.epochs
        eps.append(ep)
        print(f"  running {arm} (lr={lr:g}, {ep} ep/task) ...", flush=True)
        tr, te, acc = run(arm, opt_kind=args.opt, lr=lr, epochs=ep,
                          driver_loss=args.driver_loss, log=lambda s: print(s, flush=True))
        ref = REF.get((args.point, arm, args.opt)) if args.epochs is None else None
        tag = "" if ref is None else f"   (pt7 ledger {ref:.4f}, delta {acc - ref:+.4f})"
        print(f"  {arm:5s} final avg class-IL acc = {acc:.4f}{tag}", flush=True)
        out[f"{arm}|lr"] = np.array(lr); out[f"{arm}|epochs"] = np.array(ep)
        for phase, t in (("train", tr), ("test", te)):
            for (k, b, s), v in t.d.items():
                out[f"{arm}|{phase}|{k}|{b}|{s}"] = np.asarray(v, dtype=np.float32)
            out[f"{arm}|{phase}|bounds"] = np.asarray(t.bounds, dtype=np.int32)
        out[f"{arm}|acc"] = np.asarray(acc, dtype=np.float32)
    np.savez_compressed(NPZ, **out)
    print(f"  wrote {NPZ}", flush=True)
    plot_all(out, args.opt)
    summarise(out, log=lambda s: print(s, flush=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
