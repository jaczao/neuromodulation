"""NOVELTY-DRIVER FORM under the TEMPERATURE and SLOPE gate forms (class-IL).

Sibling of novelty_drivers.py: the same drivers and the same four axes (norm, mean_mode,
standardisation, learned-vs-frozen projection), but the rank-K LINEAR gain gate
`Gamma = 1 + m @ P` is replaced by the two positive EXPONENTIAL SCALAR forms of
results/pt7_plast_tempslope.py:

    temp   logits *= exp(m @ p_out)                 — softmax temperature, out layer only
    slope  h0 *= exp(m @ p_h0);  h1 *= exp(m @ p_h1) — per-hidden-layer ReLU-slope gain

`p` is a (K,) VECTOR, so `m @ p` is one SCALAR per sample and the gain is uniform across the layer.
That is the structural difference from the gain study, and it drives every prediction below:

  1. THE GATE IS UNIFORM, NOT PER-UNIT. The linear gate gives each of the 810 gated units its own
     coefficient; here a whole layer is scaled by one number. So the "vector form says WHERE a sample
     is unusual" story has no way to express itself even in principle — all K driver dimensions are
     collapsed into a single scalar before anything is modulated. If the vector form still differs
     from the norm form here, it is through the SCALE and CONDITIONING of that scalar, not through
     any spatial structure.

  2. THE PARAMETER COUNT COLLAPSES, so the capacity confound disappears. `p` is K (temp) or 2K
     (slope) parameters — 784 and 1,568 at the widest, i.e. 0.0016x and 0.0033x the backbone, against
     the linear gate's 635,040 (1.33x) for the same driver. The gain study could not separate "the
     784-d vector form is bad" from "635k extra trainable parameters are bad"; this study has no such
     confound anywhere in it, which makes it the cleaner test of the norm axis.

  3. exp() IS THE OPPOSITE OF FORGIVING. The linear gate is affine in m, so a large driver produces a
     large but finite deviation; `exp(m @ p)` is exponential in the driver, and pt7 already measured
     the consequence — bounded drivers land on ER, while large unstandardised ones blow the gate up
     and collapse to chance. The raw drivers here span |m| ~ 0.5 (vector) to ~23 (vec_x norm), so the
     std axis is expected to matter far MORE than it did for the linear gate, and in the opposite
     direction: there, standardising a vector with constant dimensions was the thing that detonated.

  4. temp IS ARGMAX-INVARIANT AT EVAL. A uniform positive scale on the logits cannot change which
     logit is largest, so `temp` has NO inference-time lever at all: its entire effect is a per-sample
     reweighting of the TRAINING loss by novelty. `slope` is in the eval forward and does have one.
     pt7's asymmetry (a blown-up `temp` degrades far less than a blown-up `slope`, 0.39 vs 0.10)
     follows from exactly this, and is re-testable here across the whole grid rather than at one point.

AXES (128 grid cells): form {temp, slope} x kind {vec_x, vecproj} x norm {0,1} x std {0,1} x
mean_mode {ema, cumulative, trueavg, ema+trueavg} x proj {learned, random}. `mean_mode` is included
even though the request enumerated only the other three axes: it is a superset, so the table answers
either reading, and dropping it would have lost the trueavg / ema+trueavg comparison under a new gate
form — which is the one thing this pair of studies exists to vary.

ARM / OPERATING POINT: class-IL er-own, Adam, buffer 1000, lr 3e-4 / 5 ep/task from
`neurocore.tuned` — NOT retuned, the same val-selected ER point the gain study used, so the two are
directly comparable. 1 seed; the 1-seed MPS noise floor is 0.007.

CONTROLS: `dead` freezes p at ZERO, so exp(0) = 1 exactly and the gate is provably inert — the
rule-#10 RNG-matched baseline, checked against the plain ungated run rather than assumed. `random`
freezes p at N(0, 0.1^2), the same scale the gain study used, so "learned vs frozen" means the same
thing in both. NOTE the frozen scale is NOT scale-free here: under exp() a frozen p interacts
multiplicatively with the driver's raw magnitude, so a saturated random arm is a statement about
sigma = 0.1 for THIS driver, not about frozen projections in general.

Run:  uv run python novelty_drivers/tempslope.py --part anchor
      uv run python -m neurocore.shard --script novelty_drivers/tempslope.py \
          --ledger novelty_drivers/tempslope_results.tsv \
          --split forms=temp,slope --split kinds=vec_x,vecproj --split norms=0,1 \
          --args "--part grid --resume" --workers 6 --device mps
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))
# HERE goes FIRST so `novelty_drivers` resolves to the sibling MODULE, not to this package
# DIRECTORY. The repo convention is <package>/<same-name>.py, so the two are ambiguous: run under
# neurocore.shard (cwd = ROOT) the directory wins and `novelty_drivers.novelty_drivers` imports,
# but run directly (sys.path[0] = HERE) the file wins and the dotted form dies with "'
# novelty_drivers' is not a package". Importing the file by its bare name works under BOTH.
sys.path.insert(0, str(HERE))

import pt7_neuromodulators as p7                                   # noqa: E402  (frozen, read-only)
from prototype.data import SplitMNIST                              # noqa: E402
from neurocore import shard                                        # noqa: E402
from neurocore.buffers import Reservoir                            # noqa: E402
from neurocore.controls import probe as task_probe                 # noqa: E402
from neurocore.cost import Cost, count_params                      # noqa: E402
from neurocore.ledger import Ledger                                # noqa: E402
from neurocore.signals import dataset_mean                         # noqa: E402
from neurocore.tuned import tuned_main                             # noqa: E402
from neurocore.utils import DEV, seed_all                          # noqa: E402
from novelty_drivers import (BUFFER, KINDS, MEAN_MODES, PROJ_SEED_BASE, PROJ_STD,   # noqa: E402
                             PROJS, SEED, build_driver, cl_true_mean)

CE = nn.CrossEntropyLoss()
FORMS = ("temp", "slope")

TSV = shard.ledger_path(Path(__file__).resolve().parent / "tempslope_results.tsv")
KEYS = ["form", "kind", "std", "norm", "mean_mode", "proj", "opt", "regime", "seed"]
METRICS = ["acc", "probe", "mabs", "g_h0", "g_h1", "g_out"]

# Same bit-exact reference the gain study anchors on (results/pt7_tuned_syn: report|er|adam, macro).
ANCHOR_CLASSIL = 0.8975


# ============================================================ the gate
class GateForm(nn.Module):
    """exp(m @ p) applied as a per-sample SCALAR gain. Copy-forward of
    results/pt7_plast_tempslope.GateForm, with the projection made freezable.

    p is zero-init, so exp(0) = 1 and the gate starts at exact parity — the same init-parity property
    the `unbounded` linear form has, reached multiplicatively.
    """

    def __init__(self, form, K):
        super().__init__()
        self.form = form
        if form == "temp":
            self.p_out = nn.Parameter(torch.zeros(K))
        else:
            self.p_h0 = nn.Parameter(torch.zeros(K)); self.p_h1 = nn.Parameter(torch.zeros(K))
        self.to(DEV)

    def params(self):
        return [self.p_out] if self.form == "temp" else [self.p_h0, self.p_h1]

    def forward(self, net, m, x):
        x = x.view(x.size(0), -1)
        if self.form == "temp":
            h0 = F.relu(net.l0(x)); h1 = F.relu(net.l1(h0))
            return net.l2(h1) * torch.exp(m @ self.p_out).unsqueeze(1)
        h0 = F.relu(net.l0(x)) * torch.exp(m @ self.p_h0).unsqueeze(1)
        h1 = F.relu(net.l1(h0)) * torch.exp(m @ self.p_h1).unsqueeze(1)
        return net.l2(h1)

    @torch.no_grad()
    def mag(self, m):
        """|gain - 1| per layer. Zero on the layers this form does not touch, which is the record of
        WHICH lever each form actually pulls."""
        if self.form == "temp":
            return {"h0": 0.0, "h1": 0.0, "out": (torch.exp(m @ self.p_out) - 1).abs().mean().item()}
        return {"h0": (torch.exp(m @ self.p_h0) - 1).abs().mean().item(),
                "h1": (torch.exp(m @ self.p_h1) - 1).abs().mean().item(), "out": 0.0}


def build_gate(form, K, proj, seed=SEED):
    """(gate, trainable). `learned` is the only mode the optimizer sees.

    Neither `random` nor `dead` touches the global torch RNG — the frozen draw comes from a private
    generator and the dead p is `torch.zeros` — which is what makes `dead` an RNG-matched baseline.
    """
    gate = GateForm(form, K)
    if proj == "learned":
        return gate, True
    g = torch.Generator().manual_seed(PROJ_SEED_BASE + seed)
    for p in gate.params():
        r = torch.zeros(p.shape) if proj == "dead" else torch.randn(p.shape, generator=g) * PROJ_STD
        p.data = r.to(DEV)
        p.requires_grad_(False)
    return gate, False


def cell_cost(gate, buf):
    net_params = 784 * 400 + 400 + 400 * 400 + 400 + 400 * 10 + 10
    return Cost(backbone_params=net_params,
                extra_params=count_params(gate) if gate is not None else 0,
                buffer_bytes=buf.nbytes() if buf is not None else 0,
                # vec_x / vecproj are PRE-forward, so the gate costs no extra pass at train or eval.
                fwd_train=1.0, bwd_train=1.0, fwd_infer=1.0, bwd_infer=0.0)


# ============================================================ the loop
def run_cell(form, kind, norm, mean_mode, std, proj, lr, epochs, loaders, seed=SEED):
    """class-IL er-own: main net and (when learned) p trained jointly on the ER batch under plain CE.
    Same loop as the gain study with the gate swapped, so the two are cell-for-cell comparable."""
    seed_all(seed)
    net = p7.Net().to(DEV)
    drv = build_driver(kind, norm, mean_mode, std)
    gate, trainable = build_gate(form, drv.K(), proj, seed)
    params = list(net.parameters()) + (gate.params() if trainable else [])
    opt = torch.optim.Adam(params, lr)
    buf = Reservoir(BUFFER)
    for t in range(5):
        if drv.mean_mode == "trueavg":
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
    if drv.uses_true_mean(inference=True):
        drv.set_true_mean(dataset_mean([loaders[i][1] for i in range(5)], space="x"))
    res = evaluate(net, gate, drv, loaders)
    res["cost"] = cell_cost(gate, buf)
    return res


def run_plain(lr, epochs, loaders, seed=SEED):
    """Plain ungated ER — the bit-exact anchor, identical to the gain study's."""
    seed_all(seed)
    net = p7.Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr)
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
    return dict(acc=acc, probe=float("nan"), mabs=0.0, g_h0=0.0, g_h1=0.0, g_out=0.0,
                cost=cell_cost(None, buf))


@torch.no_grad()
def evaluate(net, gate, drv, loaders):
    """Macro class-IL accuracy under the gate, plus |gain - 1| per layer, mean |m| and the probe.

    Accuracy is the MEAN OF THE FIVE PER-TASK ACCURACIES, the same convention the gain study and its
    baselines use. Diagnostics are pooled and guarded against a diverged run — under exp() a
    divergence is an expected outcome, not an error, and a NaN probe input would otherwise raise
    instead of recording it.
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
            pl = gate.mag(m)
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
def point():
    tp = tuned_main("splitmnist", "classil", "er", "adam")
    return tp["lr"], tp["epochs_per_task"]


def build_cells(part, stds):
    """[(form, kind, std, norm, mean_mode, proj)]; form None = the ungated plain baseline."""
    cells = []
    if part in ("all", "anchor"):
        cells.append((None, None, "-", 0, "-", "-"))
        for form in FORMS:
            for std in stds:
                for kind in KINDS:
                    for norm in (0, 1):
                        cells.append((form, kind, std, norm, "ema", "dead"))
    if part in ("all", "grid"):
        for form in FORMS:
            for std in stds:
                for kind in KINDS:
                    for norm in (0, 1):
                        for mm in MEAN_MODES:
                            for proj in PROJS:
                                cells.append((form, kind, std, norm, mm, proj))
    return cells


def fmt(r):
    pr = "probe=nan" if not np.isfinite(r["probe"]) else f"probe={r['probe']:.3f}"
    return (f"acc={r['acc']:.4f}  {pr}  |m|={r['mabs']:.3g}  "
            f"|g-1|(h0/h1/out)={r['g_h0']:.3g}/{r['g_h1']:.3g}/{r['g_out']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "anchor", "grid", "table"])
    ap.add_argument("--forms", default=None, help="comma filter: temp,slope")
    ap.add_argument("--kinds", default=None, help="comma filter: vec_x,vecproj")
    ap.add_argument("--norms", default=None, help="comma filter: 0,1")
    ap.add_argument("--std", default="0,1", help="standardisation arm(s): 0, 1 or 0,1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    led = Ledger(TSV, keys=KEYS, metrics=METRICS, with_cost=True)
    if args.part == "table":
        print(table(led.rows()))
        return

    lr, ep = point()
    print(f"device={DEV}  temp/slope gate forms, class-IL er-own, adam, 1 seed, NOT retuned\n"
          f"  lr {lr:g}, {ep} ep/task, buffer {BUFFER}   [anchor {ANCHOR_CLASSIL:.4f}]\n", flush=True)
    ffil = set(args.forms.split(",")) if args.forms else None
    kfil = set(args.kinds.split(",")) if args.kinds else None
    nfil = {int(v) for v in args.norms.split(",")} if args.norms else None
    stds = [int(v) for v in args.std.split(",")]
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(5)]

    for form, kind, std, norm, mm, proj in build_cells(args.part, stds):
        if form is not None:
            if ffil and form not in ffil:
                continue
            if kfil and kind not in kfil:
                continue
            if nfil and norm not in nfil:
                continue
        key = dict(form=form or "none", kind=kind or "none", std=std, norm=norm, mean_mode=mm,
                   proj=proj, opt="adam", regime="normal", seed=args.seed)
        if args.resume and led.is_done(**key):
            continue
        if form is None:
            r = run_plain(lr, ep, loaders, seed=args.seed)
            note = f"   [anchor {ANCHOR_CLASSIL:.4f}, d={r['acc'] - ANCHOR_CLASSIL:+.4f}]"
        else:
            r = run_cell(form, kind, bool(norm), mm, std, proj, lr, ep, loaders, seed=args.seed)
            note = ""
        print(f"  {form or 'plain':6s} {kind or '-':8s} std{std} norm{norm} {mm:12s} {proj:7s} | "
              f"{fmt(r)}{note}", flush=True)
        led.append(key, {k: r[k] for k in METRICS}, cost=r["cost"])
    print("ALL SELECTED CELLS DONE", flush=True)


# ============================================================ table
def table(rows):
    """One block per (form, kind, norm): the two standardisation arms side by side, vs the dead gate.

    exp(0) = 1 exactly, so the dead gate is parity whatever the driver does — all dead cells are one
    control, and the header reports their spread as the check that this holds.
    """
    out = []
    plain = next((float(r["acc"]) for r in rows if r["form"] == "none"), float("nan"))
    deads = [r for r in rows if r["proj"] == "dead"]
    dead = float(np.mean([float(r["acc"]) for r in deads])) if deads else None
    head = f"\n=== class-IL er-own (adam, tuned)   plain {plain:.4f}"
    if dead is not None:
        spread = max(float(r["acc"]) for r in deads) - min(float(r["acc"]) for r in deads)
        head += f"   dead {dead:.4f}   (n={len(deads)}, spread {spread:.6f}, d-plain {dead - plain:+.6f})"
    out.append(head)
    for form in FORMS:
        for kind in KINDS:
            for norm in (0, 1):
                sel = [r for r in rows if r["form"] == form and r["kind"] == kind
                       and int(r["norm"]) == norm and r["proj"] in PROJS]
                if not sel:
                    continue
                ex = sel[0]
                out.append(f"\n  {form} · {kind} norm{norm}   (extra params "
                           f"{int(ex['extra_params']):,}, {float(ex['param_ratio']):.4f}x backbone)")
                out.append(f"    {'':<22s}{'--- std=0 (raw) ---':^36s}  {'--- std=1 ---':^36s}")
                out.append(f"    {'mean_mode':<13s}{'proj':<9s}" +
                           2 * f"{'acc':>9s}{'d-dead':>9s}{'probe':>8s}{'|g-1|':>10s}")
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
                            g = max(float(r["g_h0"]), float(r["g_h1"]), float(r["g_out"]))
                            line += (f"{float(r['acc']):>9.4f}{dd:>9s}"
                                     f"{(f'{pr:.3f}' if np.isfinite(pr) else '-'):>8s}{g:>10.3g}")
                        if any_row:
                            out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    main()
