"""CAPACITY ABLATION for the SIGNAL NET and the SIGNAL NET + GRU (user-requested).

WHAT THIS EXTENDS. `results/pt7_capacity.py` asked whether the pt5/6/7 gating null is an
OVER-PARAMETERIZATION artifact — shrink the hidden width H until the baseline itself degrades, then watch
the (gate - baseline) delta. Its answer for the canonical pt7 `all4` gain-neuron gate was NO: the delta
never emerged, all the way down to a collapsed H=5 backbone, even though gate ENGAGEMENT rose ~15x. That
sweep ran ONE mechanism. This one runs the two richest mechanisms in the project at the same widths:

  signalnet      a 23-dim signal feature vector -> MLP 23-64-64-64-K -> rank-K gate over (h0,h1,out)
  signalnet-gru  the same, with the low-D code passed through a stateful GRUCell before the gate

both in the ACTUAL-ENTROPY variant (see below), at H in {400, 10, 5} (the user's grid: one healthy width
plus the two degraded ones where `pt7_capacity` found the baseline unstable and the verdict is decided).

THE TWO DEVIATIONS FROM `results/pt7_signalnet.py`, both mandatory:

(1) ENGAGE=TRUE. The canonical config zero-inits BOTH the signal net's output layer and the gate P, which
    is the double-zero-init saddle: dL/dP is proportional to m = 0 and dL/d(snet) is proportional to
    P = 0, so neither bootstraps and the code stays EXACTLY 0 for the whole run. That cell is numerically
    pt7's `free` control, not a test of the mechanism. `engage=True` gives the module's OUTPUT layer a
    normal init while keeping P zero-init, so gamma = 1 at step 0 (parity) but the code can bootstrap.

(2) ACTUAL ENTROPY. Feature column 8, "predicted entropy of the current sample", is replaced by the ACTUAL
    entropy from an extra unmodulated forward (entropy needs no labels, so it is exactly computable at
    inference). This is `driver_traces/signalnet_traces.py`'s variant, which at H=400 moved signalnet
    0.5215 -> 0.7137 and signalnet-gru 0.8657 -> 0.8845. The substitution goes through a shim on
    `SignalHeads`, never by editing `SignalFeatures.build`: `build` reads col 8 as `pred[:, 0:1]`, so
    replacing head output column 0 changes exactly that one feature, and does so BEFORE the 23-dim
    standardisation, which is the only place it can land without making the running stats inconsistent
    with the values they normalise. The head's own MSE training step still sees the UNWRAPPED head.

ARMS per (H, seed). The dead-gate controls are the point of the design, not padding:
    er            plain ER, Adam. Context only — NOT RNG-matched to the gate arms.
    sn-dead       signalnet, engage=FALSE. The double-zero saddle pins |g| at exactly 0, so this is
                  numerically ER, but it constructs the identical modules and therefore consumes the
                  IDENTICAL torch RNG -> it is the RNG-MATCHED baseline for `sn`. pt7_capacity measured
                  that this matters ~0.002 at H=400 and ~0.06 at H=5: reading against plain `er` at a
                  collapsing width manufactures a spurious win. `d-dead` is the honest delta.
    sn            signalnet, engage=TRUE, actual-H. The mechanism under test.
    sngru-dead    signalnet-gru, engage=FALSE. A SEPARATE control: the GRUCell consumes extra RNG, so
                  sn-dead and sngru-dead are different draws and are not interchangeable.
    sngru         signalnet-gru, engage=TRUE, actual-H.
The dead arms are insensitive to the actual-H shim (code is identically 0 whatever the features are), so
they are run once and shared between the two entropy variants.

THE CAPACITY CONFOUND, STATED UP FRONT. The modulator's own size is held CONSTANT while the backbone
shrinks, so the `mod/net` column in the table is the number to read first. At H=400 the modulator is ~0.1x
the backbone; at H=5 it is ~9-12x it. pt7_capacity's `er+freefix` arm hit +0.334 at H=5 with a
CONTENT-FREE gate purely by adding parameters — so any positive delta at H=5 here is a capacity-addition
candidate before it is a neuromodulation result, and the ratio must be reported alongside it.

NOT re-tuned per width (same caveat as pt7_capacity): lr/epochs are the standard pt7 point at every H, so
ABSOLUTE accuracies at small H are pessimistic. Every arm at a given H gets the identical budget, so the
matched-H DELTA — the quantity this study is about — is unaffected.

IMPLEMENTATION. Width enters ONLY by rebinding p7's H0/H1/GATEDIM module globals inside the `width(H)`
context manager (p7.Net / NeuronGate read them at CALL time, never captured at import), exactly as
pt7_capacity does. The training loop is a COPY-FORWARD of `pt7_signalnet.run_signalnet` + `_eval_signalnet`
rather than a call into them, per the repo extraction rule (frozen study keeps reproducing untouched) and
because the shim has to sit inside the loop. Module construction ORDER matches the frozen study exactly, so
H=400/seed42 reproduces both frozen references bit-exact — checked by `--part anchor`:
    pred-H    signalnet 0.5215  signalnet-gru 0.8657   (results/pt7_signalnet_results.tsv, `eng` rows)
    actual-H  signalnet 0.7137  signalnet-gru 0.8845   (driver_traces/signalnet_traces.md)
    er        0.8946                                    (pt7 ledger)

Eval protocol is FROZEN (`update=False`, running scalars frozen at test) — the frozen study's protocol and
the one both anchors were measured under. This study does not inherit live_traces.py's live-stats premise.

Ledger signalnet_capacity_results.tsv (`--resume` skips ledgered cells). Log signalnet_capacity.log.
Table `--part table`; plots `--part plot` -> figs/*.png.
"""
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "results"))
import pt7_neuromodulators as p7                                     # noqa: E402
import pt7_signalnet as psn                                          # noqa: E402
sys.path.insert(0, str(REPO / "prototype"))
from data import SplitMNIST                                          # noqa: E402

DEV = p7.DEV
CE = nn.CrossEntropyLoss()
TSV = HERE / "signalnet_capacity_results.tsv"
FIGDIR = HERE / "figs"

WIDTHS = [400, 10, 5]
SEEDS = [42, 43, 44]
ARMS = ["er", "sn-dead", "sn", "sngru-dead", "sngru"]
PAIR = {"sn": "sn-dead", "sngru": "sngru-dead"}                      # mechanism -> its RNG-matched control

K = 4
LR, EPOCHS, BUFFER, OPT, GRAN = 1e-3, 5, 1000, "adam", "neuron"

# H=400 / seed 42 bit-exact anchors. pred-H: results/pt7_signalnet_results.tsv (`eng` rows).
# actual-H: driver_traces/signalnet_traces.md (frozen-protocol column). er: the pt7 ledger.
ANCHOR = {("er", None): 0.8946,
          ("sn", False): 0.5215, ("sngru", False): 0.8657,
          ("sn", True): 0.7137, ("sngru", True): 0.8845}


@contextmanager
def width(H):
    """Rebind the pt7 module globals that define the main net's hidden width.

    p7.Net / NeuronGate read H0/H1/GATEDIM at CALL time, so rebinding them here makes every downstream
    pt7/psn function build and run an H-wide net with no other edit. Single-threaded and sequential;
    restored on exit. NOTE psn captured H0/H1/OUT/GATEDIM at import into its own module globals, but only
    `H1Gate` (a different part of that study) reads them — the signalnet path goes through p7.make_gate
    and p7.Net, both of which read p7's globals live.
    """
    old = (p7.H0, p7.H1, p7.GATEDIM)
    p7.H0 = p7.H1 = H
    p7.GATEDIM = 2 * H + p7.OUT
    try:
        yield
    finally:
        p7.H0, p7.H1, p7.GATEDIM = old


def n_params(H):
    """Main-net parameter count for a width-H MLP 784->H->H->10."""
    return 784 * H + H + H * H + H + H * 10 + 10


def n_mod_params(H, use_gru):
    """Modulator parameter count: SignalHeads + SignalNet + gate P (+ GRUCell). Held constant in H except
    for P, whose width is GATEDIM = 2H + 10 — i.e. the modulator barely shrinks while the backbone does."""
    heads = 784 * 32 + 32 + 32 * psn.NPRED + psn.NPRED
    snet = 23 * 64 + 64 + 2 * (64 * 64 + 64) + 64 * K + K
    proj = K * (2 * H + p7.OUT)
    gru = (3 * (64 * K + 64 * 64 + 2 * 64) + (K + 64) * K + K) if use_gru else 0
    return heads + snet + proj + gru


# ------------------------------- the actual-entropy shim -------------------------------
class ActualHShim(nn.Module):
    """Wraps SignalHeads so output column 0 carries the ACTUAL entropy instead of the head's prediction.

    Copy-forward of `driver_traces/signalnet_traces.ActualHShim` (importing that module would pull in its
    matplotlib/figure stack for a 10-line class). Holds no parameters, so building it consumes no RNG and
    the pred-H arm stays bit-identical to the frozen study.
    """

    def __init__(self, heads, net):
        super().__init__()
        self.heads = heads
        self.net = net

    def forward(self, x):
        p = self.heads(x)
        with torch.no_grad():
            H = p7.entropy(self.net.plain(x)[0])
        return torch.cat([H.unsqueeze(1), p[:, 1:]], 1)


# ------------------------------- run (copy-forward of psn.run_signalnet) -------------------------------
def run_signalnet(use_gru, engage, actual_h, seed, epochs=EPOCHS):
    """Copy-forward of `pt7_signalnet.run_signalnet` (gran=neuron, K=4, standardize=True, adam) with the
    col-8 shim optionally installed. Construction order is byte-identical to the frozen study
    (Net -> gate -> SignalHeads -> SignalFeatures -> SignalNet -> GRUOnVec), which is what makes the
    H=400 anchors reproduce; the shim is built afterwards and holds no parameters."""
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV)
    gate = psn._mk(GRAN, K)
    heads = psn.SignalHeads().to(DEV)
    feat = psn.SignalFeatures(True)
    snet = psn.SignalNet(K, engage=engage).to(DEV)
    gru = psn.GRUOnVec(K, K, engage=engage).to(DEV) if use_gru else None
    fheads = ActualHShim(heads, net) if actual_h else heads
    params = list(net.parameters()) + gate.params() + list(snet.parameters()) \
        + (list(gru.parameters()) if gru else [])
    main_opt = p7._opt(OPT, params, LR)
    head_opt = torch.optim.Adam(heads.parameters(), LR)
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
                f = feat.build(net, fheads, Xm, Ym, update=True).detach()
                code = snet(f)                                       # (B,K)
                m = gru(code) if gru else code
                loss = CE(gate(net, m, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                T = feat.targets(net, Xm, Ym)                        # component-head regression (+replay)
                hloss = F.mse_loss(heads(Xm), T)                     # the RAW head, never the shim
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)
    return evaluate(net, gate, feat, snet, gru, fheads, loaders)


@torch.no_grad()
def evaluate(net, gate, feat, snet, gru, fheads, loaders):
    """Copy-forward of `psn._eval_signalnet`. FROZEN protocol (update=False): the running scalars stay at
    their end-of-training values, which is the protocol both H=400 anchors were measured under."""
    net.eval()
    c = tot = 0
    mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            f = feat.build(net, fheads, x, y=None, update=False)
            code = snet(f)
            m = gru(code) if gru else code
            c += (gate(net, m, x).argmax(1) == y).sum().item()
            pl = gate.per_layer_mag(m)
            for k in mags:
                mags[k] += pl[k] * b
            tot += b
    return {"acc": c / tot, **{k: v / tot for k, v in mags.items()}}


def run_arm(arm, H, seed, actual_h=True, epochs=EPOCHS):
    """-> dict(acc=..., h0/h1/out=...). The `er` baseline reports acc only."""
    with width(H):
        if arm == "er":
            return {"acc": p7.train_baseline("er", OPT, lr=LR, epochs=epochs,
                                             buffer=BUFFER, seed=seed)}
        use_gru = arm.startswith("sngru")
        engage = not arm.endswith("-dead")
        return run_signalnet(use_gru, engage, actual_h, seed, epochs=epochs)


# ------------------------------- ledger -------------------------------
def load_rows():
    if not TSV.exists():
        return []
    rows = []
    for ln in TSV.read_text().splitlines():
        if not ln.strip():
            continue
        f = ln.split("\t")
        rows.append({"H": int(f[0]), "seed": int(f[1]), "arm": f[2], "acc": float(f[3]),
                     "h0": _f(f, 4), "h1": _f(f, 5), "out": _f(f, 6)})
    return rows


def _f(fields, i):
    return float(fields[i]) if i < len(fields) and fields[i] not in ("", "-") else float("nan")


def record(H, seed, arm, r):
    cells = [H, seed, arm, f"{r['acc']:.4f}"]
    if "h0" in r:
        cells += [f"{r['h0']:.4f}", f"{r['h1']:.4f}", f"{r['out']:.4f}"]
    with open(TSV, "a") as f:
        f.write("\t".join(str(c) for c in cells) + "\n")


# ------------------------------- aggregation -------------------------------
def agg(rows, H, arm):
    a = [r["acc"] for r in rows if r["H"] == H and r["arm"] == arm]
    return (float(np.mean(a)), float(np.std(a)), len(a)) if a else None


def paired_delta(rows, H, arm, ref):
    """Per-seed paired delta arm-ref at width H -> (mean, sd, n, per-seed list).

    Seeds come from the LEDGER, never from the SEEDS constant, so a width carrying extra seeds cannot
    silently pair only the first three while the mean column averages all of them (the pt7_capacity trap).
    """
    d = []
    for s in sorted({r["seed"] for r in rows if r["H"] == H}):
        a = [r["acc"] for r in rows if r["H"] == H and r["arm"] == arm and r["seed"] == s]
        b = [r["acc"] for r in rows if r["H"] == H and r["arm"] == ref and r["seed"] == s]
        if a and b:
            d.append(a[0] - b[0])
    if not d:
        return None
    return float(np.mean(d)), float(np.std(d)), len(d), d


def print_table(rows):
    print("\n  accuracy by width (mean +- s.d. over seeds; actual-entropy variant, frozen eval)", flush=True)
    print("  " + f"{'H':<5s}{'net':>9s}{'n':>3s}" + "".join(f"{a:>17s}" for a in ARMS), flush=True)
    for H in WIDTHS:
        line = f"  {H:<5d}{n_params(H):>9,d}"
        n = max([agg(rows, H, a)[2] for a in ARMS if agg(rows, H, a)] or [0])
        line += f"{n:>3d}"
        for a in ARMS:
            g = agg(rows, H, a)
            line += f"{g[0]:>11.4f}+-{g[1]:<4.3f}" if g else f"{'-':>17s}"
        print(line, flush=True)

    print("\n  paired deltas (per seed). d-dead is the honest one: `*-dead` is the RNG-matched", flush=True)
    print("  dead-gate control (|g| == 0 by the double-zero saddle), plain `er` is NOT.", flush=True)
    print("  " + f"{'H':<5s}{'mod/net':>9s}" + f"{'sn d-dead':>17s}{'sn d-er':>17s}"
          + f"{'sngru d-dead':>17s}{'sngru d-er':>17s}", flush=True)
    for H in WIDTHS:
        ratio = n_mod_params(H, False) / n_params(H)
        line = f"  {H:<5d}{ratio:>8.2f}x"
        for arm in ("sn", "sngru"):
            for ref in (PAIR[arm], "er"):
                d = paired_delta(rows, H, arm, ref)
                if d:
                    sem = (float(np.std(d[3], ddof=1)) / np.sqrt(d[2])) if d[2] > 1 else 0.0
                    line += f"{d[0]:>+11.4f}+-{sem:<4.3f}"
                else:
                    line += f"{'-':>17s}"
        print(line, flush=True)

    print("\n  gate engagement, mean over seeds: per-layer |gate deviation|", flush=True)
    print("  " + f"{'H':<5s}{'arm':<12s}{'|g| h0':>10s}{'|g| h1':>10s}{'|g| out':>10s}", flush=True)
    for H in WIDTHS:
        for arm in ("sn-dead", "sn", "sngru-dead", "sngru"):
            sel = [r for r in rows if r["H"] == H and r["arm"] == arm]
            if not sel:
                continue
            g = [float(np.mean([r[k] for r in sel])) for k in ("h0", "h1", "out")]
            print(f"  {H:<5d}{arm:<12s}{g[0]:>10.4f}{g[1]:>10.4f}{g[2]:>10.4f}", flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(exist_ok=True)
    have = [H for H in WIDTHS if agg(rows, H, "er")]
    if not have:
        print("  nothing to plot yet", flush=True)
        return
    style = {"er": ("#1f77b4", "s", "ER"),
             "sn-dead": ("#9ecae1", "^", "signalnet dead gate (RNG-matched)"),
             "sn": ("#d62728", "D", "signalnet (actual-H, engaged)"),
             "sngru-dead": ("#a1d99b", "v", "signalnet-gru dead gate (RNG-matched)"),
             "sngru": ("#2ca02c", "o", "signalnet-gru (actual-H, engaged)")}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    for arm in ARMS:
        xs = [H for H in have if agg(rows, H, arm)]
        if not xs:
            continue
        m = [agg(rows, H, arm)[0] for H in xs]; s = [agg(rows, H, arm)[1] for H in xs]
        c, mk, lab = style[arm]
        ax.errorbar(xs, m, yerr=s, marker=mk, color=c, label=lab, capsize=3, lw=1.8, ms=6)
    ax.set_xscale("log"); ax.set_xticks(WIDTHS); ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.invert_xaxis()
    ax.set_xlabel("hidden width H  (both layers; shrinking ->)")
    ax.set_ylabel("final avg class-IL accuracy  (±1 s.d. over seeds)")
    ax.set_title("Signal-net capacity ablation — accuracy vs width\n"
                 f"(Split MNIST, Adam, lr {LR} / ep {EPOCHS} / buffer {BUFFER}, actual entropy)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for arm, c, mk in (("sn", "#d62728", "D"), ("sngru", "#2ca02c", "o")):
        xs, m, s = [], [], []
        for H in have:
            d = paired_delta(rows, H, arm, PAIR[arm])
            if d:
                sd = float(np.std(d[3], ddof=1)) if d[2] > 1 else 0.0
                xs.append(H); m.append(d[0]); s.append(sd / np.sqrt(d[2]))
        if xs:
            ax.errorbar(xs, m, yerr=s, marker=mk, color=c, capsize=3, lw=1.8, ms=6,
                        label=f"{arm} − dead gate (RNG-matched, paired)")
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-0.007, 0.007, color="k", alpha=0.08, label="1-seed noise floor (±0.007)")
    ax.set_xscale("log"); ax.set_xticks(WIDTHS); ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.invert_xaxis()
    ax.set_xlabel("hidden width H  (shrinking ->)")
    ax.set_ylabel("paired accuracy delta  (±1 s.e.m.)")
    ax.set_title("Does the signal net start helping once capacity is scarce?\n(paired per seed)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    p = FIGDIR / "signalnet_capacity.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}", flush=True)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for arm, c in (("sn", "#d62728"), ("sngru", "#2ca02c")):
        for k, ls in (("h0", "-"), ("h1", "--"), ("out", ":")):
            xs, ys = [], []
            for H in WIDTHS:
                sel = [r[k] for r in rows if r["H"] == H and r["arm"] == arm]
                if sel:
                    xs.append(H); ys.append(max(float(np.mean(sel)), 1e-6))
            if xs:
                ax.plot(xs, ys, marker="o", ls=ls, color=c, label=f"{arm} |g| {k}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(WIDTHS); ax.set_xticklabels([str(w) for w in WIDTHS]); ax.invert_xaxis()
    ax.set_xlabel("hidden width H  (shrinking ->)"); ax.set_ylabel("mean |gate deviation|")
    ax.set_title("Gate engagement vs capacity")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGDIR / "gate_engagement.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}", flush=True)


# ------------------------------- driver -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="sweep", choices=["sweep", "anchor", "smoke", "table", "plot"])
    ap.add_argument("--widths", default=None, help="comma filter, e.g. 400,10")
    ap.add_argument("--seeds", default=None, help="comma filter, e.g. 42")
    ap.add_argument("--arms", default=None, help="comma filter on arm")
    ap.add_argument("--resume", action="store_true", help="skip cells already in the ledger")
    args = ap.parse_args()

    if args.part in ("table", "plot"):
        rows = load_rows()
        print_table(rows)
        if args.part == "plot":
            plot(rows)
        return

    print(f"device={DEV}  signalnet capacity ablation | class-IL | Adam lr={LR} ep={EPOCHS} "
          f"buffer={BUFFER} | gain-{GRAN} K={K} | actual entropy | engage=True\n", flush=True)

    if args.part == "smoke":
        import time
        for H in WIDTHS:
            with width(H):
                net = p7.Net().to(DEV)
                assert net.l0.out_features == H and net.l1.out_features == H
                print(f"  H={H:<4d} net={n_params(H):>8,d}  mod(sn)={n_mod_params(H, False):>8,d} "
                      f"({n_mod_params(H, False) / n_params(H):.2f}x)  "
                      f"mod(sngru)={n_mod_params(H, True):>8,d} "
                      f"({n_mod_params(H, True) / n_params(H):.2f}x)", flush=True)
        t0 = time.time()
        r = run_arm("sn", 400, 42, epochs=1)
        dt = time.time() - t0
        print(f"  1-epoch sn @H=400: acc={r['acc']:.4f} |g|={r['h0']:.4f}/{r['h1']:.4f}/{r['out']:.4f}"
              f"  [{dt:.0f}s -> ~{dt * EPOCHS / 60:.1f} min per {EPOCHS}-epoch cell]", flush=True)
        assert (p7.H0, p7.H1, p7.GATEDIM) == (400, 400, 810), "width() failed to restore globals"
        print("  globals restored OK", flush=True)
        return

    if args.part == "anchor":
        print("  H=400 seed 42 must reproduce the frozen references bit-exact:", flush=True)
        for (arm, ah), exp in ANCHOR.items():
            r = run_arm(arm, 400, 42, actual_h=bool(ah))
            tag = "er" if ah is None else f"{arm} {'actual-H' if ah else 'pred-H'}"
            ok = "OK " if abs(r["acc"] - exp) < 5e-5 else "MISMATCH"
            print(f"    {tag:<18s} got {r['acc']:.4f}  expected {exp:.4f}  [{ok}]", flush=True)
        return

    ws = [int(w) for w in args.widths.split(",")] if args.widths else WIDTHS
    ss = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    aa = args.arms.split(",") if args.arms else ARMS
    done = {(r["H"], r["seed"], r["arm"]) for r in load_rows()} if args.resume else set()

    for H in ws:
        for seed in ss:
            for arm in aa:
                if (H, seed, arm) in done:
                    continue
                r = run_arm(arm, H, seed)
                record(H, seed, arm, r)
                extra = "" if "h0" not in r else f"  |g|={r['h0']:.4f}/{r['h1']:.4f}/{r['out']:.4f}"
                print(f"  H={H:<4d} seed={seed} {arm:<11s} acc={r['acc']:.4f}{extra}", flush=True)
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
