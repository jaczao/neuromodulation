"""CAPACITY ABLATION (user-requested): is the pt5/6/7 gating null an OVER-PARAMETERIZATION artifact?

Hypothesis under test. Across pt5-pt7 a soft, main-loss-trained multiplicative gate was always ABSORBED by
the backbone (the h1-gate sat at mean 0.281 yet landed exactly on ER; a fixed-random projection matched a
learned one; the L1 penalty only shuffled magnitude between gamma and W). One explanation for that null is
that the MLP (784->400->400->10) is so over-parameterized for 2 classes at a time that there is never any
SCARCITY to allocate — gain control is a resource-allocation mechanism with no scarce resource to allocate.
The one pt5 cell that worked (Iter 1 disjoint gain) manufactured scarcity by hand (1/5 capacity per task).

So: shrink the hidden width H until the BASELINE ITSELF degrades (scarcity appears), then re-run a gating
mechanism at each H and watch the (gate - baseline) delta.
  - delta ~ 0 at EVERY H, even where ER has collapsed  -> capacity was never the issue; the null is real
    and the absorption story stands (an airtight negative).
  - delta turns POSITIVE as H shrinks                  -> gain control does allocate contested capacity;
    the earlier null was an over-parameterization artifact.
Honest prior (user's): the first. The disjoint-gain win was more plausibly the hard {0,1} freeze (kills the
gradient, un-absorbable) than scarcity per se. Run as CONFOUND ELIMINATION, not as a hoped-for win.

SETUP. Class-IL Split MNIST, Adam, lr=1e-3 / ep=5 / buffer=1000, 3 seeds {42,43,44}, hidden width
H in {400,200,100,50,25,10} (BOTH hidden layers = H). One gating mechanism, held fixed across the sweep:
the canonical pt7 `all4` gain-NEURON gate on ER (er-own) — K=4 heads m_k(x) regress the standardized bio
signals DA/ACh/NE/5HT, driving Gamma = 1 + sum_k m_k P_k over (h0,h1,out). Arms per (H, seed):
    naive     naive + masked-CE (context: how much a mechanism-free net degrades)
    er        ER, plain CE                                   <- THE baseline the gate is judged against
    er+free   pt7 `free` control: K=4 zero-init heads, NO bio target, gate trained end-to-end. |g| stays
              exactly 0 (the double-zero-init saddle), so it is numerically ER but consumes the SAME torch
              RNG for head construction -> it is the RNG-MATCHED baseline (pt7_tuned_neuro gotcha: `free`
              "beats" plain ER by ~0.006 purely by shifting the replay draws). d-free is the honest delta.
    er+all4   the gate under test.

IMPLEMENTATION. Reuses the pt7 code path VERBATIM (p7.Net / NeuronGate / Heads / Signals / train_erown /
train_baseline / eval_cell); width enters only by rebinding p7's H0/H1/GATEDIM module globals inside the
`width(H)` context manager, which those classes read at call time. Consequence: H=400 reproduces the frozen
pt7 ledger BIT-EXACT (naive 0.3900 / er 0.8946 / free 0.8760 / all4 0.8816 at seed 42) — that agreement is
the sanity anchor for the whole sweep, checked by `--part anchor`.

NOT re-tuned per width (rule #1 caveat, stated up front): lr/epochs are the standard pt7 point for every H,
so ABSOLUTE accuracies at small H may be pessimistic. Every arm at a given H gets the identical budget, so
the DELTA at matched H — the quantity this study is about — is unaffected. The neuromodulator's own capacity
(head 784->32->K, gate P) is held CONSTANT on purpose: what shrinks is the MODULATED resource, not the
modulator (at H=10 the head is larger than the main net; that is the intended asymmetry).

Ledger results/pt7_capacity_results.tsv (--resume skips ledgered cells). Log pt7_capacity.log.
Plots via `--plot` -> results/pt7_capacity/*.png. 3 seeds.
"""
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_neuromodulators as p7                                  # noqa: E402

DEV = p7.DEV
HERE = Path(__file__).resolve().parent
TSV = HERE / "pt7_capacity_results.tsv"
PLOTDIR = HERE / "pt7_capacity"

# H=5 extends the user's requested grid: at H=10 the baseline degrades but has not collapsed, and the
# negative is only airtight if it survives DEEP scarcity. Seeds 45,46 are run at H in {10,5} ONLY — the two
# widths where per-seed spread grows (±0.02) and where the verdict is actually decided; the large-H cells
# stay at 3 seeds. Seed count per cell is reported as `n` in the table, never averaged over silently.
WIDTHS = [400, 200, 100, 50, 25, 10, 5]
SEEDS = [42, 43, 44]
EXTRA_SEEDS = {10: [45, 46], 5: [45, 46]}
ARMS = ["naive", "er", "er+free", "er+all4"]
LR, EPOCHS, BUFFER, OPT = 1e-3, 5, 1000, "adam"
GRAN = "neuron"

# frozen pt7 ledger values at H=400, seed 42, adam, lr1e-3/ep5/buffer1000 (the bit-exact anchor)
ANCHOR = {"naive": 0.3900, "er": 0.8946, "er+free": 0.8760, "er+all4": 0.8816}


@contextmanager
def width(H):
    """Rebind the pt7 module globals that define the main net's hidden width.

    p7.Net / NeuronGate / SynapseGate read H0/H1/OUT/GATEDIM at CALL time (never captured at import), so
    rebinding them here makes every downstream pt7 function build and run an H-wide net with no other edit.
    Single-threaded and sequential; restored on exit.
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


def run_cell_seeded(name, arm_train, seed):
    """p7.run_cell, but with the seed threaded through (p7.run_cell hardcodes build(seed=42))."""
    loaders, net, gate, heads, sig, is_free, is_const = p7.build(name, GRAN, seed=seed, standardize=True)
    p7.net_loaders = loaders
    arm_train(name, GRAN, net, gate, heads, sig, is_free, is_const,
              OPT, lr=LR, epochs=EPOCHS, buffer=BUFFER)
    return p7.eval_cell(name, GRAN, net, gate, heads, sig, is_const, loaders)


def run_arm(arm, H, seed):
    """-> dict(acc=..., probe=..., h0/h1/out=...). Baselines report acc only."""
    with width(H):
        if arm in ("naive", "er"):
            acc = p7.train_baseline(arm, OPT, lr=LR, epochs=EPOCHS, buffer=BUFFER, seed=seed)
            return {"acc": acc}
        name = {"er+free": "free", "er+all4": "all4"}[arm]
        r = run_cell_seeded(name, p7.train_erown, seed)
        pl = r["per_layer"]
        return {"acc": r["pred"], "probe": r["probe"],
                "h0": pl["h0"], "h1": pl["h1"], "out": pl["out"]}


# ------------------------------- ledger -------------------------------
COLS = ["H", "seed", "arm", "acc", "probe", "h0", "h1", "out"]


def load_rows():
    if not TSV.exists():
        return []
    rows = []
    for ln in TSV.read_text().splitlines():
        if not ln.strip():
            continue
        f = ln.split("\t")
        rows.append({"H": int(f[0]), "seed": int(f[1]), "arm": f[2], "acc": float(f[3]),
                     "probe": _f(f, 4), "h0": _f(f, 5), "h1": _f(f, 6), "out": _f(f, 7)})
    return rows


def _f(fields, i):
    return float(fields[i]) if i < len(fields) and fields[i] not in ("", "-") else float("nan")


def record(H, seed, arm, r):
    cells = [H, seed, arm, f"{r['acc']:.4f}"]
    if "probe" in r:
        cells += [f"{r['probe']:.3f}", f"{r['h0']:.4f}", f"{r['h1']:.4f}", f"{r['out']:.4f}"]
    with open(TSV, "a") as f:
        f.write("\t".join(str(c) for c in cells) + "\n")


# ------------------------------- aggregation / plotting -------------------------------
def agg(rows):
    """(H, arm) -> (mean, std, n) over seeds."""
    out = {}
    for H in WIDTHS:
        for arm in ARMS:
            a = [r["acc"] for r in rows if r["H"] == H and r["arm"] == arm]
            if a:
                out[(H, arm)] = (float(np.mean(a)), float(np.std(a)), len(a))
    return out


def paired_delta(rows, H, arm, ref):
    """Per-seed paired delta arm-ref at width H -> (mean, std, n, per-seed list).

    Seeds are taken from the LEDGER, not from the SEEDS constant: widths carrying EXTRA_SEEDS would
    otherwise pair only the first 3 while the mean column averaged all of them (a silent mismatch).
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
    A = agg(rows)
    print("\n  H     params  n  " + "".join(f"{a:>18s}" for a in ARMS)
          + f"{'d-er':>10s}{'d-free':>10s}", flush=True)
    for H in WIDTHS:
        cells = ""
        nseed = 0
        for arm in ARMS:
            if (H, arm) in A:
                m, s, n = A[(H, arm)]
                nseed = max(nseed, n)
                cells += f"{m:>12.4f}+-{s:<4.3f}"
            else:
                cells += f"{'-':>18s}"
        de = paired_delta(rows, H, "er+all4", "er")
        df = paired_delta(rows, H, "er+all4", "er+free")
        cells += f"{de[0]:>+10.4f}" if de else f"{'-':>10s}"
        cells += f"{df[0]:>+10.4f}" if df else f"{'-':>10s}"
        print(f"  {H:<5d} {n_params(H):>8,d} {nseed:<2d} {cells}", flush=True)

    print("\n  gate engagement (er+all4, mean over seeds): per-layer |g| and task-probe", flush=True)
    print(f"  {'H':<5s}{'|g| h0':>10s}{'|g| h1':>10s}{'|g| out':>10s}{'probe':>9s}", flush=True)
    for H in WIDTHS:
        sel = [r for r in rows if r["H"] == H and r["arm"] == "er+all4"]
        if not sel:
            continue
        g = [float(np.mean([r[k] for r in sel])) for k in ("h0", "h1", "out")]
        pr = float(np.mean([r["probe"] for r in sel]))
        print(f"  {H:<5d}{g[0]:>10.4f}{g[1]:>10.4f}{g[2]:>10.4f}{pr:>9.3f}", flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTDIR.mkdir(exist_ok=True)
    A = agg(rows)
    have = [H for H in WIDTHS if (H, "er") in A]
    if not have:
        print("  nothing to plot yet", flush=True)
        return
    style = {"naive": ("#888888", "o", "naive + masked-CE"),
             "er": ("#1f77b4", "s", "ER"),
             "er+free": ("#2ca02c", "^", "ER + free (dead gate, RNG-matched)"),
             "er+all4": ("#d62728", "D", "ER + all4 gate")}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for arm in ARMS:
        xs = [H for H in have if (H, arm) in A]
        if not xs:
            continue
        m = [A[(H, arm)][0] for H in xs]; s = [A[(H, arm)][1] for H in xs]
        c, mk, lab = style[arm]
        ax.errorbar(xs, m, yerr=s, marker=mk, color=c, label=lab, capsize=3, lw=1.8, ms=6)
    ax.set_xscale("log"); ax.set_xticks(WIDTHS); ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.invert_xaxis()
    ax.set_xlabel("hidden width H  (both layers; shrinking ->)")
    ax.set_ylabel("final avg class-IL accuracy  (±1 s.d. over seeds)")
    ns = sorted({A[(H, "er")][2] for H in have})
    ax.set_title("Capacity ablation — accuracy vs width\n"
                 f"(Split MNIST, Adam, lr 1e-3 / ep 5 / buffer 1000, {min(ns)}–{max(ns)} seeds)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    for arm, ref, c, mk, lab in (("er+all4", "er", "#d62728", "D", "all4 − ER"),
                                 ("er+all4", "er+free", "#7f4fbf", "o", "all4 − free (RNG-matched)")):
        xs, m, s = [], [], []
        for H in have:
            d = paired_delta(rows, H, arm, ref)
            if d:
                # s.e.m. of the PAIRED delta: the question is whether the mean differs from 0, and the
                # paired spread at small H is dominated by how unstable the run itself is, not the gate.
                sd = float(np.std(d[3], ddof=1)) if d[2] > 1 else 0.0
                xs.append(H); m.append(d[0]); s.append(sd / np.sqrt(d[2]))
        if xs:
            ax.errorbar(xs, m, yerr=s, marker=mk, color=c, label=lab, capsize=3, lw=1.8, ms=6)
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-0.007, 0.007, color="k", alpha=0.08, label="1-seed noise floor (±0.007)")
    ax.set_xscale("log"); ax.set_xticks(WIDTHS); ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.invert_xaxis()
    ax.set_xlabel("hidden width H  (shrinking ->)")
    ax.set_ylabel("paired accuracy delta  (±1 s.e.m.)")
    ax.set_title("Does the gate start helping once capacity is scarce?\n(paired per seed)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.tight_layout()
    p = PLOTDIR / "capacity_ablation.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}", flush=True)

    # gate engagement vs width — does the gate engage more when capacity is scarce?
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for k, c in (("h0", "#1f77b4"), ("h1", "#ff7f0e"), ("out", "#d62728")):
        xs, ys = [], []
        for H in WIDTHS:
            sel = [r[k] for r in rows if r["H"] == H and r["arm"] == "er+all4"]
            if sel:
                xs.append(H); ys.append(float(np.mean(sel)))
        if xs:
            ax.plot(xs, ys, marker="o", color=c, label=f"|g| {k}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(WIDTHS); ax.set_xticklabels([str(w) for w in WIDTHS]); ax.invert_xaxis()
    ax.set_xlabel("hidden width H  (shrinking ->)"); ax.set_ylabel("mean |gate deviation|")
    ax.set_title("Gate engagement vs capacity (ER + all4)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    p = PLOTDIR / "gate_engagement.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}", flush=True)


# ------------------------------- driver -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="sweep", choices=["sweep", "anchor", "smoke", "table", "plot"])
    ap.add_argument("--widths", default=None, help="comma filter, e.g. 400,200")
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

    print(f"device={DEV}  capacity ablation | class-IL | Adam lr={LR} ep={EPOCHS} buffer={BUFFER} "
          f"| gain-{GRAN} all4 on ER | 3 seeds\n", flush=True)

    if args.part == "smoke":
        for H in (400, 10):
            with width(H):
                net = p7.Net().to(DEV)
                assert net.l0.out_features == H and net.l1.out_features == H
                print(f"  smoke H={H:<4d} params={n_params(H):>8,d} "
                      f"l0={tuple(net.l0.weight.shape)} l1={tuple(net.l1.weight.shape)} "
                      f"l2={tuple(net.l2.weight.shape)}", flush=True)
                r = run_cell_seeded_short("all4", 42)
                print(f"    1-epoch all4 er-own: pred={r['pred']:.4f}  "
                      f"|g|={r['per_layer']['h0']:.4f}/{r['per_layer']['h1']:.4f}/"
                      f"{r['per_layer']['out']:.4f}", flush=True)
        assert (p7.H0, p7.H1, p7.GATEDIM) == (400, 400, 810), "width() failed to restore globals"
        print("  globals restored OK", flush=True)
        return

    if args.part == "anchor":
        print("  H=400 seed 42 must reproduce the frozen pt7 ledger bit-exact:", flush=True)
        for arm in ARMS:
            r = run_arm(arm, 400, 42)
            ok = "OK " if abs(r["acc"] - ANCHOR[arm]) < 5e-5 else "MISMATCH"
            print(f"    {arm:9s} got {r['acc']:.4f}  expected {ANCHOR[arm]:.4f}  [{ok}]", flush=True)
        return

    ws = [int(w) for w in args.widths.split(",")] if args.widths else WIDTHS
    ss = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    aa = args.arms.split(",") if args.arms else ARMS
    done = {(r["H"], r["seed"], r["arm"]) for r in load_rows()} if args.resume else set()

    for H in ws:
        for seed in (ss if ss is not None else SEEDS + EXTRA_SEEDS.get(H, [])):
            for arm in aa:
                if (H, seed, arm) in done:
                    continue
                r = run_arm(arm, H, seed)
                record(H, seed, arm, r)
                extra = ("" if "probe" not in r else
                         f"  probe={r['probe']:.3f}  |g|={r['h0']:.4f}/{r['h1']:.4f}/{r['out']:.4f}")

                print(f"  H={H:<4d} seed={seed} {arm:9s} acc={r['acc']:.4f}{extra}", flush=True)
    print("ALL SELECTED CELLS DONE", flush=True)


def run_cell_seeded_short(name, seed):
    loaders, net, gate, heads, sig, is_free, is_const = p7.build(name, GRAN, seed=seed)
    p7.net_loaders = loaders
    p7.train_erown(name, GRAN, net, gate, heads, sig, is_free, is_const, OPT, lr=LR, epochs=1, buffer=BUFFER)
    return p7.eval_cell(name, GRAN, net, gate, heads, sig, is_const, loaders)


if __name__ == "__main__":
    main()
