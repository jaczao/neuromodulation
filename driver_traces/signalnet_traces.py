"""SIGNAL-NET CODE TRACES — what the signal net (and the signal net + GRU) actually emits, per dimension.

MECHANISM RUN, not observer-only: the signal net has no target of its own — it is trained end-to-end by
the main CE through the gate — so its code only exists if the gate is applied. This runs the real
`pt7_signalnet` mechanism (er-own, class-IL Split MNIST) and traces the K-dim vectors inside it.

ENGAGE=TRUE throughout. The canonical config zero-inits BOTH the signal net's output layer and the gate
P, which is the double-zero-init saddle: dL/dP ∝ m = 0 and dL/d(snet) ∝ P = 0, so neither bootstraps and
the code stays EXACTLY 0 for the whole run (that is pt7's `free` control, |g| = 0.000, = ER). Tracing it
would give K flat zero lines. `engage=True` gives the module's output layer a normal init while keeping
P zero-init, so gamma = 1 at step 0 (parity) but the code can bootstrap.

TRACED (per dimension, K=4, never as a norm):
  code{j}    the SignalNet output, 23 -> 64 -> 64 -> 64 -> K            (both cells)
  gruout{j}  the GRUCell output that the gate actually consumes         (signalnet-gru only)
For the -gru cell both are traced, so the GRU's contribution is the difference between the two.
Rows are {batch mean, batch SD}: the mean alone cannot tell a live per-sample dimension from one that
has collapsed to a batch-constant, and that distinction is what decided the previous driver study.

THE ONE DEVIATION from `results/pt7_signalnet.py`: feature column 8, "predicted entropy of the current
sample", is replaced by the ACTUAL entropy from an extra unmodulated forward. Entropy needs no labels, so
it is exactly computable at inference for the price of that forward — the head was never necessary for
it. Everything else is byte-identical to the frozen study, INCLUDING the eval protocol (`update=False`,
running scalars frozen at test) — this study does not carry over live_traces.py's live-stats premise.

The substitution is applied through a shim on `SignalHeads`, not by editing `SignalFeatures.build`:
`build` reads col 8 as `pred[:, 0:1]`, so replacing head output column 0 changes exactly that one feature
and does so BEFORE the 23-dim standardisation, which is where it has to happen. The shim is used only
for `feat.build`; the head's own MSE training step still sees the unwrapped head.

ANCHOR: `--actual-h off` must reproduce pt7_signalnet's frozen ledger bit-exact (the tracing is read-only
and consumes no RNG):
  signalnet     |neuron|K4|std1|er-own|adam|eng  pred 0.5215  |g| 0.4726/0.5269/0.8037
  signalnet-gru |neuron|K4|std1|er-own|adam|eng  pred 0.8657  |g| 1.5248/1.3156/1.9400
Those two runs double as the pred-H comparison arm, so every figure overlays actual-H against pred-H.

Cells: {signalnet, signalnet-gru} x {actual-H, pred-H}, K=4, neuron, std1, er-own, Adam,
lr 1e-3 / 5 ep per task / buffer 1000, seed 42. 1 seed.

Outputs: signalnet_traces.npz, figs_signalnet/*.png, signalnet_traces.log.
"""
import argparse
import copy
import sys
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

from live_traces import GRID, INK, INK2, SURFACE, _smooth, _wrap     # noqa: E402

DEV = p7.DEV
CE = nn.CrossEntropyLoss()
K = 4

NPZ = HERE / "signalnet_traces.npz"
FIGDIR = HERE / "figs_signalnet"

# pt7_signalnet_results.tsv, the `eng` rows: pred, then |g| h0/h1/out.
REF = {"signalnet": (0.5215, (0.4726, 0.5269, 0.8037)),
       "signalnet-gru": (0.8657, (1.5248, 1.3156, 1.9400))}

CELLS = [("signalnet", True), ("signalnet", False),
         ("signalnet-gru", True), ("signalnet-gru", False)]


def cell_id(kind, actual_h):
    return f"{'sngru' if kind == 'signalnet-gru' else 'sn'}|{'actualH' if actual_h else 'predH'}"


class ActualHShim(nn.Module):
    """Wraps SignalHeads so output column 0 carries the ACTUAL entropy instead of the head's prediction.

    `SignalFeatures.build` reads feature col 8 from `pred[:, 0:1]` and every other use of the head output
    (col 1 = predicted loss, cols 2-7 = predicted EMAs/stds) is untouched, so this is exactly the
    one-column change — and it lands before the 23-dim standardisation, which is the only place it can
    land without making the running stats inconsistent with the values they normalise.

    Costs one extra `net.plain` forward. At train `build` already computes the actual entropy internally
    for its running state, so the forward is redundant there; at EVAL it is the whole point, since that is
    where the quantity was previously only ever predicted.
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


class VecTrace:
    """Per-dimension (batch mean, batch SD). The SD row is the diagnostic: a dimension whose SD sits at
    ~0 is a batch-constant and carries no per-sample information into the gate, however large its mean."""

    def __init__(self, keys):
        self.d = {(k, s): [] for k in keys for s in ("mean", "sd")}
        self.bounds = []
        self._n = 0

    def add(self, rec):
        for k, v in rec.items():
            self.d[(k, "mean")].append(float(v.mean()))
            self.d[(k, "sd")].append(float(v.std()) if v.numel() > 1 else 0.0)
        self._n += 1

    def mark(self):
        self.bounds.append(self._n)


def trace_keys(use_gru):
    return [f"code{j}" for j in range(K)] + ([f"gruout{j}" for j in range(K)] if use_gru else [])


def _record(tr, code, m, use_gru):
    rec = {f"code{j}": code[:, j] for j in range(K)}
    if use_gru:
        rec.update({f"gruout{j}": m[:, j] for j in range(K)})
    tr.add(rec)


# ------------------------------- run (copy-forward of pt7_signalnet.run_signalnet) -------------------
def run(kind, actual_h, opt_kind="adam", seed=42, lr=1e-3, epochs=5, buffer=1000, ckpt=None, log=print):
    """Copy-forward of `pt7_signalnet.run_signalnet` (gran=neuron, K=4, standardize=True, engage=True)
    with the code/GRU read-outs added and the col-8 shim optionally installed. Copied rather than called
    so the frozen study keeps reproducing untouched (repo extraction rule), and so the trace hooks sit
    inside the loop without perturbing it — they only read tensors that already exist."""
    use_gru = kind == "signalnet-gru"
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV)
    gate = psn._mk("neuron", K)
    heads = psn.SignalHeads().to(DEV)
    feat = psn.SignalFeatures(True)
    snet = psn.SignalNet(K, engage=True).to(DEV)
    gru = psn.GRUOnVec(K, K, engage=True).to(DEV) if use_gru else None
    # the shim holds no parameters of its own, so building it consumes no RNG and the pred-H arm stays
    # bit-identical to the frozen study
    fheads = ActualHShim(heads, net) if actual_h else heads
    params = list(net.parameters()) + gate.params() + list(snet.parameters()) \
        + (list(gru.parameters()) if gru else [])
    main_opt = p7._opt(opt_kind, params, lr)
    head_opt = torch.optim.Adam(heads.parameters(), lr)
    buf = p7.Reservoir(buffer)
    tr = VecTrace(trace_keys(use_gru))

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
                code = snet(f)                                   # (B,K)
                m = gru(code) if gru else code
                loss = CE(gate(net, m, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                _record(tr, code.detach(), m.detach(), use_gru)
                T = feat.targets(net, Xm, Ym)                    # component-head regression (+replay)
                hloss = F.mse_loss(heads(Xm), T)                 # the RAW head, never the shim
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)
        tr.mark()
        log(f"    [{cell_id(kind, actual_h)}] task {t} done ({tr.bounds[-1]} steps)")

    if ckpt is not None:
        save_ckpt(ckpt, net, gate, heads, feat, snet, gru, kind, actual_h)
        log(f"    [{cell_id(kind, actual_h)}] checkpoint -> {ckpt.name}")
    return tr, both_evals(net, gate, heads, feat, snet, gru, fheads, loaders, use_gru)


def both_evals(net, gate, heads, feat, snet, gru, fheads, loaders, use_gru):
    """Run the test pass TWICE from the same trained state: `frozen` (update=False, the frozen study's
    protocol and the ledger anchor) and `live` (update=True, nothing frozen).

    The two passes must start from IDENTICAL state, and neither is free of side effects:
      - `feat` holds the running scalars, which update=True mutates;
      - `GRUOnVec.forward` defaults to update_state=True, so the GRU hidden advances at eval in BOTH
        modes (an inconsistency already present in the frozen study: its running scalars are frozen at
        test but its GRU hidden is not).
    So snapshot both before the first pass and restore before the second. Frozen runs FIRST because
    update=False leaves `feat` untouched, which keeps the anchor exact.
    """
    snap_feat = copy.deepcopy(feat)
    snap_hidden = gru.hidden.clone() if gru is not None else None
    out = {}
    for mode, upd in (("frozen", False), ("live", True)):
        if mode == "live":
            feat = snap_feat
            if gru is not None:
                gru.hidden = snap_hidden
        out[mode] = evaluate(net, gate, heads, feat, snet, gru, fheads, loaders, use_gru, update=upd)
    return out


@torch.no_grad()
def evaluate(net, gate, heads, feat, snet, gru, fheads, loaders, use_gru, update):
    """Copy-forward of `_eval_signalnet` + the read-outs, with the running scalars either frozen
    (`update=False`, as the frozen study) or live (`update=True`). `SignalFeatures.build` already
    supports update=True with y=None — it falls back to the PREDICTED loss for the actual-loss running
    state — so the live pass needs no labels and stays a legal inference protocol."""
    net.eval()
    te = VecTrace(trace_keys(use_gru))
    c = tot = 0
    mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            f = feat.build(net, fheads, x, y=None, update=update)
            code = snet(f)
            m = gru(code) if gru else code
            c += (gate(net, m, x).argmax(1) == y).sum().item()
            pl = gate.per_layer_mag(m)
            for k in mags:
                mags[k] += pl[k] * b
            _record(te, code, m, use_gru)
            tot += b
        te.mark()
    return te, c / tot, {k: v / tot for k, v in mags.items()}


def save_ckpt(path, net, gate, heads, feat, snet, gru, kind, actual_h):
    """Everything a test pass needs. The previous version of this study had no checkpointing, so this
    eval-side change cost a full retrain — the same trap CLAUDE.md records for pt7_driver_traces."""
    torch.save({"net": net.state_dict(), "gate": gate.state_dict(), "heads": heads.state_dict(),
                "snet": snet.state_dict(), "gru": (gru.state_dict() if gru else None),
                "gru_hidden": (gru.hidden.cpu() if gru else None), "feat": feat,
                "kind": kind, "actual_h": actual_h}, path)


# ------------------------------- plotting -------------------------------
COLOR = {"actualH": "#2a78d6", "predH": "#eb6834"}
LABEL = {"actualH": "col 8 = ACTUAL entropy (extra forward)", "predH": "col 8 = head-predicted (frozen study)"}
ROW = {"mean": "batch mean", "sd": "batch SD (per-sample dispersion)"}
PHASES = ("train", "test", "testlive")
PHASE_TITLE = {"train": "training", "test": "test — running scalars FROZEN",
               "testlive": "test — running scalars LIVE"}


def _panel(ax, data, mod, key, stat, phase, title):
    used = []
    for var in ("actualH", "predH"):
        cid = f"{mod}|{var}"
        kk = f"{cid}|{phase}|{key}|{stat}"
        if kk not in data or not len(data[kk]):
            continue
        v = data[kk]
        w = max(1, len(v) // 180)
        ax.plot(v, color=COLOR[var], lw=0.7, alpha=0.16, zorder=1)
        ax.plot(_smooth(v, w), color=COLOR[var], lw=1.6, label=LABEL[var], zorder=3)
        used.append(v)
        for b in data[f"{cid}|{phase}|bounds"][:-1]:
            ax.axvline(b, color=GRID, lw=0.8, ls="--", zorder=0)
    sym = False
    if used:
        a = np.concatenate([np.abs(s[np.isfinite(s)]) for s in used])
        typ = np.median(a[a > 0]) if (a > 0).any() else 0.0
        if typ > 0 and a.max() > 50 * typ and a.max() > 10:
            ax.set_yscale("symlog", linthresh=max(typ, 1e-6)); sym = True
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


VEC_DESC = {"code": "SignalNet output (23→64→64→64→K), the low-D code",
            "gruout": "GRUCell output — what the rank-K gate actually consumes"}
MOD_DESC = {"sn": "signalnet", "sngru": "signalnet-gru"}


def plot_all(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(exist_ok=True)
    setup = ("MECHANISM RUN (gate applied) · engage=True · K=4 · neuron · std1 · er-own · class-IL · "
             "ADAM · lr 0.001 · 5 ep/task · buffer 1000 · seed 42 · running scalars FROZEN at test")
    n = 0
    for mod in ("sn", "sngru"):
        vecs = ("code", "gruout") if mod == "sngru" else ("code",)
        for vec in vecs:
            for j in range(K):
                key = f"{vec}{j}"
                width = 4.8 * len(PHASES)
                fig, axes = plt.subplots(2, len(PHASES), figsize=(width, 6.0),
                                         facecolor=SURFACE, squeeze=False)
                fig.suptitle(f"{MOD_DESC[mod]}   —   {key}", fontsize=12, color=INK,
                             x=0.012, ha="left", y=0.988, va="top")
                fig.text(0.012, 0.918, _wrap(f"{VEC_DESC[vec]}, dimension {j} of {K} · {setup}", width),
                         fontsize=7.5, color=INK2, ha="left", va="top")
                for r, stat in enumerate(("mean", "sd")):
                    for c, phase in enumerate(PHASES):
                        ax = axes[r][c]
                        ax.set_facecolor(SURFACE)
                        _panel(ax, data, mod, key, stat, phase,
                               f"{ROW[stat]} — {PHASE_TITLE[phase]}")
                        if c == 0:
                            ax.set_ylabel(ROW[stat].split(" (")[0], fontsize=8, color=INK2)
                h, l = axes[0][0].get_legend_handles_labels()
                fig.legend(h, l, loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2,
                           ncol=2, bbox_to_anchor=(0.99, 0.995))
                fig.tight_layout(rect=[0, 0, 1, 0.895])
                fig.savefig(FIGDIR / f"{mod}_{key}.png", dpi=150, facecolor=SURFACE)
                plt.close(fig)
                n += 1

    # contact sheet: every traced dimension, batch mean, one sheet per phase
    for phase in PHASES:
        items = [(m, f"{v}{j}") for m in ("sn", "sngru")
                 for v in (("code", "gruout") if m == "sngru" else ("code",)) for j in range(K)]
        cols, rows = 4, int(np.ceil(len(items) / 4))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 2.5 * rows), facecolor=SURFACE, squeeze=False)
        fig.suptitle(f"signal-net traced dimensions — batch mean, {PHASE_TITLE[phase]}",
                     fontsize=14, color=INK,
                     x=0.008, ha="left", y=0.996, va="top")
        fig.text(0.008, 0.972, setup, fontsize=9, color=INK2, ha="left", va="top")
        flat = axes.ravel()
        for i, ax in enumerate(flat):
            ax.set_facecolor(SURFACE)
            if i >= len(items):
                ax.axis("off"); continue
            mod, key = items[i]
            _panel(ax, data, mod, key, "mean", phase, f"{MOD_DESC[mod]} {key}")
        h, l = flat[0].get_legend_handles_labels()
        if len(items) < len(flat):
            flat[len(items)].legend(h, l, loc="center", frameon=False, fontsize=10, labelcolor=INK2)
        else:
            fig.legend(h, l, loc="lower center", frameon=False, fontsize=10, labelcolor=INK2)
        fig.tight_layout(rect=[0, 0, 1, 0.958])
        fig.savefig(FIGDIR / f"_contact_sheet_{phase}.png", dpi=140, facecolor=SURFACE)
        plt.close(fig)
        n += 1
    print(f"  wrote {n} figures to {FIGDIR}")


def summarise(data, log=print):
    """Per-dimension mean/SD at train and test. The `sd/|mean|` column is the read: ~0 means the
    dimension is a batch-constant, i.e. it feeds the gate no per-sample information at all."""
    log("\n  cell              vec       phase      batch-mean    mean|.|     batch-SD    sd/|mean|")
    log("  " + "-" * 88)
    for mod in ("sn", "sngru"):
        for var in ("actualH", "predH"):
            cid = f"{mod}|{var}"
            vecs = ("code", "gruout") if mod == "sngru" else ("code",)
            for vec in vecs:
                for j in range(K):
                    for phase in PHASES:
                        mk = f"{cid}|{phase}|{vec}{j}|mean"
                        sk = f"{cid}|{phase}|{vec}{j}|sd"
                        if mk not in data or not len(data[mk]):
                            continue
                        m, s = data[mk], data[sk]
                        am = np.nanmean(np.abs(m))
                        log(f"  {cid:<17s} {vec}{j:<8d} {phase:<9s} {np.nanmean(m):>11.4g} "
                            f"{am:>10.4g} {np.nanmean(s):>12.4g} "
                            f"{(np.nanmean(s) / am if am > 0 else float('nan')):>11.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="override epochs/task (smoke tests)")
    ap.add_argument("--only", default="", help="run one cell id, e.g. 'sn|actualH'")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        z = np.load(NPZ, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        plot_all(d); summarise(d)
        return
    if NPZ.exists() and not args.force:
        raise SystemExit(f"{NPZ.name} exists — use --plot-only or --force")

    print(f"device={DEV}  signal-net code traces  (mechanism run, engage=True, K={K}, neuron, std1, "
          f"er-own, adam, seed 42)\n", flush=True)
    out = {}
    for kind, actual_h in CELLS:
        cid = cell_id(kind, actual_h)
        if args.only and args.only != cid:
            continue
        print(f"  running {cid} ({kind}) ...", flush=True)
        ck = HERE / f"ckpt_sn_{cid.replace('|', '_')}.pt"
        tr, res = run(kind, actual_h, epochs=(args.epochs or 5), ckpt=ck,
                      log=lambda s: print(s, flush=True))
        te, acc, mags = res["frozen"]
        tel, accl, magsl = res["live"]
        ref = REF[kind] if (not actual_h and args.epochs is None) else None
        if ref is None:
            tag = ""
        else:
            dg = " ".join(f"{mags[k] - ref[1][i]:+.4f}" for i, k in enumerate(("h0", "h1", "out")))
            tag = f"   (frozen ledger {ref[0]:.4f}, delta {acc - ref[0]:+.4f}; d|g| {dg})"
        print(f"  {cid:16s} FROZEN pred={acc:.4f}  |g| h0={mags['h0']:.4f} h1={mags['h1']:.4f} "
              f"out={mags['out']:.4f}{tag}", flush=True)
        print(f"  {cid:16s} LIVE   pred={accl:.4f}  |g| h0={magsl['h0']:.4f} h1={magsl['h1']:.4f} "
              f"out={magsl['out']:.4f}   (d-live {accl - acc:+.4f})", flush=True)
        out[f"{cid}|acc_live"] = np.asarray(accl, dtype=np.float32)
        for phase, t in (("train", tr), ("test", te), ("testlive", tel)):
            for (k, s), v in t.d.items():
                out[f"{cid}|{phase}|{k}|{s}"] = np.asarray(v, dtype=np.float32)
            out[f"{cid}|{phase}|bounds"] = np.asarray(t.bounds, dtype=np.int32)
        out[f"{cid}|acc"] = np.asarray(acc, dtype=np.float32)
    np.savez_compressed(NPZ, **out)
    print(f"  wrote {NPZ}", flush=True)
    plot_all(out)
    summarise(out, log=lambda s: print(s, flush=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
