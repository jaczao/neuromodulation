"""pt7 TUNED CL regime — gain-SYNAPSE mechanisms at a VAL-TUNED operating point (user-requested).

Motivation: every pt5/pt6/pt7 CL number used a FIXED, inherited lr=1e-3, ep=5 (never val-tuned;
CLAUDE.md flags "5ep SGD@1e-3 under-trains"). This script closes that gap for the pt7 gain-SYNAPSE
er-own cells the user cares about (NE, vecproj, all4), separately per optimizer, and re-checks the
all4 standardise-vs-not question at the tuned point.

Protocol (obeys the non-negotiable rules):
  TUNE  — SplitMNIST(sequence=make_sequence(7), val_frac=0.1); train on each task's train split,
          eval on the HELD-OUT val split (never the test set). Sweep (lr, epochs) for the ER
          reference per optimizer; select argmax val avg-final-acc -> (lr*, ep*). Identical grid
          both optimizers (rule #3). Buffer fixed 1000 (matches the pt7 study).
  REPORT — SplitMNIST(sequence=SEQ default order, val_frac=0); eval on the official TEST set at the
          tuned (lr*, ep*). Cells: naive, er, and the gain-synapse er-own mechanisms.

The mechanism is transferred the ER-tuned operating point (rule #3: identical budget; ER is the
reference the er-own cells are judged against). gain-SYNAPSE throughout; K-rank linear gate.

Ledger results/pt7_tuned_syn_results.tsv (own; `--resume` skips done rows). 1 seed (42).
Sanity rows reproduce the frozen ep5/lr1e-3 ledger to prove the harness is faithful before trusting
any tuned number.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_neuromodulators as p7                                  # noqa: E402
import pt7_variants as pv                                         # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from data import SplitMNIST, make_sequence                       # noqa: E402

DEV = p7.DEV
CE = nn.CrossEntropyLoss()
SEQ = p7.SEQ                                                      # reporting (default) order
VAL_SEQ = make_sequence(7)                                        # tuning order (non-negotiable rule #1)
VAL_FRAC = 0.1
BUFFER = 1000
SEED = 42
TSV = Path(__file__).resolve().parent / "pt7_tuned_syn_results.tsv"

# ---- tuning grid (per optimizer); dominant knob = epochs, SGD wants larger lr ----
GRID = {
    "sgd":  {"lr": [3e-3, 1e-2, 3e-2], "ep": [5, 10, 20]},
    "adam": {"lr": [3e-4, 1e-3, 3e-3], "ep": [5, 10, 20]},
}
# gain-synapse er-own mechanisms to report (name, standardize)
MECHS = [("NE", True), ("vecproj", True), ("all4", True), ("all4", False)]


# ------------------------------- loaders -------------------------------
def build_loaders(sequence, val_frac):
    """Return [(train_loader, eval_loader)] x5. eval = val split (val_frac>0) or test (val_frac=0)."""
    ds = SplitMNIST(sequence=sequence, val_frac=val_frac)
    if val_frac > 0:
        return [(ds.get_task_loaders(t, 64)[0], ds.get_task_val_loader(t, 64)) for t in range(5)]
    return [ds.get_task_loaders(t, 64) for t in range(5)]


# ------------------------------- baselines (loaders-parametrized) -------------------------------
def run_baseline(method, opt_kind, lr, epochs, loaders, seed=SEED):
    p7.seed_all(seed)
    net = p7.Net().to(DEV)
    opt = p7._opt(opt_kind, net.parameters(), lr)
    buf = p7.Reservoir(BUFFER)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if method == "naive":
                    loss = p7.masked_ce(net.plain(x)[0], y)
                else:
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    loss = CE(net.plain(torch.cat(Xs))[0], torch.cat(Ys))
                opt.zero_grad(); loss.backward(); opt.step()
                if method == "er":
                    buf.add(x, y)
    acc = float(np.mean([p7._acc_plain(net, loaders[i][1]) for i in range(5)]))
    return dict(pred=acc, true=float("nan"), probe=float("nan"),
                per_layer={"h0": 0.0, "h1": 0.0, "out": 0.0})


# ------------------- head-based gain-synapse er-own (NE, all4): mirror p7.run_cell -------------------
def run_head(name, opt_kind, standardize, lr, epochs, loaders, seed=SEED):
    p7.seed_all(seed)                                            # same order as p7.build (SplitMNIST has no torch RNG)
    drivers, is_free, is_const = p7.cell_spec(name)
    layers = p7.DRIVER_LAYERS.get(name, None)
    net = p7.Net().to(DEV)
    gate = p7.make_gate("synapse", len(drivers), layers)
    heads = None if is_const else p7.Heads(len(drivers)).to(DEV)
    sig = None if (is_free or is_const) else p7.Signals(drivers, standardize=standardize)
    p7.net_loaders = loaders
    p7.train_erown(name, "synapse", net, gate, heads, sig, is_free, is_const,
                   opt_kind, lr=lr, epochs=epochs, buffer=BUFFER)
    return p7.eval_cell(name, "synapse", net, gate, heads, sig, is_const, loaders)


# ------------------- headless gain-synapse er-own (vecproj): mirror pv.run_ne er-own -------------------
def run_headless(kind, opt_kind, standardize, lr, epochs, loaders, seed=SEED):
    p7.seed_all(seed)
    net = p7.Net().to(DEV)
    drv = pv.NEDriver(kind, standardize)
    K = drv.K()
    gate = p7.make_gate("synapse", K, None)
    opt = p7._opt(opt_kind, list(net.parameters()) + gate.params(), lr)
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
                loss = CE(gate(net, drv.value(net, Xm), Xm), Ym)
                opt.zero_grad(); loss.backward(); opt.step(); buf.add(x, y)
    net.eval(); c = tot = 0; mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    with torch.no_grad():
        for i in range(5):
            for x, y in loaders[i][1]:
                x, y = x.to(DEV), y.to(DEV); b = x.size(0)
                v = drv.value(net, x, update=False)
                c += (gate(net, v, x).argmax(1) == y).sum().item()
                pl = gate.per_layer_mag(v, net)
                for k in mags:
                    mags[k] += pl[k] * b
                tot += b
    return dict(pred=c / tot, true=float("nan"), probe=float("nan"),
                per_layer={k: mags[k] / tot for k in mags})


def run_mech(name, opt_kind, standardize, lr, epochs, loaders):
    if name in ("vecproj", "vec_h1proj"):
        return run_headless(name, opt_kind, standardize, lr, epochs, loaders)
    return run_head(name, opt_kind, standardize, lr, epochs, loaders)


# ------------------------------- ledger -------------------------------
def load_done():
    if not TSV.exists():
        return {}
    out = {}
    for ln in TSV.read_text().splitlines():
        if ln.strip():
            f = ln.split("\t")
            out[f[0]] = float(f[1])
    return out


def record(tag, res):
    pl = res["per_layer"]
    row = (f"{tag}\t{res['pred']:.4f}\t{res['true']:.4f}\t{res['probe']:.3f}"
           f"\t{pl['h0']:.4f}\t{pl['h1']:.4f}\t{pl['out']:.4f}\n")
    with open(TSV, "a") as fh:
        fh.write(row)


def fmt(res):
    pl = res["per_layer"]
    return (f"pred={res['pred']:.4f}  |g|(h0/h1/out)={pl['h0']:.3f}/{pl['h1']:.3f}/{pl['out']:.3f}")


# ------------------------------- phases -------------------------------
def do_tune(done):
    val_loaders = build_loaders(VAL_SEQ, VAL_FRAC)
    for opt_kind in ("sgd", "adam"):
        for lr in GRID[opt_kind]["lr"]:
            for ep in GRID[opt_kind]["ep"]:
                tag = f"tune|er|{opt_kind}|lr{lr:g}|ep{ep}"
                if tag in done:
                    continue
                res = run_baseline("er", opt_kind, lr, ep, val_loaders)
                print(f"  TUNE er {opt_kind:4s} lr{lr:<7g} ep{ep:<3d} | val_acc={res['pred']:.4f}", flush=True)
                record(tag, res)
                done[tag] = res["pred"]


def best_setting(done, opt_kind):
    cells = [(k, v) for k, v in done.items() if k.startswith(f"tune|er|{opt_kind}|")]
    if not cells:
        return None
    k, v = max(cells, key=lambda kv: kv[1])
    _, _, _, lrs, eps = k.split("|")
    return float(lrs[2:]), int(eps[2:]), v


def do_report(done):
    test_loaders = build_loaders(SEQ, 0.0)
    # sanity: reproduce the frozen ledger at ep5/lr1e-3 (harness fidelity)
    for tag, fn in [("sanity|er|adam|lr0.001|ep5", lambda: run_baseline("er", "adam", 1e-3, 5, test_loaders)),
                    ("sanity|all4|synapse|er-own|adam|std1|lr0.001|ep5",
                     lambda: run_head("all4", "adam", True, 1e-3, 5, test_loaders))]:
        if tag not in done:
            res = fn(); print(f"  SANITY {tag} | {fmt(res)}", flush=True); record(tag, res); done[tag] = res["pred"]

    for opt_kind in ("sgd", "adam"):
        bs = best_setting(done, opt_kind)
        if bs is None:
            print(f"  [report] no tuned setting for {opt_kind} yet — run --phase tune first", flush=True)
            continue
        lr, ep, valacc = bs
        print(f"  [tuned {opt_kind}] lr={lr:g} ep={ep} (val_acc={valacc:.4f})", flush=True)
        # baselines at tuned point
        for m in ("naive", "er"):
            tag = f"report|{m}|-|-|{opt_kind}|-|lr{lr:g}|ep{ep}"
            if tag not in done:
                res = run_baseline(m, opt_kind, lr, ep, test_loaders)
                print(f"  REPORT {m:6s} {opt_kind:4s} | {fmt(res)}", flush=True); record(tag, res); done[tag] = res["pred"]
        # gain-synapse er-own mechanisms at tuned point
        for name, std in MECHS:
            tag = f"report|{name}|synapse|er-own|{opt_kind}|std{int(std)}|lr{lr:g}|ep{ep}"
            if tag not in done:
                res = run_mech(name, opt_kind, std, lr, ep, test_loaders)
                print(f"  REPORT {name:8s} synapse er-own {opt_kind:4s} std{int(std)} | {fmt(res)}", flush=True)
                record(tag, res); done[tag] = res["pred"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "tune", "report"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={DEV}  (pt7 tuned gain-synapse; 1 seed; val_seq={VAL_SEQ})\n", flush=True)
    done = load_done() if args.resume else {}
    if args.phase in ("all", "tune"):
        do_tune(done)
    if args.phase in ("all", "report"):
        do_report(done)
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
