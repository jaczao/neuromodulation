"""pt7 NEUROMOD-NET tuning — class-IL er-own, main net FROZEN at the ER-tuned point (user-requested).

Motivation: pt7_tuned_syn val-tuned the MAIN net (lr, ep) for the ER *baseline*. The neuromodulator
net (regression heads + rank-K gate projection P) was never tuned on its own — in pt7 both ride the
MAIN lr (gate.P is folded into main_opt; heads.head_opt uses the same lr). This script tunes the
neuromod net's OWN learning rate while holding the main net at ER-best, for the class-IL er-own arm.

DECOUPLING (the crux of "tuning the neuromod net"): the main net steps at the inherited ER-best lr;
the gate P and heads step at a SEPARATE `neuro_lr` (Adam). Splitting params across two Adam
optimizers at the same lr is byte-identical to pt7's single optimizer (Adam has no cross-param state),
so `neuro_lr = main_lr = 3e-4` reproduces the frozen pt7_tuned_syn cell exactly (sanity anchor 0.9074
for all4-synapse er-own std1) — proving the harness before any tuned number is trusted.

Protocol (obeys the non-negotiable rules):
  TUNE   — SplitMNIST(make_sequence(7), val_frac=0.1); train each task's train split, eval the HELD-OUT
           val split (never test). Sweep neuro_lr for each (mechanism, granularity); seed 42; select
           argmax val avg-final-acc. head_hidden FIXED at 32 (pt7 default). Budget 5 points <= ER's 9.
  REPORT — SplitMNIST(SEQ default order, val_frac=0); official TEST set; the selected neuro_lr over
           3 seeds {42,43,44}, vs the 3-seed tuned-ER baseline (re-run here to stay self-contained).

Mechanisms: all4 (std1, the borderline-positive composite) + free (content-free control, tuned on the
IDENTICAL grid — if tuned free matches tuned all4, the win is module capacity/lr, not the neuromod
SIGNAL, and the pt7 negative holds). Granularity: BOTH neuron and synapse. Adam main net only.

CONFIG STORE (configs.TUNED_MAIN) is keyed (metric, base, optimizer) — class-IL vs task-IL, naive vs
er, sgd vs adam — kept SEPARATE. This study reads the (classil, er, adam) slot.

Ledger results/pt7_tuned_neuro_results.tsv (own; `--resume` skips done rows).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_neuromodulators as p7                                  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from data import SplitMNIST, make_sequence                       # noqa: E402
from configs import TUNED_MAIN                                    # noqa: E402  (single source of truth)

DEV = p7.DEV
CE = p7.CE
SEQ = p7.SEQ                                                      # reporting (default) order
VAL_SEQ = make_sequence(7)                                        # tuning order (non-negotiable rule #1)
VAL_FRAC = 0.1
BUFFER = 1000
HEAD_HID = 32                                                     # fixed (pt7 default); a hid-sweep would add its own axis
TSV = Path(__file__).resolve().parent / "pt7_tuned_neuro_results.tsv"

# ---- INHERITED main-net operating point: keyed (metric, base, optimizer) in configs.TUNED_MAIN
# (single source of truth; SEPARATE for class-IL/task-IL, naive/er, AND optimizer). Adapt to this
# study's (opt, lr, ep) shape; optimizer now lives in the key. ----
MAIN_CFG = {k: dict(opt=k[2], lr=v["lr"], ep=v["epochs_per_task"]) for k, v in TUNED_MAIN.items()}

METRIC = "classil"                                               # this study tunes class-IL only
BASE = "er"                                                      # er-own arm -> judged vs the ER baseline
OPT = "adam"                                                     # Adam only (the ER ceiling)
NEURO_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]                        # the swept knob (5 pts <= ER's 9)
MECHS = [("all4", True), ("free", None)]                         # (name, standardize); free has no sig
GRANS = ["neuron", "synapse"]
REPORT_SEEDS = [42, 43, 44]


# ------------------------------- loaders -------------------------------
def build_loaders(sequence, val_frac):
    ds = SplitMNIST(sequence=sequence, val_frac=val_frac)
    if val_frac > 0:
        return [(ds.get_task_loaders(t, 64)[0], ds.get_task_val_loader(t, 64)) for t in range(5)]
    return [ds.get_task_loaders(t, 64) for t in range(5)]


# ------------------------------- ER baseline (at the inherited main config) -------------------------------
def run_er(main_opt_kind, main_lr, epochs, loaders, seed):
    p7.seed_all(seed)
    net = p7.Net().to(DEV)
    opt = p7._opt(main_opt_kind, net.parameters(), main_lr)
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
                opt.zero_grad(); loss.backward(); opt.step(); buf.add(x, y)
    acc = float(np.mean([p7._acc_plain(net, loaders[i][1]) for i in range(5)]))
    return dict(pred=acc, true=float("nan"), probe=float("nan"),
                per_layer={"h0": 0.0, "h1": 0.0, "out": 0.0})


# ------------------- er-own with a DECOUPLED neuromod optimizer (neuro_lr) -------------------
def run_erown_decoupled(name, gran, standardize, main_opt_kind, main_lr, neuro_lr, epochs, loaders, seed):
    # construct exactly as p7.run/build so the seed reproduces (SplitMNIST has no torch RNG)
    p7.seed_all(seed)
    drivers, is_free, is_const = p7.cell_spec(name)
    layers = p7.DRIVER_LAYERS.get(name, None)
    net = p7.Net().to(DEV)
    gate = p7.make_gate(gran, len(drivers), layers)
    heads = None if is_const else p7.Heads(len(drivers), hid=HEAD_HID).to(DEV)
    sig = None if (is_free or is_const) else p7.Signals(drivers, standardize=standardize)
    K = p7._K(gate, gran)

    # MAIN net at the inherited lr; the NEUROMOD net (gate P + heads) at neuro_lr (Adam)
    main_opt = p7._opt(main_opt_kind, list(net.parameters()), main_lr)
    neuro_params = gate.params() + (list(heads.parameters()) if is_free else [])
    neuro_opt = torch.optim.Adam(neuro_params, neuro_lr)
    head_opt = (torch.optim.Adam(heads.parameters(), neuro_lr)
                if (heads is not None and not is_free) else None)

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
                m = p7._m(heads, is_const, Xm, K)
                m_gate = m if is_free else m.detach()
                loss = CE(gate(net, m_gate, Xm), Ym)
                main_opt.zero_grad(); neuro_opt.zero_grad()
                loss.backward()
                main_opt.step(); neuro_opt.step()
                if head_opt is not None:                          # biological head regression (+replay via Xm)
                    T = sig.targets(net, Xm, Ym)
                    hloss = F.mse_loss(heads(Xm), T)
                    head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)
    return p7.eval_cell(name, gran, net, gate, heads, sig, is_const, loaders)


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
    return f"pred={res['pred']:.4f}  |g|(h0/h1/out)={pl['h0']:.3f}/{pl['h1']:.3f}/{pl['out']:.3f}"


def mech_std(std):
    return "-" if std is None else f"std{int(std)}"


# ------------------------------- phases -------------------------------
def do_tune(done):
    cfg = MAIN_CFG[(METRIC, BASE, OPT)]
    val_loaders = build_loaders(VAL_SEQ, VAL_FRAC)
    print(f"[tune] main={cfg} (inherited, frozen); sweeping neuro_lr {NEURO_LRS}\n", flush=True)
    for name, std in MECHS:
        for gran in GRANS:
            for nlr in NEURO_LRS:
                tag = f"tune|{METRIC}|{BASE}-own|{name}|{gran}|{mech_std(std)}|hid{HEAD_HID}|nlr{nlr:g}|s42"
                if tag in done:
                    continue
                res = run_erown_decoupled(name, gran, std if std is not None else True,
                                          cfg["opt"], cfg["lr"], nlr, cfg["ep"], val_loaders, seed=42)
                print(f"  TUNE {name:5s} {gran:7s} nlr{nlr:<7g} | val={fmt(res)}", flush=True)
                record(tag, res); done[tag] = res["pred"]


def best_nlr(done, name, gran, std):
    pref = f"tune|{METRIC}|{BASE}-own|{name}|{gran}|{mech_std(std)}|hid{HEAD_HID}|nlr"
    cells = [(k, v) for k, v in done.items() if k.startswith(pref)]
    if not cells:
        return None
    k, v = max(cells, key=lambda kv: kv[1])
    nlr = float(k.split("|nlr")[1].split("|")[0])
    return nlr, v


def do_report(done):
    cfg = MAIN_CFG[(METRIC, BASE, OPT)]
    test_loaders = build_loaders(SEQ, 0.0)

    # SANITY: neuro_lr = main_lr must reproduce the frozen pt7_tuned_syn cell (0.9074, all4-synapse er-own std1)
    stag = f"sanity|{METRIC}|{BASE}-own|all4|synapse|std1|hid{HEAD_HID}|nlr{cfg['lr']:g}|s42"
    if stag not in done:
        res = run_erown_decoupled("all4", "synapse", True, cfg["opt"], cfg["lr"], cfg["lr"], cfg["ep"],
                                  test_loaders, seed=42)
        print(f"  SANITY all4 synapse er-own nlr={cfg['lr']:g} | {fmt(res)}  (expect ~0.9074)", flush=True)
        record(stag, res); done[stag] = res["pred"]

    # ER baseline at the inherited main config, 3 seeds (self-contained reference)
    er_accs = []
    for seed in REPORT_SEEDS:
        tag = f"report|{METRIC}|{BASE}|-|-|-|-|s{seed}"
        if tag not in done:
            res = run_er(cfg["opt"], cfg["lr"], cfg["ep"], test_loaders, seed)
            print(f"  REPORT er seed{seed} | {fmt(res)}", flush=True)
            record(tag, res); done[tag] = res["pred"]
        er_accs.append(done[tag])
    print(f"  [ER baseline] {np.mean(er_accs):.4f} ± {np.std(er_accs):.4f}\n", flush=True)

    # tuned neuromod cells, 3 seeds at the selected neuro_lr
    for name, std in MECHS:
        for gran in GRANS:
            bn = best_nlr(done, name, gran, std)
            if bn is None:
                print(f"  [report] no tuned neuro_lr for {name}/{gran} yet — run --phase tune first", flush=True)
                continue
            nlr, valacc = bn
            accs = []
            for seed in REPORT_SEEDS:
                tag = f"report|{METRIC}|{BASE}-own|{name}|{gran}|{mech_std(std)}|hid{HEAD_HID}|nlr{nlr:g}|s{seed}"
                if tag not in done:
                    res = run_erown_decoupled(name, gran, std if std is not None else True,
                                              cfg["opt"], cfg["lr"], nlr, cfg["ep"], test_loaders, seed)
                    print(f"  REPORT {name:5s} {gran:7s} nlr{nlr:<7g} seed{seed} | {fmt(res)}", flush=True)
                    record(tag, res); done[tag] = res["pred"]
                accs.append(done[tag])
            d = np.mean(accs) - np.mean(er_accs)
            print(f"  [{name}/{gran}] best nlr={nlr:g} (val {valacc:.4f}) -> "
                  f"test {np.mean(accs):.4f} ± {np.std(accs):.4f}  (dER {d:+.4f})\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "tune", "report"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={DEV}  (pt7 neuromod-net tuning; class-IL er-own; main frozen at "
          f"{MAIN_CFG[(METRIC, BASE, OPT)]}; val_seq={VAL_SEQ})\n", flush=True)
    done = load_done() if args.resume else {}
    if args.phase in ("all", "tune"):
        do_tune(done)
    if args.phase in ("all", "report"):
        do_report(done)
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
