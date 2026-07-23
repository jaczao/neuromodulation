"""pt7 TASK-IL tuning — main net + neuromod net, Adam only, gain-SYNAPSE only (user-requested).

Task-IL here = this repo's `taskil` convention: per-sample MASKED-CE training (each sample's loss
restricted to its own task's 2 classes; works with ER replay since it masks by label->task-pair) and
2-way MASKED eval (task id known at test -> argmax over that task's 2 classes). Adam throughout.

Flow (baselines FIRST to tune the main net, THEN modulated):
  1. TUNE-MAIN  — naive and er baselines, grid lr {3e-4,1e-3,3e-3} x ep {5,10,20} (rule #3, same grid
                  as the ER class-IL tuning), on the VAL sequence make_sequence(7), val_frac=0.1
                  (never test). Select argmax val task-IL acc per method -> (lr*, ep*) for naive and er.
  2. TUNE-NEURO — for each base {naive, er}, main FROZEN at that method's (lr*, ep*), tune the neuromod
                  net's own lr (decoupled Adam over gate P + heads) over {1e-4,3e-4,1e-3,3e-3,1e-2} for
                  free + all4, gain-synapse. naive-modulated = nobuf arm (gate on a naive main, NO
                  replay); er-modulated = er-own arm (gate on the ER batch). head_hidden=32.
  3. REPORT     — TEST set (SEQ default order, val_frac=0), 3 seeds {42,43,44}, at the selected configs.
                  Final table: naive vs er vs naive+{free,all4} vs er+{free,all4}.

Ledger results/pt7_tuned_neuro_taskil_results.tsv (own; `--resume` skips done rows). Selected configs
written back to prototype/configs.py by hand after the run (TUNED_MAIN taskil slots + TUNED_NEURO_LR).
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

DEV = p7.DEV
SEQ = p7.SEQ                                                      # reporting (default) order
VAL_SEQ = make_sequence(7)                                        # tuning order (non-negotiable rule #1)
VAL_FRAC = 0.1
BUFFER = 1000
HEAD_HID = 32
GRAN = "synapse"                                                  # gain-SYNAPSE only
OPT = "adam"                                                      # Adam only
METRIC = "taskil"
MAIN_GRID = {"lr": [3e-4, 1e-3, 3e-3], "ep": [5, 10, 20]}         # same budget as the class-IL ER tune
NEURO_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
MECHS = [("free", None), ("all4", True)]                          # (name, standardize); free has no sig
BASES = ["naive", "er"]
REPORT_SEEDS = [42, 43, 44]
TSV = Path(__file__).resolve().parent / "pt7_tuned_neuro_taskil_results.tsv"


# ------------------------------- loaders -------------------------------
def build_loaders(sequence, val_frac):
    ds = SplitMNIST(sequence=sequence, val_frac=val_frac)
    if val_frac > 0:
        loaders = [(ds.get_task_loaders(t, 64)[0], ds.get_task_val_loader(t, 64)) for t in range(5)]
    else:
        loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    return loaders, list(sequence)


# ------------------------------- task-IL eval (2-way masked) -------------------------------
@torch.no_grad()
def _mask_to(logits, allowed):
    add = torch.full_like(logits, float("-inf"))
    add[:, allowed[0]] = 0.0; add[:, allowed[1]] = 0.0
    return logits + add


@torch.no_grad()
def taskil_eval_baseline(net, loaders, sequence):
    net.eval(); accs = []
    for i in range(5):
        allowed = sequence[i]; c = tot = 0
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV)
            pred = _mask_to(net.plain(x)[0], allowed).argmax(1)
            c += (pred == y).sum().item(); tot += len(y)
        accs.append(c / tot)
    return float(np.mean(accs))


@torch.no_grad()
def taskil_eval_gate(net, gate, heads, is_const, loaders, sequence, K):
    net.eval(); accs = []
    for i in range(5):
        allowed = sequence[i]; c = tot = 0
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV)
            m = torch.ones(x.size(0), K, device=DEV) if is_const else heads(x)
            pred = _mask_to(gate(net, m, x), allowed).argmax(1)
            c += (pred == y).sum().item(); tot += len(y)
        accs.append(c / tot)
    return float(np.mean(accs))


# ------------------------------- baselines (masked-CE training) -------------------------------
def run_baseline(method, lr, epochs, loaders, sequence, seed):
    p7.seed_all(seed)
    net = p7.Net().to(DEV)
    opt = p7._opt(OPT, net.parameters(), lr)
    buf = p7.Reservoir(BUFFER)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if method == "naive":
                    loss = p7.masked_ce(net.plain(x)[0], y)
                else:                                              # er: masked-CE per-sample over current+replay
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    loss = p7.masked_ce(net.plain(torch.cat(Xs))[0], torch.cat(Ys))
                opt.zero_grad(); loss.backward(); opt.step()
                if method == "er":
                    buf.add(x, y)
    return taskil_eval_baseline(net, loaders, sequence)


# --------------- modulated: nobuf (naive+gate) / er-own (er+gate), decoupled neuro_lr ---------------
def _build_mod(name, standardize, seed):
    p7.seed_all(seed)
    drivers, is_free, is_const = p7.cell_spec(name)
    layers = p7.DRIVER_LAYERS.get(name, None)
    net = p7.Net().to(DEV)
    gate = p7.make_gate(GRAN, len(drivers), layers)
    heads = None if is_const else p7.Heads(len(drivers), hid=HEAD_HID).to(DEV)
    sig = None if (is_free or is_const) else p7.Signals(drivers, standardize=standardize)
    return net, gate, heads, sig, is_free, is_const


def _opts(net, gate, heads, is_free, main_lr, neuro_lr):
    main_opt = p7._opt(OPT, list(net.parameters()), main_lr)
    neuro_opt = torch.optim.Adam(gate.params() + (list(heads.parameters()) if is_free else []), neuro_lr)
    head_opt = (torch.optim.Adam(heads.parameters(), neuro_lr)
                if (heads is not None and not is_free) else None)
    return main_opt, neuro_opt, head_opt


def run_modulated(base, name, standardize, main_lr, neuro_lr, epochs, loaders, sequence, seed):
    net, gate, heads, sig, is_free, is_const = _build_mod(name, standardize if standardize is not None else True, seed)
    K = p7._K(gate, GRAN)
    main_opt, neuro_opt, head_opt = _opts(net, gate, heads, is_free, main_lr, neuro_lr)
    buf = p7.Reservoir(BUFFER) if base == "er" else None
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if base == "er":                                   # er-own: gate on the current+replay batch
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                else:                                              # nobuf: naive main, current task only
                    Xm, Ym = x.view(x.size(0), -1), y
                m = p7._m(heads, is_const, Xm, K)
                m_gate = m if is_free else m.detach()
                loss = p7.masked_ce(gate(net, m_gate, Xm), Ym)     # per-sample masked (task-IL training)
                main_opt.zero_grad(); neuro_opt.zero_grad()
                loss.backward()
                main_opt.step(); neuro_opt.step()
                if head_opt is not None:                           # bio head regression (replay via Xm for er)
                    T = sig.targets(net, Xm, Ym)
                    hloss = F.mse_loss(heads(Xm), T)
                    head_opt.zero_grad(); hloss.backward(); head_opt.step()
                if base == "er":
                    buf.add(x, y)
    return taskil_eval_gate(net, gate, heads, is_const, loaders, sequence, K)


# ------------------------------- ledger -------------------------------
def load_done():
    if not TSV.exists():
        return {}
    out = {}
    for ln in TSV.read_text().splitlines():
        if ln.strip():
            f = ln.split("\t"); out[f[0]] = float(f[1])
    return out


def record(tag, acc):
    with open(TSV, "a") as fh:
        fh.write(f"{tag}\t{acc:.4f}\n")


def mstd(std):
    return "-" if std is None else f"std{int(std)}"


# ------------------------------- phases -------------------------------
def do_tune_main(done):
    val_loaders, val_seq = build_loaders(VAL_SEQ, VAL_FRAC)
    print("[tune-main] task-IL, adam; grid", MAIN_GRID, "\n", flush=True)
    for base in BASES:
        for lr in MAIN_GRID["lr"]:
            for ep in MAIN_GRID["ep"]:
                tag = f"tune-main|{METRIC}|{base}|adam|lr{lr:g}|ep{ep}"
                if tag in done:
                    continue
                acc = run_baseline(base, lr, ep, val_loaders, val_seq, seed=42)
                print(f"  TUNE-MAIN {base:5s} lr{lr:<7g} ep{ep:<3d} | val={acc:.4f}", flush=True)
                record(tag, acc); done[tag] = acc


def best_main(done, base):
    cells = [(k, v) for k, v in done.items() if k.startswith(f"tune-main|{METRIC}|{base}|adam|")]
    if not cells:
        return None
    k, v = max(cells, key=lambda kv: kv[1])
    lr = float(k.split("|lr")[1].split("|")[0]); ep = int(k.split("|ep")[1])
    return lr, ep, v


def do_tune_neuro(done):
    val_loaders, val_seq = build_loaders(VAL_SEQ, VAL_FRAC)
    for base in BASES:
        bm = best_main(done, base)
        if bm is None:
            print(f"  [tune-neuro] no tuned main for {base} — run --phase tune-main first", flush=True); continue
        main_lr, ep, _ = bm
        print(f"\n[tune-neuro] base={base} main frozen lr{main_lr:g} ep{ep}; sweep neuro_lr {NEURO_LRS}", flush=True)
        for name, std in MECHS:
            for nlr in NEURO_LRS:
                tag = f"tune-neuro|{METRIC}|{base}|{name}|synapse|{mstd(std)}|hid{HEAD_HID}|nlr{nlr:g}"
                if tag in done:
                    continue
                acc = run_modulated(base, name, std, main_lr, nlr, ep, val_loaders, val_seq, seed=42)
                print(f"  TUNE-NEURO {base:5s} {name:5s} nlr{nlr:<7g} | val={acc:.4f}", flush=True)
                record(tag, acc); done[tag] = acc


def best_nlr(done, base, name, std):
    pref = f"tune-neuro|{METRIC}|{base}|{name}|synapse|{mstd(std)}|hid{HEAD_HID}|nlr"
    cells = [(k, v) for k, v in done.items() if k.startswith(pref)]
    if not cells:
        return None
    k, v = max(cells, key=lambda kv: kv[1])
    return float(k.split("|nlr")[1]), v


def do_report(done):
    test_loaders, test_seq = build_loaders(SEQ, 0.0)
    rows = []                                                      # (label, mean, std, cfg)
    for base in BASES:
        bm = best_main(done, base)
        if bm is None:
            print(f"  [report] no tuned main for {base}", flush=True); continue
        main_lr, ep, valacc = bm
        # baseline, 3 seeds
        accs = []
        for seed in REPORT_SEEDS:
            tag = f"report|{METRIC}|{base}|-|-|adam|lr{main_lr:g}|ep{ep}|s{seed}"
            if tag not in done:
                acc = run_baseline(base, main_lr, ep, test_loaders, test_seq, seed)
                print(f"  REPORT {base:5s} seed{seed} | test={acc:.4f}", flush=True)
                record(tag, acc); done[tag] = acc
            accs.append(done[tag])
        rows.append((base, np.mean(accs), np.std(accs), f"lr{main_lr:g} ep{ep}"))
        # modulated, 3 seeds
        for name, std in MECHS:
            bn = best_nlr(done, base, name, std)
            if bn is None:
                print(f"  [report] no tuned neuro_lr for {base}/{name}", flush=True); continue
            nlr, nval = bn
            maccs = []
            for seed in REPORT_SEEDS:
                tag = f"report|{METRIC}|{base}|{name}|synapse|{mstd(std)}|adam|lr{main_lr:g}|ep{ep}|nlr{nlr:g}|s{seed}"
                if tag not in done:
                    acc = run_modulated(base, name, std, main_lr, nlr, ep, test_loaders, test_seq, seed)
                    print(f"  REPORT {base:5s}+{name:5s} nlr{nlr:<7g} seed{seed} | test={acc:.4f}", flush=True)
                    record(tag, acc); done[tag] = acc
                maccs.append(done[tag])
            rows.append((f"{base}+{name}", np.mean(maccs), np.std(maccs), f"lr{main_lr:g} ep{ep} nlr{nlr:g}"))

    print("\n================  TASK-IL  (adam, gain-synapse, best configs, 3 seeds)  ================", flush=True)
    base_means = {r[0]: r[1] for r in rows}
    for label, mean, std, cfg in rows:
        delta = ""
        if "+" in label:
            b = label.split("+")[0]
            if b in base_means:
                delta = f"  (d{b} {mean - base_means[b]:+.4f})"
        print(f"  {label:14s} {mean:.4f} ± {std:.4f}   [{cfg}]{delta}", flush=True)
    # the clean gate test: all4 vs free at matched (base, main, ep)
    for base in BASES:
        if f"{base}+all4" in base_means and f"{base}+free" in base_means:
            print(f"  >> {base}: all4 - free = {base_means[f'{base}+all4'] - base_means[f'{base}+free']:+.4f} "
                  f"(gate signal vs dead-gate control)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "tune-main", "tune-neuro", "report"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={DEV}  (pt7 TASK-IL tuning; adam; gain-synapse; val_seq={VAL_SEQ})\n", flush=True)
    done = load_done() if args.resume else {}
    if args.phase in ("all", "tune-main"):
        do_tune_main(done)
    if args.phase in ("all", "tune-neuro"):
        do_tune_neuro(done)
    if args.phase in ("all", "report"):
        do_report(done)
    print("\nALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
