"""pt7 STANDARD regime, TUNED (user-requested): full MNIST, single task, 10-way CE.

Drivers 5ht-const, NE, vecproj, vec_h1proj, all4 (+ vanilla baseline), standardize on/off, gran {neuron,
synapse}. "Tuned" = epochs selected on the held-out VAL split (never test) at a standard-good lr per
optimizer (adam 1e-3, sgd 1e-2); a full lr grid over neu+syn x full-MNIST was out of compute budget
(documented scope). Reports test acc at the best-val epoch. Ledger pt7_std_tuned_results.tsv (`--resume`).
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_neuromodulators as p7                                  # noqa: E402
import pt7_variants as pv                                         # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from data import get_standard_loaders                            # noqa: E402

DEV = p7.DEV; CE = nn.CrossEntropyLoss()
TSV = Path(__file__).resolve().parent / "pt7_std_tuned_results.tsv"
LR = {"adam": 1e-3, "sgd": 1e-2}
MAX_EPOCHS = 6


@torch.no_grad()
def _acc(fwd, loader):
    c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        c += (fwd(x).argmax(1) == y).sum().item(); tot += len(y)
    return c / tot


def run_std_tuned(driver, gran, std, opt_kind, seed=42):
    p7.seed_all(seed)
    tr, va, te = get_standard_loaders(batch_size=64)
    lr = LR[opt_kind]; net = p7.Net().to(DEV)

    if driver == "vanilla":
        opt = p7._opt(opt_kind, net.parameters(), lr)
        fwd = lambda x: net.plain(x)[0]                                          # noqa: E731

        def step(x, y):
            loss = CE(fwd(x), y); opt.zero_grad(); loss.backward(); opt.step()

    elif driver in ("vecproj", "vec_h1proj"):
        drv = pv.NEDriver(driver, std); gate = p7.make_gate(gran, drv.K(), None)
        opt = p7._opt(opt_kind, list(net.parameters()) + gate.params(), lr)
        fwd = lambda x: gate(net, drv.value(net, x, update=False), x)            # noqa: E731

        def step(x, y):
            loss = CE(gate(net, drv.value(net, x), x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    elif driver == "5ht-const":
        gate = p7.make_gate(gran, 1, None)
        opt = p7._opt(opt_kind, list(net.parameters()) + gate.params(), lr)
        fwd = lambda x: gate(net, torch.ones(x.size(0), 1, device=DEV), x)       # noqa: E731

        def step(x, y):
            loss = CE(gate(net, torch.ones(x.size(0), 1, device=DEV), x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    else:                                                                        # NE, all4 (head-based)
        drivers, _, _ = p7.cell_spec(driver); layers = p7.DRIVER_LAYERS.get(driver, None)
        gate = p7.make_gate(gran, len(drivers), layers); heads = p7.Heads(len(drivers)).to(DEV)
        sig = p7.Signals(drivers, standardize=std, loss_fn=pv.per_sample_ce_plain)
        main_opt = p7._opt(opt_kind, list(net.parameters()) + gate.params(), lr)
        head_opt = torch.optim.Adam(heads.parameters(), lr)
        fwd = lambda x: gate(net, heads(x), x)                                   # noqa: E731

        def step(x, y):
            m = heads(x).detach()
            loss = CE(gate(net, m, x), y)
            main_opt.zero_grad(); loss.backward(); main_opt.step()
            hl = F.mse_loss(heads(x), sig.targets(net, x, y))
            head_opt.zero_grad(); hl.backward(); head_opt.step()

    best_val = -1.0; best_test = 0.0; best_ep = 0
    for ep in range(MAX_EPOCHS):
        net.train()
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV); step(x, y)
        net.eval()
        v = _acc(fwd, va)
        if v > best_val:
            best_val = v; best_test = _acc(fwd, te); best_ep = ep + 1
    return dict(pred=best_test, val=best_val, epoch=best_ep)


def build_cells():
    cells = []                                          # (driver, gran, std)
    for opt in ("sgd", "adam"):
        cells.append(("vanilla", "-", True, opt))
        for gran in ("neuron", "synapse"):
            cells.append(("5ht-const", gran, True, opt))
            for n in ("NE", "all4", "vecproj", "vec_h1proj"):
                for std in (True, False):
                    cells.append((n, gran, std, opt))
    return cells


def load_done():
    return {ln.split("\t", 1)[0] for ln in TSV.read_text().splitlines()} if TSV.exists() else set()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); done = load_done() if args.resume else set()
    print(f"device={DEV}  (pt7 standard TUNED; lr adam 1e-3 / sgd 1e-2, val-selected epochs<= {MAX_EPOCHS})\n",
          flush=True)
    for driver, gran, std, opt in build_cells():
        tag = f"{driver}|{gran}|std{int(std)}|{opt}"
        if tag in done:
            continue
        r = run_std_tuned(driver, gran, std, opt)
        print(f"  {driver:11s} {gran:7s} std{int(std)} {opt:4s} | test={r['pred']:.4f} "
              f"(val={r['val']:.4f} @ep{r['epoch']})", flush=True)
        with open(TSV, "a") as f:
            f.write(f"{tag}\t{r['pred']:.4f}\t{r['val']:.4f}\t{r['epoch']}\n")
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
