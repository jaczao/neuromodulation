"""Discriminator (b): does the BIG selector (784->400->400->T) just need more optimization?

Sweep the selector's own Adam lr and its training epochs, main/gate optimization pinned at the
standard 1e-3, and read train-infer / test-infer / acc. If train+test infer climb toward the
small net's ~0.89, the (a) under-training diagnosis is confirmed and the fix is budget/lr.

Selector lr is DECOUPLED from main+gate lr here (p6s.train couples them). infer is optimizer-
independent (shown in (a)), so we run a single optimizer (adam). class-IL, seed 42, buffer 1000.
Reference (a): small lr1e-3 ep5 test-infer 0.892 (er) / 0.883 (buf); big 0.864 / 0.870.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pt6_synapse as p6s                       # noqa: E402
import pt6_softmlp_bigsel as bs                 # noqa: E402
from pt6_synapse import CE, DEV, Reservoir, fwd_grouped, masked_ce  # noqa: E402

LEDGER = HERE / "pt6_softmlp_selsweep_results.tsv"
OPT = "adam"          # infer is opt-independent; acc uses this main optimizer
MAIN_LR = GATE_LR = 1e-3


def train_bufown(mech, net, loaders, inf_lr, epochs):
    buf = Reservoir(bs.BUF)
    inf_opt = torch.optim.Adam(mech.inf_params(), lr=inf_lr)
    main_opt = p6s._opt(OPT, net.parameters(), MAIN_LR)
    gate_opt = torch.optim.Adam(mech.gate_params(), lr=GATE_LR)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV); tids = torch.full((x.size(0),), t, device=DEV)
                loss = masked_ce(fwd_grouped(net, mech, x, tids), y)
                net.zero_grad(); mech.zero_grad(); loss.backward(); main_opt.step()
                buf.add(x, y)
                Xi, Ti = [x.view(x.size(0), -1)], [tids]
                r = buf.sample_any(64)
                if r is not None:
                    Xi.append(r[0].to(DEV)); Ti.append(torch.div(r[1].to(DEV), 2, rounding_mode="floor"))
                ce = CE(mech.task_logits(torch.cat(Xi)), torch.cat(Ti))
                inf_opt.zero_grad(); ce.backward(); inf_opt.step()
                Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [tids]
                for j in range(t):
                    s = buf.sample_task(j, 64)
                    if s is not None:
                        Xs.append(s[0].to(DEV)); Ys.append(s[1].to(DEV))
                        Ts.append(torch.full((s[0].size(0),), j, device=DEV))
                meta = masked_ce(fwd_grouped(net, mech, torch.cat(Xs), torch.cat(Ts)), torch.cat(Ys))
                net.zero_grad(); mech.zero_grad(); meta.backward(); gate_opt.step()


def train_erown(mech, net, loaders, inf_lr, epochs):
    buf = Reservoir(bs.BUF)
    inf_opt = torch.optim.Adam(mech.inf_params(), lr=inf_lr)
    opt = p6s._opt(OPT, list(net.parameters()) + list(mech.gate_params()), MAIN_LR)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [torch.full((x.size(0),), t, device=DEV)]
                r = buf.sample_any(64)
                if r is not None:
                    xr, yr = r[0].to(DEV), r[1].to(DEV)
                    Xs.append(xr); Ys.append(yr); Ts.append(torch.div(yr, 2, rounding_mode="floor"))
                Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                loss = CE(fwd_grouped(net, mech, Xm, Tm), Ym)
                opt.zero_grad(); inf_opt.zero_grad(); loss.backward(); opt.step()
                ce = CE(mech.task_logits(Xm), Tm)
                inf_opt.zero_grad(); ce.backward(); inf_opt.step()
                buf.add(x, y)


@torch.no_grad()
def train_infer(mech, net, loaders):
    net.eval(); c = n = 0
    for i in range(5):
        for x, y in loaders[i][0]:
            x = x.to(DEV)
            c += (mech.task_logits(x).argmax(1) == i).sum().item(); n += x.size(0)
    return c / n


def run(arm, inf_lr, epochs):
    loaders, net, mech = bs.build(True)                 # always the BIG selector
    (train_bufown if arm == "buf-own" else train_erown)(mech, net, loaders, inf_lr, epochs)
    tr = train_infer(mech, net, loaders)
    ev = p6s.evaluate(mech, net, loaders)               # test infer + soft/oracle acc
    return tr, ev


CONFIGS = [(3e-4, 5), (1e-3, 5), (3e-3, 5), (1e-3, 10), (1e-3, 20)]


def main():
    done = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                done.add(line.split("\t")[0])
    print(f"device={DEV}  BIG selector lr/epoch sweep  main/gate lr={MAIN_LR:g}  opt={OPT}  seed=42\n", flush=True)
    print(f"{'arm':8s} {'inf_lr':>7s} {'ep':>3s} | {'tr-inf':>7s} {'te-inf':>7s} {'soft':>7s} {'oracle':>7s}", flush=True)
    for arm in ("buf-own", "er-own"):
        for inf_lr, ep in CONFIGS:
            tag = f"{arm}|big|inflr{inf_lr:g}|ep{ep}"
            if tag in done:
                print(f"{arm:8s} {inf_lr:7g} {ep:3d} | (cached)", flush=True); continue
            tr, ev = run(arm, inf_lr, ep)
            with LEDGER.open("a") as f:
                f.write(f"{tag}\t{tr:.4f}\t{ev['infer']:.4f}\t{ev['soft']:.4f}\t{ev['oracle']:.4f}\n")
            print(f"{arm:8s} {inf_lr:7g} {ep:3d} | {tr:7.4f} {ev['infer']:7.4f} "
                  f"{ev['soft']:7.4f} {ev['oracle']:7.4f}", flush=True)


if __name__ == "__main__":
    main()
