"""soft_mlp (gain-SYNAPSE) with the task-inference selector at TWO sizes, all three arms.

User-requested: re-run soft_mlp across arms {nobuf, buf-own, er-own} with the task-inference
net (a) AS-IS (784->128->T, the pt6_synapse default) and (b) AS BIG AS THE MAIN NET
(784->400->400->T, matching Net's two 400-wide hidden layers). Show oracle-free acc (soft &
hard) and task-inference acc (infer).

Operating point = pt6_synapse convention: class-IL, seed 42, lr 1e-3, ep 5, buffer 1000, both
optimizers. small-net cells reproduce pt6_synapse.md (sanity anchor); the ONLY new variable is
selector size. Gate stays per-synapse (P:(T, n_syn)). nobuf = buf-own minus the buffer (selector
+ gate + main all on the current task only, no replay anywhere).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pt6_synapse as p6s             # noqa: E402

DEV = p6s.DEV
T = p6s.T
SEED, LR, EP, BUF = 42, 1e-3, 5, 1000
LEDGER = HERE / "pt6_softmlp_bigsel_results.tsv"


class SoftMLPSynBig(p6s.SoftMLPSyn):
    """Same per-synapse gate table P, but the task-inference net matches the main net:
    784 -> 400 -> 400 -> T (vs the default 784 -> 128 -> T)."""

    def __init__(self):
        super().__init__()                              # keeps P; sets small gh/go (overridden below)
        self.gh = nn.Linear(784, 400)
        self.g2 = nn.Linear(400, 400)
        self.go = nn.Linear(400, T)

    def task_logits(self, x):
        h = F.relu(self.gh(x.view(x.size(0), -1)))
        h = F.relu(self.g2(h))
        return self.go(h)

    def inf_params(self):
        return list(self.gh.parameters()) + list(self.g2.parameters()) + list(self.go.parameters())


def build(big):
    p6s.seed_all(SEED)                                  # mirror p6s.build exactly (loaders -> net -> mech)
    ds = p6s.SplitMNIST(sequence=p6s.SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    net = p6s.Net().to(DEV)
    mech = (SoftMLPSynBig() if big else p6s.SoftMLPSyn()).to(DEV)
    return loaders, net, mech


def train_nobuf(mech, net, loaders, opt_kind):
    """No buffer anywhere: naive masked-CE main + selector + gate meta, current task only."""
    inf_opt = torch.optim.Adam(mech.inf_params(), lr=LR)
    main_opt = p6s._opt(opt_kind, net.parameters(), LR)
    gate_opt = torch.optim.Adam(mech.gate_params(), lr=LR)
    for t in range(5):
        for _ in range(EP):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                tids = torch.full((x.size(0),), t, device=DEV)
                loss = p6s.masked_ce(p6s.fwd_grouped(net, mech, x, tids), y)
                net.zero_grad(); mech.zero_grad(); loss.backward(); main_opt.step()
                ce = p6s.CE(mech.task_logits(x), tids)
                inf_opt.zero_grad(); ce.backward(); inf_opt.step()
                meta = p6s.masked_ce(p6s.fwd_grouped(net, mech, x, tids), y)
                net.zero_grad(); mech.zero_grad(); meta.backward(); gate_opt.step()


def run(arm, big, opt_kind):
    loaders, net, mech = build(big)
    if arm == "nobuf":
        train_nobuf(mech, net, loaders, opt_kind)
    else:
        p6s.train(mech, net, loaders, arm, opt_kind, lr=LR, epochs=EP, buffer=BUF)
    return p6s.evaluate(mech, net, loaders)


def _done():
    d = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                d.add(line.split("\t")[0])
    return d


def main():
    done = _done()
    print(f"device={DEV}  soft_mlp gain-SYNAPSE  selector-size sweep  "
          f"seed={SEED} lr={LR:g} ep={EP} buf={BUF}\n", flush=True)
    print(f"{'arm':8s} {'sel':6s} {'opt':4s} | {'soft':>7s} {'hard':>7s} {'oracle':>7s} {'infer':>7s}", flush=True)
    for opt in ("sgd", "adam"):
        for arm in ("nobuf", "buf-own", "er-own"):
            for big in (False, True):
                sel = "big" if big else "small"
                tag = f"soft_mlp|synapse|{arm}|{sel}|{opt}|lr{LR:g}|ep{EP}"
                if tag in done:
                    print(f"{arm:8s} {sel:6s} {opt:4s} | (cached)", flush=True)
                    continue
                r = run(arm, big, opt)
                with LEDGER.open("a") as f:
                    f.write(tag + "\t" + "\t".join(
                        f"{r[k]:.4f}" for k in ("soft", "hard", "oracle", "infer")) + "\n")
                print(f"{arm:8s} {sel:6s} {opt:4s} | {r['soft']:7.4f} {r['hard']:7.4f} "
                      f"{r['oracle']:7.4f} {r['infer']:7.4f}", flush=True)


if __name__ == "__main__":
    main()
