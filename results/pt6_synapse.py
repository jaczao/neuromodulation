"""pt6 — gain-SYNAPSE for soft_mlp (soft-blend AND hard-argmax eval).

Per the corrected deferral rationale (pt6_driver_mechanisms.md): per-synapse `soft_mlp` is RUNNABLE.
It gates via a (T, n_syn) LOOKUP (2.4M params -- the 374M blow-up is mean_image/embedding only), and
it trains on TRUE task ids so a batch holds <= T=5 distinct gates and groups into <=5 masked matmuls.

Extra simplification used here for the SOFT blend (removes the "per-sample Gamma" blocker entirely):
    Gamma_i = sum_t p_it * Gamma_t   and   (Gamma . W) x  is LINEAR in Gamma, so
    (Gamma_i . W) x_i + b = sum_t p_it * [ (Gamma_t . W) x_i + b ]        (uses sum_t p_it = 1)
i.e. the blended layer output is the p-weighted sum of the T per-task gated outputs -> T matmuls per
layer, NO (B, d_out, d_in) expansion. Exact, not an approximation (verified against a per-sample
reference in the smoke test).

Gated layers = net0 (400x784), net2 (400x400), net4 (10x400)  -> n_syn = 477 600 ("h0,h1,out").
Gain form: Gamma = 1 + P (unbounded), matching the neuron study. class-IL, seed42 lr1e-3 ep5 buf1000,
1 seed. Refs: er-sgd 0.723, er-adam 0.895; NEURON soft_mlp soft er-own/adam 0.886, er-own/sgd 0.856.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pt6_driver_mechanisms import (  # noqa: E402
    CE, DEV, Net, Reservoir, SEQ, _opt, masked_ce, seed_all,
)
from data import SplitMNIST  # noqa: E402

LAYERS = ((400, 784), (400, 400), (10, 400))
NSYN = sum(a * b for a, b in LAYERS)          # 477_600
T = 5


class SoftMLPSyn(nn.Module):
    """Per-synapse gate table P:(T, n_syn) + replay-trained task-inference net g(x)."""

    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(torch.zeros(T, NSYN))
        self.gh = nn.Linear(784, 128); self.go = nn.Linear(128, T)

    def mats(self):
        out, off = [], 0
        for a, b in LAYERS:
            out.append(self.P[:, off:off + a * b].view(T, a, b)); off += a * b
        return out

    def task_logits(self, x):
        return self.go(F.relu(self.gh(x.view(x.size(0), -1))))

    def gate_params(self):
        return [self.P]

    def inf_params(self):
        return list(self.gh.parameters()) + list(self.go.parameters())

    def layer_stats(self):
        off, out = 0, {}
        for name, (a, b) in zip(("net0", "net2", "net4"), LAYERS):
            out[name] = float(self.P[:, off:off + a * b].detach().abs().mean()); off += a * b
        return out


def _wb(net):
    return ((net.l0.weight, net.l0.bias), (net.l1.weight, net.l1.bias), (net.l2.weight, net.l2.bias))


def fwd_grouped(net, mech, X, tids):
    """Each sample carries ONE task -> group and do <=T masked matmuls (the pt5 er_task_id path)."""
    X = X.view(X.size(0), -1); M = mech.mats(); wb = _wb(net)
    out = torch.zeros(X.size(0), 10, device=X.device, dtype=X.dtype)
    for t in tids.unique():
        idx = (tids == t).nonzero().squeeze(1)
        h = X[idx]
        for li in range(3):
            W, b = wb[li]
            h = F.linear(h, (1 + M[li][t]) * W, b)
            if li < 2:
                h = F.relu(h)
        out = out.index_copy(0, idx, h)
    return out


def fwd_mixed(net, mech, X, p):
    """Soft blend, EXACT: layer output = sum_t p_t * [(Gamma_t . W) h + b]  (linear in Gamma)."""
    h = X.view(X.size(0), -1); M = mech.mats(); wb = _wb(net)
    for li in range(3):
        W, b = wb[li]
        acc = 0
        for t in range(T):
            acc = acc + p[:, t:t + 1] * F.linear(h, (1 + M[li][t]) * W, b)
        h = acc
        if li < 2:
            h = F.relu(h)
    return h


def build(seed=42):
    seed_all(seed)
    ds = SplitMNIST(sequence=SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    return loaders, Net().to(DEV), SoftMLPSyn().to(DEV)


def train(mech, net, loaders, arm, opt_kind, lr=1e-3, epochs=5, buffer=1000):
    buf = Reservoir(buffer)
    inf_opt = torch.optim.Adam(mech.inf_params(), lr=lr)

    def inf_step(X, Tt):
        ce = CE(mech.task_logits(X), Tt)
        inf_opt.zero_grad(); ce.backward(); inf_opt.step()

    if arm == "er-own":
        opt = _opt(opt_kind, list(net.parameters()) + list(mech.gate_params()), lr)
        for t in range(5):
            for _ in range(epochs):
                for x, y in loaders[t][0]:
                    x, y = x.to(DEV), y.to(DEV)
                    Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [torch.full((x.size(0),), t, device=DEV)]
                    r = buf.sample_any(64)
                    if r is not None:
                        xr, yr = r[0].to(DEV), r[1].to(DEV)
                        Xs.append(xr); Ys.append(yr)
                        Ts.append(torch.div(yr, 2, rounding_mode="floor"))
                    Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                    loss = CE(fwd_grouped(net, mech, Xm, Tm), Ym)
                    opt.zero_grad(); inf_opt.zero_grad(); loss.backward(); opt.step()
                    inf_step(Xm, Tm)
                    buf.add(x, y)
        return

    main_opt = _opt(opt_kind, net.parameters(), lr)
    gate_opt = torch.optim.Adam(mech.gate_params(), lr=lr)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV); tids = torch.full((x.size(0),), t, device=DEV)
                loss = masked_ce(fwd_grouped(net, mech, x, tids), y)   # main: naive (gate detached below)
                net.zero_grad(); mech.zero_grad(); loss.backward(); main_opt.step()
                buf.add(x, y)
                Xi, Ti = [x.view(x.size(0), -1)], [tids]
                r = buf.sample_any(64)
                if r is not None:
                    Xi.append(r[0].to(DEV)); Ti.append(torch.div(r[1].to(DEV), 2, rounding_mode="floor"))
                inf_step(torch.cat(Xi), torch.cat(Ti))
                Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [tids]
                for j in range(t):
                    s = buf.sample_task(j, 64)
                    if s is not None:
                        Xs.append(s[0].to(DEV)); Ys.append(s[1].to(DEV))
                        Ts.append(torch.full((s[0].size(0),), j, device=DEV))
                Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                meta = masked_ce(fwd_grouped(net, mech, Xm, Tm), Ym)
                net.zero_grad(); mech.zero_grad(); meta.backward(); gate_opt.step()


@torch.no_grad()
def evaluate(mech, net, loaders):
    net.eval()
    o = s = h = inf = tot = 0
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            p = F.softmax(mech.task_logits(x), dim=1)
            tid_o = torch.full((b,), i, device=DEV)
            o += (fwd_grouped(net, mech, x, tid_o).argmax(1) == y).sum().item()
            s += (fwd_mixed(net, mech, x, p).argmax(1) == y).sum().item()
            h += (fwd_grouped(net, mech, x, p.argmax(1)).argmax(1) == y).sum().item()
            inf += (p.argmax(1) == i).sum().item(); tot += b
    return dict(oracle=o / tot, soft=s / tot, hard=h / tot, infer=inf / tot) | mech.layer_stats()


def main():
    print(f"device={DEV}  gain-SYNAPSE soft_mlp (n_syn={NSYN})")
    print("refs: er-sgd 0.723, er-adam 0.895 | NEURON soft er-own/adam 0.886, er-own/sgd 0.856\n")
    for arm in ("er-own", "buf-own"):
        for opt in ("sgd", "adam"):
            loaders, net, mech = build()
            train(mech, net, loaders, arm, opt)
            r = evaluate(mech, net, loaders)
            print(f"  {arm:7s} {opt:4s} | oracle={r['oracle']:.4f} soft={r['soft']:.4f} "
                  f"hard={r['hard']:.4f} infer={r['infer']:.4f} | |P| net0={r['net0']:.2e} "
                  f"net2={r['net2']:.2e} net4={r['net4']:.2e}", flush=True)


if __name__ == "__main__":
    main()
