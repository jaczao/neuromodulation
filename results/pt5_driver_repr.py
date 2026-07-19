"""pt5 DRIVER-REPRESENTATION study: is the one-hot task_id driver replaceable, and does any
alternative survive without the eval-time oracle?  (self-contained; class-IL Split MNIST)

The pt5 gate is  driver -> projection -> per-task gain gate.  Here we vary the DRIVER:
  onehot : raw_t = P[t]                 independent per-task rows (the pt5 default)
  lin    : raw_t = mu_t @ W             shared LINEAR map (784->800) over the task-mean image mu_t
  mlp    : raw_t = gf(relu(gh(mu_t)))   shared NONLINEAR map (784->128->800)
each optionally with CENTERED mu (mu_t - mean_t mu_t; inter-task cos 0.82 -> -0.24).

Arms (gain-neuron, unbounded gain g=1+raw on h0,h1; seed42 lr1e-3 ep5 buffer1000):
  buf-own : main net naive on the current task (masked CE); gate trained on a per-task replay
            META-loss (Adam@1e-3). main optimizer {sgd,adam}.
  er-own  : main net + gate trained JOINTLY on the ER batch (current+replay), each sample gated
            by its own task; single optimizer {sgd,adam}; plain CE.

Eval, three ways (the last two need NO task id at inference):
  oracle    : gate for task i = raw_table[i]
  per-image : gate = project(x - center)                             (driver = the image itself)
  nearest   : nn = argmin_i ||(x-center) - mu_i||; gate = raw_table[nn]   (nearest-prototype infer)

KEY FINDINGS (1 seed; full numbers in pt5_driver_repr.md):
  * A CENTERED lin driver MATCHES one-hot UNDER THE ORACLE (er-own/adam ~0.99). The differentiation
    comes from decorrelated inputs + a linear map preserving that geometry; the mlp's relu
    RE-correlates the centered features, so mlp < lin.
  * EVERY non-oracle eval (per-image, nearest) falls BELOW plain ER (~0.75 vs ER-adam 0.894),
    capped by nearest-prototype task inference (~0.76). The gate is fundamentally task-IL: the
    oracle carried the headline, not the driver representation.
  * Overlap is measured on dev(=raw), NOT g(=1+raw): the shared parity offset (the all-ones 1)
    inflates cos(g) toward +1 for gentle gates (|dev| small), hiding real differentiation.

Run:  uv run python results/pt5_driver_repr.py    (full grid ~20 min on MPS; 1 seed, ORACLE-caveat)
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from data import SplitMNIST  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available()
                   else ("cuda" if torch.cuda.is_available() else "cpu"))
SEQ = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]     # default class-IL sequence; label c -> task c//2
CE = nn.CrossEntropyLoss()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def masked_ce(logits, y):
    """Per-sample masked CE: restrict each sample's logits to its own task-pair (pt3 lever B)."""
    task = torch.div(y, 2, rounding_mode="floor")
    allowed = torch.stack([2 * task, 2 * task + 1], dim=1)
    add = torch.full_like(logits, float("-inf"))
    add.scatter_(1, allowed, 0.0)
    return F.cross_entropy(logits + add, y)


class Reservoir:
    def __init__(self, cap):
        self.cap = cap; self.X = torch.zeros(cap, 784); self.Y = torch.zeros(cap, dtype=torch.long)
        self.n = 0; self.filled = 0

    def add(self, x, y):
        x = x.view(x.size(0), -1).cpu(); y = y.cpu()
        for i in range(len(x)):
            if self.filled < self.cap:
                self.X[self.filled] = x[i]; self.Y[self.filled] = y[i]; self.filled += 1
            else:
                j = random.randint(0, self.n)
                if j < self.cap:
                    self.X[j] = x[i]; self.Y[j] = y[i]
            self.n += 1

    def sample_task(self, j, b):
        idx = (torch.div(self.Y[:self.filled], 2, rounding_mode="floor") == j).nonzero().squeeze(1)
        if len(idx) == 0:
            return None
        pick = idx[torch.randint(0, len(idx), (b,))]
        return self.X[pick], self.Y[pick]

    def sample_any(self, b):
        if self.filled == 0:
            return None
        idx = torch.randint(0, self.filled, (b,))
        return self.X[idx], self.Y[idx]


class Net(nn.Module):
    """Main MLP 784-400-400-10 (matches prototype/model.py)."""

    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(784, 400); self.l1 = nn.Linear(400, 400); self.l2 = nn.Linear(400, 10)


def forward_ps(net, raw_bs, x):                     # per-sample gain gate; raw_bs: (B,800)
    x = x.view(x.size(0), -1)
    z0 = F.relu(net.l0(x)) * (1 + raw_bs[:, :400])
    z1 = F.relu(net.l1(z0)) * (1 + raw_bs[:, 400:])
    return net.l2(z1)


class Gate(nn.Module):
    """driver -> (5,800) per-task gate table. kind in {onehot, lin, mlp}. mus already centered
    iff the study requested centering (onehot ignores mus)."""

    def __init__(self, kind, mus):
        super().__init__()
        self.kind = kind
        self.register_buffer("mus", mus)             # (5,784)
        if kind == "onehot":
            self.P = nn.Parameter(torch.zeros(5, 800))
        elif kind == "lin":
            self.W = nn.Parameter(torch.zeros(784, 800))
        elif kind == "mlp":
            self.gh = nn.Linear(784, 128)
            self.gf = nn.Linear(128, 800)
            nn.init.normal_(self.gf.weight, std=1e-3); nn.init.zeros_(self.gf.bias)  # near-parity, nonzero
        else:
            raise ValueError(kind)

    def raw_table(self, T):
        if self.kind == "onehot":
            return self.P[:T]
        if self.kind == "lin":
            return self.mus[:T] @ self.W
        return self.gf(F.relu(self.gh(self.mus[:T])))

    def project(self, d):                            # gate for an arbitrary driver d:(B,784) (lin/mlp)
        if self.kind == "lin":
            return d @ self.W
        return self.gf(F.relu(self.gh(d)))


def build(kind, center, seed=42):
    seed_all(seed)
    ds = SplitMNIST(sequence=SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    raw_mus = torch.stack([torch.cat([xb.view(xb.size(0), -1) for xb, _ in loaders[t][0]]).mean(0)
                           for t in range(5)]).to(DEV)
    cvec = raw_mus.mean(0) if center else torch.zeros(784, device=DEV)   # offline centering
    return loaders, Net().to(DEV), Gate(kind, raw_mus - cvec).to(DEV), cvec


def train_bufown(net, gate, loaders, opt_kind, lr=1e-3, epochs=5, buffer=1000):
    main_opt = (torch.optim.SGD if opt_kind == "sgd" else torch.optim.Adam)(net.parameters(), lr=lr)
    gate_opt = torch.optim.Adam(gate.parameters(), lr=lr)
    buf = Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                raw_t = gate.raw_table(t + 1)[t].detach().unsqueeze(0).expand(x.size(0), -1)
                loss = masked_ce(forward_ps(net, raw_t, x), y)          # main step: naive, gate detached
                net.zero_grad(); gate.zero_grad(); loss.backward(); main_opt.step()
                buf.add(x, y)
                Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [torch.full((x.size(0),), t, device=DEV)]
                for j in range(t):                                       # meta step: per-task replay
                    s = buf.sample_task(j, 64)
                    if s is not None:
                        Xs.append(s[0].to(DEV)); Ys.append(s[1].to(DEV))
                        Ts.append(torch.full((s[0].size(0),), j, device=DEV))
                Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                meta = masked_ce(forward_ps(net, gate.raw_table(t + 1)[Tm], Xm), Ym)
                net.zero_grad(); gate.zero_grad(); meta.backward(); gate_opt.step()


def train_erown(net, gate, loaders, opt_kind, lr=1e-3, epochs=5, buffer=1000):
    opt = (torch.optim.SGD if opt_kind == "sgd" else torch.optim.Adam)(
        list(net.parameters()) + list(gate.parameters()), lr=lr)
    buf = Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [torch.full((x.size(0),), t, device=DEV)]
                rep = buf.sample_any(64)                                # ER batch: current + replay
                if rep is not None:
                    xr, yr = rep[0].to(DEV), rep[1].to(DEV)
                    Xs.append(xr); Ys.append(yr); Ts.append(torch.div(yr, 2, rounding_mode="floor"))
                Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                loss = CE(forward_ps(net, gate.raw_table(t + 1)[Tm], Xm), Ym)   # plain CE, own-task gate
                opt.zero_grad(); loss.backward(); opt.step()
                buf.add(x, y)


def train_baseline(method, opt_kind, lr=1e-3, epochs=5, buffer=1000, seed=42):
    """No-gate baselines. naive = masked CE; er = plain CE + reservoir replay."""
    seed_all(seed)
    ds = SplitMNIST(sequence=SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    net = Net().to(DEV)
    opt = (torch.optim.SGD if opt_kind == "sgd" else torch.optim.Adam)(net.parameters(), lr=lr)
    buf = Reservoir(buffer)

    def fwd(x):
        x = x.view(x.size(0), -1)
        return net.l2(F.relu(net.l1(F.relu(net.l0(x)))))

    hist = []
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if method == "naive":
                    loss = masked_ce(fwd(x), y)
                else:
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    rep = buf.sample_any(64)
                    if rep is not None:
                        Xs.append(rep[0].to(DEV)); Ys.append(rep[1].to(DEV))
                    loss = CE(fwd(torch.cat(Xs)), torch.cat(Ys))
                opt.zero_grad(); loss.backward(); opt.step()
                if method == "er":
                    buf.add(x, y)
        hist.append([_acc(fwd, loaders[i][1]) for i in range(5)])
    return float(np.mean(hist[-1]))


@torch.no_grad()
def _acc(fwd, loader):
    c = t = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        c += (fwd(x).argmax(1) == y).sum().item(); t += len(y)
    return c / t


@torch.no_grad()
def eval_all(net, gate, center, loaders):
    """Returns (oracle, per_image, nearest, task_infer_acc). per_image/nearest are None for onehot."""
    net.eval()
    G = gate.raw_table(5)
    o = tot = 0
    p = nn_ = inf = 0
    have_img = gate.kind in ("lin", "mlp")
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            o += (forward_ps(net, G[i].unsqueeze(0).expand(b, -1), x).argmax(1) == y).sum().item()
            tot += b
            if have_img:
                d = x.view(b, -1) - center
                p += (forward_ps(net, gate.project(d), x).argmax(1) == y).sum().item()
                nnid = (d[:, None, :] - gate.mus[None]).pow(2).sum(-1).argmin(1)
                nn_ += (forward_ps(net, G[nnid], x).argmax(1) == y).sum().item()
                inf += (nnid == i).sum().item()
    if have_img:
        return o / tot, p / tot, nn_ / tot, inf / tot
    return o / tot, None, None, None


def overlap_cos(gate):
    """Mean off-diagonal cosine of the 5 per-task gate DEV vectors (raw), per hidden layer.
    Measured on dev(=raw), not g(=1+raw): the shared parity 1 inflates cos(g) for gentle gates."""
    raw = gate.raw_table(5).detach().cpu()

    def offcos(M):
        Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = Mn @ Mn.t()
        return float(np.mean([C[i, j].item() for i in range(5) for j in range(5) if i < j]))
    return offcos(raw[:, :400]), offcos(raw[:, 400:])


ARMS = {"buf-own": train_bufown, "er-own": train_erown}


def main():
    print(f"device={DEV}\n")
    print("no-gate baselines:")
    for m in ("naive", "er"):
        for opt in ("sgd", "adam"):
            print(f"  {m:5s} {opt:4s}: {train_baseline(m, opt):.4f}", flush=True)

    print("\ndriver grid (gain-neuron; acc oracle / per-image / nearest (infer); cos h0/h1 on dev):")
    hdr = f"  {'driver':6s} {'cen':3s} {'arm':7s} {'opt':4s} | {'oracle':>7s} {'per-img':>7s} {'near':>7s} (infer) | cos h0/h1"
    print(hdr)
    for kind in ("onehot", "lin", "mlp"):
        for center in (False, True):
            if kind == "onehot" and center:
                continue                                    # onehot ignores the image
            for arm, trainer in ARMS.items():
                for opt in ("sgd", "adam"):
                    loaders, net, gate, cvec = build(kind, center)
                    trainer(net, gate, loaders, opt)
                    o, p, n, inf = eval_all(net, gate, cvec, loaders)
                    c0, c1 = overlap_cos(gate)
                    pf = f"{p:.4f}" if p is not None else "  -   "
                    nf = f"{n:.4f} ({inf:.3f})" if n is not None else "  -   "
                    print(f"  {kind:6s} {'yes' if center else 'no ':3s} {arm:7s} {opt:4s} | "
                          f"{o:7.4f} {pf:>7s} {nf:>13s} | {c0:+.3f}/{c1:+.3f}", flush=True)


if __name__ == "__main__":
    main()
