"""pt6 — content & inference-net MECHANISMS of the task_id driver + an EVAL-RESOLUTION axis.
Self-contained; class-IL Split MNIST; gain target on (h0,h1,out); gain-NEURON (810 gains).
Synapse granularity is a documented follow-on, and NOT blocked for every mechanism (see the .md /
CLAUDE.md rationale): the 374M-param content projection is `mean_image`/lin only, and the per-sample-Gamma
expansion blocks TRAINING only for `embedding`. Per-synapse `soft_mlp` was runnable and simply not run.

MECHANISMS (task_id -> per-task/per-sample gate):
  onehot     : raw_t = P[t]                          (reference; oracle only)
  mean_image : raw_t = proj(mu_t), proj in {lin,mlp}; +/- centering. Resolution axis at eval.
  soft_mlp   : gate table P (onehot-style) + inference net g(x):784->128->T trained WITH REPLAY;
               eval blends gate = sum_t softmax(g(x))_t * P[t].
  embedding  : gate = proj(e(x)), e(x)=hidden(128) of the SAME inference net g; per-image, oracle-free.

EVAL RESOLUTION (mean_image; one training -> all modes):
  oracle | per-image | nearest | soft-nearest(tau)  (soft_mlp uses softmax(g); embedding is per-image)

ARMS: buf-own (naive main + per-task replay META-loss on the gate) and er-own (main+gate joint on the
ER batch, own-task gating); inference nets (soft_mlp/embedding) trained on task-CE over the reservoir.
Baselines naive/er (no gate). opt {sgd,adam}, seed42 lr1e-3 ep5 buffer1000, 1 seed. ORACLE caveat.
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
SEQ = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
CE = nn.CrossEntropyLoss()
GATEDIM = 810                                  # 400 (h0) + 400 (h1) + 10 (out)
TAUS = (0.03, 0.1, 0.3, 1.0)                   # soft-nearest temperatures (over mean squared distance)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def masked_ce(logits, y):
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
        p = idx[torch.randint(0, len(idx), (b,))]
        return self.X[p], self.Y[p]

    def sample_any(self, b):
        if self.filled == 0:
            return None
        idx = torch.randint(0, self.filled, (b,))
        return self.X[idx], self.Y[idx]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(784, 400); self.l1 = nn.Linear(400, 400); self.l2 = nn.Linear(400, 10)


def forward_gain(net, raw, x):                 # raw:(B,810) per-sample gains on h0,h1,out
    x = x.view(x.size(0), -1)
    z0 = F.relu(net.l0(x)) * (1 + raw[:, :400])
    z1 = F.relu(net.l1(z0)) * (1 + raw[:, 400:800])
    return net.l2(z1) * (1 + raw[:, 800:810])


# ------------------------------- mechanisms -------------------------------
class OneHot(nn.Module):
    modes = ("oracle",)
    has_inference = False

    def __init__(self, mus):
        super().__init__(); self.register_buffer("mus", mus)
        self.P = nn.Parameter(torch.zeros(5, GATEDIM))

    def raw_table(self):
        return self.P

    def train_gate(self, x, tids):
        return self.P[tids]

    def gate_params(self):
        return [self.P]

    def inf_params(self):
        return []


class MeanImage(nn.Module):
    modes = ("oracle", "per-image", "nearest", "soft-nearest")
    has_inference = False

    def __init__(self, mus, proj):
        super().__init__(); self.register_buffer("mus", mus); self.proj = proj   # mus: centered iff study asked
        if proj == "lin":
            self.W = nn.Parameter(torch.zeros(784, GATEDIM))
        else:
            self.gh = nn.Linear(784, 128); self.gf = nn.Linear(128, GATEDIM)
            nn.init.normal_(self.gf.weight, std=1e-3); nn.init.zeros_(self.gf.bias)

    def _project(self, d):
        return d @ self.W if self.proj == "lin" else self.gf(F.relu(self.gh(d)))

    def raw_table(self):
        return self._project(self.mus)

    def train_gate(self, x, tids):
        return self.raw_table()[tids]

    def gate_params(self):
        return list(self.parameters())

    def inf_params(self):
        return []


class SoftMLP(nn.Module):
    modes = ("oracle", "soft-mlp")
    has_inference = True

    def __init__(self, mus):
        super().__init__(); self.register_buffer("mus", mus)
        self.P = nn.Parameter(torch.zeros(5, GATEDIM))
        self.gh = nn.Linear(784, 128); self.go = nn.Linear(128, 5)     # task-inference net

    def raw_table(self):
        return self.P

    def task_logits(self, x):
        return self.go(F.relu(self.gh(x.view(x.size(0), -1))))

    def train_gate(self, x, tids):
        return self.P[tids]                                            # oracle task at train

    def blend(self, x):
        return F.softmax(self.task_logits(x), dim=1) @ self.P          # eval: sum_t p_t P[t]

    def gate_params(self):
        return [self.P]

    def inf_params(self):
        return list(self.gh.parameters()) + list(self.go.parameters())


class Embedding(nn.Module):
    modes = ("per-image",)
    has_inference = True

    def __init__(self, mus, proj):
        super().__init__(); self.register_buffer("mus", mus); self.proj = proj
        self.gh = nn.Linear(784, 128); self.go = nn.Linear(128, 5)     # inference net (trained w/ replay)
        if proj == "lin":
            self.W = nn.Parameter(torch.zeros(128, GATEDIM))
        else:
            self.pf1 = nn.Linear(128, 128); self.pf2 = nn.Linear(128, GATEDIM)
            nn.init.normal_(self.pf2.weight, std=1e-3); nn.init.zeros_(self.pf2.bias)

    def embed(self, x):
        return F.relu(self.gh(x.view(x.size(0), -1)))

    def task_logits(self, x):
        return self.go(self.embed(x))

    def gate_per_sample(self, x):
        e = self.embed(x)
        return e @ self.W if self.proj == "lin" else self.pf2(F.relu(self.pf1(e)))

    def train_gate(self, x, tids):
        return self.gate_per_sample(x)                                 # per-image, no oracle

    def gate_params(self):                                            # projection only (embedding via inf)
        return [self.W] if self.proj == "lin" else list(self.pf1.parameters()) + list(self.pf2.parameters())

    def inf_params(self):
        return list(self.gh.parameters()) + list(self.go.parameters())


def make_mech(kind, mus, proj):
    if kind == "onehot":
        return OneHot(mus)
    if kind == "mean_image":
        return MeanImage(mus, proj)
    if kind == "soft_mlp":
        return SoftMLP(mus)
    if kind == "embedding":
        return Embedding(mus, proj)
    raise ValueError(kind)


# ------------------------------- build / train -------------------------------
def build(kind, proj, center, seed=42):
    seed_all(seed)
    ds = SplitMNIST(sequence=SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    raw_mus = torch.stack([torch.cat([xb.view(xb.size(0), -1) for xb, _ in loaders[t][0]]).mean(0)
                           for t in range(5)]).to(DEV)
    cvec = raw_mus.mean(0) if center else torch.zeros(784, device=DEV)
    return loaders, Net().to(DEV), make_mech(kind, raw_mus - cvec, proj).to(DEV), cvec


def _opt(kind, params, lr):
    return (torch.optim.SGD if kind == "sgd" else torch.optim.Adam)(params, lr=lr)


def train_bufown(mech, net, loaders, opt_kind, lr=1e-3, epochs=5, buffer=1000):
    main_opt = _opt(opt_kind, net.parameters(), lr)
    gate_opt = torch.optim.Adam(mech.gate_params(), lr=lr) if mech.gate_params() else None
    inf_opt = torch.optim.Adam(mech.inf_params(), lr=lr) if mech.has_inference else None
    buf = Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV); tids = torch.full((x.size(0),), t, device=DEV)
                # main step: naive on current task, gate detached
                loss = masked_ce(forward_gain(net, mech.train_gate(x, tids).detach(), x), y)
                net.zero_grad(); mech.zero_grad(); loss.backward(); main_opt.step()
                buf.add(x, y)
                # inference-net step: task-CE over current + replay (learn task inference, no forget)
                if inf_opt is not None:
                    Xi, Ti = [x.view(x.size(0), -1)], [tids]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xi.append(r[0].to(DEV)); Ti.append(torch.div(r[1].to(DEV), 2, rounding_mode="floor"))
                    ce = CE(mech.task_logits(torch.cat(Xi)), torch.cat(Ti))
                    inf_opt.zero_grad(); ce.backward(); inf_opt.step()
                # meta step: per-task replay meta-loss trains the gate/projection
                if gate_opt is not None:
                    Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [tids]
                    for j in range(t):
                        s = buf.sample_task(j, 64)
                        if s is not None:
                            Xs.append(s[0].to(DEV)); Ys.append(s[1].to(DEV))
                            Ts.append(torch.full((s[0].size(0),), j, device=DEV))
                    Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                    meta = masked_ce(forward_gain(net, mech.train_gate(Xm, Tm), Xm), Ym)
                    net.zero_grad(); mech.zero_grad(); meta.backward(); gate_opt.step()


def train_erown(mech, net, loaders, opt_kind, lr=1e-3, epochs=5, buffer=1000):
    opt = _opt(opt_kind, list(net.parameters()) + list(mech.gate_params()), lr)
    inf_opt = torch.optim.Adam(mech.inf_params(), lr=lr) if mech.has_inference else None
    buf = Reservoir(buffer)
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
                loss = CE(forward_gain(net, mech.train_gate(Xm, Tm), Xm), Ym)        # plain CE, own-task gate
                opt.zero_grad(); inf_opt and inf_opt.zero_grad(); loss.backward(); opt.step()
                if inf_opt is not None:
                    ce = CE(mech.task_logits(Xm), Tm)
                    inf_opt.zero_grad(); ce.backward(); inf_opt.step()
                buf.add(x, y)


def train_baseline(method, opt_kind, lr=1e-3, epochs=5, buffer=1000, seed=42):
    seed_all(seed)
    ds = SplitMNIST(sequence=SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    net = Net().to(DEV); opt = _opt(opt_kind, net.parameters(), lr); buf = Reservoir(buffer)

    def fwd(x):
        x = x.view(x.size(0), -1)
        return net.l2(F.relu(net.l1(F.relu(net.l0(x)))))
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if method == "naive":
                    loss = masked_ce(fwd(x), y)
                else:
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    loss = CE(fwd(torch.cat(Xs)), torch.cat(Ys))
                opt.zero_grad(); loss.backward(); opt.step()
                if method == "er":
                    buf.add(x, y)
        pass
    return float(np.mean([_acc(fwd, loaders[i][1]) for i in range(5)]))


@torch.no_grad()
def _acc(fwd, loader):
    c = t = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        c += (fwd(x).argmax(1) == y).sum().item(); t += len(y)
    return c / t


# ------------------------------- eval (resolution modes) -------------------------------
@torch.no_grad()
def eval_modes(mech, net, center, loaders):
    net.eval()
    out = {}
    G = mech.raw_table() if hasattr(mech, "raw_table") else None       # (5,810)
    for mode in mech.modes:
        if mode == "soft-nearest":                                     # sweep tau in one data pass
            cor = {tau: 0 for tau in TAUS}; tot = 0
            for i in range(5):
                for x, y in loaders[i][1]:
                    x, y = x.to(DEV), y.to(DEV); b = x.size(0); d = x.view(b, -1) - center
                    dist = (d[:, None, :] - mech.mus[None]).pow(2).mean(-1)     # (B,5)
                    for tau in TAUS:
                        raw = F.softmax(-dist / tau, dim=1) @ G
                        cor[tau] += (forward_gain(net, raw, x).argmax(1) == y).sum().item()
                    tot += b
            for tau in TAUS:
                out[f"soft-near@{tau}"] = cor[tau] / tot
            continue
        c = tot = inf = 0
        for i in range(5):
            for x, y in loaders[i][1]:
                x, y = x.to(DEV), y.to(DEV); b = x.size(0); d = x.view(b, -1) - center
                if mode == "oracle":
                    raw = G[i].unsqueeze(0).expand(b, -1)
                elif mode == "per-image":
                    raw = mech.gate_per_sample(x) if isinstance(mech, Embedding) else mech._project(d)
                elif mode == "nearest":
                    nn_ = (d[:, None, :] - mech.mus[None]).pow(2).sum(-1).argmin(1); raw = G[nn_]
                    inf += (nn_ == i).sum().item()
                elif mode == "soft-mlp":
                    raw = mech.blend(x); inf += (mech.task_logits(x).argmax(1) == i).sum().item()
                c += (forward_gain(net, raw, x).argmax(1) == y).sum().item(); tot += b
        out[mode] = c / tot
        if mode in ("nearest", "soft-mlp"):
            out[mode + "/infer"] = inf / tot
    return out


# ------------------------------- grid -------------------------------
CELLS = [  # (mechanism, proj, center) — proj/center ignored where N/A
    ("onehot", "-", False),
    ("mean_image", "lin", True), ("mean_image", "lin", False),
    ("mean_image", "mlp", True),
    ("soft_mlp", "-", False),
    ("embedding", "lin", False), ("embedding", "mlp", False),
]
ARMS = {"buf-own": train_bufown, "er-own": train_erown}


def main():
    print(f"device={DEV}  (gain-NEURON on h0,h1,out; 1 seed)\n")
    print("baselines (no gate):")
    for m in ("naive", "er"):
        for opt in ("sgd", "adam"):
            print(f"  {m:5s} {opt:4s}: {train_baseline(m, opt):.4f}", flush=True)
    print("\nmechanism grid  (acc per resolution mode):")
    for kind, proj, center in CELLS:
        for arm, trainer in ARMS.items():
            for opt in ("sgd", "adam"):
                loaders, net, mech, cvec = build(kind, proj, center)
                trainer(mech, net, loaders, opt)
                res = eval_modes(mech, net, cvec, loaders)
                tag = f"{kind}/{proj}{'/cen' if center else ''}"
                body = "  ".join(f"{k}={v:.4f}" for k, v in res.items())
                print(f"  {tag:22s} {arm:7s} {opt:4s} | {body}", flush=True)


if __name__ == "__main__":
    main()
