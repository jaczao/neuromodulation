"""pt7 — the four classic neuromodulators (DA, ACh, NE, 5-HT) as PRE-FORWARD gate drivers.

Self-contained; class-IL Split MNIST; gain target on (h0,h1,out); {neuron, synapse}.
See prototype/SPEC-proto-pt7.md. Every gate signal is computable BEFORE the main forward: a small head
m_k(x) regresses a biological per-sample signal tau_k (DA/ACh/NE/5HT), trained WITH REPLAY, and drives the
per-sample gate at train AND eval (oracle-free by construction). NE_emb is a within-forward signal (last
hidden novelty gates the out layer). The gate is rank-K linear: Gamma_i = 1 + sum_k m_ik P_k, so per-synapse
needs only K+1 matmuls per layer (no (B,d_out,d_in) expansion) and P is (K, n_syn).

DRIVERS (tau_k, per-sample true signal; head predicts it from x):
  DA      : (loss_i - ema_slow)/std              reward prediction error (phasic)     -> out/plasticity
  ACh     : H_i (predictive entropy)             expected uncertainty                 -> h0 (bottom-up)
  NE      : relu((|DA_i| - ach_vol)/ach_vol)     unexpected uncertainty / gain        -> out
  NE_emb  : ||h1_i - mean_h1||                   embedding novelty (within-forward)   -> out only
  5HT     : -loss_i (reward; tonic=ema_reward)   average reward / critic              -> global
  tonic ablations: DA_step, ACh_vol (scalar), NE_rise (scalar); nulls: 5ht-const, free.

ARMS: buf-own (naive main + per-task replay META-loss on P; heads regress tau + replay) and er-own (main+P
joint on the ER batch, own-task; heads regress tau + replay). CONTROLS: free (K=4 heads, NO bio target,
trained end-to-end) and 5ht-const (learned constant gate, no x-dependence = scale-degeneracy null).
EVAL: pred (heads, oracle-free = THE number), true (2-pass, uses labels = diagnostic upper bound), probe
(task-decodability of m(x)), per-layer |gate|. opt {sgd,adam}, seed42 lr1e-3 ep5 buffer1000, 1 seed.
"""
import argparse
import math
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
H0, H1, OUT = 400, 400, 10
GATEDIM = H0 + H1 + OUT                          # 810 (neuron)
BF, BS = 0.1, 0.02                               # fast / slow EMA rates
EPS = 1e-6

# driver -> the gate LAYERS it is allowed to touch (per SPEC "native" mapping). None = all three.
DRIVER_LAYERS = {"NE_emb": ("out",)}             # within-forward: last hidden novelty -> out only


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def per_sample_masked_ce(logits, y):
    task = torch.div(y, 2, rounding_mode="floor")
    allowed = torch.stack([2 * task, 2 * task + 1], dim=1)
    add = torch.full_like(logits, float("-inf")); add.scatter_(1, allowed, 0.0)
    return F.cross_entropy(logits + add, y, reduction="none")


def masked_ce(logits, y):
    return per_sample_masked_ce(logits, y).mean()


def entropy(logits):
    logp = F.log_softmax(logits, dim=1)
    return -(logp.exp() * logp).sum(1)


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
        self.l0 = nn.Linear(784, H0); self.l1 = nn.Linear(H0, H1); self.l2 = nn.Linear(H1, OUT)

    def plain(self, x):                            # unmodulated forward -> (logits, h1)
        x = x.view(x.size(0), -1)
        h0 = F.relu(self.l0(x)); h1 = F.relu(self.l1(h0))
        return self.l2(h1), h1


# ------------------------------- gate (rank-K linear) -------------------------------
class NeuronGate(nn.Module):
    """Gamma = 1 + m @ P over the 810 neuron gains; P:(K, 810)."""
    def __init__(self, K, layers):
        super().__init__(); self.P = nn.Parameter(torch.zeros(K, GATEDIM)); self.set_layers(layers)

    def set_layers(self, layers):                  # zero (and freeze grad on) disallowed columns
        m = torch.zeros(GATEDIM)
        if layers is None or "h0" in layers:  m[:H0] = 1
        if layers is None or "h1" in layers:  m[H0:H0 + H1] = 1
        if layers is None or "out" in layers: m[H0 + H1:] = 1
        self.register_buffer("lm", m)

    def raw(self, m):                              # m:(B,K) -> (B,810)
        return (m @ self.P) * self.lm

    def forward(self, net, m, x, detach_P=False):
        P = self.P.detach() if detach_P else self.P
        raw = ((m @ P) * self.lm)
        x = x.view(x.size(0), -1)
        z0 = F.relu(net.l0(x)) * (1 + raw[:, :H0])
        z1 = F.relu(net.l1(z0)) * (1 + raw[:, H0:H0 + H1])
        return net.l2(z1) * (1 + raw[:, H0 + H1:])

    def params(self):
        return [self.P]

    @torch.no_grad()
    def per_layer_mag(self, m):
        raw = (m @ self.P) * self.lm
        return {"h0": raw[:, :H0].abs().mean().item(),
                "h1": raw[:, H0:H0 + H1].abs().mean().item(),
                "out": raw[:, H0 + H1:].abs().mean().item()}


class SynapseGate(nn.Module):
    """(Gamma ⊙ W)x = Wx + sum_k m_k (P_k ⊙ W)x ; P per layer (K, d_out, d_in). Layers 0,2,4 map l0,l1,l2."""
    def __init__(self, K, layers):
        super().__init__()
        self.on = {"h0": layers is None or "h0" in layers, "h1": layers is None or "h1" in layers,
                   "out": layers is None or "out" in layers}
        self.P0 = nn.Parameter(torch.zeros(K, H0, 784)) if self.on["h0"] else None
        self.P1 = nn.Parameter(torch.zeros(K, H1, H0)) if self.on["h1"] else None
        self.P2 = nn.Parameter(torch.zeros(K, OUT, H1)) if self.on["out"] else None

    @staticmethod
    def _layer(inp, lin, P, m, detach_P):
        base = F.linear(inp, lin.weight, lin.bias)
        if P is None:
            return base
        Pw = (P.detach() if detach_P else P) * lin.weight.unsqueeze(0)      # (K,do,di)
        mod = torch.einsum("kod,bd->bko", Pw, inp)                          # (B,K,do)
        return base + (m.unsqueeze(-1) * mod).sum(1)

    def forward(self, net, m, x, detach_P=False):
        x = x.view(x.size(0), -1)
        z0 = F.relu(self._layer(x, net.l0, self.P0, m, detach_P))
        z1 = F.relu(self._layer(z0, net.l1, self.P1, m, detach_P))
        return self._layer(z1, net.l2, self.P2, m, detach_P)

    def params(self):
        return [p for p in (self.P0, self.P1, self.P2) if p is not None]

    @torch.no_grad()
    def per_layer_mag(self, m, net):               # cheap proxy: mean_k |m_k| * |P_k ⊙ W|  (no B×K×do×di tensor)
        out = {}
        mk = m.abs().mean(0)                        # (K,)
        for name, lin, P in (("h0", net.l0, self.P0), ("h1", net.l1, self.P1), ("out", net.l2, self.P2)):
            if P is None:
                out[name] = 0.0; continue
            pw = (P * lin.weight.unsqueeze(0)).abs().mean(dim=(1, 2))   # (K,)
            out[name] = float((mk * pw).sum().item())
        return out


# ------------------------------- heads + biological signals -------------------------------
class Heads(nn.Module):
    """m_k(x): 784 -> h -> K. K per-sample scalars; drives the gate (train & eval)."""
    def __init__(self, K, hid=32):
        super().__init__(); self.f1 = nn.Linear(784, hid); self.f2 = nn.Linear(hid, K)
        nn.init.zeros_(self.f2.weight); nn.init.zeros_(self.f2.bias)   # start ~0 -> gate ~parity

    def forward(self, x):
        return self.f2(F.relu(self.f1(x.view(x.size(0), -1))))


class Signals:
    """Per-sample biological targets tau_k from a plain (unmodulated) detached forward + running EMA state.

    Targets are STANDARDIZED per driver by running mean/var so the K drivers enter the linear gate at a
    comparable unit scale (else a large-magnitude driver like 5HT=-loss blows the K=4 gate up). Only the
    per-sample VARIATION matters to the gate (scale is absorbed by P), so standardizing is neutral; it also
    makes the `true`-eval a clean upper bound (same scale the head regressed). Linear ⇒ synapse-safe.
    """
    def __init__(self, drivers, standardize=True, loss_fn=None):
        self.drivers = drivers; K = len(drivers); self.standardize = standardize
        self.loss_fn = loss_fn or per_sample_masked_ce      # standard regime passes plain per-sample CE
        self.ef = self.es = self.esq = self.er = self.prev = self.emaH = None
        self.mh1 = None
        self.run_mean = torch.zeros(K, device=DEV); self.run_var = torch.ones(K, device=DEV)
        self.inited = False

    @torch.no_grad()
    def targets(self, net, x, y, update=True):
        logits, h1 = net.plain(x)
        ell = self.loss_fn(logits, y); Hs = entropy(logits)
        Lm = ell.mean().item()
        if self.ef is None:
            self.ef = self.es = Lm; self.esq = 0.0; self.er = -Lm; self.prev = Lm
            self.emaH = Hs.mean().item(); self.mh1 = h1.mean(0)
        std = ell.std() + EPS
        ach_vol = math.sqrt(max(self.esq, 0.0))
        da = (ell - self.es) / std
        cols = []
        for d in self.drivers:
            if d == "DA":        cols.append(da)
            elif d == "DA_step": cols.append((ell - self.prev) / std)
            elif d == "DA_fast": cols.append((ell - self.ef) / (abs(self.ef) + EPS))   # /ema_fast baseline
            elif d == "ACh":     cols.append(Hs)
            elif d == "ACh_ema": cols.append(torch.full_like(ell, self.emaH))          # lag-1 running entropy (scalar)
            elif d == "ACh_vol": cols.append(torch.full_like(ell, ach_vol))
            elif d == "ACh_vol_ps": cols.append((ell - self.ef).abs())                 # per-sample |loss - ema_fast|
            elif d == "NE":      cols.append(F.relu((da.abs() - ach_vol) / (ach_vol + EPS)))
            elif d == "NE_rise": cols.append(torch.full_like(ell, max(self.ef - self.es, 0.0)))
            elif d == "NE_emb":  cols.append((h1 - self.mh1).norm(dim=1))
            elif d == "5HT":     cols.append(-ell)
            elif d == "5HT_ema": cols.append(torch.full_like(ell, -self.es))           # ema_slow(-loss) (tonic scalar)
            else: raise ValueError(d)
        T = torch.stack(cols, 1)                    # (B,K) raw
        if update:
            self.ef += BF * (Lm - self.ef); self.es += BS * (Lm - self.es)
            self.esq += BS * ((Lm - self.ef) ** 2 - self.esq); self.er += BS * (-Lm - self.er)
            self.emaH += BS * (Hs.mean().item() - self.emaH)
            self.mh1 += BS * (h1.mean(0) - self.mh1); self.prev = Lm
            if self.standardize:
                bm = T.mean(0); bv = T.var(0, unbiased=False)
                if not self.inited:
                    self.run_mean = bm.clone(); self.run_var = bv.clone(); self.inited = True
                else:
                    self.run_mean = 0.99 * self.run_mean + 0.01 * bm
                    self.run_var = 0.99 * self.run_var + 0.01 * bv
        if not self.standardize:
            return T
        return (T - self.run_mean) / (self.run_var.sqrt() + EPS)   # standardized (running stats)


# ------------------------------- cells -------------------------------
# name -> (drivers, is_free, is_const)
def cell_spec(name):
    if name == "free":       return (["free"] * 4, True, False)
    if name == "5ht-const":  return (["const"], False, True)
    if name == "all4":       return (["DA", "ACh", "NE", "5HT"], False, False)
    return ([name], False, False)


def make_gate(gran, K, layers):
    return (NeuronGate(K, layers) if gran == "neuron" else SynapseGate(K, layers)).to(DEV)


def build(name, gran, seed=42, standardize=True):
    seed_all(seed)
    ds = SplitMNIST(sequence=SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    drivers, is_free, is_const = cell_spec(name)
    layers = DRIVER_LAYERS.get(name, None)
    net = Net().to(DEV)
    gate = make_gate(gran, len(drivers), layers)
    heads = None if is_const else Heads(len(drivers)).to(DEV)
    sig = None if (is_free or is_const) else Signals(drivers, standardize=standardize)
    return loaders, net, gate, heads, sig, is_free, is_const


def _opt(kind, params, lr):
    return (torch.optim.SGD if kind == "sgd" else torch.optim.Adam)(params, lr=lr)


def _m(heads, is_const, x, K):
    if is_const:
        return torch.ones(x.size(0), K, device=DEV)        # constant driver
    return heads(x)


def train_erown(name, gran, net, gate, heads, sig, is_free, is_const,
                opt_kind, lr=1e-3, epochs=5, buffer=1000):
    K = gate.P.size(0) if gran == "neuron" else next(p for p in (gate.P0, gate.P1, gate.P2)
                                                     if p is not None).size(0)
    main_opt = _opt(opt_kind, list(net.parameters()) + gate.params()
                    + (list(heads.parameters()) if is_free else []), lr)
    head_opt = torch.optim.Adam(heads.parameters(), lr) if (heads is not None and not is_free) else None
    buf = Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders_train(net_loaders, t):
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                m = _m(heads, is_const, Xm, K)
                m_gate = m if is_free else m.detach()
                logits = gate(net, m_gate, Xm)
                loss = CE(logits, Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                if head_opt is not None:                      # biological head regression (+replay via Xm)
                    T = sig.targets(net, Xm, Ym)
                    hloss = F.mse_loss(heads(Xm), T)
                    head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)


def train_nobuf(name, gran, net, gate, heads, sig, is_free, is_const,
                opt_kind, lr=1e-3, epochs=5, buffer=1000):
    """Standalone, NO buffer (pt6-followup-(B) stress test): naive masked-CE main + gate jointly on the
    CURRENT task only; heads regress tau on the current task only. No replay anywhere."""
    K = gate.P.size(0) if gran == "neuron" else next(p for p in (gate.P0, gate.P1, gate.P2)
                                                     if p is not None).size(0)
    main_opt = _opt(opt_kind, list(net.parameters()) + gate.params()
                    + (list(heads.parameters()) if is_free else []), lr)
    head_opt = torch.optim.Adam(heads.parameters(), lr) if (heads is not None and not is_free) else None
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders_train(net_loaders, t):
                x, y = x.to(DEV), y.to(DEV)
                m = _m(heads, is_const, x, K)
                m_gate = m if is_free else m.detach()
                loss = masked_ce(gate(net, m_gate, x), y)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                if head_opt is not None:
                    T = sig.targets(net, x, y)
                    hloss = F.mse_loss(heads(x), T)
                    head_opt.zero_grad(); hloss.backward(); head_opt.step()


def train_bufown(name, gran, net, gate, heads, sig, is_free, is_const,
                 opt_kind, lr=1e-3, epochs=5, buffer=1000):
    K = gate.P.size(0) if gran == "neuron" else next(p for p in (gate.P0, gate.P1, gate.P2)
                                                     if p is not None).size(0)
    main_opt = _opt(opt_kind, net.parameters(), lr)
    gate_opt = torch.optim.Adam(gate.params() + (list(heads.parameters()) if is_free else []), lr)
    head_opt = torch.optim.Adam(heads.parameters(), lr) if (heads is not None and not is_free) else None
    buf = Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders_train(net_loaders, t):
                x, y = x.to(DEV), y.to(DEV)
                # main: naive on current task under a DETACHED gate (net grad only)
                m = _m(heads, is_const, x, K).detach()
                logits = gate(net, m, x, detach_P=True)
                loss = masked_ce(logits, y)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                buf.add(x, y)
                # head: regress tau on current + replay
                if head_opt is not None:
                    rh = buf.sample_any(64)
                    Xh = torch.cat([x.view(x.size(0), -1)] + ([rh[0].to(DEV)] if rh else []))
                    Yh = torch.cat([y] + ([rh[1].to(DEV)] if rh else []))
                    T = sig.targets(net, Xh, Yh)
                    hloss = F.mse_loss(heads(Xh), T)
                    head_opt.zero_grad(); hloss.backward(); head_opt.step()
                # meta: per-task replay meta-loss trains P (heads detached; free trains heads here)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                for j in range(t):
                    s = buf.sample_task(j, 64)
                    if s is not None:
                        Xs.append(s[0].to(DEV)); Ys.append(s[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                mm = _m(heads, is_const, Xm, K)
                mm = mm if is_free else mm.detach()
                meta = masked_ce(gate(net, mm, Xm), Ym)
                gate_opt.zero_grad(); meta.backward(); gate_opt.step()


# module-level handle so the inner loops can see the current cell's loaders (set in run_cell)
net_loaders = None


def loaders_train(ldrs, t):
    return ldrs[t][0]


# ------------------------------- baselines -------------------------------
def train_baseline(method, opt_kind, lr=1e-3, epochs=5, buffer=1000, seed=42):
    seed_all(seed)
    ds = SplitMNIST(sequence=SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    net = Net().to(DEV); opt = _opt(opt_kind, net.parameters(), lr); buf = Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if method == "naive":
                    loss = masked_ce(net.plain(x)[0], y)
                else:
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    loss = CE(net.plain(torch.cat(Xs))[0], torch.cat(Ys))
                opt.zero_grad(); loss.backward(); opt.step()
                if method == "er":
                    buf.add(x, y)
    return float(np.mean([_acc_plain(net, loaders[i][1]) for i in range(5)]))


@torch.no_grad()
def _acc_plain(net, loader):
    net.eval(); c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        c += (net.plain(x)[0].argmax(1) == y).sum().item(); tot += len(y)
    return c / tot


# ------------------------------- eval -------------------------------
@torch.no_grad()
def eval_cell(name, gran, net, gate, heads, sig, is_const, loaders):
    net.eval(); K = _K(gate, gran)
    c_pred = c_true = tot = 0
    mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    Ms, Ts = [], []                                # for the task-decodability probe
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            m = torch.ones(b, K, device=DEV) if is_const else heads(x)
            c_pred += (gate(net, m, x).argmax(1) == y).sum().item()
            if sig is not None:                    # diagnostic upper bound (uses labels)
                mt = sig.targets(net, x, y, update=False)
                c_true += (gate(net, mt, x).argmax(1) == y).sum().item()
            pl = gate.per_layer_mag(m) if gran == "neuron" else gate.per_layer_mag(m, net)
            for k in mags:
                mags[k] += pl[k] * b
            Ms.append(m.cpu()); Ts.append(torch.full((b,), i))
            tot += b
    out = {"pred": c_pred / tot, "true": (c_true / tot) if sig is not None else float("nan")}
    out["per_layer"] = {k: v / tot for k, v in mags.items()}
    out["probe"] = _probe(torch.cat(Ms), torch.cat(Ts), K)
    return out


def _K(gate, gran):
    return gate.P.size(0) if gran == "neuron" else next(p for p in (gate.P0, gate.P1, gate.P2)
                                                        if p is not None).size(0)


def _probe(M, T, K):
    """Linear probe task-acc from m(x) in R^K (diagnostic: is the modulatory code task-decodable?)."""
    M = (M - M.mean(0)) / (M.std(0) + EPS)
    clf = nn.Linear(K, 5)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    with torch.enable_grad():
        for _ in range(300):
            opt.zero_grad(); F.cross_entropy(clf(M), T).backward(); opt.step()
    return (clf(M).argmax(1) == T).float().mean().item()


# ------------------------------- grid runner -------------------------------
def run_cell(name, gran, arm, opt_kind, standardize=True):
    global net_loaders
    loaders, net, gate, heads, sig, is_free, is_const = build(name, gran, standardize=standardize)
    net_loaders = loaders
    trainer = {"buf-own": train_bufown, "er-own": train_erown, "nobuf": train_nobuf}[arm]
    trainer(name, gran, net, gate, heads, sig, is_free, is_const, opt_kind)
    return eval_cell(name, gran, net, gate, heads, sig, is_const, loaders)


def fmt(res):
    pl = res["per_layer"]
    return (f"pred={res['pred']:.4f}  true={res['true']:.4f}  probe={res['probe']:.3f}  "
            f"|g|(h0/h1/out)={pl['h0']:.3f}/{pl['h1']:.3f}/{pl['out']:.3f}")


NEURON_MAIN = ["DA", "ACh", "NE", "NE_emb", "5HT", "all4", "free"]
NULLS = ["DA_step", "ACh_vol", "NE_rise", "5ht-const"]
SYN = ["DA", "all4", "free"]
RESULTS_TSV = Path(__file__).resolve().parent / "pt7_results.tsv"    # resume ledger + data source


def build_cells(part):
    """Return the ordered list of (kind, name, gran, arm, opt) tuples for a --part selection."""
    cells = []
    if part in ("all", "baselines"):
        cells += [("base", m, "-", "-", opt) for m in ("naive", "er") for opt in ("sgd", "adam")]
    if part in ("all", "neuron-main"):
        cells += [("cell", n, "neuron", arm, opt) for n in NEURON_MAIN
                  for arm in ("nobuf", "buf-own", "er-own") for opt in ("sgd", "adam")]
    if part in ("all", "nulls"):
        cells += [("cell", n, "neuron", "er-own", opt) for n in NULLS for opt in ("sgd", "adam")]
    if part in ("all", "synapse"):
        for n in SYN:
            arms = ("er-own", "buf-own") if n == "all4" else ("er-own",)
            cells += [("cell", n, "synapse", arm, opt) for arm in arms for opt in ("sgd", "adam")]
    return cells


def load_done():
    if not RESULTS_TSV.exists():
        return set()
    return {ln.split("\t", 1)[0] for ln in RESULTS_TSV.read_text().splitlines() if ln.strip()}


def record(tag, res):
    if res.get("base") is not None:
        row = f"{tag}\t{res['base']:.4f}"
    else:
        pl = res["per_layer"]
        row = (f"{tag}\t{res['pred']:.4f}\t{res['true']:.4f}\t{res['probe']:.3f}"
               f"\t{pl['h0']:.4f}\t{pl['h1']:.4f}\t{pl['out']:.4f}")
    with open(RESULTS_TSV, "a") as f:
        f.write(row + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "baselines", "neuron-main", "nulls", "synapse", "smoke"])
    ap.add_argument("--drivers", default=None, help="comma filter on mechanism name")
    ap.add_argument("--arms", default=None, help="comma filter on arm (nobuf,buf-own,er-own)")
    ap.add_argument("--opts", default=None, help="comma filter on optimizer (sgd,adam)")
    ap.add_argument("--resume", action="store_true", help="skip cells already in the TSV ledger")
    args = ap.parse_args()
    print(f"device={DEV}  (gain (h0,h1,out); rank-K linear gate; 1 seed)\n", flush=True)

    if args.part == "smoke":
        for gran in ("neuron", "synapse"):
            for name in ("DA", "all4", "free", "5ht-const", "NE_emb"):
                for arm in ("er-own", "buf-own"):
                    r = _run_short(name, gran, arm, "adam")
                    print(f"  smoke {name:9s} {gran:7s} {arm:7s}: pred={r['pred']:.3f}", flush=True)
        return

    dfil = set(args.drivers.split(",")) if args.drivers else None
    afil = set(args.arms.split(",")) if args.arms else None
    ofil = set(args.opts.split(",")) if args.opts else None
    done = load_done() if args.resume else set()

    for kind, name, gran, arm, opt in build_cells(args.part):
        if kind == "cell":
            if dfil and name not in dfil:  continue
            if afil and arm not in afil:   continue
        if ofil and opt not in ofil:       continue
        tag = f"{name}|{gran}|{arm}|{opt}"
        if tag in done:
            continue
        if kind == "base":
            acc = train_baseline(name, opt)
            print(f"  base {name:5s} {opt:4s}: {acc:.4f}", flush=True)
            record(tag, {"base": acc})
        else:
            r = run_cell(name, gran, arm, opt)
            print(f"  {name:9s} {gran:7s} {arm:7s} {opt:4s} | {fmt(r)}", flush=True)
            record(tag, r)
    print("ALL SELECTED CELLS DONE", flush=True)


def _run_short(name, gran, arm, opt_kind):
    """1-epoch smoke of a single cell (all code paths, tiny)."""
    global net_loaders
    loaders, net, gate, heads, sig, is_free, is_const = build(name, gran)
    net_loaders = loaders
    trainer = train_bufown if arm == "buf-own" else train_erown
    trainer(name, gran, net, gate, heads, sig, is_free, is_const, opt_kind, epochs=1)
    return eval_cell(name, gran, net, gate, heads, sig, is_const, loaders)


if __name__ == "__main__":
    main()
