"""pt7 SIGNAL-NET + GRU drivers (user-requested follow-up to pt7_neuromodulators.py).

Four user-requested mechanisms, all class-IL Split MNIST, gain (h0,h1,out), er-own, ADAM, seed42
lr1e-3 ep5 buffer1000 (the pt7 study default — Adam-ER is operating-point-insensitive, ~0.895 untuned
≈ 0.8975 tuned, so the study default is comparable to the tuned point for these Adam cells).
Reuses the pt7 primitives (Net, NeuronGate, SynapseGate, Heads, Signals, Reservoir). Ledger
pt7_signalnet_results.tsv (`--resume` skips done rows). Baselines (pt7): er-adam 0.8946, naive-adam 0.390,
all4 neuron er-own adam 0.8816.

PART `reset` (task 1) — all4, er-own, adam, standardised, 3 seeds, with NEUROMODULATOR-NET RESET.
  "Reset the learning of the neuromodulator net at every task switch": at the start of each new task
  (t>0) the neuromodulator net (the heads m_k(x) AND the gate P) is restored to its start-of-training
  weights + fresh optimizer state; the MAIN net is NOT reset (it keeps learning across tasks) and the
  replay buffer persists (so task-2 training still sees task-1 replay samples under the freshly-reset
  neuromod net — the reset is triggered ONLY by task boundaries, not by replay). At INFERENCE the last
  (end-of-task-5) neuromod net is used — no reset. reset=OFF reproduces pt7 all4 er-own adam 0.8816.
  (Splitting gate.P into its own Adam at the same lr is identical to keeping it in main_opt — Adam is
  per-parameter — so reset=OFF is a faithful reproduction, verified by the sanity cell.)

PART `gru-all4` (task 2) — a GRU on the fixed all4 signal vector. The predicted all4 vector (Heads(x),
  trained WITH REPLAY to regress the 4 standardized bio tau) is fed as input to a stateful GRUCell whose
  output (K=4) drives the rank-K gate. Hidden persists across batches (detached each step = truncated
  BPTT len 1), mirrors pt7_stateful. 1 seed, adam, er-own.

PART `signalnet` (task 3) — a SIGNAL NET: an MLP with 3 hidden layers that ingests a rich 23-dim signal
  vector and outputs a low-D code (K in {4,16}) that is UPPROJECTED by the rank-K gate P to the gain
  vector (neuron) / matrices (synapse). Trained end-to-end by the main CE (like the `free` control, but
  with structured inputs). The 23 inputs (per-sample where marked `[ps]`, else running-scalar broadcast;
  `pred`=a component head regressing the quantity, trained WITH REPLAY, oracle-free at eval):
    entropy:  1) H of last prediction (actual, scalar)   2-4) 3 EMAs of entropy (slow,mid,fast)
              5-7) 3 stds of entropy                      8) [ps] pred of entropy of curr sample
    novelty:  9) [ps] ||h1-mean_h1|| (mean=EMA, h1 from a partial extra forward)
             10) [ps] ||x -mean_x||
    loss:    11) [ps] pred of loss (head)                12-14) 3 EMAs of preds of loss
             15-17) 3 preds of EMAs of loss [ps]          18-20) 3 stds of preds of loss
             21-23) 3 preds of stds of loss [ps]
  Component heads (one shared MLP 784->32->8) regress [H, L, ema_s/m/f(L), std_s/m/f(L)] by MSE+replay;
  their outputs supply the 8 predicted features (cols 8,11,15-17,21-23). Running scalars are frozen at
  eval (mirrors pt7_stateful frozen mode); the per-sample heads/norms carry the eval-time variation.
  Inputs tried BOTH standardised (running mean/var over the 23-dim vector) and not. 1 seed, adam, er-own;
  {neuron, synapse} x K{4,16}.

PART `signalnet-gru` (task 4) — the signal net's low-D output is fed to a stateful GRUCell (as gru-all4)
  whose output drives the gate. 1 seed, adam, er-own; neuron x K{4,16} x std{on,off}.
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_neuromodulators as p7                                  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from data import SplitMNIST, get_standard_loaders                 # noqa: E402

DEV, EPS = p7.DEV, p7.EPS
H0, H1, OUT, GATEDIM = p7.H0, p7.H1, p7.OUT, p7.GATEDIM
CE = nn.CrossEntropyLoss()
BETAS = (0.01, 0.05, 0.2)                                         # 3 EMA timescales: slow, mid, fast
TSV = Path(__file__).resolve().parent / "pt7_signalnet_results.tsv"


def _mk(gran, K):
    return p7.make_gate(gran, K, None)


# =============================== PART reset (task 1) ===============================
def train_erown_reset(net, gate, heads, sig, opt_kind, reset, loaders,
                      lr=1e-3, epochs=5, buffer=1000):
    """all4 er-own with optional neuromodulator-net reset at each task switch.
    Neuromod net = heads (regress tau) + gate P. main_opt owns net only (never reset); neuromod_opt owns
    gate.P; head_opt owns heads. On a task switch (t>0, reset=True) restore heads+gate.P to their init
    snapshot and rebuild their optimizers (fresh Adam moments). Buffer persists across the reset."""
    K = gate.P.size(0)
    main_opt = p7._opt(opt_kind, net.parameters(), lr)
    init_gate = copy.deepcopy(gate.state_dict())
    init_heads = copy.deepcopy(heads.state_dict())
    neuromod_opt = torch.optim.Adam(gate.params(), lr)
    head_opt = torch.optim.Adam(heads.parameters(), lr)
    buf = p7.Reservoir(buffer)
    for t in range(5):
        if reset and t > 0:
            gate.load_state_dict(init_gate); heads.load_state_dict(init_heads)
            neuromod_opt = torch.optim.Adam(gate.params(), lr)
            head_opt = torch.optim.Adam(heads.parameters(), lr)
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                m = heads(Xm).detach()
                logits = gate(net, m, Xm)                       # gate P + net trained by CE
                loss = CE(logits, Ym)
                main_opt.zero_grad(); neuromod_opt.zero_grad()
                loss.backward()
                main_opt.step(); neuromod_opt.step()
                T = sig.targets(net, Xm, Ym)                    # bio head regression (+replay via Xm)
                hloss = F.mse_loss(heads(Xm), T)
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)


def run_reset(reset, opt_kind, seed):
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    drivers = ["DA", "ACh", "NE", "5HT"]
    net = p7.Net().to(DEV); gate = p7.NeuronGate(4, None).to(DEV)
    heads = p7.Heads(4).to(DEV); sig = p7.Signals(drivers, standardize=True)
    train_erown_reset(net, gate, heads, sig, opt_kind, reset, loaders)
    return p7.eval_cell("all4", "neuron", net, gate, heads, sig, False, loaders)


# =============================== PART gru-all4 (task 2) ===============================
class GRUOnVec(nn.Module):
    """Stateful GRU whose input is a (B,Kin) per-sample vector; output (B,Kout) drives the gate.
    Single hidden state persists across batches (updated from the batch mean, detached = BPTT len 1),
    broadcast to per-sample and concatenated with the per-sample input (mirrors pt7_stateful)."""
    def __init__(self, kin, kout, hid=64, engage=False):
        super().__init__()
        self.cell = nn.GRUCell(kin, hid)
        self.out = nn.Linear(kin + hid, kout)
        if not engage:                                           # zero-init -> m=0 -> gate parity (dead saddle
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # with a zero-init gate P)
        self.hid = hid; self.hidden = torch.zeros(1, hid, device=DEV)

    def forward(self, a, update_state=True):
        h_new = self.cell(a.mean(0, keepdim=True), self.hidden)
        if update_state:
            self.hidden = h_new.detach()
        return self.out(torch.cat([a, h_new.expand(a.size(0), -1)], 1))

    def reset_hidden(self):
        self.hidden = torch.zeros(1, self.hid, device=DEV)


def run_gru_all4(gran, opt_kind, seed=42, lr=1e-3, epochs=5, buffer=1000, engage=False):
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    drivers = ["DA", "ACh", "NE", "5HT"]; K = 4
    net = p7.Net().to(DEV); gate = _mk(gran, K)
    heads = p7.Heads(K).to(DEV); sig = p7.Signals(drivers, standardize=True)
    gru = GRUOnVec(K, K, engage=engage).to(DEV)
    main_opt = p7._opt(opt_kind, list(net.parameters()) + gate.params() + list(gru.parameters()), lr)
    head_opt = torch.optim.Adam(heads.parameters(), lr)
    buf = p7.Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                a = heads(Xm).detach()                          # predicted all4 vector (B,4)
                m = gru(a)                                      # GRU -> (B,4) gate signal
                loss = CE(gate(net, m, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                T = sig.targets(net, Xm, Ym)
                hloss = F.mse_loss(heads(Xm), T)
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)
    return _eval_gru(net, gate, heads, gru, gran, loaders)


@torch.no_grad()
def _eval_gru(net, gate, heads, gru, gran, loaders):
    net.eval(); c = tot = 0; mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            a = heads(x.view(b, -1))
            m = gru(a)
            c += (gate(net, m, x).argmax(1) == y).sum().item()
            pl = gate.per_layer_mag(m) if gran == "neuron" else gate.per_layer_mag(m, net)
            for k in mags:
                mags[k] += pl[k] * b
            tot += b
    return dict(pred=c / tot, true=float("nan"), probe=float("nan"),
                per_layer={k: mags[k] / tot for k in mags})


# =============================== PARTS signalnet / signalnet-gru (tasks 3,4) ===============================
NFEAT = 23
NPRED = 8                                                        # component-head outputs: H,L,emaL x3,stdL x3


class SignalHeads(nn.Module):
    """One MLP 784->32->8 regressing [H, L, ema_s/m/f(L), std_s/m/f(L)] (per-sample; the EMA/std cols are
    broadcast scalar targets). Supplies the 8 PREDICTED features; trained by MSE WITH REPLAY."""
    def __init__(self):
        super().__init__(); self.f1 = nn.Linear(784, 32); self.f2 = nn.Linear(32, NPRED)
        nn.init.zeros_(self.f2.weight); nn.init.zeros_(self.f2.bias)

    def forward(self, x):
        return self.f2(F.relu(self.f1(x.view(x.size(0), -1))))


class SignalFeatures:
    """Builds the (B,23) signal vector from running state + the component heads. Running scalars update at
    train (update=True) and freeze at eval (update=False). Optional standardization of the 23-dim vector."""
    def __init__(self, standardize):
        self.standardize = standardize
        self.h_last = None                                       # actual entropy of last batch (scalar)
        self.eH = list(BETAS); self.vH = list(BETAS)             # EMA mean/var of actual entropy (placeholder)
        self.eLp = list(BETAS); self.vLp = list(BETAS)           # EMA mean/var of mean predicted loss
        self.eL = list(BETAS); self.vL = list(BETAS)             # EMA mean/var of actual loss (head targets)
        self.mh1 = None; self.mx = None
        self.inited = False
        self.run_mean = torch.zeros(NFEAT, device=DEV); self.run_var = torch.ones(NFEAT, device=DEV)
        self.stats_inited = False

    def _boot(self, Hm, Lm, Lpm, h1, x2):
        self.h_last = Hm
        self.eH = [Hm] * 3; self.vH = [0.0] * 3
        self.eLp = [Lpm] * 3; self.vLp = [0.0] * 3
        self.eL = [Lm] * 3; self.vL = [0.0] * 3
        self.mh1 = h1.mean(0).clone(); self.mx = x2.mean(0).clone()
        self.inited = True

    @staticmethod
    def _ema(lst, vlst, val):
        for i, b in enumerate(BETAS):
            vlst[i] = (1 - b) * vlst[i] + b * (val - lst[i]) ** 2
            lst[i] = (1 - b) * lst[i] + b * val

    def targets(self, net, x, y):
        """Regression targets for SignalHeads: [H_i, L_i, ema_s/m/f(L), std_s/m/f(L)] (B,8)."""
        with torch.no_grad():
            logits, _ = net.plain(x)
            H = p7.entropy(logits); L = p7.per_sample_masked_ce(logits, y)
            emaL = torch.tensor(self.eL, device=DEV) if self.inited else L.mean().repeat(3)
            stdL = torch.sqrt(torch.tensor(self.vL, device=DEV).clamp_min(0)) if self.inited \
                else torch.zeros(3, device=DEV)
            B = x.size(0)
            cols = [H.unsqueeze(1), L.unsqueeze(1),
                    emaL.unsqueeze(0).expand(B, -1), stdL.unsqueeze(0).expand(B, -1)]
            return torch.cat(cols, 1)

    def build(self, net, heads, x, y=None, update=True):
        """Return (B,23) feature matrix. At train (update=True) y is used for actual-loss running state
        (never enters the gate autograd path — it is detached running scalars); at eval update=False."""
        x2 = x.view(x.size(0), -1); B = x2.size(0)
        pred = heads(x2)                                          # (B,8): H,L,emaL x3,stdL x3  (per-sample)
        with torch.no_grad():
            _, h1 = net.plain(x2)
            if update:
                logits2, _ = net.plain(x2)
                Hact = p7.entropy(logits2)
                Lact = p7.per_sample_masked_ce(logits2, y) if y is not None else pred[:, 1]
                Hm, Lm = Hact.mean().item(), Lact.mean().item()
                Lpm = pred[:, 1].mean().item()
                if not self.inited:
                    self._boot(Hm, Lm, Lpm, h1, x2)
                self._ema(self.eH, self.vH, Hm)
                self._ema(self.eLp, self.vLp, Lpm)
                self._ema(self.eL, self.vL, Lm)
                self.mh1 += p7.BS * (h1.mean(0) - self.mh1)
                self.mx += p7.BS * (x2.mean(0) - self.mx)
                self.h_last = Hm
            elif not self.inited:                                # eval before any train step (smoke only)
                self._boot(pred[:, 0].mean().item(), pred[:, 1].mean().item(),
                           pred[:, 1].mean().item(), h1, x2)
            nrm_h1 = (h1 - self.mh1).norm(dim=1, keepdim=True)
            nrm_x = (x2 - self.mx).norm(dim=1, keepdim=True)
        # scalar running features (broadcast); detached constants w.r.t. the gate graph
        def sc(v):
            return torch.full((B, 1), float(v), device=DEV)
        scalars = torch.cat(
            [sc(self.h_last)]
            + [sc(self.eH[i]) for i in range(3)]
            + [sc(np.sqrt(max(self.vH[i], 0.0))) for i in range(3)]
            + [sc(self.eLp[i]) for i in range(3)]
            + [sc(np.sqrt(max(self.vLp[i], 0.0))) for i in range(3)], 1)   # (B,13)
        # predicted per-sample features from the heads
        pred_H = pred[:, 0:1]; pred_L = pred[:, 1:2]
        pred_emaL = pred[:, 2:5]; pred_stdL = pred[:, 5:8]
        feats = torch.cat([
            scalars[:, 0:7],           # H_last, 3 EMA H, 3 std H
            pred_H,                    # 8) pred entropy [ps]
            nrm_h1, nrm_x,             # 9,10) novelty [ps]
            pred_L,                    # 11) pred loss [ps]
            scalars[:, 7:10],          # 12-14) EMAs of preds of loss
            pred_emaL,                 # 15-17) preds of EMAs of loss [ps]
            scalars[:, 10:13],         # 18-20) stds of preds of loss
            pred_stdL,                 # 21-23) preds of stds of loss [ps]
        ], 1)                          # (B,23)
        if not self.standardize:
            return feats
        with torch.no_grad():
            bm = feats.mean(0); bv = feats.var(0, unbiased=False)
            if not self.stats_inited:
                self.run_mean = bm.clone(); self.run_var = bv.clone(); self.stats_inited = True
            elif update:
                self.run_mean = 0.99 * self.run_mean + 0.01 * bm
                self.run_var = 0.99 * self.run_var + 0.01 * bv
        return (feats - self.run_mean) / (self.run_var.sqrt() + EPS)


class SignalNet(nn.Module):
    """MLP with 3 hidden layers: 23 -> 64 -> 64 -> 64 -> K. Output = low-D code upprojected by the gate P."""
    def __init__(self, K, hid=64, engage=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NFEAT, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, K))
        if not engage:                                           # zero-init output -> m=0 -> gate parity
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        return self.net(f)


def run_signalnet(gran, K, standardize, use_gru, opt_kind, seed=42, lr=1e-3, epochs=5, buffer=1000,
                  engage=False):
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV); gate = _mk(gran, K)
    heads = SignalHeads().to(DEV); feat = SignalFeatures(standardize)
    snet = SignalNet(K, engage=engage).to(DEV)
    gru = GRUOnVec(K, K, engage=engage).to(DEV) if use_gru else None
    params = list(net.parameters()) + gate.params() + list(snet.parameters()) \
        + (list(gru.parameters()) if gru else [])
    main_opt = p7._opt(opt_kind, params, lr)
    head_opt = torch.optim.Adam(heads.parameters(), lr)
    buf = p7.Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                f = feat.build(net, heads, Xm, Ym, update=True).detach()
                code = snet(f)                                   # (B,K)
                m = gru(code) if gru else code
                loss = CE(gate(net, m, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                T = feat.targets(net, Xm, Ym)                    # component-head regression (+replay)
                hloss = F.mse_loss(heads(Xm), T)
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)
    return _eval_signalnet(net, gate, heads, feat, snet, gru, gran, loaders)


@torch.no_grad()
def _eval_signalnet(net, gate, heads, feat, snet, gru, gran, loaders):
    net.eval(); c = tot = 0; mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            f = feat.build(net, heads, x, y=None, update=False)
            code = snet(f)
            m = gru(code) if gru else code
            c += (gate(net, m, x).argmax(1) == y).sum().item()
            pl = gate.per_layer_mag(m) if gran == "neuron" else gate.per_layer_mag(m, net)
            for k in mags:
                mags[k] += pl[k] * b
            tot += b
    return dict(pred=c / tot, true=float("nan"), probe=float("nan"),
                per_layer={k: mags[k] / tot for k in mags})


# =============================== PART h1gate (user-requested) ===============================
class H1Gate(nn.Module):
    """Sibling net with the main net's architecture up to h1 (784->400->400), same input x, output squashed
    to [0,1] by sigmoid; the 400-d output gates the main net's h1 by element-wise multiply. Trained jointly
    with the main net by the ER loss (no separate target). NOTE sigmoid(0)=0.5 => not parity at init (halves
    h1 from step 1) — fine under Adam, which absorbs a uniform rescale (CLAUDE.md bounded01 gotcha)."""
    def __init__(self):
        super().__init__(); self.g0 = nn.Linear(784, H0); self.g1 = nn.Linear(H0, H1)

    def forward(self, x):
        return torch.sigmoid(self.g1(F.relu(self.g0(x.view(x.size(0), -1)))))


def _h1gate_logits(net, gate, x):
    x2 = x.view(x.size(0), -1)
    h0 = F.relu(net.l0(x2)); h1 = F.relu(net.l1(h0))
    return net.l2(h1 * gate(x2)), gate(x2)


def run_h1gate(opt_kind, seed=42, lr=1e-3, epochs=5, buffer=1000):
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV); gate = H1Gate().to(DEV)
    opt = p7._opt(opt_kind, list(net.parameters()) + list(gate.parameters()), lr)
    buf = p7.Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                logits, _ = _h1gate_logits(net, gate, Xm)
                loss = CE(logits, Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                buf.add(x, y)
    net.eval(); c = tot = 0; gsum = 0.0
    with torch.no_grad():
        for i in range(5):
            for x, y in loaders[i][1]:
                x, y = x.to(DEV), y.to(DEV); b = x.size(0)
                logits, g = _h1gate_logits(net, gate, x)
                c += (logits.argmax(1) == y).sum().item(); gsum += g.mean().item() * b; tot += b
    return dict(pred=c / tot, true=float("nan"), probe=float("nan"),
                per_layer={"h0": 0.0, "h1": gsum / tot, "out": 0.0})   # h1 = mean gate value (1.0 = parity)


# =============================== PART h1gate-std (h1-gate in the STANDARD regime) ===============================
@torch.no_grad()
def _std_h1_acc(net, gate, loader):
    net.eval(); c = tot = 0; gsum = 0.0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV); b = x.size(0)
        logits, g = _h1gate_logits(net, gate, x)
        c += (logits.argmax(1) == y).sum().item(); gsum += g.mean().item() * b; tot += b
    return c / tot, gsum / tot


def run_h1gate_standard(with_gate, opt_kind="adam", seed=42, lr=1e-3, epochs=5):
    """Full-MNIST single-task 10-way CE. with_gate=False = vanilla MLP baseline (goal #2 reference)."""
    p7.seed_all(seed)
    tr, _, te = get_standard_loaders(batch_size=64)
    net = p7.Net().to(DEV)
    gate = H1Gate().to(DEV) if with_gate else None
    params = list(net.parameters()) + (list(gate.parameters()) if gate else [])
    opt = p7._opt(opt_kind, params, lr)
    for _ in range(epochs):
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            logits = _h1gate_logits(net, gate, x)[0] if gate else net.plain(x)[0]
            loss = CE(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    if gate is None:
        net.eval()
        with torch.no_grad():
            c = tot = 0
            for x, y in te:
                x, y = x.to(DEV), y.to(DEV)
                c += (net.plain(x)[0].argmax(1) == y).sum().item(); tot += len(y)
        return dict(pred=c / tot, true=float("nan"), probe=float("nan"),
                    per_layer={"h0": 0.0, "h1": 1.0, "out": 0.0})
    acc, gmean = _std_h1_acc(net, gate, te)
    return dict(pred=acc, true=float("nan"), probe=float("nan"),
                per_layer={"h0": 0.0, "h1": gmean, "out": 0.0})


# =============================== PART all4fixed (all4 with a FIXED RANDOM projection) ===============================
def _freeze_random_proj(gate, scale, dist, seed):
    """Overwrite the gate's learned P with a FIXED RANDOM tensor and freeze it (not in any optimizer)."""
    g = torch.Generator().manual_seed(7000 + seed)
    for P in gate.params():
        if dist == "gaussian":
            r = torch.randn(P.shape, generator=g) * scale
        else:                                                    # rademacher +-scale
            r = (torch.randint(0, 2, P.shape, generator=g).float() * 2 - 1) * scale
        P.data = r.to(DEV); P.requires_grad_(False)


def run_all4_fixedproj(gran, scale, dist, seed, opt_kind="adam", lr=1e-3, epochs=5, buffer=1000):
    """pt7 all4 er-own, but the rank-K projection P is FIXED RANDOM (not learned). Heads still regress the
    standardized bio tau (with replay); the main net adapts to the random-direction modulation. Only the
    per-sample coefficients m_k(x) (heads) and the backbone learn — P never gets a gradient."""
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    drivers = ["DA", "ACh", "NE", "5HT"]; K = 4
    net = p7.Net().to(DEV); gate = p7.make_gate(gran, K, None)
    _freeze_random_proj(gate, scale, dist, seed)
    heads = p7.Heads(K).to(DEV); sig = p7.Signals(drivers, standardize=True)
    main_opt = p7._opt(opt_kind, net.parameters(), lr)           # P frozen -> NOT in any optimizer
    head_opt = torch.optim.Adam(heads.parameters(), lr)
    buf = p7.Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                m = heads(Xm).detach()
                loss = CE(gate(net, m, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                T = sig.targets(net, Xm, Ym)
                hloss = F.mse_loss(heads(Xm), T)
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)
    return p7.eval_cell("all4", gran, net, gate, heads, sig, False, loaders)


# =============================== grid / ledger ===============================
def load_done():
    if not TSV.exists():
        return set()
    return {ln.split("\t", 1)[0] for ln in TSV.read_text().splitlines() if ln.strip()}


def record(tag, res):
    pl = res["per_layer"]
    row = (f"{tag}\t{res['pred']:.4f}\t{res['true']:.4f}\t{res['probe']:.3f}"
           f"\t{pl['h0']:.4f}\t{pl['h1']:.4f}\t{pl['out']:.4f}")
    with open(TSV, "a") as f:
        f.write(row + "\n")


def fmt(res):
    pl = res["per_layer"]
    return (f"pred={res['pred']:.4f}  |g|(h0/h1/out)={pl['h0']:.3f}/{pl['h1']:.3f}/{pl['out']:.3f}")


def build_cells(part):
    cells = []                                                   # (kind, tag, kwargs)
    if part in ("all", "reset"):
        cells.append(("reset", "all4|reset0|neuron|er-own|adam|seed42", dict(reset=False, seed=42)))  # sanity
        for s in (42, 43, 44):
            cells.append(("reset", f"all4|reset1|neuron|er-own|adam|seed{s}", dict(reset=True, seed=s)))
    if part in ("all", "gru-all4"):
        for gran in ("neuron", "synapse"):
            cells.append(("gru-all4", f"gru-all4|{gran}|er-own|adam", dict(gran=gran)))
    if part in ("all", "signalnet"):
        for gran in ("neuron", "synapse"):
            for K in (4, 16):
                stds = (True, False) if gran == "neuron" else (True,)
                for std in stds:
                    cells.append(("signalnet",
                                  f"signalnet|{gran}|K{K}|std{int(std)}|er-own|adam",
                                  dict(gran=gran, K=K, standardize=std, use_gru=False)))
    if part in ("all", "signalnet-gru"):
        for K in (4, 16):
            for std in (True, False):
                cells.append(("signalnet-gru",
                              f"signalnet-gru|neuron|K{K}|std{int(std)}|er-own|adam",
                              dict(gran="neuron", K=K, standardize=std, use_gru=True)))
    if part in ("all", "engage"):
        # symmetry-broken re-run: module OUTPUT layer gets normal init (gate P stays zero-init -> parity at
        # step 0, but P can now bootstrap because m != 0). Tests whether the mechanisms help once engaged.
        cells.append(("gru-all4", "gru-all4|neuron|er-own|adam|eng", dict(gran="neuron", engage=True)))
        for K in (4, 16):
            cells.append(("signalnet", f"signalnet|neuron|K{K}|std1|er-own|adam|eng",
                          dict(gran="neuron", K=K, standardize=True, use_gru=False, engage=True)))
            cells.append(("signalnet-gru", f"signalnet-gru|neuron|K{K}|std1|er-own|adam|eng",
                          dict(gran="neuron", K=K, standardize=True, use_gru=True, engage=True)))
    if part in ("all", "h1gate"):
        cells.append(("h1gate", "h1gate|sibling784-400-400|h1-sigmoid-mult|er-own|adam", dict()))
    if part in ("all", "h1gate-std"):
        cells.append(("h1gate-std", "h1gate-std|vanilla|adam", dict(with_gate=False)))
        cells.append(("h1gate-std", "h1gate-std|h1-sigmoid-mult|adam", dict(with_gate=True)))
    if part in ("all", "all4fixed"):
        # all4 with a FIXED RANDOM projection, 3 ways x 2 seeds. Ways: neuron(scale 0.1 / 0.3), synapse(0.1).
        ways = [("neuron", 0.1, "gaussian"), ("neuron", 0.3, "gaussian"), ("synapse", 0.1, "gaussian")]
        for gran, scale, dist in ways:
            for s in (42, 43):
                cells.append(("all4fixed",
                              f"all4fixed|{gran}|{dist}|scale{scale}|er-own|adam|seed{s}",
                              dict(gran=gran, scale=scale, dist=dist, seed=s)))
    return cells


def run_one(kind, kwargs):
    if kind == "reset":
        return run_reset(kwargs["reset"], "adam", kwargs["seed"])
    if kind == "gru-all4":
        return run_gru_all4(kwargs["gran"], "adam", engage=kwargs.get("engage", False))
    if kind == "h1gate":
        return run_h1gate("adam")
    if kind == "h1gate-std":
        return run_h1gate_standard(kwargs["with_gate"], "adam")
    if kind == "all4fixed":
        return run_all4_fixedproj(kwargs["gran"], kwargs["scale"], kwargs["dist"], kwargs["seed"], "adam")
    return run_signalnet(kwargs["gran"], kwargs["K"], kwargs["standardize"],
                         kwargs["use_gru"], "adam", engage=kwargs.get("engage", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "reset", "gru-all4", "signalnet", "signalnet-gru", "engage",
                             "h1gate", "h1gate-std", "all4fixed", "smoke"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={DEV}  (pt7 signalnet; er-adam 0.8946 naive-adam 0.390 all4-er-own-adam 0.8816)\n",
          flush=True)

    if args.part == "smoke":
        for kind, tag, kw in ([("reset", "reset", dict(reset=True, seed=42))]
                              + [("gru-all4", "gru", dict(gran="neuron"))]
                              + [("signalnet", "sn-n", dict(gran="neuron", K=4, standardize=True, use_gru=False))]
                              + [("signalnet", "sn-s", dict(gran="synapse", K=4, standardize=True, use_gru=False))]
                              + [("signalnet-gru", "sng", dict(gran="neuron", K=4, standardize=False, use_gru=True))]):
            r = _smoke(kind, kw)
            print(f"  smoke {tag:6s}: pred={r['pred']:.3f}  |g|out={r['per_layer']['out']:.3f}", flush=True)
        return

    done = load_done() if args.resume else set()
    for kind, tag, kwargs in build_cells(args.part):
        if tag in done:
            continue
        r = run_one(kind, kwargs)
        print(f"  {tag:48s} | {fmt(r)}", flush=True)
        record(tag, r)
    print("ALL SELECTED CELLS DONE", flush=True)


def _smoke(kind, kw):
    """1-epoch smoke of each path."""
    kw = dict(kw)
    if kind == "reset":
        p7.seed_all(kw["seed"])
        ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
        net = p7.Net().to(DEV); gate = p7.NeuronGate(4, None).to(DEV)
        heads = p7.Heads(4).to(DEV); sig = p7.Signals(["DA", "ACh", "NE", "5HT"], standardize=True)
        train_erown_reset(net, gate, heads, sig, "adam", kw["reset"], loaders, epochs=1)
        return p7.eval_cell("all4", "neuron", net, gate, heads, sig, False, loaders)
    if kind == "gru-all4":
        return run_gru_all4(kw["gran"], "adam", epochs=1)
    return run_signalnet(kw["gran"], kw["K"], kw["standardize"], kw["use_gru"], "adam", epochs=1)


if __name__ == "__main__":
    main()
