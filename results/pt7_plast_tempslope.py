"""pt7 follow-up (user-requested): the neuromodulator drivers on a PLASTICITY target, and two new GATE
FORMS (softmax-temperature, ReLU-slope). class-IL Split MNIST, er-own, seed42 lr1e-3 ep5 buffer1000, 1 seed.
Self-contained (pt7 style); reuses pt7_neuromodulators / pt7_variants / pt7_stateful primitives.

SET 1 — PLASTICITY target (SGD main net, NON-standardised drivers, er-own):
  drivers: da_fast, ach_ema, ach_gru, 5ht_ema  (all K=1, head-predicted, trained WITH REPLAY)
  mechanisms:
    neuron  : per-neuron LR gate alpha=exp(mbar@P), P:(1,810); gates each layer's incoming weight+bias grads
    synapse : per-synapse LR gate per layer alpha_l=exp(mbar@P_l); gates WEIGHT grads only (biases plastic)
    global  : a single scalar alpha=exp(mbar*p) (scalar->scalar projection) scaling ALL lrs equally
  All three train P via the pt5 LOOKAHEAD meta-loss (the gate is on the gradient, so P gets no grad from the
  main loss under in-place SGD gating — pt5 caveat): W_fast = W.detach() - lr*(alpha@g) [alpha keeps grad, g
  detached], meta-CE on the SAME ER batch (replay in the batch => retention meta-loss) trains ONLY P via Adam,
  then the real gated SGD step commits with the detached alpha. Eval is the PLAIN (unmodulated) net — the gate
  only shaped LEARNING, so pred = plain acc; report per-layer |alpha-1| + the task-decodability probe.

SET 2 — GATE FORMS (Adam main net, NON-standardised drivers, er-own):
  drivers: ne, ne_emb_all, ne_vecproj, nerisez, nerisez_gru
  forms (both positive scalar gains, exp(m@p), parity at init, P trained JOINTLY by the ER loss — in forward):
    temp  : logits_out *= exp(m@p_out)  — softmax TEMPERATURE (out only). A uniform positive scale is
            ARGMAX-INVARIANT at eval, so temp's only effect is via TRAINING dynamics (per-sample loss/confidence
            reweighting by novelty); reported as the key interpretive point.
    slope : h0 *= exp(m@p_h0); h1 *= exp(m@p_h1) — per-hidden-layer ReLU-slope gain (hidden only). Affects eval.

Baselines (this harness, matched opt): naive/er sgd & adam. Ledger pt7_plast_tempslope_results.tsv (--resume).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_neuromodulators as p7                                   # noqa: E402
from pt7_stateful import StatefulDriver                           # noqa: E402
from pt7_variants import NEDriver                                 # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from data import SplitMNIST                                       # noqa: E402

DEV, EPS, BS = p7.DEV, p7.EPS, p7.BS
H0, H1, OUT, GATEDIM = p7.H0, p7.H1, p7.OUT, p7.GATEDIM
CE = nn.CrossEntropyLoss()
TSV = Path(__file__).resolve().parent / "pt7_plast_tempslope_results.tsv"

SET1_DRIVERS = ["da_fast", "ach_ema", "ach_gru", "5ht_ema"]        # SGD, plasticity
SET1_MECHS = ["neuron", "synapse", "global"]
TUNED_SGD_LR = 0.03                                                # pt7_tuned_syn val-tuned ER-SGD (ep5) -> 0.9034
DEFAULT_NEURO_LR = 1e-3                                            # pt7_tuned_neuro default (neuro net un-swept here)
SET2_DRIVERS = ["ne", "ne_emb_all", "ne_vecproj", "nerisez", "nerisez_gru"]   # Adam, gate forms
SET2_FORMS = ["temp", "slope"]


# ------------------------------- stateful driver with a standardize toggle -------------------------------
class StatefulStd(StatefulDriver):
    """StatefulDriver + a `standardize` flag. For `ach` (predicted entropy) standardize=False returns the raw
    prediction. `nerisez` keeps its intrinsic z-score (that IS the mechanism, not the standardize toggle)."""
    def __init__(self, mech, gru, standardize):
        super().__init__(mech, gru)
        self.standardize = standardize

    def driver(self, x, update_state, update_stats):
        Hpred = self.predictH(x, update_state)
        if self.mech == "ach":
            return self._standardize(Hpred, update_stats) if self.standardize else Hpred
        if self.emaH is None:                                     # nerisez bootstrap (batch z-score)
            return F.relu(Hpred - Hpred.mean()) / (Hpred.std() + EPS)
        if update_stats:
            self.upd_actual(Hpred.detach())
        return F.relu((Hpred - self.emaH) / math.sqrt(self.varH + EPS))


# ------------------------------- unified driver provider -------------------------------
class Driver:
    """m(x) provider + its own (replay-trained) head optimizer. All drivers here are oracle-free."""
    HEAD_KEY = {"da_fast": "DA_fast", "ach_ema": "ACh_ema", "5ht_ema": "5HT_ema", "ne": "NE"}

    def __init__(self, name, standardize, lr):
        self.name = name
        if name in self.HEAD_KEY:                                 # Signals head (784->32->1 regresses tau)
            self.kind = "head"; self.K = 1
            self.heads = p7.Heads(1).to(DEV)
            self.sig = p7.Signals([self.HEAD_KEY[name]], standardize=standardize)
            self.opt = torch.optim.Adam(self.heads.parameters(), lr)
        elif name in ("ne_emb_all", "ne_vecproj"):                # headless NE novelty (no head to train)
            self.kind = "ne"
            self.drv = NEDriver({"ne_emb_all": "emb_all", "ne_vecproj": "vecproj"}[name], standardize)
            self.K = self.drv.K()
        elif name in ("nerisez", "nerisez_gru", "ach_gru"):       # stateful entropy predictor
            self.kind = "stateful"; self.K = 1
            self.mech = "ach" if name == "ach_gru" else "nerisez"
            self.drv = StatefulStd(self.mech, gru=(name != "nerisez"), standardize=standardize).to(DEV)
            self.opt = torch.optim.Adam(self.drv.parameters(), lr)
        else:
            raise ValueError(name)

    def value(self, net, X, update=True):
        """(B,K) driver, DETACHED (the gate/plasticity path never grads back into the driver)."""
        if self.kind == "head":
            return self.heads(X).detach()
        if self.kind == "ne":
            return self.drv.value(net, X, update=update).detach()
        us = update and (self.mech == "ach")                      # stats: predicted (ach) vs actual (nerisez)
        return self.drv.driver(X, update_state=update, update_stats=us).detach()

    def train_head(self, net, X, Y):
        if self.kind == "ne":
            return                                                # deterministic, nothing to train
        if self.kind == "head":
            T = self.sig.targets(net, X, Y)
            hloss = F.mse_loss(self.heads(X), T)
        else:
            with torch.no_grad():
                Hact = p7.entropy(net.plain(X)[0]).unsqueeze(1)
            if self.mech == "nerisez":
                self.drv.upd_actual(Hact)
            hloss = F.mse_loss(self.drv.predictH(X, update_state=False), Hact)
        self.opt.zero_grad(); hloss.backward(); self.opt.step()


# ------------------------------- SET 1: plasticity gate + lookahead -------------------------------
NPARAMS = 6                                                       # l0.w,l0.b,l1.w,l1.b,l2.w,l2.b


class PlastGate(nn.Module):
    """Per-neuron / per-synapse / global learning-rate gate alpha = exp(raw), raw = mbar@P (P zero-init =>
    alpha=1 parity). exp keeps alpha>0 and can amplify OR suppress the LR (unlike sigmoid, which only cuts)."""
    def __init__(self, mech, K, lr):
        super().__init__()
        self.mech = mech
        if mech == "neuron":
            self.P = nn.Parameter(torch.zeros(K, GATEDIM))
        elif mech == "synapse":
            self.P0 = nn.Parameter(torch.zeros(K, H0 * 784))
            self.P1 = nn.Parameter(torch.zeros(K, H1 * H0))
            self.P2 = nn.Parameter(torch.zeros(K, OUT * H1))
        else:                                                     # global: scalar->scalar
            self.P = nn.Parameter(torch.zeros(K))
        self.to(DEV)
        self.opt = torch.optim.Adam(self.parameters(), lr)

    def mult(self, mbar):
        """mbar:(K,) detached -> (per-parameter multiplier dict {0..5}, per-layer alpha tuple for reporting)."""
        one = lambda n: torch.ones(n, device=DEV)                 # noqa: E731  (plastic bias -> alpha 1)
        if self.mech == "neuron":
            a = torch.exp(mbar @ self.P)                          # (810,)
            ah0, ah1, ao = a[:H0], a[H0:H0 + H1], a[H0 + H1:]
            return ({0: ah0[:, None], 1: ah0, 2: ah1[:, None], 3: ah1, 4: ao[:, None], 5: ao},
                    (ah0, ah1, ao))
        if self.mech == "synapse":
            a0 = torch.exp(mbar @ self.P0).view(H0, 784)
            a1 = torch.exp(mbar @ self.P1).view(H1, H0)
            a2 = torch.exp(mbar @ self.P2).view(OUT, H1)
            return ({0: a0, 1: one(H0), 2: a1, 3: one(H1), 4: a2, 5: one(OUT)}, (a0, a1, a2))
        a = torch.exp(mbar @ self.P)                              # scalar
        return ({i: a for i in range(NPARAMS)}, (a,))


def _net_params(net):
    return [net.l0.weight, net.l0.bias, net.l1.weight, net.l1.bias, net.l2.weight, net.l2.bias]


def _fwd_fast(Wf, x):
    x = x.view(x.size(0), -1)
    h0 = F.relu(x @ Wf[0].t() + Wf[1])
    h1 = F.relu(h0 @ Wf[2].t() + Wf[3])
    return h1 @ Wf[4].t() + Wf[5]


def run_plast(driver_name, mech, main_lr=1e-3, neuro_lr=None, epochs=5, buffer=1000, seed=42):
    """main_lr = the SGD lr of the MAIN net (commit + lookahead W_fast). neuro_lr = the Adam lr of the
    neuromod net (gate P meta-opt + driver head); None => main_lr (the untuned run, bit-exact). The tuned
    variant decouples them: main_lr=0.03 (pt7_tuned_syn ER-SGD operating point), neuro_lr=1e-3 (default)."""
    neuro_lr = main_lr if neuro_lr is None else neuro_lr
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV)
    drv = Driver(driver_name, standardize=False, lr=neuro_lr)
    gate = PlastGate(mech, drv.K, neuro_lr)
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
                mbar = drv.value(net, Xm).mean(0)                 # (K,) detached
                params = _net_params(net)
                g = torch.autograd.grad(CE(net.plain(Xm)[0], Ym), params)   # detached grads
                mult, _ = gate.mult(mbar)                         # alpha differentiable in P
                Wf = [params[i].detach() - main_lr * (mult[i] * g[i]) for i in range(NPARAMS)]
                meta = CE(_fwd_fast(Wf, Xm), Ym)                  # retention meta-loss (replay in Xm) trains P
                gate.opt.zero_grad(); meta.backward(); gate.opt.step()
                with torch.no_grad():                             # real gated SGD step (detached alpha, same g)
                    for i in range(NPARAMS):
                        params[i].add_(mult[i].detach() * g[i], alpha=-main_lr)
                buf.add(x, y)
                drv.train_head(net, Xm, Ym)
    return eval_plast(net, drv, gate, loaders)


@torch.no_grad()
def eval_plast(net, drv, gate, loaders):
    net.eval()
    pred = float(np.mean([p7._acc_plain(net, loaders[i][1]) for i in range(5)]))
    mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}; tot = 0; Ms, Ts = [], []
    for i in range(5):
        for x, y in loaders[i][1]:
            x = x.to(DEV); b = x.size(0)
            m = drv.value(net, x, update=False)
            _, structs = gate.mult(m.mean(0))
            if gate.mech == "global":
                dev = (structs[0] - 1).abs().item()
                for k in mags:
                    mags[k] += dev * b
            else:
                a0, a1, a2 = structs
                mags["h0"] += (a0 - 1).abs().mean().item() * b
                mags["h1"] += (a1 - 1).abs().mean().item() * b
                mags["out"] += (a2 - 1).abs().mean().item() * b
            Ms.append(m.cpu()); Ts.append(torch.full((b,), i)); tot += b
    return dict(pred=pred, true=float("nan"), probe=p7._probe(torch.cat(Ms), torch.cat(Ts), drv.K),
                per_layer={k: mags[k] / tot for k in mags})


# ------------------------------- SET 2: temperature / slope gate forms -------------------------------
class GateForm(nn.Module):
    """Positive scalar gain forms, exp(m@p), p zero-init => parity. temp gates the OUT logits (softmax
    temperature); slope gates the hidden ReLUs (h0,h1). P trained jointly by the main ER loss (in forward)."""
    def __init__(self, form, K):
        super().__init__()
        self.form = form
        if form == "temp":
            self.p_out = nn.Parameter(torch.zeros(K))
        else:
            self.p_h0 = nn.Parameter(torch.zeros(K)); self.p_h1 = nn.Parameter(torch.zeros(K))
        self.to(DEV)

    def forward(self, net, m, x):
        x = x.view(x.size(0), -1)
        if self.form == "temp":
            h0 = F.relu(net.l0(x)); h1 = F.relu(net.l1(h0))
            return net.l2(h1) * torch.exp(m @ self.p_out).unsqueeze(1)
        h0 = F.relu(net.l0(x)) * torch.exp(m @ self.p_h0).unsqueeze(1)
        h1 = F.relu(net.l1(h0)) * torch.exp(m @ self.p_h1).unsqueeze(1)
        return net.l2(h1)

    @torch.no_grad()
    def mag(self, m):
        if self.form == "temp":
            return {"h0": 0.0, "h1": 0.0, "out": (torch.exp(m @ self.p_out) - 1).abs().mean().item()}
        return {"h0": (torch.exp(m @ self.p_h0) - 1).abs().mean().item(),
                "h1": (torch.exp(m @ self.p_h1) - 1).abs().mean().item(), "out": 0.0}


def run_gateform(driver_name, form, lr=1e-3, epochs=5, buffer=1000, seed=42):
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV)
    drv = Driver(driver_name, standardize=False, lr=lr)
    gate = GateForm(form, drv.K)
    opt = torch.optim.Adam(list(net.parameters()) + list(gate.parameters()), lr)   # joint (in-forward)
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
                m = drv.value(net, Xm)                            # detached (head trained separately)
                loss = CE(gate(net, m, Xm), Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                buf.add(x, y)
                drv.train_head(net, Xm, Ym)
    return eval_gateform(net, drv, gate, loaders)


@torch.no_grad()
def eval_gateform(net, drv, gate, loaders):
    net.eval(); c = tot = 0; mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}; Ms, Ts = [], []
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            m = drv.value(net, x, update=False)
            c += (gate(net, m, x).argmax(1) == y).sum().item()
            pl = gate.mag(m)
            for k in mags:
                mags[k] += pl[k] * b
            Ms.append(m.cpu()); Ts.append(torch.full((b,), i)); tot += b
    return dict(pred=c / tot, true=float("nan"), probe=p7._probe(torch.cat(Ms), torch.cat(Ts), drv.K),
                per_layer={k: mags[k] / tot for k in mags})


# ------------------------------- ledger + grid -------------------------------
def load_done():
    if not TSV.exists():
        return set()
    return {ln.split("\t", 1)[0] for ln in TSV.read_text().splitlines() if ln.strip()}


def record(tag, res):
    if res.get("base") is not None:
        row = f"{tag}\t{res['base']:.4f}"
    else:
        pl = res["per_layer"]
        row = (f"{tag}\t{res['pred']:.4f}\t{res['true']:.4f}\t{res['probe']:.3f}"
               f"\t{pl['h0']:.4f}\t{pl['h1']:.4f}\t{pl['out']:.4f}")
    with open(TSV, "a") as f:
        f.write(row + "\n")


def fmt(res):
    pl = res["per_layer"]
    return (f"pred={res['pred']:.4f}  probe={res['probe']:.3f}  "
            f"|a-1|(h0/h1/out)={pl['h0']:.3f}/{pl['h1']:.3f}/{pl['out']:.3f}")


def build_cells(part):
    cells = []                                                    # (group, name, sub, opt)
    if part in ("all", "baselines"):
        cells += [("base", m, "-", opt) for m in ("naive", "er") for opt in ("sgd", "adam")]
    if part in ("all", "plast"):
        cells += [("plast", d, mech, "sgd") for d in SET1_DRIVERS for mech in SET1_MECHS]
    if part == "plast-tuned":                                     # main net at tuned SGD lr=0.03 (ep5)
        cells += [("base", m, "-", "sgd-tuned") for m in ("naive", "er")]
        cells += [("plast", d, mech, "sgd-tuned") for d in SET1_DRIVERS for mech in SET1_MECHS]
    if part in ("all", "gateform"):
        cells += [("gateform", d, form, "adam") for d in SET2_DRIVERS for form in SET2_FORMS]
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "baselines", "plast", "plast-tuned", "gateform", "smoke"])
    ap.add_argument("--drivers", default=None, help="comma filter on driver name")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={DEV}  (pt7 plasticity + temp/slope gate forms; class-IL er-own; 1 seed)\n", flush=True)

    if args.part == "smoke":
        for d, mech in (("da_fast", "neuron"), ("ach_gru", "synapse"), ("5ht_ema", "global")):
            r = run_plast(d, mech, epochs=1)
            print(f"  smoke plast    {d:12s} {mech:7s}: {fmt(r)}", flush=True)
        for d, form in (("ne", "temp"), ("ne_vecproj", "slope"), ("nerisez_gru", "temp")):
            r = run_gateform(d, form, epochs=1)
            print(f"  smoke gateform {d:12s} {form:5s}: {fmt(r)}", flush=True)
        return

    dfil = set(args.drivers.split(",")) if args.drivers else None
    done = load_done() if args.resume else set()
    for group, name, sub, opt in build_cells(args.part):
        if dfil and group != "base" and name not in dfil:
            continue
        tag = f"{group}|{name}|{sub}|{opt}"
        if tag in done:
            continue
        if group == "base":
            lr = TUNED_SGD_LR if opt == "sgd-tuned" else 1e-3
            acc = p7.train_baseline(name, "sgd" if opt == "sgd-tuned" else opt, lr=lr)
            print(f"  base {name:5s} {opt:9s}: {acc:.4f}", flush=True)
            record(tag, {"base": acc})
        elif group == "plast":
            r = (run_plast(name, sub, main_lr=TUNED_SGD_LR, neuro_lr=DEFAULT_NEURO_LR)
                 if opt == "sgd-tuned" else run_plast(name, sub))
            print(f"  plast    {name:12s} {sub:7s} {opt:9s} | {fmt(r)}", flush=True)
            record(tag, r)
        else:
            r = run_gateform(name, sub)
            print(f"  gateform {name:12s} {sub:5s} adam | {fmt(r)}", flush=True)
            record(tag, r)
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
