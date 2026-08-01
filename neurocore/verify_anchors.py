"""Parity check: neurocore primitives reproduce the frozen pt7 ledger BIT-EXACT.

The extraction was a copy-forward — prototype/ and results/ were not edited — so the archived path
reproduces its numbers trivially. The real question is whether the EXTRACTED primitives are faithful,
which this answers by rebuilding the pt7 `er-own` / baseline cells out of neurocore parts and
comparing against the frozen values in results/pt7_results.tsv.

Bit-exactness requires the RNG stream to match, so the module-construction order here is identical to
results/pt7_neuromodulators.build (seed_all -> SplitMNIST -> loaders -> Net -> gate -> Heads ->
Signals) and the training loops are reproduced step for step. That the loops live HERE and not in
neurocore is the intended split: a problem package owns its data, backbone, baselines and loop, and
assembles them from core primitives. This file doubles as the reference example of that.

Anchors (seed 42, Adam, lr 1e-3 / ep 5 / buffer 1000, gain-neuron on h0,h1,out):
    naive   0.3900     er   0.8946     er+free  0.8760     er+all4  0.8816

Run: uv run python neurocore/verify_anchors.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))

from neurocore.buffers import Reservoir                                    # noqa: E402
from neurocore.controls import cell_spec, probe                            # noqa: E402
from neurocore.gates import gate_K, make_gate                              # noqa: E402
from neurocore.gates import Heads                                          # noqa: E402
from neurocore.signals import DRIVER_LAYERS, Signals, masked_ce            # noqa: E402
from neurocore.utils import DEV, seed_all                                  # noqa: E402

from data import SplitMNIST                                                # noqa: E402

SEQ = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
CE = nn.CrossEntropyLoss()
H0, H1, OUT = 400, 400, 10
LR, EPOCHS, BUFFER, OPT = 1e-3, 5, 1000, "adam"
GRAN = "neuron"

ANCHORS = {"naive": 0.3900, "er": 0.8946, "er+free": 0.8760, "er+all4": 0.8816}
TOL = 5e-5


# ------------------------------- problem-owned backbone -------------------------------
class Net(nn.Module):
    """The Split-MNIST MLP. Satisfies neurocore's backbone contract: plain(x) -> (logits, features)."""
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(784, H0); self.l1 = nn.Linear(H0, H1); self.l2 = nn.Linear(H1, OUT)

    def plain(self, x):
        x = x.view(x.size(0), -1)
        h0 = F.relu(self.l0(x)); h1 = F.relu(self.l1(h0))
        return self.l2(h1), h1


def _opt(kind, params, lr):
    return (torch.optim.SGD if kind == "sgd" else torch.optim.Adam)(params, lr=lr)


# ------------------------------- loops -------------------------------
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


def train_erown(loaders, net, gate, heads, sig, is_free, is_const, gran,
                opt_kind=OPT, lr=LR, epochs=EPOCHS, buffer=BUFFER):
    K = gate_K(gate, gran)
    main_opt = _opt(opt_kind, list(net.parameters()) + gate.params()
                    + (list(heads.parameters()) if is_free else []), lr)
    head_opt = torch.optim.Adam(heads.parameters(), lr) if (heads is not None and not is_free) else None
    buf = Reservoir(buffer)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                m = torch.ones(Xm.size(0), K, device=DEV) if is_const else heads(Xm)
                m_gate = m if is_free else m.detach()
                loss = CE(gate(net, m_gate, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                if head_opt is not None:                      # biological head regression (+replay via Xm)
                    T = sig.targets(net, Xm, Ym)
                    hloss = F.mse_loss(heads(Xm), T)
                    head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)


def train_baseline(method, opt_kind=OPT, lr=LR, epochs=EPOCHS, buffer=BUFFER, seed=42):
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


@torch.no_grad()
def eval_cell(net, gate, heads, sig, is_const, loaders, gran=GRAN):
    net.eval(); K = gate_K(gate, gran)
    c_pred = tot = 0
    mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    Ms, Ts = [], []
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            m = torch.ones(b, K, device=DEV) if is_const else heads(x)
            c_pred += (gate(net, m, x).argmax(1) == y).sum().item()
            pl = gate.per_layer_mag(m) if gran == "neuron" else gate.per_layer_mag(m, net)
            for k in mags:
                mags[k] += pl[k] * b
            Ms.append(m.cpu()); Ts.append(torch.full((b,), i))
            tot += b
    return {"pred": c_pred / tot,
            "per_layer": {k: v / tot for k, v in mags.items()},
            "probe": probe(torch.cat(Ms), torch.cat(Ts), K)}


def run_arm(arm):
    if arm in ("naive", "er"):
        return {"acc": train_baseline(arm)}
    name = {"er+free": "free", "er+all4": "all4"}[arm]
    loaders, net, gate, heads, sig, is_free, is_const = build(name, GRAN)
    train_erown(loaders, net, gate, heads, sig, is_free, is_const, GRAN)
    r = eval_cell(net, gate, heads, sig, is_const, loaders)
    pl = r["per_layer"]
    return {"acc": r["pred"], "probe": r["probe"], "h0": pl["h0"], "h1": pl["h1"], "out": pl["out"]}


def main():
    print(f"device={DEV}  neurocore parity vs frozen pt7 ledger "
          f"(seed 42, {OPT} lr={LR} ep={EPOCHS} buffer={BUFFER}, gain-{GRAN})\n", flush=True)
    ok_all = True
    for arm, want in ANCHORS.items():
        r = run_arm(arm)
        ok = abs(r["acc"] - want) < TOL
        ok_all &= ok
        extra = ("" if "probe" not in r else
                 f"  probe={r['probe']:.3f}  |g|={r['h0']:.4f}/{r['h1']:.4f}/{r['out']:.4f}")
        print(f"  {arm:9s} got {r['acc']:.4f}  expected {want:.4f}  "
              f"[{'OK ' if ok else 'MISMATCH'}]{extra}", flush=True)
    print("\n" + ("ALL ANCHORS MATCH" if ok_all else "ANCHOR MISMATCH — extraction is not faithful"),
          flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
