"""Parity check: neurocore primitives reproduce the frozen pt6/pt7 numbers.

The extraction was a copy-forward — prototype/ and results/ were not edited — so the archived path
reproduces its numbers trivially. The real question is whether the EXTRACTED primitives are faithful,
which this answers by rebuilding the pt7 `er-own` / baseline cells out of neurocore parts and
comparing against the frozen values in results/pt7_results.tsv.

Bit-exactness requires the RNG stream to match, so the module-construction order here is identical to
results/pt7_neuromodulators.build (seed_all -> SplitMNIST -> loaders -> Net -> gate -> Heads ->
Signals) and the training loops are reproduced step for step. That the loops live HERE and not in
neurocore is the intended split: a problem package owns its data, backbone, baselines and loop, and
assembles them from core primitives. This file doubles as the reference example of that.

pt7 anchors (seed 42, Adam, lr 1e-3 / ep 5 / buffer 1000, gain-neuron on h0,h1,out):
    naive   0.3900     er   0.8946     er+free  0.8760     er+all4  0.8816

pt6 anchors (seed 42, lr 1e-3 / ep 5 / buffer 1000, gain-neuron), for neurocore.task_selection.
NOT all bit-exact — see PT6_ANCHORS below for why (soft_mlp gates via `P[tids]`, whose scatter-add
backward is nondeterministic on MPS, so pt6 does not reproduce its own log either). `~` marks a
metric held to the noise floor rather than bitwise:
    soft_mlp  er-own  adam : ~oracle 0.9913  ~soft 0.8850   infer 0.8843
    soft_mlp  buf-own sgd  : ~oracle 0.9393  ~soft 0.8562   infer 0.8648
    embedding er-own  sgd  :  per-image 0.8888

Run: uv run python neurocore/verify_anchors.py [--part pt7|pt6|all]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurocore.buffers import Reservoir                                    # noqa: E402
from neurocore.controls import cell_spec, probe                            # noqa: E402
from neurocore.gates import apply_neuron_gain, gate_K, make_gate           # noqa: E402
from neurocore.gates import Heads                                          # noqa: E402
from neurocore.signals import DRIVER_LAYERS, Signals, masked_ce            # noqa: E402
from neurocore.task_selection import EmbeddingSelector, SoftMLPSelector    # noqa: E402
from neurocore.utils import DEV, seed_all                                  # noqa: E402

# Imported as `prototype.data`, NOT via a sys.path insert into prototype/ the way the frozen
# results/pt*.py scripts do it. Same module, same code, but statically resolvable, so an IDE can
# follow it. Inert numerically: data.py holds no module-level RNG (its generators are local), and
# every entry point calls seed_all() after imports anyway.
from prototype.data import SplitMNIST                                      # noqa: E402

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


# ------------------------------- pt6: task selectors -------------------------------
# pt6 reduces its masked CE with F.cross_entropy's own mean, not per-sample-then-mean. Kept exactly
# so the pt6 cells reproduce bit-exact; the two are mathematically equal but need not be bitwise.
def _pt6_masked_ce(logits, y):
    task = torch.div(y, 2, rounding_mode="floor")
    allowed = torch.stack([2 * task, 2 * task + 1], dim=1)
    add = torch.full_like(logits, float("-inf")); add.scatter_(1, allowed, 0.0)
    return F.cross_entropy(logits + add, y)


# pt6 anchors carry PER-METRIC tolerances, because pt6's soft_mlp is not bit-reproducible even from
# its own unmodified code. Cause (measured, not assumed): soft_mlp gates via `P[tids]`, whose backward
# is an atomic SCATTER-ADD — nondeterministic on MPS at max|d| 3.8e-6 per step, versus exactly 0 for a
# matmul. Compounded through Adam over ~4750 steps that becomes ~3e-3 of weight drift.
#   soft_mlp er-own/adam oracle:  logged 0.9913 | pt6's own re-run 0.9885 | this extraction 0.9904
# Everything reached through matmuls IS bit-exact and is held to TOL: the inference net (`infer`,
# pure matmuls) and the whole `embedding` mechanism (`e @ W`). That split is the evidence the
# extraction is faithful — it is exact wherever exactness is achievable at all.
# Use SoftMLPSelector(deterministic=True) for a bit-reproducible variant (one_hot(tids) @ P).
PT6_NOISE = 0.007        # documented pt6 1-seed MPS floor (CLAUDE.md pt6-followups: 0.007-0.016)

PT6_ANCHORS = {
    ("soft_mlp", "er-own", "adam"):  {"oracle": (0.9913, PT6_NOISE), "soft": (0.8850, PT6_NOISE),
                                      "infer": (0.8843, TOL)},
    ("soft_mlp", "buf-own", "sgd"):  {"oracle": (0.9393, PT6_NOISE), "soft": (0.8562, PT6_NOISE),
                                      "infer": (0.8648, TOL)},
    ("embedding", "er-own", "sgd"):  {"per-image": (0.8888, TOL)},
}


def pt6_build(kind, seed=42, proj="lin"):
    """Mirror results/pt6_driver_mechanisms.build EXACTLY, including the mean-image pass.

    That pass iterates the SHUFFLED train loaders and therefore consumes RNG *before* Net() and the
    selector are constructed. soft_mlp never uses the means, but removing the computation would shift
    every subsequent draw — so it stays. This is the RNG-order coupling that makes copy-forward the
    right extraction strategy.
    """
    seed_all(seed)
    ds = SplitMNIST(sequence=SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    _mus = torch.stack([torch.cat([xb.view(xb.size(0), -1) for xb, _ in loaders[t][0]]).mean(0)
                        for t in range(5)]).to(DEV)                       # RNG-consuming, load-bearing
    net = Net().to(DEV)
    mech = (SoftMLPSelector() if kind == "soft_mlp"
            else EmbeddingSelector(proj=proj)).to(DEV)
    return loaders, net, mech


def pt6_train(mech, net, loaders, arm, opt_kind, lr=LR, epochs=EPOCHS, buffer=BUFFER):
    buf = Reservoir(buffer)
    inf_opt = torch.optim.Adam(mech.inf_params(), lr=lr)
    if arm == "er-own":
        opt = _opt(opt_kind, list(net.parameters()) + list(mech.gate_params()), lr)
    else:
        main_opt = _opt(opt_kind, net.parameters(), lr)
        gate_opt = torch.optim.Adam(mech.gate_params(), lr=lr)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                tids = torch.full((x.size(0),), t, device=DEV)
                if arm == "er-own":
                    Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [tids]
                    r = buf.sample_any(64)
                    if r is not None:
                        xr, yr = r[0].to(DEV), r[1].to(DEV)
                        Xs.append(xr); Ys.append(yr)
                        Ts.append(torch.div(yr, 2, rounding_mode="floor"))
                    Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                    loss = CE(apply_neuron_gain(net, mech.train_gate(Xm, Tm), Xm), Ym)
                    opt.zero_grad(); inf_opt.zero_grad(); loss.backward(); opt.step()
                    ce = CE(mech.task_logits(Xm), Tm)
                    inf_opt.zero_grad(); ce.backward(); inf_opt.step()
                    buf.add(x, y)
                else:                                    # buf-own: naive main, meta-loss on the gate
                    loss = _pt6_masked_ce(
                        apply_neuron_gain(net, mech.train_gate(x, tids).detach(), x), y)
                    net.zero_grad(); mech.zero_grad(); loss.backward(); main_opt.step()
                    buf.add(x, y)
                    Xi, Ti = [x.view(x.size(0), -1)], [tids]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xi.append(r[0].to(DEV))
                        Ti.append(torch.div(r[1].to(DEV), 2, rounding_mode="floor"))
                    ce = CE(mech.task_logits(torch.cat(Xi)), torch.cat(Ti))
                    inf_opt.zero_grad(); ce.backward(); inf_opt.step()
                    Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [tids]
                    for j in range(t):
                        s = buf.sample_task(j, 64)
                        if s is not None:
                            Xs.append(s[0].to(DEV)); Ys.append(s[1].to(DEV))
                            Ts.append(torch.full((s[0].size(0),), j, device=DEV))
                    Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                    meta = _pt6_masked_ce(
                        apply_neuron_gain(net, mech.train_gate(Xm, Tm), Xm), Ym)
                    net.zero_grad(); mech.zero_grad(); meta.backward(); gate_opt.step()


@torch.no_grad()
def pt6_eval(mech, net, loaders, kind):
    net.eval()
    c = {"oracle": 0, "soft": 0, "per-image": 0}; inf = tot = 0
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            if kind == "soft_mlp":
                tids = torch.full((b,), i, device=DEV)
                c["oracle"] += (apply_neuron_gain(
                    net, mech.eval_gate(x, "oracle", tids), x).argmax(1) == y).sum().item()
                c["soft"] += (apply_neuron_gain(
                    net, mech.eval_gate(x, "soft"), x).argmax(1) == y).sum().item()
                inf += (mech.task_logits(x).argmax(1) == i).sum().item()
            else:
                c["per-image"] += (apply_neuron_gain(
                    net, mech.eval_gate(x, "per-image"), x).argmax(1) == y).sum().item()
            tot += b
    out = {k: v / tot for k, v in c.items() if v}
    if kind == "soft_mlp":
        out["infer"] = inf / tot
    return out


def run_pt6():
    ok_all = True
    for (kind, arm, opt_kind), want in PT6_ANCHORS.items():
        loaders, net, mech = pt6_build(kind)
        pt6_train(mech, net, loaders, arm, opt_kind)
        got = pt6_eval(mech, net, loaders, kind)
        parts, ok_cell = [], True
        for k, (w, tol) in want.items():
            hit = abs(got[k] - w) < tol
            ok_cell &= hit
            mark = "" if tol <= TOL else "~"          # ~ = held to the noise floor, not bit-exact
            parts.append(f"{k}={got[k]:.4f}({mark}exp {w:.4f}){'' if hit else ' <-MISMATCH'}")
        ok_all &= ok_cell
        print(f"  {kind:9s} {arm:7s} {opt_kind:4s} [{'OK ' if ok_cell else 'MISMATCH'}]  "
              + "  ".join(parts), flush=True)
    return ok_all


def run_pt7():
    ok_all = True
    for arm, want in ANCHORS.items():
        r = run_arm(arm)
        ok = abs(r["acc"] - want) < TOL
        ok_all &= ok
        extra = ("" if "probe" not in r else
                 f"  probe={r['probe']:.3f}  |g|={r['h0']:.4f}/{r['h1']:.4f}/{r['out']:.4f}")
        print(f"  {arm:9s} got {r['acc']:.4f}  expected {want:.4f}  "
              f"[{'OK ' if ok else 'MISMATCH'}]{extra}", flush=True)
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "pt7", "pt6"])
    args = ap.parse_args()
    print(f"device={DEV}  neurocore parity vs frozen ledgers "
          f"(seed 42, lr={LR} ep={EPOCHS} buffer={BUFFER}, gain-{GRAN})\n", flush=True)
    ok_all = True
    if args.part in ("all", "pt7"):
        print("pt7 — drivers / rank-K gate / controls:", flush=True)
        ok_all &= run_pt7()
    if args.part in ("all", "pt6"):
        print("\npt6 — task selectors (neurocore.task_selection):", flush=True)
        ok_all &= run_pt6()
    print("\n" + ("ALL ANCHORS MATCH" if ok_all else "ANCHOR MISMATCH — extraction is not faithful"),
          flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
