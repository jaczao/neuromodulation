"""pt7 follow-up (user-requested): does neuromodulation (all4, Adam) help CONVERGENCE / EFFICIENCY —
i.e. reach a given accuracy in fewer epochs — on class-IL Split MNIST AND standard full-MNIST? The pt7
headline already established all4 ties the baseline at the FINAL plateau (er-adam / vanilla); this asks the
orthogonal question: even when the endpoint ties, is the LEARNING CURVE faster/steeper?

all4 = the canonical pt7 gate: K=4 heads m_k(x) regress the four bio signals (DA/ACh/NE/5HT, STANDARDIZED),
trained WITH REPLAY, driving a gain-NEURON gate Gamma=1+sum_k m_k P_k over (h0,h1,out). Adam, seed42,
lr1e-3, buffer1000. Self-contained (reuses pt7_neuromodulators / pt7_variants primitives). 1 seed.

Parts:
  standard  : full MNIST single-task, 10-way CE. vanilla vs all4. Per-EPOCH test-acc learning curve (E=15).
              Efficiency = acc@early-epochs + epochs-to-threshold (0.97, 0.98).
  cl-sweep  : class-IL, er-own. EPOCHS-PER-TASK budget sweep {1,2,3,5,8}. ER vs ER+all4 final avg acc — does
              the gate reach the plateau at a smaller budget (sample efficiency)?
  cl-traj   : class-IL, er-own, ep=5. Per-EPOCH avg class-IL acc trajectory (25 checkpoints) — a finer view
              of WHEN accuracy is gained through the task sequence. ER vs ER+all4.

Ledger results/pt7_convergence_results.tsv (--resume). Log pt7_convergence.log. Baselines cross-check the pt7
harness (er-adam ~0.895, vanilla ~0.98).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_neuromodulators as p7                                   # noqa: E402
from pt7_variants import per_sample_ce_plain, _std_acc            # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from data import SplitMNIST, get_standard_loaders                # noqa: E402

DEV = p7.DEV
CE = nn.CrossEntropyLoss()
ALL4 = ["DA", "ACh", "NE", "5HT"]
TSV = Path(__file__).resolve().parent / "pt7_convergence_results.tsv"


def _all4_parts(seed_built_net):
    """Build the canonical all4 gate/head/signal trio (gain-neuron, standardized, K=4)."""
    gate = p7.NeuronGate(4, None).to(DEV)
    heads = p7.Heads(4).to(DEV)
    sig = p7.Signals(ALL4, standardize=True)                       # bio-target loss = masked CE (CL default)
    return gate, heads, sig


# ------------------------------- STANDARD regime (full MNIST) -------------------------------
def run_standard_curve(driver, epochs=15, seed=42):
    p7.seed_all(seed)
    tr, _, te = get_standard_loaders(batch_size=64)
    net = p7.Net().to(DEV)
    curve = []
    if driver == "vanilla":
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        for _ in range(epochs):
            for x, y in tr:
                x, y = x.to(DEV), y.to(DEV)
                loss = CE(net.plain(x)[0], y)
                opt.zero_grad(); loss.backward(); opt.step()
            curve.append(_std_acc(net, te))
        return curve
    gate, heads, _ = _all4_parts(net)
    sig = p7.Signals(ALL4, standardize=True, loss_fn=per_sample_ce_plain)   # standard: plain 10-way CE target
    main_opt = torch.optim.Adam(list(net.parameters()) + gate.params(), 1e-3)
    head_opt = torch.optim.Adam(heads.parameters(), 1e-3)
    for _ in range(epochs):
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            loss = CE(gate(net, heads(x).detach(), x), y)
            main_opt.zero_grad(); loss.backward(); main_opt.step()
            hloss = F.mse_loss(heads(x), sig.targets(net, x, y))
            head_opt.zero_grad(); hloss.backward(); head_opt.step()
        curve.append(_std_acc(net, te, gate, heads))
    return curve


# ------------------------------- CL class-IL (Split MNIST) -------------------------------
@torch.no_grad()
def _cl_avg_plain(net, loaders):
    return float(np.mean([p7._acc_plain(net, loaders[i][1]) for i in range(5)]))


@torch.no_grad()
def _cl_avg_gate(net, gate, heads, loaders):
    net.eval(); accs = []
    for i in range(5):
        c = tot = 0
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV)
            c += (gate(net, heads(x), x).argmax(1) == y).sum().item(); tot += len(y)
        accs.append(c / tot)
    return float(np.mean(accs))


def run_cl(method, ep_per_task=5, buffer=1000, seed=42):
    """method in {er, er_all4}. Returns (final_avg_acc, trajectory=[(task, epoch, avg_acc)])."""
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV); buf = p7.Reservoir(buffer); traj = []
    if method == "er":
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        for t in range(5):
            for e in range(ep_per_task):
                for x, y in loaders[t][0]:
                    x, y = x.to(DEV), y.to(DEV)
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    loss = CE(net.plain(torch.cat(Xs))[0], torch.cat(Ys))
                    opt.zero_grad(); loss.backward(); opt.step(); buf.add(x, y)
                traj.append((t, e, _cl_avg_plain(net, loaders)))
        return traj[-1][2], traj
    gate, heads, sig = _all4_parts(net)
    main_opt = torch.optim.Adam(list(net.parameters()) + gate.params(), 1e-3)
    head_opt = torch.optim.Adam(heads.parameters(), 1e-3)
    for t in range(5):
        for e in range(ep_per_task):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                m = heads(Xm).detach()
                loss = CE(gate(net, m, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step(); buf.add(x, y)
                hloss = F.mse_loss(heads(Xm), sig.targets(net, Xm, Ym))
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
            traj.append((t, e, _cl_avg_gate(net, gate, heads, loaders)))
    return traj[-1][2], traj


# ------------------------------- ledger + grid -------------------------------
def load_done():
    if not TSV.exists():
        return set()
    return {"\t".join(ln.split("\t")[:3]) for ln in TSV.read_text().splitlines() if ln.strip()}


def rec(*fields):
    with open(TSV, "a") as f:
        f.write("\t".join(str(x) for x in fields) + "\n")


def _thr(curve, th):
    for i, a in enumerate(curve, 1):
        if a >= th:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "standard", "cl-sweep", "cl-traj", "smoke"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={DEV}  (pt7 convergence/efficiency; all4 gain-neuron; Adam; 1 seed)\n", flush=True)

    if args.part == "smoke":
        c = run_standard_curve("all4", epochs=1); print(f"  smoke standard all4 ep1: {c[-1]:.4f}", flush=True)
        f, _ = run_cl("er_all4", ep_per_task=1); print(f"  smoke cl er_all4 ep1: {f:.4f}", flush=True)
        return

    done = load_done() if args.resume else set()

    if args.part in ("all", "standard"):
        for driver in ("vanilla", "all4"):
            if f"standard\t{driver}\t-" in done:
                continue
            curve = run_standard_curve(driver, epochs=15)
            for e, a in enumerate(curve, 1):
                rec("standard", driver, "-", e, f"{a:.4f}")
            snap = "  ".join(f"e{e}={curve[e-1]:.4f}" for e in (1, 2, 3, 5, 10, 15))
            print(f"  standard {driver:8s} | {snap}  | ->0.97@ep{_thr(curve, 0.97)} 0.98@ep{_thr(curve, 0.98)}",
                  flush=True)

    if args.part in ("all", "cl-sweep"):
        for method in ("er", "er_all4"):
            for ep in (1, 2, 3, 5, 8):
                if f"cl-sweep\t{method}\t{ep}" in done:
                    continue
                final, _ = run_cl(method, ep_per_task=ep)
                rec("cl-sweep", method, ep, f"{final:.4f}")
                print(f"  cl-sweep {method:8s} ep/task={ep} | final avg acc = {final:.4f}", flush=True)

    if args.part in ("all", "cl-traj"):
        for method in ("er", "er_all4"):
            if f"cl-traj\t{method}\t5" in done:
                continue
            final, traj = run_cl(method, ep_per_task=5)
            for k, (t, e, a) in enumerate(traj):
                rec("cl-traj", method, 5, k, t, e, f"{a:.4f}")
            snap = "  ".join(f"t{t}e{e}={a:.3f}" for (t, e, a) in traj if e == 4)   # end-of-task checkpoints
            print(f"  cl-traj  {method:8s} ep=5 | end-of-task: {snap}  final={final:.4f}", flush=True)

    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
