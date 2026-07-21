"""pt7 VARIANTS (follow-up to pt7_neuromodulators.py) — user-requested extra mechanisms.

Reuses the pt7 primitives (Net, gates, Heads, Signals, train loops) and adds:
  A. STANDARD regime (full MNIST, single task, 10-way CE): all4 gate vs vanilla, sgd+adam.
  B. New head-based CL drivers (recognised by the extended Signals; run via pt7n.run_cell):
       DA_fast   = (loss - ema_fast)/ema_fast
       ACh_ema   = ema(entropy)            (lag-1 scalar)
       ACh_vol_ps= |loss - ema_fast|       (PER-SAMPLE volatility)
       5HT_ema   = ema_slow(-loss)         (tonic scalar)
  C. NE double-forward / multidim drivers (NO head — computed directly, gates ALL layers):
       NE_emb_all = ||h1-mean_h1|| scalar, double forward, gates (h0,h1,out)
       NE_vec_h1  = h1-mean_h1  (400-dim multidim driver), double forward
       NE_vec_x   = x -mean_x   (784-dim multidim driver, pre-forward: input novelty)
       NE_vecproj = R(x-mean_x) (PROJ_DIM-dim random projection of the 784-dim input diff)
  D. STANDARDISATION ablation: run B+C and the OLD {ACh, NE_emb, DA_step} with standardize on/off
     (old drivers: only the OFF runs are new; the ON runs already live in pt7_results.tsv).

Ledger results/pt7_variants_results.tsv (own; `--resume` skips done). class-IL Split MNIST unless STANDARD;
gain (h0,h1,out) neuron; seed 42, lr 1e-3, ep 5, buffer 1000, 1 seed. Table: pt7_variants_make_table (below).
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
import pt7_neuromodulators as p7                                  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from data import SplitMNIST, get_standard_loaders               # noqa: E402

DEV, EPS, BS = p7.DEV, p7.EPS, p7.BS
H0, H1, OUT, GATEDIM = p7.H0, p7.H1, p7.OUT, p7.GATEDIM
CE = nn.CrossEntropyLoss()
PROJ_DIM = 32
TSV = Path(__file__).resolve().parent / "pt7_variants_results.tsv"


def per_sample_ce_plain(logits, y):
    return F.cross_entropy(logits, y, reduction="none")


# ------------------------------- C. NE double-forward / multidim drivers -------------------------------
class NEDriver:
    """Computes a (B,K) NE driver directly (no head); gates all layers. emb_all=scalar h1-novelty;
    vec_h1=h1 diff (double fwd); vec_x=input diff (pre-forward); vecproj=random projection of the input diff."""
    def __init__(self, kind, standardize, seed=0, mean_mode="ema"):
        self.kind = kind; self.standardize = standardize; self.mean_mode = mean_mode
        self.mh1 = None; self.mx = None; self.ch1 = 0; self.cx = 0     # counts for cumulative mean
        self.run_mean = None; self.run_var = None; self.inited = False
        g = torch.Generator().manual_seed(seed)
        if kind == "vecproj":
            self.R = (torch.randn(784, PROJ_DIM, generator=g) / math.sqrt(784)).to(DEV)
        elif kind == "vec_h1proj":
            self.R = (torch.randn(H1, PROJ_DIM, generator=g) / math.sqrt(H1)).to(DEV)

    def K(self):
        return {"emb_all": 1, "vec_h1": H1, "vec_x": 784,
                "vecproj": PROJ_DIM, "vec_h1proj": PROJ_DIM}[self.kind]

    def _upd_mean(self, m, cur, cattr):                          # ema (recent-weighted) or cumulative (true mean)
        if self.mean_mode == "ema":
            return m + BS * (cur.mean(0) - m)
        c = getattr(self, cattr) + cur.size(0); setattr(self, cattr, c)
        return m + (cur.sum(0) - cur.size(0) * m) / c            # incremental cumulative mean

    @torch.no_grad()
    def value(self, net, x, update=True):
        x2 = x.view(x.size(0), -1)
        if self.kind in ("vec_x", "vecproj"):                    # input-space novelty (pre-forward)
            if self.mx is None:
                self.mx = x2.mean(0).clone(); self.cx = x2.size(0)
            elif update:
                self.mx = self._upd_mean(self.mx, x2, "cx")
            diff = x2 - self.mx
            v = diff if self.kind == "vec_x" else diff @ self.R
        else:                                                    # h1 novelty (double forward)
            _, h1 = net.plain(x)
            if self.mh1 is None:
                self.mh1 = h1.mean(0).clone(); self.ch1 = h1.size(0)
            elif update:
                self.mh1 = self._upd_mean(self.mh1, h1, "ch1")
            diff = h1 - self.mh1
            if self.kind == "emb_all":
                v = diff.norm(dim=1, keepdim=True)
            elif self.kind == "vec_h1":
                v = diff
            else:                                                # vec_h1proj: downproject the 400-dim h1 diff
                v = diff @ self.R
        if update and self.standardize:
            bm = v.mean(0); bv = v.var(0, unbiased=False)
            if not self.inited:
                self.run_mean = bm.clone(); self.run_var = bv.clone(); self.inited = True
            else:
                self.run_mean = 0.99 * self.run_mean + 0.01 * bm
                self.run_var = 0.99 * self.run_var + 0.01 * bv
        if self.standardize and self.inited:
            return (v - self.run_mean) / (self.run_var.sqrt() + EPS)
        return v


def run_ne(kind, arm, opt_kind, standardize, mean_mode="ema", lr=1e-3, epochs=5, buffer=1000, seed=42):
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV); drv = NEDriver(kind, standardize, mean_mode=mean_mode); K = drv.K()
    gate = p7.NeuronGate(K, None).to(DEV)
    if arm == "buf-own":                                         # naive main + per-task replay meta-loss on P
        main_opt = p7._opt(opt_kind, net.parameters(), lr)
        gate_opt = torch.optim.Adam(gate.params(), lr)
    else:
        opt = p7._opt(opt_kind, list(net.parameters()) + gate.params(), lr)
    buf = p7.Reservoir(buffer) if arm in ("er-own", "buf-own") else None
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if arm == "er-own":
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                    v = drv.value(net, Xm)
                    loss = CE(gate(net, v, Xm), Ym)
                    opt.zero_grad(); loss.backward(); opt.step(); buf.add(x, y)
                elif arm == "nobuf":
                    v = drv.value(net, x)
                    loss = p7.masked_ce(gate(net, v, x), y)
                    opt.zero_grad(); loss.backward(); opt.step()
                else:                                            # buf-own
                    v = drv.value(net, x)                        # updates running mean/stats
                    loss = p7.masked_ce(gate(net, v, x, detach_P=True), y)   # main net naive, P frozen
                    main_opt.zero_grad(); loss.backward(); main_opt.step(); buf.add(x, y)
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    for j in range(t):
                        s = buf.sample_task(j, 64)
                        if s is not None:
                            Xs.append(s[0].to(DEV)); Ys.append(s[1].to(DEV))
                    Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                    vm = drv.value(net, Xm, update=False)
                    meta = p7.masked_ce(gate(net, vm, Xm), Ym)   # meta-loss trains P
                    gate_opt.zero_grad(); meta.backward(); gate_opt.step()
    net.eval(); c = tot = 0; mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}
    with torch.no_grad():
        for i in range(5):
            for x, y in loaders[i][1]:
                x, y = x.to(DEV), y.to(DEV); b = x.size(0)
                v = drv.value(net, x, update=False)
                c += (gate(net, v, x).argmax(1) == y).sum().item()
                pl = gate.per_layer_mag(v)
                for k in mags:
                    mags[k] += pl[k] * b
                tot += b
    return dict(pred=c / tot, true=float("nan"), probe=float("nan"),
                per_layer={k: mags[k] / tot for k in mags})


# ------------------------------- A. STANDARD regime (full MNIST) -------------------------------
@torch.no_grad()
def _std_acc(net, loader, gate=None, heads=None):
    net.eval(); c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        logits = net.plain(x)[0] if gate is None else gate(net, heads(x), x)
        c += (logits.argmax(1) == y).sum().item(); tot += len(y)
    return c / tot


def run_standard(gate_on, opt_kind, lr=1e-3, epochs=5, seed=42):
    p7.seed_all(seed)
    tr, _, te = get_standard_loaders(batch_size=64)
    net = p7.Net().to(DEV)
    if not gate_on:
        opt = p7._opt(opt_kind, net.parameters(), lr)
        for _ in range(epochs):
            for x, y in tr:
                x, y = x.to(DEV), y.to(DEV)
                loss = CE(net.plain(x)[0], y)
                opt.zero_grad(); loss.backward(); opt.step()
        return dict(pred=_std_acc(net, te), true=float("nan"), probe=float("nan"),
                    per_layer={"h0": 0.0, "h1": 0.0, "out": 0.0})
    drivers = ["DA", "ACh", "NE", "5HT"]; K = 4
    gate = p7.NeuronGate(K, None).to(DEV); heads = p7.Heads(K).to(DEV)
    sig = p7.Signals(drivers, standardize=True, loss_fn=per_sample_ce_plain)
    main_opt = p7._opt(opt_kind, list(net.parameters()) + gate.params(), lr)
    head_opt = torch.optim.Adam(heads.parameters(), lr)
    for _ in range(epochs):
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            m = heads(x).detach()
            loss = CE(gate(net, m, x), y)
            main_opt.zero_grad(); loss.backward(); main_opt.step()
            T = sig.targets(net, x, y)
            hloss = F.mse_loss(heads(x), T)
            head_opt.zero_grad(); hloss.backward(); head_opt.step()
    m0 = heads
    net.eval()
    pl = gate.per_layer_mag(heads(next(iter(te))[0].to(DEV)))
    return dict(pred=_std_acc(net, te, gate, m0), true=float("nan"), probe=float("nan"), per_layer=pl)


# ------------------------------- ledger + grid -------------------------------
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


NEW_HEAD = ["DA_fast", "ACh_ema", "ACh_vol_ps", "5HT_ema"]       # via pt7n.run_cell
OLD_HEAD = ["ACh", "NE_emb", "DA_step"]                          # standardize=OFF only (ON already in pt7)
NE_KINDS = ["emb_all", "vec_h1", "vec_h1proj", "vec_x", "vecproj"]   # via run_ne
ARMS = ["er-own", "nobuf", "buf-own"]
OPTS = ["sgd", "adam"]


CUM_KINDS = ["emb_all", "vec_h1", "vec_h1proj", "vec_x"]         # NE kinds to also try with a cumulative mean


def build_cells(part):
    cells = []  # (kind, name, arm, opt, standardize, mean_mode)
    if part in ("all", "standard"):
        for gate_on in (True, False):
            for opt in OPTS:
                cells.append(("standard", "all4" if gate_on else "vanilla", "-", opt, True, "ema"))
    if part in ("all", "new-head"):
        for n in NEW_HEAD:
            for std in (True, False):
                for arm in ARMS:
                    for opt in OPTS:
                        cells.append(("head", n, arm, opt, std, "ema"))
    if part in ("all", "old-head"):
        for n in OLD_HEAD:                                       # OFF only
            for arm in ARMS:
                for opt in OPTS:
                    cells.append(("head", n, arm, opt, False, "ema"))
    if part in ("all", "ne"):
        for k in NE_KINDS:
            for std in (True, False):
                for arm in ARMS:
                    for opt in OPTS:
                        cells.append(("ne", k, arm, opt, std, "ema"))
    if part in ("all", "extra"):
        for k in CUM_KINDS:                                      # cumulative-mean NE (vs the EMA runs)
            for arm in ("er-own", "nobuf"):
                for opt in OPTS:
                    cells.append(("ne", k, arm, opt, True, "cumulative"))
        for arm in ARMS:                                         # NE_rise (tonic) WITHOUT standardization
            for opt in OPTS:
                cells.append(("head", "NE_rise", arm, opt, False, "ema"))
    return cells


def fmt(res):
    pl = res["per_layer"]
    return (f"pred={res['pred']:.4f}  true={res['true']:.4f}  probe={res['probe']:.3f}  "
            f"|g|(h0/h1/out)={pl['h0']:.3f}/{pl['h1']:.3f}/{pl['out']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "standard", "new-head", "old-head", "ne", "extra"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={DEV}  (pt7 variants; 1 seed)\n", flush=True)
    done = load_done() if args.resume else set()

    for kind, name, arm, opt, std, mean_mode in build_cells(args.part):
        tag = f"{kind}|{name}|{arm}|{opt}|std{int(std)}" + ("|cum" if mean_mode == "cumulative" else "")
        if tag in done:
            continue
        if kind == "standard":
            r = run_standard(name == "all4", opt)
        elif kind == "head":
            r = p7.run_cell(name, "neuron", arm, opt, standardize=std)
        else:
            r = run_ne(name, arm, opt, std, mean_mode=mean_mode)
        mtag = " cum" if mean_mode == "cumulative" else ""
        print(f"  {kind:8s} {name:10s} {arm:7s} {opt:4s} std{int(std)}{mtag} | {fmt(r)}", flush=True)
        record(tag, r)
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
