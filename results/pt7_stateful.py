"""pt7 STATEFUL / z-score drivers (user-requested). CL Split MNIST, gain-NEURON (K=1), class-IL.

Mechanisms (driver = per-sample scalar m(x); gate Gamma = 1 + m*P over 810 gains):
  nerisez : relu((H - ema_H)/sqrt(var_H + eps)) — a z-scored (rectified) ENTROPY-surprise driver. H is
            PREDICTED by a head; ema_H/var_H are running stats of the ACTUAL past entropies (from the real
            forward at train, NOT predicted). Eval: frozen (end-of-train stats) vs running (update on stream).
  ach     : standardized predicted entropy (= ACh), but the predictor is a GRU.
Predictor: MLP (784->32->1) or a stateful GRU (784->proj->GRUCell->[proj|hidden]->1; hidden persists across
batches, detached each step = truncated BPTT len 1). At inference the GRU hidden state + running stats are
either FROZEN (end of training) or RUNNING (keep updating on the eval stream, from PREDICTED H since actual
H is not available pre-forward). One `eval_mode` flag controls both.

Grid: (nerisez MLP, nerisez GRU, ach GRU) x {frozen,running} x {er-own,nobuf} x {sgd,adam} = 24 cells.
Head trained WITH REPLAY (regress actual H, MSE). seed42 lr1e-3 ep5 buffer1000, 1 seed.
Baselines (from pt7): naive 0.629/0.390, er 0.723/0.895 (sgd/adam). Ledger pt7_stateful_results.tsv.
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_neuromodulators as p7                                  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from data import SplitMNIST                                       # noqa: E402

DEV, EPS, BS = p7.DEV, p7.EPS, p7.BS
CE = nn.CrossEntropyLoss()
TSV = Path(__file__).resolve().parent / "pt7_stateful_results.tsv"


class StatefulDriver(nn.Module):
    def __init__(self, mech, gru, proj=64, hid=64):
        super().__init__()
        self.mech = mech; self.gru = gru
        if gru:
            self.projx = nn.Linear(784, proj); self.cell = nn.GRUCell(proj, hid)
            self.out = nn.Linear(proj + hid, 1)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
            self.hidden = torch.zeros(1, hid, device=DEV)
        else:
            self.f1 = nn.Linear(784, 32); self.f2 = nn.Linear(32, 1)
            nn.init.zeros_(self.f2.weight); nn.init.zeros_(self.f2.bias)
        self.emaH = None; self.varH = None                       # actual-entropy stats (nerisez z-score)
        self.rm = None; self.rv = None                           # standardization of H_pred (ach)

    def predictH(self, x, update_state=True):
        x = x.view(x.size(0), -1)
        if self.gru:
            p = F.relu(self.projx(x)); h_new = self.cell(p.mean(0, keepdim=True), self.hidden)
            if update_state:
                self.hidden = h_new.detach()
            return self.out(torch.cat([p, h_new.expand(p.size(0), -1)], 1))
        return self.f2(F.relu(self.f1(x)))

    @torch.no_grad()
    def upd_actual(self, Hact):                                  # EMA mean/var of actual entropy
        m = Hact.mean().item()
        if self.emaH is None:
            self.emaH = m; self.varH = Hact.var(unbiased=False).item()
        else:
            self.varH = (1 - BS) * self.varH + BS * ((Hact - self.emaH) ** 2).mean().item()
            self.emaH = (1 - BS) * self.emaH + BS * m

    def _standardize(self, v, update):
        with torch.no_grad():
            bm = v.mean(0); bv = v.var(0, unbiased=False)
            if self.rm is None:
                self.rm = bm.clone(); self.rv = bv.clone()
            elif update:
                self.rm = 0.99 * self.rm + 0.01 * bm; self.rv = 0.99 * self.rv + 0.01 * bv
        return (v - self.rm) / (self.rv.sqrt() + EPS)

    def driver(self, x, update_state, update_stats):
        Hpred = self.predictH(x, update_state)                   # (B,1)
        if self.mech == "ach":
            return self._standardize(Hpred, update_stats)
        if self.emaH is None:                                    # bootstrap (batch z-score)
            return F.relu(Hpred - Hpred.mean()) / (Hpred.std() + EPS)
        if update_stats:                                         # eval-running: update from PREDICTED H
            self.upd_actual(Hpred.detach())
        return F.relu((Hpred - self.emaH) / math.sqrt(self.varH + EPS))


def run_stateful(mech, gru, eval_mode, arm, opt_kind, lr=1e-3, epochs=5, buffer=1000, seed=42):
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ); loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV); drv = StatefulDriver(mech, gru).to(DEV); gate = p7.NeuronGate(1, None).to(DEV)
    opt = p7._opt(opt_kind, list(net.parameters()) + gate.params(), lr)
    head_opt = torch.optim.Adam(drv.parameters(), lr)
    buf = p7.Reservoir(buffer) if arm == "er-own" else None
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
                else:
                    Xm, Ym = x.view(x.size(0), -1), y
                m = drv.driver(Xm, update_state=True, update_stats=(mech == "ach")).detach()
                logits = gate(net, m, Xm)
                loss = CE(logits, Ym) if arm == "er-own" else p7.masked_ce(logits, Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    Hact = p7.entropy(net.plain(Xm)[0]).unsqueeze(1)
                if mech == "nerisez":
                    drv.upd_actual(Hact)
                hloss = F.mse_loss(drv.predictH(Xm, update_state=False), Hact)
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                if arm == "er-own":
                    buf.add(x, y)
    net.eval(); upd = eval_mode == "running"; c = tot = 0; mag = 0.0
    with torch.no_grad():
        for i in range(5):
            for x, y in loaders[i][1]:
                x, y = x.to(DEV), y.to(DEV); b = x.size(0)
                m = drv.driver(x.view(b, -1), update_state=upd, update_stats=upd)
                c += (gate(net, m, x).argmax(1) == y).sum().item()
                mag += gate.per_layer_mag(m)["out"] * b; tot += b
    return dict(pred=c / tot, magout=mag / tot)


MECHS = [("nerisez", False), ("nerisez", True), ("ach", True)]


def load_done():
    return {ln.split("\t", 1)[0] for ln in TSV.read_text().splitlines()} if TSV.exists() else set()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); done = load_done() if args.resume else set()
    print(f"device={DEV}  (pt7 stateful; er 0.723/0.895, naive 0.629/0.390)\n", flush=True)
    for mech, gru in MECHS:
        for eval_mode in ("frozen", "running"):
            for arm in ("er-own", "nobuf"):
                for opt in ("sgd", "adam"):
                    tag = f"{mech}|gru{int(gru)}|{eval_mode}|{arm}|{opt}"
                    if tag in done:
                        continue
                    r = run_stateful(mech, gru, eval_mode, arm, opt)
                    print(f"  {mech:8s} gru{int(gru)} {eval_mode:7s} {arm:7s} {opt:4s} | "
                          f"pred={r['pred']:.4f}  |g|out={r['magout']:.3f}", flush=True)
                    with open(TSV, "a") as f:
                        f.write(f"{tag}\t{r['pred']:.4f}\t{r['magout']:.4f}\n")
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
