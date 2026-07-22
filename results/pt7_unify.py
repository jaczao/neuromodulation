"""pt7 UNIFY-12 — a new "all4-like" composite that unifies 12 heterogeneous neuromodulator drivers
into ONE rank-K linear gate (user-requested). Same design as all4 (Γ=1+Σ_k m_k P_k, driver detached,
predictors trained WITH REPLAY), just K=12 and the 12 columns come from THREE sources:

  head-regressed (shared Heads MLP 784->32->8 regresses p7.Signals targets, oracle-free at eval):
    DA, ACh, NE, NE_emb, 5HT, DA_step, DA_fast, ACh_vol_ps            (8)
  headless (pt7_variants.NEDriver, ||h1-mean_h1|| double-forward):
    emb_all                                                           (1)
  stateful (pt7_stateful.StatefulDriver, own MLP/GRU predictor of entropy, trained w/ replay):
    nerisez-MLP, nerisez-GRU, ach-GRU                                 (3)

All 12 driver columns are assembled per step, DETACHED, and UNIFIED-standardised (running mean/var over
the 12-vector, std1) before the gate — mandatory (mixed scales; cf. all4-std0 collapse). Predictors are
trained separately (Heads8 MSE to Signals targets; each stateful predictH MSE to actual entropy). Gate
driven by PREDICTED values only -> oracle-free at eval (stateful eval = FROZEN, per pt7_stateful's
frozen≈running finding).

Operating point matches the recent tuned runs: Adam, TUNED lr=3e-4/ep5, er-own, std1, seed 42; gain
NEURON and SYNAPSE. Reference: ER-adam-tuned 0.9029±0.0043, all4 0.9104±0.0022 (3-seed).
Ledger results/pt7_unify_results.tsv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_neuromodulators as p7                                  # noqa: E402
import pt7_variants as pv                                         # noqa: E402
import pt7_stateful as ps                                         # noqa: E402
import pt7_tuned_syn as T                                         # noqa: E402

DEV, EPS = p7.DEV, p7.EPS
CE = nn.CrossEntropyLoss()
LR, EP, BUFFER, SEED = 3e-4, 5, 1000, 42
HEAD_DRIVERS = ["DA", "ACh", "NE", "NE_emb", "5HT", "DA_step", "DA_fast", "ACh_vol_ps"]
K = len(HEAD_DRIVERS) + 1 + 3                                     # 8 head + emb_all + 3 stateful = 12
TSV = Path(__file__).resolve().parent / "pt7_unify_results.tsv"


class RunStd:
    """Unified running-stats standardizer over the assembled 12-vector (matches Signals' scheme)."""
    def __init__(self):
        self.rm = None; self.rv = None

    def __call__(self, v, update):
        with torch.no_grad():
            bm = v.mean(0); bv = v.var(0, unbiased=False)
            if self.rm is None:
                self.rm = bm.clone(); self.rv = bv.clone()
            elif update:
                self.rm = 0.99 * self.rm + 0.01 * bm
                self.rv = 0.99 * self.rv + 0.01 * bv
        return (v - self.rm) / (self.rv.sqrt() + EPS)


def build_unify(gran, seed=SEED):
    p7.seed_all(seed)
    net = p7.Net().to(DEV)
    gate = p7.make_gate(gran, K, None)
    heads8 = p7.Heads(len(HEAD_DRIVERS)).to(DEV)                  # predicts the 8 Signals targets
    sig8 = p7.Signals(HEAD_DRIVERS, standardize=False)            # raw targets; unified std handles scale
    nedrv = pv.NEDriver("emb_all", standardize=False)            # headless h1-novelty
    stateful = [ps.StatefulDriver("nerisez", gru=False).to(DEV),  # nerisez-MLP
                ps.StatefulDriver("nerisez", gru=True).to(DEV),   # nerisez-GRU
                ps.StatefulDriver("ach", gru=True).to(DEV)]       # ach-GRU
    return net, gate, heads8, sig8, nedrv, stateful


def driver_columns(net, x, heads8, nedrv, stateful, *, update):
    """Assemble the (B,12) PREDICTED driver vector. update=True during training (advances state/means)."""
    cols = [heads8(x)]                                            # (B,8) head predictions
    cols.append(nedrv.value(net, x, update=update))              # emb_all (B,1)
    for drv in stateful[:2]:                                      # nerisez-MLP, nerisez-GRU
        cols.append(drv.driver(x, update_state=update, update_stats=False))
    cols.append(stateful[2].driver(x, update_state=update, update_stats=update))  # ach-GRU
    return torch.cat(cols, 1)


def run_unify(gran, seed=SEED):
    net, gate, heads8, sig8, nedrv, stateful = build_unify(gran, seed)
    loaders = T.build_loaders(T.SEQ, 0.0)                        # TEST loaders (report)
    rstd = RunStd()
    main_opt = p7._opt("adam", list(net.parameters()) + gate.params(), LR)
    head_opt = torch.optim.Adam(heads8.parameters(), LR)
    stateful_params = [p for drv in stateful for p in drv.parameters()]
    state_opt = torch.optim.Adam(stateful_params, LR)
    buf = p7.Reservoir(BUFFER)
    for t in range(5):
        for _ in range(EP):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                # --- gate step: assemble predicted driver, detach, unified-standardize ---
                m = driver_columns(net, Xm, heads8, nedrv, stateful, update=True).detach()
                m = rstd(m, update=True)
                loss = CE(gate(net, m, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                # --- predictor training (separate opts, WITH replay via Xm) ---
                T8 = sig8.targets(net, Xm, Ym)
                h8loss = F.mse_loss(heads8(Xm), T8)
                head_opt.zero_grad(); h8loss.backward(); head_opt.step()
                with torch.no_grad():
                    Hact = p7.entropy(net.plain(Xm)[0]).unsqueeze(1)
                for drv in stateful[:2]:                          # nerisez -> update actual-entropy stats
                    drv.upd_actual(Hact)
                sloss = sum(F.mse_loss(drv.predictH(Xm, update_state=False), Hact) for drv in stateful)
                state_opt.zero_grad(); sloss.backward(); state_opt.step()
                buf.add(x, y)
    # --- eval (FROZEN stateful/means), class-IL 10-way ---
    net.eval(); c = tot = 0
    with torch.no_grad():
        for i in range(5):
            for x, y in loaders[i][1]:
                x, y = x.to(DEV), y.to(DEV)
                m = driver_columns(net, x.view(x.size(0), -1), heads8, nedrv, stateful, update=False)
                m = rstd(m, update=False)
                c += (gate(net, m, x).argmax(1) == y).sum().item(); tot += len(y)
    return c / tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={DEV}  (pt7 UNIFY-12; Adam tuned lr={LR:g}/ep{EP}, er-own, std1, seed{SEED}; K={K})\n", flush=True)
    done = {}
    if args.resume and TSV.exists():
        done = {ln.split('\t')[0]: float(ln.split('\t')[1]) for ln in TSV.read_text().splitlines() if ln.strip()}
    for gran in ("neuron", "synapse"):
        tag = f"all12|{gran}|er-own|adam|std1|lr{LR:g}|ep{EP}|seed{SEED}"
        if tag in done:
            acc = done[tag]
        else:
            acc = run_unify(gran)
            with open(TSV, "a") as fh:
                fh.write(f"{tag}\t{acc:.4f}\n")
        print(f"  all12 {gran:8s} er-own adam std1 | acc={acc:.4f}  (ER 0.9029, all4 0.9104)", flush=True)
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
