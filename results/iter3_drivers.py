"""Iteration 3 (driver comparison): drivers on the weight_mask target.

Matched conditions per SPEC: target (weight_mask, layer 2, full-rank), modulator
architecture, and hyperparameters are all FIXED at the Iteration 2 best (lr=1e-3,
epochs_per_task=5); only the modulator's INPUT (the driver) changes. driver=none is
the Iteration 2 baseline (0.1979). Each driver: one validation-sequence sanity run
(seed=7) then 3 test seeds (42/43/44). Every driver is detached (control signal only).

Sub-iterations: 3a surprise, 3b uncertainty, 3c activation_stats.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig
from prototype.data import make_sequence
from prototype.train import cl_train

LR, EP = 1e-3, 5
DRIVERS = [("3a", "surprise"), ("3b", "uncertainty"), ("3c", "activation_stats")]
NAIVE = 0.1979
ITER12_BEST = 0.1992  # best avg_final_acc from Iterations 1-2 (the bar to beat by +5pts)

results = {}
for sub, drv in DRIVERS:
    print(f"\n############ Sub-iteration {sub}: driver={drv} ############")

    def mk(seed):
        return CLConfig(
            seed=seed, lr=LR, epochs_per_task=EP,
            use_neuromod=True, neuromod_target="weight_mask",
            neuromod_mask_layer=2, neuromod_mask_rank=0, neuromod_mask_init=0.99,
            neuromod_driver=drv,
        )

    print(f"--- {drv}: validation sanity run (make_sequence(7)) ---")
    vacc, vforget = cl_train(mk(7), "naive", no_wandb=True, sequence=make_sequence(7))
    print(f">>> {drv} VAL: acc={vacc:.4f} forget={vforget:.4f}\n")

    print(f"--- {drv}: 3 test seeds ---")
    accs, forgets = [], []
    for seed in (42, 43, 44):
        acc, forget = cl_train(mk(seed), "naive", no_wandb=True, sequence=None)
        accs.append(acc)
        forgets.append(forget)
        print(f">>> {drv} seed={seed}: acc={acc:.4f} forget={forget:.4f}\n")
    results[drv] = (np.mean(accs), np.std(accs), np.mean(forgets), np.std(forgets))

print("\n=== ITERATION 3 RESULTS (weight_mask target, matched config lr=1e-3 ep=5) ===")
print(f"{'driver':18s} {'avg_final_acc':>16s} {'forgetting':>16s}")
print(f"{'none (Iter 2)':18s} {NAIVE:>10.4f} ± 0.0000 {'0.7982 ± 0.0002':>16s}")
for _, drv in DRIVERS:
    a, sa, f, sf = results[drv]
    print(f"{drv:18s} {a:>10.4f} ± {sa:.4f} {f:>10.4f} ± {sf:.4f}")
best_drv = max(results, key=lambda d: results[d][0])
ba = results[best_drv][0]
print(f"\nBest driver: {best_drv} acc={ba:.4f}")
print(f"Bar to beat (Iter1-2 best {ITER12_BEST} + 5pts) = {ITER12_BEST + 0.05:.4f}")
print(f"beats_by_5pts = {ba - ITER12_BEST >= 0.05}")
