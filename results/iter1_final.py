"""Iteration 1 (plasticity): LR-ratio debugging probe (val) + 3-seed test eval.

Best val config from iter1_val_sweep.py: lr=0.1, epochs_per_task=10, mod_lr=1e-3.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig
from prototype.data import make_sequence
from prototype.train import cl_train

VAL = make_sequence(7)
BEST_LR, BEST_EP = 0.1, 10

print("=== Debugging checklist item 4: modulator LR ratio (validation seq) ===")
for mod_lr in [1e-4, 1e-2]:
    cfg = CLConfig(
        seed=7, lr=BEST_LR, epochs_per_task=BEST_EP,
        use_neuromod=True, neuromod_target="plasticity", neuromod_lr=mod_lr,
    )
    acc, forget = cl_train(cfg, "naive", no_wandb=True, sequence=VAL)
    print(f">>> plasticity mod_lr={mod_lr}: acc={acc:.4f} forget={forget:.4f}\n")

print("=== Iteration 1 final: 3-seed TEST eval at best val config ===")
accs, forgets = [], []
for seed in (42, 43, 44):
    cfg = CLConfig(
        seed=seed, lr=BEST_LR, epochs_per_task=BEST_EP,
        use_neuromod=True, neuromod_target="plasticity", neuromod_lr=1e-3,
    )
    acc, forget = cl_train(cfg, "naive", no_wandb=True, sequence=None)  # test sequence
    accs.append(acc)
    forgets.append(forget)
    print(f">>> seed={seed}: acc={acc:.4f} forget={forget:.4f}\n")

print("=== ITERATION 1 RESULT ===")
print(f"plasticity (per-neuron, lookahead, SGD) | test seeds {accs}")
print(f"avg_final_acc = {np.mean(accs):.4f} +- {np.std(accs):.4f}")
print(f"forgetting    = {np.mean(forgets):.4f} +- {np.std(forgets):.4f}")
print(f"Naive (frozen baseline) = 0.1979 +- 0.0003")
print(f"beats_naive_by_5pts = {np.mean(accs) - 0.1979 >= 0.05}")
