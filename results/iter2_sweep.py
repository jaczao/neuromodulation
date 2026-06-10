"""Iteration 2 (weight_mask): validation sweep + 3-seed test eval.

Per-synapse mask on the 2nd linear layer (400x400, full-rank head). Mask is in the
forward graph so the modulator trains by ordinary backprop (single Adam optimizer
over net+modulator). Tuning ONLY on the validation sequence make_sequence(7); same
2x2 budget as the sprint neuromod sweep: lr x epochs_per_task. Composes with naive.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig
from prototype.data import make_sequence
from prototype.train import cl_train

VAL = make_sequence(7)
LRS = [3e-4, 1e-3]
EPOCHS = [5, 10]

print("=== Iteration 2 weight_mask: validation sweep (seq=make_sequence(7)) ===")
rows = []
for lr in LRS:
    for ep in EPOCHS:
        cfg = CLConfig(
            seed=7, lr=lr, epochs_per_task=ep,
            use_neuromod=True, neuromod_target="weight_mask",
            neuromod_mask_layer=2, neuromod_mask_rank=0, neuromod_mask_init=0.99,
        )
        acc, forget = cl_train(cfg, "naive", no_wandb=True, sequence=VAL)
        rows.append((lr, ep, acc, forget))
        print(f">>> weight_mask lr={lr} ep={ep}: acc={acc:.4f} forget={forget:.4f}\n")

print("=== SUMMARY (validation) ===")
for lr, ep, acc, forget in rows:
    print(f"weight_mask lr={lr:<6} ep={ep:<3} acc={acc:.4f} forget={forget:.4f}")
best = max(rows, key=lambda r: r[2])
print(f"\nBEST weight_mask (val): lr={best[0]} ep={best[1]} acc={best[2]:.4f}")

print("\n=== Iteration 2 final: 3-seed TEST eval at best val config ===")
accs, forgets = [], []
for seed in (42, 43, 44):
    cfg = CLConfig(
        seed=seed, lr=best[0], epochs_per_task=best[1],
        use_neuromod=True, neuromod_target="weight_mask",
        neuromod_mask_layer=2, neuromod_mask_rank=0, neuromod_mask_init=0.99,
    )
    acc, forget = cl_train(cfg, "naive", no_wandb=True, sequence=None)  # test sequence
    accs.append(acc)
    forgets.append(forget)
    print(f">>> seed={seed}: acc={acc:.4f} forget={forget:.4f}\n")

print("=== ITERATION 2 RESULT ===")
print(f"weight_mask (per-synapse, layer 2, full-rank) | test seeds {accs}")
print(f"avg_final_acc = {np.mean(accs):.4f} +- {np.std(accs):.4f}")
print(f"forgetting    = {np.mean(forgets):.4f} +- {np.std(forgets):.4f}")
print("Naive (frozen baseline) = 0.1979 +- 0.0003")
print(f"beats_naive_by_5pts = {np.mean(accs) - 0.1979 >= 0.05}")
