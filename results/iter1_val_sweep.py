"""Iteration 1 (plasticity) validation sweep + Naive-SGD control.

Tuning is done ONLY on the validation sequence make_sequence(7), never the test
sequence (non-negotiable rule 1). Same budget shape as the sprint: lr x epochs
2x2 grid. Modulator LR fixed at 1e-3 for the main sweep (LR-ratio probe is a
separate debugging-checklist step). The Naive-SGD control isolates the
plasticity mechanism from the Adam->SGD optimizer switch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.data import make_sequence
from prototype.train import cl_train

VAL = make_sequence(7)
LRS = [0.01, 0.1]
EPOCHS = [5, 10]

print("=== Iteration 1 plasticity: validation sweep (seq=make_sequence(7)) ===")
rows = []
for lr in LRS:
    for ep in EPOCHS:
        cfg = CLConfig(
            seed=7, lr=lr, epochs_per_task=ep,
            use_neuromod=True, neuromod_target="plasticity", neuromod_lr=1e-3,
        )
        acc, forget = cl_train(cfg, "naive", no_wandb=True, sequence=VAL)
        rows.append(("plasticity", lr, ep, acc, forget))
        print(f">>> plasticity lr={lr} ep={ep}: acc={acc:.4f} forget={forget:.4f}\n")

print("=== Naive-SGD control (same grid, no neuromod, matched SGD optimizer) ===")
for lr in LRS:
    for ep in EPOCHS:
        cfg = CLConfig(seed=7, lr=lr, epochs_per_task=ep, optimizer="sgd")
        acc, forget = cl_train(cfg, "naive", no_wandb=True, sequence=VAL)
        rows.append(("naive_sgd", lr, ep, acc, forget))
        print(f">>> naive_sgd lr={lr} ep={ep}: acc={acc:.4f} forget={forget:.4f}\n")

print("=== SUMMARY ===")
for name, lr, ep, acc, forget in rows:
    print(f"{name:12s} lr={lr:<5} ep={ep:<3} acc={acc:.4f} forget={forget:.4f}")
best = max((r for r in rows if r[0] == "plasticity"), key=lambda r: r[3])
print(f"\nBEST plasticity (val): lr={best[1]} ep={best[2]} acc={best[3]:.4f}")
