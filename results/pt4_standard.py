"""pt4: every neuromod mechanism in the STANDARD (single-task full-MNIST) regime.

Reports vanilla vs neuromod test accuracy, mean +- std over 3 seeds, at the frozen tuned
standard config (lr=3e-4, epochs=20, batch=64) behind results/standard_mnist_table.md.

Group R (runnable in standard), all run here:
  R1 activation gain   (target=activation)
  R2 weight mask       (target=weight_mask)
  R3 logit calibration (target=logit)
  R4 plasticity/meta-LR (target=plasticity)  -- SGD main net, compared to the SGD-vanilla ref
  R5 importance gating (target=importance)

Group N (task_route, consolidation, logit+recency, weight_mask drivers, stateful) are N/A by
construction in single-task standard learning (no task boundaries / no task sequence); see
SPEC-proto-pt4.md. They are NOT run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import StandardConfig
from prototype.train import train_standard

SEEDS = (42, 43, 44)
BASE = dict(lr=3e-4, epochs=20, batch_size=64)


def run3(tag, **extra):
    accs = []
    for s in SEEDS:
        cfg = StandardConfig(seed=s, **BASE, **extra)
        _, test_acc = train_standard(cfg, no_wandb=True)
        accs.append(test_acc)
        print(f"  [{tag}] seed={s} test_acc={test_acc:.4f}")
    a, sa = float(np.mean(accs)), float(np.std(accs))
    print(f">>> {tag:28s} test_acc={a:.4f}±{sa:.4f}")
    return tag, a, sa


rows = []
rows.append(run3("vanilla (adam)"))
rows.append(run3("R1 activation",  use_neuromod=True, neuromod_target="activation"))
rows.append(run3("R2 weight_mask", use_neuromod=True, neuromod_target="weight_mask"))
rows.append(run3("R3 logit",       use_neuromod=True, neuromod_target="logit"))
rows.append(run3("R5 importance",  use_neuromod=True, neuromod_target="importance"))
# R4 plasticity trains the main net with plain SGD (Adam-moments caveat), so its fair reference
# is an SGD-vanilla at the same config, not the Adam vanilla above.
rows.append(run3("vanilla (sgd ref)", optimizer="sgd"))
rows.append(run3("R4 plasticity",  use_neuromod=True, neuromod_target="plasticity"))

print("\n==== pt4 STANDARD-LEARNING SUMMARY (full MNIST, 3 seeds 42/43/44) ====")
print(f"{'config':28s} test_acc")
for tag, a, sa in rows:
    print(f"{tag:28s} {a:.4f}±{sa:.4f}")
