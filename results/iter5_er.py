"""Iteration 5 addendum: neuromod+ER vs ER under the task-IL regime (pt3 dual-comparison B).

Frozen ER config (BEST_CL_ER): lr=3e-4, epochs_per_task=5, buffer=1000. Regime: task-IL
(output_masking=taskil), matching the iter5 diagnostic. Seed=42, test sequence. Tests whether
the pt2 hidden-layer mechanisms add anything on top of replay once the head bottleneck is gone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED = 42
ER = dict(lr=3e-4, epochs_per_task=5, er_buffer_size=1000)  # frozen BEST_CL_ER


def run(tag, masking, **neuromod):
    cfg = CLConfig(seed=SEED, output_masking=masking, **ER, **neuromod)
    acc, forget = cl_train(cfg, "er", no_wandb=True, sequence=None)
    print(f">>> {tag:26s} [{masking:6s}]  acc={acc:.4f}  forget={forget:.4f}\n")
    return tag, acc, forget


rows = []
print("==== ER and neuromod+ER under task-IL (frozen ER config) ====")
rows.append(run("ER", "taskil"))
rows.append(run("weight_mask+ER", "taskil", use_neuromod=True, neuromod_target="weight_mask",
                neuromod_mask_layer=2, neuromod_mask_rank=0))
rows.append(run("activation_gain+ER", "taskil", use_neuromod=True, neuromod_target="activation"))

print("==== iter5 +ER SUMMARY (task-IL, seed=42) ====")
er_acc = rows[0][1]
print(f"{'config':22s} {'avg_final_acc':>14s} {'forgetting':>12s} {'vs ER':>10s}")
for tag, acc, forget in rows:
    print(f"{tag:22s} {acc:>14.4f} {forget:>12.4f} {acc - er_acc:>+10.4f}")
print(f"\n(class-IL ER reference = 0.9023 frozen; this run is task-IL ER = {er_acc:.4f})")
