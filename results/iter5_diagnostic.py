"""Iteration 5 (pt3) diagnostic: is the shared output head the class-IL bottleneck?

Runs the same mechanisms under three regimes (seed=42, test sequence, lr=1e-3 ep=5):
  none   = class-IL (default, all 10 logits compete at train and eval)
  loss   = mask the TRAIN loss to the current task's 2 classes; class-IL eval (all 10)
  taskil = mask train AND eval to the task's 2 classes (full task-IL)

Decision gate (no accept/reject): if forgetting collapses for plain Naive under taskil
(vs ~0.798 in class-IL), the shared-head logit competition is confirmed as the dominant
cause, justifying the head-reaching retries in Iterations 6-10. The `loss` regime isolates
how much of the gap is the train-side "don't push down absent classes" effect (lever B).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP = 42, 1e-3, 5


def run(tag, masking, **neuromod):
    cfg = CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, output_masking=masking, **neuromod)
    acc, forget = cl_train(cfg, "naive", no_wandb=True, sequence=None)
    print(f">>> {tag:34s} [{masking:6s}]  acc={acc:.4f}  forget={forget:.4f}\n")
    return tag, masking, acc, forget


rows = []
print("==== Naive (no neuromod) across regimes ====")
rows.append(run("naive", "none"))
rows.append(run("naive", "loss"))
rows.append(run("naive", "taskil"))

print("==== weight_mask (pt2 Iter 2 mechanism) ====")
wm = dict(use_neuromod=True, neuromod_target="weight_mask", neuromod_mask_layer=2, neuromod_mask_rank=0)
rows.append(run("weight_mask", "none", **wm))
rows.append(run("weight_mask", "taskil", **wm))

print("==== activation gain (sprint mechanism) ====")
ag = dict(use_neuromod=True, neuromod_target="activation")
rows.append(run("activation_gain", "none", **ag))
rows.append(run("activation_gain", "taskil", **ag))

print("==== ITERATION 5 DIAGNOSTIC SUMMARY (seed=42, test seq) ====")
print(f"{'mechanism':18s} {'regime':8s} {'avg_final_acc':>14s} {'forgetting':>12s}")
for tag, masking, acc, forget in rows:
    print(f"{tag:18s} {masking:8s} {acc:>14.4f} {forget:>12.4f}")

naive = {m: (a, f) for t, m, a, f in rows if t == "naive"}
print("\n-- decision gate --")
print(f"Naive forgetting: class-IL(none)={naive['none'][1]:.4f}  loss={naive['loss'][1]:.4f}  taskil={naive['taskil'][1]:.4f}")
print(f"Naive avg acc:    class-IL(none)={naive['none'][0]:.4f}  loss={naive['loss'][0]:.4f}  taskil={naive['taskil'][0]:.4f}")
drop = naive["none"][1] - naive["taskil"][1]
print(f"Forgetting drop (none -> taskil) = {drop:.4f}  => head_is_bottleneck = {drop > 0.3}")
