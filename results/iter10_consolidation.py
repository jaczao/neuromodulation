"""Iteration 10 (pt3): stateful boundary detector + EWC-style consolidation, dual comparison.

Running surprise (loss EMA) detects task boundaries with no task ID; at each detected boundary the
model snapshots params and accumulates an importance anchor (online EWC at detected boundaries).
Tune the penalty lambda on the validation sequence, then 3 test seeds for naive (standalone) and er.
Reports boundaries detected (4 true internal). Refs: Naive 0.1979, ER 0.9023.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig
from prototype.data import make_sequence
from prototype.train import cl_train

SEEDS = (42, 43, 44)
CON = dict(use_neuromod=True, neuromod_target="consolidation")

print("==== lambda sweep on validation sequence (seed=7, naive) ====")
val = []
for lam in (0.1, 1.0, 10.0):
    acc, _ = cl_train(CLConfig(seed=7, lr=1e-3, epochs_per_task=5, neuromod_importance_lambda=lam, **CON),
                      "naive", no_wandb=True, sequence=make_sequence(7))
    val.append((lam, acc)); print(f">>> lambda={lam} val_acc={acc:.4f}\n")
best_lam = max(val, key=lambda r: r[1])[0]
print(f"BEST lambda (val) = {best_lam}\n")


def run3(tag, method, **base):
    accs = [cl_train(CLConfig(seed=s, neuromod_importance_lambda=best_lam, **base, **CON),
                     method, no_wandb=True, sequence=None)[0] for s in SEEDS]
    a, sa = np.mean(accs), np.std(accs)
    print(f">>> {tag:24s} acc={a:.4f}±{sa:.4f}")
    return a, sa


print("==== 3-seed test (class-IL) at best lambda ====")
a_a, sa_a = run3("consolidation + naive", "naive", lr=1e-3, epochs_per_task=5)
a_b, sa_b = run3("consolidation + ER",    "er",    lr=3e-4, epochs_per_task=5, er_buffer_size=1000)

print("\n==== ITERATION 10 SUMMARY (lambda=%s) ====" % best_lam)
print("refs: Naive=0.1979  ER=0.9023")
print(f"(A) consolidation+naive = {a_a:.4f}±{sa_a:.4f}  beats_naive_5pts={a_a-0.1979>=0.05}")
print(f"(B) consolidation+ER    = {a_b:.4f}±{sa_b:.4f}  beats_er_2pts={a_b-0.9023>=0.02}  delta={a_b-0.9023:+.4f}")
