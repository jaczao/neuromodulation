"""Iteration 8 (pt3): task-inferred routing (simplified HAT / lever C), dual comparison.

Masked-loss main net + a task-inference net g(x); at eval, route each input to the inferred
task's output classes. Standalone (naive): g trained sequentially, no replay. +ER: a shared
reservoir buffer trains BOTH the main net and g (so g need not forget). The binding constraint is
g's routing accuracy, reported directly.

Refs: Naive 0.1979, naive+masked-loss 0.3777, ER 0.9023, task-IL oracle (perfect routing) 0.9286.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig
from prototype.train import cl_train

SEEDS = (42, 43, 44)
TR = dict(use_neuromod=True, neuromod_target="task_route")


def run3(tag, method, **base):
    accs, forgets = [], []
    for s in SEEDS:
        acc, forget = cl_train(CLConfig(seed=s, **base, **TR), method, no_wandb=True, sequence=None)
        accs.append(acc); forgets.append(forget)
    a, sa = np.mean(accs), np.std(accs)
    print(f">>> {tag:24s} acc={a:.4f}±{sa:.4f}  forget={np.mean(forgets):.4f}")
    return tag, a, sa


print("==== Iteration 8 task-inferred routing (3 seeds, class-IL) ====")
r_a = run3("task_route + naive", "naive", lr=1e-3, epochs_per_task=5)
r_b = run3("task_route + ER",    "er",    lr=3e-4, epochs_per_task=5, er_buffer_size=1000)

print("\n==== ITERATION 8 SUMMARY ====")
print("refs: Naive=0.1979  naive+maskloss=0.3777  ER=0.9023  task-IL oracle=0.9286")
print(f"(A) task_route+naive = {r_a[1]:.4f}±{r_a[2]:.4f}  beats_naive_5pts={r_a[1]-0.1979>=0.05}  beats_maskloss={r_a[1]-0.3777>=0.0}")
print(f"(B) task_route+ER    = {r_b[1]:.4f}±{r_b[2]:.4f}  beats_er_2pts={r_b[1]-0.9023>=0.02}  delta={r_b[1]-0.9023:+.4f}")
print("(routing accuracy printed per run above as 'mean routing acc'; final value in [task_route debug])")
