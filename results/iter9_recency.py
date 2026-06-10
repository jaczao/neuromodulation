"""Iteration 9 (pt3): retention driver (per-class recency) on the logit calibrator, dual comparison.

Gives iter6's logit calibrator the retention signal it lacked: a per-class presence EMA fed to the
modulator. Tests whether a retention DRIVER (rather than masked loss) rescues logit calibration.
Standalone (naive) and +ER (replay). 3 test seeds, class-IL.
Refs: Naive 0.1979, ER 0.9023.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig
from prototype.train import cl_train

SEEDS = (42, 43, 44)
REC = dict(use_neuromod=True, neuromod_target="logit", neuromod_driver="recency")


def run3(tag, method, **base):
    accs = [cl_train(CLConfig(seed=s, **base, **REC), method, no_wandb=True, sequence=None)[0] for s in SEEDS]
    a, sa = np.mean(accs), np.std(accs)
    print(f">>> {tag:26s} acc={a:.4f}±{sa:.4f}")
    return a, sa


print("==== Iteration 9 logit+recency driver (3 seeds, class-IL) ====")
a_a, sa_a = run3("logit+recency + naive", "naive", lr=1e-3, epochs_per_task=5)
a_b, sa_b = run3("logit+recency + ER",    "er",    lr=3e-4, epochs_per_task=5, er_buffer_size=1000)

print("\n==== ITERATION 9 SUMMARY ====")
print("refs: Naive=0.1979  ER=0.9023")
print(f"(A) logit+recency+naive = {a_a:.4f}±{sa_a:.4f}  beats_naive_5pts={a_a-0.1979>=0.05}")
print(f"(B) logit+recency+ER    = {a_b:.4f}±{sa_b:.4f}  beats_er_2pts={a_b-0.9023>=0.02}  delta={a_b-0.9023:+.4f}")
