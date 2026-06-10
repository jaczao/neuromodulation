"""Iteration 7 (pt3): importance-gated plasticity, dual comparison.

Online per-parameter importance omega (sum of raw grad^2, never reset across tasks) gates each
parameter's effective LR: alpha_p = 1/(1+lambda*omega_p), applied to ALL params incl. the head,
so params important to past tasks are frozen. This is the retention signal iter6 lacked.

Tune lambda on the validation sequence (make_sequence(7)) only. Then 3 test seeds for:
  (A) standalone: importance+naive vs Naive 0.1979 (and vs naive+masked-loss 0.3777 to isolate value)
  (B) complementarity: importance+ER vs ER 0.9023
Bar set by iter6: a real mechanism must beat naive+masked-loss (~0.38) standalone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig
from prototype.data import make_sequence
from prototype.train import cl_train

SEEDS = (42, 43, 44)
NAIVE, NAIVE_LOSS, ER = 0.1979, 0.3777, 0.9023
IMP = dict(use_neuromod=True, neuromod_target="importance")

print("==== lambda sweep on validation sequence (seed=7, naive, class-IL) ====")
val = []
for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0):
    cfg = CLConfig(seed=7, lr=1e-3, epochs_per_task=5, neuromod_importance_lambda=lam, **IMP)
    acc, forget = cl_train(cfg, "naive", no_wandb=True, sequence=make_sequence(7))
    val.append((lam, acc, forget))
    print(f">>> lambda={lam:<8} val_acc={acc:.4f} forget={forget:.4f}\n")
best_lam = max(val, key=lambda r: r[1])[0]
print(f"BEST lambda (val) = {best_lam}\n")


def run3(tag, method, base, **extra):
    accs, forgets = [], []
    for s in SEEDS:
        acc, forget = cl_train(CLConfig(seed=s, neuromod_importance_lambda=best_lam, **base, **extra, **IMP),
                               method, no_wandb=True, sequence=None)
        accs.append(acc); forgets.append(forget)
    a, sa, f, sf = np.mean(accs), np.std(accs), np.mean(forgets), np.std(forgets)
    print(f">>> {tag:28s} acc={a:.4f}±{sa:.4f}  forget={f:.4f}±{sf:.4f}")
    return tag, a, sa


naive_cfg = dict(lr=1e-3, epochs_per_task=5)
er_cfg = dict(lr=3e-4, epochs_per_task=5, er_buffer_size=1000)
print("==== 3-seed test (class-IL) at best lambda ====")
r_a  = run3("importance + naive",          "naive", naive_cfg, output_masking="none")
r_al = run3("importance + naive + maskloss","naive", naive_cfg, output_masking="loss")
r_b  = run3("importance + ER",             "er",    er_cfg,    output_masking="none")

print("\n==== ITERATION 7 SUMMARY (class-IL, 3 seeds, lambda=%s) ====" % best_lam)
print(f"Naive={NAIVE}  naive+maskloss={NAIVE_LOSS}  ER={ER}")
print(f"(A) importance+naive          = {r_a[1]:.4f}±{r_a[2]:.4f}  beats_naive_5pts={r_a[1]-NAIVE>=0.05}")
print(f"    importance+naive+maskloss = {r_al[1]:.4f}±{r_al[2]:.4f}  beats_naive+maskloss={r_al[1]-NAIVE_LOSS>=0.0}")
print(f"(B) importance+ER             = {r_b[1]:.4f}±{r_b[2]:.4f}  beats_er_2pts={r_b[1]-ER>=0.02}  delta={r_b[1]-ER:+.4f}")
