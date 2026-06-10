"""Iteration 6 (pt3): logit calibration (FiLM on the output logits), dual comparison.

Mechanism: a context-driven per-sample FiLM on the 10 logits, logits'=(1+gamma(x))*logits+beta(x).
It reaches the head directly. Trained on the current task alone it just favors current classes,
so per the SPEC it is paired with a retention term (output_masking='loss', lever B). Class-IL
(the real benchmark). 3 test seeds (42/43/44).

Comparisons:
  (A) standalone vs Naive (0.1979): isolate the modulator's value over masked-loss alone, so we
      report naive+loss (lever B, no neuromod), logit+none (no retention), logit+loss.
  (B) complementarity vs ER (0.9023): logit+ER (frozen ER config), class-IL.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig
from prototype.train import cl_train

SEEDS = (42, 43, 44)
NAIVE, ER = 0.1979, 0.9023
LOGIT = dict(use_neuromod=True, neuromod_target="logit")
ER_CFG = dict(lr=3e-4, epochs_per_task=5, er_buffer_size=1000)


def run3(tag, method, base, **extra):
    accs, forgets = [], []
    for s in SEEDS:
        cfg = CLConfig(seed=s, **base, **extra)
        acc, forget = cl_train(cfg, method, no_wandb=True, sequence=None)
        accs.append(acc); forgets.append(forget)
    a, sa, f, sf = np.mean(accs), np.std(accs), np.mean(forgets), np.std(forgets)
    print(f">>> {tag:30s} acc={a:.4f}±{sa:.4f}  forget={f:.4f}±{sf:.4f}")
    return tag, a, sa, f, sf


naive_cfg = dict(lr=1e-3, epochs_per_task=5)
print("==== (A) standalone (class-IL, naive side) ====")
r_nl  = run3("naive + masked-loss",        "naive", naive_cfg, output_masking="loss")
r_ln  = run3("logit (no retention)",       "naive", naive_cfg, output_masking="none", **LOGIT)
r_ll  = run3("logit + masked-loss",        "naive", naive_cfg, output_masking="loss", **LOGIT)
print("\n==== (B) complementarity (class-IL, ER side) ====")
r_le  = run3("logit + ER",                 "er",    ER_CFG,    output_masking="none", **LOGIT)

print("\n==== ITERATION 6 SUMMARY (class-IL, 3 seeds) ====")
print(f"Naive (frozen)            = {NAIVE:.4f}")
print(f"ER (frozen)               = {ER:.4f}")
for tag, a, sa, f, sf in (r_nl, r_ln, r_ll, r_le):
    print(f"{tag:26s} acc={a:.4f}±{sa:.4f}  forget={f:.4f}±{sf:.4f}")
print("\n-- decision --")
print(f"(A) logit+loss vs Naive : {r_ll[1]:.4f} vs {NAIVE:.4f}  beats_naive_5pts={r_ll[1]-NAIVE>=0.05}")
print(f"    logit+loss vs naive+loss (isolate neuromod): {r_ll[1]:.4f} vs {r_nl[1]:.4f}  delta={r_ll[1]-r_nl[1]:+.4f}")
print(f"(B) logit+ER vs ER      : {r_le[1]:.4f} vs {ER:.4f}  beats_er_2pts={r_le[1]-ER>=0.02}  delta={r_le[1]-ER:+.4f}")
