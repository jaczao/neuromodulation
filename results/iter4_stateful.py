"""Iteration 4 (stateful modulator): GRU modulator on weight_mask + surprise driver.

Variant=stateful (GRU cell, hidden state persisted across steps AND task boundaries,
never reset), on the best target so far (weight_mask, layer 2, full-rank) with the
default driver (surprise, since no driver won in Iteration 3). Matched conditions:
lr=1e-3, epochs_per_task=5 (Iter 2 best). Validation-sequence sanity run (seed=7) then
3 test seeds (42/43/44). A hidden=32 validation probe checks robustness to GRU size.

Accept: beats best-so-far (0.1992, Iter 1) by >=5 pts, OR matches with materially less
forgetting (baseline forgetting ~0.798).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig
from prototype.data import make_sequence
from prototype.train import cl_train

LR, EP = 1e-3, 5
BEST_SO_FAR = 0.1992


def mk(seed, hidden=64):
    return CLConfig(
        seed=seed, lr=LR, epochs_per_task=EP,
        use_neuromod=True, neuromod_target="weight_mask", neuromod_variant="stateful",
        neuromod_mask_layer=2, neuromod_mask_rank=0, neuromod_mask_init=0.99,
        neuromod_driver="surprise", neuromod_stateful_hidden=hidden,
    )


print("=== Iteration 4 stateful: validation sanity (hidden=64, make_sequence(7)) ===")
vacc, vforget = cl_train(mk(7, 64), "naive", no_wandb=True, sequence=make_sequence(7))
print(f">>> stateful h=64 VAL: acc={vacc:.4f} forget={vforget:.4f}\n")

print("=== hidden=32 validation probe ===")
vacc32, vforget32 = cl_train(mk(7, 32), "naive", no_wandb=True, sequence=make_sequence(7))
print(f">>> stateful h=32 VAL: acc={vacc32:.4f} forget={vforget32:.4f}\n")

print("=== Iteration 4 final: 3 test seeds (hidden=64) ===")
accs, forgets = [], []
for seed in (42, 43, 44):
    acc, forget = cl_train(mk(seed, 64), "naive", no_wandb=True, sequence=None)
    accs.append(acc)
    forgets.append(forget)
    print(f">>> stateful seed={seed}: acc={acc:.4f} forget={forget:.4f}\n")

print("=== ITERATION 4 RESULT ===")
print(f"stateful (GRU h=64) weight_mask + surprise | test seeds {accs}")
print(f"avg_final_acc = {np.mean(accs):.4f} +- {np.std(accs):.4f}")
print(f"forgetting    = {np.mean(forgets):.4f} +- {np.std(forgets):.4f}")
print(f"best-so-far (Iter 1) = {BEST_SO_FAR};  Naive = 0.1979 +- 0.0003")
print(f"beats_by_5pts = {np.mean(accs) - BEST_SO_FAR >= 0.05}")
