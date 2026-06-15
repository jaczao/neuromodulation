"""Direct-gain modulator (no low-dim signal + projection; the neuromod net is a direct
image->gain head of shape 784 x layer_width), in BOTH regimes, all 4 gate configs.

Variant of the pt1 GainModulator: instead of signal_net 784->64->k=8 then a fixed projection
P_l (k->hidden), each gated layer gets its own Linear(784 -> layer_width) gain head (zero-init,
(1+m)*h FiLM, vanilla parity at init). Gate configs:
  last_hidden          : gate h2
  two_hidden           : gate h1, h2          (the pt1 layout)
  last_hidden_output   : gate h2, logits
  two_hidden_output    : gate h1, h2, logits  (logits = the pt3 class-IL bottleneck)

Standard (full MNIST): test acc vs vanilla 0.9796 / pt1 gain 0.9806 (frozen std config).
Class-IL Split MNIST: (A) standalone vs Naive 0.1979, (B) +ER vs ER 0.9023 (frozen CL configs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import CLConfig, StandardConfig
from prototype.train import cl_train, train_standard

SEEDS = (42, 43, 44)
GATES = ("last_hidden", "two_hidden", "last_hidden_output", "two_hidden_output")
DG = dict(use_neuromod=True, neuromod_target="direct_gain")

# Frozen references (same configs, already published).
VAN_STD, GAIN_STD = 0.9796, 0.9806
NAIVE, ER = 0.1979, 0.9023


def std3(gate):
    accs = []
    for s in SEEDS:
        cfg = StandardConfig(seed=s, lr=3e-4, epochs=20, batch_size=64,
                             neuromod_gain_gate=gate, **DG)
        _, t = train_standard(cfg, no_wandb=True)
        accs.append(t)
    return float(np.mean(accs)), float(np.std(accs))


def cl3(gate, method, base):
    accs, forgets = [], []
    for s in SEEDS:
        cfg = CLConfig(seed=s, neuromod_gain_gate=gate, **base, **DG)
        a, f = cl_train(cfg, method, no_wandb=True, sequence=None)
        accs.append(a); forgets.append(f)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(forgets))


naive_base = dict(lr=1e-3, epochs_per_task=5, output_masking="none")
er_base = dict(lr=3e-4, epochs_per_task=5, er_buffer_size=1000, output_masking="none")

print("######## STANDARD (full MNIST) ########")
std_rows = []
for g in GATES:
    a, sa = std3(g)
    std_rows.append((g, a, sa))
    print(f">>> std {g:22s} test_acc={a:.4f}±{sa:.4f}")

print("\n######## CLASS-IL Split MNIST: (A) standalone (naive) ########")
cln_rows = []
for g in GATES:
    a, sa, f = cl3(g, "naive", naive_base)
    cln_rows.append((g, a, sa, f))
    print(f">>> naive {g:22s} acc={a:.4f}±{sa:.4f} forget={f:.4f}")

print("\n######## CLASS-IL Split MNIST: (B) complementarity (+ER) ########")
cle_rows = []
for g in GATES:
    a, sa, f = cl3(g, "er", er_base)
    cle_rows.append((g, a, sa, f))
    print(f">>> er {g:22s} acc={a:.4f}±{sa:.4f} forget={f:.4f}")

print("\n==== DIRECT-GAIN SUMMARY (3 seeds 42/43/44) ====")
print(f"[standard] vanilla={VAN_STD:.4f}  pt1 gain={GAIN_STD:.4f}")
for g, a, sa in std_rows:
    print(f"  std  {g:22s} {a:.4f}±{sa:.4f}  (vs vanilla {a-VAN_STD:+.4f})")
print(f"[class-IL] Naive={NAIVE:.4f}  ER={ER:.4f}")
for g, a, sa, f in cln_rows:
    print(f"  A naive {g:22s} {a:.4f}±{sa:.4f}  (vs Naive {a-NAIVE:+.4f})  forget={f:.4f}")
for g, a, sa, f in cle_rows:
    print(f"  B +ER   {g:22s} {a:.4f}±{sa:.4f}  (vs ER {a-ER:+.4f})  forget={f:.4f}")
