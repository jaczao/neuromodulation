"""pt5 LEARNED-projection task-mask overlap — ADAM, out config, GAIN only (user-requested).

Same overlap study as results/pt5_mask_overlap{,_gate}.py but under ADAM. The dump prints BOTH bases
per layer: 'dev' (= m, subtract the parity gate γ(0)=1) and 'gate' (the applied gate itself, 1+m),
so we can read whether the learned gates form iter-1-style disjoint subnetworks (gate basis:
cos≈0/frac≈0.2) or stay a near-parity mostly-on gate (cos≈1/frac≈1).

Mechanisms: gain {neuron, synapse}. Arms: standalone buf-own (naive + modulator-only replay
meta-loss) and er-own (+ER, per-task gating). out config = modulate the h0/h1/out layers (gate/
mask_layers 0,2,4). Plus Adam naive/ER baselines (no neuromod). class-IL, learned proj,
gain_form=unbounded, seed 42, lr 1e-3, ep 5, buffer 1000, er_task_id ON. Oracle caveat (task id at eval).

Run: PT5_DUMP_OVERLAP=1 uv run python results/pt5_mask_overlap_adam.py > results/pt5_mask_overlap_adam.log 2>&1
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PT5_DUMP_OVERLAP", "1")   # both bases by default
sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000

MECH = {
    "gain-neuron":  dict(neuromod_target="activation", neuromod_granularity="neuron",
                         neuromod_gain_form="unbounded", neuromod_gain_layers="0,2,4"),
    "gain-synapse": dict(neuromod_target="activation", neuromod_granularity="synapse",
                         neuromod_gain_form="unbounded", neuromod_mask_layers="0,2,4"),
}
ARMS = {
    "buf-own": ("naive", dict(output_masking="loss", neuromod_meta_replay=True)),
    "er-own": ("er", dict(output_masking="none")),
}

results = {}

# --- Adam baselines (no neuromod) ---
print("\n" + "#" * 90 + "\n# BASELINES (Adam, no neuromod)\n" + "#" * 90, flush=True)
a, f = cl_train(CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer="adam",
                         output_masking="loss"), "naive", no_wandb=True)
results["naive"] = (a, f); print(f">>> naive-Adam (masked-loss)  acc={a:.4f} forget={f:.4f}", flush=True)
a, f = cl_train(CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer="adam",
                         output_masking="none", er_buffer_size=BUFFER), "er", no_wandb=True)
results["er"] = (a, f); print(f">>> ER-Adam (buffer={BUFFER})       acc={a:.4f} forget={f:.4f}", flush=True)

# --- neuromod cells ---
for mech in ("gain-neuron", "gain-synapse"):
    for arm, (method, extra) in ARMS.items():
        print("\n" + "#" * 90)
        print(f"# CELL  {mech}  arm={arm}  out(0,2,4)  method={method}  [ADAM]")
        print("#" * 90, flush=True)
        c = CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer="adam",
                     use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                     neuromod_projection="learned", er_buffer_size=BUFFER, neuromod_er_task_id=True,
                     **MECH[mech], **extra)
        a, f = cl_train(c, method, no_wandb=True, sequence=None)
        results[(mech, arm)] = (a, f)
        print(f">>> {mech} {arm}  acc={a:.4f} forget={f:.4f}", flush=True)

print("\n\n" + "=" * 60)
print("pt5 ADAM overlap study — accuracy summary")
print("=" * 60)
print(f"  baseline naive-Adam : acc={results['naive'][0]:.4f} forget={results['naive'][1]:.4f}")
print(f"  baseline ER-Adam    : acc={results['er'][0]:.4f} forget={results['er'][1]:.4f}")
print(f"\n  {'mechanism':14s} {'arm':8s} {'acc':>8s} {'forget':>8s}")
for mech in ("gain-neuron", "gain-synapse"):
    for arm in ("buf-own", "er-own"):
        a, f = results[(mech, arm)]
        print(f"  {mech:14s} {arm:8s} {a:>8.4f} {f:>8.4f}")
print("\nPer-cell [pt5 overlap] lines above give dev(=m) and gate(=1+m/α) metrics per layer. "
      "iter-1 disjoint subnets => gate basis cos~0/IoU~0/frac~0.2; mostly-on gate => cos~1/frac~1.")
