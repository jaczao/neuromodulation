"""pt5 LEARNED-projection task-mask overlap — GATE BASIS (user-requested follow-up).

Same metrics as results/pt5_mask_overlap.py, but computed on the APPLIED GATE γ itself
(PT5_OVERLAP_BASIS=gate → offset 0), not on γ−parity. This is the apples-to-apples view against
iter-1's fixed {0,1} gate: does the learned gate form DISJOINT SUBNETWORKS like iter-1
(cos≈0, IoU≈0, frac≈1/T=0.2 for T=5), or does it keep everything ≈on (γ≈1 → cos≈1, frac≈1)?

Only the `out` config (modulate h0, h1, AND out/logits). 4 cells: gain {neuron, synapse} ×
{standalone buf-own, er-own}. class-IL, SGD, learned proj, gain_form=unbounded, seed 42, lr 1e-3,
ep 5, buffer 1000, --neuromod-er-task-id ON. Oracle caveat carries.

Run: PT5_DUMP_OVERLAP=1 PT5_OVERLAP_BASIS=gate uv run python results/pt5_mask_overlap_gate.py > results/pt5_mask_overlap_gate.log 2>&1
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PT5_DUMP_OVERLAP", "1")
os.environ.setdefault("PT5_OVERLAP_BASIS", "gate")
sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000

GRAN = {
    "neuron": dict(neuromod_granularity="neuron", neuromod_gain_layers="0,2,4"),
    "synapse": dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2,4"),
}
ARMS = {
    "buf-own": ("naive", dict(output_masking="loss", neuromod_meta_replay=True)),
    "er-own": ("er", dict(output_masking="none")),
}

results = {}
for gran in ("neuron", "synapse"):
    for arm, (method, extra) in ARMS.items():
        print("\n" + "#" * 90)
        print(f"# CELL  gain-{gran}  arm={arm}  layers=out(0,2,4)  method={method}  [GATE basis]")
        print("#" * 90, flush=True)
        c = CLConfig(
            seed=SEED, lr=LR, epochs_per_task=EP, optimizer="sgd",
            use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
            neuromod_target="activation", neuromod_gain_form="unbounded",
            neuromod_projection="learned", er_buffer_size=BUFFER, neuromod_er_task_id=True,
            **GRAN[gran], **extra,
        )
        a, f = cl_train(c, method, no_wandb=True, sequence=None)
        results[(gran, arm)] = (a, f)
        print(f">>> gain-{gran} {arm}  acc={a:.4f} forget={f:.4f}", flush=True)

print("\n\n" + "=" * 66)
print("pt5 GATE-basis overlap (metrics on γ itself) — do learned gates make subnetworks?")
print("=" * 66)
print(f"  {'granularity':11s} {'arm':8s} {'acc':>8s} {'forget':>8s}")
for gran in ("neuron", "synapse"):
    for arm in ("buf-own", "er-own"):
        a, f = results[(gran, arm)]
        print(f"  {gran:11s} {arm:8s} {a:>8.4f} {f:>8.4f}")
print("\nRead the [pt5 overlap] basis=gate lines above: iter-1 disjoint subnetworks would show "
      "cos≈0 / IoU≈0 / frac≈0.2; a near-parity mostly-on gate shows cos≈1 / IoU≈1 / frac≈1.")
