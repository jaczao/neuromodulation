"""pt5 LEARNED-projection task-mask OVERLAP (user-requested; needs a run because the learned P is
never checkpointed — nothing on disk to read).

For each cell we train the learned-projection gain model, then (env PT5_DUMP_OVERLAP, read in
cl_train) dump per-gated-layer overlap of the T=5 per-task gate rows γ_t: mean |deviation from
parity|, mean off-diagonal cosine similarity of the deviation vectors, and mean off-diagonal
Jaccard of each task's top-25%-|dev| engaged-unit set. cos≈0 / IoU≈0.25(=chance for top-25%) ⇒
disjoint-like allocation; cos→1 / IoU→1 ⇒ tasks reuse the same units.

Grid (8 cells): gain {neuron, synapse} × arm {standalone buf-own, er-own} × layers {hid=0,2, out=0,2,4}.
Fixed: class-IL, SGD, learned projection, gain_form=unbounded, seed 42, lr 1e-3, ep 5, buffer 1000,
--neuromod-er-task-id ON (per-task gating of buffered samples). Oracle caveat (task id at eval).

Run: PT5_DUMP_OVERLAP=1 uv run python results/pt5_mask_overlap.py   (redirect to .log)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PT5_DUMP_OVERLAP", "1")
sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000

# (granularity, layer-arg) -> the layer flag differs by granularity (gain_layers vs mask_layers)
GRAN = {
    "neuron": lambda layers: dict(neuromod_granularity="neuron", neuromod_gain_layers=layers),
    "synapse": lambda layers: dict(neuromod_granularity="synapse", neuromod_mask_layers=layers),
}
LAYERS = {"hid": "0,2", "out": "0,2,4"}
# arm -> (method, extra config). standalone = naive + modulator-only replay meta-loss; er = +ER.
ARMS = {
    "buf-own": ("naive", dict(output_masking="loss", neuromod_meta_replay=True)),
    "er-own": ("er", dict(output_masking="none")),
}

results = {}
for gran in ("neuron", "synapse"):
    for arm, (method, extra) in ARMS.items():
        for lname, layers in LAYERS.items():
            tag = f"{gran:7s} {arm:7s} {lname}"
            print("\n" + "#" * 90)
            print(f"# CELL  gain-{gran}  arm={arm}  layers={lname}({layers})  method={method}")
            print("#" * 90, flush=True)
            c = CLConfig(
                seed=SEED, lr=LR, epochs_per_task=EP, optimizer="sgd",
                use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                neuromod_target="activation", neuromod_gain_form="unbounded",
                neuromod_projection="learned", er_buffer_size=BUFFER,
                neuromod_er_task_id=True,
                **GRAN[gran](layers), **extra,
            )
            a, f = cl_train(c, method, no_wandb=True, sequence=None)
            results[(gran, arm, lname)] = (a, f)
            print(f">>> {tag}  acc={a:.4f} forget={f:.4f}", flush=True)

print("\n\n" + "=" * 72)
print("pt5 LEARNED-proj mask overlap — cell accuracy summary (overlap stats inline above)")
print("=" * 72)
print(f"  {'granularity':11s} {'arm':8s} {'layers':6s} {'acc':>8s} {'forget':>8s}")
for gran in ("neuron", "synapse"):
    for arm in ("buf-own", "er-own"):
        for lname in ("hid", "out"):
            a, f = results[(gran, arm, lname)]
            print(f"  {gran:11s} {arm:8s} {lname:6s} {a:>8.4f} {f:>8.4f}")
print("\nSee each cell's [pt5 overlap] lines above for per-layer |dev| / cos / IoU@top25 and the "
      "full 5x5 per-task cosine matrix. Oracle caveat carries.")
