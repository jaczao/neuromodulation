"""pt5 buffer-size sweep for the STANDALONE meta-loss (buf-meta-own) arm, out stage, class-IL.

In the standalone meta arm the buffer trains ONLY the gain gate P (the main net steps naive,
buffer-free), so buffer size changes the QUALITY of P's per-task calibration, not the backbone. This
sweeps 4 total buffer sizes for {sgd, adam} x {gain-neuron, gain-synapse}, out stage (learned P,
gain-neuron gate 0,2,4 / gain-synapse mask 0,2,4).

buffer=1000 is NOT re-run: those cells are the buf-meta-own `out` cells of results/pt5_out_bias.log
(same config: learned P, class-IL, meta_replay ON, er_task_id ON, gain_form unbounded, seed 42, lr
1e-3, ep 5). Merged in as constants. New sizes: {200, 500, 2000} => 12 runs.

Reference floors (no buffer, no gain): naive class-IL SGD 0.6296 / Adam 0.3894.

Run: uv run python results/pt5_out_meta_buffer_sweep.py   (redirect to .log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP = 42, 1e-3, 5
SIZES_NEW = [200, 500, 2000]
SIZE_EXISTING = 1000
OPTIMIZERS = ["sgd", "adam"]

GRAN = [
    ("gain-neuron", dict(neuromod_granularity="neuron", neuromod_gain_layers="0,2,4")),
    ("gain-synapse", dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2,4")),
]

# buffer=1000 buf-meta-own `out` cells from results/pt5_out_bias.log: (opt, gran) -> (acc, forget)
EXISTING = {
    ("sgd", "gain-neuron"): (0.9625, 0.0009), ("sgd", "gain-synapse"): (0.9894, 0.0006),
    ("adam", "gain-neuron"): (0.7229, 0.2647), ("adam", "gain-synapse"): (0.6907, 0.2913),
}
FLOOR = {"sgd": 0.6296, "adam": 0.3894}   # naive class-IL (no buffer, no gain)


def run(tag, size, optimizer, gkw):
    c = CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                 output_masking="loss", er_buffer_size=size,
                 use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                 neuromod_target="activation", neuromod_projection="learned",
                 neuromod_meta_replay=True, neuromod_er_task_id=True, **gkw)
    acc, forget = cl_train(c, "naive", no_wandb=True, sequence=None)
    print(f">>> {tag:44s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


# results[(opt, gran, size)] = (acc, forget)
results = {}
for (opt, gran), v in EXISTING.items():
    results[(opt, gran, SIZE_EXISTING)] = v
for optimizer in OPTIMIZERS:
    print(f"\n################ class-IL  optimizer={optimizer} ################", flush=True)
    for gran, gkw in GRAN:
        for size in SIZES_NEW:
            results[(optimizer, gran, size)] = run(
                f"[{optimizer}] {gran} buf={size}", size, optimizer, gkw)

ALL_SIZES = sorted(SIZES_NEW + [SIZE_EXISTING])

print("\n\n" + "=" * 84)
print("pt5 buffer-size sweep — standalone meta-loss (buf-meta-own), out stage, class-IL")
print("1 seed (42), lr=1e-3, ep=5. buffer trains ONLY the gain P (main net naive). *=merged (buf=1000)")
print("=" * 84)
for optimizer in OPTIMIZERS:
    print(f"\n--- optimizer={optimizer.upper()} ---   floor: naive (no buffer) = {FLOOR[optimizer]:.4f}")
    header = "  ".join(f"buf={s}" for s in ALL_SIZES)
    print(f"  {'mechanism':14s} {header}   (forgetting in parens)")
    for gran, _ in GRAN:
        cells = []
        for s in ALL_SIZES:
            a, f = results[(optimizer, gran, s)]
            mark = "*" if s == SIZE_EXISTING else " "
            cells.append(f"{a:.4f}{mark}")
        print(f"  {gran:14s} " + "  ".join(f"{c:>9s}" for c in cells))
        fcells = [f"({results[(optimizer, gran, s)][1]:.3f})" for s in ALL_SIZES]
        print(f"  {'':14s} " + "  ".join(f"{c:>9s}" for c in fcells))

print("\nbuffer trains ONLY the gate P (main net steps naive, never sees the buffer), so size changes "
      "per-task gate calibration, not the backbone: smaller buffer -> fewer distinct past-task exemplars "
      "(reservoir evicts the oldest tasks hardest) -> the meta-loss draws k=64 WITH replacement from a "
      "narrow pool, so P overfits those exemplars. Oracle + 1-seed caveats carry.")
