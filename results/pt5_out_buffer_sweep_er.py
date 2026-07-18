"""pt5 buffer-size sweep, out stage, class-IL — (A) rerun the standalone buffer=1000 cells fresh
(replacing the merged constants in results/pt5_out_meta_buffer_sweep.log), and (B) the SAME sweep for
the er-own arm, all 4 sizes run fresh (incl. buffer=1000, which also exists in pt5_out_bias.log).

Two arms, KEY difference in what the buffer does:
  - buf-meta-own (standalone): buffer trains ONLY the gain P (main net naive). Size -> gate
    calibration quality only. {200,500,2000} merged from pt5_out_meta_buffer_sweep.log; 1000 rerun.
  - er-own (+ER): buffer feeds the BACKBONE (replay) AND each replayed sample is routed through its
    own P[j] (er_task_id). Size -> what the backbone re-learns. All 4 sizes run fresh.

learned projection, out stage (gain-neuron gate 0,2,4 / gain-synapse mask 0,2,4), seed 42, lr 1e-3,
ep 5, gain_form unbounded, er_task_id ON. Cross-checks: standalone 1000 should reproduce
pt5_out_meta_buffer_sweep (sgd-neu 0.9625, sgd-syn 0.9894, adam-neu 0.7229, adam-syn 0.6907); er-own
1000 should reproduce pt5_out_bias (sgd-neu 0.9466, sgd-syn 0.7347, adam-neu 0.9875, adam-syn 0.9915).

Run: uv run python results/pt5_out_buffer_sweep_er.py   (redirect to .log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP = 42, 1e-3, 5
SIZES = [200, 500, 1000, 2000]
OPTIMIZERS = ["sgd", "adam"]

GRAN = [
    ("gain-neuron", dict(neuromod_granularity="neuron", neuromod_gain_layers="0,2,4")),
    ("gain-synapse", dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2,4")),
]

# standalone {200,500,2000} merged from results/pt5_out_meta_buffer_sweep.log (identical config).
# Only buffer=1000 is rerun for the standalone arm (the * cells the user asked to rerun).
STANDALONE_MERGE = {
    ("sgd", "gain-neuron", 200): (0.9459, 0.0163), ("sgd", "gain-neuron", 500): (0.9570, 0.0056),
    ("sgd", "gain-neuron", 2000): (0.9591, 0.0045),
    ("sgd", "gain-synapse", 200): (0.9890, 0.0009), ("sgd", "gain-synapse", 500): (0.9881, 0.0020),
    ("sgd", "gain-synapse", 2000): (0.9899, 0.0004),
    ("adam", "gain-neuron", 200): (0.6109, 0.3768), ("adam", "gain-neuron", 500): (0.7227, 0.2650),
    ("adam", "gain-neuron", 2000): (0.7256, 0.2620),
    ("adam", "gain-synapse", 200): (0.7199, 0.2621), ("adam", "gain-synapse", 500): (0.7108, 0.2712),
    ("adam", "gain-synapse", 2000): (0.7080, 0.2740),
}
# cross-check refs at buffer=1000
CK_STANDALONE = {("sgd", "gain-neuron"): 0.9625, ("sgd", "gain-synapse"): 0.9894,
                 ("adam", "gain-neuron"): 0.7229, ("adam", "gain-synapse"): 0.6907}
CK_ER = {("sgd", "gain-neuron"): 0.9466, ("sgd", "gain-synapse"): 0.7347,
         ("adam", "gain-neuron"): 0.9875, ("adam", "gain-synapse"): 0.9915}
FLOOR_NAIVE = {"sgd": 0.6296, "adam": 0.3894}
FLOOR_ER = {"sgd": 0.7226, "adam": 0.9053}


def run(tag, arm, size, optimizer, gkw):
    replay = arm == "er-own"
    c = CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                 output_masking=("none" if replay else "loss"), er_buffer_size=size,
                 use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                 neuromod_target="activation", neuromod_projection="learned",
                 neuromod_meta_replay=(not replay), neuromod_er_task_id=True, **gkw)
    acc, forget = cl_train(c, ("er" if replay else "naive"), no_wandb=True, sequence=None)
    print(f">>> {tag:46s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


standalone = dict(STANDALONE_MERGE)
er = {}
for optimizer in OPTIMIZERS:
    print(f"\n################ class-IL  optimizer={optimizer} ################", flush=True)
    for gran, gkw in GRAN:
        # (A) rerun standalone buffer=1000
        standalone[(optimizer, gran, 1000)] = run(
            f"[{optimizer}] {gran} standalone buf=1000 (RERUN)", "buf-meta-own", 1000, optimizer, gkw)
        # (B) er-own, all sizes
        for size in SIZES:
            er[(optimizer, gran, size)] = run(
                f"[{optimizer}] {gran} er-own buf={size}", "er-own", size, optimizer, gkw)


def table(title, data, floor, ck):
    print(f"\n{title}")
    for optimizer in OPTIMIZERS:
        print(f"  --- {optimizer.upper()} ---  floor={floor[optimizer]:.4f}   "
              f"(cross-check buf=1000: neu {ck[(optimizer,'gain-neuron')]:.4f} / "
              f"syn {ck[(optimizer,'gain-synapse')]:.4f})")
        print(f"  {'mechanism':14s} " + "  ".join(f"buf={s:<5d}" for s in SIZES))
        for gran, _ in GRAN:
            accs = "  ".join(f"{data[(optimizer, gran, s)][0]:.4f}" for s in SIZES)
            fs = "  ".join(f"({data[(optimizer, gran, s)][1]:.3f})" for s in SIZES)
            print(f"  {gran:14s} {accs}")
            print(f"  {'':14s} {fs}")


print("\n\n" + "=" * 88)
print("pt5 buffer-size sweep, out stage, class-IL — standalone (gate-only) vs er-own (backbone replay)")
print("1 seed (42), lr=1e-3, ep=5. standalone {200,500,2000} merged; 1000 rerun. er-own all fresh.")
print("=" * 88)
table("STANDALONE (buf-meta-own): buffer trains ONLY the gate P (main net naive)",
      standalone, FLOOR_NAIVE, CK_STANDALONE)
table("ER-OWN (+ER): buffer feeds the BACKBONE (replay), each replayed sample via its own P[j]",
      er, FLOOR_ER, CK_ER)

print("\n--- CROSS-CHECK (fresh buf=1000 vs prior logs) ---")
for opt in OPTIMIZERS:
    for gran, _ in GRAN:
        s = standalone[(opt, gran, 1000)][0]; sr = CK_STANDALONE[(opt, gran)]
        e = er[(opt, gran, 1000)][0]; er_ = CK_ER[(opt, gran)]
        print(f"  {opt:4s} {gran:12s} standalone {s:.4f} vs {sr:.4f} "
              f"{'OK' if abs(s-sr)<5e-4 else '**DIFF**'} | er-own {e:.4f} vs {er_:.4f} "
              f"{'OK' if abs(e-er_)<5e-4 else '**DIFF**'}")

print("\nStandalone: buffer only calibrates the gate (saturates fast, SGD ~insensitive). er-own: buffer "
      "drives what the backbone replays, so its size effect is the classic replay curve. Oracle + 1-seed.")
