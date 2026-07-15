"""pt5 Iteration 3 follow-up 3 — modulator-only replay meta-loss for GAIN (user-requested).

Gain is a FORWARD target: its learned P sits in model.parameters() and iter-3 trained it via the
MAIN loss (current task only, no retention signal). This adds `--neuromod-meta-replay` support for
gain: a SEPARATE optimizer trains ONLY P on a modulator-only replay meta-loss, while the main net
trains naive on the current task (P excluded from the main optimizer). Because gain gates the
FORWARD (unlike plasticity's grad gate), the meta-loss is PER-TASK: each seen task j's samples
(fresh current batch for j=t, buffer samples for j<t) are forwarded under ITS OWN gate P[j] and the
losses summed; only P[j] gets a gradient (the one-hot zeroes the other rows). The oracle selects
P[i] at eval (as everywhere in pt5).

Both gain mechs x {class-IL, task-IL} x {SGD, Adam}, standalone (naive), 1 seed (42), lr=1e-3, ep=5,
buffer=1000. OFF (iter-3, P trained by the main loss) values are the fixed reference. Oracle caveat.

Run: uv run python results/pt5_iter3_gain_metareplay.py   (redirect to .log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000

# iter-3 standalone (meta_replay OFF) reference (acc, forget), from results/pt5_iter3.log
OFF = {
    ("classil", "sgd", "gain-neuron"): (0.6311, 0.1242), ("classil", "sgd", "gain-synapse"): (0.6295, 0.1239),
    ("classil", "adam", "gain-neuron"): (0.3770, 0.5883), ("classil", "adam", "gain-synapse"): (0.4202, 0.5668),
    ("taskil", "sgd", "gain-neuron"): (0.9768, 0.0017), ("taskil", "sgd", "gain-synapse"): (0.9771, 0.0016),
    ("taskil", "adam", "gain-neuron"): (0.9102, 0.0860), ("taskil", "adam", "gain-synapse"): (0.9665, 0.0306),
}
MECH = {"gain-neuron": dict(neuromod_granularity="neuron", neuromod_gain_layers="0,2"),
        "gain-synapse": dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2")}


def run(metric, opt, mech):
    masking = "taskil" if metric == "taskil" else "loss"
    c = CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=opt, output_masking=masking,
                 use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                 neuromod_target="activation", neuromod_gain_form="unbounded",
                 neuromod_projection="learned", neuromod_meta_replay=True, er_buffer_size=BUFFER,
                 **MECH[mech])
    a, f = cl_train(c, "naive", no_wandb=True, sequence=None)
    print(f">>> [{metric} {opt}] {mech} meta_replay=ON  acc={a:.4f} forget={f:.4f}", flush=True)
    return a, f


ON = {}
for metric in ("classil", "taskil"):
    for opt in ("sgd", "adam"):
        for mech in ("gain-neuron", "gain-synapse"):
            ON[(metric, opt, mech)] = run(metric, opt, mech)

print("\n\n" + "=" * 82)
print("pt5 ITER3 — GAIN modulator-only replay meta-loss (does a retention signal help gain?)")
print("=" * 82)
for metric in ("classil", "taskil"):
    for opt in ("sgd", "adam"):
        print(f"\n--- {metric.upper()}  {opt.upper()} (standalone / naive) ---")
        print(f"  {'mechanism':14s} {'OFF acc':>8s} {'ON acc':>8s} {'dAcc':>8s} "
              f"{'OFF forg':>9s} {'ON forg':>8s} {'dForg':>8s}")
        for mech in ("gain-neuron", "gain-synapse"):
            oa, of = OFF[(metric, opt, mech)]; na, nf = ON[(metric, opt, mech)]
            print(f"  {mech:14s} {oa:>8.4f} {na:>8.4f} {na-oa:>+8.4f} "
                  f"{of:>9.4f} {nf:>8.4f} {nf-of:>+8.4f}")
print("\nOFF = iter-3 (gain P trained by the MAIN loss, current task only); ON = per-task "
      "modulator-only replay meta-loss trains ONLY P, main net stays naive. Oracle caveat carries.")
