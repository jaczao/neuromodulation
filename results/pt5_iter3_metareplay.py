"""pt5 Iteration 3 follow-up — standalone modulator-only replay meta-loss (user-requested).

The iter-3 standalone (neurom) plasticity meta-loop trained P on the CURRENT batch only (no
retention signal). This adds a modulator-only replay buffer (`--neuromod-meta-replay`): a reservoir
of past examples augments the META-loss (which trains ONLY P) while the MAIN net stays naive. This
is the SPEC's iter-3 "modulator-only replay meta-loss" for the standalone condition. Question: does
giving the learned plasticity P a real retention signal rescue standalone plasticity?

Standalone only (method=naive); both plasticity mechs x {class-IL, task-IL} x {SGD, Adam}, 1 seed
(42), lr=1e-3, ep=5, buffer=1000. OFF (iter-3, meta on current batch only) values are the fixed
reference (from results/pt5_iter3.log). class-IL uses output_masking='loss', task-IL 'taskil'.

Run: uv run python results/pt5_iter3_metareplay.py   (redirect to .log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000

# iter-3 standalone (meta_replay OFF) reference: (acc, forget), from results/pt5_iter3.log
OFF = {
    ("classil", "sgd", "plast-neuron"): (0.6456, 0.1285), ("classil", "sgd", "plast-synapse"): (0.6430, 0.1130),
    ("classil", "adam", "plast-neuron"): (0.3866, 0.5841), ("classil", "adam", "plast-synapse"): (0.3820, 0.5933),
    ("taskil", "sgd", "plast-neuron"): (0.9739, 0.0008), ("taskil", "sgd", "plast-synapse"): (0.9737, 0.0005),
    ("taskil", "adam", "plast-neuron"): (0.9092, 0.0875), ("taskil", "adam", "plast-synapse"): (0.9191, 0.0786),
}
MECH = {"plast-neuron": dict(neuromod_granularity="neuron", neuromod_plasticity_layers="0,2,4",
                             neuromod_plasticity_scope="both"),
        "plast-synapse": dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2")}


def run(metric, opt, mech):
    masking = "taskil" if metric == "taskil" else "loss"
    c = CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=opt, output_masking=masking,
                 use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                 neuromod_target="plasticity", neuromod_projection="learned",
                 neuromod_meta_replay=True, er_buffer_size=BUFFER, **MECH[mech])
    a, f = cl_train(c, "naive", no_wandb=True, sequence=None)
    print(f">>> [{metric} {opt}] {mech} meta_replay=ON  acc={a:.4f} forget={f:.4f}", flush=True)
    return a, f


ON = {}
for metric in ("classil", "taskil"):
    for opt in ("sgd", "adam"):
        for mech in ("plast-neuron", "plast-synapse"):
            ON[(metric, opt, mech)] = run(metric, opt, mech)

print("\n\n" + "=" * 78)
print("pt5 ITER3 — standalone modulator-only replay meta-loss (does a retention signal help?)")
print("=" * 78)
for metric in ("classil", "taskil"):
    for opt in ("sgd", "adam"):
        print(f"\n--- {metric.upper()}  {opt.upper()} (standalone / naive) ---")
        print(f"  {'mechanism':14s} {'OFF acc':>8s} {'ON acc':>8s} {'dAcc':>8s} "
              f"{'OFF forg':>9s} {'ON forg':>8s} {'dForg':>8s}")
        for mech in ("plast-neuron", "plast-synapse"):
            oa, of = OFF[(metric, opt, mech)]; na, nf = ON[(metric, opt, mech)]
            print(f"  {mech:14s} {oa:>8.4f} {na:>8.4f} {na-oa:>+8.4f} "
                  f"{of:>9.4f} {nf:>8.4f} {nf-of:>+8.4f}")
print("\nOFF = iter-3 (meta-loss on current batch only); ON = modulator-only replay buffer feeds the "
      "meta-loss (main net stays naive). Oracle caveat carries.")
