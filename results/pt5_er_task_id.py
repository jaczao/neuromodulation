"""pt5 --neuromod-er-task-id study (user-requested feature): under +ER, apply each replayed
sample's OWN task mask P[j] instead of the whole current+replay batch under the current task P[t].

Forward targets (gain) gate the forward per task (split the batch, forward each subset under P[j],
scatter the logits back); plasticity gates each task's gradient per task (split the batch, backward
each subset weighted n_j/N, gate by P[j], accumulate, one step). Default OFF = parity (whole ER batch
under P[t]). No sparsity reg (neuromod_sparsity_lambda=0, default).

Cells: er OFF vs er ON, disjoint projection, 1 seed (42), lr=1e-3, ep=5, buffer=1000, class-IL.
ER uses output_masking='none' (pt5 convention: masked-loss OFF for replay cells, matching iter-1/3 —
ER calibrates the head via replay, no masked loss). gain-neuron {SGD,Adam}, gain-synapse {SGD},
plast-neuron {SGD}. OFF for gain-neuron reproduces the iter-1 er+gain baselines (SGD 0.8264 /
Adam 0.9901). Oracle caveat carries.

Run: uv run python results/pt5_er_task_id.py   (redirect to .log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000

MECH = {
    "gain-neuron": dict(neuromod_target="activation", neuromod_gain_form="unbounded",
                        neuromod_granularity="neuron", neuromod_gain_layers="0,2"),
    "gain-synapse": dict(neuromod_target="activation", neuromod_gain_form="unbounded",
                         neuromod_granularity="synapse", neuromod_mask_layers="0,2"),
    "plast-neuron": dict(neuromod_target="plasticity", neuromod_granularity="neuron",
                         neuromod_plasticity_layers="0,2,4", neuromod_plasticity_scope="both"),
}
# which optimizers to run per mechanism
CELLS = [("gain-neuron", "sgd"), ("gain-neuron", "adam"),
         ("gain-synapse", "sgd"), ("plast-neuron", "sgd")]


def run(mech, opt, er_task_id):
    # ER (replay) cells use masked-loss OFF ('none') per the pt5 convention (iter-1/iter-3): the
    # replay-calibrated head does class-IL 10-way, no masked training loss.
    c = CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=opt, output_masking="none",
                 use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                 neuromod_projection="disjoint", er_buffer_size=BUFFER,
                 neuromod_er_task_id=er_task_id, **MECH[mech])
    a, f = cl_train(c, "er", no_wandb=True, sequence=None)
    print(f">>> [{opt}] {mech} er_task_id={'ON ' if er_task_id else 'OFF'}  "
          f"acc={a:.4f} forget={f:.4f}", flush=True)
    return a, f


RES = {}
for mech, opt in CELLS:
    RES[(mech, opt, False)] = run(mech, opt, False)
    RES[(mech, opt, True)] = run(mech, opt, True)

print("\n\n" + "=" * 78)
print("pt5 --neuromod-er-task-id: +ER OFF (batch under P[t]) vs ON (each sample under its P[j])")
print("=" * 78)
print(f"  {'mechanism':13s} {'opt':4s} {'OFF acc':>8s} {'ON acc':>8s} {'dAcc':>8s} "
      f"{'OFF forg':>9s} {'ON forg':>8s} {'dForg':>8s}")
for mech, opt in CELLS:
    oa, of = RES[(mech, opt, False)]; na, nf = RES[(mech, opt, True)]
    print(f"  {mech:13s} {opt:4s} {oa:>8.4f} {na:>8.4f} {na-oa:>+8.4f} "
          f"{of:>9.4f} {nf:>8.4f} {nf-of:>+8.4f}")
print("\nDisjoint projection, class-IL (ER masked-loss OFF), seed 42, lr=1e-3, ep=5, buffer=1000, no "
      "sparsity reg. Oracle (task id at eval). Forward gain gates per task in the forward; plasticity "
      "gates per-task grads.")
