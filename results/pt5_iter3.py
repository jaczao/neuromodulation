"""pt5 Iteration 3 — LEARNED projection (`projection=learned`).

User-requested slice: 4 mechanisms x {naive, neurom, er, neurom+er} x {adam, sgd} x {class-IL,
task-IL}, 1 seed (screening, SPEC Methodology 3). The 4 mechanisms are:
  - gain per-NEURON   : activation, granularity=neuron,  gate = (h0,h1)   [gain_layers 0,2]
  - gain per-SYNAPSE  : activation, granularity=synapse, layers = (net.0,net.2) [mask_layers 0,2]
  - plasticity per-NEURON  : plasticity, granularity=neuron,  layers=(0,2,4) scope=both
  - plasticity per-SYNAPSE : plasticity, granularity=synapse, layers=(net.0,net.2)

Learned P training (this iteration's mechanism):
  - gain (FORWARD target): P is an nn.Parameter INSIDE the model, so the pt5 main optimizer trains
    it via the ordinary main loss (one-hot => only the current task's row P[t] gets a gradient, so
    rows specialise per task and freeze after their task).
  - plasticity (GRAD-GATE target): the gate is applied to grads in place, so a learned P gets no
    main-loss gradient. It is trained by a per-batch lookahead / first-order meta-gradient wired in
    cl_train's pt5 branch (W_fast = W - lr*(gate.g), meta-loss on the SAME batch trains ONLY P). For
    the +ER cells cx/cy already carry replayed past-task samples, so the meta-loss is the SPEC's
    modulator-only replay meta-loss.

Masking per metric (the ONLY difference between class-IL and task-IL):
  - class-IL: naive/neurom use output_masking='loss' (masked train loss, class-IL 10-way eval);
    er/neurom+er use 'none' (ER supplies its own retention). class-IL 10-way eval throughout.
  - task-IL : all four cells use output_masking='taskil' (masked train loss AND eval masked to each
    task's 2 classes, 2-way argmax). The pt5 eval branch honours taskil (allowed=sequence[i]).

Layer sets are held FIXED across cells per mechanism (no per-condition head switching): per-synapse
gain/plasticity gate the two HIDDEN layers only (0,2) in every cell (an explicit head gate fights
replay, cf. iter-1 weight_mask+ER -0.61); per-neuron plasticity keeps 0,2,4 (that reaches net.4
only via the implicit outgoing-column coupling a1, the cooperative-with-ER kind, not an explicit
head mask). gain per-neuron gates (h0,h1). gain_form=unbounded (parity at zero-init P).

SGD and Adam are BOTH run (user request). Caveat: for plasticity, Adam re-triggers the Adam-moments
issue (the grad is gated before .step(), so Adam's moments come from the gated grad) AND the
lookahead inner step is SGD-style, so plasticity+Adam is a first-order surrogate; SGD is the clean
plasticity comparison. gain is a forward target (Adam is legitimate).

Run: uv run python results/pt5_iter3.py   (redirect to results/pt5_iter3.log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED = 42
LR = 1e-3
EP = 5
BUFFER = 1000
PROJECTION = "learned"          # Iteration 3
OPTIMIZERS = ["sgd", "adam"]
METRICS = ["classil", "taskil"]

# mechanism -> (neuromod_target, extra kwargs shared by neurom + neurom+er)
MECHANISMS = [
    ("gain-neuron", "activation", dict(
        neuromod_granularity="neuron", neuromod_gain_layers="0,2", neuromod_gain_form="unbounded")),
    ("gain-synapse", "activation", dict(
        neuromod_granularity="synapse", neuromod_mask_layers="0,2", neuromod_gain_form="unbounded")),
    ("plast-neuron", "plasticity", dict(
        neuromod_granularity="neuron", neuromod_plasticity_layers="0,2,4",
        neuromod_plasticity_scope="both")),
    ("plast-synapse", "plasticity", dict(
        neuromod_granularity="synapse", neuromod_mask_layers="0,2")),
]


def _mask_for(metric, replay):
    """output_masking per (metric, is-replay-cell). class-IL: loss for non-replay, none for replay.
    task-IL: taskil everywhere (masks train loss + eval to the task's 2 classes)."""
    if metric == "taskil":
        return "taskil"
    return "none" if replay else "loss"


def _base(optimizer, metric, replay, **kw) -> CLConfig:
    masking = _mask_for(metric, replay)
    extra = dict(er_buffer_size=BUFFER) if replay else {}
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                    output_masking=masking, **extra, **kw)


def _neurom(optimizer, metric, replay, target, extra) -> CLConfig:
    kw = dict(use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
              neuromod_target=target, neuromod_projection=PROJECTION, **extra)
    return _base(optimizer, metric, replay, **kw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:48s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


# results[(metric, optimizer)] = dict with baselines + per-mechanism (neurom, neurom+er)
results = {}
for metric in METRICS:
    for optimizer in OPTIMIZERS:
        print(f"\n################ metric={metric}  optimizer={optimizer} ################", flush=True)
        cell = {}
        cell["naive"] = run(f"[{metric} {optimizer}] naive (baseline)",
                            _base(optimizer, metric, replay=False), "naive")
        cell["er"] = run(f"[{metric} {optimizer}] er (baseline)",
                         _base(optimizer, metric, replay=True), "er")
        for name, target, extra in MECHANISMS:
            cell[(name, "neurom")] = run(
                f"[{metric} {optimizer}] {name} neurom",
                _neurom(optimizer, metric, replay=False, target=target, extra=extra), "naive")
            cell[(name, "neurom+er")] = run(
                f"[{metric} {optimizer}] {name} neurom+er",
                _neurom(optimizer, metric, replay=True, target=target, extra=extra), "er")
        results[(metric, optimizer)] = cell


print("\n\n" + "=" * 92)
print("pt5 ITERATION 3 (projection=learned) — 1 seed (42), lr=1e-3, ep=5, buffer=1000")
print("standalone bar: neurom vs naive (same-opt, same-metric) | +ER bar: neurom+er vs er (>=2pts)")
print("=" * 92)
for metric in METRICS:
    for optimizer in OPTIMIZERS:
        cell = results[(metric, optimizer)]
        nb = cell["naive"][0]; eb = cell["er"][0]
        print(f"\n--- metric={metric.upper()}  optimizer={optimizer.upper()} ---")
        print(f"  baselines: naive={nb:.4f} (f={cell['naive'][1]:.4f})   "
              f"er={eb:.4f} (f={cell['er'][1]:.4f})")
        print(f"  {'mechanism':16s} {'neurom':>9s} {'(vs naive)':>11s} "
              f"{'neurom+er':>10s} {'(vs er)':>10s}")
        for name, _, _ in MECHANISMS:
            s = cell[(name, "neurom")][0]; e = cell[(name, "neurom+er")][0]
            print(f"  {name:16s} {s:>9.4f} {s - nb:>+11.4f} {e:>10.4f} {e - eb:>+10.4f}")
print("\nCaveat: ORACLE task id at train+eval (task-IL-style even on the class-IL metric). "
      "plasticity+Adam is a first-order surrogate (Adam-moments); SGD is the clean plasticity read.")
