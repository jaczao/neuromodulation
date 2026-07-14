"""pt5 Iteration 3 follow-up — two user-requested probes on the LEARNED projection.

(A) INIT-BIAS sweep (plasticity only). The learned plasticity gate starts at sigmoid(0)=0.5, i.e.
    every weight (incl. replayed-sample grads) is throttled to half-LR from step 1 — a plausible
    driver of the iter-3 plast+ER collapse (-0.15). Sweep the initial per-side gate alpha via a
    logit bias (`neuromod_plasticity_init`): {0.9, 0.95, 0.99} vs the iter-3 default 0.5. Gain is NOT
    swept: its learned init is already 1.0 (parity/neutral) and its failure is mechanistic (a soft
    gate never becomes a hard freeze), not an init-value issue.

(B) SPARSITY regularization (all four learned mechanisms). Add an L1 penalty on the projected GATE
    (`neuromod_sparsity_lambda` * mean|gate|), pushing each task toward a sparse active subset — i.e.
    toward the disjoint {0,1} allocation that actually worked in iter-1. NB L1 on P itself is
    degenerate here (gate at P=0 is 1.0 for gain / 0.5 for plasticity, not 0), so the meaningful
    sparsity target is the gate, not P. gain trains its P via the main loss (penalty added there);
    plasticity trains its P via the meta-loss (penalty added there). Sweep lambda in {0.1, 1.0}.

Both sections: 1 seed (42), lr=1e-3, ep=5, buffer=1000, class-IL. Gain sparsity is run on ADAM
(where the iter-1 disjoint win was strongest, er+gain 0.9901); plasticity on SGD (the clean read,
and where the collapse is). Baselines and iter-3 (init=0.5 / lambda=0) values are shown as fixed
reference rows (already verified bit-exact; not re-run).

Run: uv run python results/pt5_iter3_followup.py   (redirect to results/pt5_iter3_followup.log)
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

# iter-3 reference numbers (projection=learned, init=0.5, lambda=0), class-IL, from results/pt5_iter3.log
REF = {
    ("sgd", "naive"): 0.6296, ("sgd", "er"): 0.7226,
    ("adam", "naive"): 0.3894, ("adam", "er"): 0.9053,
    ("sgd", "gain-neuron"): (0.6311, 0.7271), ("sgd", "gain-synapse"): (0.6295, 0.7266),
    ("sgd", "plast-neuron"): (0.6456, 0.5676), ("sgd", "plast-synapse"): (0.6430, 0.5847),
    ("adam", "gain-neuron"): (0.3770, 0.8842), ("adam", "gain-synapse"): (0.4202, 0.9169),
    ("adam", "plast-neuron"): (0.3866, 0.9057), ("adam", "plast-synapse"): (0.3820, 0.8900),
}

MECH = {  # name -> (target, extra kwargs)
    "gain-neuron": ("activation", dict(neuromod_granularity="neuron", neuromod_gain_layers="0,2",
                                        neuromod_gain_form="unbounded")),
    "gain-synapse": ("activation", dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2",
                                         neuromod_gain_form="unbounded")),
    "plast-neuron": ("plasticity", dict(neuromod_granularity="neuron",
                                         neuromod_plasticity_layers="0,2,4",
                                         neuromod_plasticity_scope="both")),
    "plast-synapse": ("plasticity", dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2")),
}


def cfg(optimizer, mech, replay, **extra_over):
    target, extra = MECH[mech]
    masking = "none" if replay else "loss"
    kw = dict(use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
              neuromod_target=target, neuromod_projection="learned", **extra, **extra_over)
    base = dict(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer, output_masking=masking)
    if replay:
        base["er_buffer_size"] = BUFFER
    return CLConfig(**base, **kw)


def run(tag, config, method):
    acc, f = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:52s} acc={acc:.4f}  forget={f:.4f}", flush=True)
    return acc, f


# ------------------------------------------------------------------ (A) init-bias sweep
print("\n################ (A) INIT-BIAS sweep — plasticity, SGD, class-IL ################", flush=True)
INITS = [0.9, 0.95, 0.99]
A = {}  # (mech, init, cell) -> acc
for mech in ("plast-neuron", "plast-synapse"):
    for init in INITS:
        A[(mech, init, "neurom")] = run(f"[A] {mech} init={init} neurom",
                                        cfg("sgd", mech, False, neuromod_plasticity_init=init), "naive")[0]
        A[(mech, init, "neurom+er")] = run(f"[A] {mech} init={init} neurom+er",
                                           cfg("sgd", mech, True, neuromod_plasticity_init=init), "er")[0]

# ------------------------------------------------------------------ (B) sparsity sweep
# Calibrated grids (see scratchpad calib): gate L1 penalty, mean-normalized, so Adam (grad-normalized)
# bites near lambda~1 for gain (inverted-U peak); SGD plasticity's Adam meta-opt saturates (plateau),
# and sparsity pushes plasticity toward the frozen disjoint-plasticity regime (~0.43). So gain gets a
# fine grid around the peak; plasticity gets a single point to document the hurt.
print("\n################ (B) SPARSITY sweep — gate L1, class-IL ################", flush=True)
B = {}  # (opt, mech, lam, cell) -> acc
# (opt, mech, [lambdas])
SPARSE_SPEC = [
    ("adam", "gain-neuron", [0.3, 1.0, 3.0]),
    ("adam", "gain-synapse", [0.3, 1.0, 3.0]),
    ("sgd", "plast-neuron", [1.0]),
    ("sgd", "plast-synapse", [1.0]),
]
for opt, mech, lams in SPARSE_SPEC:
    for lam in lams:
        B[(opt, mech, lam, "neurom")] = run(f"[B] {opt} {mech} lam={lam} neurom",
                                            cfg(opt, mech, False, neuromod_sparsity_lambda=lam), "naive")[0]
        B[(opt, mech, lam, "neurom+er")] = run(f"[B] {opt} {mech} lam={lam} neurom+er",
                                               cfg(opt, mech, True, neuromod_sparsity_lambda=lam), "er")[0]


# ------------------------------------------------------------------ tables
print("\n\n" + "=" * 88)
print("pt5 ITER 3 FOLLOW-UP — 1 seed (42), lr=1e-3, ep=5, buffer=1000, class-IL")
print("=" * 88)

print("\n--- (A) INIT-BIAS (plasticity, SGD) ---")
print(f"  baselines: naive={REF[('sgd','naive')]:.4f}  er={REF[('sgd','er')]:.4f}")
print(f"  {'mechanism':14s} {'init':>5s} {'neurom':>9s} {'(vs naive)':>11s} {'neurom+er':>10s} {'(vs er)':>10s}")
for mech in ("plast-neuron", "plast-synapse"):
    s0, e0 = REF[("sgd", mech)]
    print(f"  {mech:14s} {'0.50*':>5s} {s0:>9.4f} {s0-REF[('sgd','naive')]:>+11.4f} {e0:>10.4f} {e0-REF[('sgd','er')]:>+10.4f}   (iter3)")
    for init in INITS:
        s = A[(mech, init, "neurom")]; e = A[(mech, init, "neurom+er")]
        print(f"  {mech:14s} {init:>5.2f} {s:>9.4f} {s-REF[('sgd','naive')]:>+11.4f} {e:>10.4f} {e-REF[('sgd','er')]:>+10.4f}")

print("\n--- (B) SPARSITY (gate L1) ---")
for opt, mech, lams in SPARSE_SPEC:
    nb = REF[(opt, "naive")]; eb = REF[(opt, "er")]
    s0, e0 = REF[(opt, mech)]
    print(f"  {mech} [{opt}]  baselines naive={nb:.4f} er={eb:.4f}")
    print(f"    {'lambda':>7s} {'neurom':>9s} {'(vs naive)':>11s} {'neurom+er':>10s} {'(vs er)':>10s}")
    print(f"    {'0.0*':>7s} {s0:>9.4f} {s0-nb:>+11.4f} {e0:>10.4f} {e0-eb:>+10.4f}   (iter3)")
    for lam in lams:
        s = B[(opt, mech, lam, "neurom")]; e = B[(opt, mech, lam, "neurom+er")]
        print(f"    {lam:>7.2f} {s:>9.4f} {s-nb:>+11.4f} {e:>10.4f} {e-eb:>+10.4f}")

print("\n* = iter-3 reference (init=0.5 / lambda=0), not re-run (verified bit-exact). Oracle caveat "
      "carries (task-id at train+eval).")
