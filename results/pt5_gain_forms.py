"""pt5 GAIN FORM sweep — {unbounded, bounded01, positive} x {per-neuron, per-synapse} gain.

User-requested slice: 2 gain granularities x 3 gain forms x {naive, neurom, er, neurom+er} x
{sgd, adam}, class-IL only, 1 seed (screening, SPEC Methodology 3). 28 runs total (4 baselines +
2x3x2x2 = 24 mechanism cells).

WHY projection=learned (not a free choice): gain_form is INERT under a fixed binary P — gain_gamma
returns the raw {0,1} gate for every form (disjoint/shared), so the three forms would emit three
IDENTICAL numbers. The form only exists under the learned projection, hence this is an iter-3
(learned) slice, and the `unbounded` column is directly comparable to results/pt5_iter3.py.

The three forms (all inert under fixed P; see neuromod.gain_gamma):
  - unbounded : gamma = 1 + raw            range (-inf,+inf), init 1.0, INVERTS below 0  [iter-3 default]
  - bounded01 : gamma = sigmoid(raw)       range (0,1),       init 0.5, suppress-only
  - positive  : gamma = softplus(raw+b)    range (0,+inf),    init 1.0, amplifies, never inverts
                (b = ln(e-1), so gamma(0)=1.0 exactly => same zero-init parity as unbounded)

`positive` is an ABLATION of unbounded's sign inversion ("is positivity enough, or is the inversion
doing real work?"), NOT an expected win: it cannot hard-freeze (softplus reaches 0 only
asymptotically) and the exact-zero freeze is the iter-1 lever. No sparsity here (lambda=0), so the
L1-pull asymmetry between forms is NOT exercised in this sweep — that is a separate axis.

BUILT-IN REGRESSION CHECK: `unbounded` must reproduce results/pt5_iter3.py bit-exact, since the
gate= -> gain_form refactor is behaviour-preserving:
    baselines    class-IL SGD  naive 0.6296 / er 0.7226 ; Adam naive 0.3894 / er 0.9053
    gain-neuron  class-IL SGD  0.6311        ; Adam 0.3770
    gain-synapse class-IL Adam 0.4202
A mismatch means the refactor moved something and the sweep is void.

Masking (class-IL convention, matching iter-3): naive/neurom use output_masking='loss' (masked train
loss, class-IL 10-way eval); er/neurom+er use 'none' (ER supplies its own retention). Layer sets are
held FIXED across cells: per-neuron gain gates (h0,h1) [gain_layers 0,2]; per-synapse gain gates the
two HIDDEN layers (net.0,net.2) [mask_layers 0,2] — never the head, since an explicit head gate
fights replay (cf. iter-1 weight_mask+ER -0.61).

Caveat: ORACLE task id at train+eval, so these are task-IL-style results on the class-IL metric.

Run: uv run python results/pt5_gain_forms.py   (redirect to results/pt5_gain_forms.log)
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
PROJECTION = "learned"          # the only projection where gain_form is live
METRIC = "classil"
OPTIMIZERS = ["sgd", "adam"]
FORMS = ["unbounded", "bounded01", "positive"]

# granularity -> (neuromod_target, kwargs shared by neurom + neurom+er; gain_form added per-form)
GRANULARITIES = [
    ("gain-neuron", "activation", dict(
        neuromod_granularity="neuron", neuromod_gain_layers="0,2")),
    ("gain-synapse", "activation", dict(
        neuromod_granularity="synapse", neuromod_mask_layers="0,2")),
]

# results/pt5_iter3.py, class-IL, projection=learned, gain_form=unbounded (regression reference)
ITER3_REF = {
    ("sgd", "naive"): 0.6296, ("sgd", "er"): 0.7226,
    ("adam", "naive"): 0.3894, ("adam", "er"): 0.9053,
    ("sgd", "gain-neuron"): 0.6311, ("adam", "gain-neuron"): 0.3770,
    ("adam", "gain-synapse"): 0.4202,
}


def _base(optimizer, replay, **kw) -> CLConfig:
    """class-IL: masked train loss for non-replay, none for replay (ER brings its own retention)."""
    masking = "none" if replay else "loss"
    extra = dict(er_buffer_size=BUFFER) if replay else {}
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                    output_masking=masking, **extra, **kw)


def _neurom(optimizer, replay, target, extra, form) -> CLConfig:
    kw = dict(use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
              neuromod_target=target, neuromod_projection=PROJECTION,
              neuromod_gain_form=form, **extra)
    return _base(optimizer, replay, **kw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:52s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


results = {}
for optimizer in OPTIMIZERS:
    print(f"\n################ metric={METRIC}  optimizer={optimizer} ################", flush=True)
    cell = {}
    cell["naive"] = run(f"[{optimizer}] naive (baseline)", _base(optimizer, replay=False), "naive")
    cell["er"] = run(f"[{optimizer}] er (baseline)", _base(optimizer, replay=True), "er")
    for name, target, extra in GRANULARITIES:
        for form in FORMS:
            cell[(name, form, "neurom")] = run(
                f"[{optimizer}] {name} {form} neurom",
                _neurom(optimizer, False, target, extra, form), "naive")
            cell[(name, form, "neurom+er")] = run(
                f"[{optimizer}] {name} {form} neurom+er",
                _neurom(optimizer, True, target, extra, form), "er")
    results[optimizer] = cell


print("\n\n" + "=" * 96)
print("pt5 GAIN FORMS (projection=learned, class-IL) — 1 seed (42), lr=1e-3, ep=5, buffer=1000")
print("standalone bar: neurom vs naive (same-opt) | +ER bar: neurom+er vs er (>=2pts)")
print("=" * 96)
for optimizer in OPTIMIZERS:
    cell = results[optimizer]
    nb = cell["naive"][0]; eb = cell["er"][0]
    print(f"\n--- optimizer={optimizer.upper()} ---")
    print(f"  baselines: naive={nb:.4f} (f={cell['naive'][1]:.4f})   er={eb:.4f} (f={cell['er'][1]:.4f})")
    print(f"  {'mechanism':14s} {'form':10s} {'neurom':>9s} {'(vs naive)':>11s} "
          f"{'forget':>8s} {'neurom+er':>10s} {'(vs er)':>9s} {'forget':>8s}")
    for name, _, _ in GRANULARITIES:
        for form in FORMS:
            s, sf = cell[(name, form, "neurom")]
            e, ef = cell[(name, form, "neurom+er")]
            print(f"  {name:14s} {form:10s} {s:>9.4f} {s - nb:>+11.4f} {sf:>8.4f} "
                  f"{e:>10.4f} {e - eb:>+9.4f} {ef:>8.4f}")

print("\n--- REGRESSION CHECK vs results/pt5_iter3.py (unbounded must match bit-exact) ---")
ok = True
for optimizer in OPTIMIZERS:
    cell = results[optimizer]
    checks = [("naive", cell["naive"][0]), ("er", cell["er"][0])]
    for name, _, _ in GRANULARITIES:
        checks.append((name, cell[(name, "unbounded", "neurom")][0]))
    for key, got in checks:
        ref = ITER3_REF.get((optimizer, key))
        if ref is None:
            continue
        match = abs(got - ref) < 5e-4
        ok &= match
        print(f"  {optimizer:4s} {key:14s} got={got:.4f}  iter3={ref:.4f}  "
              f"{'MATCH' if match else '*** MISMATCH ***'}")
print(f"  => {'all reproduce iter-3' if ok else 'MISMATCH: refactor moved a number, sweep is void'}")

print("\nCaveat: ORACLE task id at train+eval (task-IL-style result on the class-IL metric). "
      "sparsity_lambda=0 throughout, so the forms' L1-pull asymmetry is not exercised here.")
