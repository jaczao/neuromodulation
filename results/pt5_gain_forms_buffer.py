"""pt5 GAIN FORMS x BUFFER — {unbounded, bounded01, positive} x {per-neuron, per-synapse} gain,
standalone WITH the modulator-only replay buffer, and +ER with correct per-sample task ids.

Follow-up to results/pt5_gain_forms.py (same grid, no buffer). User-requested. 40 runs:
  - 4  baselines            : {naive, er} x {sgd, adam}
  - 24 standalone + buffer  : 2 granularity x 3 form x 2 meta-arm x 2 opt   (method=naive)
  - 12 +ER, correct task ids: 2 granularity x 3 form x 2 opt                (method=er)

THE THREE ARMS (what "the buffer" and "correct task ids" mean per cell):
  - standalone, meta_task_id=ON  : --neuromod-meta-replay + --neuromod-er-task-id. The gain P is
    trained by a modulator-only PER-TASK meta-loss: each seen task j is forwarded under ITS OWN gate
    P[j] (buffer samples for j<t via label_to_task, fresh batch for j=t). Main net stays naive; P is
    excluded from its optimizer. This is iter3-followup-3's mechanism.
  - standalone, meta_task_id=OFF : the WRONG-TASK ablation. IDENTICAL sample composition, but every
    meta batch is forwarded under the CURRENT task's gate P[t]. Since the one-hot routes every meta
    gradient to row t, the past rows P[j] (j<t) get NO gradient => the retention signal is gone. The
    gate is the ONLY difference between the two arms, so this isolates per-task meta gating.
  - +ER, er_task_id=ON          : each REPLAYED sample in the main batch passes through its own gate
    P[j] (batch split by task, logits scattered back) instead of the current P[t].

`--neuromod-er-task-id` selects BOTH the meta arm (standalone) and the main-batch arm (+ER) — one
flag, one meaning ("gate a buffered sample by its own task id"). NOTE it was replay-gated before, so
the meta-loop was per-task unconditionally; results/pt5_iter3_gain_metareplay.py now passes it
explicitly to reproduce its numbers.

projection=learned throughout: gain_form is INERT under a fixed P (gain_gamma returns raw {0,1} for
every form), and the meta-loop only trains a LEARNED P.

REFERENCES (must reproduce; verified by hand before this sweep):
  - standalone meta ON,  gain-neuron class-IL SGD unbounded = 0.9074 (f=0.0205)  [followup-3]
  - standalone meta OFF, gain-neuron class-IL SGD unbounded = 0.7155 (f=0.0485)  [new arm]
  - no-buffer baselines for the same grid: results/pt5_gain_forms.log
    (SGD naive 0.6296 / er 0.7226 ; Adam naive 0.3894 / er 0.9053)

Masking (class-IL convention): naive/standalone use output_masking='loss'; er/+ER use 'none'.
Layer sets fixed: per-neuron gain gates (h0,h1); per-synapse gain gates hidden net.0,net.2 only.

CAVEATS carried from followup-3: (1) ORACLE task id at train+eval => task-IL-style result on the
class-IL metric; (2) the standalone arms USE the buffer (replay on the MODULATOR, not the backbone),
so they are NOT apples-to-apples with a no-buffer naive baseline, and beating ER here is not beating
ER (ER has no oracle); (3) 1 seed.

Run: uv run python results/pt5_gain_forms_buffer.py   (redirect to results/pt5_gain_forms_buffer.log)
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
PROJECTION = "learned"
OPTIMIZERS = ["sgd", "adam"]
FORMS = ["unbounded", "bounded01", "positive"]

GRANULARITIES = [
    ("gain-neuron", dict(neuromod_granularity="neuron", neuromod_gain_layers="0,2")),
    ("gain-synapse", dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2")),
]

# (arm, meta_replay, er_task_id, method, replay) -- replay drives masking + er_buffer_size
ARMS = [
    ("buf-meta-own",  True,  True,  "naive", False),   # standalone, per-task meta gate  [followup-3]
    ("buf-meta-cur",  True,  False, "naive", False),   # standalone, wrong-task meta gate [ablation]
    ("er-own",        False, True,  "er",    True),    # +ER, replayed sample -> its own P[j]
]

# hand-verified references (see docstring)
REFS = {("buf-meta-own", "sgd", "gain-neuron", "unbounded"): 0.9074,
        ("buf-meta-cur", "sgd", "gain-neuron", "unbounded"): 0.7155}
# Comparison points from results/pt5_gain_forms.log (the no-buffer sweep, same grid/seed/config).
# NOBUF   = its `neurom` cells    : standalone, NO buffer (meta_replay OFF).
# ER_CUR  = its `neurom+er` cells : +ER with er_task_id OFF (replayed sample under the CURRENT P[t]).
# ER_CUR is a valid pairing for this sweep's er-own arm — identical config except the flag, since
# meta_replay is inert for +ER cells (gain_meta_replay_on requires `not use_replay`). Hence er-cur is
# NOT re-run here (it would reproduce these to 4 dp); the table below merges it in.
NOBUF = {
    ("sgd", "gain-neuron", "unbounded"): 0.6311, ("sgd", "gain-neuron", "bounded01"): 0.4638,
    ("sgd", "gain-neuron", "positive"): 0.6303, ("sgd", "gain-synapse", "unbounded"): 0.6295,
    ("sgd", "gain-synapse", "bounded01"): 0.4643, ("sgd", "gain-synapse", "positive"): 0.6291,
    ("adam", "gain-neuron", "unbounded"): 0.3770, ("adam", "gain-neuron", "bounded01"): 0.4073,
    ("adam", "gain-neuron", "positive"): 0.4623, ("adam", "gain-synapse", "unbounded"): 0.4202,
    ("adam", "gain-synapse", "bounded01"): 0.4271, ("adam", "gain-synapse", "positive"): 0.4389,
}
ER_CUR = {
    ("sgd", "gain-neuron", "unbounded"): 0.7271, ("sgd", "gain-neuron", "bounded01"): 0.2901,
    ("sgd", "gain-neuron", "positive"): 0.7243, ("sgd", "gain-synapse", "unbounded"): 0.7266,
    ("sgd", "gain-synapse", "bounded01"): 0.2855, ("sgd", "gain-synapse", "positive"): 0.7261,
    ("adam", "gain-neuron", "unbounded"): 0.8842, ("adam", "gain-neuron", "bounded01"): 0.9031,
    ("adam", "gain-neuron", "positive"): 0.8960, ("adam", "gain-synapse", "unbounded"): 0.9169,
    ("adam", "gain-synapse", "bounded01"): 0.9100, ("adam", "gain-synapse", "positive"): 0.8839,
}


def _cfg(optimizer, replay, **kw) -> CLConfig:
    masking = "none" if replay else "loss"
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                    output_masking=masking, er_buffer_size=BUFFER, **kw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:56s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


results = {}
for optimizer in OPTIMIZERS:
    print(f"\n################ class-IL  optimizer={optimizer} ################", flush=True)
    cell = {}
    cell["naive"] = run(f"[{optimizer}] naive (baseline)",
                        _cfg(optimizer, replay=False), "naive")
    cell["er"] = run(f"[{optimizer}] er (baseline)",
                     _cfg(optimizer, replay=True), "er")
    for gname, gkw in GRANULARITIES:
        for form in FORMS:
            for arm, meta, task_id, method, replay in ARMS:
                cfg = _cfg(optimizer, replay,
                           use_neuromod=True, neuromod_drivers="task_id=onehot",
                           neuromod_context="none", neuromod_target="activation",
                           neuromod_projection=PROJECTION, neuromod_gain_form=form,
                           neuromod_meta_replay=meta, neuromod_er_task_id=task_id, **gkw)
                cell[(gname, form, arm)] = run(
                    f"[{optimizer}] {gname} {form} {arm}", cfg, method)
    results[optimizer] = cell


print("\n\n" + "=" * 104)
print("pt5 GAIN FORMS x BUFFER (projection=learned, class-IL) — 1 seed (42), lr=1e-3, ep=5, buffer=1000")
print("buf-meta-own = modulator-only replay, per-task meta gate | buf-meta-cur = wrong-task ablation")
print("er-own = +ER with each replayed sample under its own P[j] | no-buf from results/pt5_gain_forms.log")
print("=" * 104)
for optimizer in OPTIMIZERS:
    cell = results[optimizer]
    nb = cell["naive"][0]; eb = cell["er"][0]
    print(f"\n--- optimizer={optimizer.upper()} ---   baselines: naive={nb:.4f}  er={eb:.4f}")
    print("  STANDALONE (main net naive; the buffer trains ONLY the gain P) | +ER (buffer trains the backbone)")
    print(f"  {'mechanism':14s} {'form':10s} {'no-buf':>8s} {'meta-cur':>9s} {'meta-own':>9s} "
          f"{'d-meta':>7s} {'own-naive':>10s} | {'er-cur':>8s} {'er-own':>8s} {'d-er':>7s} {'own-er':>7s}")
    for gname, _ in GRANULARITIES:
        for form in FORMS:
            own, ownf = cell[(gname, form, "buf-meta-own")]
            cur, curf = cell[(gname, form, "buf-meta-cur")]
            ero, erof = cell[(gname, form, "er-own")]
            nbuf = NOBUF.get((optimizer, gname, form), float("nan"))
            ecur = ER_CUR.get((optimizer, gname, form), float("nan"))
            print(f"  {gname:14s} {form:10s} {nbuf:>8.4f} {cur:>9.4f} {own:>9.4f} "
                  f"{own - cur:>+7.4f} {own - nb:>+10.4f} | {ecur:>8.4f} {ero:>8.4f} "
                  f"{ero - ecur:>+7.4f} {ero - eb:>+7.4f}")
    print("  d-meta = per-task minus wrong-task meta gate | d-er = same flag under ER (er-cur from "
          "results/pt5_gain_forms.log) | own-er = er-own vs the ER baseline (+2pt bar)")

print("\n--- REFERENCE CHECK (hand-verified before the sweep) ---")
for (arm, opt, gname, form), ref in REFS.items():
    got = results[opt][(gname, form, arm)][0]
    match = abs(got - ref) < 5e-4
    print(f"  {opt:4s} {gname:12s} {form:10s} {arm:12s} got={got:.4f} ref={ref:.4f} "
          f"{'MATCH' if match else '*** MISMATCH ***'}")

print("\nCaveats: ORACLE task id at train+eval (task-IL-style on the class-IL metric). The standalone "
      "meta arms USE the buffer (replay on the MODULATOR, not the backbone), so they are NOT "
      "apples-to-apples with the no-buffer naive baseline, and beating ER is not beating ER (no "
      "oracle there). sparsity_lambda=0 throughout. 1 seed.")
