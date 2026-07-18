"""pt5 SPARSITY (gate L1) sweep — continue iter3-followup (B) across BOTH optimizers and the THREE
gain arms {standalone, buff-own, er-own}, class-IL, learned projection, unbounded gain form.

WHY: iter3-followup (B) only swept the L1 sparsity penalty for gain under ADAM, and only the
{standalone, +ER} pair — and that +ER used the OLD er_task_id=OFF (wrong-task) default. It never ran
(a) SGD gain sparsity at all, (b) the modulator-only replay meta arm (buff-own) with sparsity, or
(c) er-own (each replayed sample under its own P[j], now the default). This sweep fills all three.

THE THREE ARMS (identical to results/pt5_gain_forms_buffer.py; sparsity_lambda added to each):
  - standalone : method=naive, NO buffer. Gain P trained by the MAIN loss (current task only) + the
                 L1 penalty on the projected gate (train.py:1141). masking='loss' (class-IL 10-way).
  - buff-own   : method=naive, --neuromod-meta-replay --neuromod-er-task-id. A SEPARATE optimizer
                 trains ONLY P on a modulator-only PER-TASK meta-loss (each seen task j forwarded
                 under its own gate P[j]); the L1 is added to that meta-loss (train.py:1233). Main net
                 stays naive. masking='loss'.
  - er-own     : method=er, --neuromod-er-task-id. Each replayed sample in the main batch passes
                 through its own gate P[j]; P trained by the (replay-augmented) main loss + L1
                 (train.py:1141). masking='none' (ER brings its own retention).

The L1 (neuromod_sparsity_lambda * mean|gate|, gate = 1+raw for unbounded gain) pushes each task's
gate toward a sparse active subset — i.e. toward the iter-1 disjoint {0,1} freeze, but soft/learned.

LAMBDA GRIDS are per-granularity (per-synapse has a larger fan-in, so its mean-normalised penalty
bites at ~10x higher lambda — the D-scaling noted in iter3-followup): neuron {0,.1,.3,1,3} brackets
the Adam inverted-U peak (~0.3); synapse {0,1,3,10,30} brackets the still-rising Adam curve (0.7632
at 10). lambda=0 is run LIVE (not referenced) as the per-arm anchor AND a regression check — with
sparsity_lambda=0 the L1 code is gated off, so each lambda=0 cell must reproduce
results/pt5_gain_forms_buffer.log bit-exact (unbounded rows), asserted at the end.

NB SGD's useful lambda scale is UNKNOWN (SGD grads are not normalised like Adam's, so the followup's
"bites near lambda~1" calibration is Adam-specific) — the SGD columns are exploratory; the grid may
under- or over-shoot the SGD peak.

1 seed (42), lr=1e-3, ep=5, buffer=1000. ORACLE task id at train+eval (task-IL-style on the class-IL
metric) — same caveat as all pt5 gain work; buff-own/er-own USE the buffer so beating ER is not
beating ER (ER has no oracle).

Run: uv run python results/pt5_sparsity_arms.py   (redirect to results/pt5_sparsity_arms.log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000
PROJECTION = "learned"
FORM = "unbounded"          # the sparsity mechanism's form (constant |1| L1 pull, can reach 0)
OPTIMIZERS = ["sgd", "adam"]

# granularity -> (kwargs, lambda grid). Per-synapse gate has ~larger fan-in -> higher useful lambda.
GRANS = {
    "gain-neuron":  (dict(neuromod_granularity="neuron",  neuromod_gain_layers="0,2"),
                     [0.0, 0.1, 0.3, 1.0, 3.0]),
    "gain-synapse": (dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2"),
                     [0.0, 1.0, 3.0, 10.0, 30.0]),
}

# (arm, meta_replay, er_task_id, method, replay). replay drives masking + er_buffer_size.
ARMS = [
    ("standalone", False, False, "naive", False),   # no buffer, P via main loss
    ("buff-own",   True,  True,  "naive", False),    # modulator-only replay, per-task meta gate
    ("er-own",     False, True,  "er",    True),     # +ER, replayed sample -> its own P[j]
]

# well-established baselines (reproduced across pt5; not re-run)
BASE = {("sgd", "naive"): 0.6296, ("sgd", "er"): 0.7226,
        ("adam", "naive"): 0.3894, ("adam", "er"): 0.9053}

# lambda=0 anchors from results/pt5_gain_forms_buffer.log (unbounded rows) — regression targets.
REF0 = {
    ("sgd", "gain-neuron", "standalone"): 0.6311, ("sgd", "gain-neuron", "buff-own"): 0.9074,
    ("sgd", "gain-neuron", "er-own"): 0.7376,
    ("sgd", "gain-synapse", "standalone"): 0.6295, ("sgd", "gain-synapse", "buff-own"): 0.9871,
    ("sgd", "gain-synapse", "er-own"): 0.7282,
    ("adam", "gain-neuron", "standalone"): 0.3770, ("adam", "gain-neuron", "buff-own"): 0.5075,
    ("adam", "gain-neuron", "er-own"): 0.9887,
    ("adam", "gain-synapse", "standalone"): 0.4202, ("adam", "gain-synapse", "buff-own"): 0.6304,
    ("adam", "gain-synapse", "er-own"): 0.9900,
}


def cfg(optimizer, gkw, arm_meta, arm_taskid, replay, lam) -> CLConfig:
    masking = "none" if replay else "loss"
    extra = dict(er_buffer_size=BUFFER) if (replay or arm_meta) else {}
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer, output_masking=masking,
                    use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                    neuromod_target="activation", neuromod_projection=PROJECTION, neuromod_gain_form=FORM,
                    neuromod_meta_replay=arm_meta, neuromod_er_task_id=arm_taskid,
                    neuromod_sparsity_lambda=lam, **extra, **gkw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:56s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


R = {}  # (opt, gran, arm, lam) -> (acc, forget)
for optimizer in OPTIMIZERS:
    for gname, (gkw, lams) in GRANS.items():
        for arm, meta, taskid, method, replay in ARMS:
            for lam in lams:
                print(f"\n#### {optimizer} {gname} {arm} lambda={lam} ####", flush=True)
                R[(optimizer, gname, arm, lam)] = run(
                    f"[{optimizer}] {gname} {arm} lam={lam}",
                    cfg(optimizer, gkw, meta, taskid, replay, lam), method)


# ------------------------------------------------------------------ tables
print("\n\n" + "=" * 100)
print("pt5 SPARSITY (gate L1) x THREE ARMS x {sgd,adam} — 1 seed (42), lr=1e-3, ep=5, buffer=1000, class-IL")
print("standalone = no buffer (P via main loss) | buff-own = modulator-only replay meta-loss |")
print("er-own = +ER, replayed sample under its own P[j].  gain_form=unbounded, projection=learned")
print("=" * 100)
for optimizer in OPTIMIZERS:
    nb = BASE[(optimizer, "naive")]; eb = BASE[(optimizer, "er")]
    print(f"\n################ optimizer={optimizer.upper()}   baselines: naive={nb:.4f}  er={eb:.4f} ################")
    for gname, (_, lams) in GRANS.items():
        print(f"\n--- {gname} [{optimizer}] ---")
        print(f"  {'lambda':>7s} | {'standalone':>10s} {'d0':>7s} | {'buff-own':>9s} {'d0':>7s} "
              f"{'forget':>7s} | {'er-own':>8s} {'d0':>7s} {'(vs er)':>8s}")
        for lam in lams:
            sa, saf = R[(optimizer, gname, "standalone", lam)]
            bo, bof = R[(optimizer, gname, "buff-own", lam)]
            eo, eof = R[(optimizer, gname, "er-own", lam)]
            sa0 = R[(optimizer, gname, "standalone", 0.0)][0]
            bo0 = R[(optimizer, gname, "buff-own", 0.0)][0]
            eo0 = R[(optimizer, gname, "er-own", 0.0)][0]
            mark = "*" if lam == 0.0 else " "
            print(f"  {lam:>6.2f}{mark} | {sa:>10.4f} {sa-sa0:>+7.4f} | {bo:>9.4f} {bo-bo0:>+7.4f} "
                  f"{bof:>7.4f} | {eo:>8.4f} {eo-eo0:>+7.4f} {eo-eb:>+8.4f}")
    print("  d0 = delta vs this arm's own lambda=0 anchor (marked *) | (vs er) = er-own minus ER baseline")

print("\n--- REGRESSION CHECK: lambda=0 must reproduce results/pt5_gain_forms_buffer.log (unbounded) ---")
ok = True
for (opt, gname, arm), ref in REF0.items():
    got = R[(opt, gname, arm, 0.0)][0]
    match = abs(got - ref) < 5e-4
    ok &= match
    print(f"  {opt:4s} {gname:12s} {arm:11s} got={got:.4f} ref={ref:.4f} "
          f"{'MATCH' if match else '*** MISMATCH ***'}")
print(f"  => {'all lambda=0 anchors reproduce the buffer study' if ok else 'MISMATCH: a code path moved'}")

print("\nCaveats: ORACLE task id at train+eval (task-IL-style on the class-IL metric). buff-own/er-own "
      "USE the buffer (beating ER is not beating ER — no oracle there). SGD lambda scale is exploratory "
      "(SGD grads unnormalised, unlike Adam). 1 seed.")
