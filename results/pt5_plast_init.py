"""pt5 LEARNED-PLASTICITY init sweep — init in {0.5, 0.8, 0.9, 0.99} x {per-neuron, per-synapse}
x {sgd, adam} x {standalone+buffer, +ER}. class-IL, 1 seed. 36 runs (4 baselines + 4x2x2x2 = 32).

WHY THIS SWEEP (fills the gap the iter3-followup init sweep skipped). The learned plasticity gate is
`sigmoid(init_bias + raw)` with `init_bias = logit(neuromod_plasticity_init)`, so the init sets BOTH
the starting LR throttle AND the gate's own trainability, and the two fight:

  init  bias    gate at init   d gate/d raw = s(1-s)   consequence
  0.5   0.000   0.5            0.2500 (max)            HALVES every grad from step one (an effective
                                                       1/2 LR) -> this is what throttled the REPLAYED
                                                       grads and caused the iter-3 plast+ER collapse
  0.8   1.386   0.8            0.1600                  20% throttle, 16x the gradient of 0.99
  0.9   2.197   0.9            0.0900                  10% throttle,  9x the gradient of 0.99
  0.99  4.595   0.99           0.0099 (saturated)      ~no throttle, but P gets ~25x smaller grads
                                                       -> gate stays ~1 -> degenerates to VANILLA

The recorded followup sweep only ran the two ENDS (0.5 and 0.99) and found 0.99 monotonically cures
the +ER collapse (plast-neuron +ER -0.155 -> -0.003) but returns to ~=ER with no net win ("gate -> ~1
-> vanilla"). That "no win at either end" is exactly what a saturation/throttle tradeoff predicts, so
0.8/0.9 test whether a middle exists: enough headroom to not throttle replay, enough gradient to
still train P. If both middles are also ~= their baselines, the tradeoff has no sweet spot and the
ceiling is the LEVER (pt3's conclusion), not the init.

projection=learned throughout: under a FIXED P the gate is raw {0,1} and init_bias is IGNORED
(`raw if self.fixed else sigmoid(init_bias + raw)`), so the init only exists here.

ARMS
  - standalone (non-ER): --neuromod-meta-replay, so the buffer trains ONLY P via the lookahead
    meta-loss while the main net steps naive. NOTE plasticity gates the GRADIENT, so its meta-loss
    applies ONE gate to the whole summed gradient (train.py augments the batch, no per-task split) --
    there is no meta-own/meta-cur distinction here, unlike gain (that is a forward target).
  - +ER: --neuromod-er-task-id ON (each replayed sample's grad gated by its OWN P[j], per-task
    backward + accumulate + one step). This is the arm where the init matters most: the collapse the
    init cured was the 0.5 gate throttling REPLAYED grads.

Adam CAVEAT (user asked for both): plasticity gates grads BEFORE .step(), so Adam's moments are
computed from the gated grad (the Adam-moments caveat) AND the lookahead inner step is SGD-style =>
plasticity+Adam is a first-order surrogate. SGD is the clean plasticity read.

REFERENCE (hand-verified before the sweep, confirms the gain-meta flag rewiring left plasticity
untouched): plast-neuron, adam, standalone+buffer, init 0.5 = 0.4244 (f=0.5059) [iter3-followup-2].

Masking (class-IL convention): standalone 'loss', +ER 'none'. Caveat: ORACLE task id at train+eval.

Run: uv run python results/pt5_plast_init.py   (redirect to results/pt5_plast_init.log)
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
INITS = [0.5, 0.8, 0.9, 0.99]

MECHS = [
    ("plast-neuron", dict(neuromod_granularity="neuron",
                          neuromod_plasticity_layers="0,2,4",
                          neuromod_plasticity_scope="both")),
    ("plast-synapse", dict(neuromod_granularity="synapse",
                           neuromod_mask_layers="0,2")),
]

# (arm, meta_replay, er_task_id, method, replay)
ARMS = [
    ("standalone", True,  False, "naive", False),   # buffer trains ONLY P (meta lookahead)
    ("er",         False, True,  "er",    True),    # replayed grads gated by their own P[j]
]

REF = {("plast-neuron", "adam", "standalone", 0.5): 0.4244}


def _cfg(optimizer, replay, **kw) -> CLConfig:
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                    output_masking=("none" if replay else "loss"),
                    er_buffer_size=BUFFER, **kw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:52s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


results = {}
for optimizer in OPTIMIZERS:
    print(f"\n################ class-IL  optimizer={optimizer} ################", flush=True)
    cell = {}
    cell["naive"] = run(f"[{optimizer}] naive (baseline)", _cfg(optimizer, replay=False), "naive")
    cell["er"] = run(f"[{optimizer}] er (baseline)", _cfg(optimizer, replay=True), "er")
    for mname, mkw in MECHS:
        for arm, meta, task_id, method, replay in ARMS:
            for init in INITS:
                cfg = _cfg(optimizer, replay,
                           use_neuromod=True, neuromod_drivers="task_id=onehot",
                           neuromod_context="none", neuromod_target="plasticity",
                           neuromod_projection=PROJECTION, neuromod_plasticity_init=init,
                           neuromod_meta_replay=meta, neuromod_er_task_id=task_id, **mkw)
                cell[(mname, arm, init)] = run(
                    f"[{optimizer}] {mname} {arm} init={init}", cfg, method)
    results[optimizer] = cell


print("\n\n" + "=" * 100)
print("pt5 LEARNED-PLASTICITY INIT SWEEP (projection=learned, class-IL) — 1 seed (42), lr=1e-3, ep=5, buffer=1000")
print("standalone = --neuromod-meta-replay (buffer trains ONLY P) | er = --neuromod-er-task-id ON")
print("bar: standalone vs naive | er vs er (>=2pts). init sets BOTH the LR throttle and the gate gradient.")
print("=" * 100)
for optimizer in OPTIMIZERS:
    cell = results[optimizer]
    nb = cell["naive"][0]; eb = cell["er"][0]
    print(f"\n--- optimizer={optimizer.upper()} ---  baselines: naive={nb:.4f} (f={cell['naive'][1]:.4f})  "
          f"er={eb:.4f} (f={cell['er'][1]:.4f})")
    for mname, _ in MECHS:
        print(f"  {mname}")
        print(f"    {'init':>5s} {'bias':>7s} {'dgate':>7s} | {'standalone':>10s} {'vs naive':>9s} "
              f"{'forget':>7s} | {'er':>8s} {'vs er':>8s} {'forget':>7s}")
        for init in INITS:
            import math
            b = math.log(init / (1 - init))
            dg = init * (1 - init)
            s, sf = cell[(mname, "standalone", init)]
            e, ef = cell[(mname, "er", init)]
            star = " *" if (e - eb) >= 0.02 or (s - nb) >= 0.05 else ""
            print(f"    {init:>5.2f} {b:>7.3f} {dg:>7.4f} | {s:>10.4f} {s - nb:>+9.4f} {sf:>7.4f} "
                  f"| {e:>8.4f} {e - eb:>+8.4f} {ef:>7.4f}{star}")

print("\n--- REFERENCE CHECK (iter3-followup-2; also confirms the gain-meta flag rewiring left plasticity alone) ---")
for (mname, opt, arm, init), ref in REF.items():
    got = results[opt][(mname, arm, init)][0]
    print(f"  {opt} {mname} {arm} init={init}: got={got:.4f} ref={ref:.4f} "
          f"{'MATCH' if abs(got - ref) < 5e-4 else '*** MISMATCH ***'}")

print("\nCaveats: ORACLE task id at train+eval (task-IL-style on the class-IL metric). plasticity+Adam is a "
      "first-order surrogate (grads gated before .step() => Adam moments from the gated grad; the lookahead "
      "inner step is SGD-style) -- SGD is the clean read. 1 seed, sparsity_lambda=0.")
