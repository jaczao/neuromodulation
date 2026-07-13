"""pt5 Iteration 2 — shared backbone + private capacity (`projection=shared`, `shared_frac=0.5`).

Mirrors the LAST run of Iteration 1 exactly (the 3-seed, both-optimizers gain study), changing ONLY
the projection from `disjoint` to `shared`:
  - target = gain (activation), per-neuron, gate = (h0, h1)  [`--neuromod-gain-layers 0,2`]
  - 3 seeds {42, 43, 44} x 2 optimizers {adam, sgd} x {naive, naive+gain, er, er+gain} = 24 runs
  - lr=1e-3, ep=5, ER buffer=1000, class-IL eval, task-id oracle
Baselines are same-optimizer per condition: naive+masked-loss (standalone bar) and ER (the +ER bar).
gain_form is inert under a fixed binary P (disjoint/shared collapse to a {0,1} gate); kept for parity
with iter 1. Reports acc mean+-std over the 3 seeds per optimizer, matching the iter-1 finding format.

Run: uv run python results/pt5_iter2_gain.py   (logs also to results/pt5_iter2_gain.log)
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEEDS = [42, 43, 44]
OPTIMIZERS = ["adam", "sgd"]
LR = 1e-3
EP = 5
BUFFER = 1000
PROJECTION = "shared"       # Iteration 2 (iter 1 was "disjoint")
SHARED_FRAC = 0.5
GAIN_LAYERS = "0,2"         # gate the two hidden activations (h0, h1); iter-1 last-run gate


def _base(seed, optimizer, **kw) -> CLConfig:
    return CLConfig(seed=seed, lr=LR, epochs_per_task=EP, optimizer=optimizer, **kw)


def _gain(seed, optimizer, masked) -> CLConfig:
    """gain (activation) neuromod config. masked=True -> naive + masked-loss (standalone);
    masked=False -> ER, masking off. Shared projection, (h0,h1) gate, task-id oracle."""
    kw = dict(
        use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
        neuromod_target="activation", neuromod_projection=PROJECTION,
        neuromod_shared_frac=SHARED_FRAC, neuromod_gain_layers=GAIN_LAYERS,
        neuromod_gain_form="unbounded",
    )
    if masked:
        return _base(seed, optimizer, output_masking="loss", **kw)
    return _base(seed, optimizer, output_masking="none", er_buffer_size=BUFFER, **kw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:40s} acc={acc:.4f}  forget={forget:.4f}\n")
    return acc, forget


def agg(xs):
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


# cell -> {optimizer -> [per-seed acc]}
CELLS = ["naive", "naive+gain", "er", "er+gain"]
acc = {c: {o: [] for o in OPTIMIZERS} for c in CELLS}
fgt = {c: {o: [] for o in OPTIMIZERS} for c in CELLS}

for optimizer in OPTIMIZERS:
    for seed in SEEDS:
        print(f"######## optimizer={optimizer} seed={seed} ########")
        a, f = run(f"[{optimizer} s{seed}] naive+masked-loss",
                   _base(seed, optimizer, output_masking="loss"), "naive")
        acc["naive"][optimizer].append(a); fgt["naive"][optimizer].append(f)

        a, f = run(f"[{optimizer} s{seed}] naive+gain (shared)",
                   _gain(seed, optimizer, masked=True), "naive")
        acc["naive+gain"][optimizer].append(a); fgt["naive+gain"][optimizer].append(f)

        a, f = run(f"[{optimizer} s{seed}] ER",
                   _base(seed, optimizer, output_masking="none", er_buffer_size=BUFFER), "er")
        acc["er"][optimizer].append(a); fgt["er"][optimizer].append(f)

        a, f = run(f"[{optimizer} s{seed}] er+gain (shared)",
                   _gain(seed, optimizer, masked=False), "er")
        acc["er+gain"][optimizer].append(a); fgt["er+gain"][optimizer].append(f)

print("\n\n==== pt5 ITERATION 2 (projection=shared, frac=0.5) — gain, gate=(h0,h1) ====")
print(f"seeds={SEEDS}  lr={LR}  ep={EP}  buffer={BUFFER}  class-IL eval, task-id oracle\n")
for optimizer in OPTIMIZERS:
    nb, nbs = agg(acc["naive"][optimizer])
    ng, ngs = agg(acc["naive+gain"][optimizer])
    eb, ebs = agg(acc["er"][optimizer])
    eg, egs = agg(acc["er+gain"][optimizer])
    print(f"-- {optimizer.upper()} --")
    print(f"  naive           {nb:.4f} +- {nbs:.4f}")
    print(f"  naive+gain      {ng:.4f} +- {ngs:.4f}   (delta {ng - nb:+.4f})")
    print(f"  er              {eb:.4f} +- {ebs:.4f}")
    print(f"  er+gain         {eg:.4f} +- {egs:.4f}   (delta {eg - eb:+.4f})")
    print(f"  forgetting: naive+gain {agg(fgt['naive+gain'][optimizer])[0]:.4f}  "
          f"er+gain {agg(fgt['er+gain'][optimizer])[0]:.4f}\n")

print("accept-for-confirm bars: standalone naive+gain > naive+masked-loss; +ER er+gain > ER by >=2pts "
      "(same-optimizer). Compare against iter-1 (disjoint) to see whether partial sharing helps or hurts.")
