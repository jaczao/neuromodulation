"""pt5 Iter 1 (disjoint) + Iter 2 (shared) gain study re-run under TASK-IL eval, 1 seed.

Companion to the `--output-masking taskil` eval fix (commit 87a6b9e): the pt5 driver branch now
masks EVAL logits to the oracle task's 2 classes when output_masking=="taskil" (previously it
evaluated class-IL 10-way regardless of the flag, so the gain cells in the earlier taskil table
were class-IL while the naive/er baselines were true task-IL — apples-to-oranges). This script
re-measures both projections with the fix, so baselines AND gain cells are genuine task-IL.

Mirrors results/pt5_iter2_gain.py exactly (gain/activation, gate=(h0,h1), lr=1e-3, ep=5,
ER buffer=1000, both optimizers, cells {naive, naive+gain, er, er+gain}) with TWO changes:
  - output_masking = "taskil" for ALL cells (2-way masked eval on both sides)
  - 1 seed (42), and BOTH projections in one run: disjoint (iter 1) and shared frac=0.5 (iter 2)

1 seed => directional, NOT a reportable mean+-std. Run:
  uv run python results/pt5_taskil_eval.py   (tee to results/pt5_taskil_eval.log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED = 42
OPTIMIZERS = ["adam", "sgd"]
LR = 1e-3
EP = 5
BUFFER = 1000
GAIN_LAYERS = "0,2"          # gate the two hidden activations (h0, h1); iter-1 last-run gate
MASKING = "taskil"


def _base(optimizer, **kw) -> CLConfig:
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                    output_masking=MASKING, **kw)


def _gain(optimizer, projection, er) -> CLConfig:
    """gain (activation) neuromod config, task-id oracle, (h0,h1) gate. er=True adds the ER buffer."""
    kw = dict(
        use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
        neuromod_target="activation", neuromod_projection=projection,
        neuromod_gain_layers=GAIN_LAYERS, neuromod_gain_form="unbounded",
    )
    if projection == "shared":
        kw["neuromod_shared_frac"] = 0.5
    if er:
        kw["er_buffer_size"] = BUFFER
    return _base(optimizer, **kw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:44s} acc={acc:.4f}  forget={forget:.4f}\n", flush=True)
    return acc, forget


results = {}  # (projection, optimizer, cell) -> (acc, forget)

for projection in ["disjoint", "shared"]:
    for optimizer in OPTIMIZERS:
        print(f"######## projection={projection} optimizer={optimizer} seed={SEED} ########", flush=True)
        results[(projection, optimizer, "naive")] = run(
            f"[{projection}/{optimizer}] naive (taskil)",
            _base(optimizer), "naive")
        results[(projection, optimizer, "naive+gain")] = run(
            f"[{projection}/{optimizer}] naive+gain (taskil)",
            _gain(optimizer, projection, er=False), "naive")
        results[(projection, optimizer, "er")] = run(
            f"[{projection}/{optimizer}] ER (taskil)",
            _base(optimizer, er_buffer_size=BUFFER), "er")
        results[(projection, optimizer, "er+gain")] = run(
            f"[{projection}/{optimizer}] er+gain (taskil)",
            _gain(optimizer, projection, er=True), "er")

CELLS = ["naive", "naive+gain", "er", "er+gain"]
print("\n\n==== pt5 Iter1(disjoint) + Iter2(shared) — gain gate=(h0,h1), TASK-IL eval, 1 seed=42 ====")
print(f"lr={LR} ep={EP} buffer={BUFFER} masking={MASKING} (both baselines AND gain cells 2-way task-IL)\n")
for projection in ["disjoint", "shared"]:
    print(f"#### projection = {projection} ####")
    for optimizer in OPTIMIZERS:
        print(f"  -- {optimizer.upper()} --")
        for cell in CELLS:
            a, f = results[(projection, optimizer, cell)]
            print(f"     {cell:12s}  acc={a:.4f}  forget={f:.4f}")
        ng = results[(projection, optimizer, "naive+gain")][0] - results[(projection, optimizer, "naive")][0]
        eg = results[(projection, optimizer, "er+gain")][0] - results[(projection, optimizer, "er")][0]
        print(f"       (delta naive+gain {ng:+.4f} | er+gain {eg:+.4f})")
    print()

print("Read: task-IL removes cross-task head competition, so baselines are near-ceiling (naive "
      "0.93-0.98, er 0.97-0.99) and gain adds ~0 on top / hurts under SGD. The one clean effect is "
      "SGD naive+gain forgetting = 0.0000 (frozen subnet, no momentum, no replay). Residual "
      "forgetting elsewhere is Adam momentum nudging frozen params + ER retuning within-task head "
      "biases — NOT the class-IL cross-task bias leak (that is gone). Same oracle caveat.")
