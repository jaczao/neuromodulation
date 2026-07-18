"""EWC + ER combined baseline (`--method ewc_er`, the new `EWCER` CL method).

Question asked: can EWC (weight-space Fisher anchor) STACKED on ER (replay) beat ER alone?
`EWCER` (prototype/methods.py) runs the gradient step on ER's replay-augmented batch and adds
the EWC quadratic penalty (λ/2)·Σ_i F_i·(θ−θ_i*)² to that batch loss; Fisher/snapshot bookkeeping
is inherited from EWC, the reservoir buffer from ER. It is a plain CL method (no neuromod), so it
flows through cl_train's generic make_cl_method path and composes with both metrics/optimizers.

Settings throughout: seed 42, lr=1e-3, epochs_per_task=5, er_buffer_size=1000, 1 seed (screening).
  - task-IL : output_masking='taskil' (masked train loss + eval masked to each task's 2 classes).
  - class-IL: ER path uses output_masking='none' (10-way eval; ER supplies its own retention).
  - λ chosen per optimizer: Adam λ=1e5 (the CLConfig default, tuned for Adam); SGD λ=1e3 — the
    default λ=1e5 COLLAPSES under SGD (the quadratic penalty dominates the small SGD step and pins
    the net to task 0: task-IL 0.09, all later tasks 0.000). λ is the EWC strength (0 = pure ER).

λ tuning (class-IL) is done on the VALIDATION sequence (make_sequence(7), held-out val split) — never
the test set (non-negotiable rule #1) — then the val-selected λ is reported on the test set.

FINDINGS (see .log for the numbers):
  - task-IL: ER ≈ EWC ≈ EWC+ER, all at the ceiling (~0.97 SGD / ~0.99 Adam). Combining is a wash —
    task-IL removes the shared-head competition, so replay already tops out and the anchor adds ~0.
  - class-IL: combining does NOT beat ER. At the untuned per-opt λ, EWC+ER is BELOW ER
    (SGD −0.088 at λ=1e3, Adam −0.010 at λ=1e5). The λ sweep shows the big SGD loss was purely a
    too-large λ: the best val λ=10 recovers to ≈ER (+0.007 on val) — but that flips to −0.007 on the
    TEST set (a noise-level bump that does not transfer), and under Adam NO λ beats ER at all.
  - Reading: on class-IL, replay owns the output-head bottleneck; a weight-space regulariser at best
    gets out of the way (≈ER), never adds to it. Consistent with the whole pt2/pt3 arc (replay is the
    only lever; regularisation fails class-IL). EWC ALONE on class-IL ≈ Naive (~0.20).

Run: uv run python results/ewc_er_baseline.py   (redirect to results/ewc_er_baseline.log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.data import make_sequence
from prototype.train import cl_train

SEED = 42
LR = 1e-3
EP = 5
BUFFER = 1000
VAL_SEQ_SEED = 7
LAMBDA_ADAM = 1e5   # CLConfig default (tuned for Adam)
LAMBDA_SGD = 1e3    # default 1e5 collapses under SGD
LAMBDA_GRID = [10, 100, 1000, 10000, 100000]


def _cfg(optimizer, masking, ewc_lambda=None, replay_buffer=True) -> CLConfig:
    kw = dict(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer, output_masking=masking)
    if replay_buffer:
        kw["er_buffer_size"] = BUFFER
    if ewc_lambda is not None:
        kw["ewc_lambda"] = ewc_lambda
    return CLConfig(**kw)


def run(tag, config, method, eval_split="test", sequence=None):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=sequence, eval_split=eval_split)
    print(f">>> {tag:44s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


def main():
    # ---- task-IL: er / ewc / ewc_er, both optimizers ----
    print("\n===== task-IL (output_masking=taskil) =====", flush=True)
    for opt in ("sgd", "adam"):
        lam = LAMBDA_SGD if opt == "sgd" else LAMBDA_ADAM
        run(f"[taskil {opt}] er", _cfg(opt, "taskil"), "er")
        run(f"[taskil {opt}] ewc (lam={lam:g})", _cfg(opt, "taskil", lam), "ewc")
        run(f"[taskil {opt}] ewc_er (lam={lam:g})", _cfg(opt, "taskil", lam), "ewc_er")

    # ---- class-IL: er / ewc_er at the per-opt lambda ----
    print("\n===== class-IL (output_masking=none) =====", flush=True)
    for opt in ("sgd", "adam"):
        lam = LAMBDA_SGD if opt == "sgd" else LAMBDA_ADAM
        run(f"[classil {opt}] er", _cfg(opt, "none"), "er")
        run(f"[classil {opt}] ewc_er (lam={lam:g})", _cfg(opt, "none", lam), "ewc_er")

    # ---- class-IL lambda tuning on the VALIDATION sequence (rule #1: never tune on test) ----
    print("\n===== class-IL lambda sweep on VALIDATION (make_sequence(7), --val) =====", flush=True)
    val_seq = make_sequence(VAL_SEQ_SEED)
    for opt in ("sgd", "adam"):
        run(f"[val {opt}] er", _cfg(opt, "none"), "er", eval_split="val", sequence=val_seq)
        for lam in LAMBDA_GRID:
            run(f"[val {opt}] ewc_er lam={lam:g}", _cfg(opt, "none", lam), "ewc_er",
                eval_split="val", sequence=val_seq)

    # ---- report the val-selected candidate (SGD lam=10) on the TEST set ----
    print("\n===== TEST confirm of val-selected candidate =====", flush=True)
    run("[test sgd] ewc_er lam=10 (val-selected)", _cfg("sgd", "none", 10), "ewc_er")


if __name__ == "__main__":
    main()
