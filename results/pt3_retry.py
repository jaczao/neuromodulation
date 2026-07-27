"""pt3 RETRY — Iterations 6-10 at a VAL-TUNED operating point, 3 seeds, both optimizers (user-requested).

Motivation. The original pt3 iterations (6 logit calibration, 7 importance-gated plasticity,
8 task-inferred routing, 9 recency driver on the calibrator, 10 boundary-detected consolidation)
all ran at a FIXED, inherited lr=1e-3 / ep=5 under ADAM ONLY, and their reference baselines were
frozen scalars (Naive 0.1979, naive+masked 0.3777, ER 0.9023). Two things have changed since:
  (a) pt7_tuned_syn showed the CL regime was never val-tuned, and that tuning ER lifts SGD
      0.723 -> 0.903 and DISSOLVES apparent mechanism wins that were really closing an
      under-fit (the "gate compensates for an untuned baseline" pattern);
  (b) `--optimizer` matters enormously under a masked loss (CLAUDE.md: naive+masked = 0.39 Adam
      vs 0.63 SGD), so an Adam-only verdict is half the picture.
This script re-runs all five mechanisms at the tuned point, on both optimizers, over 3 seeds.

Arms (fixed by the request):
  A "naive+masked" — method=naive, output_masking='loss'  (the pt3 standalone bar, lever B)
  B "er"           — method=er,   output_masking='none'   (replay, NO masked loss)
Comparison baselines: naive+masked, er, and EWC+ER (`method=ewc_er`, the later-added combined
method) — the last one answers "does a weight-space anchor stacked on replay do better than any
of these neuromodulators?".

Protocol (obeys the non-negotiable rules):
  TUNE  — sequence=make_sequence(7), val_frac=0.1, eval on the HELD-OUT val split, seed 42,
          NEVER the test set (rule #1).
          * main point for arm A (naive+masked): swept here per optimizer, lr x ep, same grid as
            pt7_tuned_syn (rule #3). TUNED_MAIN has no ("classil","naive","sgd") entry and its
            ("classil","naive","adam") entry is a transferred ER point, so neither is reused.
          * main point for arm B (er): reused from configs.TUNED_MAIN — already val-tuned by
            pt7_tuned_syn with this exact grid at output_masking='none'.
          * per-mechanism hyperparameter: the two lambda-bearing mechanisms (importance,
            consolidation) and the EWC+ER baseline each get a 5-point lambda sweep at their arm's
            tuned main point, per optimizer AND per arm (identical budget, rule #3). The other
            three mechanisms (logit, task_route, logit+recency) have no free hyperparameter.
  REPORT— default sequence, full train set, official MNIST test set, seeds {42,43,44} (rule #5).

Everything routes through prototype/train.py's real pt3 branches (no re-implementation), so this
measures the same mechanisms the original iterations measured.

Ledger results/pt3_retry_results.tsv; `--resume` skips completed rows; `--part` chunks the run
(Bash 600s timeout DETACHES rather than kills, so chunk deliberately).

Run: uv run python results/pt3_retry.py --part all --resume   (redirect to results/pt3_retry.log)
"""
import argparse
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prototype.configs import CLConfig, TUNED_MAIN  # noqa: E402
from prototype.data import make_sequence            # noqa: E402
from prototype.train import cl_train                # noqa: E402

TSV = Path(__file__).resolve().parent / "pt3_retry_results.tsv"
COLS = ["stage", "mech", "arm", "opt", "lr", "ep", "lam", "seed", "acc", "forget", "extra"]

SEEDS = (42, 43, 44)
# 1-seed MPS run-to-run spread (CLAUDE.md pt6-followups: identical configs differ by ~0.007-0.016).
# Used as the tie-break band when selecting a tuned point, so selection never chases noise.
NOISE_FLOOR = 0.007
TUNE_SEED = 42
VAL_SEQ_SEED = 7
VAL_FRAC = 0.1
BUFFER = 1000

# Main-point grid for arm A. Started as pt7_tuned_syn's ER grid (rule #3) and was EXTENDED TWO
# DECADES DOWNWARD after the first pass selected the grid FLOOR in both optimizers with val acc
# monotone increasing as lr fell (sgd 3e-3 0.5232 > 1e-2 0.4276 > 3e-2 0.4274; adam 3e-4 0.4653 >
# 1e-3 0.3777 > 3e-3 0.2346). A boundary selection is a truncated grid, not an optimum. The naive
# arm wants a much smaller lr than ER does — replay tolerates (and needs) big steps, while an
# unreplayed net forgets harder the larger the step (CLAUDE.md: "naive DROPS at the tuned lr").
# Historical corroboration: naive+masked SGD@1e-3 = 0.6296 on test.
GRID = {
# SGD's peak came out INTERIOR on the first extension (1e-3/ep5, neighbours lower both sides, and
# it reproduces the historical naive+masked SGD test number 0.6296 at that lr). ADAM stayed pinned
# at the floor for a second pass (monotone 3e-3 0.235 -> 3e-5 0.579, forgetting falling 0.696 ->
# 0.116), so it is extended two decades further. Adam's per-parameter normalisation overwrites old
# tasks fast, so the unreplayed arm keeps wanting a smaller step; the turnover has to come once the
# step is too small to fit the current task at all.
    "sgd":  {"lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "ep": [5, 10, 20]},
    "adam": {"lr": [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3], "ep": [5, 10, 20]},
}

# lambda grids — 5 decade-spaced points each, so every lambda-bearing cell gets the SAME BUDGET
# (rule #3), but centred per optimizer. The tuned SGD lr (3e-2) is 100x the tuned Adam lr (3e-4),
# and a quadratic/importance penalty competes with the step size, so a grid centred for Adam is
# entirely in the collapsed regime under SGD (verified: consolidation lam=100 and ewc_er lam=1e3
# both hit chance 0.0927 at the tuned SGD lr). ewc_er_baseline.py set the same precedent
# (Adam lam=1e5, SGD lam=1e3). Adam rows keep each mechanism's ORIGINAL pt3 grid.
LAM_GRID = {
    ("importance",    "adam"): [1.0, 10.0, 100.0, 1e3, 1e4],    # iter-7's original grid
    ("importance",    "sgd"):  [0.1, 1.0, 10.0, 100.0, 1e3],
    ("consolidation", "adam"): [0.1, 1.0, 10.0, 100.0, 1e3],     # iter-10's grid, extended to 5
    ("consolidation", "sgd"):  [1e-3, 1e-2, 0.1, 1.0, 10.0],
    ("ewc_er",        "adam"): [10.0, 100.0, 1e3, 1e4, 1e5],     # ewc_er_baseline's grid
    ("ewc_er",        "sgd"):  [0.1, 1.0, 10.0, 100.0, 1e3],
}

# mechanism -> the neuromod config it selects in prototype/train.py
MECHS = {
    "logit":         dict(use_neuromod=True, neuromod_target="logit"),                          # iter 6
    "importance":    dict(use_neuromod=True, neuromod_target="importance"),                     # iter 7
    "task_route":    dict(use_neuromod=True, neuromod_target="task_route"),                     # iter 8
    "logit_recency": dict(use_neuromod=True, neuromod_target="logit",
                          neuromod_driver="recency"),                                           # iter 9
    "consolidation": dict(use_neuromod=True, neuromod_target="consolidation"),                  # iter 10
}
# SCOPE (user-requested): retry only Iteration 6 (logit calibration) and Iteration 10
# (boundary-detected consolidation). The other three branches stay defined above (and their
# hardcoded-Adam fix stands), so widening the scope later is a one-line change here.
MECH_ORDER = ["logit", "consolidation"]
LAM_MECHS = ("consolidation",)          # iter 6 has no free hyperparameter to tune

# arm -> (method, output_masking)
ARMS = {"naive_masked": ("naive", "loss"), "er": ("er", "none")}

# debug lines worth keeping alongside the accuracy
DEBUG_PATTERNS = [
    (re.compile(r"routing\) accuracy = ([0-9.]+)"), "route_acc"),
    (re.compile(r"mean gate = ([0-9.eE+-]+)"), "gate_mean"),
    (re.compile(r"boundaries detected = (\d+)"), "boundaries"),
]


# --------------------------------------------------------------------------- ledger
def load_ledger():
    if not TSV.exists():
        return {}
    rows = {}
    for line in TSV.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        f = line.split("\t")
        rows[tuple(f[:8])] = (float(f[8]), float(f[9]))
    return rows


def append(key, acc, forget, extra=""):
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with TSV.open("a") as fh:
        fh.write("\t".join(list(key) + [f"{acc:.6f}", f"{forget:.6f}", extra]) + "\n")


def key_of(stage, mech, arm, opt, lr, ep, lam, seed):
    return (stage, mech, arm, opt, f"{lr:g}", str(ep), f"{lam:g}" if lam is not None else "-",
            str(seed))


# --------------------------------------------------------------------------- one cell
def run_cell(stage, mech, arm, opt, lr, ep, lam, seed, ledger, split="test"):
    """Run (or skip, if ledgered) one configuration. Returns (acc, forget)."""
    key = key_of(stage, mech, arm, opt, lr, ep, lam, seed)
    if key in ledger:
        acc, forget = ledger[key]
        print(f"[skip] {'|'.join(key)} acc={acc:.4f}")
        return acc, forget

    method, masking = ARMS[arm]
    kw = dict(seed=seed, lr=lr, epochs_per_task=ep, optimizer=opt,
              output_masking=masking, er_buffer_size=BUFFER)
    if mech in MECHS:
        kw.update(MECHS[mech])
    if mech == "ewc_er":
        method = "ewc_er"
        kw["ewc_lambda"] = lam
    elif lam is not None:
        kw["neuromod_importance_lambda"] = lam

    cfg = CLConfig(**kw)
    sequence = make_sequence(VAL_SEQ_SEED) if split == "val" else None

    buf = io.StringIO()
    with redirect_stdout(buf):
        acc, forget = cl_train(cfg, method, no_wandb=True, sequence=sequence, eval_split=split)
    out = buf.getvalue()

    extra = []
    for pat, name in DEBUG_PATTERNS:
        m = pat.search(out)
        if m:
            extra.append(f"{name}={m.group(1)}")
    extra = ",".join(extra)

    append(key, acc, forget, extra)
    ledger[key] = (acc, forget)
    print(f"[run ] {'|'.join(key)} acc={acc:.4f} forget={forget:.4f} {extra}")
    return acc, forget


# --------------------------------------------------------------------------- stages
def tune_main(ledger):
    """Arm A (naive+masked) main point per optimizer, on the val split. Arm B reuses TUNED_MAIN."""
    best = {}
    for opt in ("sgd", "adam"):
        results = []
        for lr in GRID[opt]["lr"]:
            for ep in GRID[opt]["ep"]:
                acc, _ = run_cell("tune_main", "baseline", "naive_masked", opt, lr, ep, None,
                                  TUNE_SEED, ledger, split="val")
                results.append((acc, lr, ep))
        # Selection with an explicit NOISE-FLOOR TIE-BREAK. The raw 2D argmax is not trustworthy
        # here: (lr, epochs) trade off along a ridge (what matters is roughly total movement
        # lr*epochs), so the argmax slides down that ridge on 1-seed noise. Adam's top three cells
        # spanned 0.0068 < NOISE_FLOOR. Rule: among cells within NOISE_FLOOR of the best, take the
        # SMALLEST epochs_per_task (cheapest, and pt7_convergence showed CL avg-acc DECAYS with
        # more epochs/task), breaking any remaining tie on val acc. Selection stays on val only.
        top = max(results)[0]
        tied = [r for r in results if r[0] >= top - NOISE_FLOOR]
        acc, lr, ep = min(tied, key=lambda r: (r[2], -r[0]))
        best[opt] = dict(lr=lr, epochs_per_task=ep, val=acc)
        note = "" if acc == top else f" [tie-break: {len(tied)} cells within {NOISE_FLOOR} of {top:.4f}]"
        print(f"\n>>> TUNED naive+masked [{opt}]: lr={lr:g} ep={ep} (val {acc:.4f}){note}\n")
    return best


def main_point(arm, opt, tuned_naive):
    """(lr, ep) for an arm+optimizer: arm B from configs.TUNED_MAIN, arm A from tune_main."""
    if arm == "er":
        p = TUNED_MAIN[("classil", "er", opt)]
        return p["lr"], p["epochs_per_task"]
    p = tuned_naive[opt]
    return p["lr"], p["epochs_per_task"]


def tune_lambda(ledger, tuned_naive):
    """5-point lambda sweep per (mechanism, arm, optimizer) at that arm's tuned main point."""
    best = {}
    cells = [(m, a) for m in LAM_MECHS for a in ARMS] + [("ewc_er", "er")]
    for mech, arm in cells:
        for opt in ("sgd", "adam"):
            lr, ep = main_point(arm, opt, tuned_naive)
            results = []
            for lam in LAM_GRID[(mech, opt)]:
                acc, _ = run_cell("tune_lam", mech, arm, opt, lr, ep, lam, TUNE_SEED, ledger,
                                  split="val")
                results.append((acc, lam))
            acc, lam = max(results)
            best[(mech, arm, opt)] = lam
            print(f"\n>>> TUNED lambda [{mech}|{arm}|{opt}] = {lam:g} (val {acc:.4f})\n")
    return best


def test(ledger, tuned_naive, tuned_lam):
    """3-seed test-set report: baselines + all five mechanisms x both arms x both optimizers."""
    table = {}

    def three(mech, arm, opt, lam):
        lr, ep = main_point(arm, opt, tuned_naive)
        accs, forgets, extras = [], [], []
        for s in SEEDS:
            a, f = run_cell("test", mech, arm, opt, lr, ep, lam, s, ledger, split="test")
            accs.append(a); forgets.append(f)
        table[(mech, arm, opt)] = (float(np.mean(accs)), float(np.std(accs)),
                                   float(np.mean(forgets)), lr, ep, lam)
        return table[(mech, arm, opt)]

    for opt in ("sgd", "adam"):
        # baselines
        three("baseline", "naive_masked", opt, None)
        three("baseline", "er", opt, None)
        three("ewc_er", "er", opt, tuned_lam[("ewc_er", "er", opt)])
        # mechanisms
        for mech in MECH_ORDER:
            for arm in ("naive_masked", "er"):
                lam = tuned_lam.get((mech, arm, opt))
                three(mech, arm, opt, lam)
    return table


def report(table):
    print("\n" + "=" * 100)
    print("pt3 RETRY — class-IL, tuned operating point, 3 seeds (42/43/44), test set")
    print("=" * 100)
    for opt in ("sgd", "adam"):
        for arm in ("naive_masked", "er"):
            base = table[("baseline", arm, opt)]
            print(f"\n---- {opt.upper()} | arm = {arm} | baseline {base[0]:.4f}±{base[1]:.4f} "
                  f"(lr={base[3]:g} ep={base[4]}) ----")
            print(f"{'mechanism':16s} {'acc':>16s} {'forget':>8s} {'lam':>7s} {'delta_vs_base':>14s}")
            print(f"{'(baseline)':16s} {base[0]:>9.4f}±{base[1]:.4f} {base[2]:>8.4f} "
                  f"{'-':>7s} {'-':>14s}")
            if arm == "er":
                e = table[("ewc_er", "er", opt)]
                print(f"{'EWC+ER':16s} {e[0]:>9.4f}±{e[1]:.4f} {e[2]:>8.4f} "
                      f"{e[5]:>7g} {e[0] - base[0]:>+14.4f}")
            for mech in MECH_ORDER:
                r = table[(mech, arm, opt)]
                lam = f"{r[5]:g}" if r[5] is not None else "-"
                print(f"{mech:16s} {r[0]:>9.4f}±{r[1]:.4f} {r[2]:>8.4f} {lam:>7s} "
                      f"{r[0] - base[0]:>+14.4f}")
    print("\nBars (pt3 convention): standalone must beat the naive+masked baseline; "
          "+ER must beat ER by >= +0.02.")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "tune-main", "tune-lam", "test", "report"])
    ap.add_argument("--resume", action="store_true", help="skip cells already in the ledger")
    args = ap.parse_args()

    ledger = load_ledger() if args.resume else {}

    tuned_naive = None
    if args.part in ("all", "tune-main"):
        tuned_naive = tune_main(ledger)
    if args.part == "tune-main":
        return

    if tuned_naive is None:                      # recover the selection from the ledger
        tuned_naive = tune_main(ledger)          # all cells ledgered -> pure lookup, no training

    tuned_lam = None
    if args.part in ("all", "tune-lam"):
        tuned_lam = tune_lambda(ledger, tuned_naive)
    if args.part == "tune-lam":
        return
    if tuned_lam is None:
        tuned_lam = tune_lambda(ledger, tuned_naive)

    table = test(ledger, tuned_naive, tuned_lam)
    report(table)


if __name__ == "__main__":
    main()
