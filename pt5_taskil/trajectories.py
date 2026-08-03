"""Per-task accuracy TRAJECTORIES for every task-IL cell, at its val-tuned point, seed 42.

`plast_taskil.py` stores only the final (acc, forget) per cell — `run_cell` redirects `cl_train`'s
stdout to a throwaway buffer — so the A[t,i] matrix it prints ("After task t/T | seen tasks: [...]")
never reaches disk. This re-runs every cell the study reports and captures it.

Covers ALL cells, including the later arms: the 5 baselines plus {plast, dead} x
{nobuf, bufcur, bufown, ercur, erown} x {neuron, synapse}.

Three readings per cell:
  - `avg_seen[t]` = mean accuracy over the t+1 tasks seen so far, after training task t. The running
    version of the reported metric (avg_seen[4] == the study's headline number). pt7_convergence used
    this shape to show CL avg-acc DECAYS with more epochs/task.
  - `first[t]`    = accuracy on task 0 after training task t — the cleanest single retention curve.
  - the full A[t,i] row is kept in the TSV so any other slice can be derived without re-running.

TUNED POINTS ARE READ FROM THE LEDGER, never re-derived: `plast_taskil`'s own selection functions are
called with `run_cell` patched to a LOOKUP-ONLY version, so this script can never start a training run
to fill a tuning gap. If the study has not finished, it exits naming the missing cell instead of
silently training it.

Seed 42 only. Trajectories are for SHAPE; every reported number in the study is 3 seeds, and the
final-column values here should match that study's seed-42 ledger rows exactly (a free consistency
check — they are the same configs).

Run (only after plast_taskil.py has finished): uv run python pt5_taskil/trajectories.py
"""
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                       # noqa: E402

import plast_taskil as S                                 # noqa: E402
from prototype.train import cl_train                     # noqa: E402

TSV = Path(__file__).resolve().parent / "trajectories.tsv"
SEED = 42
LINE = re.compile(r"After task (\d+)/\d+ \| seen tasks: \[([^\]]*)\]")


def lookup_only(stage, cell, lr, ep, lam, nlr, seed, ledger, split):
    """Stand-in for S.run_cell during selection recovery: never trains, only reads the ledger."""
    key = S.key_of(stage, cell, lr, ep, lam, nlr, seed)
    if key not in ledger:
        raise SystemExit(f"NOT LEDGERED YET: {'|'.join(key)}\n"
                         f"Let plast_taskil.py finish before running this.")
    return ledger[key]


def selections():
    """(tuned_main, tuned_lam, tuned_nlr) recovered from the ledger with no training."""
    S.run_cell = lookup_only
    ledger = S.load_ledger()
    buf = io.StringIO()
    with redirect_stdout(buf):                           # selection printing is noise here
        tuned_main = S.tune_main(ledger)
        tuned_lam = S.tune_lambda(ledger, tuned_main)
        tuned_nlr = S.tune_neuro(ledger, tuned_main)
    return tuned_main, tuned_lam, tuned_nlr


def cell_list(tuned_main, tuned_lam, tuned_nlr):
    """Every reported cell as (label, cell, lr, ep, lam, nlr) — mirrors plast_taskil.test().

    `label` is the display/TSV key and `cell` is what build_config consumes; they differ only where
    one cell name is run at two hyperparameters (EWC at its tuned lambda and at the lambda=0 control),
    which would otherwise collide in the results dict.
    """
    cells = [("naive", "naive", None, None),
             ("naive_unmasked", "naive_unmasked", None, None),
             ("er", "er", None, None),
             ("ewc", "ewc", tuned_lam["ewc"], None),
             # EWC's RNG-matched control: the Fisher pass still runs, the penalty is off. It is what
             # showed EWC's +0.0035 is the penalty rather than the Fisher pass shifting the stream,
             # so its trajectory belongs next to EWC's.
             ("ewc_lam0", "ewc", 0.0, None),
             ("ewc_er", "ewc_er", tuned_lam["ewc_er"], None)]
    for arm in S.ARM_ORDER:
        for gran in S.GRANS:
            if arm != "bufown":
                name = S.cell_name("dead", arm, gran)
                cells.append((name, name, None, None))
            name = S.cell_name("plast", arm, gran)
            cells.append((name, name, None, tuned_nlr[(arm, gran)]))
    out = []
    for label, cell, lam, nlr in cells:
        lr, ep = S.main_point(cell, tuned_main)
        out.append((label, cell, lr, ep, lam, nlr))
    return out


def trajectory(cell, lr, ep, lam, nlr):
    config, method = S.build_config(cell, lr, ep, lam, nlr, SEED)
    buf = io.StringIO()
    with redirect_stdout(buf):
        if cell == "naive_unmasked":
            with S.unmasked_loss():
                acc, forget = cl_train(config, method, no_wandb=True, sequence=None,
                                       eval_split="test")
        else:
            acc, forget = cl_train(config, method, no_wandb=True, sequence=None, eval_split="test")
    rows = [[float(v) for v in m.group(2).split(",")] for m in LINE.finditer(buf.getvalue())]
    return rows, acc, forget


def main():
    tuned_main, tuned_lam, tuned_nlr = selections()
    cells = cell_list(tuned_main, tuned_lam, tuned_nlr)
    ledger = S.load_ledger()

    out = ["\t".join(["cell", "after_task", "avg_seen", "first_task", "per_task"])]
    results = {}
    for label, cell, lr, ep, lam, nlr in cells:
        rows, acc, forget = trajectory(cell, lr, ep, lam, nlr)
        results[label] = ([float(np.mean(r)) for r in rows], [r[0] for r in rows], acc)
        for t, r in enumerate(rows):
            out.append(f"{label}\t{t}\t{np.mean(r):.6f}\t{r[0]:.6f}\t"
                       f"{','.join(f'{v:.4f}' for v in r)}")
        # free consistency check: same config as the study's seed-42 test row, so it must match
        key = S.key_of("test", cell, lr, ep, lam, nlr, SEED)
        ref = ledger.get(key)
        # 1e-6 = the ledger's storage precision ("%.6f"): `acc` is full precision, `ref` was rounded
        # on write, so a tighter tolerance flags identical runs as mismatched.
        flag = "" if ref is None else ("  OK" if abs(ref[0] - acc) < 1e-6
                                       else f"  !! MISMATCH vs ledger {ref[0]:.6f}")
        print(f"[done] {label:22s} final={acc:.4f} forget={forget:.4f}{flag}", flush=True)
    TSV.write_text("\n".join(out) + "\n")

    names = [c[0] for c in cells]
    for title, idx in (("AVG ACCURACY OVER TASKS SEEN SO FAR", 0),
                       ("ACCURACY ON TASK 0 AS TRAINING PROCEEDS (retention)", 1)):
        print("\n" + "=" * 84)
        print(f"{title} — task-IL, tuned, seed 42, test set")
        print("=" * 84)
        print(f"{'cell':22s} " + " ".join(f"{'T' + str(t):>8s}" for t in range(5))
              + ("   drop" if idx == 1 else ""))
        for name in names:
            series = results[name][idx]
            tail = f"   {series[0] - series[-1]:+.4f}" if idx == 1 else ""
            print(f"{name:22s} " + " ".join(f"{v:>8.4f}" for v in series) + tail)
    print(f"\nWritten to {TSV.name}. Seed 42 only — shape, not a reported number.")


if __name__ == "__main__":
    main()
