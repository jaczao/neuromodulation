"""pt5 Iteration 3 (LEARNED projection) — PLASTICITY, TASK-IL, SGD, at a VAL-TUNED operating point.

User-requested. The original `results/pt5_iter3.py` ran its task-IL plasticity cells at a FIXED,
inherited lr=1e-3 / ep=5 with the neuromod net pinned to `CLConfig.neuromod_lr=1e-3`, against
baselines that were never tuned either. Two things since then say that verdict cannot be trusted as
stated:
  (a) pt7_tuned_syn / pt3_retry: tuning the BASELINE is what dissolves apparent mechanism wins — the
      untuned baseline leaves a gap the mechanism was really just closing (ER-SGD 0.723 -> 0.903).
  (b) pt7_tuned_neuro: the neuromod net's OWN lr is a real, separately-tunable axis (it has an
      interior optimum), and `TUNED_MAIN` has NO ("taskil", *, "sgd") entry at all, so nothing here
      could be looked up.
This script therefore re-runs the task-IL / SGD plasticity slice with BOTH nets val-tuned, over
3 seeds, against four baselines (naive, ER, EWC, EWC+ER) and an RNG-matched dead-gate control.

MECHANISM (the only mechanism arm, per the request: "standalone buf cur")
  pt5 iter-3 learned-P plasticity, STANDALONE (naive main net, NO replay on the backbone) with the
  modulator-only replay meta-loss `--neuromod-meta-replay`: a reservoir buffer feeds the LOOKAHEAD
  meta-loss that trains ONLY the projection P (retention signal), while the main net steps naive on
  the current task. "buf" = that meta-replay buffer; "cur" = the gate applied inside the meta-loss is
  the CURRENT task's row P[t] — `neuromod_er_task_id` is replay-gated in `cl_train`, so it cannot
  fire on a naive run, and plasticity's meta path is not per-task (unlike gain's). The `[pt5 debug]`
  line prints `er_task_id=False` for these cells, which is the tell that they are the `cur` arm.
  Two granularities: per-NEURON (alpha per hidden unit, layers 0,2,4, scope=both — reaches net.4's
  head COLUMNS via the outgoing coupling) and per-SYNAPSE (alpha per weight, hidden layers 0,2 only;
  an explicit head gate fights replay, cf. iter-1 weight_mask+ER -0.61).

  `neuromod_plasticity_init` stays at 0.5 (NOT re-swept): `results/pt5_plast_init.py` already swept
  it {0.1..0.99} under SGD and found the STANDALONE optimum is the interior point 0.5, and that the
  init never rescues plasticity. Re-sweeping it here would spend budget re-deriving a known answer.

DEAD-GATE CONTROL (non-negotiable rule #10)
  `dead-neuron` / `dead-synapse` = the SAME mechanism config with `neuromod_lr = 0.0` and
  `neuromod_plasticity_init = 0.9999`. The modulator is CONSTRUCTED (same torch RNG consumed), the
  meta-replay buffer is filled and sampled (same `random` stream consumed), the lookahead runs — but
  the meta-optimizer's step size is zero, so P stays at its zero init and the gate is pinned at
  alpha = sigmoid(logit(0.9999) + 0) = 0.9999 for every unit, i.e. a 0.9999x uniform LR rescale =
  numerically the naive baseline. That makes it the RNG-MATCHED baseline for the mechanism, which
  the plain `naive` run is NOT (CLAUDE.md: constructing an unused modulator shifts the replay draws,
  worth ~0.002 at width 400 and ~0.06 at a destabilised operating point). Report `d-dead` as the
  mechanism's effect and `d-naive` only as context.

PROTOCOL (rules #1, #3, #5)
  TUNE  (seed 42, VAL sequence make_sequence(7), val_frac=0.1, eval on the held-out val split — the
         test set is never touched during selection):
    1. main lr x ep for `naive` and `er`, grid lr {1e-3,3e-3,1e-2,3e-2,1e-1} x ep {5,10,20}
       (5x3 = 15 cells each, identical budget). Selection uses pt3_retry's NOISE-FLOOR TIE-BREAK:
       among cells within 0.007 val of the best, take the smallest epochs_per_task, then the best
       val acc — (lr, ep) trade off along a ridge and the raw 2D argmax slides on 1-seed noise.
       A selection landing on a grid EDGE is flagged: that is a truncated grid, not an optimum.
    2. `ewc` inherits naive's main point and `ewc_er` inherits er's (they are those methods plus a
       weight-space penalty), then each gets a 5-point ewc_lambda sweep — the same arrangement
       `results/pt3_retry.py` used for its EWC+ER baseline, and it keeps the per-method budget equal.
    3. NEURO lr for each granularity: main FROZEN at naive's tuned point (the mechanism's backbone
       is naive), sweep `neuromod_lr` over {1e-4,3e-4,1e-3,3e-3,1e-2}. NOTE the pt5 plasticity
       meta-optimizer is Adam by construction in `prototype/train.py` (frozen code) even when the
       MAIN net is SGD, so this tunes an Adam neuro lr under an SGD main net — the same split
       pt7_tuned_neuro used. The dead control is not tuned (its lr is 0 by definition).
  REPORT (default task order, full train set, official MNIST test set, seeds {42,43,44}, mean +/- std).

Everything routes through `prototype/train.py`'s real pt5 branch — no re-implementation — so this
measures the same mechanism `results/pt5_iter3.py` measured, only at a tuned operating point.
`prototype/` and `results/` are FROZEN (rule #9): this study is NEW code in its own package and
imports the frozen path read-only, exactly as `driver_traces/` does.

Ledger pt5_taskil/plast_taskil_results.tsv; `--resume` skips completed rows; `--part` chunks the run
(a Bash 600s timeout DETACHES rather than kills, so chunk deliberately).

Run: uv run python pt5_taskil/plast_taskil.py --part all --resume  (redirect to plast_taskil.log)
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prototype.configs import CLConfig            # noqa: E402
from prototype.data import make_sequence          # noqa: E402
from prototype.train import cl_train              # noqa: E402

TSV = Path(__file__).resolve().parent / "plast_taskil_results.tsv"
COLS = ["stage", "cell", "opt", "lr", "ep", "lam", "nlr", "seed", "acc", "forget"]

OPT = "sgd"                     # the requested optimizer (main net)
METRIC = "taskil"               # masked-CE train + 2-way masked eval
SEEDS = (42, 43, 44)
TUNE_SEED = 42
VAL_SEQ = make_sequence(7)      # rule #1: tuning order, never the reporting order
VAL_FRAC = 0.1
BUFFER = 1000                   # matches every pt5 study (the CLI default 200 is NOT comparable)
NOISE_FLOOR = 0.007             # 1-seed MPS run-to-run spread; the tie-break band

MAIN_GRID = {"lr": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], "ep": [5, 10, 20]}
# 5 decade-ish points, centred on the historical SGD picks (ewc_er_baseline used lam=1e3 under SGD;
# pt3_retry found lam>=1e3 collapses to chance at a tuned SGD lr). Same budget for both EWC arms.
LAM_GRID = [0.1, 1.0, 10.0, 100.0, 1e3]
NEURO_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

# mechanism -> the pt5 config knobs that select it in prototype/train.py
MECH_KW = {
    "neuron": dict(neuromod_granularity="neuron", neuromod_plasticity_layers="0,2,4",
                   neuromod_plasticity_scope="both"),
    "synapse": dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2"),
}
GRANS = ("neuron", "synapse")


# --------------------------------------------------------------------------- ledger
def load_ledger():
    if not TSV.exists():
        return {}
    rows = {}
    for line in TSV.read_text().splitlines()[1:]:
        if line.strip():
            f = line.split("\t")
            rows[tuple(f[:8])] = (float(f[8]), float(f[9]))
    return rows


def append(key, acc, forget):
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with TSV.open("a") as fh:
        fh.write("\t".join(list(key) + [f"{acc:.6f}", f"{forget:.6f}"]) + "\n")


def key_of(stage, cell, lr, ep, lam, nlr, seed):
    g = lambda v: "-" if v is None else f"{v:g}"          # noqa: E731
    return (stage, cell, OPT, f"{lr:g}", str(ep), g(lam), g(nlr), str(seed))


# --------------------------------------------------------------------------- one cell
def build_config(cell, lr, ep, lam, nlr, seed):
    """(CLConfig, method_name) for a cell name. Cells: naive | er | ewc | ewc_er |
    plast-<gran> (mechanism, buf-cur) | dead-<gran> (RNG-matched inert gate)."""
    kw = dict(seed=seed, lr=lr, epochs_per_task=ep, optimizer=OPT,
              output_masking=METRIC, er_buffer_size=BUFFER)
    if cell in ("naive", "er"):
        return CLConfig(**kw), cell
    if cell in ("ewc", "ewc_er"):
        return CLConfig(ewc_lambda=lam, **kw), cell
    kind, gran = cell.split("-")
    kw.update(dict(use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                   neuromod_target="plasticity", neuromod_projection="learned",
                   neuromod_meta_replay=True, **MECH_KW[gran]))
    if kind == "dead":
        # Inert gate, everything else identical: meta-optimizer step size 0 (P frozen at its zero
        # init) and the gate pinned at ~1.0, so the main net's update is the naive one.
        kw.update(neuromod_lr=0.0, neuromod_plasticity_init=0.9999)
    else:
        kw.update(neuromod_lr=nlr, neuromod_plasticity_init=0.5)
    return CLConfig(**kw), "naive"                      # standalone: naive backbone


def run_cell(stage, cell, lr, ep, lam, nlr, seed, ledger, split):
    key = key_of(stage, cell, lr, ep, lam, nlr, seed)
    if key in ledger:
        acc, forget = ledger[key]
        print(f"[skip] {'|'.join(key)} acc={acc:.4f}", flush=True)
        return acc, forget
    config, method = build_config(cell, lr, ep, lam, nlr, seed)
    sequence = VAL_SEQ if split == "val" else None
    config.val_frac = VAL_FRAC
    buf = io.StringIO()
    with redirect_stdout(buf):
        acc, forget = cl_train(config, method, no_wandb=True, sequence=sequence, eval_split=split)
    append(key, acc, forget)
    ledger[key] = (acc, forget)
    print(f"[run ] {'|'.join(key)} acc={acc:.4f} forget={forget:.4f}", flush=True)
    return acc, forget


# --------------------------------------------------------------------------- stage 1: main net
def _edge_warn(label, value, grid):
    if value in (grid[0], grid[-1]):
        print(f"    !! {label}={value:g} is at a GRID EDGE — truncated grid, not an optimum; "
              f"extend before trusting it", flush=True)


def tune_main(ledger):
    """lr x ep on the val split for the two base methods. Returns {cell: dict(lr, epochs, val)}."""
    best = {}
    for cell in ("naive", "er"):
        results = []
        for lr in MAIN_GRID["lr"]:
            for ep in MAIN_GRID["ep"]:
                acc, _ = run_cell("tune_main", cell, lr, ep, None, None, TUNE_SEED, ledger, "val")
                results.append((acc, lr, ep))
        top = max(results)[0]
        tied = [r for r in results if r[0] >= top - NOISE_FLOOR]
        acc, lr, ep = min(tied, key=lambda r: (r[2], -r[0]))
        best[cell] = dict(lr=lr, epochs_per_task=ep, val=acc)
        note = "" if acc == top else f"  [tie-break: {len(tied)} cells within {NOISE_FLOOR} of {top:.4f}]"
        print(f"\n>>> TUNED main [{cell}]: lr={lr:g} ep={ep} (val {acc:.4f}){note}", flush=True)
        _edge_warn("lr", lr, MAIN_GRID["lr"])
        _edge_warn("ep", ep, MAIN_GRID["ep"])
        print(flush=True)
    return best


def main_point(cell, tuned_main):
    """(lr, ep) for a cell. EWC inherits naive's point, EWC+ER inherits er's, the mechanism and its
    dead control inherit naive's (their backbone is a naive main net)."""
    base = {"er": "er", "ewc_er": "er"}.get(cell, "naive")
    p = tuned_main[base]
    return p["lr"], p["epochs_per_task"]


# --------------------------------------------------------------------------- stage 2: ewc lambda
def tune_lambda(ledger, tuned_main):
    best = {}
    for cell in ("ewc", "ewc_er"):
        lr, ep = main_point(cell, tuned_main)
        results = []
        for lam in LAM_GRID:
            acc, _ = run_cell("tune_lam", cell, lr, ep, lam, None, TUNE_SEED, ledger, "val")
            results.append((acc, lam))
        acc, lam = max(results)
        best[cell] = lam
        print(f"\n>>> TUNED lambda [{cell}] = {lam:g} (val {acc:.4f}) at lr={lr:g} ep={ep}", flush=True)
        _edge_warn("lambda", lam, LAM_GRID)
        print(flush=True)
    return best


# --------------------------------------------------------------------------- stage 3: neuro lr
def tune_neuro(ledger, tuned_main):
    best = {}
    lr, ep = tuned_main["naive"]["lr"], tuned_main["naive"]["epochs_per_task"]
    for gran in GRANS:
        results = []
        for nlr in NEURO_LRS:
            acc, _ = run_cell("tune_neuro", f"plast-{gran}", lr, ep, None, nlr, TUNE_SEED,
                              ledger, "val")
            results.append((acc, nlr))
        acc, nlr = max(results)
        best[gran] = nlr
        print(f"\n>>> TUNED neuro_lr [plast-{gran}] = {nlr:g} (val {acc:.4f}) "
              f"at main lr={lr:g} ep={ep}", flush=True)
        _edge_warn("neuro_lr", nlr, NEURO_LRS)
        print(flush=True)
    return best


# --------------------------------------------------------------------------- stage 4: test report
def test(ledger, tuned_main, tuned_lam, tuned_nlr):
    table = {}
    cells = [("naive", None, None), ("er", None, None),
             ("ewc", tuned_lam["ewc"], None), ("ewc_er", tuned_lam["ewc_er"], None)]
    for gran in GRANS:
        cells.append((f"dead-{gran}", None, None))
        cells.append((f"plast-{gran}", None, tuned_nlr[gran]))
    for cell, lam, nlr in cells:
        lr, ep = main_point(cell, tuned_main)
        accs, forgets = [], []
        for s in SEEDS:
            a, f = run_cell("test", cell, lr, ep, lam, nlr, s, ledger, "test")
            accs.append(a); forgets.append(f)
        table[cell] = dict(accs=accs, mean=float(np.mean(accs)), std=float(np.std(accs)),
                           forget=float(np.mean(forgets)), lr=lr, ep=ep, lam=lam, nlr=nlr)
    return table


def report(table):
    print("\n" + "=" * 104, flush=True)
    print("pt5 iter3 LEARNED plasticity — TASK-IL, SGD, VAL-TUNED (main net + neuromod net), "
          "3 seeds (42/43/44), TEST set", flush=True)
    print("=" * 104, flush=True)
    print(f"{'cell':16s} {'acc (mean±std)':>18s} {'forget':>8s} {'lr':>7s} {'ep':>4s} "
          f"{'lam':>7s} {'nlr':>7s}   config", flush=True)
    for cell in ("naive", "er", "ewc", "ewc_er",
                 "dead-neuron", "plast-neuron", "dead-synapse", "plast-synapse"):
        r = table[cell]
        lam = f"{r['lam']:g}" if r["lam"] is not None else "-"
        nlr = f"{r['nlr']:g}" if r["nlr"] is not None else ("0" if cell.startswith("dead") else "-")
        print(f"{cell:16s} {r['mean']:>11.4f}±{r['std']:.4f} {r['forget']:>8.4f} "
              f"{r['lr']:>7g} {r['ep']:>4d} {lam:>7s} {nlr:>7s}", flush=True)

    print("\n--- mechanism deltas (d-dead is THE number: RNG-matched inert gate, rule #10) ---",
          flush=True)
    naive = table["naive"]["mean"]
    for gran in GRANS:
        m, d = table[f"plast-{gran}"], table[f"dead-{gran}"]
        per_seed = [a - b for a, b in zip(m["accs"], d["accs"])]
        signs = sum(1 for v in per_seed if v > 0)
        print(f"  plast-{gran:8s} d-dead {m['mean'] - d['mean']:+.4f}  "
              f"(per seed {', '.join(f'{v:+.4f}' for v in per_seed)}; {signs}/3 positive)   "
              f"d-naive {m['mean'] - naive:+.4f}", flush=True)
    for gran in GRANS:
        d = table[f"dead-{gran}"]
        print(f"  dead-{gran:9s} d-naive {d['mean'] - naive:+.4f}  "
              f"(RNG shift of constructing the modulator; NOT a mechanism effect)", flush=True)
    print("\nBars: the mechanism must beat its own dead-gate control by more than the 3-seed spread; "
          "and the whole arm is only interesting if it approaches the best baseline (ER / EWC+ER).",
          flush=True)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "tune-main", "tune-lam", "tune-neuro", "test", "report"])
    ap.add_argument("--resume", action="store_true", help="skip cells already in the ledger")
    args = ap.parse_args()
    print(f"pt5 task-IL plasticity study | opt={OPT} metric={METRIC} buffer={BUFFER} "
          f"val_seq={VAL_SEQ}\n", flush=True)

    ledger = load_ledger() if args.resume else {}

    tuned_main = tune_main(ledger)                     # pure lookup once ledgered
    if args.part == "tune-main":
        return
    tuned_lam = tune_lambda(ledger, tuned_main)
    if args.part == "tune-lam":
        return
    tuned_nlr = tune_neuro(ledger, tuned_main)
    if args.part == "tune-neuro":
        return
    table = test(ledger, tuned_main, tuned_lam, tuned_nlr)
    report(table)


if __name__ == "__main__":
    main()
