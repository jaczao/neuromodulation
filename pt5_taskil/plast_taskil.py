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

BASELINES
  `naive`, `er`, `ewc`, `ewc_er`, all at `output_masking='taskil'` (masked train loss + 2-way masked
  eval), plus `naive_unmasked` — naive trained on the FULL 10-way CE with the task-IL eval kept. That
  ablation separates the two things this repo's `taskil` setting does at once: pt3 Iteration 5 showed
  the masked TRAIN loss (lever B) is worth a large chunk on its own, so a task-IL number quoted
  without it is a different baseline. `output_masking` cannot express it, hence the `unmasked_loss()`
  shim below. It gets its own lr x ep sweep (identical budget, rule #3) — an unmasked net sees a
  different loss surface and has no reason to share naive's operating point.

PROTOCOL (rules #1, #3, #5)
  TUNE  (seed 42, VAL sequence make_sequence(7), val_frac=0.1, eval on the held-out val split — the
         test set is never touched during selection):
    1. main lr x ep for `naive`, `naive_unmasked` and `er`, grid lr {1e-3,3e-3,1e-2,3e-2,1e-1} x
       ep {5,10,20} (5x3 = 15 cells each, identical budget). Selection uses the NOISE-FLOOR TIE-BREAK:
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
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prototype.train as train_mod               # noqa: E402  (for the unmasked-loss shim)
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
# Per-cell UPWARD extension of the lr grid, added after the first pass because ER selected the grid
# TOP (lr=1e-1) with val acc MONOTONE INCREASING in lr at every epoch budget (ep=5: 0.9687 0.9795
# 0.9847 0.9875 0.9909; ep=10 and ep=20 the same shape) — never turning over, which is the truncated-
# grid signature, not an optimum. Only ER is extended: `naive` and `naive_unmasked` both turn over
# INSIDE the grid and then collapse at lr=1e-1 (0.9332 and 0.7076), so extending them upward would
# add only strictly-worse cells and cannot move their selection. This is the same asymmetry
# `results/pt3_retry.py` used when it extended its naive arm DOWNWARD and left ER's grid alone: the
# arms sit at genuinely different operating points (replay tolerates and wants large steps; an
# unreplayed net forgets harder the larger the step), so one grid cannot serve both.
MAIN_GRID_EXT = {"er": [3e-1, 1.0]}


def lr_grid(cell):
    return MAIN_GRID["lr"] + MAIN_GRID_EXT.get(cell, [])
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

# The five mechanism ARMS. Each is (backbone method, meta_replay, er_task_id, which base's main point).
#   nobuf  — iter-3 DEFAULT: naive main, learned P trained by a lookahead meta-loss on the CURRENT
#            batch only. No buffer anywhere, so no retention signal reaches P.
#   bufcur — naive main + the modulator-only replay buffer (`--neuromod-meta-replay`): P gets a
#            retention signal, the backbone stays naive. Gated by the CURRENT task's row.
#   bufown — REQUESTED, but NOT EXPRESSIBLE for plasticity, and this is a property of the frozen
#            loop rather than a choice here. `neuromod_er_task_id` reaches two places in cl_train:
#            `er_task_id_on` (replay-gated, so it cannot fire on a naive run) and `meta_task_id_on`,
#            which is read ONLY inside `if gain_meta_replay_on` — the GAIN meta-loop. Plasticity's
#            standalone meta-loss (train.py ~1332) builds its batch from the buffer and gates it with
#            the CURRENT task's `factors`, consulting neither flag. So `bufown` is byte-identical to
#            `bufcur` by construction. It is kept as a real ledger row to DEMONSTRATE that identity
#            rather than assert it (the report checks it), and it reuses bufcur's tuned neuro_lr
#            instead of spending a second identical sweep on it.
#            (CLAUDE.md already notes "'cur' is the accurate label" for this arm; this pins down why.)
#   ercur  — ER backbone, whole mixed current+replay batch gated by the CURRENT task P[t]: the
#            legacy wrong-task ablation.
#   erown  — ER backbone, each sample gated by ITS OWN task P[j]: cl_train splits the batch by task,
#            backwards each subset weighted n_j/N, gates by P[j], accumulates, one step. Up to 5
#            backward passes per batch, so this arm is several times slower than the rest.
ARMS = {
    "nobuf":  dict(method="naive", meta_replay=False, er_task_id=False, base="naive"),
    "bufcur": dict(method="naive", meta_replay=True,  er_task_id=False, base="naive"),
    "bufown": dict(method="naive", meta_replay=True,  er_task_id=True,  base="naive"),
    "ercur":  dict(method="er",    meta_replay=False, er_task_id=False, base="er"),
    "erown":  dict(method="er",    meta_replay=False, er_task_id=True,  base="er"),
}
ARM_ORDER = ("nobuf", "bufcur", "bufown", "ercur", "erown")
# bufcur was the study's original (and only) arm, so its cells are named `plast-<gran>` / `dead-<gran>`
# in the ledger. Keep that spelling so those rows stay valid instead of re-running identical configs.
TUNE_ARMS = ("nobuf", "ercur", "erown")          # bufcur already tuned; bufown == bufcur


def cell_name(kind, arm, gran):
    return f"{kind}-{gran}" if arm == "bufcur" else f"{kind}-{arm}-{gran}"


def parse_cell(cell):
    """'plast-neuron' -> (plast, bufcur, neuron); 'dead-erown-synapse' -> (dead, erown, synapse)."""
    parts = cell.split("-")
    return (parts[0], "bufcur", parts[1]) if len(parts) == 2 else tuple(parts)


# ------------------------------------------------------- unmasked-loss task-IL (`naive_unmasked`)
# This repo's `taskil` convention masks TWO things: the per-sample training loss (each sample's CE
# restricted to its own task's 2 classes) AND the eval logits (2-way argmax). `output_masking` has no
# setting for "unmasked training loss, masked eval", so the ablation "task-IL WITHOUT the masked
# loss" — train on the full 10-way CE, still evaluate task-IL — cannot be expressed by config alone.
# It needs no change to the frozen loop: `MaskedCE` is documented to be plain CE while `pairs is
# None`, and every branch of `cl_train` enables masking by ASSIGNING `criterion.pairs`. So a subclass
# whose `pairs` assignment is a no-op keeps the criterion at plain CE while `output_masking='taskil'`
# still drives the masked eval (`allowed=sequence[i]`). Patched into `prototype.train` for the
# duration of one call, never edited in place (rule #9), and it consumes no RNG (the real MaskedCE
# also only builds an nn.CrossEntropyLoss), so this arm stays RNG-comparable to `naive`.
class _UnmaskedCE(train_mod.MaskedCE):
    @property
    def pairs(self):
        return None

    @pairs.setter
    def pairs(self, value):                      # swallow cl_train's `criterion.pairs = ...`
        pass


@contextmanager
def unmasked_loss():
    original = train_mod.MaskedCE
    train_mod.MaskedCE = _UnmaskedCE             # isinstance checks still pass (it is a subclass)
    try:
        yield
    finally:
        train_mod.MaskedCE = original


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
    """(CLConfig, method_name) for a cell name. Cells: naive | naive_unmasked | er | ewc | ewc_er |
    plast-<gran> (mechanism, buf-cur) | dead-<gran> (RNG-matched inert gate)."""
    kw = dict(seed=seed, lr=lr, epochs_per_task=ep, optimizer=OPT,
              output_masking=METRIC, er_buffer_size=BUFFER)
    if cell == "naive_unmasked":                 # same config; run_cell applies the unmasked shim
        return CLConfig(**kw), "naive"
    if cell in ("naive", "er"):
        return CLConfig(**kw), cell
    if cell in ("ewc", "ewc_er"):
        return CLConfig(ewc_lambda=lam, **kw), cell
    kind, arm, gran = parse_cell(cell)
    spec = ARMS[arm]
    kw.update(dict(use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
                   neuromod_target="plasticity", neuromod_projection="learned",
                   neuromod_meta_replay=spec["meta_replay"],
                   neuromod_er_task_id=spec["er_task_id"], **MECH_KW[gran]))
    if kind == "dead":
        # Inert gate, everything else identical: meta-optimizer step size 0 (P frozen at its zero
        # init) and the gate pinned at ~1.0, so the main net's update is the naive one. Each arm needs
        # its OWN dead control — `erown`'s split-by-task path runs a backward per task and accumulates,
        # which is a different floating-point summation order from `ercur`'s single backward even when
        # the gate is uniform, so one control cannot serve both.
        kw.update(neuromod_lr=0.0, neuromod_plasticity_init=0.9999)
    else:
        kw.update(neuromod_lr=nlr, neuromod_plasticity_init=0.5)
    return CLConfig(**kw), spec["method"]


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
        if cell == "naive_unmasked":
            with unmasked_loss():
                acc, forget = cl_train(config, method, no_wandb=True, sequence=sequence,
                                       eval_split=split)
        else:
            acc, forget = cl_train(config, method, no_wandb=True, sequence=sequence,
                                   eval_split=split)
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
    for cell in ("naive", "naive_unmasked", "er"):
        results = []
        for lr in lr_grid(cell):
            for ep in MAIN_GRID["ep"]:
                acc, _ = run_cell("tune_main", cell, lr, ep, None, None, TUNE_SEED, ledger, "val")
                results.append((acc, lr, ep))
        top = max(results)[0]
        tied = [r for r in results if r[0] >= top - NOISE_FLOOR]
        acc, lr, ep = min(tied, key=lambda r: (r[2], -r[0]))
        best[cell] = dict(lr=lr, epochs_per_task=ep, val=acc)
        note = "" if acc == top else f"  [tie-break: {len(tied)} cells within {NOISE_FLOOR} of {top:.4f}]"
        print(f"\n>>> TUNED main [{cell}]: lr={lr:g} ep={ep} (val {acc:.4f}){note}", flush=True)
        _edge_warn("lr", lr, lr_grid(cell))
        _edge_warn("ep", ep, MAIN_GRID["ep"])
        print(flush=True)
    return best


def main_point(cell, tuned_main):
    """(lr, ep) for a cell. EWC inherits naive's point, EWC+ER inherits er's, the mechanism and its
    dead control inherit naive's (their backbone is a naive main net)."""
    if "-" in cell:                                   # a mechanism or dead-control cell
        base = ARMS[parse_cell(cell)[1]]["base"]      # naive-backbone arms vs ER-backbone arms
    else:
        base = {"er": "er", "ewc_er": "er", "naive_unmasked": "naive_unmasked"}.get(cell, "naive")
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
    """neuro_lr per (arm, granularity), main net frozen at that arm's backbone point. `bufown` is
    byte-identical to `bufcur` (see ARMS), so it reuses bufcur's selection rather than re-sweeping."""
    best = {}
    for arm in ("bufcur",) + TUNE_ARMS:
        cell_lr, cell_ep = main_point(cell_name("plast", arm, "neuron"), tuned_main)
        for gran in GRANS:
            results = []
            for nlr in NEURO_LRS:
                acc, _ = run_cell("tune_neuro", cell_name("plast", arm, gran), cell_lr, cell_ep,
                                  None, nlr, TUNE_SEED, ledger, "val")
                results.append((acc, nlr))
            acc, nlr = max(results)
            best[(arm, gran)] = nlr
            print(f"\n>>> TUNED neuro_lr [plast-{arm}-{gran}] = {nlr:g} (val {acc:.4f}) "
                  f"at main lr={cell_lr:g} ep={cell_ep}", flush=True)
            _edge_warn("neuro_lr", nlr, NEURO_LRS)
            print(flush=True)
    for gran in GRANS:
        best[("bufown", gran)] = best[("bufcur", gran)]
    return best


# --------------------------------------------------------------------------- stage 4: test report
def test(ledger, tuned_main, tuned_lam, tuned_nlr):
    table = {}
    cells = [("naive", None, None), ("naive_unmasked", None, None), ("er", None, None),
             ("ewc", tuned_lam["ewc"], None), ("ewc_er", tuned_lam["ewc_er"], None)]
    for arm in ARM_ORDER:
        for gran in GRANS:
            # bufown shares bufcur's dead control (identical config), so it is not run twice.
            if arm != "bufown":
                cells.append((cell_name("dead", arm, gran), None, None))
            cells.append((cell_name("plast", arm, gran), None, tuned_nlr[(arm, gran)]))
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
    print(f"{'cell':22s} {'acc (mean±std)':>18s} {'forget':>8s} {'lr':>7s} {'ep':>4s} "
          f"{'lam':>7s} {'nlr':>7s}", flush=True)
    order = ["naive", "naive_unmasked", "er", "ewc", "ewc_er"]
    for arm in ARM_ORDER:
        for gran in GRANS:
            if arm != "bufown":
                order.append(cell_name("dead", arm, gran))
            order.append(cell_name("plast", arm, gran))
    for cell in order:
        r = table[cell]
        lam = f"{r['lam']:g}" if r["lam"] is not None else "-"
        nlr = f"{r['nlr']:g}" if r["nlr"] is not None else ("0" if cell.startswith("dead") else "-")
        print(f"{cell:22s} {r['mean']:>11.4f}±{r['std']:.4f} {r['forget']:>8.4f} "
              f"{r['lr']:>7g} {r['ep']:>4d} {lam:>7s} {nlr:>7s}", flush=True)

    print("\n--- mechanism deltas (d-dead is THE number: RNG-matched inert gate, rule #10) ---",
          flush=True)
    print(f"{'arm':10s} {'gran':8s} {'d-dead':>9s}  {'per seed':>26s}  {'pos':>4s} "
          f"{'d-base':>9s}  base", flush=True)
    for arm in ARM_ORDER:
        # bufown reuses bufcur's dead control; every other arm has its own.
        dead_arm = "bufcur" if arm == "bufown" else arm
        base_cell = ARMS[arm]["base"]
        for gran in GRANS:
            m, d = table[cell_name("plast", arm, gran)], table[cell_name("dead", dead_arm, gran)]
            per_seed = [a - b for a, b in zip(m["accs"], d["accs"])]
            signs = sum(1 for v in per_seed if v > 0)
            print(f"{arm:10s} {gran:8s} {m['mean'] - d['mean']:>+9.4f}  "
                  f"{', '.join(f'{v:+.4f}' for v in per_seed):>26s}  {signs}/3  "
                  f"{m['mean'] - table[base_cell]['mean']:>+9.4f}  {base_cell}", flush=True)

    print("\n--- dead-gate controls vs their plain baseline (RNG shift only, NOT a mechanism) ---",
          flush=True)
    for arm in ARM_ORDER:
        if arm == "bufown":
            continue
        for gran in GRANS:
            d, b = table[cell_name("dead", arm, gran)], table[ARMS[arm]["base"]]
            print(f"  dead-{arm}-{gran:8s} d-{ARMS[arm]['base']:5s} {d['mean'] - b['mean']:+.4f}",
                  flush=True)

    # bufown is byte-identical to bufcur by construction (neuromod_er_task_id never reaches the
    # standalone plasticity meta-loop). Check it rather than assert it.
    print("\n--- bufown == bufcur check (er_task_id does not reach plasticity's meta-loop) ---",
          flush=True)
    # TOLERANCE = the LEDGER's precision, not machine epsilon. Accuracies are stored as "%.6f", so a
    # cell read back from the ledger is rounded while one just computed in this process is not; a
    # 1e-12 comparison between the two reports False on runs that are in fact identical. Match the
    # tolerance to how the numbers were produced (cf. CLAUDE.md's rule about not setting a bit-exact
    # tolerance without checking what produced the value).
    for gran in GRANS:
        a = table[cell_name("plast", "bufown", gran)]["accs"]
        b = table[cell_name("plast", "bufcur", gran)]["accs"]
        worst = max(abs(x - y) for x, y in zip(a, b))
        print(f"  {gran:8s} identical_to_ledger_precision={worst < 1e-6}  (max|Δ|={worst:.2e})  "
              f"bufown {['%.6f' % v for v in a]} vs bufcur {['%.6f' % v for v in b]}", flush=True)

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
