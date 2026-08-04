"""Standalone buf-cur plasticity: joint (epochs x neuro_lr) tuning at three gate INITS.

Follow-up to `plast_taskil.py`, which found the standalone buf-cur arm null against its dead gate and
— via `gate_stats.py` — explained why: the per-NEURON gate engages hard (alpha 0.5 -> 0.93) but every
task learns the SAME deviation, i.e. it spends itself as a global LR knob UNDOING its own 0.5 init;
the per-SYNAPSE gate never leaves parity at all (|alpha-0.5| = 0.0007, tuned neuro_lr at the grid
floor). Both readings implicate the INIT, and neither was tested at a tuned epoch budget.

WHAT THIS ADDS over `results/pt5_plast_init.py` (which swept init {0.1..0.99} and rejected at every
value): that study ran at a FIXED inherited lr=1e-3/ep=5 with the neuromod net pinned to the main lr,
and its conclusion — "the init never rescues plasticity" — was drawn at an operating point nobody had
tuned. pt7's plasticity arc is the cautionary case: an untuned SGD baseline made a global LR boost
look like a +0.11 win that dissolved at a tuned lr. Here the main lr is already tuned (3e-3, from
plast_taskil), and this sweeps the two axes that were never tuned for the mechanism:

  init  {0.5, 0.9, 0.99}   the starting gate alpha = sigmoid(logit(init)). 0.5 is plast_taskil's
                           value (a half-LR handicap the gate must undo); 0.9/0.99 start near-open,
                           so if the neuron gate's whole job was opening itself up, these hand it
                           that for free and the mechanism has to show something else or tie.
  ep    {5, 10, 20}        epochs_per_task. Held at the backbone's 5 in plast_taskil so the mechanism
                           and its control were matched; a gated run may want a different budget,
                           which was listed there as an untuned axis.
  nlr   {1e-4 .. 1e-2}     the neuromod net's own lr (5 points, as before).

x {neuron, synapse} = 90 val cells. Main lr FIXED at naive's tuned 3e-3 (this is the standalone arm,
so its backbone is naive); tuning it jointly too would be a fourth axis on an effect measured at
-0.0006.

THE DEAD CONTROL IS PER-INIT, and that is the point. `dead` = the same config with neuromod_lr = 0, so
P stays at zero and alpha is pinned at exactly `init` for every unit and task — a uniform init x LR
rescale. Comparing a live gate at init 0.9 against a dead gate at init 0.9 separates "the gate learned
something" from "this init is simply a better effective learning rate", which a single dead control at
one init cannot do. Each is also RNG-matched (modulator built, reservoir filled and sampled, lookahead
run), per rule #10.

GATE STUDY on every selected cell: per-layer alpha statistics including the requested alpha>0.95 and
alpha>0.99 fractions, plus cos(dev) (does each task learn a DIFFERENT gate, or the same one?) and the
raw per-task alpha saved to .npz so further threshold questions need no re-run.

Protocol: tune on the VAL sequence make_sequence(7)/val_frac=0.1 seed 42, never the test set (rule
#1); identical budget per (init, gran) (rule #3); report 3 seeds on test (rule #5). Noise-floor
tie-break on selection, preferring the smallest epochs among cells within 0.007 of the best.

Ledger pt5_taskil/plast_init_results.tsv; `--resume` skips done rows; `--part` chunks the run.

Run: uv run python pt5_taskil/plast_init.py --part all --resume  (redirect to plast_init.log)
"""
import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_stats as G                                   # noqa: E402
import plast_taskil as S                                 # noqa: E402
import prototype.train as train_mod                      # noqa: E402
from prototype.configs import CLConfig                   # noqa: E402
from prototype.train import cl_train                     # noqa: E402

TSV = Path(__file__).resolve().parent / "plast_init_results.tsv"
COLS = ["stage", "gran", "init", "ep", "nlr", "seed", "acc", "forget"]

INITS = (0.5, 0.9, 0.99)
EPOCHS = (5, 10, 20)
NEURO_LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
MAIN_LR = 3e-3                                           # naive's tuned point (plast_taskil)
SEEDS = (42, 43, 44)
TUNE_SEED = 42
NOISE_FLOOR = 0.007
DEAD_NLR = 0.0                                           # nlr = 0 => P frozen at 0 => alpha == init


# --------------------------------------------------------------------------- ledger
def load_ledger():
    if not TSV.exists():
        return {}
    rows = {}
    for line in TSV.read_text().splitlines()[1:]:
        if line.strip():
            f = line.split("\t")
            rows[tuple(f[:6])] = (float(f[6]), float(f[7]))
    return rows


def append(key, acc, forget):
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with TSV.open("a") as fh:
        fh.write("\t".join(list(key) + [f"{acc:.6f}", f"{forget:.6f}"]) + "\n")


def key_of(stage, gran, init, ep, nlr, seed):
    return (stage, gran, f"{init:g}", str(ep), f"{nlr:g}", str(seed))


# --------------------------------------------------------------------------- one cell
def build(gran, init, ep, nlr, seed):
    """Standalone buf-cur plasticity: naive backbone + the modulator-only replay meta-loss."""
    return CLConfig(
        seed=seed, lr=MAIN_LR, epochs_per_task=ep, optimizer="sgd",
        output_masking="taskil", er_buffer_size=1000,
        use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
        neuromod_target="plasticity", neuromod_projection="learned",
        neuromod_meta_replay=True, neuromod_er_task_id=False,
        neuromod_lr=nlr, neuromod_plasticity_init=init,
        **S.MECH_KW[gran],
    )


def run_cell(stage, gran, init, ep, nlr, seed, ledger, split):
    key = key_of(stage, gran, init, ep, nlr, seed)
    if key in ledger:
        acc, forget = ledger[key]
        print(f"[skip] {'|'.join(key)} acc={acc:.4f}", flush=True)
        return acc, forget
    config = build(gran, init, ep, nlr, seed)
    config.val_frac = 0.1
    buf = io.StringIO()
    with redirect_stdout(buf):
        acc, forget = cl_train(config, "naive", no_wandb=True,
                               sequence=S.VAL_SEQ if split == "val" else None, eval_split=split)
    append(key, acc, forget)
    ledger[key] = (acc, forget)
    print(f"[run ] {'|'.join(key)} acc={acc:.4f} forget={forget:.4f}", flush=True)
    return acc, forget


# --------------------------------------------------------------------------- stages
def tune(ledger):
    """(ep, nlr) per (gran, init) on val. Tie-break: within NOISE_FLOOR of best, smallest ep."""
    best = {}
    for gran in S.GRANS:
        for init in INITS:
            results = []
            for ep in EPOCHS:
                for nlr in NEURO_LRS:
                    acc, _ = run_cell("tune", gran, init, ep, nlr, TUNE_SEED, ledger, "val")
                    results.append((acc, ep, nlr))
            top = max(results)[0]
            tied = [r for r in results if r[0] >= top - NOISE_FLOOR]
            acc, ep, nlr = min(tied, key=lambda r: (r[1], -r[0]))
            best[(gran, init)] = (ep, nlr, acc)
            note = "" if acc == top else f"  [tie-break: {len(tied)}/{len(results)} within {NOISE_FLOOR}]"
            print(f"\n>>> TUNED [{gran} init={init:g}] ep={ep} nlr={nlr:g} (val {acc:.4f}){note}",
                  flush=True)
            for label, v, grid in (("ep", ep, EPOCHS), ("nlr", nlr, NEURO_LRS)):
                if v in (grid[0], grid[-1]):
                    print(f"    !! {label}={v:g} at a GRID EDGE", flush=True)
            print(flush=True)
    return best


def test(ledger, best):
    """3 seeds on test: the tuned live cell and its PER-INIT dead control, at the same (ep)."""
    table = {}
    for gran in S.GRANS:
        for init in INITS:
            ep, nlr, _ = best[(gran, init)]
            for kind, use_nlr in (("live", nlr), ("dead", DEAD_NLR)):
                accs, forgets = [], []
                for s in SEEDS:
                    a, f = run_cell("test", gran, init, ep, use_nlr, s, ledger, "test")
                    accs.append(a); forgets.append(f)
                table[(gran, init, kind)] = dict(
                    accs=accs, mean=float(np.mean(accs)), std=float(np.std(accs)),
                    forget=float(np.mean(forgets)), ep=ep, nlr=use_nlr)
    return table


def gate(best):
    """Per-layer alpha stats at each selected cell, seed 42, incl. >0.95 / >0.99. Saves .npz."""
    for gran in S.GRANS:
        for init in INITS:
            ep, nlr, _ = best[(gran, init)]
            G.COLLECTED.clear()
            os.environ["PT5_DUMP_OVERLAP"] = "1"
            original = train_mod._pt5_dump_overlap
            train_mod._pt5_dump_overlap = G.collector
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    acc, forget = cl_train(build(gran, init, ep, nlr, TUNE_SEED), "naive",
                                           no_wandb=True, sequence=None, eval_split="test")
            finally:
                train_mod._pt5_dump_overlap = original
                os.environ.pop("PT5_DUMP_OVERLAP", None)
            data = dict(G.COLLECTED)
            parity = data["parity"]
            print(f"\n{'=' * 104}")
            print(f"GATE  {gran} init={init:g}  ep={ep} nlr={nlr:g}  |  test acc {acc:.4f} "
                  f"forget {forget:.4f}  |  parity alpha {parity:.4f}")
            print("=" * 104)
            print(f"{'layer':7s} {'n':>9s} {'|a-par|':>9s} {'a mean':>8s} {'a sd':>8s} "
                  f"{'a min':>8s} {'a max':>8s} {'>0.75':>7s} {'>0.95':>7s} {'>0.99':>7s} "
                  f"{'cos(dev)':>9s} {'frac':>6s}")
            for name, A in data["layers"].items():
                dev = A - parity
                print(f"{name:7s} {A[0].size:>9d} {np.abs(dev).mean():>9.5f} {A.mean():>8.4f} "
                      f"{A.std():>8.4f} {A.min():>8.4f} {A.max():>8.4f} "
                      f"{(A > 0.75).mean():>7.3f} {(A > 0.95).mean():>7.3f} "
                      f"{(A > 0.99).mean():>7.3f} "
                      f"{G._offdiag_cos(dev):>9.4f} {G._engaged_frac(dev):>6.3f}")
            np.savez(Path(__file__).resolve().parent / f"gate_alpha_{gran}_init{init:g}.npz",
                     parity=parity, **data["layers"])


def report(table, best):
    print("\n" + "=" * 104)
    print("STANDALONE BUF-CUR — joint (epochs x neuro_lr) tuning at three gate inits, "
          "3 seeds, TEST set")
    print("=" * 104)
    print(f"{'gran':8s} {'init':>5s} {'ep':>4s} {'nlr':>8s} {'live acc':>17s} "
          f"{'dead acc (same init)':>21s} {'d-dead':>8s} {'per seed':>28s} {'pos':>4s}")
    for gran in S.GRANS:
        for init in INITS:
            L, D = table[(gran, init, "live")], table[(gran, init, "dead")]
            ps = [a - b for a, b in zip(L["accs"], D["accs"])]
            print(f"{gran:8s} {init:>5g} {L['ep']:>4d} {L['nlr']:>8g} "
                  f"{L['mean']:>10.4f}±{L['std']:.4f} {D['mean']:>14.4f}±{D['std']:.4f} "
                  f"{L['mean'] - D['mean']:>+8.4f} "
                  f"{', '.join(f'{v:+.4f}' for v in ps):>28s} "
                  f"{sum(1 for v in ps if v > 0)}/3")
    print("\n  reference (plast_taskil, ep=5, main lr 3e-3): naive 0.9784  er 0.9946")
    print("  d-dead isolates the LEARNED gate; the dead column isolates what the INIT alone buys "
          "(a uniform alpha x LR rescale).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "tune", "test", "gate", "report"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"standalone buf-cur | init sweep {INITS} x ep {EPOCHS} x nlr {NEURO_LRS} "
          f"| main lr {MAIN_LR:g} sgd taskil\n", flush=True)

    ledger = load_ledger() if args.resume else {}
    best = tune(ledger)
    if args.part == "tune":
        return
    if args.part == "gate":
        gate(best)
        return
    table = test(ledger, best)
    if args.part in ("all", "gate"):
        gate(best)
    report(table, best)


if __name__ == "__main__":
    main()
