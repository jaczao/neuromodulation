"""pt7 TUNED gain-synapse — 3-seed run of 9 more drivers (user-requested; same setup as seeds run 1).

Same operating point as pt7_tuned_syn_seeds.py: Adam, TUNED lr=3e-4/ep5, gain-SYNAPSE, er-own, std1,
seeds {42,43,44}, class-IL Split MNIST, test set. Drivers:
  head-based (p7.cell_spec + Signals): ACh, NE_emb, NE_rise, 5ht-const, free, DA_fast, ACh_ema
  headless  (NEDriver):                emb_all, vec_h1proj
ER 3-seed reference (0.9029±0.0042) is reused from pt7_tuned_syn_seeds_results.tsv (not re-run).

Ledger results/pt7_tuned_syn_seeds2_results.tsv (own; `--resume` skips done).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_tuned_syn as T                                         # noqa: E402

LR, EP = 3e-4, 5
SEEDS = [42, 43, 44]
DRIVERS = ["ACh", "NE_emb", "NE_rise", "5ht-const", "free", "DA_fast", "ACh_ema", "emb_all", "vec_h1proj"]
HEADLESS = {"emb_all", "vec_h1proj", "vecproj", "vec_h1", "vec_x"}
TSV = Path(__file__).resolve().parent / "pt7_tuned_syn_seeds2_results.tsv"
ER_TSV = Path(__file__).resolve().parent / "pt7_tuned_syn_seeds_results.tsv"   # ER reference source


def load_done(tsv):
    if not tsv.exists():
        return {}
    return {ln.split("\t")[0]: float(ln.split("\t")[1])
            for ln in tsv.read_text().splitlines() if ln.strip()}


def record(tag, acc):
    with open(TSV, "a") as fh:
        fh.write(f"{tag}\t{acc:.4f}\n")


def run_one(name, seed, loaders):
    if name in HEADLESS:
        return T.run_headless(name, "adam", True, LR, EP, loaders, seed=seed)["pred"]
    return T.run_head(name, "adam", True, LR, EP, loaders, seed=seed)["pred"]


def er_reference():
    d = load_done(ER_TSV)
    accs = [v for k, v in d.items() if k.startswith("er|")]
    return (float(np.mean(accs)), float(np.std(accs))) if len(accs) == 3 else (None, None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={T.DEV}  (pt7 tuned gain-syn Adam 3-seed, batch 2; lr={LR:g} ep={EP})\n", flush=True)
    done = load_done(TSV) if args.resume else {}
    loaders = T.build_loaders(T.SEQ, 0.0)                         # TEST loaders
    accs = {n: [] for n in DRIVERS}
    for name in DRIVERS:
        for seed in SEEDS:
            tag = f"{name}|synapse|er-own|adam|std1|lr{LR:g}|ep{EP}|seed{seed}"
            if tag in done:
                accs[name].append(done[tag]); continue
            acc = run_one(name, seed, loaders)
            print(f"  {name:11s} seed{seed} | acc={acc:.4f}", flush=True)
            record(tag, acc); accs[name].append(acc)

    er_m, er_s = er_reference()
    print(f"\n=== 3-seed summary (Adam, tuned lr=3e-4 ep=5, gain-syn er-own std1) ===", flush=True)
    print(f"  {'er (ref)':12s} {er_m:.4f} ± {er_s:.4f}" if er_m is not None else "  er ref missing", flush=True)
    for name in DRIVERS:
        a = np.array(accs[name])
        d = f"  Δ vs ER {a.mean()-er_m:+.4f}" if er_m is not None else ""
        print(f"  {name:12s} {a.mean():.4f} ± {a.std():.4f}  "
              f"(seeds {', '.join(f'{x:.4f}' for x in a)}){d}", flush=True)
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
