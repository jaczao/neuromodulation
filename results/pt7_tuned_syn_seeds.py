"""pt7 TUNED gain-synapse — 3-seed confirm of the Adam er-own cells (user-requested).

The 1-seed tuned report had NE +0.011 and all4 +0.010 over tuned ER (Adam) — inside 1-seed noise
(~0.02). This runs seeds {42,43,44} at the TUNED Adam point (lr=3e-4, ep=5) for ER baseline, NE, and
all4 (gain-synapse, er-own, standardised), reporting mean±std on the TEST set. Reuses pt7_tuned_syn.

Ledger results/pt7_tuned_syn_seeds_results.tsv (own; `--resume` skips done). class-IL Split MNIST.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_tuned_syn as T                                         # noqa: E402

LR, EP = 3e-4, 5                                                  # tuned Adam operating point (val 0.9079)
SEEDS = [42, 43, 44]
TSV = Path(__file__).resolve().parent / "pt7_tuned_syn_seeds_results.tsv"


def load_done():
    if not TSV.exists():
        return {}
    return {ln.split("\t")[0]: float(ln.split("\t")[1])
            for ln in TSV.read_text().splitlines() if ln.strip()}


def record(tag, acc):
    with open(TSV, "a") as fh:
        fh.write(f"{tag}\t{acc:.4f}\n")


def _run_mech_seed(name, seed, loaders):
    # T.run_head / run_headless take a seed kwarg; route by mechanism type
    if name in ("vecproj", "vec_h1proj"):
        return T.run_headless(name, "adam", True, LR, EP, loaders, seed=seed)["pred"]
    return T.run_head(name, "adam", True, LR, EP, loaders, seed=seed)["pred"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"device={T.DEV}  (pt7 tuned gain-syn Adam 3-seed; lr={LR:g} ep={EP})\n", flush=True)
    done = load_done() if args.resume else {}
    loaders = T.build_loaders(T.SEQ, 0.0)                         # TEST loaders (report)
    accs = {n: [] for n in ("er", "NE", "all4")}
    for name in ("er", "NE", "all4"):
        for seed in SEEDS:
            tag = f"{name}|synapse|er-own|adam|std1|lr{LR:g}|ep{EP}|seed{seed}"
            if tag in done:
                accs[name].append(done[tag]); continue
            acc = _run_mech_seed(name, seed, loaders) if name != "er" \
                else T.run_baseline("er", "adam", LR, EP, loaders, seed=seed)["pred"]
            print(f"  {name:5s} seed{seed} | acc={acc:.4f}", flush=True)
            record(tag, acc); accs[name].append(acc)
    print("\n=== 3-seed summary (Adam, tuned lr=3e-4 ep=5, gain-syn er-own std1) ===", flush=True)
    er_m = float(np.mean(accs["er"]))
    for name in ("er", "NE", "all4"):
        a = np.array(accs[name]); d = f"  Δ vs ER {a.mean()-er_m:+.4f}" if name != "er" else ""
        print(f"  {name:5s}  {a.mean():.4f} ± {a.std():.4f}  (seeds {', '.join(f'{x:.4f}' for x in a)}){d}", flush=True)
    print("ALL SELECTED CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
