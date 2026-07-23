"""pt7 NEUROMOD-NET tuning under SGD — class-IL, ER only, gain-SYNAPSE, all4 (+ free control).

SGD regime: MAIN net SGD (class-IL er-sgd reused from configs.TUNED_MAIN = lr0.03/ep5, tuned by
pt7_tuned_syn) AND the NEUROMOD net (gate P + heads) on SGD via a decoupled optimizer. Tune the
neuro_lr over 3 SGD-scale values on the val sequence; report at 1 seed on test. Reuses
pt7_tuned_neuro.run_er / run_erown_decoupled (the latter's neuro_opt_kind='sgd').

Ledger results/pt7_tuned_neuro_sgd_results.tsv (own; `--resume`). 1 seed for report (user-requested).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt7_tuned_neuro as tn                                       # noqa: E402  (run_er, run_erown_decoupled, loaders)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from configs import TUNED_MAIN                                     # noqa: E402

METRIC, BASE, MAIN_OPT, NEURO_OPT, GRAN = "classil", "er", "sgd", "sgd", "synapse"
MAIN = TUNED_MAIN[(METRIC, BASE, MAIN_OPT)]                        # {lr:0.03, epochs_per_task:5}
MAIN_LR, EP = MAIN["lr"], MAIN["epochs_per_task"]
NEURO_LRS = [3e-3, 1e-2, 3e-2]                                     # 3 SGD-scale neuro lrs (user-requested)
SEED = 42
TSV = Path(__file__).resolve().parent / "pt7_tuned_neuro_sgd_results.tsv"


def load_done():
    if not TSV.exists():
        return {}
    return {ln.split("\t")[0]: float(ln.split("\t")[1]) for ln in TSV.read_text().splitlines() if ln.strip()}


def record(tag, acc):
    with open(TSV, "a") as fh:
        fh.write(f"{tag}\t{acc:.4f}\n")


def erown(name, std, nlr, loaders, seed):
    return tn.run_erown_decoupled(name, GRAN, std, MAIN_OPT, MAIN_LR, nlr, EP, loaders, seed,
                                  neuro_opt_kind=NEURO_OPT)["pred"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    done = load_done() if args.resume else {}
    print(f"device={tn.DEV}  (pt7 SGD neuromod tuning; class-IL er-own; gain-synapse; "
          f"main sgd lr{MAIN_LR:g}/ep{EP}; neuro sgd; 1-seed report)\n", flush=True)
    val_loaders = tn.build_loaders(tn.VAL_SEQ, tn.VAL_FRAC)
    test_loaders = tn.build_loaders(tn.SEQ, 0.0)

    # ---- neuro tune (val, seed 42): all4 er-own, 3 SGD neuro_lrs ----
    print("[tune] all4 er-own, sweep neuro_lr", NEURO_LRS, flush=True)
    for nlr in NEURO_LRS:
        tag = f"tune|{METRIC}|{BASE}-own|all4|synapse|sgd|nlr{nlr:g}|s{SEED}"
        if tag not in done:
            acc = erown("all4", True, nlr, val_loaders, SEED)
            print(f"  TUNE all4 nlr{nlr:<7g} | val={acc:.4f}", flush=True); record(tag, acc); done[tag] = acc
    best_nlr = max(NEURO_LRS, key=lambda l: done[f"tune|{METRIC}|{BASE}-own|all4|synapse|sgd|nlr{l:g}|s{SEED}"])
    print(f"  -> best neuro_lr = {best_nlr:g} (val {done[f'tune|{METRIC}|{BASE}-own|all4|synapse|sgd|nlr{best_nlr:g}|s{SEED}']:.4f})\n", flush=True)

    # ---- report (test, 1 seed): er baseline, er+all4 (best nlr), er+free (nlr inert) ----
    results = {}
    er_tag = f"report|{METRIC}|{BASE}|-|sgd|s{SEED}"
    if er_tag not in done:
        acc = tn.run_er(MAIN_OPT, MAIN_LR, EP, test_loaders, SEED)["pred"]
        print(f"  REPORT er | test={acc:.4f}", flush=True); record(er_tag, acc); done[er_tag] = acc
    results["er"] = done[er_tag]
    for name, std, nlr in [("all4", True, best_nlr), ("free", True, best_nlr)]:
        tag = f"report|{METRIC}|{BASE}-own|{name}|synapse|sgd|nlr{nlr:g}|s{SEED}"
        if tag not in done:
            acc = erown(name, std, nlr, test_loaders, SEED)
            print(f"  REPORT er+{name:4s} nlr{nlr:<7g} | test={acc:.4f}", flush=True); record(tag, acc); done[tag] = acc
        results[f"er+{name}"] = done[tag]

    print("\n========  CLASS-IL SGD (er-own, gain-synapse, 1 seed)  ========", flush=True)
    for k in ("er", "er+free", "er+all4"):
        d = f"  (dfree {results[k] - results['er+free']:+.4f})" if k == "er+all4" else ""
        print(f"  {k:9s} {results[k]:.4f}{d}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
