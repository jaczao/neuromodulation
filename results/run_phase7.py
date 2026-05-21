#!/usr/bin/env python
"""Phase 7 final evaluation: standard MNIST table and CL table.

Uses Phase 6 best hyperparameters from configs.py.
Runs each configuration over seeds [42, 43, 44], aggregates mean ± std,
and writes:
  results/standard_mnist_table.{csv,md}
  results/split_mnist_table.{csv,md}

Usage:
  uv run python results/run_phase7.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from prototype.configs import (
    BEST_CL_ER,
    BEST_CL_EWC,
    BEST_CL_JOINT,
    BEST_CL_NAIVE,
    BEST_CL_NEUROMOD,
    BEST_STANDARD_NEUROMOD,
    BEST_STANDARD_VANILLA,
)
from prototype.train import cl_train, train_standard

SEEDS = [42, 43, 44]
RESULTS_DIR = Path(__file__).parent


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Standard MNIST
# ---------------------------------------------------------------------------

def run_standard_table() -> list[dict]:
    print("\n" + "=" * 60)
    print("Standard MNIST evaluation (official test set, 3 seeds)")
    print("=" * 60)
    rows = []
    for label, base_config in [
        ("vanilla MLP", BEST_STANDARD_VANILLA),
        ("neuromod MLP", BEST_STANDARD_NEUROMOD),
    ]:
        accs = []
        for seed in SEEDS:
            config = base_config.__class__(**{**base_config.__dict__, "seed": seed})
            print(f"\n[{label}] seed={seed}  lr={config.lr}  epochs={config.epochs}")
            _, test_acc = train_standard(config, no_wandb=True)
            accs.append(test_acc)
        m, s = _mean_std(accs)
        print(f"=> {label}: {m:.4f} ± {s:.4f}")
        rows.append({"method": label, "test_acc_mean": m, "test_acc_std": s})
    return rows


def save_standard_table(rows: list[dict]) -> None:
    csv_path = RESULTS_DIR / "standard_mnist_table.csv"
    md_path = RESULTS_DIR / "standard_mnist_table.md"

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "test_acc_mean", "test_acc_std"])
        w.writeheader()
        w.writerows(rows)

    v, n = BEST_STANDARD_VANILLA, BEST_STANDARD_NEUROMOD
    with open(md_path, "w") as f:
        f.write("# Standard MNIST Results\n\n")
        f.write(
            f"Hyperparameters (Phase 6a sweep): vanilla lr={v.lr}, epochs={v.epochs};"
            f" neuromod lr={n.lr}, epochs={n.epochs}; batch_size=64\n\n"
        )
        f.write("| Method | Test Acc (mean ± std) |\n")
        f.write("|--------|----------------------|\n")
        for r in rows:
            f.write(f"| {r['method']} | {r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f} |\n")

    print(f"Saved {csv_path.name} and {md_path.name}")


# ---------------------------------------------------------------------------
# CL Split MNIST
# ---------------------------------------------------------------------------

_CL_BEST = {
    "naive": BEST_CL_NAIVE,
    "joint": BEST_CL_JOINT,
    "ewc":   BEST_CL_EWC,
    "er":    BEST_CL_ER,
}


def run_cl_table() -> tuple[list[dict], str]:
    print("\n" + "=" * 60)
    print("Split MNIST CL evaluation (test sequence seed=42, 3 seeds)")
    print("=" * 60)

    raw: dict[str, dict] = {}

    # --- Baselines ---
    for method, base_config in _CL_BEST.items():
        accs, forgettings = [], []
        for seed in SEEDS:
            config = base_config.__class__(**{**base_config.__dict__, "seed": seed})
            print(
                f"\n[{method}] seed={seed}  lr={config.lr}"
                f"  ept={config.epochs_per_task}"
            )
            acc, fgt = cl_train(config, method, no_wandb=True)
            accs.append(acc)
            forgettings.append(fgt)
        raw[method] = {"accs": accs, "forgettings": forgettings}

    # Best of {ewc, er} by mean avg_final_acc (excludes oracle joint and no-CL naive)
    best_bl = max(["ewc", "er"], key=lambda m: float(np.mean(raw[m]["accs"])))
    print(f"\n=> Best CL baseline: {best_bl}")

    # --- Neuromod standalone ---
    accs, forgettings = [], []
    for seed in SEEDS:
        config = BEST_CL_NEUROMOD.__class__(**{**BEST_CL_NEUROMOD.__dict__, "seed": seed})
        print(
            f"\n[neuromod] seed={seed}  lr={config.lr}"
            f"  ept={config.epochs_per_task}"
            f"  learned_proj={config.neuromod_learned_projection}"
        )
        acc, fgt = cl_train(config, "naive", no_wandb=True)
        accs.append(acc)
        forgettings.append(fgt)
    raw["neuromod"] = {"accs": accs, "forgettings": forgettings}

    # --- Neuromod + best baseline ---
    combined_key = f"neuromod+{best_bl}"
    best_bl_config = _CL_BEST[best_bl]
    accs, forgettings = [], []
    for seed in SEEDS:
        # Merge neuromod flags into best-baseline config
        merged = {**best_bl_config.__dict__, "seed": seed,
                  "use_neuromod": True,
                  "neuromod_variant": BEST_CL_NEUROMOD.neuromod_variant,
                  "neuromod_target": BEST_CL_NEUROMOD.neuromod_target,
                  "neuromod_learned_projection": BEST_CL_NEUROMOD.neuromod_learned_projection}
        config = best_bl_config.__class__(**merged)
        print(f"\n[{combined_key}] seed={seed}  lr={config.lr}  ept={config.epochs_per_task}")
        acc, fgt = cl_train(config, best_bl, no_wandb=True)
        accs.append(acc)
        forgettings.append(fgt)
    raw[combined_key] = {"accs": accs, "forgettings": forgettings}

    # Build rows in display order
    row_order = ["naive", "joint", "ewc", "er", "neuromod", combined_key]
    rows = []
    for method in row_order:
        r = raw[method]
        m_acc, s_acc = _mean_std(r["accs"])
        m_fgt, s_fgt = _mean_std(r["forgettings"])
        rows.append({
            "method": method,
            "avg_final_acc_mean": m_acc,
            "avg_final_acc_std": s_acc,
            "forgetting_mean": m_fgt,
            "forgetting_std": s_fgt,
        })
        print(f"=> {method}: acc={m_acc:.4f}±{s_acc:.4f}  forget={m_fgt:.4f}±{s_fgt:.4f}")

    return rows, best_bl


def save_cl_table(rows: list[dict]) -> None:
    csv_path = RESULTS_DIR / "split_mnist_table.csv"
    md_path = RESULTS_DIR / "split_mnist_table.md"

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "avg_final_acc_mean", "avg_final_acc_std",
            "forgetting_mean", "forgetting_std",
        ])
        w.writeheader()
        w.writerows(rows)

    with open(md_path, "w") as f:
        f.write("# Split MNIST CL Results\n\n")
        f.write(
            f"Hyperparameters (Phase 6b sweep, val-seq seed=7):"
            f" naive lr={BEST_CL_NAIVE.lr} ept={BEST_CL_NAIVE.epochs_per_task};"
            f" joint lr={BEST_CL_JOINT.lr} ept={BEST_CL_JOINT.epochs_per_task};"
            f" ewc lr={BEST_CL_EWC.lr} ept={BEST_CL_EWC.epochs_per_task}"
            f" λ={BEST_CL_EWC.ewc_lambda:.0e};"
            f" er lr={BEST_CL_ER.lr} ept={BEST_CL_ER.epochs_per_task}"
            f" buf={BEST_CL_ER.er_buffer_size};"
            f" neuromod lr={BEST_CL_NEUROMOD.lr} ept={BEST_CL_NEUROMOD.epochs_per_task}\n\n"
        )
        f.write("Seeds: 42, 43, 44 (test sequence class order: seed=42)\n\n")
        f.write("| Method | Avg Final Acc (mean ± std) | Forgetting (mean ± std) |\n")
        f.write("|--------|---------------------------|------------------------|\n")
        for r in rows:
            f.write(
                f"| {r['method']} "
                f"| {r['avg_final_acc_mean']:.4f} ± {r['avg_final_acc_std']:.4f} "
                f"| {r['forgetting_mean']:.4f} ± {r['forgetting_std']:.4f} |\n"
            )

    print(f"Saved {csv_path.name} and {md_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standard_rows = run_standard_table()
    save_standard_table(standard_rows)

    cl_rows, best_method = run_cl_table()
    save_cl_table(cl_rows)

    print("\n" + "=" * 60)
    print("Phase 7 complete")
    print("=" * 60)
    print("\nStandard MNIST:")
    for r in standard_rows:
        print(f"  {r['method']}: {r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f}")
    print(f"\nSplit MNIST (best CL baseline: {best_method}):")
    for r in cl_rows:
        print(
            f"  {r['method']}: "
            f"acc={r['avg_final_acc_mean']:.4f}±{r['avg_final_acc_std']:.4f}  "
            f"forget={r['forgetting_mean']:.4f}±{r['forgetting_std']:.4f}"
        )
