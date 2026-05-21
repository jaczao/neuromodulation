"""Phase 6 hyperparameter sweep — never touches any test set.

(a) Standard-MNIST sweep:
    lr ∈ {3e-4, 1e-3}  ×  epochs ∈ {10, 20}  ×  use_neuromod ∈ {False, True}
    → 8 trials; select by val_acc (held-out 10k of training set)

(b) CL sweep (validation sequence seed=7):
    lr ∈ {3e-4, 1e-3}  ×  epochs_per_task ∈ {5, 10}
    + EWC:     ewc_lambda ∈ {1e4, 1e5}
    + ER:      er_buffer_size ∈ {200, 1000}
    + neuromod: learned_projection ∈ {False, True}
    → select by avg_final_acc on validation sequence

Usage:
    uv run python prototype/sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig, StandardConfig
from prototype.data import make_sequence
from prototype.train import cl_train, train_standard

# Validation sequence for CL sweep — NEVER the test sequence (seed=42)
VAL_SEQUENCE = make_sequence(7)

# Fixed seed for all sweep trials (reproducibility, not selection)
SWEEP_SEED = 42


# ---------------------------------------------------------------------------
# (a) Standard-MNIST sweep
# ---------------------------------------------------------------------------

def standard_sweep() -> dict[str, dict]:
    """Returns best config dict per model name: 'vanilla' and 'neuromod'."""
    print("\n" + "=" * 60)
    print("(a) Standard-MNIST sweep — select by val_acc")
    print("=" * 60)

    lrs = [3e-4, 1e-3]
    epochs_list = [10, 20]
    models = [("vanilla", False), ("neuromod", True)]
    best_per_model: dict[str, dict] = {}

    for model_name, use_neuromod in models:
        print(f"\n--- {model_name} MLP ---")
        best_val = -1.0
        best_cfg: dict = {}

        for lr in lrs:
            for epochs in epochs_list:
                config = StandardConfig(
                    lr=lr,
                    epochs=epochs,
                    seed=SWEEP_SEED,
                    use_neuromod=use_neuromod,
                )
                val_acc, test_acc = train_standard(config, no_wandb=True)
                tag = f"lr={lr:.0e} epochs={epochs:>2}"
                print(f"  {tag}  val={val_acc:.4f}  test={test_acc:.4f}")
                if val_acc > best_val:
                    best_val = val_acc
                    best_cfg = {
                        "lr": lr,
                        "epochs": epochs,
                        "use_neuromod": use_neuromod,
                        "val_acc": val_acc,
                    }

        best_per_model[model_name] = best_cfg
        print(
            f"  => Best: lr={best_cfg['lr']:.0e}  epochs={best_cfg['epochs']}"
            f"  val_acc={best_cfg['val_acc']:.4f}"
        )

    return best_per_model


# ---------------------------------------------------------------------------
# (b) CL sweep
# ---------------------------------------------------------------------------

def _cl_trial(config: CLConfig, method: str) -> tuple[float, float]:
    """Run one CL trial on the validation sequence."""
    return cl_train(config, method, no_wandb=True, sequence=VAL_SEQUENCE)


def cl_sweep() -> dict[str, dict]:
    """Returns best config dict per CL method."""
    print("\n" + "=" * 60)
    print("(b) CL sweep — validation sequence seed=7, select by avg_final_acc")
    print("=" * 60)

    lrs = [3e-4, 1e-3]
    epochs_list = [5, 10]
    best_per_method: dict[str, dict] = {}

    # --- naive ---
    print("\n--- naive ---")
    best_acc = -1.0
    best_cfg: dict = {}
    for lr in lrs:
        for epochs in epochs_list:
            config = CLConfig(lr=lr, epochs_per_task=epochs, seed=SWEEP_SEED)
            avg_acc, fgt = _cl_trial(config, "naive")
            print(f"  lr={lr:.0e} ept={epochs}  acc={avg_acc:.4f}  fgt={fgt:.4f}")
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_cfg = {"lr": lr, "epochs_per_task": epochs, "val_acc": avg_acc}
    best_per_method["naive"] = best_cfg
    print(f"  => Best: lr={best_cfg['lr']:.0e}  ept={best_cfg['epochs_per_task']}  acc={best_cfg['val_acc']:.4f}")

    # --- joint ---
    print("\n--- joint ---")
    best_acc = -1.0
    best_cfg = {}
    for lr in lrs:
        for epochs in epochs_list:
            config = CLConfig(lr=lr, epochs_per_task=epochs, seed=SWEEP_SEED)
            avg_acc, fgt = _cl_trial(config, "joint")
            print(f"  lr={lr:.0e} ept={epochs}  acc={avg_acc:.4f}  fgt={fgt:.4f}")
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_cfg = {"lr": lr, "epochs_per_task": epochs, "val_acc": avg_acc}
    best_per_method["joint"] = best_cfg
    print(f"  => Best: lr={best_cfg['lr']:.0e}  ept={best_cfg['epochs_per_task']}  acc={best_cfg['val_acc']:.4f}")

    # --- EWC ---
    # λ ∈ {1e4, 1e5}: per CLAUDE.md, per-sample Fisher ≈ 1e-5 so λ=1000 is negligible
    print("\n--- ewc ---")
    ewc_lambdas = [1e4, 1e5]
    best_acc = -1.0
    best_cfg = {}
    for lr in lrs:
        for epochs in epochs_list:
            for lam in ewc_lambdas:
                config = CLConfig(lr=lr, epochs_per_task=epochs, ewc_lambda=lam, seed=SWEEP_SEED)
                avg_acc, fgt = _cl_trial(config, "ewc")
                print(f"  lr={lr:.0e} ept={epochs} lam={lam:.0e}  acc={avg_acc:.4f}  fgt={fgt:.4f}")
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_cfg = {
                        "lr": lr,
                        "epochs_per_task": epochs,
                        "ewc_lambda": lam,
                        "val_acc": avg_acc,
                    }
    best_per_method["ewc"] = best_cfg
    print(
        f"  => Best: lr={best_cfg['lr']:.0e}  ept={best_cfg['epochs_per_task']}"
        f"  lam={best_cfg['ewc_lambda']:.0e}  acc={best_cfg['val_acc']:.4f}"
    )

    # --- ER ---
    print("\n--- er ---")
    buffers = [200, 1000]
    best_acc = -1.0
    best_cfg = {}
    for lr in lrs:
        for epochs in epochs_list:
            for buf in buffers:
                config = CLConfig(lr=lr, epochs_per_task=epochs, er_buffer_size=buf, seed=SWEEP_SEED)
                avg_acc, fgt = _cl_trial(config, "er")
                print(f"  lr={lr:.0e} ept={epochs} buf={buf:>4}  acc={avg_acc:.4f}  fgt={fgt:.4f}")
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_cfg = {
                        "lr": lr,
                        "epochs_per_task": epochs,
                        "er_buffer_size": buf,
                        "val_acc": avg_acc,
                    }
    best_per_method["er"] = best_cfg
    print(
        f"  => Best: lr={best_cfg['lr']:.0e}  ept={best_cfg['epochs_per_task']}"
        f"  buf={best_cfg['er_buffer_size']}  acc={best_cfg['val_acc']:.4f}"
    )

    # --- neuromod standalone (naive sequential + gain modulator) ---
    print("\n--- neuromod (gain, hidden, naive sequential) ---")
    learned_proj_opts = [False, True]
    best_acc = -1.0
    best_cfg = {}
    for lr in lrs:
        for epochs in epochs_list:
            for lp in learned_proj_opts:
                config = CLConfig(
                    lr=lr,
                    epochs_per_task=epochs,
                    seed=SWEEP_SEED,
                    use_neuromod=True,
                    neuromod_learned_projection=lp,
                )
                avg_acc, fgt = _cl_trial(config, "naive")
                print(f"  lr={lr:.0e} ept={epochs} learned_proj={lp!s:<5}  acc={avg_acc:.4f}  fgt={fgt:.4f}")
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_cfg = {
                        "lr": lr,
                        "epochs_per_task": epochs,
                        "neuromod_learned_projection": lp,
                        "val_acc": avg_acc,
                    }
    best_per_method["neuromod"] = best_cfg
    print(
        f"  => Best: lr={best_cfg['lr']:.0e}  ept={best_cfg['epochs_per_task']}"
        f"  learned_proj={best_cfg['neuromod_learned_projection']}  acc={best_cfg['val_acc']:.4f}"
    )

    return best_per_method


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(std_best: dict[str, dict], cl_best: dict[str, dict]) -> None:
    print("\n" + "=" * 60)
    print("PHASE 6 SUMMARY — best configs (by validation metric)")
    print("=" * 60)

    print("\n# Standard MNIST best configs (Phase 6a)")
    v = std_best["vanilla"]
    print(f"BEST_STANDARD_VANILLA  = StandardConfig(lr={v['lr']}, epochs={v['epochs']})")
    n = std_best["neuromod"]
    print(f"BEST_STANDARD_NEUROMOD = StandardConfig(lr={n['lr']}, epochs={n['epochs']}, use_neuromod=True)")

    print("\n# CL best configs (Phase 6b, validation sequence seed=7)")
    naive = cl_best["naive"]
    print(f"BEST_CL_NAIVE  = CLConfig(lr={naive['lr']}, epochs_per_task={naive['epochs_per_task']})")
    joint = cl_best["joint"]
    print(f"BEST_CL_JOINT  = CLConfig(lr={joint['lr']}, epochs_per_task={joint['epochs_per_task']})")
    ewc = cl_best["ewc"]
    print(f"BEST_CL_EWC    = CLConfig(lr={ewc['lr']}, epochs_per_task={ewc['epochs_per_task']}, ewc_lambda={ewc['ewc_lambda']})")
    er = cl_best["er"]
    print(f"BEST_CL_ER     = CLConfig(lr={er['lr']}, epochs_per_task={er['epochs_per_task']}, er_buffer_size={er['er_buffer_size']})")
    nm = cl_best["neuromod"]
    print(
        f"BEST_CL_NEUROMOD = CLConfig(lr={nm['lr']}, epochs_per_task={nm['epochs_per_task']},"
        f" use_neuromod=True, neuromod_learned_projection={nm['neuromod_learned_projection']})"
    )


if __name__ == "__main__":
    std_best = standard_sweep()
    cl_best = cl_sweep()
    print_summary(std_best, cl_best)
