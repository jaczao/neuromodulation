import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from prototype.configs import CLConfig, StandardConfig
from prototype.data import SplitMNIST, get_standard_loaders
from prototype.methods import make_cl_method
from prototype.model import MLP

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on MPS/CPU, needed for CUDA portability


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train_standard(config: StandardConfig, no_wandb: bool = False) -> float:
    """Train vanilla MLP on full MNIST. Returns test accuracy."""
    device = _device()
    seed_everything(config.seed)

    train_loader, val_loader, test_loader = get_standard_loaders(
        batch_size=config.batch_size,
        val_size=config.val_size,
    )
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    use_wandb = not no_wandb and _WANDB_AVAILABLE
    if use_wandb:
        _wandb.init(
            project="neuromod-cl-prototype",
            config={"lr": config.lr, "epochs": config.epochs, "batch_size": config.batch_size, "seed": config.seed},
            tags=["method=standard", "dataset=standard_mnist", f"seed={config.seed}", "use_neuromod=False"],
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:>2}/{config.epochs} | val_acc={val_acc:.4f}")
        if use_wandb:
            _wandb.log({"val_acc": val_acc, "epoch": epoch})

    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    if use_wandb:
        _wandb.log({"test_acc": test_acc})
        _wandb.finish()
    return test_acc


def _train_joint(
    model: nn.Module,
    split_mnist: SplitMNIST,
    config: CLConfig,
    device: torch.device,
    criterion: nn.Module,
) -> None:
    """Train on the union of all task data for config.epochs_per_task epochs."""
    all_datasets = [
        split_mnist.get_task_loaders(t, config.batch_size)[0].dataset
        for t in range(split_mnist.n_tasks)
    ]
    combined_loader = DataLoader(
        ConcatDataset(all_datasets), batch_size=config.batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(1, config.epochs_per_task + 1):
        model.train()
        for x, y in combined_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"  Joint epoch {epoch}/{config.epochs_per_task}")


def cl_train(
    config: CLConfig,
    method_name: str,
    no_wandb: bool = False,
) -> tuple[float, float]:
    """CL training loop. Returns (avg_final_acc, forgetting)."""
    device = _device()
    seed_everything(config.seed)

    split_mnist = SplitMNIST()
    T = split_mnist.n_tasks
    # A[t, i] = accuracy on task i after training on task t; NaN = not yet evaluated
    A = np.full((T, T), np.nan)
    criterion = nn.CrossEntropyLoss()
    model = MLP().to(device)

    use_wandb = not no_wandb and _WANDB_AVAILABLE
    if use_wandb:
        _wandb.init(
            project="neuromod-cl-prototype",
            config={
                "lr": config.lr,
                "epochs_per_task": config.epochs_per_task,
                "batch_size": config.batch_size,
                "seed": config.seed,
                "method": method_name,
            },
            tags=[
                f"method={method_name}",
                "dataset=split_mnist",
                f"seed={config.seed}",
                "use_neuromod=False",
                "neuromod_variant=none",
                "neuromod_target=none",
            ],
        )

    if method_name == "joint":
        _train_joint(model, split_mnist, config, device, criterion)
        t = T - 1
        for i in range(T):
            _, test_loader_i = split_mnist.get_task_loaders(i, config.batch_size)
            A[t, i] = evaluate(model, test_loader_i, device)
            if use_wandb:
                _wandb.log({f"acc/task_{i}": A[t, i]})
        print(f"Joint | per-task accs: [{', '.join(f'{A[t,i]:.3f}' for i in range(T))}]")
    else:
        method = make_cl_method(method_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        for t in range(T):
            train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
            method.train_task(t, model, train_loader, optimizer, criterion, device, config)
            method.on_task_end(t, model, train_loader, device, config)
            for i in range(t + 1):
                _, test_loader_i = split_mnist.get_task_loaders(i, config.batch_size)
                A[t, i] = evaluate(model, test_loader_i, device)
                if use_wandb:
                    _wandb.log({f"acc/task_{i}": A[t, i], "after_task": t})
            seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
            print(f"After task {t + 1}/{T} | seen tasks: [{seen}]")

    # avg_final_acc = mean over all tasks of final row
    avg_final_acc = float(np.nanmean(A[T - 1, :]))

    # forgetting = mean over all tasks of (peak acc seen - final acc)
    # Per spec: mean over i < T; last task always contributes 0
    forget_vals = []
    for i in range(T):
        col = [A[t, i] for t in range(i, T) if not np.isnan(A[t, i])]
        if col:
            forget_vals.append(max(col) - A[T - 1, i])
    forgetting = float(np.mean(forget_vals)) if forget_vals else 0.0

    print(f"\navg_final_acc={avg_final_acc:.4f} | forgetting={forgetting:.4f}")
    if use_wandb:
        _wandb.log({"avg_final_acc": avg_final_acc, "forgetting": forgetting})
        _wandb.finish()

    return avg_final_acc, forgetting


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuromodulation prototype training")
    parser.add_argument("--standard", action="store_true", help="Run standard MNIST training")
    parser.add_argument("--method", choices=["naive", "joint", "ewc", "er"], default="naive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    # Hyperparameter overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epochs-per-task", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--er-buffer-size", type=int, default=None)
    # Neuromod flags (wired in Phase 5)
    parser.add_argument("--use-neuromod", action="store_true")
    parser.add_argument("--neuromod-variant", type=str, default=None)
    parser.add_argument("--neuromod-target", type=str, default=None)
    args = parser.parse_args()

    if args.standard:
        config = StandardConfig(seed=args.seed)
        if args.lr is not None:
            config.lr = args.lr
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        train_standard(config, no_wandb=args.no_wandb)
    else:
        config = CLConfig(seed=args.seed)
        if args.lr is not None:
            config.lr = args.lr
        if args.epochs_per_task is not None:
            config.epochs_per_task = args.epochs_per_task
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.ewc_lambda is not None:
            config.ewc_lambda = args.ewc_lambda
        if args.er_buffer_size is not None:
            config.er_buffer_size = args.er_buffer_size
        cl_train(config, args.method, no_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
