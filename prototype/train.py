import argparse
import random
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script (python prototype/train.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from prototype.configs import StandardConfig, CLConfig
from prototype.data import get_standard_loaders
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


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuromodulation prototype training")
    parser.add_argument("--standard", action="store_true", help="Run standard MNIST training")
    parser.add_argument("--method", choices=["naive", "joint", "ewc", "er"], default="naive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    # Hyperparameter overrides (Phase 6 sweeps will use these)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    # Neuromod flags (wired up in Phase 5)
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
        raise NotImplementedError("CL training loop implemented in Phase 3")


if __name__ == "__main__":
    main()
