from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])

_ROOT = Path.home() / ".cache" / "mnist"
_DEFAULT_CLASS_PAIRS: list[tuple[int, int]] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def make_sequence(seed: int) -> list[tuple[int, int]]:
    """Return a permutation of the 5 class pairs, seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    pairs = list(_DEFAULT_CLASS_PAIRS)
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order]


def task_indices(
    task_id: int,
    split: str,
    sequence: list[tuple[int, int]] | None = None,
) -> list[int]:
    """Return MNIST dataset indices belonging to task_id's class pair.

    Args:
        task_id: 0-indexed task number.
        split: 'train' or 'test'.
        sequence: list of (class_a, class_b) pairs; defaults to _DEFAULT_CLASS_PAIRS.
    """
    if sequence is None:
        sequence = _DEFAULT_CLASS_PAIRS
    classes = set(sequence[task_id])
    ds = datasets.MNIST(root=str(_ROOT), train=(split == "train"), download=True, transform=_TRANSFORM)
    targets: list[int] = ds.targets.tolist()
    return [i for i, label in enumerate(targets) if label in classes]


class SplitMNIST:
    """5-task Split MNIST — each task is a 2-class subset, one shared head (class-IL)."""

    def __init__(
        self,
        root: str | None = None,
        sequence: list[tuple[int, int]] | None = None,
    ) -> None:
        self.root = root or str(_ROOT)
        self.sequence = sequence if sequence is not None else list(_DEFAULT_CLASS_PAIRS)
        self._train_ds = datasets.MNIST(root=self.root, train=True, download=True, transform=_TRANSFORM)
        self._test_ds = datasets.MNIST(root=self.root, train=False, download=True, transform=_TRANSFORM)

    @property
    def n_tasks(self) -> int:
        return len(self.sequence)

    def get_task_loaders(self, task_id: int, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader) for the given task."""
        classes = set(self.sequence[task_id])
        train_idx = [i for i, label in enumerate(self._train_ds.targets.tolist()) if label in classes]
        test_idx = [i for i, label in enumerate(self._test_ds.targets.tolist()) if label in classes]
        train_loader = DataLoader(Subset(self._train_ds, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(self._test_ds, test_idx), batch_size=batch_size, shuffle=False)
        return train_loader, test_loader


def get_standard_loaders(
    root: str | None = None,
    batch_size: int = 64,
    val_size: int = 10_000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for full-MNIST standard training.

    Val split is the last val_size examples of the 60k training set, never the official test set.
    """
    root = root or str(_ROOT)
    full_train = datasets.MNIST(root=root, train=True, download=True, transform=_TRANSFORM)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=_TRANSFORM)

    n = len(full_train)
    train_idx = list(range(n - val_size))
    val_idx = list(range(n - val_size, n))

    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
