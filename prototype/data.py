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
        val_frac: float = 0.0,
        val_seed: int = 12345,
    ) -> None:
        self.root = root or str(_ROOT)
        self.sequence = sequence if sequence is not None else list(_DEFAULT_CLASS_PAIRS)
        self.val_frac = val_frac
        self.val_seed = val_seed
        self._train_ds = datasets.MNIST(root=self.root, train=True, download=True, transform=_TRANSFORM)
        self._test_ds = datasets.MNIST(root=self.root, train=False, download=True, transform=_TRANSFORM)

    @property
    def n_tasks(self) -> int:
        return len(self.sequence)

    def _task_train_val_idx(self, task_id: int) -> tuple[list[int], list[int]]:
        """Partition task_id's TRAIN indices into (train_idx, val_idx).

        val_frac <= 0 → val_idx is empty and train_idx is the full task train set (the
        historical default, so non-`--val` runs are unchanged). The held-out val split is
        seeded by the class pair — NOT the model seed or task order — so the same images
        are held out across every seed and every task ordering (never touches the test set).
        """
        classes = sorted(self.sequence[task_id])
        all_idx = [i for i, label in enumerate(self._train_ds.targets.tolist()) if label in set(classes)]
        if self.val_frac <= 0.0:
            return all_idx, []
        rng = np.random.default_rng([self.val_seed, classes[0], classes[1]])
        perm = rng.permutation(len(all_idx))
        n_val = int(round(len(all_idx) * self.val_frac))
        all_arr = np.asarray(all_idx)
        return all_arr[perm[n_val:]].tolist(), all_arr[perm[:n_val]].tolist()

    def get_task_loaders(self, task_id: int, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader). train excludes the val split when val_frac > 0."""
        train_idx, _ = self._task_train_val_idx(task_id)
        classes = set(self.sequence[task_id])
        test_idx = [i for i, label in enumerate(self._test_ds.targets.tolist()) if label in classes]
        train_loader = DataLoader(Subset(self._train_ds, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(self._test_ds, test_idx), batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def get_task_val_loader(self, task_id: int, batch_size: int = 64) -> DataLoader:
        """Return the held-out validation loader (carved from TRAIN). Requires val_frac > 0."""
        _, val_idx = self._task_train_val_idx(task_id)
        if not val_idx:
            raise ValueError("get_task_val_loader requires val_frac > 0 (no validation split configured)")
        return DataLoader(Subset(self._train_ds, val_idx), batch_size=batch_size, shuffle=False)


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
