from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CLMethod(ABC):
    """Base interface for CL methods used in the sequential task loop."""

    @abstractmethod
    def train_task(
        self,
        task_id: int,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config,
    ) -> None: ...

    def on_task_end(
        self,
        task_id: int,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config,
    ) -> None:
        """Hook called after train_task, before evaluating the next task."""
        pass


class Naive(CLMethod):
    """Standard sequential fine-tuning — no forgetting prevention."""

    def train_task(self, task_id, model, train_loader, optimizer, criterion, device, config):
        model.train()
        for _ in range(config.epochs_per_task):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()


_REGISTRY: dict[str, type[CLMethod]] = {
    "naive": Naive,
}


def make_cl_method(name: str) -> CLMethod:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown CL method: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
