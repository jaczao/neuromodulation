import random
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


class EWC(CLMethod):
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Accumulates per-task Fisher penalties; each new task's loss adds
    (λ/2) Σ_i F_i · (θ - θ_i*)² summed over previous tasks.
    """

    def __init__(self):
        self.fishers: list[dict[str, torch.Tensor]] = []
        self.snapshots: list[dict[str, torch.Tensor]] = []

    def train_task(self, task_id, model, train_loader, optimizer, criterion, device, config):
        model.train()
        for _ in range(config.epochs_per_task):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                for fisher, snapshot in zip(self.fishers, self.snapshots):
                    for name, p in model.named_parameters():
                        if name in fisher:
                            loss = loss + (config.ewc_lambda / 2) * (
                                fisher[name] * (p - snapshot[name]) ** 2
                            ).sum()
                loss.backward()
                optimizer.step()

    def on_task_end(self, task_id, model, train_loader, device, config):
        """Compute Fisher diagonal from ~ewc_samples training samples, store θ*."""
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()
        }
        model.eval()

        # Collect exactly ewc_samples examples (or all available)
        xs, ys = [], []
        count = 0
        for x, y in train_loader:
            take = config.ewc_samples - count
            xs.append(x[:take])
            ys.append(y[:take])
            count += min(take, len(x))
            if count >= config.ewc_samples:
                break

        samples_x = torch.cat(xs).to(device)
        samples_y = torch.cat(ys).to(device)
        n_samples = len(samples_x)

        # Per-sample gradients are required: batch-mean gradients squared underestimate
        # the Fisher diagonal by ~B× due to the inequality E[g²] ≥ (E[g])².
        ewc_criterion = nn.CrossEntropyLoss()
        for xi, yi in zip(samples_x, samples_y):
            model.zero_grad()
            loss = ewc_criterion(model(xi.unsqueeze(0)), yi.unsqueeze(0))
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.data ** 2

        for name in fisher:
            fisher[name] /= n_samples

        self.fishers.append({n: f.detach().clone() for n, f in fisher.items()})
        self.snapshots.append({n: p.detach().clone() for n, p in model.named_parameters()})


class ER(CLMethod):
    """Experience Replay with reservoir sampling (fixed-size buffer)."""

    def __init__(self):
        self.buf_x: list[torch.Tensor] = []  # stored on CPU
        self.buf_y: list[torch.Tensor] = []
        self._n_seen: int = 0

    def train_task(self, task_id, model, train_loader, optimizer, criterion, device, config):
        model.train()
        for _ in range(config.epochs_per_task):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # Buffer update happens before the gradient step (per spec)
                self._update_buffer(x, y, config.er_buffer_size)

                if self.buf_x:
                    buf_x, buf_y = self._sample(len(x), device)
                    combined_x = torch.cat([x, buf_x], dim=0)
                    combined_y = torch.cat([y, buf_y], dim=0)
                else:
                    combined_x, combined_y = x, y

                optimizer.zero_grad()
                loss = criterion(model(combined_x), combined_y)
                loss.backward()
                optimizer.step()

    def _update_buffer(self, x: torch.Tensor, y: torch.Tensor, max_size: int) -> None:
        for xi, yi in zip(x.cpu(), y.cpu()):
            self._n_seen += 1
            if len(self.buf_x) < max_size:
                self.buf_x.append(xi)
                self.buf_y.append(yi)
            else:
                # Reservoir sampling: keep with probability max_size / n_seen
                j = random.randrange(self._n_seen)
                if j < max_size:
                    self.buf_x[j] = xi
                    self.buf_y[j] = yi

    def _sample(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = random.choices(range(len(self.buf_x)), k=n)
        buf_x = torch.stack([self.buf_x[i] for i in indices]).to(device)
        buf_y = torch.stack([self.buf_y[i] for i in indices]).to(device)
        return buf_x, buf_y


_REGISTRY: dict[str, type[CLMethod]] = {
    "naive": Naive,
    "ewc": EWC,
    "er": ER,
}


def make_cl_method(name: str) -> CLMethod:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown CL method: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
