"""Iteration 2 debugging checklist: did the weight mask actually move / differentiate?

Trains weight_mask on the validation sequence (naive, best val config) and, after each
task, logs: (1) mask range over the whole layer, (2) the mask computed on EACH seen
task's own data, to test the "different contexts -> different masks" hypothesis. Also
runs a low-rank (rank=16) variant and a separate higher modulator-LR variant as
checklist items 1/4/10 (output distribution, LR ratio, capacity).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from prototype.configs import CLConfig
from prototype.data import SplitMNIST, make_sequence
from prototype.model import MLP
from prototype.neuromod import WeightMaskMLP, make_modulator
from prototype.train import _device, evaluate, seed_everything


def _task_context(split, t, device, n=256):
    loader, _ = split.get_task_loaders(t, 64)
    xs = []
    for x, _ in loader:
        xs.append(x)
        if sum(len(z) for z in xs) >= n:
            break
    return torch.cat(xs)[:n].to(device)


def run(rank: int, mod_lr_mult: float, tag: str):
    print(f"\n===== {tag}: rank={rank} mod_lr_mult={mod_lr_mult} =====")
    device = _device()
    seed_everything(7)
    split = SplitMNIST(sequence=make_sequence(7))
    T = split.n_tasks

    base = MLP().to(device)
    lin = base.net[2]
    mod = make_modulator(
        "weight_mask", mask_dims=(lin.out_features, lin.in_features),
        mask_rank=rank, mask_init=0.99,
    )
    model = WeightMaskMLP(base, mod, layer_idx=2).to(device)

    if mod_lr_mult == 1.0:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        mod_ids = {id(p) for p in model.modulator.parameters()}
        net_params = [p for p in model.parameters() if id(p) not in mod_ids]
        opt = torch.optim.Adam([
            {"params": net_params, "lr": 1e-3},
            {"params": list(model.modulator.parameters()), "lr": 1e-3 * mod_lr_mult},
        ])
    crit = nn.CrossEntropyLoss()

    for t in range(T):
        loader, _ = split.get_task_loaders(t, 64)
        model.train()
        for _ in range(5):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
        with torch.no_grad():
            full = model.modulator.compute_mask(_task_context(split, t, device))
            per_task = [model.modulator.compute_mask(_task_context(split, i, device)) for i in range(t + 1)]
            # pairwise max abs difference between task masks (are they distinct?)
            maxdiff = 0.0
            for i in range(len(per_task)):
                for j in range(i + 1, len(per_task)):
                    maxdiff = max(maxdiff, (per_task[i] - per_task[j]).abs().max().item())
        print(f"  after task {t}: mask[min={full.min():.4f} max={full.max():.4f} "
              f"mean={full.mean():.4f} std={full.std():.4f}] "
              f"max cross-task mask diff={maxdiff:.4f}")

    accs = []
    for i in range(T):
        _, test = split.get_task_loaders(i, 64)
        accs.append(evaluate(model, test, device))
    print(f"  per-task acc: {[round(a,3) for a in accs]}  avg={sum(accs)/T:.4f}")


if __name__ == "__main__":
    run(rank=0, mod_lr_mult=1.0, tag="full-rank, shared LR (as swept)")
    run(rank=0, mod_lr_mult=50.0, tag="full-rank, modulator LR x50 (checklist item 4)")
    run(rank=16, mod_lr_mult=50.0, tag="low-rank r=16, modulator LR x50 (checklist items 1/10)")
