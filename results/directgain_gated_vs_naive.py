"""Difference between the GATED last-hidden vector (direct_gain, last_hidden) and the
NON-GATED last-hidden vector from the NAIVE baseline (vanilla MLP, no modulation).

Regimes: standard (full MNIST) and CL (Split MNIST, method=naive).
Variants: direct_gain without tanh (gain=1+m) and with tanh (gain=1+tanh(m)).
For each (regime, variant) we train, on the SAME seed/init, a vanilla baseline and the
direct_gain model, then on probe batches (digits {0,1} and {8,9}) report:
  ||h2_naive||  ||h2_gated||  ||Δ||=||h2_gated-h2_naive||  ||Δ||/||h2_naive||  cos(naive,gated)
No rank analysis. NOTE: baseline and direct_gain are two separately trained nets (same init),
so the 400 hidden units are only approximately aligned; read cos with that caveat.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from prototype.configs import CLConfig, StandardConfig
from prototype.data import SplitMNIST, get_standard_loaders
from prototype.neuromod import ModulatedMLP
from prototype.train import _build_model, _device, evaluate, seed_everything

dev = _device()
sm = SplitMNIST(sequence=None)          # probe batches (digits {0,1}=task0, {8,9}=task4)


def train(cfg, regime):
    seed_everything(cfg.seed)
    model = _build_model(cfg, dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if regime == "standard":
        tr, _va, te = get_standard_loaders(batch_size=cfg.batch_size, val_size=10_000)
        for _ in range(cfg.epochs):
            model.train()
            for x, y in tr:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        return model, evaluate(model, te, dev)
    for t in range(sm.n_tasks):          # cl naive (sequential fine-tuning)
        tl, _ = sm.get_task_loaders(t, cfg.batch_size)
        model.train()
        for _ in range(cfg.epochs_per_task):
            for x, y in tl:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model, None


def last_hidden(model, x):
    """Last-hidden vector actually used by net[4]: gated if direct_gain, plain otherwise."""
    xf = x.view(x.size(0), -1)
    if isinstance(model, ModulatedMLP):
        base, mod = model.base, model.modulator
        h2 = base.net[3](base.net[2](base.net[1](base.net[0](xf))))
        gain = mod._gain(mod.heads["h1"](xf))     # 1+tanh(m) or 1+m
        return gain * h2
    net = model.net
    return net[3](net[2](net[1](net[0](xf))))


def probe(t, n=512):
    _, te = sm.get_task_loaders(t, n)
    x, y = next(iter(te))
    return x.to(dev), y.to(dev)


def cfg_for(regime, **extra):
    if regime == "standard":
        return StandardConfig(seed=42, lr=3e-4, epochs=20, batch_size=64, **extra)
    return CLConfig(seed=42, lr=1e-3, epochs_per_task=5, batch_size=64, **extra)


DG = dict(use_neuromod=True, neuromod_target="direct_gain", neuromod_gain_gate="last_hidden")

for regime in ("standard", "cl"):
    naive_model, acc = train(cfg_for(regime), regime)
    accs = f"(naive test_acc={acc:.4f})" if acc is not None else ""
    print("\n" + "#" * 80)
    print(f"REGIME = {regime}   {accs}")
    for bounded in (False, True):
        dg_model, dacc = train(cfg_for(regime, neuromod_gain_bounded=bounded, **DG), regime)
        tag = "tanh " if bounded else "no-tanh"
        dstr = f" (dg test_acc={dacc:.4f})" if dacc is not None else ""
        print(f"-- direct_gain {tag}{dstr} : gated vs non-gated(naive) --")
        for t in (0, 4):
            x, _y = probe(t)
            with torch.no_grad():
                hn = last_hidden(naive_model, x)          # non-gated (naive baseline)
                hg = last_hidden(dg_model, x)             # gated (direct_gain)
                nn_ = hn.norm(dim=1)
                ng_ = hg.norm(dim=1)
                d = (hg - hn).norm(dim=1)
                rel = (d / (nn_ + 1e-8)).mean()
                cos = torch.cosine_similarity(hn, hg, dim=1).mean()
            print(f"   task{t} (digits {sorted(set(_y.tolist()))}): "
                  f"||h2_naive||={nn_.mean():.3f}  ||h2_gated||={ng_.mean():.3f}  "
                  f"||Δ||={d.mean():.3f}  ||Δ||/||h2_naive||={rel:.3f}  cos={cos:.3f}")
