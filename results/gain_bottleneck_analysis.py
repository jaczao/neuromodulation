"""Bottlenecked gain net (pt1 GainModulator: 784->64->k=8 signal, then fixed projection P_l):
gated-vs-naive difference AND rank analysis, on standard and CL.

Contrast with the no-bottleneck direct_gain runs. The bottlenecked net gates h2 with
gain = 1 + signal_net(x) @ P_1, where signal_net outputs k=8 and P_1 is (8,400). So the gain
delta lives in an <=8-dim subspace BY CONSTRUCTION (rank cap = k). We show:
  (rank) P_1 projection matrix: singular values + effective rank (<= 8)
  (1) gain delta / gain distribution
  (3) per-sample gain-set effective rank (should hit the k=8 cap)
  (4) gated h2 vs non-gated h2 from the NAIVE baseline: ||.||, ||Δ||, ||Δ||/||naive||, cos
Probe batches: digits {0,1}=task0, {8,9}=task4. NOTE: naive and gain are two separately trained
nets (same init), so units are only approximately aligned (read cos with that caveat).
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
sm = SplitMNIST(sequence=None)


def eff_rank(s):
    s = s[s > 0]
    return float((s.sum() ** 2) / (s ** 2).sum())


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
    for t in range(sm.n_tasks):
        tl, _ = sm.get_task_loaders(t, cfg.batch_size)
        model.train()
        for _ in range(cfg.epochs_per_task):
            for x, y in tl:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model, None


def naive_h2(model, x):
    xf = x.view(x.size(0), -1)
    net = model.net
    return net[3](net[2](net[1](net[0](xf))))


def gain_h2(model, x):
    """Bottlenecked GainModulator: real gated last-hidden (h1 gated upstream too) + the gain vec."""
    xf = x.view(x.size(0), -1)
    base, mod = model.base, model.modulator
    h1 = mod.modulate(base.net[1](base.net[0](xf)), xf, 0)      # gated h1
    h2_pre = base.net[3](base.net[2](h1))                       # last hidden, pre-gain
    h2_g = mod.modulate(h2_pre, xf, 1)                          # gated last hidden
    gain = 1.0 + (mod.signal_net(xf) @ mod.P_1)                 # the h2 gain vector (B,400)
    return h2_g, gain


def probe(t, n=512):
    _, te = sm.get_task_loaders(t, n)
    x, y = next(iter(te))
    return x.to(dev), y.to(dev)


def cfg_for(regime, **extra):
    if regime == "standard":
        return StandardConfig(seed=42, lr=3e-4, epochs=20, batch_size=64, **extra)
    return CLConfig(seed=42, lr=1e-3, epochs_per_task=5, batch_size=64, **extra)


GAIN = dict(use_neuromod=True, neuromod_target="activation")   # pt1 bottlenecked gain modulator

for regime in ("standard", "cl"):
    naive_model, nacc = train(cfg_for(regime), regime)
    gain_model, gacc = train(cfg_for(regime, **GAIN), regime)
    mod = gain_model.modulator
    P1 = mod.P_1.detach()                                       # (8, 400) fixed projection
    sp = torch.linalg.svdvals(P1.float().cpu())
    print("\n" + "#" * 80)
    na = f"naive={nacc:.4f} gain={gacc:.4f}" if nacc is not None else "(class-IL)"
    print(f"REGIME = {regime}   {na}")
    print(f"(rank) projection P_1 shape={tuple(P1.shape)}  singular values="
          f"{[round(v,3) for v in sp.tolist()]}  eff_rank={eff_rank(sp):.2f}  (cap = k = 8)")
    for t in (0, 4):
        x, y = probe(t)
        with torch.no_grad():
            hn = naive_h2(naive_model, x)
            hg, gain = gain_h2(gain_model, x)
            gd = gain - 1.0                                     # gain delta = sig @ P_1
            Gc = gain - gain.mean(0, keepdim=True)
            sg = torch.linalg.svdvals(Gc.float().cpu())
            d = (hg - hn).norm(dim=1)
            rel = (d / (hn.norm(dim=1) + 1e-8)).mean()
            cos = torch.cosine_similarity(hn, hg, dim=1).mean()
        print(f"  task{t} (digits {sorted(set(y.tolist()))}):")
        print(f"    (1) gain delta: mean={gd.mean():+.4f} std={gd.std():.4f} "
              f"[{gd.min():+.3f},{gd.max():+.3f}]   gain: mean={gain.mean():+.4f} "
              f"frac(gain<0)={(gain<0).float().mean():.3f}")
        print(f"    (3) gain-set eff_rank={eff_rank(sg):.2f}  (cap = k = 8)  "
              f"top-8 sv={[round(v,2) for v in sg[:8].tolist()]}")
        print(f"    (4) ||h2_naive||={hn.norm(dim=1).mean():.3f}  ||h2_gain||={hg.norm(dim=1).mean():.3f}  "
              f"||Δ||={d.mean():.3f}  ||Δ||/||naive||={rel:.3f}  cos={cos:.3f}")
