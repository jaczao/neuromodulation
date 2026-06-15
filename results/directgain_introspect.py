"""Introspect the direct-gain modulator (last_hidden gate) from the last CL experiment.

Reproduces CL naive, seed 42, target=direct_gain, gate=last_hidden, then on test batches dumps:
  (1) outputs of the neuromod net: gain m(x) and gain=1+m distribution
  (2) "projection matrix": direct_gain has NONE by design; the analogous learned object is the
      head weight W (400x784). Report its singular values / effective rank (does the direct head
      re-learn a low-rank map, i.e. reinvent pt1's k=8 bottleneck?). pt1's fixed P_l shown for ref.
  (3) projection of the neuromod outputs: SVD of the per-sample gain matrix G (B,400): how many
      effective dims the gains span (vs pt1's hard k=8).
  (4) non-gated vs gated last hidden layer: h2 vs (1+m)*h2.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from prototype.configs import CLConfig
from prototype.data import SplitMNIST
from prototype.neuromod import make_modulator
from prototype.train import _build_model, _device, seed_everything


def eff_rank(s: torch.Tensor) -> float:
    """Participation-ratio effective rank of singular values s: (sum s)^2 / sum s^2."""
    s = s[s > 0]
    return float((s.sum() ** 2) / (s ** 2).sum())


ap = argparse.ArgumentParser()
ap.add_argument("--bounded", action="store_true", help="gain=1+tanh(m) in [0,2] vs unbounded 1+m")
args = ap.parse_args()
print(f"### direct-gain introspection | gate=last_hidden | bounded={args.bounded} ###")

seed_everything(42)
dev = _device()
cfg = CLConfig(seed=42, lr=1e-3, epochs_per_task=5, batch_size=64,
               use_neuromod=True, neuromod_target="direct_gain", neuromod_gain_gate="last_hidden",
               neuromod_gain_bounded=args.bounded)
model = _build_model(cfg, dev)
sm = SplitMNIST(sequence=None)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
crit = nn.CrossEntropyLoss()
for t in range(sm.n_tasks):
    tl, _ = sm.get_task_loaders(t, cfg.batch_size)
    model.train()
    for _ in range(cfg.epochs_per_task):
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()

model.eval()
mod = model.modulator
head = mod.heads["h1"]          # Linear(784 -> 400): the direct gain head (last hidden layer)
W = head.weight.detach()        # (400, 784)
base = model.base               # MLP

# Collect a test batch from an OLD task (0) and the LAST task (4) to compare input-conditioned gains.
def task_batch(t, n=512):
    _, te = sm.get_task_loaders(t, n)
    x, y = next(iter(te))
    return x.to(dev), y.to(dev)

print("=" * 78)
print("(2) 'PROJECTION MATRIX': direct_gain has none. Analogous object = head weight W(400x784).")
print(f"    W shape={tuple(W.shape)}  ||W||_F={W.norm():.4f}  mean={W.mean():+.5f}  std={W.std():.5f}")
sv = torch.linalg.svdvals(W.float().cpu())
print(f"    top-10 singular values: {[round(v,3) for v in sv[:10].tolist()]}")
print(f"    effective rank (participation ratio) of W = {eff_rank(sv):.2f}  (full = min(400,784)=400)")
print(f"    -> pt1 used a FIXED random P_l of shape (k=8, 400); here the 784->400 map is learned,")
print(f"       eff-rank tells whether it collapses toward a low-dim (pt1-like) map.")

for t in (0, 4):
    x, y = task_batch(t)
    xf = x.view(x.size(0), -1)
    with torch.no_grad():
        m = head(xf)                    # (B,400) raw neuromod-net output
        gain = mod._gain(m)             # 1+tanh(m) if bounded else 1+m  (the applied gain)
        # recompute raw (non-gated) last hidden h2, then the gated version
        h1 = base.net[1](base.net[0](xf))
        h2_raw = base.net[3](base.net[2](h1))      # (B,400) ungated last hidden
        h2_gated = gain * h2_raw                    # what ModulatedMLP actually feeds to net[4]
        # SVD of the per-sample gain matrix (how many effective dims the gains span)
        Gc = gain - gain.mean(0, keepdim=True)
        sg = torch.linalg.svdvals(Gc.float().cpu())
    print("=" * 78)
    print(f"TASK {t} TEST BATCH (B={x.size(0)})  classes={sorted(set(y.tolist()))}")
    print(f"(1) neuromod-net output m:  mean={m.mean():+.4f} std={m.std():.4f} "
          f"min={m.min():+.4f} max={m.max():+.4f}")
    print(f"    gain=1+m:               mean={gain.mean():+.4f} std={gain.std():.4f} "
          f"min={gain.min():+.4f} max={gain.max():+.4f}  frac(gain<0)={ (gain<0).float().mean():.4f}")
    print(f"(3) per-sample gain vectors (B,400): top-8 singular values "
          f"{[round(v,2) for v in sg[:8].tolist()]}")
    print(f"    effective rank of the gain set = {eff_rank(sg):.2f}  (pt1 hard-capped this at k=8)")
    print(f"(4) non-gated vs gated last hidden:")
    print(f"    h2_raw   : mean={h2_raw.mean():.4f} ||.||={h2_raw.norm(dim=1).mean():.3f} "
          f"sparsity={(h2_raw<=1e-6).float().mean():.3f}")
    print(f"    h2_gated : mean={h2_gated.mean():.4f} ||.||={h2_gated.norm(dim=1).mean():.3f} "
          f"sparsity={(h2_gated<=1e-6).float().mean():.3f}")
    cos = torch.cosine_similarity(h2_raw, h2_gated, dim=1).mean()
    rel = ((h2_gated - h2_raw).norm(dim=1) / (h2_raw.norm(dim=1) + 1e-8)).mean()
    print(f"    cos(h2_raw,h2_gated)={cos:.4f}  mean relative change ||Δ||/||h2||={rel:.4f}")
