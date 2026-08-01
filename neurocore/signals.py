"""The driver / signal bank — the thesis contribution, reused by every problem direction.

Extracted verbatim from results/pt7_neuromodulators.py (`Signals`, the loss/entropy primitives) and
results/pt7_variants.py (`NEDriver`, the head-free novelty drivers).

BACKBONE CONTRACT. `Signals.targets` and `NEDriver.value` need exactly one thing from the network
they observe:

    net.plain(x) -> (logits, features)

`features` is the penultimate representation (h1 in the pt7 MLP) and is what the embedding-novelty
drivers (NE_emb, emb_all, vec_h1, vec_h1proj) difference against a running mean. Any backbone that
exposes `plain` can drive the whole bank; nothing else about the architecture is assumed.

STANDARDIZATION (two rules, both learned the hard way in pt7 — see the driver taxonomy below):
  - PER-SAMPLE drivers: standardize. They carry per-sample variation at wildly different raw scales
    (5HT ~ -3 vs NE up to 1.2e6), and an unstandardized mixed-scale composite blows an unbounded
    rank-K gate to NaN (all4 un-standardised at a tuned SGD lr -> 0.098 = chance).
  - TONIC / scalar drivers: NEVER standardize. Their per-batch variance is ~0, so dividing by
    sqrt(run_var) divides by ~0: |g| blows to 10-17 and the run collapses. The same driver is merely
    INERT if left unstandardized. Standardization turns "inert" into "catastrophic" here.
Standardization is LINEAR, so it preserves the K+1-matmul trick that makes the per-synapse gate
tractable, and it makes a true-signal eval a clean upper bound (same scale the head regressed).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import BF, BS, DEV, EPS

H1 = 400                                         # feature (penultimate) width the novelty drivers see
PROJ_DIM = 32                                    # random-projection width for vecproj / vec_h1proj

# driver -> the gate LAYERS it is allowed to touch (per SPEC "native" mapping). None = all three.
DRIVER_LAYERS = {"NE_emb": ("out",)}             # within-forward: last hidden novelty -> out only


# ------------------------------- loss / entropy primitives -------------------------------
def per_sample_masked_ce(logits, y):
    task = torch.div(y, 2, rounding_mode="floor")
    allowed = torch.stack([2 * task, 2 * task + 1], dim=1)
    add = torch.full_like(logits, float("-inf")); add.scatter_(1, allowed, 0.0)
    return F.cross_entropy(logits + add, y, reduction="none")


def masked_ce(logits, y):
    return per_sample_masked_ce(logits, y).mean()


def per_sample_ce_plain(logits, y):
    return F.cross_entropy(logits, y, reduction="none")


def entropy(logits):
    logp = F.log_softmax(logits, dim=1)
    return -(logp.exp() * logp).sum(1)


# ------------------------------- head-regressed biological drivers -------------------------------
class Signals:
    """Per-sample biological targets tau_k from a plain (unmodulated) detached forward + running EMA state.

    Targets are STANDARDIZED per driver by running mean/var so the K drivers enter the linear gate at a
    comparable unit scale (else a large-magnitude driver like 5HT=-loss blows the K=4 gate up). Only the
    per-sample VARIATION matters to the gate (scale is absorbed by P), so standardizing is neutral; it also
    makes the `true`-eval a clean upper bound (same scale the head regressed). Linear => synapse-safe.

    `loss_fn` is the hook that selects which loss the drivers are read under: masked 2-way CE (the CL
    default) or plain 10-way CE (the standard regime, and the honest choice for an ER arm that actually
    trains on plain CE). Directly loss-proportional drivers rescale with it; ratio-form drivers barely
    move; ACh (entropy) is byte-identical because entropy does not depend on the loss function.
    """
    def __init__(self, drivers, standardize=True, loss_fn=None):
        self.drivers = drivers; K = len(drivers); self.standardize = standardize
        self.loss_fn = loss_fn or per_sample_masked_ce      # standard regime passes plain per-sample CE
        self.ef = self.es = self.esq = self.er = self.prev = self.emaH = None
        self.mh1 = None
        self.run_mean = torch.zeros(K, device=DEV); self.run_var = torch.ones(K, device=DEV)
        self.inited = False

    @torch.no_grad()
    def targets(self, net, x, y, update=True):
        logits, h1 = net.plain(x)
        ell = self.loss_fn(logits, y); Hs = entropy(logits)
        Lm = ell.mean().item()
        if self.ef is None:
            self.ef = self.es = Lm; self.esq = 0.0; self.er = -Lm; self.prev = Lm
            self.emaH = Hs.mean().item(); self.mh1 = h1.mean(0)
        std = ell.std() + EPS
        ach_vol = math.sqrt(max(self.esq, 0.0))
        da = (ell - self.es) / std
        cols = []
        for d in self.drivers:
            if d == "DA":        cols.append(da)
            elif d == "DA_step": cols.append((ell - self.prev) / std)
            elif d == "DA_fast": cols.append((ell - self.ef) / (abs(self.ef) + EPS))   # /ema_fast baseline
            elif d == "ACh":     cols.append(Hs)
            elif d == "ACh_ema": cols.append(torch.full_like(ell, self.emaH))          # lag-1 running entropy (scalar)
            elif d == "ACh_vol": cols.append(torch.full_like(ell, ach_vol))
            elif d == "ACh_vol_ps": cols.append((ell - self.ef).abs())                 # per-sample |loss - ema_fast|
            elif d == "NE":      cols.append(F.relu((da.abs() - ach_vol) / (ach_vol + EPS)))
            elif d == "NE_rise": cols.append(torch.full_like(ell, max(self.ef - self.es, 0.0)))
            elif d == "NE_emb":  cols.append((h1 - self.mh1).norm(dim=1))
            elif d == "5HT":     cols.append(-ell)
            elif d == "5HT_ema": cols.append(torch.full_like(ell, -self.es))           # ema_slow(-loss) (tonic scalar)
            else: raise ValueError(d)
        T = torch.stack(cols, 1)                    # (B,K) raw
        if update:
            self.ef += BF * (Lm - self.ef); self.es += BS * (Lm - self.es)
            self.esq += BS * ((Lm - self.ef) ** 2 - self.esq); self.er += BS * (-Lm - self.er)
            self.emaH += BS * (Hs.mean().item() - self.emaH)
            self.mh1 += BS * (h1.mean(0) - self.mh1); self.prev = Lm
            if self.standardize:
                bm = T.mean(0); bv = T.var(0, unbiased=False)
                if not self.inited:
                    self.run_mean = bm.clone(); self.run_var = bv.clone(); self.inited = True
                else:
                    self.run_mean = 0.99 * self.run_mean + 0.01 * bm
                    self.run_var = 0.99 * self.run_var + 0.01 * bv
        if not self.standardize:
            return T
        return (T - self.run_mean) / (self.run_var.sqrt() + EPS)   # standardized (running stats)


# ------------------------------- head-free novelty drivers -------------------------------
class NEDriver:
    """Computes a (B,K) NE driver directly (no head); gates all layers. emb_all=scalar h1-novelty;
    vec_h1=h1 diff (double fwd); vec_x=input diff (pre-forward); vecproj=random projection of the input diff.

    These are the only drivers that need no head at eval (a novelty signal is computable from the input
    or the forward itself), and the only ones well-conditioned in all four panels of the pt7 trace study
    (train/test x raw/standardized) because they depend on no frozen loss statistic.

    mean_mode: `ema` (recent-weighted) or `cumulative` (true running mean). EMA is REQUIRED for the
    embedding-space kinds — the cumulative mean lags the drifting representation and destabilises Adam
    (vec_h1 cumulative + Adam collapses to chance) — while the stationary input-space kinds are
    mean-mode-agnostic.
    """
    def __init__(self, kind, standardize, seed=0, mean_mode="ema"):
        self.kind = kind; self.standardize = standardize; self.mean_mode = mean_mode
        self.mh1 = None; self.mx = None; self.ch1 = 0; self.cx = 0     # counts for cumulative mean
        self.run_mean = None; self.run_var = None; self.inited = False
        g = torch.Generator().manual_seed(seed)
        if kind == "vecproj":
            self.R = (torch.randn(784, PROJ_DIM, generator=g) / math.sqrt(784)).to(DEV)
        elif kind == "vec_h1proj":
            self.R = (torch.randn(H1, PROJ_DIM, generator=g) / math.sqrt(H1)).to(DEV)

    def K(self):
        return {"emb_all": 1, "vec_h1": H1, "vec_x": 784,
                "vecproj": PROJ_DIM, "vec_h1proj": PROJ_DIM}[self.kind]

    def _upd_mean(self, m, cur, cattr):                          # ema (recent-weighted) or cumulative (true mean)
        if self.mean_mode == "ema":
            return m + BS * (cur.mean(0) - m)
        c = getattr(self, cattr) + cur.size(0); setattr(self, cattr, c)
        return m + (cur.sum(0) - cur.size(0) * m) / c            # incremental cumulative mean

    @torch.no_grad()
    def value(self, net, x, update=True):
        x2 = x.view(x.size(0), -1)
        if self.kind in ("vec_x", "vecproj"):                    # input-space novelty (pre-forward)
            if self.mx is None:
                self.mx = x2.mean(0).clone(); self.cx = x2.size(0)
            elif update:
                self.mx = self._upd_mean(self.mx, x2, "cx")
            diff = x2 - self.mx
            v = diff if self.kind == "vec_x" else diff @ self.R
        else:                                                    # h1 novelty (double forward)
            _, h1 = net.plain(x)
            if self.mh1 is None:
                self.mh1 = h1.mean(0).clone(); self.ch1 = h1.size(0)
            elif update:
                self.mh1 = self._upd_mean(self.mh1, h1, "ch1")
            diff = h1 - self.mh1
            if self.kind == "emb_all":
                v = diff.norm(dim=1, keepdim=True)
            elif self.kind == "vec_h1":
                v = diff
            else:                                                # vec_h1proj: downproject the 400-dim h1 diff
                v = diff @ self.R
        if update and self.standardize:
            bm = v.mean(0); bv = v.var(0, unbiased=False)
            if not self.inited:
                self.run_mean = bm.clone(); self.run_var = bv.clone(); self.inited = True
            else:
                self.run_mean = 0.99 * self.run_mean + 0.01 * bm
                self.run_var = 0.99 * self.run_var + 0.01 * bv
        if self.standardize and self.inited:
            return (v - self.run_mean) / (self.run_var.sqrt() + EPS)
        return v
