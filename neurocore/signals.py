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

The head-free NOVELTY drivers carry two further axes, both defaulting to the historical behaviour:
`norm` (the difference VECTOR, or just its magnitude) and `mean_mode` (what "normal" is measured
against: a streaming ema, an exact running mean, an exact dataset mean, or ema-at-train and
exact-at-inference). See NEDriver.
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import BF, BS, DEV, EPS, rng_frozen

DEFAULT_FEAT_DIM = 400        # penultimate width the novelty drivers see (constructor arg, not a global)
DEFAULT_IN_DIM = 784
PROJ_DIM = 32                 # random-projection width for vecproj / vec_h1proj

# Novelty kinds that emit a VECTOR (and so can be reduced to its norm); emb_all is already the norm.
VECTOR_KINDS = ("vec_h1", "vec_x", "vecproj", "vec_h1proj")
# Kinds that difference in INPUT space (pre-forward, no model needed); the rest difference h1.
INPUT_KINDS = ("vec_x", "vecproj")
# How the reference vector the driver differences against is obtained. See NEDriver.
MEAN_MODES = ("ema", "cumulative", "trueavg", "ema+trueavg")

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

    Every kind is `reduce(current - reference)`, so the driver has exactly two design axes beyond the
    space it differences in:

    NORM (`norm`, default False = the historical behaviour).
      False: the driver is the DIFFERENCE VECTOR itself, K = its dimensionality (784 for vec_x, 32 for
             the projected kinds). Each dimension gets its own gate coefficient, so the gate can react
             to WHERE a sample is unusual.
      True:  the driver is the scalar L2 NORM of that difference, K = 1 — HOW unusual, with the
             direction discarded. `emb_all` is by definition the norm of the vec_h1 difference, so
             `vec_h1` + norm=True reproduces it exactly and norm is a no-op on emb_all itself.
      The norm is taken BEFORE standardisation, which is the only correct order: standardising a K-dim
      vector per-dimension and THEN taking the norm concentrates it at sqrt(K) (a measured trap, see
      results/pt7_driver_traces.md), whereas norm-then-standardise z-scores a genuine scalar.
      It is also a CAPACITY axis, and a large one: the rank-K gate's projection is (K, gate_width), so
      norm=True takes vec_x's per-neuron P from 784x810 = 635k parameters (1.33x the Split-MNIST
      backbone, i.e. a capacity confound by neurocore.cost's guard) down to 810 (0.0017x).
      And it sidesteps a numerical trap the vector form has: 212 of vec_x's 784 input dimensions are
      MNIST border pixels with zero running variance, so standardising the vector divides by ~eps
      there (measured |m| up to 2e6 at test in pt5_taskil/plast_drivers.py) whereas the norm of the
      RAW difference has real variance. `norm=True` is the promoted form of the `VecNorm` wrapper
      that pt5_taskil/plast_drivers.py and position_paper/drivers.py each hand-rolled for exactly
      this reason; new code should use the flag.

    REFERENCE MEAN (`mean_mode`), i.e. what "normal" is measured against:
      `ema`         recent-weighted running mean, updated per batch (the default, and the historical
                    behaviour). REQUIRED for the embedding-space kinds: h1 drifts as the backbone
                    trains, and a lagging reference destabilises Adam (vec_h1 + cumulative + Adam
                    collapses to chance). The stationary input-space kinds are mean-mode-agnostic.
      `cumulative`  exact running arithmetic mean over every sample seen SO FAR (incremental, causal,
                    no lag parameter — but it is built up online, so early batches see a mean estimated
                    from very little and it can never un-see old data).
      `trueavg`     the exact arithmetic mean of a DATA SET, computed by a full pass and installed with
                    `set_true_mean` — no online estimation at all. At training that costs one pass over
                    the images before (or during) the run; at inference the test stream can be averaged
                    on the way through, which is why it is deployable rather than an oracle.
      `ema+trueavg` the ema at TRAINING and the true average at INFERENCE: the reference the gate was
                    trained under is the cheap streaming one, but the reference it is deployed under is
                    exact. This is the only mode in which the two phases disagree, and it is the reason
                    `value` has an explicit `inference` switch.

    `set_true_mean` must be called before a `trueavg` / `ema+trueavg` driver is read in the phase that
    wants the true mean; `neurocore.signals.dataset_mean` computes it RNG-neutrally. A missing true
    mean RAISES rather than silently falling back to the ema — a silent fallback would make the two
    modes indistinguishable in a ledger.
    """
    def __init__(self, kind, standardize, seed=0, mean_mode="ema", norm=False,
                 feat_dim=DEFAULT_FEAT_DIM, in_dim=DEFAULT_IN_DIM):
        if mean_mode not in MEAN_MODES:
            raise ValueError(f"unknown mean_mode {mean_mode!r}; known: {' | '.join(MEAN_MODES)}")
        self.kind = kind; self.standardize = standardize; self.mean_mode = mean_mode
        self.norm = bool(norm)
        self.feat_dim = feat_dim; self.in_dim = in_dim
        self.mh1 = None; self.mx = None; self.ch1 = 0; self.cx = 0     # counts for cumulative mean
        self.true_mean = None                                          # installed by set_true_mean
        self.run_mean = None; self.run_var = None; self.inited = False
        g = torch.Generator().manual_seed(seed)
        if kind == "vecproj":
            self.R = (torch.randn(in_dim, PROJ_DIM, generator=g) / math.sqrt(in_dim)).to(DEV)
        elif kind == "vec_h1proj":
            self.R = (torch.randn(feat_dim, PROJ_DIM, generator=g) / math.sqrt(feat_dim)).to(DEV)

    # ---- shape ----
    def K(self):
        if self.norm:
            return 1                                             # the L2 norm is one scalar per sample
        return {"emb_all": 1, "vec_h1": self.feat_dim, "vec_x": self.in_dim,
                "vecproj": PROJ_DIM, "vec_h1proj": PROJ_DIM}[self.kind]

    def space(self):
        """Which representation this kind differences in — what a true mean must be computed over."""
        return "x" if self.kind in INPUT_KINDS else "h1"

    def mean_dim(self):
        return self.in_dim if self.space() == "x" else self.feat_dim

    # ---- the reference mean ----
    def set_true_mean(self, mean):
        """Install the exact arithmetic mean this driver differences against (see `dataset_mean`).

        Call it again to swap the reference between phases — a `trueavg` CL run installs the mean of
        the training images seen so far at each task boundary, then the test-stream mean before eval.
        """
        v = torch.as_tensor(mean, dtype=torch.float32).to(DEV).reshape(-1)
        if v.numel() != self.mean_dim():
            raise ValueError(f"true mean for kind {self.kind!r} must have {self.mean_dim()} entries, "
                             f"got {v.numel()}")
        self.true_mean = v
        return self

    def uses_true_mean(self, inference):
        """Whether THIS read uses the installed true mean rather than the streaming one."""
        return (self.mean_mode == "trueavg"
                or (self.mean_mode == "ema+trueavg" and inference))

    def _reference(self, cur, mattr, cattr, update, inference):
        """The vector subtracted from `cur`, advancing the streaming mean only when it is in use."""
        if self.uses_true_mean(inference):
            if self.true_mean is None:
                raise RuntimeError(
                    f"mean_mode={self.mean_mode!r} needs an exact mean but none was installed. Call "
                    f"set_true_mean(dataset_mean(...)) for the {self.space()} space "
                    f"{'before inference' if self.mean_mode == 'ema+trueavg' else 'before use'}.")
            return self.true_mean.to(cur.device)              # no-op when already co-located
        m = getattr(self, mattr)
        if m is None:                                            # first read: seed from this batch
            m = cur.mean(0).clone(); setattr(self, cattr, cur.size(0))
        elif update:
            m = self._upd_mean(m, cur, cattr)
        setattr(self, mattr, m)
        return m

    def _upd_mean(self, m, cur, cattr):                          # ema (recent-weighted) or cumulative (true mean)
        if self.mean_mode == "cumulative":
            c = getattr(self, cattr) + cur.size(0); setattr(self, cattr, c)
            return m + (cur.sum(0) - cur.size(0) * m) / c        # incremental cumulative mean
        return m + BS * (cur.mean(0) - m)                        # ema (also the train phase of ema+trueavg)

    # ---- read-out ----
    @torch.no_grad()
    def value(self, net, x, update=True, inference=None):
        """(B, K) driver value.

        `update` advances the streaming mean and the standardisation statistics, as before.
        `inference` selects the reference mean for `ema+trueavg` and DEFAULTS TO `not update`, since
        `update=False` has meant "test pass" everywhere in this project. That default is wrong in one
        place: a read-out taken DURING training with `update=False` (a meta-loss batch, a diagnostic)
        would be treated as inference and switch reference mid-training — pass `inference=False`
        explicitly there. It is inert for every other mean_mode.
        """
        inference = (not update) if inference is None else bool(inference)
        x2 = x.view(x.size(0), -1)
        if self.kind in INPUT_KINDS:                             # input-space novelty (pre-forward)
            diff = x2 - self._reference(x2, "mx", "cx", update, inference)
            v = diff if self.kind == "vec_x" else diff @ self.R
        else:                                                    # h1 novelty (double forward)
            _, h1 = net.plain(x)
            diff = h1 - self._reference(h1, "mh1", "ch1", update, inference)
            if self.kind == "emb_all":
                v = diff.norm(dim=1, keepdim=True)
            elif self.kind == "vec_h1":
                v = diff
            else:                                                # vec_h1proj: downproject the 400-dim h1 diff
                v = diff @ self.R
        if self.norm and v.size(1) > 1:                          # norm BEFORE standardising (see class doc)
            v = v.norm(dim=1, keepdim=True)
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


def _loader_list(loaders):
    """Normalise "one loader or several" without eating a batch by mistake.

    A loader yields BATCHES `(x, y)`; a list of loaders yields loaders. Passing one loader where a
    list was expected would otherwise iterate its batches as if each were a loader and sum the LABELS
    alongside the images — a silent factor-of-two in the mean. Disambiguated on whether the first
    item is itself a batch (a sequence whose first element is a tensor).
    """
    if not isinstance(loaders, (list, tuple)):
        return [loaders]
    first = loaders[0] if loaders else None
    if isinstance(first, (list, tuple)) and first and torch.is_tensor(first[0]):
        return [loaders]                                    # a bare sequence of (x, y) batches
    return list(loaders)


@torch.no_grad()
def dataset_mean(loaders, net=None, space="x", device=DEV, rng_neutral=True):
    """Exact arithmetic mean over every sample in `loaders` — one pass, no EMA, no lag.

    `loaders` is a DataLoader or a list of them (a CL run passes the task loaders it is allowed to
    average over). `space="h1"` needs `net` and means the mean is taken over the PENULTIMATE
    representation, which drifts as the backbone trains — so an h1 true mean is exact only w.r.t. the
    weights at the moment it was computed and has to be recomputed when they move. Input-space means
    are stationary and can be computed once.

    RNG-NEUTRAL BY DEFAULT: iterating a DataLoader draws a base seed from the global torch generator
    even with shuffle=False, so this pass would otherwise shift a training run's data order and move
    it off its reference trajectory. `rng_neutral=False` only if the caller is already inside a guard.
    """
    loaders = _loader_list(loaders)
    if space not in ("x", "h1"):
        raise ValueError(f"space must be 'x' or 'h1', got {space!r}")
    if space == "h1" and net is None:
        raise ValueError("dataset_mean(space='h1') needs the net whose h1 is being averaged")
    with rng_frozen() if rng_neutral else contextlib.nullcontext():
        total, n = None, 0
        for ld in loaders:
            for batch in ld:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device).view(x.size(0), -1)
                v = x if space == "x" else net.plain(x)[1]
                total = v.sum(0) if total is None else total + v.sum(0)
                n += v.size(0)
    if not n:
        raise ValueError("dataset_mean: no samples")
    return total / n
