"""LIVE DRIVER TRACES — every neuromodulator formula under a NOTHING-IS-FROZEN inference protocol.

Successor to the frozen `results/pt7_driver_traces*` study. Same observer-only idea (a PLAIN net trains,
every driver is a passive read-out, nothing the observer computes touches the loss / the parameters / the
RNG), but the eval protocol is inverted and three predictor-based drivers are added.

WHAT CHANGES vs the frozen study
--------------------------------
1. NOTHING IS FROZEN AT INFERENCE. The frozen study ran its test pass with `update=False`, so every EMA
   (ema_fast/slow, ema_H, mean_h1, mean_x) and every running standardisation stat stopped at the last
   training batch. Here the test pass runs with `update=True`: all state keeps advancing on the test
   stream. Consequence: the TONIC drivers (ACh_ema, ACh_vol, NE_rise, 5HT_ema) are no longer flat lines
   at a stale constant — they track the test stream across the five task blocks. (`--freeze-at-test`
   reproduces the old protocol for an A/B; it is a test-side flag, so `--retest` gets it in seconds.)

2. ACh_ema IS THE ACTUAL RUNNING EMA, NOT A PREDICTION, and so is ACh. Both are functions of the
   predictive entropy H(logits), which needs NO LABELS — one extra unmodulated forward and they are
   exactly computable at inference. This study therefore always pays that extra forward rather than
   regressing them with a head. Same for NE_emb / nerisez / the vec_* novelty drivers.

3. THE LOSS-DEPENDENT DRIVERS GET A THIRD PANEL: "as if at true inference". A driver containing the
   per-sample loss ell_i cannot be computed at inference at any price — ell_i needs the label. Those
   nine (DA, DA_step, DA_fast, ACh_vol, ACh_vol_ps, NE, NE_rise, 5HT, 5HT_ema) are what pt7's head
   m_k(x) exists to regress, so for them the test figure carries a third column showing the HEAD's
   output — what the deployed mechanism would actually feed its gate. Two heads are trained (784->32->12,
   pt7's `Heads`, Adam, MSE), one against the raw targets and one against the standardised targets, so
   the raw and standardised predicted panels are each a faithful "what the deployed head emits".
   The label-free drivers get no such column BY CONSTRUCTION: their true-inference value IS the actual
   value plotted in column 2.

4. THREE PREDICTOR-BASED DRIVERS ARE TRACED (`results/pt7_stateful.py`'s mechanisms):
     nerisez_mlp  relu((H_pred - ema_H)/sqrt(var_H))   H from an MLP  784->32->1
     nerisez_gru  same formula                          H from a stateful GRU (hidden persists, detached)
     ACh_gru      standardised H_pred                   H from the same GRU
   Their ema_H/var_H are updated from the ACTUAL entropy (this study's premise: an extra forward is
   allowed), exactly as pt7_stateful's TRAINING loop does — pt7_stateful's eval-`running` mode instead
   fed them predicted H because it assumed no extra forward. `nerisez_mlp` is included as the MLP
   counterpart that makes the GRU panel readable; `nerisez` (actual H) is the no-predictor reference,
   and `ACh` is `ACh_gru`'s no-predictor reference.

EVERY DRIVER GETS raw x std x {training, test} panels (plus the third test column where it applies).

ANCHOR: the observer is RNG-neutral — the pt7 modules it builds are constructed inside `rng_frozen()`,
which snapshots and restores torch/numpy/random state, so the main net's trajectory is bit-identical to
a plain run and both arms must land on the frozen ledger. References for the default operating point
(tuned Adam, --driver-loss arm, --shuffle-test): naive 0.5514, er 0.8988. That agreement is the sanity
check; if it moves, the observer has started perturbing training.

REGIMES: naive (masked per-sample CE, current task only, no buffer) and er (plain 10-way CE over
current + reservoir-1000 replay) — pt7's two class-IL baselines.

Outputs: live_traces.npz, figs/*.png, live_traces.log.
Usage:
  uv run python driver_traces/live_traces.py                          # train both arms + plot
  uv run python driver_traces/live_traces.py --plot-only              # re-plot from the npz
  uv run python driver_traces/live_traces.py --retest live            # test pass only (seconds)
  uv run python driver_traces/live_traces.py --retest live --freeze-at-test --suffix _frozen
"""
import argparse
import contextlib
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "results"))
import pt7_neuromodulators as p7                                     # noqa: E402
import pt7_stateful as ps                                           # noqa: E402
import pt7_variants as pv                                           # noqa: E402
from torch.utils.data import DataLoader, Subset                     # noqa: E402

sys.path.insert(0, str(REPO / "prototype"))
from configs import TUNED_MAIN                                      # noqa: E402
from data import SplitMNIST                                         # noqa: E402

DEV, EPS, BS = p7.DEV, p7.EPS, p7.BS
CE = nn.CrossEntropyLoss()

HEAD_DRIVERS = ["DA", "DA_step", "DA_fast", "ACh", "ACh_ema", "ACh_vol", "ACh_vol_ps",
                "NE", "NE_rise", "NE_emb", "5HT", "5HT_ema"]
NE_KINDS = ["vec_h1", "vec_h1proj", "vec_x", "vecproj"]             # emb_all == NE_emb, not duplicated
PRED_DRIVERS = ["nerisez_mlp", "nerisez_gru", "ACh_gru"]            # H from a learned predictor
ALL_DRIVERS = HEAD_DRIVERS + ["nerisez"] + NE_KINDS + PRED_DRIVERS

# Drivers containing the per-sample loss ell_i. No extra forward can supply ell_i at inference (it needs
# the LABEL), so these — and only these — get the head-predicted "true inference" column. Everything else
# is a function of x / logits / activations and is exactly computable with one extra unmodulated forward.
LOSS_DEPENDENT = ["DA", "DA_step", "DA_fast", "ACh_vol", "ACh_vol_ps", "NE", "NE_rise", "5HT", "5HT_ema"]
PRED_SUFFIX = "@pred"

# label -> (formula shown on the panel, neuromodulator family, dimensionality)
META = {
    "DA":          (r"$(\ell_i-\mathrm{ema}_{slow})/\mathrm{std}(\ell)$", "DA", 1),
    "DA_step":     (r"$(\ell_i-L_{t-1})/\mathrm{std}(\ell)$", "DA", 1),
    "DA_fast":     (r"$(\ell_i-\mathrm{ema}_{fast})/|\mathrm{ema}_{fast}|$", "DA", 1),
    "ACh":         (r"$H(\mathrm{logits}_i)$", "ACh", 1),
    "ACh_ema":     (r"$\mathrm{ema}(H)$  (tonic)", "ACh", 1),
    "ACh_vol":     (r"$\sqrt{\mathrm{ema}((L-\mathrm{ema}_{fast})^2)}$  (tonic)", "ACh", 1),
    "ACh_vol_ps":  (r"$|\ell_i-\mathrm{ema}_{fast}|$", "ACh", 1),
    "NE":          (r"$\mathrm{relu}((|DA_i|-\mathrm{ACh}_{vol})/\mathrm{ACh}_{vol})$", "NE", 1),
    "NE_rise":     (r"$\max(\mathrm{ema}_{fast}-\mathrm{ema}_{slow},0)$  (tonic)", "NE", 1),
    "NE_emb":      (r"$\|h_1^{(i)}-\overline{h_1}\|$", "NE", 1),
    "nerisez":     (r"$\mathrm{relu}((H_i-\mathrm{ema}_H)/\sqrt{\mathrm{var}_H})$   (ACTUAL $H$)", "NE", 1),
    "nerisez_mlp": (r"$\mathrm{relu}((\hat{H}_i-\mathrm{ema}_H)/\sqrt{\mathrm{var}_H})$   ($\hat{H}$: MLP)",
                    "NE", 1),
    "nerisez_gru": (r"$\mathrm{relu}((\hat{H}_i-\mathrm{ema}_H)/\sqrt{\mathrm{var}_H})$   ($\hat{H}$: GRU)",
                    "NE", 1),
    "ACh_gru":     (r"$\mathrm{std}(\hat{H}_i)$   ($\hat{H}$: GRU)", "ACh", 1),
    "5HT":         (r"$-\ell_i$", "5-HT", 1),
    "5HT_ema":     (r"$-\mathrm{ema}_{slow}$  (tonic)", "5-HT", 1),
    "vec_h1":      (r"$\|h_1^{(i)}-\overline{h_1}\|$  (400-d)", "NE", 400),
    "vec_h1proj":  (r"$\|R(h_1^{(i)}-\overline{h_1})\|$  (32-d)", "NE", 32),
    "vec_x":       (r"$\|x_i-\overline{x}\|$  (784-d)", "NE", 784),
    "vecproj":     (r"$\|R(x_i-\overline{x})\|$  (32-d)", "NE", 32),
}

NPZ = HERE / "live_traces.npz"
FIGDIR = HERE / "figs"

# Sanity anchor. The observer is RNG-neutral, so these must reproduce the frozen study's own run at the
# same operating point (results/pt7_driver_traces_armloss_shuftest.log), itself within the ~0.007-0.016
# MPS 1-seed noise floor of the pt7 ledger (naive 0.5545 / er 0.8975).
REF = {("tuned", "naive", "adam"): 0.5514, ("tuned", "er", "adam"): 0.8988}
UNTUNED = dict(lr=1e-3, epochs_per_task=5)


def hparams(point, arm, opt_kind):
    """(lr, epochs) for an arm. Tuned values come from configs.TUNED_MAIN (rule #7: no hardcoded
    hyperparameters); a missing key is a hard error meaning "that combination was never swept"."""
    if point == "untuned":
        return UNTUNED["lr"], UNTUNED["epochs_per_task"]
    try:
        c = TUNED_MAIN[("classil", arm, opt_kind)]
    except KeyError:
        raise SystemExit(f"no tuned entry for (classil, {arm}, {opt_kind}) — tune it first, "
                         f"or pass --point untuned")
    return c["lr"], c["epochs_per_task"]


@contextlib.contextmanager
def rng_frozen():
    """Snapshot/restore every RNG stream around observer construction. The frozen study was RNG-neutral
    for free (it built no parameters); this one builds two heads and three predictors, and initialising
    them would consume torch RNG and shift the DataLoader's shuffling — silently moving the main net off
    its reference trajectory and breaking the anchor. Restoring makes the observer neutral again."""
    t, n, r = torch.get_rng_state(), np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(t); np.random.set_state(n); random.setstate(r)


# ------------------------------- standardisation (pt7's, reproduced once) -------------------------------
class Standardizer:
    """Running per-dim standardisation, identical to Signals/NEDriver: update 0.99/0.01 (init from the
    first batch), then (v - run_mean)/(sqrt(run_var) + eps). Unlike the frozen study this keeps updating
    at test unless --freeze-at-test is passed."""

    def __init__(self):
        self.rm = self.rv = None

    def __call__(self, v, update=True):
        if update:
            bm, bv = v.mean(0), v.var(0, unbiased=False)
            if self.rm is None:
                self.rm, self.rv = bm.clone(), bv.clone()
            else:
                self.rm = 0.99 * self.rm + 0.01 * bm
                self.rv = 0.99 * self.rv + 0.01 * bv
        if self.rm is None:
            return v
        return (v - self.rm) / (self.rv.sqrt() + EPS)


def _scalarise(v):
    """(B,K) -> per-sample scalar: the value itself if K==1 (sign kept), else the L2 norm."""
    return v.squeeze(1) if v.size(1) == 1 else v.norm(dim=1)


# ------------------------------- the observer -------------------------------
class LiveObserver:
    """Evaluates every driver on a batch, and owns the learned parts (two heads + three H-predictors).

    Read-only w.r.t. the main net: no grad flows into it, and its own modules train from detached
    targets with their own optimizers. Construction is wrapped in rng_frozen() by the caller."""

    def __init__(self, lr, loss_fn=None, proj_seed=0):
        with rng_frozen():
            self.sig = p7.Signals(HEAD_DRIVERS, standardize=False, loss_fn=loss_fn)
            self.ne = {k: pv.NEDriver(k, standardize=False, seed=proj_seed) for k in NE_KINDS}
            # pt7_stateful's predictors: ("nerisez", gru?) and ("ach", gru)
            self.pred = {"nerisez_mlp": ps.StatefulDriver("nerisez", gru=False).to(DEV),
                         "nerisez_gru": ps.StatefulDriver("nerisez", gru=True).to(DEV),
                         "ACh_gru": ps.StatefulDriver("ach", gru=True).to(DEV)}
            # one head per basis: the deployed head regresses whatever targets pt7 hands it, and pt7
            # standardises by default — training both makes the raw AND standardised predicted panels
            # each a genuine "what the deployed head emits", not one derived from the other.
            self.head = {b: p7.Heads(len(HEAD_DRIVERS)).to(DEV) for b in ("raw", "std")}
        self.pred_opt = {k: torch.optim.Adam(v.parameters(), lr) for k, v in self.pred.items()}
        self.head_opt = {b: torch.optim.Adam(v.parameters(), lr) for b, v in self.head.items()}
        self.emaH = self.varH = None                                # nerisez stats (ACTUAL entropy)
        self.std = {k: Standardizer() for k in ALL_DRIVERS}

    # ---- read-out ----
    @torch.no_grad()
    def __call__(self, net, X, Y, update=True):
        """Returns (traces, targets) where traces[k] = (raw_scalar, std_scalar) per sample and targets is
        {"raw": (B,12), "std": (B,12)} — the head-training targets for this batch."""
        vals = {}
        T = self.sig.targets(net, X, Y, update=update)               # (B,12) raw, loss-based state advances
        for i, k in enumerate(HEAD_DRIVERS):
            vals[k] = T[:, i:i + 1]
        for k, d in self.ne.items():
            vals[k] = d.value(net, X, update=update)
        Hact = p7.entropy(net.plain(X)[0]).unsqueeze(1)              # the extra forward, label-free
        vals["nerisez"] = self._nerisez(Hact, update)
        for k, d in self.pred.items():
            # ach standardises its own predicted H (update_stats); nerisez z-scores against ACTUAL-H
            # stats, which we advance below from Hact rather than from H_pred (pt7_stateful's train-loop
            # convention, valid here because this study allows the extra forward at inference too).
            vals[k] = d.driver(X, update_state=update, update_stats=(update and d.mech == "ach"))
            if update and d.mech == "nerisez":
                d.upd_actual(Hact)
        self._Hact = Hact

        out, std_cols = {}, {}
        for k, v in vals.items():
            s = self.std[k](v, update=update)                        # update-then-transform, as pt7 does
            out[k] = (_scalarise(v), _scalarise(s))
            std_cols[k] = s
        tgt = {"raw": T, "std": torch.cat([std_cols[k] for k in HEAD_DRIVERS], dim=1)}
        return out, tgt

    def _nerisez(self, H, update):
        """relu((H - ema_H)/sqrt(var_H)) on the ACTUAL entropy — the no-predictor reference for
        nerisez_mlp / nerisez_gru."""
        if self.emaH is None:
            self.emaH = H.mean().item(); self.varH = H.var(unbiased=False).item()
            return F.relu(H - H.mean()) / (H.std() + EPS)            # bootstrap (batch z-score), as pt7
        v = F.relu((H - self.emaH) / math.sqrt(self.varH + EPS))
        if update:                                                   # var uses the OLD mean, then the mean
            self.varH = (1 - BS) * self.varH + BS * ((H - self.emaH) ** 2).mean().item()
            self.emaH = (1 - BS) * self.emaH + BS * H.mean().item()
        return v

    # ---- the observer's own learning (heads + H-predictors) ----
    def learn(self, X, tgt):
        """One MSE step for each head and each H-predictor. Mirrors pt7's `F.mse_loss(heads(Xm), T)` and
        pt7_stateful's `F.mse_loss(drv.predictH(Xm, update_state=False), Hact)`. All targets are detached
        (they come out of the no-grad read-out), so nothing here can reach the main net."""
        for b, h in self.head.items():
            loss = F.mse_loss(h(X), tgt[b])
            self.head_opt[b].zero_grad(); loss.backward(); self.head_opt[b].step()
        for k, d in self.pred.items():
            loss = F.mse_loss(d.predictH(X, update_state=False), self._Hact)
            self.pred_opt[k].zero_grad(); loss.backward(); self.pred_opt[k].step()

    @torch.no_grad()
    def head_out(self, X):
        """What the DEPLOYED mechanism would feed its gate: {basis: (B,12)} from x alone, no labels."""
        return {b: h(X) for b, h in self.head.items()}


class Trace:
    """Accumulates (batch mean, batch sd) of each series' per-sample scalar, raw and standardised.
    Series not present in a given record simply stay empty (the predicted series exist only at test)."""

    def __init__(self, keys):
        self.d = {(k, b, s): [] for k in keys for b in ("raw", "std") for s in ("mean", "sd")}
        self.bounds = []
        self._n = 0

    def add(self, rec):
        for k, (raw, std) in rec.items():
            for b, v in (("raw", raw), ("std", std)):
                self.d[(k, b, "mean")].append(float(v.mean()))
                self.d[(k, b, "sd")].append(float(v.std()) if v.numel() > 1 else 0.0)
        self._n += 1

    def mark(self):
        self.bounds.append(self._n)


TRACE_KEYS = ALL_DRIVERS + [k + PRED_SUFFIX for k in LOSS_DEPENDENT]


# ------------------------------- run -------------------------------
def test_loaders(ds, batch_size=64, shuffle=True, seed=1234):
    """Test loaders, task 0->4. `shuffle` randomises order WITHIN each task while keeping the task blocks
    in sequence. The MNIST test file is ordered (mean |x| rises ~25->28 with file index, rho~+0.64), and
    under this study's live-update protocol that ramp would feed straight into the running stats, so
    shuffling is the DEFAULT here (the frozen study made it opt-in). Order cannot change accuracy."""
    out = []
    for t in range(5):
        cls = set(p7.SEQ[t])
        idx = [i for i, lab in enumerate(ds._test_ds.targets.tolist()) if lab in cls]
        g = torch.Generator().manual_seed(seed + t) if shuffle else None
        out.append(DataLoader(Subset(ds._test_ds, idx), batch_size=batch_size,
                              shuffle=shuffle, generator=g))
    return out


def evaluate(net, obs, ds, shuffle_test=True, freeze=False):
    """Test pass. `freeze=False` (the point of this study) keeps every EMA and running stat advancing on
    the test stream; `freeze=True` reproduces the frozen study's protocol for an A/B. Also records, for
    each loss-dependent driver, the HEAD's output — the value the deployed mechanism would actually use,
    since ell_i is unavailable without labels."""
    net.eval()
    te = Trace(TRACE_KEYS)
    c = tot = 0
    with torch.no_grad():
        for i, tl in enumerate(test_loaders(ds, shuffle=shuffle_test)):
            for x, y in tl:
                x, y = x.to(DEV), y.to(DEV)
                Xm = x.view(x.size(0), -1)
                c += (net.plain(Xm)[0].argmax(1) == y).sum().item(); tot += len(y)
                rec, _ = obs(net, Xm, y, update=not freeze)
                hp = obs.head_out(Xm)
                for k in LOSS_DEPENDENT:
                    j = HEAD_DRIVERS.index(k)
                    rec[k + PRED_SUFFIX] = (hp["raw"][:, j], hp["std"][:, j])
                te.add(rec)
            te.mark()
    return te, c / tot


def save_ckpt(path, net, obs, arm, lr, ep, driver_loss):
    """Checkpoint the trained net + the whole observer (frozen stats AND the trained heads/predictors —
    both halves are needed for a test pass, so they travel together). Lets a test-side change cost
    seconds instead of a full retrain."""
    torch.save({"net": net.state_dict(), "obs": obs, "arm": arm,
                "lr": lr, "epochs": ep, "driver_loss": driver_loss}, path)


def run(arm, opt_kind="adam", lr=3e-4, epochs=5, buffer=1000, seed=42, driver_loss="arm",
        shuffle_test=True, freeze=False, ckpt=None, log=print):
    """Train a PLAIN net under `arm` while tracing every driver and training the observer's own heads and
    H-predictors alongside. Returns (train_trace, test_trace, acc).

    driver_loss: "arm" (default here) = the loss inside DA/5HT/... follows the arm's own training loss
    (plain CE under er); "pt7" = masked CE in both arms, as pt7's Signals does even in er-own.
    """
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=64) for t in range(5)]
    net = p7.Net().to(DEV)
    opt = p7._opt(opt_kind, net.parameters(), lr)
    buf = p7.Reservoir(buffer)
    obs = LiveObserver(lr, loss_fn=(pv.per_sample_ce_plain if (driver_loss == "arm" and arm == "er")
                                    else None))
    tr = Trace(TRACE_KEYS)

    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if arm == "naive":                                  # masked CE, current task only
                    Xm, Ym = x.view(x.size(0), -1), y
                    loss = p7.masked_ce(net.plain(Xm)[0], Ym)
                else:                                               # er: plain CE over current + replay
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                    loss = CE(net.plain(Xm)[0], Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                rec, tgt = obs(net, Xm, Ym, update=True)             # observe AFTER the step, as pt7 does
                tr.add(rec)
                obs.learn(Xm, tgt)                                  # heads + predictors (detached targets)
                if arm == "er":
                    buf.add(x, y)
        tr.mark()
        log(f"    [{arm}] task {t} done ({tr.bounds[-1]} steps)")

    if ckpt is not None:
        save_ckpt(ckpt, net, obs, arm, lr, epochs, driver_loss)
        log(f"    [{arm}] checkpoint -> {ckpt.name}")
    te, acc = evaluate(net, obs, ds, shuffle_test=shuffle_test, freeze=freeze)
    return tr, te, acc


# ------------------------------- plotting -------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
COLOR = {"naive": "#2a78d6", "er": "#eb6834"}                        # categorical slots 1 & 2
LABEL = {"naive": "naive + masked loss", "er": "ER (plain CE)"}
COL_TITLE = {"train": "training", "test": "test — ACTUAL value",
             "testpred": "test — HEAD-PREDICTED"}


def _smooth(v, w):
    """Centred moving average with edge padding (mode='same' would droop at both ends)."""
    if w <= 1 or len(v) < w:
        return v
    pad = w // 2
    return np.convolve(np.pad(v, (pad, w - 1 - pad), mode="edge"), np.ones(w) / w, mode="valid")


# Panels whose values are SMALL in absolute terms but span many decades, so the generic rule below (which
# also requires max > 10) leaves them linear and unreadable.
FORCE_SYMLOG = {("NE_rise", "raw", "train"): 5e-4}

# Panels where the moving average is itself MISLEADING and the raw trace is the honest view: a mean over
# a sparse spike train smears each spike into a plateau one window wide at 1/w of its height.
NO_SMOOTH = {("NE_rise", "raw", "train")}

# Panels whose faint per-step trace spikes far above the smoothed signal, squashing it against the axis.
FIT_SMOOTHED = {("nerisez", "raw", "train"), ("nerisez", "std", "train"),
                ("nerisez_mlp", "raw", "train"), ("nerisez_mlp", "std", "train"),
                ("nerisez_gru", "raw", "train"), ("nerisez_gru", "std", "train")}


def _pick_scale(ax, series, key=None, basis=None, phase=None):
    """Linear unless the dynamic range is extreme (the tonic drivers blow up once standardised), in which
    case symlog around the typical magnitude so both regimes stay legible in one panel."""
    a = np.concatenate([np.abs(s[np.isfinite(s)]) for s in series if len(s)])
    if not len(a):
        return False
    forced = FORCE_SYMLOG.get((key, basis, phase))
    if forced is not None:
        ax.set_yscale("symlog", linthresh=forced)
        return True
    typ = np.median(a[a > 0]) if (a > 0).any() else 0.0
    if typ > 0 and a.max() > 50 * typ and a.max() > 10:
        ax.set_yscale("symlog", linthresh=max(typ, 1e-6))
        return True
    return False


def _fit_to_smoothed(ax, key, basis, phase, used, smoothed):
    if (key, basis, phase) not in FIT_SMOOTHED or not smoothed:
        return False
    s = np.concatenate([x[np.isfinite(x)] for x in smoothed if len(x)])
    r = np.concatenate([x[np.isfinite(x)] for x in used if len(x)])
    if not len(s):
        return False
    lo, hi = float(s.min()), float(s.max())
    pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)
    return bool(len(r) and (r.max() > hi or r.min() < lo))     # only flag if the raw trace really clips


def _panel(ax, data, key, basis, phase, title):
    """`phase` is the COLUMN ('train'|'test'|'testpred'); the npz phase and series key are derived from
    it, since the predicted series live under the test phase with a '@pred' key suffix."""
    npz_phase = "test" if phase == "testpred" else phase
    series_key = key + PRED_SUFFIX if phase == "testpred" else key
    used, smoothed = [], []
    for arm in ("naive", "er"):
        mk = f"{arm}|{npz_phase}|{series_key}|{basis}|mean"
        if mk not in data:
            continue
        m = data[mk]
        if not len(m):
            continue
        if (series_key, basis, phase) in NO_SMOOTH:
            ax.plot(m, color=COLOR[arm], lw=0.9, alpha=0.85, label=LABEL[arm], zorder=3)
            used.append(m); smoothed.append(m)
        else:
            w = max(1, len(m) // 180)
            s = _smooth(m, w)
            ax.plot(m, color=COLOR[arm], lw=0.7, alpha=0.16, zorder=1)
            ax.plot(s, color=COLOR[arm], lw=1.6, label=LABEL[arm], zorder=3)
            used.append(m); smoothed.append(s)
        for b in data[f"{arm}|{npz_phase}|bounds"][:-1]:
            ax.axvline(b, color=GRID, lw=0.8, ls="--", zorder=0)
    sym = _pick_scale(ax, used, series_key, basis, phase)
    clipped = _fit_to_smoothed(ax, series_key, basis, phase, used, smoothed)
    ax.set_title(title + (" — symlog" if sym else "") + (" — y fit to smoothed" if clipped else ""),
                 fontsize=8.5, color=INK2, pad=4, loc="left")
    ax.set_xlabel("training step" if phase == "train" else "test batch (tasks 0→4)",
                  fontsize=7.5, color=INK2)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=7, colors=INK2, length=3)


def _setup_line(data):
    """Human-readable operating point, taken from the run itself (per arm, since a tuned point may differ)."""
    opt, point = str(data["opt"]), str(data["point"])
    hp = {a: (float(data[f"{a}|lr"]), int(data[f"{a}|epochs"])) for a in ("naive", "er")}
    same = hp["naive"] == hp["er"]
    hs = (f"lr {hp['naive'][0]:g} · {hp['naive'][1]} ep/task" if same else
          " · ".join(f"{a}: lr {l:g}/{e} ep" for a, (l, e) in hp.items()))
    frozen = bool(data["freeze_at_test"]) if "freeze_at_test" in data else False
    proto = "stats FROZEN at test (legacy protocol)" if frozen else "nothing frozen at test (live stats)"
    return (f"plain net, NO modulation applied · {proto} · {opt.upper()} ({point}) · {hs} · "
            f"buffer 1000 · seed 42")


def _wrap(head, width_in):
    """Greedy wrap of a ' · '-joined header to the figure width. Splitting the parts in half (what the
    frozen study did) overflows as soon as the parts are uneven — the budget has to be in CHARACTERS.
    ~18.5 chars/inch at 7.5pt with the default sans face, measured against the rendered figures."""
    budget = max(40, int(18.5 * width_in))
    lines, cur = [], ""
    for p in head.split(" · "):
        cand = p if not cur else f"{cur} · {p}"
        if len(cand) > budget and cur:
            lines.append(cur); cur = p
        else:
            cur = cand
    lines.append(cur)
    return "\n".join(lines)


def plot_all(data, panels="all"):
    """One figure per driver: rows {raw, standardised} x columns {training, test, [test-predicted]}.
    The third column exists only for the loss-dependent drivers — for every other driver the
    true-inference value IS the actual value already shown in column 2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup = _setup_line(data)
    FIGDIR.mkdir(exist_ok=True)
    for key in ALL_DRIVERS:
        formula, fam, dim = META[key]
        phases = ["train", "test"] if panels == "all" else ["test"]
        if key in LOSS_DEPENDENT:
            phases.append("testpred")
        width = 4.8 * len(phases)
        fig, axes = plt.subplots(2, len(phases), figsize=(width, 6.0),
                                 facecolor=SURFACE, squeeze=False)
        note = "batch-mean of the per-sample value" if dim == 1 else \
               f"batch-mean of the per-sample L2 norm ({dim}-d driver)"
        if key in LOSS_DEPENDENT:
            note += " · needs the LABEL → head-regressed at true inference"
        elif key in PRED_DRIVERS:
            note += " · H from a learned predictor; label-free"
        else:
            note += " · label-free → exactly computable at inference (one extra forward)"
        # tall LaTeX (norms, fractions) grows the title box downward, so keep a wide gap to the subtitle
        fig.suptitle(f"{key}   ({fam})      {formula}", fontsize=12,
                     color=INK, x=0.012, ha="left", y=0.988, va="top")
        fig.text(0.012, 0.918, _wrap(f"{note} · {setup}", width),
                 fontsize=7.5, color=INK2, ha="left", va="top")
        for r, basis in enumerate(("raw", "std")):
            for c, phase in enumerate(phases):
                ax = axes[r][c]
                ax.set_facecolor(SURFACE)
                lab = "non-standardised" if basis == "raw" else "standardised (running stats)"
                _panel(ax, data, key, basis, phase, f"{lab} — {COL_TITLE[phase]}")
                if c == 0:
                    ax.set_ylabel(lab.split(" (")[0], fontsize=8, color=INK2)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", frameon=False, fontsize=8.5,
                   labelcolor=INK2, ncol=2, bbox_to_anchor=(0.99, 0.995))
        fig.tight_layout(rect=[0, 0, 1, 0.895])
        fig.savefig(FIGDIR / f"{key}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    n_sheets = 0
    for cphase in (("train", "test") if panels == "all" else ("test",)):
        n_sheets += _contact_sheet(plt, data, setup, ALL_DRIVERS, cphase, f"_contact_sheet_{cphase}_raw")
    # a third sheet: actual vs head-predicted at test, for the nine loss-dependent drivers
    n_sheets += _contact_sheet(plt, data, setup, LOSS_DEPENDENT, "testpred",
                               "_contact_sheet_test_predicted")
    print(f"  wrote {len(ALL_DRIVERS) + n_sheets} figures to {FIGDIR}")


def _contact_sheet(plt, data, setup, keys, cphase, name):
    n = len(keys)
    cols, rows = 4, math.ceil(n / 4)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.5 * rows), facecolor=SURFACE, squeeze=False)
    fig.suptitle(f"neuromodulator drivers — non-standardised value, {COL_TITLE[cphase]}",
                 fontsize=14, color=INK, x=0.008, ha="left", y=0.996, va="top")
    fig.text(0.008, 0.972, setup, fontsize=9, color=INK2, ha="left", va="top")
    flat = axes.ravel()
    for i, ax in enumerate(flat):
        ax.set_facecolor(SURFACE)
        if i >= n:
            ax.axis("off"); continue
        _panel(ax, data, keys[i], "raw", cphase, keys[i])
    h, l = flat[0].get_legend_handles_labels()
    (flat[n] if n < len(flat) else fig).legend(h, l, loc="center", frameon=False, fontsize=12,
                                               labelcolor=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.958])
    fig.savefig(FIGDIR / f"{name}.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return 1


# ------------------------------- numeric summary -------------------------------
def summarise(data, log=print):
    """Per-driver range table. `spread` = max/median of |value|: a large spread on the standardised row
    is the tonic-driver blow-up signature (standardising a near-constant divides by ~0)."""
    log("\n  driver         regime phase     basis        mean        sd         min         max      spread")
    log("  " + "-" * 100)
    for k in TRACE_KEYS:
        for arm in ("naive", "er"):
            for phase in ("train", "test"):
                for basis in ("raw", "std"):
                    kk = f"{arm}|{phase}|{k}|{basis}|mean"
                    if kk not in data or not len(data[kk]):
                        continue
                    v = data[kk]
                    a = np.abs(v[np.isfinite(v)])
                    med = np.median(a[a > 0]) if (a > 0).any() else 0.0
                    spread = (a.max() / med) if med > 0 else float("inf")
                    log(f"  {k:<14s} {arm:<6s} {phase:<9s} {basis:<4s} "
                        f"{np.nanmean(v):>11.3g} {np.nanstd(v):>10.3g} "
                        f"{np.nanmin(v):>11.3g} {np.nanmax(v):>11.3g} {spread:>10.3g}")


def head_fit(data, log=print):
    """How well the head tracks the driver it would replace at inference. Pearson r between the ACTUAL
    and HEAD-PREDICTED batch-mean test traces — i.e. between column 2 and column 3 of the figure, which
    is exactly the comparison the third panel invites. A low r means the deployed mechanism is NOT
    seeing that driver, whatever the formula says."""
    log("\n  head fit at test (Pearson r between actual and head-predicted batch-mean traces)")
    log(f"  {'driver':<14s} {'raw:naive':>10s} {'raw:er':>10s} {'std:naive':>10s} {'std:er':>10s}")
    log("  " + "-" * 60)
    for k in LOSS_DEPENDENT:
        row = []
        for basis in ("raw", "std"):
            for arm in ("naive", "er"):
                a = data.get(f"{arm}|test|{k}|{basis}|mean")
                b = data.get(f"{arm}|test|{k}{PRED_SUFFIX}|{basis}|mean")
                if a is None or b is None or not len(a) or not len(b):
                    row.append(float("nan")); continue
                m = np.isfinite(a) & np.isfinite(b)
                row.append(float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 and
                           a[m].std() > 0 and b[m].std() > 0 else float("nan"))
        log(f"  {k:<14s} {row[0]:>10.3f} {row[1]:>10.3f} {row[2]:>10.3f} {row[3]:>10.3f}")


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--point", default="tuned", choices=["tuned", "untuned"],
                    help="'tuned' (default): lr/epochs from configs.TUNED_MAIN[('classil',arm,opt)]; "
                         "'untuned': the inherited pt7 point (lr 1e-3, ep 5)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs/task (smoke tests)")
    ap.add_argument("--driver-loss", default="arm", choices=["arm", "pt7"],
                    help="'arm' (default): ell_i follows the arm's training loss (plain CE under er); "
                         "'pt7': masked CE in both arms, as pt7's Signals does")
    ap.add_argument("--no-shuffle-test", dest="shuffle_test", action="store_false",
                    help="keep the MNIST test-file order (adds the |x| ordering ramp as a confound, "
                         "which under live stats also feeds the running EMAs)")
    ap.add_argument("--freeze-at-test", action="store_true",
                    help="A/B: reproduce the frozen study's protocol (all stats stop at the last "
                         "training batch) instead of this study's live-update protocol")
    ap.add_argument("--suffix", default="", help="suffix for the npz/figure outputs (side-by-side variants)")
    ap.add_argument("--figs", default="all", choices=["all", "test", "none"])
    ap.add_argument("--plot-only", action="store_true", help="re-plot from the saved npz (no training)")
    ap.add_argument("--ckpt-tag", default="live", help="checkpoints as ckpt_<tag>_<arm>.pt ('' disables)")
    ap.add_argument("--retest", default="", metavar="TAG",
                    help="skip training: reload the <TAG> checkpoints and redo ONLY the test pass "
                         "(seconds). Training traces are carried over from the source npz.")
    ap.add_argument("--force", action="store_true", help="overwrite an existing npz (re-train)")
    args = ap.parse_args()

    global NPZ, FIGDIR
    if args.suffix:
        NPZ = HERE / f"live_traces{args.suffix}.npz"
        FIGDIR = HERE / f"figs{args.suffix}"

    if args.plot_only:
        z = np.load(NPZ, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        plot_all(d, panels=("all" if args.figs == "none" else args.figs))
        summarise(d); head_fit(d)
        return
    if NPZ.exists() and not (args.force or args.retest):
        raise SystemExit(f"{NPZ.name} already exists — nothing to do.\n"
                         f"  --plot-only   re-plot from it\n"
                         f"  --retest {args.ckpt_tag or 'TAG'}  redo only the test pass (seconds)\n"
                         f"  --force       re-train from scratch")

    # --retest redoes only the test pass, so the training traces come from the CANONICAL npz (not from
    # NPZ, which under --suffix is the variant's own output file and does not exist yet).
    src = None
    if args.retest:
        base = HERE / "live_traces.npz"
        src = np.load(base, allow_pickle=True) if base.exists() else None

    proto = "FROZEN (legacy A/B)" if args.freeze_at_test else "LIVE (nothing frozen)"
    print(f"device={DEV}  live driver traces  (opt={args.opt}, point={args.point}, seed 42, buffer 1000, "
          f"driver-loss={args.driver_loss}, test stats={proto})\n", flush=True)
    out = {"opt": np.array(args.opt), "point": np.array(args.point),
           "freeze_at_test": np.array(args.freeze_at_test)}
    for arm in ("naive", "er"):
        lr, ep = hparams(args.point, arm, args.opt)
        if args.epochs is not None:
            ep = args.epochs
        if args.retest:
            cp = torch.load(HERE / f"ckpt_{args.retest}_{arm}.pt", weights_only=False)
            lr, ep = cp["lr"], cp["epochs"]
            print(f"  retesting {arm} from checkpoint (lr={lr:g}, {ep} ep/task, "
                  f"driver-loss={cp['driver_loss']}) ...", flush=True)
            net = p7.Net().to(DEV); net.load_state_dict(cp["net"])
            te, acc = evaluate(net, cp["obs"], SplitMNIST(sequence=p7.SEQ),
                               shuffle_test=args.shuffle_test, freeze=args.freeze_at_test)
            tr = None
        else:
            print(f"  running {arm} (lr={lr:g}, {ep} ep/task) ...", flush=True)
            ck = (HERE / f"ckpt_{args.ckpt_tag}_{arm}.pt") if args.ckpt_tag else None
            tr, te, acc = run(arm, opt_kind=args.opt, lr=lr, epochs=ep, driver_loss=args.driver_loss,
                              shuffle_test=args.shuffle_test, freeze=args.freeze_at_test, ckpt=ck,
                              log=lambda s: print(s, flush=True))
        ref = REF.get((args.point, arm, args.opt)) if args.epochs is None else None
        tag = "" if ref is None else f"   (frozen-study run {ref:.4f}, delta {acc - ref:+.4f})"
        print(f"  {arm:5s} final avg class-IL acc = {acc:.4f}{tag}", flush=True)
        out[f"{arm}|lr"] = np.array(lr); out[f"{arm}|epochs"] = np.array(ep)
        for phase, t in (("train", tr), ("test", te)):
            if t is None:                  # --retest: reuse the source npz's training traces
                if src is not None:
                    for kk in src.files:
                        if kk.startswith(f"{arm}|train|"):
                            out[kk] = src[kk]
                continue
            for (k, b, s), v in t.d.items():
                out[f"{arm}|{phase}|{k}|{b}|{s}"] = np.asarray(v, dtype=np.float32)
            out[f"{arm}|{phase}|bounds"] = np.asarray(t.bounds, dtype=np.int32)
        out[f"{arm}|acc"] = np.asarray(acc, dtype=np.float32)
    np.savez_compressed(NPZ, **out)
    print(f"  wrote {NPZ}", flush=True)
    if args.figs != "none":
        plot_all(out, panels=args.figs)
    summarise(out, log=lambda s: print(s, flush=True))
    head_fit(out, log=lambda s: print(s, flush=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
