"""LOSS MODULATION — THESIS-PLAN direction B, position-paper mechanism 1.

    L = sum_T f(s_T, T) . L_T

Split the batch loss into one term per TASK and take a linear combination, where the coefficient
vector comes from pt6's replay-trained task-inference net. class-IL, Adam, at the val-tuned ER
operating point. Three coefficient forms were requested:

    soft   c_T = mean over the batch of the inference posterior p(T|x)
    ema    c_T = EMA of `soft` across steps
    dev    c_T = soft - ema + 1

THE STRUCTURAL FACT THAT SHAPES THE WHOLE STUDY, and that is worth stating before any result:
plain batch-mean CE ALREADY IS a task-weighted sum, with the weights being the true composition:

    mean CE  =  sum_T (n_T / N) . L_T

So c_T = n_T/N is not a neutral choice among many — it is EXACTLY plain ER, and it is the parity
control (`truefrac`). And `soft` is an ESTIMATOR of that same vector: if the inference net were
perfect, mean_i p(T|x_i) = n_T/N identically. **So `soft` can differ from ER only by inference
error.** That is a prediction, not a hedge: it says the soft variant should land on ER, and any
deviation measures the selector's miscalibration rather than a mechanism. `ema` and `dev` are the
forms that can genuinely depart, because they carry information across steps.

  A SCALE TRAP IN `dev`, HANDLED EXPLICITLY. sum_T soft = 1 and sum_T ema = 1, so sum_T dev = T.
  The `dev` loss is therefore ~T times larger than the others, which is an LEARNING-RATE change
  wearing a mechanism's clothes — the single most repeated artifact in this project (pt7's ach_ema
  "+0.11" was a 2.6x global LR boost that dissolved at a tuned lr; wd_modulation's two per-step
  "wins" were 60x weight rescales). `dev_norm` is `dev` renormalised to sum 1 and is what separates
  the two. Report BOTH: if dev != dev_norm, the difference is the LR confound, not the mechanism.

CONTROLS (rule #10, plus the lesson from wd_modulation that a dead control is not enough):
    truefrac  c = true batch composition == plain ER exactly. The PARITY control.
    uniform   c = 1/T over tasks present. Content-free, and a real CL trick (task balancing) — so it
              measures how much of any win is just "stop letting the current task dominate".
    learned   c = softmax(free parameter), trained by the main loss. CONTENT-FREE but LEARNED: it
              has the same degrees of freedom as `soft` and no dependence on x at all.

`learned` exists because of what wd_modulation found: a gate with m(x) == 1 reproduced ~85% of that
study's +0.023, and every real driver's margin over it was inside the noise floor. A dead control
cannot separate "the inference net said something useful" from "the coefficients had freedom to
move" — only a content-free-but-learned control can. It is built in from the start here rather than
retrofitted.

ARM. `er` only, and that is a structural choice rather than a budget one: with no replay a batch
holds ONE task, so exactly one L_T is nonzero and the whole mechanism collapses to a scalar rescale
of the loss — i.e. to a learning-rate knob, with nothing task-differentiated about it. The
rehearsal-free regime is still reported (rule #12) precisely to show that degeneracy rather than to
hide it.

Ledger `loss_modulation_results.tsv`. Run:
    uv run python -m neurocore.shard --script position_paper/loss_modulation.py \\
        --ledger position_paper/loss_modulation_results.tsv --split coef=soft,ema,dev,dev_norm \\
        --args "--part test --resume" --workers 6 --device cpu

    # every mechanism form against its random-posterior twin, normal + budget
    uv run python -m neurocore.shard --script position_paper/loss_modulation.py \\
        --ledger position_paper/loss_modulation_results.tsv \\
        --split coef=randproj,randproj_ema,randproj_dev,randproj_dev_norm \\
        --split part=test,regimes \\
        --args "--resume --formulation group --regime budget" --workers 3 --device cpu

DEVICE, AND A MIXED LEDGER FOUND THE HARD WAY. Device changes numerics (a CPU and an MPS run of the
same cell differ by up to ~0.005 here), so a ledger must not mix them. These ledgers DO: the `anchor`
and `tune` parts were run directly (MPS) while `test`/`regimes` went through the sharded CPU runner,
so the parity control's seed-42 `normal` row was an MPS row sitting in a CPU table. It was caught by
`--part rngcheck` reporting SHIFTED for a pair that is identical by construction — the check found a
defect it was not written to look for. The parity row has been re-run on CPU; the `val` rows are
still MPS and are left alone deliberately, since re-running them could move the selected inference lr
and orphan every test row keyed to it (and this study is not being re-tuned).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

import pt7_neuromodulators as p7                                   # noqa: E402  (frozen, read-only)
from neurocore import shard                                        # noqa: E402
from neurocore.cost import Cost, count_params                      # noqa: E402
from neurocore.ledger import NOISE_FLOOR, Ledger, where            # noqa: E402
from neurocore.task_selection import TaskInferenceNet              # noqa: E402
from neurocore.tuned import tuned_main                             # noqa: E402
from prototype.data import SplitMNIST, make_sequence               # noqa: E402

DEV = p7.DEV
PROBLEM, METRIC, BASE, OPT = "splitmnist", "classil", "er", "adam"

# TWO FORMULATIONS of "one term per task", kept in SEPARATE ledgers so the first set of results is
# not discarded when the second is added.
#
#   group  (v1)  L = sum_T c_T . L_T with L_T the mean 10-way CE over the batch's task-T SAMPLES.
#                Splits by sample. c = n_T/N is then EXACTLY plain ER, so `truefrac` is the parity
#                control and `soft` — an estimator of that same vector — is capped at ER by algebra.
#
#   logit  (v2)  The CE has one term per CLASS in its normaliser; each is scaled by that class's
#                task coefficient:  L = -log( c_y.exp(z_y) / sum_c c_task(c).exp(z_c) ),
#                implemented as the logit adjustment  z_c <- z_c + log c_task(c)  followed by plain
#                CE. Splits by CLASS, which is what "L_T contains only the classes of task T" means.
#
# The parity control MOVES between them, and that is the main thing to keep straight:
#   group -> `truefrac` is plain ER;  logit -> `uniform` is plain ER (a constant added to every
#   logit cancels in the softmax), while `truefrac` becomes a REAL mechanism — logit adjustment by
#   observed task frequency, i.e. the balanced-softmax / long-tail correction, reached from the
#   position paper's direction.
#   sample (v3)  Each SAMPLE's whole CE scaled by its own task's coefficient:
#                 L = (1/N) sum_i c_task(i) . CE_i . Expanding, this is
#                 sum_T c_T . (n_T/N) . L_T — i.e. the `group` family with c composed with the true
#                 fraction, NOT a logit adjustment. The gradient is c_y . (p - onehot(y)): the
#                 ordinary CE gradient scaled by a scalar, so the DIRECTION is unchanged and only
#                 that sample's effective weight (its learning rate) moves. Contrast `logit`, where
#                 c enters the log-sum-exp and changes the softmax probabilities themselves, hence
#                 the relative pressure BETWEEN classes.
#
#   Its parity control is `ones` (c == 1 exactly), NOT `uniform`: the other coefficient forms sum to
#   1, so c = 1/T makes the loss T times SMALLER — a global LR change, not parity. `w_mean` (the mean
#   per-sample weight) is ledgered for this formulation precisely so that scale is visible rather
#   than inferred; a form that differs only in `w_mean` is an LR knob, not a mechanism.
FORMULATIONS = ("group", "logit", "sample")
FORMULATION = "group"          # overridden by --formulation before the ledger is opened
PARITY = {"group": "truefrac", "logit": "uniform", "sample": "ones"}
_STEM = {"group": "loss_modulation_results", "logit": "loss_modulation_logit_results",
         "sample": "loss_modulation_sample_results"}


def _tsv(formulation):
    return shard.ledger_path(Path(__file__).resolve().parent / f"{_STEM[formulation]}.tsv")


def _metrics(formulation):
    """`sample` carries one extra column; it is a NEW ledger file, so this is not schema drift."""
    return METRICS + (["w_mean"] if formulation == "sample" else [])


TSV = _tsv(FORMULATION)
KEYS = ["regime", "coef", "nlr", "seed", "split"]
METRICS = ["acc", "forget", "infer", "c_sum", "c_sd", "c_err"]

COEFS = ("soft", "ema", "dev", "dev_norm")            # the mechanism forms
CONTROLS = ("truefrac", "uniform", "learned", "ones")

# RANDPROJ is a MODIFIER ON A FORM, not a form of its own: the posterior comes from a FROZEN RANDOM
# 784->T projection instead of the replay-trained inference net, and everything downstream (the EMA,
# the deviation, the renormalisation) is computed from it exactly as before. Nothing about it is
# learned, so it isolates "does c need to carry real task information" from "does c need to vary per
# batch at all" — the latter being what made the content-free `learned` vector fail in `logit`.
#
# Applied to EVERY mechanism form rather than to `soft` alone, because the forms differ in what they
# do with the posterior and there is no reason the answer transfers. `soft` reads it directly; `ema`
# smooths it over ~100 steps, which suppresses per-batch noise and could therefore be MORE damaged by
# a random posterior (it averages away the real signal too) or LESS (a random projection of MNIST is
# a stable statistic, so its EMA is close to a constant = the `learned`/`uniform` regime); `dev` sees
# only the difference between the two, which for a random posterior is per-batch noise around zero.
#
# `randproj` (no suffix) is kept as the name for randproj+`soft` so the rows already in the ledgers
# keep their key. NOTE those rows are superseded: they drew R from the GLOBAL torch stream, which
# shifted the replay draws (rule #10) and made them not RNG-matched to their own `soft` twin. R now
# comes from a private generator; see `_rand_proj`.
RANDPROJ = ("randproj",) + tuple(f"randproj_{c}" for c in COEFS if c != "soft")
FIXP_SIGMA = 0.1
RANDPROJ_SEED_OFFSET = 9000
ALL_COEFS = COEFS + CONTROLS + RANDPROJ
SEEDS = (42, 43, 44)
TUNE_SEED = 42
BUFFERS = {"normal": 1000, "budget": 200, "rfree": 0}
NEURO_GRID = (1e-4, 1e-3, 1e-2)                       # inference-net lr
EMA_RATE = 0.01
N_TASKS = 5
VAL_SEQ = make_sequence(7)
VAL_FRAC = 0.1

# Plain ER at this operating point, from the wd_modulation harness (same tuned point, Adam).
ANCHOR_ER = 0.9019


def _label_to_task(seq):
    """label -> task index, built from the sequence (NOT label//2 — the val sequence is shuffled)."""
    m = torch.full((10,), -1, dtype=torch.long)
    for t, pair in enumerate(seq):
        for c in pair:
            m[c] = t
    return m.to(DEV)


def _split_coef(coef):
    """coef -> (mechanism form, use a random posterior?).

    `randproj` is the legacy key for randproj+`soft`; `randproj_<form>` is the general spelling.
    """
    if coef == "randproj":
        return "soft", True
    if coef.startswith("randproj_"):
        return coef[len("randproj_"):], True
    return coef, False


def _rand_proj(seed):
    """The frozen 784->T projection, drawn from a PRIVATE generator — never the global stream.

    This is the whole reason `randproj_X` can be read against `X` directly. Drawing R from the global
    stream (as the first version did) consumes RNG before training and shifts every replay draw, so
    the pair would differ by the rule-#10 shift as well as by the mechanism, and at width 400 that
    shift is worth ~0.002 — the same order as the effects here. With a private generator the two runs
    consume the global stream identically and any difference is the posterior. `--part rngcheck`
    verifies that claim rather than asserting it.
    """
    g = torch.Generator().manual_seed(seed + RANDPROJ_SEED_OFFSET)
    return (torch.randn(784, N_TASKS, generator=g) * FIXP_SIGMA).to(DEV)


def coefficients(kind, post, tids, ema, free):
    """The T-vector c, plus the true composition for reporting. `post` is (B, T), `tids` is (B,)."""
    n = torch.bincount(tids, minlength=N_TASKS).float()
    true = n / n.sum()
    if kind == "truefrac":
        return true, true
    if kind == "ones":
        return torch.ones(N_TASKS, device=post.device), true      # `sample` parity: L == plain CE
    if kind == "uniform":
        present = (n > 0).float()
        return present / present.sum(), true
    if kind == "learned":
        return torch.softmax(free, 0), true
    soft = post.mean(0)
    if kind == "soft":
        return soft, true
    if kind == "ema":
        return ema.clone(), true
    dev = soft - ema + 1.0
    if kind == "dev":
        return dev, true                               # sums to T — the scale trap, kept on purpose
    return dev / dev.sum(), true                       # dev_norm


def modulated_loss(logits, y, tids, c, formulation="group", l2t=None):
    """The task-weighted loss, in whichever formulation is selected.

    `group`: sum_T c_T . L_T over tasks PRESENT in the batch, L_T the mean CE over that task's
    samples. With c = n_T/N this is identically the plain batch-mean CE.

    `logit`: the CE's per-CLASS terms scaled by their task's coefficient. Done as an additive
    adjustment in logit space, `z_c += log c_task(c)`, which is exactly equivalent to scaling
    exp(z_c) by c_task(c) inside the softmax and is numerically far better behaved than forming the
    ratio directly. A CONSTANT c cancels (softmax is shift-invariant), which is what makes `uniform`
    an exact parity control here.
    """
    if formulation == "sample":
        # v3: scale each sample's whole CE by its own task's coefficient.
        w = c[tids]
        return (w * F.cross_entropy(logits, y, reduction="none")).mean()
    if formulation == "group":
        total = logits.new_zeros(())
        for t in range(N_TASKS):
            mask = tids == t
            if not mask.any():
                continue
            total = total + c[t] * F.cross_entropy(logits[mask], y[mask])
        return total
    adj = torch.log(c.clamp_min(1e-12))[l2t]          # (10,) per-class, from its task's coefficient
    return F.cross_entropy(logits + adj.unsqueeze(0), y)


def run(coef, seed, nlr, main_lr, epochs, buffer, split="test"):
    """One cell. Reads the module-level FORMULATION (set once from --formulation)."""
    p7.seed_all(seed)
    form, use_rp = _split_coef(coef)
    seq = VAL_SEQ if split == "val" else p7.SEQ
    ds = SplitMNIST(sequence=seq, val_frac=VAL_FRAC if split == "val" else 0.0)
    loaders = [ds.get_task_loaders(t, 64) for t in range(N_TASKS)]
    evals = [ds.get_task_val_loader(t, 64) if split == "val" else loaders[t][1]
             for t in range(N_TASKS)]
    l2t = _label_to_task(seq)

    net = p7.Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), main_lr)
    inf = TaskInferenceNet(n_tasks=N_TASKS).to(DEV)
    randR = _rand_proj(seed) if use_rp else None
    inf_opt = torch.optim.Adam(inf.params(), nlr)
    free = torch.zeros(N_TASKS, device=DEV, requires_grad=True)
    free_opt = torch.optim.Adam([free], nlr)
    buf = p7.Reservoir(buffer) if buffer > 0 else None
    ema = torch.full((N_TASKS,), 1.0 / N_TASKS, device=DEV)
    A = np.full((N_TASKS, N_TASKS), np.nan)
    cs, errs, ws = [], [], []

    for t in range(N_TASKS):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                xf = x.view(x.size(0), -1)
                rep = buf.sample_any(64) if buf is not None else None
                if rep is not None:
                    Xm = torch.cat([xf, rep[0].to(DEV)]); Ym = torch.cat([y, rep[1].to(DEV)])
                else:
                    Xm, Ym = xf, y
                tids = l2t[Ym]

                with torch.no_grad():
                    post = (torch.softmax(Xm @ randR, dim=1) if randR is not None
                            else inf.posterior(Xm))
                    soft = post.mean(0)
                    ema = (1 - EMA_RATE) * ema + EMA_RATE * soft
                c, true = coefficients(form, post, tids, ema, free)
                loss = modulated_loss(net.plain(Xm)[0], Ym, tids, c, FORMULATION, l2t)
                opt.zero_grad()
                if form == "learned":
                    free_opt.zero_grad()
                loss.backward()
                opt.step()
                if form == "learned":
                    free_opt.step()

                # the inference net trains WITH REPLAY on task-CE — pt6's finding is that replay is
                # what makes a selector work at all (no buffer -> infer collapses to chance)
                inf_opt.zero_grad()
                F.cross_entropy(inf.task_logits(Xm), tids).backward()
                inf_opt.step()

                with torch.no_grad():
                    ws.append(float(c.detach()[tids].mean()))
                    cs.append(c.detach().cpu())
                    errs.append((c.detach() / max(float(c.sum()), 1e-8) - true).abs().sum().item())
                if buf is not None:
                    buf.add(x, y)
        for i in range(t + 1):
            A[t, i] = _acc(net, evals[i])

    acc = float(np.nanmean(A[N_TASKS - 1, :]))
    forget = float(np.mean([max([A[k, i] for k in range(i, N_TASKS)]) - A[N_TASKS - 1, i]
                            for i in range(N_TASKS)]))
    C = torch.stack(cs)
    return dict(acc=acc, forget=forget, infer=_infer_acc(inf, evals, l2t),
                c_sum=float(C.sum(1).mean()), c_sd=float(C.std(0).mean()),
                c_err=float(np.mean(errs)), w_mean=float(np.mean(ws)),
                cost=Cost(backbone_params=count_params(net),
                          extra_params=count_params(inf),
                          buffer_bytes=0 if buf is None else
                          buf.X.element_size() * buf.X.nelement()
                          + buf.Y.element_size() * buf.Y.nelement(),
                          # 1 main fwd + 1 inference fwd; 1 main bwd + 1 inference bwd.
                          # Inference is FREE at test: the coefficients only ever shaped training.
                          fwd_train=2.0, bwd_train=2.0, fwd_infer=1.0, bwd_infer=0.0))


@torch.no_grad()
def _acc(net, loader):
    net.eval()
    c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        c += (net.plain(x)[0].argmax(1) == y).sum().item(); tot += len(y)
    net.train()
    return c / tot


@torch.no_grad()
def _infer_acc(inf, evals, l2t):
    """Task accuracy of the selector. Reported because pt6's routing law makes it the quantity that
    caps anything task-conditioned — and here it also bounds how far `soft` can stray from ER."""
    c = tot = 0
    for i, ld in enumerate(evals):
        for x, y in ld:
            x, y = x.to(DEV), y.to(DEV)
            p = inf.posterior(x.view(x.size(0), -1))
            c += (p.argmax(1) == l2t[y]).sum().item(); tot += len(y)
    return c / tot


# ========================================================================================== driving
def ledger():
    return Ledger(TSV, keys=KEYS, metrics=_metrics(FORMULATION), with_cost=True)


def run_cell(led, regime, coef, nlr, seed, split="test"):
    key = dict(regime=regime, coef=coef, nlr=f"{nlr:g}", seed=seed, split=split)
    if led.is_done(**key):
        rows = where(led.rows(), **key)
        return float(rows[0]["acc"])
    p = tuned_main(PROBLEM, METRIC, BASE, OPT)
    r = run(coef, seed, nlr, p["lr"], p["epochs_per_task"], BUFFERS[regime], split=split)
    led.append(key, {k: r[k] for k in _metrics(FORMULATION)}, cost=r["cost"])
    print(f"  {regime:6s} {coef:9s} nlr={nlr:<7g} s{seed} {split:4s} acc={r['acc']:.4f} "
          f"forget={r['forget']:.4f} infer={r['infer']:.4f} sum(c)={r['c_sum']:.3f} "
          f"c_err={r['c_err']:.4f}", flush=True)
    return r["acc"]


def part_anchor(led):
    """`truefrac` is algebraically plain ER, so it must land on the frozen ER number. That makes the
    parity control double as this study's anchor."""
    par = PARITY[FORMULATION]
    print(f"ANCHOR — `{par}` is plain ER by construction in the `{FORMULATION}` formulation\n",
          flush=True)
    acc = run_cell(led, "normal", par, 1e-3, 42)
    d = acc - ANCHOR_ER
    print(f"    {par} {acc:.6f} vs ER {ANCHOR_ER:.6f}  d={d:+.6f}  "
          f"{'~noise' if abs(d) < NOISE_FLOOR else 'MISMATCH'}", flush=True)


def part_rngcheck(led):
    """`randproj_truefrac` MUST be bit-identical to `truefrac`.

    `truefrac`'s coefficients are the batch composition, so they ignore the posterior entirely — the
    only thing the randproj path changes in that cell is that R is built and a random posterior is
    computed and thrown away. If the two runs agree, R costs no global RNG and every `randproj_X` is
    RNG-matched to its `X` twin; if they differ, no paired reading in this study is safe.
    """
    print("RNG-MATCH CHECK — `randproj_truefrac` vs `truefrac` (the posterior is unused in both)\n",
          flush=True)
    a = run_cell(led, "normal", "truefrac", 1e-3, 42)
    b = run_cell(led, "normal", "randproj_truefrac", 1e-3, 42)
    # 1e-6, not 0: both sides are round-tripped through the ledger's "%.6f".
    print(f"    truefrac {a:.6f}  randproj_truefrac {b:.6f}  d={b - a:+.6f}  "
          f"{'MATCHED' if abs(b - a) < 1e-6 else 'SHIFTED — paired randproj readings are unsafe'}",
          flush=True)


def part_tune(led):
    """One inference-net lr, swept once and shared: it is the SAME net for soft/ema/dev, so tuning
    per coefficient form would be tuning the same object three times on the same data (rule #3)."""
    print(f"TUNE inference-net lr on VAL — grid {NEURO_GRID}\n", flush=True)
    scores = {nlr: run_cell(led, "normal", "soft", nlr, TUNE_SEED, split="val")
              for nlr in NEURO_GRID}
    best = max(scores, key=lambda k: scores[k])
    span = max(scores.values()) - min(scores.values())
    note = "  !! span < noise floor — UNRESOLVED at 1 seed" if span < NOISE_FLOOR else ""
    print(f"  >>> inference lr = {best:g} (val {scores[best]:.4f}, span {span:.4f}){note}",
          flush=True)
    return best


def tuned_inf_lr(led):
    rows = where(led.rows(), coef="soft", split="val", seed=TUNE_SEED, regime="normal")
    if not rows:
        raise KeyError("no val rows — run --part tune first")
    return float(max(rows, key=lambda r: float(r["acc"]))["nlr"])


def part_test(led, coefs, regime="normal"):
    nlr = tuned_inf_lr(led)
    print(f"TEST — 3 seeds at inference lr {nlr:g}, regime {regime}\n", flush=True)
    for coef in coefs:
        for s in SEEDS:
            run_cell(led, regime, coef, nlr, s)


def part_regimes(led, coefs, regimes=("budget", "rfree")):
    for regime in regimes:
        part_test(led, coefs, regime=regime)


def part_report(led, regime="normal"):
    rows = led.rows()
    nlr = tuned_inf_lr(led)
    print("\n" + "=" * 96)
    print(f"LOSS MODULATION  L = sum_T c_T . L_T   |   class-IL / ER / Adam   |   regime={regime}")
    print("=" * 96)
    par = PARITY[FORMULATION]
    ref = _accs(rows, par, nlr, regime)
    print(f"  parity control `{par}` (== plain ER in the `{FORMULATION}` formulation) "
          f"{np.mean(ref):.4f} +- {np.std(ref):.4f}" if ref else f"  {par} not run")
    print(f"\n  {'coef':10s} {'acc':>9s} {'sd':>8s} {'d-' + par:>12s} {'pos':>5s} "
          f"{'forget':>8s} {'infer':>7s} {'sum(c)':>8s} {'c_err':>8s}")
    for coef in ALL_COEFS:
        a = _accs(rows, coef, nlr, regime)
        if not a or not ref:
            continue
        d = [x - y for x, y in zip(a, ref)]
        ex = _one(rows, coef, nlr, regime)
        flag = " " if abs(np.mean(d)) >= NOISE_FLOOR else "~"
        print(f"  {coef:10s} {np.mean(a):>9.4f} {np.std(a):>8.4f} {np.mean(d):>+12.4f}{flag}"
              f"{sum(x > 0 for x in d)}/{len(d):<3d} {float(ex['forget']):>8.4f} "
              f"{float(ex['infer']):>7.4f} {float(ex['c_sum']):>8.3f} {float(ex['c_err']):>8.4f}")
    print(f"\n  ~ = |d| < {NOISE_FLOOR} (noise floor). `{par}` IS plain ER algebraically in this")
    print("  formulation, so it is the parity line, not a competitor. NOTE the parity control MOVES")
    print("  between formulations: group -> truefrac, logit -> uniform (a constant added to every")
    print("  logit cancels in the softmax). In `logit`, `truefrac` is a real mechanism — logit")
    print("  adjustment by observed task frequency, i.e. balanced softmax.")
    print("  Read `dev` against `dev_norm`: a gap between them is")
    print("  the sum(c)=T LEARNING-RATE confound, not the mechanism. Read every form against")
    print("  `learned` (content-free but with the same freedom) before crediting the inference net.")
    print("  c_err = |c/sum(c) - true composition|, i.e. how far the coefficients sit from the exact")
    print("  weighting plain ER already uses — the quantity `soft` can only differ from ER through.")
    _randproj_table(rows, nlr, regime)


def _randproj_table(rows, nlr, regime):
    """Each mechanism form against ITS OWN random-posterior twin.

    R is drawn from a private generator, so a form and its twin consume the global RNG identically
    and this is a paired live-vs-live reading — the one comparison in the study that does not have to
    be routed through a d-parity column to cancel a stream shift.
    """
    print(f"\n  RANDOM-POSTERIOR TWIN  (frozen random 784->{N_TASKS} projection, nothing learned)")
    print(f"  {'form':10s} {'real':>9s} {'randproj':>10s} {'d':>10s} {'neg':>5s} "
          f"{'c_err real':>11s} {'c_err rp':>9s}")
    for form in COEFS:
        twin = "randproj" if form == "soft" else f"randproj_{form}"
        a, b = _accs(rows, form, nlr, regime), _accs(rows, twin, nlr, regime)
        if not a or not b or len(a) != len(b):
            continue
        d = [y - x for x, y in zip(a, b)]
        flag = " " if abs(np.mean(d)) >= NOISE_FLOOR else "~"
        print(f"  {form:10s} {np.mean(a):>9.4f} {np.mean(b):>10.4f} {np.mean(d):>+10.4f}{flag}"
              f"{sum(x < 0 for x in d)}/{len(d):<3d} "
              f"{float(_one(rows, form, nlr, regime)['c_err']):>11.4f} "
              f"{float(_one(rows, twin, nlr, regime)['c_err']):>9.4f}")
    print("  d = randproj - real, so NEGATIVE means the replay-trained posterior was worth something")
    print("  and `neg` counts the seeds where it was. A form whose d is ~0 does not need real task")
    print("  content at all — only a per-batch-varying vector, which is what separates this control")
    print("  from the constant `learned` one.")


def _accs(rows, coef, nlr, regime):
    sel = where(rows, coef=coef, nlr=f"{nlr:g}", split="test", regime=regime)
    return [float(r["acc"]) for r in sorted(sel, key=lambda r: int(r["seed"]))]


def _one(rows, coef, nlr, regime):
    return where(rows, coef=coef, nlr=f"{nlr:g}", split="test", regime=regime)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "anchor", "rngcheck", "tune", "test", "regimes", "report"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--coef", default=None, help="comma filter (the shard axis)")
    ap.add_argument("--regime", default="normal", help="report regime: normal,budget,rfree")
    ap.add_argument("--formulation", default="group", choices=list(FORMULATIONS),
                    help="group = split by SAMPLE (v1); logit = per-CLASS scaling (v2)")
    a = ap.parse_args()
    coefs = tuple(a.coef.split(",")) if a.coef else ALL_COEFS
    global FORMULATION, TSV
    FORMULATION = a.formulation
    TSV = _tsv(FORMULATION)                 # set BEFORE the ledger is opened
    led = ledger()
    print(f"loss modulation | device {DEV} | formulation {FORMULATION} | coefs {coefs}\n"
          f"ledger {TSV}\n", flush=True)
    if a.part in ("all", "anchor"):
        part_anchor(led)
    if a.part == "rngcheck":
        part_rngcheck(led)
    if a.part in ("all", "tune"):
        part_tune(led)
    if a.part in ("all", "test"):
        part_test(led, coefs)
    if a.part == "regimes":
        part_regimes(led, coefs, regimes=tuple(a.regime.split(",")))
    if a.part in ("all", "report"):
        part_report(led, regime=a.regime)


if __name__ == "__main__":
    main()
