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

TSV = shard.ledger_path(Path(__file__).resolve().parent / "loss_modulation_results.tsv")
KEYS = ["regime", "coef", "nlr", "seed", "split"]
METRICS = ["acc", "forget", "infer", "c_sum", "c_sd", "c_err"]

COEFS = ("soft", "ema", "dev", "dev_norm")            # the mechanism forms
CONTROLS = ("truefrac", "uniform", "learned")         # see module docstring
ALL_COEFS = COEFS + CONTROLS
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


def coefficients(kind, post, tids, ema, free):
    """The T-vector c, plus the true composition for reporting. `post` is (B, T), `tids` is (B,)."""
    n = torch.bincount(tids, minlength=N_TASKS).float()
    true = n / n.sum()
    if kind == "truefrac":
        return true, true
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


def modulated_loss(logits, y, tids, c):
    """sum_T c_T . L_T over tasks PRESENT in the batch.

    L_T is the mean CE over that task's samples, so with c = n_T/N this is identically the plain
    batch-mean CE — which is what makes `truefrac` an exact parity control rather than an approximate
    one.
    """
    total = logits.new_zeros(())
    for t in range(N_TASKS):
        mask = tids == t
        if not mask.any():
            continue
        total = total + c[t] * F.cross_entropy(logits[mask], y[mask])
    return total


def run(coef, seed, nlr, main_lr, epochs, buffer, split="test"):
    p7.seed_all(seed)
    seq = VAL_SEQ if split == "val" else p7.SEQ
    ds = SplitMNIST(sequence=seq, val_frac=VAL_FRAC if split == "val" else 0.0)
    loaders = [ds.get_task_loaders(t, 64) for t in range(N_TASKS)]
    evals = [ds.get_task_val_loader(t, 64) if split == "val" else loaders[t][1]
             for t in range(N_TASKS)]
    l2t = _label_to_task(seq)

    net = p7.Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), main_lr)
    inf = TaskInferenceNet(n_tasks=N_TASKS).to(DEV)
    inf_opt = torch.optim.Adam(inf.params(), nlr)
    free = torch.zeros(N_TASKS, device=DEV, requires_grad=True)
    free_opt = torch.optim.Adam([free], nlr)
    buf = p7.Reservoir(buffer) if buffer > 0 else None
    ema = torch.full((N_TASKS,), 1.0 / N_TASKS, device=DEV)
    A = np.full((N_TASKS, N_TASKS), np.nan)
    cs, errs = [], []

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
                    post = inf.posterior(Xm)
                    soft = post.mean(0)
                    ema = (1 - EMA_RATE) * ema + EMA_RATE * soft
                c, true = coefficients(coef, post, tids, ema, free)
                loss = modulated_loss(net.plain(Xm)[0], Ym, tids, c)
                opt.zero_grad()
                if coef == "learned":
                    free_opt.zero_grad()
                loss.backward()
                opt.step()
                if coef == "learned":
                    free_opt.step()

                # the inference net trains WITH REPLAY on task-CE — pt6's finding is that replay is
                # what makes a selector work at all (no buffer -> infer collapses to chance)
                inf_opt.zero_grad()
                F.cross_entropy(inf.task_logits(Xm), tids).backward()
                inf_opt.step()

                with torch.no_grad():
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
                c_err=float(np.mean(errs)),
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
    return Ledger(TSV, keys=KEYS, metrics=METRICS, with_cost=True)


def run_cell(led, regime, coef, nlr, seed, split="test"):
    key = dict(regime=regime, coef=coef, nlr=f"{nlr:g}", seed=seed, split=split)
    if led.is_done(**key):
        rows = where(led.rows(), **key)
        return float(rows[0]["acc"])
    p = tuned_main(PROBLEM, METRIC, BASE, OPT)
    r = run(coef, seed, nlr, p["lr"], p["epochs_per_task"], BUFFERS[regime], split=split)
    led.append(key, {k: r[k] for k in METRICS}, cost=r["cost"])
    print(f"  {regime:6s} {coef:9s} nlr={nlr:<7g} s{seed} {split:4s} acc={r['acc']:.4f} "
          f"forget={r['forget']:.4f} infer={r['infer']:.4f} sum(c)={r['c_sum']:.3f} "
          f"c_err={r['c_err']:.4f}", flush=True)
    return r["acc"]


def part_anchor(led):
    """`truefrac` is algebraically plain ER, so it must land on the frozen ER number. That makes the
    parity control double as this study's anchor."""
    print("ANCHOR — `truefrac` is plain ER by construction\n", flush=True)
    acc = run_cell(led, "normal", "truefrac", 1e-3, 42)
    d = acc - ANCHOR_ER
    print(f"    truefrac {acc:.6f} vs ER {ANCHOR_ER:.6f}  d={d:+.6f}  "
          f"{'~noise' if abs(d) < NOISE_FLOOR else 'MISMATCH'}", flush=True)


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


def part_regimes(led, coefs):
    for regime in ("budget", "rfree"):
        part_test(led, coefs, regime=regime)


def part_report(led, regime="normal"):
    rows = led.rows()
    nlr = tuned_inf_lr(led)
    print("\n" + "=" * 96)
    print(f"LOSS MODULATION  L = sum_T c_T . L_T   |   class-IL / ER / Adam   |   regime={regime}")
    print("=" * 96)
    ref = _accs(rows, "truefrac", nlr, regime)
    print(f"  parity control `truefrac` (== plain ER) {np.mean(ref):.4f} +- {np.std(ref):.4f}"
          if ref else "  truefrac not run")
    print(f"\n  {'coef':10s} {'acc':>9s} {'sd':>8s} {'d-truefrac':>12s} {'pos':>5s} "
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
    print(f"\n  ~ = |d| < {NOISE_FLOOR} (noise floor). `truefrac` IS plain ER algebraically, so it is")
    print("  the parity line, not a competitor. Read `dev` against `dev_norm`: a gap between them is")
    print("  the sum(c)=T LEARNING-RATE confound, not the mechanism. Read every form against")
    print("  `learned` (content-free but with the same freedom) before crediting the inference net.")
    print("  c_err = |c/sum(c) - true composition|, i.e. how far the coefficients sit from the exact")
    print("  weighting plain ER already uses — the quantity `soft` can only differ from ER through.")


def _accs(rows, coef, nlr, regime):
    sel = where(rows, coef=coef, nlr=f"{nlr:g}", split="test", regime=regime)
    return [float(r["acc"]) for r in sorted(sel, key=lambda r: int(r["seed"]))]


def _one(rows, coef, nlr, regime):
    return where(rows, coef=coef, nlr=f"{nlr:g}", split="test", regime=regime)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "anchor", "tune", "test", "regimes", "report"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--coef", default=None, help="comma filter (the shard axis)")
    ap.add_argument("--regime", default="normal", help="report regime: normal,budget,rfree")
    a = ap.parse_args()
    coefs = tuple(a.coef.split(",")) if a.coef else ALL_COEFS
    led = ledger()
    print(f"loss modulation | device {DEV} | coefs {coefs}\nledger {TSV}\n", flush=True)
    if a.part in ("all", "anchor"):
        part_anchor(led)
    if a.part in ("all", "tune"):
        part_tune(led)
    if a.part in ("all", "test"):
        part_test(led, coefs)
    if a.part == "regimes":
        part_regimes(led, coefs)
    if a.part in ("all", "report"):
        part_report(led, regime=a.regime)


if __name__ == "__main__":
    main()
