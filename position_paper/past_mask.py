"""PAST-ONLY OUTPUT MASKING — THESIS-PLAN B, own variant. class-IL, Adam.

User-requested: in the naive arm, use a masked loss that masks **the past tasks only**.

THE 2x2 THIS COMPLETES. pt3-iter5 established that class-IL forgetting is dominated by output-logit
competition, and that masking the training loss to a sample's OWN task recovers a large chunk
(0.198 -> 0.389 untuned; 0.5430 -> 0.5990 at the tuned point in pt3_retry). That is one of four
possible ways to restrict which logits a sample competes against, and only two had been run:

    none    every logit competes            = plain naive
    curr    only the sample's own 2 logits  = pt3's lever B
    past    everything EXCEPT older tasks   = REQUESTED — old logits get no negative gradient
    future  everything EXCEPT newer tasks   = the complement, run as the control that makes the
                                             other three readable

WHY `past` IS NOT JUST `curr` IN DISGUISE, AND WHY `future` IS THE RIGHT CONTROL. Old classes are
forgotten because current-task samples push their logits DOWN. `past` removes exactly that gradient
while leaving the current task free to compete with the not-yet-used future logits — so it should
protect old classes about as well as `curr` while training a wider head. `future` does the opposite:
it keeps the suppression of old classes and removes only the harmless part, so it should behave like
`none`. If `past ~= curr` and `future ~= none`, the mechanism is confirmed to be "stop pushing old
logits down" and nothing else. That pair of predictions is the study.

  A PROPERTY WORTH SAYING OUT LOUD: `past` is a PROGRESSIVE mask. At task 0 nothing is past, so it
  IS `none`; at the last task everything but the current pair is past, so it IS `curr`. It
  interpolates between the two arms pt3 already measured, which is what makes it interesting and
  also what caps it — it cannot beat `curr` by more than the early-task slack.

MASKING IS PER SAMPLE, by the sample's OWN task (not the loop's current task). For the naive arm
those coincide, but per-sample is what keeps the definition meaningful under replay, where a
buffered sample's "past" is older than the current task. `MaskedCE` in the frozen code masks per
sample for the same reason (else ER's replay is broken).

TUNING. Every variant gets its OWN lr grid, and that is not optional here: pt3_retry found the
naive+masked arm's Adam optimum at lr 1e-5, FIVE DECADES below the ER grid's floor, because replay
tolerates and needs large steps while an unreplayed net forgets harder the larger the step. Reusing
one arm's grid for another was worth 0.22 accuracy there. Consequence, also from pt3_retry: **never
compare arm to arm directly, only each variant to its own baseline at its own tuned point.**

ANCHORS: `none` and `curr` must reproduce pt3_retry's 3-seed tuned numbers (naive 0.5430 +- 0.0247,
naive+masked 0.5990) — they are the same configurations by construction.

Ledger `past_mask_results.tsv`.
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
from neurocore.tuned import tuned_main                             # noqa: E402
from prototype.data import SplitMNIST, make_sequence               # noqa: E402

DEV = p7.DEV
PROBLEM, METRIC, OPT = "splitmnist", "classil", "adam"

TSV = shard.ledger_path(Path(__file__).resolve().parent / "past_mask_results.tsv")
KEYS = ["regime", "arm", "mask", "lr", "seed", "split"]
METRICS = ["acc", "forget", "n_allowed"]

MASKS = ("none", "curr", "past", "future")
ARMS = ("naive", "er")
SEEDS = (42, 43, 44)
TUNE_SEED = 42
BUFFERS = {"naive": 0, "er": 1000}
# spans both known optima: naive+masked adam = 1e-5 (pt3_retry) and naive adam = 3e-4
LR_GRID = (1e-6, 1e-5, 1e-4, 3e-4, 1e-3)
N_TASKS = 5
VAL_SEQ = make_sequence(7)
VAL_FRAC = 0.1
EPOCHS = 5

# pt3_retry, 3-seed, tuned, class-IL Adam. Same configurations by construction.
ANCHORS = {"none": 0.5430, "curr": 0.5990}


def _label_to_task(seq):
    m = torch.full((10,), -1, dtype=torch.long)
    for t, pair in enumerate(seq):
        for c in pair:
            m[c] = t
    return m.to(DEV)


def allowed_mask(kind, tids, l2t):
    """(B, 10) additive mask: 0 where a logit may compete, -inf where it may not.

    `cls_task[c]` is class c's task; `tids[i]` is sample i's task. The four variants are the four
    comparisons between them, which is what makes this a 2x2 rather than a list of tricks.
    """
    if kind == "none":
        return None
    cls_task = l2t.unsqueeze(0)                       # (1, 10)
    own = tids.unsqueeze(1)                           # (B, 1)
    if kind == "curr":
        ok = cls_task == own
    elif kind == "past":
        ok = cls_task >= own                          # drop STRICTLY older tasks
    else:                                             # future
        ok = cls_task <= own                          # drop STRICTLY newer tasks
    return torch.where(ok, 0.0, float("-inf"))


def masked_ce(logits, y, mask):
    return F.cross_entropy(logits if mask is None else logits + mask, y)


def run(mask_kind, arm, seed, lr, epochs=EPOCHS, split="test"):
    p7.seed_all(seed)
    seq = VAL_SEQ if split == "val" else p7.SEQ
    ds = SplitMNIST(sequence=seq, val_frac=VAL_FRAC if split == "val" else 0.0)
    loaders = [ds.get_task_loaders(t, 64) for t in range(N_TASKS)]
    evals = [ds.get_task_val_loader(t, 64) if split == "val" else loaders[t][1]
             for t in range(N_TASKS)]
    l2t = _label_to_task(seq)

    net = p7.Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr)
    buf = p7.Reservoir(BUFFERS[arm]) if BUFFERS[arm] > 0 else None
    A = np.full((N_TASKS, N_TASKS), np.nan)
    n_allowed = []

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
                m = allowed_mask(mask_kind, l2t[Ym], l2t)
                loss = masked_ce(net.plain(Xm)[0], Ym, m)
                opt.zero_grad(); loss.backward(); opt.step()
                n_allowed.append(10.0 if m is None else float((m == 0).sum(1).float().mean()))
                if buf is not None:
                    buf.add(x, y)
        for i in range(t + 1):
            A[t, i] = _acc(net, evals[i])

    last = N_TASKS - 1
    return dict(acc=float(np.nanmean(A[last, :])),
                forget=float(np.mean([max([A[k, i] for k in range(i, N_TASKS)]) - A[last, i]
                                      for i in range(N_TASKS)])),
                n_allowed=float(np.mean(n_allowed)),
                cost=Cost(backbone_params=count_params(net), extra_params=0,
                          buffer_bytes=0 if buf is None else
                          buf.X.element_size() * buf.X.nelement()
                          + buf.Y.element_size() * buf.Y.nelement(),
                          # masking is free: it changes the loss, not the network
                          fwd_train=1.0, bwd_train=1.0, fwd_infer=1.0, bwd_infer=0.0))


@torch.no_grad()
def _acc(net, loader):
    net.eval()
    c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        c += (net.plain(x)[0].argmax(1) == y).sum().item(); tot += len(y)
    net.train()
    return c / tot


# ========================================================================================== driving
def ledger():
    return Ledger(TSV, keys=KEYS, metrics=METRICS, with_cost=True)


def run_cell(led, arm, mask, lr, seed, split="test"):
    key = dict(regime="normal", arm=arm, mask=mask, lr=f"{lr:g}", seed=seed, split=split)
    if led.is_done(**key):
        return float(where(led.rows(), **key)[0]["acc"])
    r = run(mask, arm, seed, lr, split=split)
    led.append(key, {k: r[k] for k in METRICS}, cost=r["cost"])
    print(f"  {arm:5s} {mask:6s} lr={lr:<8g} s{seed} {split:4s} acc={r['acc']:.4f} "
          f"forget={r['forget']:.4f} allowed={r['n_allowed']:.2f}/10", flush=True)
    return r["acc"]


def part_anchor(led):
    """`none` and `curr` in the naive arm are pt3_retry's two tuned configurations."""
    print("ANCHOR — vs pt3_retry (3-seed, tuned, class-IL Adam)\n", flush=True)
    for mask, base in (("none", "naive"), ("curr", "naive_masked")):
        lr = tuned_main(PROBLEM, METRIC, base, OPT)["lr"]
        accs = [run_cell(led, "naive", mask, lr, s) for s in SEEDS]
        d = float(np.mean(accs)) - ANCHORS[mask]
        print(f"    {mask:6s} lr={lr:g}  {np.mean(accs):.4f} +- {np.std(accs):.4f} vs pt3_retry "
              f"{ANCHORS[mask]:.4f}  d={d:+.4f}  "
              f"{'~noise' if abs(d) < 0.02 else 'MISMATCH'}", flush=True)


def part_tune(led, arms, masks):
    """Per VARIANT, never shared: pt3_retry showed the masked arm's optimum sits five decades from
    the unmasked one's, and reusing a grid across arms was worth 0.22 accuracy."""
    print(f"TUNE lr on VAL, PER VARIANT — grid {LR_GRID}\n", flush=True)
    for arm in arms:
        for mask in masks:
            scores = {lr: run_cell(led, arm, mask, lr, TUNE_SEED, split="val") for lr in LR_GRID}
            best = max(scores, key=lambda k: scores[k])
            span = max(scores.values()) - min(scores.values())
            edge = "  !! GRID EDGE — extend" if best in (LR_GRID[0], LR_GRID[-1]) else ""
            print(f"  >>> {arm:5s} {mask:6s} lr={best:g} (val {scores[best]:.4f}, "
                  f"span {span:.4f}){edge}", flush=True)


def tuned_lr(led, arm, mask):
    rows = where(led.rows(), arm=arm, mask=mask, split="val", seed=TUNE_SEED)
    if not rows:
        raise KeyError(f"no val rows for ({arm}, {mask}) — run --part tune first")
    return float(max(rows, key=lambda r: float(r["acc"]))["lr"])


def part_test(led, arms, masks):
    for arm in arms:
        for mask in masks:
            lr = tuned_lr(led, arm, mask)
            for s in SEEDS:
                run_cell(led, arm, mask, lr, s)


def part_report(led, arms):
    rows = led.rows()
    print("\n" + "=" * 92)
    print("PAST-ONLY OUTPUT MASKING   |   class-IL / Adam   |   each variant at its OWN tuned lr")
    print("=" * 92)
    for arm in arms:
        print(f"\n--- {arm} ---")
        print(f"  {'mask':8s} {'lr':>8s} {'acc':>9s} {'sd':>8s} {'d-none':>9s} {'pos':>5s} "
              f"{'forget':>8s} {'allowed':>9s}")
        ref = None
        for mask in MASKS:
            try:
                lr = tuned_lr(led, arm, mask)
            except KeyError:
                continue
            a = _accs(rows, arm, mask, lr)
            if not a:
                continue
            if mask == "none":
                ref = a
            ex = _one(rows, arm, mask, lr)
            d = [x - y for x, y in zip(a, ref)] if ref else [0.0]
            flag = " " if abs(np.mean(d)) >= NOISE_FLOOR else "~"
            print(f"  {mask:8s} {lr:>8g} {np.mean(a):>9.4f} {np.std(a):>8.4f} "
                  f"{np.mean(d):>+9.4f}{flag}{sum(x > 0 for x in d)}/{len(d):<3d} "
                  f"{float(ex['forget']):>8.4f} {float(ex['n_allowed']):>9.2f}")
    print("\n  PREDICTIONS (pre-registered): `past` ~= `curr` and `future` ~= `none`. If both hold,")
    print("  the mechanism is exactly 'stop pushing OLD logits down' and nothing else. `past` is a")
    print("  PROGRESSIVE mask (= `none` at task 0, = `curr` at the last task), so it cannot beat")
    print("  `curr` by more than the early-task slack. `allowed` is the mean number of competing")
    print("  logits per sample and shows the interpolation directly: 10 for none, 2 for curr.")
    print("  Arms sit at DIFFERENT tuned lrs — read each column against its own `none`, never")
    print("  across arms (pt3_retry: that error was worth 0.22).")


def _accs(rows, arm, mask, lr):
    sel = where(rows, arm=arm, mask=mask, lr=f"{lr:g}", split="test")
    return [float(r["acc"]) for r in sorted(sel, key=lambda r: int(r["seed"]))]


def _one(rows, arm, mask, lr):
    return where(rows, arm=arm, mask=mask, lr=f"{lr:g}", split="test")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "anchor", "tune", "test", "report"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--mask", default=None, help="comma filter (a good shard axis)")
    a = ap.parse_args()
    arms = tuple(a.arm.split(",")) if a.arm else ARMS
    masks = tuple(a.mask.split(",")) if a.mask else MASKS
    led = ledger()
    print(f"past-only masking | device {DEV} | arms {arms} | masks {masks}\nledger {TSV}\n",
          flush=True)
    if a.part in ("all", "anchor"):
        part_anchor(led)
    if a.part in ("all", "tune"):
        part_tune(led, arms, masks)
    if a.part in ("all", "test"):
        part_test(led, arms, masks)
    if a.part in ("all", "report"):
        part_report(led, arms)


if __name__ == "__main__":
    main()
