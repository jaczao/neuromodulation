"""ER replay buffer stored as uint8 instead of float32 — normal and memory-budgeted regimes.

NO NEUROMODULATION anywhere. This is a COST-COLUMN study, not a mechanism study: it asks what the
`buffer_bytes` column buys, given that the bytes can be cut 4x for free.

THE CODEC IS BIT-EXACT LOSSLESS, and that is what shapes the design. MNIST is uint8 upstream and the
transform is affine (ToTensor -> /255, Normalize -> (x-mu)/sigma), so
    round((x*sigma + mu)*255) -> uint8 -> ((k/255) - mu)/sigma
reproduces the stored float32 BIT-FOR-BIT (verified on the training set: torch.equal True). So
"does quantisation cost accuracy?" is not a 6-run experiment, it is an assertion plus a 2-run
integration check (`u8count`), and the only open question is what the saved bytes BUY (`u8bytes`).

ARMS (class-IL, Adam at the val-tuned ER point, no neuromodulation):
    fp32     the reference.            cap 1000 (normal) / 200 (budget)
    u8bytes  same BYTES, 4x samples.   cap 3969 (normal) / 793 (budget)
    u8count  same SAMPLES, 1/4 bytes.  cap 1000 / 200 — PARITY CHECK, 1 seed.

`u8count` must come back BIT-IDENTICAL to `fp32`: the reservoir's RNG consumption is unchanged
(`random.randint` per evicted sample, `torch.randint` per draw) and the codec draws none, so at a
matched cap the only difference is a lossless round trip. That is the end-to-end proof the codec is
wired correctly — a mismatch there means the integration is wrong, not that uint8 costs accuracy.
`u8bytes` is deliberately NOT RNG-matched to fp32 (a different cap changes the reservoir draws);
capacity IS the mechanism there, so the comparison is the honest one.

rfree is EXCLUDED, not forgotten: ER at buffer 0 is naive, a degeneracy check rather than a
rehearsal-free result (rule #12, user-set).

METRIC CONVENTION: MACRO average of per-task accuracies, matching results/pt7_tuned_syn.run_baseline
(POOLED would bias this study's deltas — see CLAUDE.md). Forgetting follows prototype/train.py:
mean over ALL tasks of (max over t>=i of A[t,i]) - A[T-1,i], so the last task contributes 0.

ANCHOR: fp32/normal/seed42 must reproduce the frozen results/pt7_tuned_syn_results.tsv ER-adam cell
at 0.897549. The mid-training eval matrix needed for forgetting is wrapped in `rng_frozen()` —
iterating a DataLoader draws from the global torch RNG even with shuffle=False, so without the guard
the added evals would silently move the run off its reference trajectory.

Run (one path for EVERY part, so the ledger cannot mix devices):
    uv run python -m neurocore.shard --script uint8_buffer/buffer_dtype.py \
        --ledger uint8_buffer/buffer_dtype_results.tsv --split arms=fp32,u8bytes,u8count \
        --args "--part test --resume" --workers 3 --device mps
    uv run python uint8_buffer/buffer_dtype.py --part report
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))
sys.path.insert(0, str(ROOT / "prototype"))

import pt7_neuromodulators as p7                                   # noqa: E402
from data import MNIST_MEAN, MNIST_STD                             # noqa: E402
from neurocore import shard                                        # noqa: E402
from neurocore.cost import Cost, count_params                      # noqa: E402
from neurocore.ledger import Ledger, NOISE_FLOOR, paired_delta, summarize, where  # noqa: E402
from neurocore.tuned import tuned_main                             # noqa: E402
from neurocore.utils import rng_frozen                             # noqa: E402

DEV = p7.DEV
CE = nn.CrossEntropyLoss()
SEEDS = [42, 43, 44]
OPT = "adam"
TSV = shard.ledger_path(Path(__file__).resolve().parent / "buffer_dtype_results.tsv")

# The frozen ER-adam class-IL cell (results/pt7_tuned_syn_results.tsv), macro, seed 42, buffer 1000.
ANCHOR = 0.897549
ANCHOR_TOL = 1e-6                       # ledger stores %.6f, so compare at the precision it was written

# The two regimes rule #12 asks for that are non-degenerate for ER. `budget` = 1/5 the bytes.
FP32_CAP = {"normal": 1000, "budget": 200}
REGIMES = list(FP32_CAP)
ARMS = ["fp32", "u8bytes", "u8count"]
PARITY_SEEDS = [42]                     # u8count is bit-exact by construction; 1 seed proves wiring


# ------------------------------- byte accounting -------------------------------
def _bytes(cap, x_itemsize):
    """Resident bytes of a cap-sample buffer: X (784 elements) + Y (int64 label)."""
    return cap * 784 * x_itemsize + cap * 8


def u8_cap_at_bytes(budget):
    """Largest uint8 cap fitting `budget`. Not exactly 4x: int64 labels stay 8 B/sample."""
    return budget // (784 + 8)


def cap_for(arm, regime):
    base = FP32_CAP[regime]
    if arm == "u8bytes":
        return u8_cap_at_bytes(_bytes(base, 4))
    return base                                   # fp32 and u8count both run at the reference cap


# ------------------------------- uint8 reservoir -------------------------------
def encode(x):
    """Normalised float32 -> the original MNIST byte. Exact: the transform is affine over k/255."""
    return torch.round((x * MNIST_STD + MNIST_MEAN) * 255).clamp_(0, 255).to(torch.uint8)


def decode(k):
    """Byte -> normalised float32, in the SAME op order as ToTensor+Normalize, so it round-trips
    bit-exact rather than merely close."""
    return ((k.float() / 255) - MNIST_MEAN) / MNIST_STD


class Uint8Reservoir:
    """p7.Reservoir with X stored as uint8.

    RNG consumption is IDENTICAL to the float32 reservoir — one `random.randint` per evicted sample,
    one `torch.randint` per draw, and the codec draws none — so at a matched cap this reproduces the
    float32 arm bit-exact. Buffer discipline is unchanged: update before the gradient step, sample
    with replacement, reservoir sampling for the fill.
    """

    def __init__(self, cap):
        self.cap = cap
        self.X = torch.zeros(cap, 784, dtype=torch.uint8)
        self.Y = torch.zeros(cap, dtype=torch.long)
        self.n = 0
        self.filled = 0

    def add(self, x, y):
        x = encode(x.view(x.size(0), -1).cpu())    # encode the batch once; no RNG either way
        y = y.cpu()
        for i in range(len(x)):
            if self.filled < self.cap:
                self.X[self.filled] = x[i]; self.Y[self.filled] = y[i]; self.filled += 1
            else:
                j = random.randint(0, self.n)
                if j < self.cap:
                    self.X[j] = x[i]; self.Y[j] = y[i]
            self.n += 1

    def sample_any(self, b):
        if self.filled == 0:
            return None
        idx = torch.randint(0, self.filled, (b,))
        return decode(self.X[idx]), self.Y[idx]

    def nbytes(self):
        return self.X.element_size() * self.X.numel() + self.Y.element_size() * self.Y.numel()


def buf_bytes(buf):
    """p7.Reservoir (frozen, rule #9) has no nbytes(), so size both kinds here."""
    if hasattr(buf, "nbytes"):
        return buf.nbytes()
    return buf.X.element_size() * buf.X.numel() + buf.Y.element_size() * buf.Y.numel()


# ------------------------------- the cell -------------------------------
def build_loaders():
    """[(train_loader, test_loader)] x5 at the reporting (default) order, full train set."""
    ds = p7.SplitMNIST(sequence=p7.SEQ, val_frac=0.0)
    return [ds.get_task_loaders(t, 64) for t in range(5)]


def run_cell(arm, cap, lr, epochs, loaders, seed):
    """Copy-forward of results/pt7_tuned_syn.run_baseline's `er` branch, with the buffer class and
    capacity parameterised and an RNG-neutral eval matrix added for forgetting."""
    p7.seed_all(seed)
    net = p7.Net().to(DEV)
    opt = p7._opt(OPT, net.parameters(), lr)
    buf = (p7.Reservoir if arm == "fp32" else Uint8Reservoir)(cap)
    A = np.zeros((5, 5))
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                loss = CE(net.plain(torch.cat(Xs))[0], torch.cat(Ys))
                opt.zero_grad(); loss.backward(); opt.step()
                buf.add(x, y)
        with rng_frozen():                       # iterating a DataLoader WRITES to the torch RNG
            for i in range(5):
                A[t, i] = p7._acc_plain(net, loaders[i][1])
        net.train()
    acc = float(np.mean(A[4]))                                            # MACRO, per run_baseline
    forget = float(np.mean([max(A[t, i] for t in range(i, 5)) - A[4, i] for i in range(5)]))
    cost = Cost(backbone_params=count_params(net), extra_params=0, buffer_bytes=buf_bytes(buf),
                fwd_train=1.0, bwd_train=1.0, fwd_infer=1.0, bwd_infer=0.0)
    return acc, forget, cost


# ------------------------------- driver -------------------------------
def part_test(args, led):
    point = tuned_main("splitmnist", "classil", "er", OPT)
    lr, epochs = point["lr"], point["epochs_per_task"]
    loaders = build_loaders()
    print(f"[point] classil/er/{OPT}: lr={lr} ep={epochs}  (neurocore.tuned, val-selected)")
    for arm in args.arms:
        seeds = PARITY_SEEDS if arm == "u8count" else args.seeds
        for regime in args.regimes:
            cap = cap_for(arm, regime)
            for seed in seeds:
                key = dict(regime=regime, arm=arm, cap=cap, seed=seed)
                if args.resume and led.is_done(**key):
                    print(f"  skip {key}"); continue
                acc, forget, cost = run_cell(arm, cap, lr, epochs, loaders, seed)
                led.append(key, dict(acc=acc, forget=forget), cost=cost)
                print(f"  {regime:<7s} {arm:<8s} cap={cap:<5d} seed={seed}  "
                      f"acc={acc:.6f} forget={forget:.4f} bytes={cost.buffer_bytes:,}")


def part_report(args, led):
    rows = led.rows()
    if not rows:
        print("empty ledger — run --part test first"); return

    a = where(rows, regime="normal", arm="fp32", seed=42)
    if a:
        d = abs(float(a[0]["acc"]) - ANCHOR)
        print(f"ANCHOR fp32/normal/seed42: {float(a[0]['acc']):.6f} vs frozen pt7_tuned_syn "
              f"{ANCHOR:.6f} -> {'OK' if d <= ANCHOR_TOL else 'MISMATCH'} (|d|={d:.2e})\n")

    print("PARITY (u8count vs fp32 at a matched cap — must be bit-identical):")
    ok = True
    for regime in REGIMES:
        for seed in PARITY_SEEDS:
            f = where(rows, regime=regime, arm="fp32", seed=seed)
            u = where(rows, regime=regime, arm="u8count", seed=seed)
            if not (f and u):
                continue
            d = abs(float(f[0]["acc"]) - float(u[0]["acc"]))
            ok &= d <= ANCHOR_TOL
            print(f"  {regime:<7s} seed={seed}  fp32 {float(f[0]['acc']):.6f}  "
                  f"u8count {float(u[0]['acc']):.6f}  |d|={d:.2e}  "
                  f"{'BIT-EXACT' if d <= ANCHOR_TOL else 'MISMATCH -> codec/integration bug'}"
                  f"   bytes {int(f[0]['buffer_bytes']):,} -> {int(u[0]['buffer_bytes']):,}")
    print(f"  => {'codec verified' if ok else 'CODEC FAILED'}\n")

    for regime in REGIMES:
        sel = where(rows, regime=regime)
        if not sel:
            continue
        print(f"{regime.upper()}  (fp32 cap {FP32_CAP[regime]}, "
              f"u8bytes cap {cap_for('u8bytes', regime)})")
        stats = summarize(sel, metric="acc", group_by=("arm",))
        print(f"  {'arm':<9s}{'cap':>6s}{'acc':>10s}{'sd':>9s}{'n':>3s}{'forget':>9s}"
              f"{'bytes':>12s}{'d-fp32':>11s}")
        for arm in ARMS:
            if (arm,) not in stats:
                continue
            m, s, n = stats[(arm,)]
            r0 = where(sel, arm=arm)[0]
            fg = float(np.mean([float(r["forget"]) for r in where(sel, arm=arm)]))
            d = paired_delta(sel, arm, "fp32", metric="acc")
            dcol = "-" if (arm == "fp32" or d is None) else \
                   f"{d[0]:+.4f}{' ' if abs(d[0]) >= NOISE_FLOOR else '~'}"
            print(f"  {arm:<9s}{int(r0['cap']):>6d}{m:>10.4f}{s:>9.4f}{n:>3d}{fg:>9.4f}"
                  f"{int(r0['buffer_bytes']):>12,d}{dcol:>11s}")
        print(f"  (~ = |delta| < {NOISE_FLOOR} noise floor: read as null)\n")

    print("rfree: EXCLUDED as structurally degenerate — ER at buffer 0 IS naive, a degeneracy check\n"
          "       rather than a rehearsal-free result (rule #12). A rehearsal-free row needs a\n"
          "       rehearsal-free base (EWC/SI/MAS/LwF, DGR ~0.91), which is a different study.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="test", choices=["test", "report"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    args.regimes = [r for r in args.regimes.split(",") if r]
    args.seeds = [int(s) for s in args.seeds.split(",") if s]

    led = Ledger(TSV, keys=["regime", "arm", "cap", "seed"], metrics=["acc", "forget"],
                 with_cost=True)
    print(f"ledger: {TSV}  device: {DEV}")
    (part_test if args.part == "test" else part_report)(args, led)


if __name__ == "__main__":
    main()
