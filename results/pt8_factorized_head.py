"""FACTORIZED TASK-INFERENCE HEAD (user-requested; no SPEC-proto-pt8 exists — the `pt8_`
prefix is a file-naming label only, to keep the results/ convention).

MECHANISM (one net, two heads off h1; no gating anywhere):
  trunk  x(784) -> h0=relu(l0 x) -> h1=relu(l1 h0)                        [784-400-400]
  head T:  task_logits = task(h1)          (5-way)  -- TRAINS THE TRUNK
  head C:  cls_logits  = cls(h1.detach())  (10-way) -- trains ONLY itself (no backprop to trunk)

  PREDICTION is factorized, P(class) = P(task) * P(class | task):
      q      = softmax(task_logits)                          (B,5), sums to 1
      p_pair = softmax(cls_logits.view(B,5,2), dim=2)        (B,5,2), each pair sums to 1
      P(2t+j) = q[:,t] * p_pair[:,t,j]                       (B,10), sums to 1  -> argmax
  i.e. the 10-class head is FIVE independent 2-way classifiers (one per task) whose outputs
  are softly weighted by the task-inference posterior. Oracle-free at eval (no task id).

LOSSES (per batch, both on the SAME samples; each sample's task = label // 2):
  L_task = CE(task_logits, task_id)          -> trunk + task head
  L_cls  = masked CE on cls_logits           -> cls head only (h1 detached)
           (masked = restricted to the true task's 2 logits, i.e. the within-pair 2-way CE)
  The trunk is therefore shaped ONLY by task discrimination; the class head is a linear probe
  on those features. Whether within-pair (0-vs-1) information survives features trained for a
  5-way between-pair objective is the empirical question.

ARMS: `naive` (current task only) and `er` (+ reservoir replay, 64 samples/step, both losses
over current+replay). The pt6 follow-up (B) result says the selector NEEDS replay or task
inference collapses to chance, so `naive` is expected to fail; it is the control.

SETUP: class-IL Split MNIST, Adam lr=3e-4, epochs/task=5, buffer 1000, batch 64 — the
val-tuned class-IL ER operating point (configs.TUNED_MAIN[("classil","er","adam")]).
3 seeds {42,43,44}. Baselines naive / er (plain 784-400-400-10 net) from pt6's harness at the
SAME point. Caveat: the mechanism is a different architecture from the baselines, so runs are
not RNG-matched (cf. the pt7_tuned_neuro RNG-matching gotcha) — hence 3 seeds + std.

REPORTED per run:  pred (headline, factorized argmax) | oracle (true task -> 2-way, upper
bound on the class head) | raw (plain 10-way argmax of cls_logits, no task scaling) | infer
(task-inference acc) | forget (mean over tasks of max-minus-final on `pred`).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pt6_driver_mechanisms as p6  # noqa: E402  (SEQ, DEV, CE, Reservoir, masked_ce, seed_all, train_baseline)

DEV = p6.DEV
LEDGER = HERE / "pt8_factorized_head_results.tsv"
LR, EP, BUF, BATCH = 3e-4, 5, 1000, 64
SEEDS = (42, 43, 44)


class FactorNet(nn.Module):
    """Shared trunk + task head (trains the trunk) + class head (gradient-isolated)."""

    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(784, 400)
        self.l1 = nn.Linear(400, 400)
        self.task = nn.Linear(400, 5)
        self.cls = nn.Linear(400, 10)

    def trunk(self, x):
        x = x.view(x.size(0), -1)
        return F.relu(self.l1(F.relu(self.l0(x))))

    def heads(self, x):
        h1 = self.trunk(x)
        return self.task(h1), self.cls(h1.detach())     # detach == "won't backprop to the rest of the net"


def factor_logits(task_logits, cls_logits):
    """log P(class) = log softmax(task) + log softmax(pair). Row t, offset j -> class 2t+j."""
    logq = F.log_softmax(task_logits, dim=1)                            # (B,5)
    logp = F.log_softmax(cls_logits.view(-1, 5, 2), dim=2)              # (B,5,2)
    return (logp + logq.unsqueeze(2)).reshape(-1, 10)                   # (B,10), exp sums to 1


def train(arm, seed, lr=LR, epochs=EP, buffer=BUF):
    p6.seed_all(seed)
    ds = p6.SplitMNIST(sequence=p6.SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=BATCH) for t in range(5)]
    net = FactorNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buf = p6.Reservoir(buffer)
    accs = np.zeros((5, 5))                                             # accs[after_task, eval_task] on `pred`

    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                if arm == "er":
                    r = buf.sample_any(BATCH)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                Tm = torch.div(Ym, 2, rounding_mode="floor")            # true task of each sample
                tl, cl = net.heads(Xm)
                loss = p6.CE(tl, Tm) + p6.masked_ce(cl, Ym)             # disjoint parameter sets
                opt.zero_grad(); loss.backward(); opt.step()
                if arm == "er":
                    buf.add(x, y)
        for i in range(5):
            accs[t, i] = evaluate(net, loaders[i][1], i)["pred"]
        net.train()
    return net, loaders, accs


@torch.no_grad()
def evaluate(net, loader, true_task):
    net.eval()
    c_pred = c_or = c_raw = c_inf = n = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        tl, cl = net.heads(x)
        c_pred += (factor_logits(tl, cl).argmax(1) == y).sum().item()
        pair = cl.view(-1, 5, 2)[:, true_task, :]                       # oracle: true task's 2 logits
        c_or += ((2 * true_task + pair.argmax(1)) == y).sum().item()
        c_raw += (cl.argmax(1) == y).sum().item()                       # no task scaling
        c_inf += (tl.argmax(1) == true_task).sum().item()
        n += len(y)
    return dict(pred=c_pred / n, oracle=c_or / n, raw=c_raw / n, infer=c_inf / n)


def run_cell(arm, seed):
    net, loaders, accs = train(arm, seed)
    res = [evaluate(net, loaders[i][1], i) for i in range(5)]
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    out["forget"] = float(np.mean([accs[:, i].max() - accs[4, i] for i in range(5)]))
    out["traj"] = "/".join(f"{accs[t].mean():.3f}" for t in range(5))
    return out


def load_ledger():
    done = {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                f = line.split("\t")
                done[f[0]] = f[1:]
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("all", "baselines", "mech"))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    done = load_ledger()
    print(f"device={DEV}  factorized task-inference head  class-IL  adam lr={LR:g} ep={EP} "
          f"buffer={BUF}  seeds={seeds}\n", flush=True)

    if args.part in ("all", "baselines"):
        print("baselines (plain 784-400-400-10, same point):", flush=True)
        for m in ("naive", "er"):
            vals = []
            for s in seeds:
                tag = f"base|{m}|{s}"
                if tag in done:
                    a = float(done[tag][0])
                else:
                    a = p6.train_baseline(m, "adam", lr=LR, epochs=EP, buffer=BUF, seed=s)
                    with LEDGER.open("a") as f:
                        f.write(f"{tag}\t{a:.4f}\n")
                vals.append(a)
                print(f"  {m:5s} seed{s}: {a:.4f}", flush=True)
            print(f"  {m:5s} MEAN  : {np.mean(vals):.4f} +/- {np.std(vals):.4f}\n", flush=True)

    if args.part in ("all", "mech"):
        print(f"{'arm':6s} {'seed':5s} | {'pred':>7s} {'oracle':>7s} {'raw':>7s} {'infer':>7s} "
              f"{'forget':>7s} | traj(avg acc after each task)", flush=True)
        for arm in ("naive", "er"):
            preds = []
            for s in seeds:
                tag = f"mech|{arm}|{s}"
                if tag in done:
                    v = done[tag]
                    o = dict(pred=float(v[0]), oracle=float(v[1]), raw=float(v[2]),
                             infer=float(v[3]), forget=float(v[4]), traj=v[5])
                else:
                    o = run_cell(arm, s)
                    with LEDGER.open("a") as f:
                        f.write(f"{tag}\t{o['pred']:.4f}\t{o['oracle']:.4f}\t{o['raw']:.4f}\t"
                                f"{o['infer']:.4f}\t{o['forget']:.4f}\t{o['traj']}\n")
                preds.append(o["pred"])
                print(f"{arm:6s} {s:<5d} | {o['pred']:7.4f} {o['oracle']:7.4f} {o['raw']:7.4f} "
                      f"{o['infer']:7.4f} {o['forget']:7.4f} | {o['traj']}", flush=True)
            print(f"{arm:6s} MEAN  | {np.mean(preds):7.4f} +/- {np.std(preds):.4f}\n", flush=True)


if __name__ == "__main__":
    main()
