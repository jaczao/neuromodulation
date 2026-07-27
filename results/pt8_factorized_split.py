"""FACTORIZED PREDICTION, variants 2 & 3 (user-requested follow-up to pt8_factorized_head.py).

Same factorized prediction as variant 1 -- P(2t+j) = softmax(task)[t] * softmax(pair_t)[j] --
but the class path is no longer a gradient-isolated probe on a task-trained trunk:

  v1 `shared`  (pt8_factorized_head.py, already run): ONE trunk, task head trains it, class
                 head sits on h1.detach(). Class path cannot shape its own features.
  v2 `split`   : TWO independent nets. main 784-400-400-10 trained END-TO-END by the masked
                 CE (full backprop, no detach); tinf 784-400-400-5 trained by task CE.
  v3 `expert<W>`: FIVE per-task nets 784-W-W-2 (one per task pair) + the same tinf net. Expert
                 j is trained ONLY on task-j samples, so the class path is STRUCTURALLY immune
                 to forgetting -- everything then rides on task inference.

Implementation note (v3): the 5 experts' 2-logit outputs are concatenated to a (B,10) vector
whose columns 2j,2j+1 come from expert j, and the SAME per-sample masked CE is applied. That
is exactly equivalent to per-expert 2-way CE with correct batch-mean weighting: masked_ce adds
-inf to the non-true pair, so softmax gives those logits zero probability and hence exactly
zero gradient -- only the owning expert is updated by each sample.

PARAMETER COUNTS (class path + tinf), verified by construction:
  split      478,410 + 476,405 =  954,815
  expert100  5 x 88,802 = 444,010 + 476,405 =  920,415   (class path ~ parameter-matched to split's main net)
  expert50   5 x 41,902 = 209,510 + 476,405 =  685,915

ARMS: `naive` (current task only) and `er` (reservoir 1000, 64/step; replay feeds BOTH the
class path and tinf). In v3's naive arm expert j receives no gradient at all once task j ends,
so its 2-way accuracy is frozen by construction.

SETUP: class-IL Split MNIST, Adam lr=3e-4, ep/task=5, buffer 1000, batch 64 (the val-tuned
class-IL ER point), 3 seeds {42,43,44}. Baselines live in pt8_factorized_head_results.tsv
(naive 0.5430+/-0.0247, er 0.9029+/-0.0042).

REPORTED: pred (class-IL headline) | oracle (task-IL: 2-way within the true task) | raw (plain
10-way argmax, no task scaling) | infer (task-inference acc) | f-cls, f-task (forgetting on the
class-IL and task-IL metrics -- BOTH trajectories are tracked here, which v1 did not do).
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
import pt6_driver_mechanisms as p6      # noqa: E402  (SEQ, DEV, CE, Reservoir, masked_ce, seed_all)
from pt8_factorized_head import factor_logits  # noqa: E402

DEV = p6.DEV
LEDGER = HERE / "pt8_factorized_split_results.tsv"
LR, EP, BUF, BATCH = 3e-4, 5, 1000, 64
SEEDS = (42, 43, 44)


def mlp(win, w, wout):
    return nn.Sequential(nn.Linear(win, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, wout))


class Split(nn.Module):
    """v2: one full-size main net, trained end-to-end by the masked CE."""

    def __init__(self):
        super().__init__()
        self.main = mlp(784, 400, 10)

    def cls(self, x):
        return self.main(x.view(x.size(0), -1))


class Experts(nn.Module):
    """v3: one net per task; column 2j,2j+1 of the output belongs to expert j."""

    def __init__(self, w):
        super().__init__()
        self.ex = nn.ModuleList([mlp(784, w, 2) for _ in range(5)])

    def cls(self, x):
        x = x.view(x.size(0), -1)
        return torch.cat([e(x) for e in self.ex], dim=1)


def build_cls(variant):
    if variant == "split":
        return Split()
    if variant.startswith("expert"):
        return Experts(int(variant[6:]))
    raise ValueError(variant)


def train(variant, arm, seed, lr=LR, epochs=EP, buffer=BUF):
    p6.seed_all(seed)
    ds = p6.SplitMNIST(sequence=p6.SEQ)
    loaders = [ds.get_task_loaders(t, batch_size=BATCH) for t in range(5)]
    clsnet = build_cls(variant).to(DEV)
    tinf = mlp(784, 400, 5).to(DEV)                      # task-inference net, always full size
    opt = torch.optim.Adam(list(clsnet.parameters()) + list(tinf.parameters()), lr=lr)
    buf = p6.Reservoir(buffer)
    a_cls = np.zeros((5, 5))                             # trajectory on the class-IL metric
    a_task = np.zeros((5, 5))                            # trajectory on the task-IL metric

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
                Tm = torch.div(Ym, 2, rounding_mode="floor")
                loss = p6.CE(tinf(Xm), Tm) + p6.masked_ce(clsnet.cls(Xm), Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                if arm == "er":
                    buf.add(x, y)
        for i in range(5):
            m = evaluate(clsnet, tinf, loaders[i][1], i)
            a_cls[t, i], a_task[t, i] = m["pred"], m["oracle"]
        clsnet.train(); tinf.train()
    return clsnet, tinf, loaders, a_cls, a_task


@torch.no_grad()
def evaluate(clsnet, tinf, loader, true_task):
    clsnet.eval(); tinf.eval()
    c_pred = c_or = c_raw = c_inf = n = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        cl, tl = clsnet.cls(x), tinf(x.view(x.size(0), -1))
        c_pred += (factor_logits(tl, cl).argmax(1) == y).sum().item()
        pair = cl.view(-1, 5, 2)[:, true_task, :]
        c_or += ((2 * true_task + pair.argmax(1)) == y).sum().item()
        c_raw += (cl.argmax(1) == y).sum().item()
        c_inf += (tl.argmax(1) == true_task).sum().item()
        n += len(y)
    return dict(pred=c_pred / n, oracle=c_or / n, raw=c_raw / n, infer=c_inf / n)


def _forget(a):
    return float(np.mean([a[:, i].max() - a[4, i] for i in range(5)]))


def run_cell(variant, arm, seed):
    clsnet, tinf, loaders, a_cls, a_task = train(variant, arm, seed)
    res = [evaluate(clsnet, tinf, loaders[i][1], i) for i in range(5)]
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    out["f_cls"], out["f_task"] = _forget(a_cls), _forget(a_task)
    out["traj"] = "/".join(f"{a_cls[t].mean():.3f}" for t in range(5))
    return out


KEYS = ("pred", "oracle", "raw", "infer", "f_cls", "f_task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="split,expert100,expert50")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    variants = args.variants.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    done = {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                f = line.split("\t"); done[f[0]] = f[1:]

    print(f"device={DEV}  factorized split / per-task experts  class-IL  adam lr={LR:g} "
          f"ep={EP} buffer={BUF}  seeds={seeds}", flush=True)
    print("baselines (pt8_factorized_head): naive 0.5430+/-0.0247   er 0.9029+/-0.0042\n", flush=True)
    hdr = f"{'variant':10s} {'arm':6s} {'seed':5s} |" + "".join(f"{k:>8s}" for k in KEYS) + " | traj"
    print(hdr, flush=True)
    for variant in variants:
        for arm in ("naive", "er"):
            acc = {k: [] for k in KEYS}
            for s in seeds:
                tag = f"{variant}|{arm}|{s}"
                if tag in done:
                    v = done[tag]
                    o = {k: float(v[i]) for i, k in enumerate(KEYS)}; o["traj"] = v[len(KEYS)]
                else:
                    o = run_cell(variant, arm, s)
                    with LEDGER.open("a") as f:
                        f.write(tag + "".join(f"\t{o[k]:.4f}" for k in KEYS) + f"\t{o['traj']}\n")
                for k in KEYS:
                    acc[k].append(o[k])
                print(f"{variant:10s} {arm:6s} {s:<5d} |" + "".join(f"{o[k]:8.4f}" for k in KEYS)
                      + f" | {o['traj']}", flush=True)
            print(f"{variant:10s} {arm:6s} MEAN  |"
                  + "".join(f"{np.mean(acc[k]):8.4f}" for k in KEYS)
                  + f" | pred std {np.std(acc['pred']):.4f}\n", flush=True)


if __name__ == "__main__":
    main()
