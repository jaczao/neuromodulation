"""Discriminator (a): train-set vs test-set task-inference acc for the soft_mlp selector,
small (784->128->T) vs big (784->400->400->T), replay arms {buf-own, er-own}, both optimizers.

Reading: if the big net's TRAIN infer stays ~ small's but TEST drops -> generalization gap
(overfitting). If big's TRAIN infer is ALSO lower -> under-training/optimization (fixed Adam
lr 1e-3 + 5-epoch budget can't fit the deeper net). Same operating point as the size sweep
(class-IL, seed 42, lr 1e-3, ep 5, buffer 1000).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pt6_synapse as p6s                       # noqa: E402
import pt6_softmlp_bigsel as bs                 # noqa: E402  (build + SoftMLPSynBig)

DEV = p6s.DEV
LEDGER = HERE / "pt6_softmlp_traintest_results.tsv"


@torch.no_grad()
def infer_train_test(mech, net):
    """Task-inference acc over each task's TRAIN split and TEST split (true task id = i)."""
    net.eval()
    tr_c = tr_n = te_c = te_n = 0
    for i in range(5):
        for x, y in LOADERS[i][0]:                       # train split
            x = x.to(DEV)
            tr_c += (mech.task_logits(x).argmax(1) == i).sum().item(); tr_n += x.size(0)
        for x, y in LOADERS[i][1]:                       # test split
            x = x.to(DEV)
            te_c += (mech.task_logits(x).argmax(1) == i).sum().item(); te_n += x.size(0)
    return tr_c / tr_n, te_c / te_n


LOADERS = None


def run(arm, big, opt_kind):
    global LOADERS
    loaders, net, mech = bs.build(big)
    LOADERS = loaders
    p6s.train(mech, net, loaders, arm, opt_kind, lr=bs.LR, epochs=bs.EP, buffer=bs.BUF)
    tr, te = infer_train_test(mech, net)
    return tr, te


def main():
    done = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                done.add(line.split("\t")[0])
    print(f"device={DEV}  soft_mlp selector train-vs-test infer  seed=42 lr={bs.LR:g} ep={bs.EP}\n", flush=True)
    print(f"{'arm':8s} {'sel':6s} {'opt':4s} | {'train':>7s} {'test':>7s} {'gap':>7s}", flush=True)
    for opt in ("sgd", "adam"):
        for arm in ("buf-own", "er-own"):
            for big in (False, True):
                sel = "big" if big else "small"
                tag = f"{arm}|{sel}|{opt}"
                if tag in done:
                    print(f"{arm:8s} {sel:6s} {opt:4s} | (cached)", flush=True)
                    continue
                tr, te = run(arm, big, opt)
                with LEDGER.open("a") as f:
                    f.write(f"{tag}\t{tr:.4f}\t{te:.4f}\t{tr - te:.4f}\n")
                print(f"{arm:8s} {sel:6s} {opt:4s} | {tr:7.4f} {te:7.4f} {tr - te:7.4f}", flush=True)


if __name__ == "__main__":
    main()
