"""STUDY 5 — PROPORTIONALLY ALLOCATED SUBNETS. THESIS-PLAN B, own variant.

User-requested. pt5 iter-1 is the one mechanism in this project that ever beat replay by a wide
margin: a FIXED DISJOINT partition of the hidden units into T subnets, gated {0,1} by the task id, so
each task trains its own private subnet and a gated-off unit receives exactly zero gradient — a hard
freeze the backbone cannot absorb. Its result carried one fatal caveat: the task id was supplied AT
EVAL, making it a task-IL-style number on a class-IL metric.

THIS STUDY REMOVES THE ORACLE AND SOFTENS THE ALLOCATION. The gate is still built on the same fixed
disjoint partition, but a sample is routed to subnets IN PROPORTION to pt6's soft_mlp posterior:

    gamma(x)_j = p_{owner(j)}(x)

where `owner(j)` is the subnet that owns hidden unit j. If the selector says (0.4, 0.4, 0.1, 0, 0),
subnets 0 and 1 are scaled by 0.4, subnet 2 by 0.1, and subnets 3-4 are off — exactly the requested
semantics. Under a ONE-HOT posterior this reduces to pt5 iter-1's {0,1} gate identically, which makes
`true` a faithful reproduction of that mechanism and every other variant a departure from it.

WHY IT IS WORTH RUNNING. `pred ~= oracle x infer` has held for hard routing (pt3-iter8, pt6) and for a
soft factorized posterior (pt8), and study 3 showed a task-DIFFERENTIATED gain gate has ~zero
tolerance for misrouting. A disjoint subnet gate is the most differentiated gate possible, so the
naive prediction is that it should be the WORST case for misrouting. The counter-argument, and the
reason to measure rather than assume: proportional allocation means a misrouted sample still puts
mass on the correct subnet, so error degrades the gate smoothly instead of selecting an actively
wrong one. Which effect wins is the question.

THE TRAINING SCHEDULE (study-3 fashion, as requested), so each task can learn to USE several subnets
rather than only its own:

    true          true-id one-hot throughout                  = pt5 iter-1, reproduced
    last          posterior for the whole last task
    last_half     posterior from the midpoint of the last task
    last_plateau  posterior once the selector's accuracy stops improving
    always        posterior throughout                        = the bound

Study 3 found these worse the earlier they fire, but that was for a LEARNED gain table whose rows
were being smeared. Here the subnets are FIXED and disjoint, so the same schedule is asking a
different question: not "do the rows differentiate less" but "does each task learn to spread across
subnets it does not own".

WHAT DIFFERS FROM pt5 ITER-1, STATED PLAINLY. (1) The gate is on the HIDDEN layers only (h0, h1),
never the output head: pt5 found that a label-aligned output gate IS task-IL masking, which would
re-introduce exactly the confound this study exists to remove. (2) Soft allocation kills the hard
freeze — a unit scaled by p gets gradient proportional to p, not zero — so the property that made
iter-1 work is deliberately traded away in the non-`true` variants. `true` keeps it.

CONTROLS. `dead` sets gamma == 1 (no subnets at all) while the selector, its replay training and all
of its RNG run identically — the RNG-matched control, which here is also the "no mechanism" baseline
since the partition is fixed rather than learned. Eval reports `soft` (oracle-free, the headline),
`oracle` (true id — pt5 iter-1's protocol, diagnostic only) and `hard` (argmax routing).

class-IL, Adam at the val-tuned ER point, buffer 1000, er-own, 3 seeds. The selector lr is REUSED
from `softmlp_switch` (1e-4) rather than re-swept — same selector, same arm, same operating point.

Ledger `subnet_soft_results.tsv`.
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
from neurocore.projections import build_disjoint_proj              # noqa: E402
from neurocore.task_selection import TaskInferenceNet, routing_ceiling  # noqa: E402
from neurocore.tuned import tuned_main                             # noqa: E402
from prototype.data import SplitMNIST, make_sequence               # noqa: E402

DEV = p7.DEV
PROBLEM, METRIC, BASE, OPT = "splitmnist", "classil", "er", "adam"

TSV = shard.ledger_path(Path(__file__).resolve().parent / "subnet_soft_results.tsv")
KEYS = ["regime", "switch", "gate", "nlr", "seed", "split"]
METRICS = ["acc", "oracle", "hard", "infer", "forget", "ceiling", "gate_mean", "switch_step"]

SWITCHES = ("true", "last", "last_half", "last_plateau", "always")
GATES = ("live", "dead")
SEEDS = (42, 43, 44)
BUFFERS = {"normal": 1000, "budget": 200, "rfree": 0}
SEL_LR = 1e-4                     # reused from softmlp_switch's val sweep (same selector and arm)
N_TASKS, H0, H1 = 5, 400, 400
PLATEAU_WIN, PLATEAU_EPS = 50, 0.005
VAL_SEQ = make_sequence(7)
VAL_FRAC = 0.1
ANCHOR_ER = 0.9019                # plain ER at this operating point (wd_modulation harness)


def _label_to_task(seq):
    m = torch.full((10,), -1, dtype=torch.long)
    for t, pair in enumerate(seq):
        for c in pair:
            m[c] = t
    return m.to(DEV)


def _owner(seed):
    """Unit -> subnet map for the 800 hidden units, from the SAME balanced disjoint partition
    primitive pt5 iter-1 used. Fixed, never learned: this study varies the ROUTING, not the
    partition."""
    P = build_disjoint_proj(N_TASKS, H0 + H1, seed)     # (T, 800) one-hot columns
    return P.argmax(0).to(DEV)


def _switch_task(kind):
    return {"true": N_TASKS, "last": N_TASKS - 1, "last_half": N_TASKS - 1,
            "last_plateau": N_TASKS - 1, "always": 0}[kind]


def _apply(net, x, gamma):
    """gamma is (B, 800): a per-sample scale for every hidden unit. Head left ungated on purpose."""
    g0, g1 = gamma[:, :H0], gamma[:, H0:]
    h0 = F.relu(net.l0(x)) * g0
    h1 = F.relu(net.l1(h0)) * g1
    return net.l2(h1)


def _gamma_from_post(post, owner):
    """gamma_j = p_{owner(j)} — subnet t scaled by the posterior mass on task t."""
    return post[:, owner]


def _gamma_onehot(tids, owner):
    """The one-hot limit: gamma_j = 1 iff owner(j) == the sample's task. == pt5 iter-1's {0,1} gate."""
    return (owner.unsqueeze(0) == tids.unsqueeze(1)).float()


def run(switch, gate, seed, nlr=SEL_LR, main_lr=3e-4, epochs=5, buffer=1000, split="test"):
    p7.seed_all(seed)
    seq = VAL_SEQ if split == "val" else p7.SEQ
    ds = SplitMNIST(sequence=seq, val_frac=VAL_FRAC if split == "val" else 0.0)
    loaders = [ds.get_task_loaders(t, 64) for t in range(N_TASKS)]
    evals = [ds.get_task_val_loader(t, 64) if split == "val" else loaders[t][1]
             for t in range(N_TASKS)]
    l2t = _label_to_task(seq)
    owner = _owner(seed)

    net = p7.Net().to(DEV)
    sel = TaskInferenceNet(n_tasks=N_TASKS).to(DEV)
    opt = torch.optim.Adam(net.parameters(), main_lr)
    sel_opt = torch.optim.Adam(sel.params(), nlr)
    buf = p7.Reservoir(buffer) if buffer > 0 else None
    live = gate == "live"
    sw = _switch_task(switch)
    in_task = switch in ("last_half", "last_plateau")
    steps_last = len(loaders[N_TASKS - 1][0]) * epochs
    step_in_task = 0; inferred_on = False; win_hits = win_n = 0; prev_win = None; switch_step = -1
    A = np.full((N_TASKS, N_TASKS), np.nan)
    gmag = []

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

                if not live:
                    gamma = torch.ones(Xm.size(0), H0 + H1, device=DEV)
                elif (t >= sw) and (inferred_on or not in_task):
                    with torch.no_grad():
                        gamma = _gamma_from_post(sel.posterior(Xm), owner)
                else:
                    gamma = _gamma_onehot(tids, owner)

                opt.zero_grad()
                F.cross_entropy(_apply(net, Xm, gamma), Ym).backward()
                opt.step()
                gmag.append(float(gamma.mean()))

                sel_opt.zero_grad()                    # selector: replay-trained task-CE (pt6)
                F.cross_entropy(sel.task_logits(Xm), tids).backward()
                sel_opt.step()
                if buf is not None:
                    buf.add(x, y)

                if in_task and t == sw and not inferred_on:
                    step_in_task += 1
                    if switch == "last_half":
                        if step_in_task >= steps_last // 2:
                            inferred_on, switch_step = True, step_in_task
                    else:
                        with torch.no_grad():
                            win_hits += (sel.task_logits(Xm).argmax(1) == tids).sum().item()
                            win_n += len(tids)
                        if win_n >= PLATEAU_WIN * 64:
                            aw = win_hits / win_n
                            if prev_win is not None and aw - prev_win < PLATEAU_EPS:
                                inferred_on, switch_step = True, step_in_task
                            prev_win, win_hits, win_n = aw, 0, 0
        for i in range(t + 1):
            A[t, i] = _eval(net, sel, evals[i], owner, "soft", live, i)

    last = N_TASKS - 1
    acc = float(np.nanmean(A[last, :]))
    modes = {m: float(np.mean([_eval(net, sel, evals[i], owner, m, live, i)
                               for i in range(N_TASKS)])) for m in ("oracle", "hard")}
    infer = _infer_acc(sel, evals, l2t)
    return dict(acc=acc, oracle=modes["oracle"], hard=modes["hard"], infer=infer,
                forget=float(np.mean([max([A[k, i] for k in range(i, N_TASKS)]) - A[last, i]
                                      for i in range(N_TASKS)])),
                ceiling=routing_ceiling(modes["oracle"], infer),
                gate_mean=float(np.mean(gmag)), switch_step=switch_step,
                cost=Cost(backbone_params=count_params(net), extra_params=count_params(sel),
                          buffer_bytes=0 if buf is None else
                          buf.X.element_size() * buf.X.nelement()
                          + buf.Y.element_size() * buf.Y.nelement(),
                          fwd_train=2.0, bwd_train=2.0,
                          # the gate IS in the forward, so inference pays for the selector too
                          fwd_infer=2.0, bwd_infer=0.0))


@torch.no_grad()
def _eval(net, sel, loader, owner, mode, live, task_i):
    net.eval()
    c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        xf = x.view(x.size(0), -1)
        if not live:
            gamma = torch.ones(xf.size(0), H0 + H1, device=DEV)
        elif mode == "oracle":
            gamma = _gamma_onehot(torch.full((xf.size(0),), task_i, device=DEV), owner)
        elif mode == "hard":
            gamma = _gamma_onehot(sel.task_logits(xf).argmax(1), owner)
        else:
            gamma = _gamma_from_post(sel.posterior(xf), owner)
        c += (_apply(net, xf, gamma).argmax(1) == y).sum().item(); tot += len(y)
    net.train()
    return c / tot


@torch.no_grad()
def _infer_acc(sel, evals, l2t):
    c = tot = 0
    for ld in evals:
        for x, y in ld:
            x, y = x.to(DEV), y.to(DEV)
            c += (sel.task_logits(x.view(x.size(0), -1)).argmax(1) == l2t[y]).sum().item()
            tot += len(y)
    return c / tot


# ========================================================================================== driving
def ledger():
    return Ledger(TSV, keys=KEYS, metrics=METRICS, with_cost=True)


def run_cell(led, regime, switch, gate, seed, split="test"):
    key = dict(regime=regime, switch=switch, gate=gate, nlr=f"{SEL_LR:g}", seed=seed, split=split)
    if led.is_done(**key):
        return float(where(led.rows(), **key)[0]["acc"])
    p = tuned_main(PROBLEM, METRIC, BASE, OPT)
    r = run(switch, gate, seed, main_lr=p["lr"], epochs=p["epochs_per_task"],
            buffer=BUFFERS[regime], split=split)
    led.append(key, {k: r[k] for k in METRICS}, cost=r["cost"])
    print(f"  {regime:6s} {switch:12s} {gate:4s} s{seed} soft={r['acc']:.4f} "
          f"oracle={r['oracle']:.4f} hard={r['hard']:.4f} infer={r['infer']:.4f} "
          f"gate={r['gate_mean']:.3f} sw_step={r['switch_step']}", flush=True)
    return r["acc"]


def part_test(led, switches, regime="normal"):
    for sw in switches:
        for g in GATES:
            for s in SEEDS:
                run_cell(led, regime, sw, g, s)


def part_report(led, regime="normal"):
    rows = led.rows()
    print("\n" + "=" * 104)
    print(f"SUBNETS ALLOCATED BY THE SOFT_MLP POSTERIOR   |   class-IL / ER / Adam   |  {regime}")
    print(f"pt5 iter-1 reference (ORACLE task id at eval): er+gain 0.9901 / standalone 0.9949")
    print("=" * 104)
    print(f"  {'switch':13s} {'soft':>8s} {'sd':>7s} {'d-dead':>9s} {'pos':>5s} {'oracle':>8s} "
          f"{'hard':>7s} {'infer':>7s} {'o x i':>7s} {'gate':>6s}")
    for sw in SWITCHES:
        live = _seeded(rows, sw, "live", regime)
        dead = _seeded(rows, sw, "dead", regime)
        both = sorted(set(live) & set(dead))
        if not both:
            continue
        d = [live[s] - dead[s] for s in both]
        a = [live[s] for s in sorted(live)]
        ex = _one(rows, sw, "live", regime)
        flag = " " if abs(np.mean(d)) >= NOISE_FLOOR else "~"
        print(f"  {sw:13s} {np.mean(a):>8.4f} {np.std(a):>7.4f} {np.mean(d):>+9.4f}{flag}"
              f"{sum(x > 0 for x in d)}/{len(d):<3d} {float(ex['oracle']):>8.4f} "
              f"{float(ex['hard']):>7.4f} {float(ex['infer']):>7.4f} "
              f"{float(ex['ceiling']):>7.4f} {float(ex['gate_mean']):>6.3f}")
    print(f"\n  ~ = |d-dead| < {NOISE_FLOOR}. `soft` is oracle-FREE and is the headline; `oracle` is")
    print("  pt5 iter-1's protocol and is DIAGNOSTIC only. Compare `soft` to `o x i`: the routing law")
    print("  says a task-conditioned gate has ~zero misrouting tolerance, so soft ~= oracle x infer.")
    print("  Proportional allocation is the reason it might not — a misrouted sample still puts mass")
    print("  on the right subnet. `gate` is the mean gamma: 0.2 under one-hot routing (1 of 5 subnets")
    print("  on), 1.0 for the dead control, and in between when the posterior spreads.")


def _seeded(rows, sw, gate, regime):
    return {int(r["seed"]): float(r["acc"])
            for r in where(rows, switch=sw, gate=gate, split="test", regime=regime)}


def _one(rows, sw, gate, regime):
    return where(rows, switch=sw, gate=gate, split="test", regime=regime)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "test", "regimes", "report"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--switch", default=None, help="comma filter (the shard axis)")
    ap.add_argument("--regime", default="normal")
    a = ap.parse_args()
    switches = tuple(a.switch.split(",")) if a.switch else SWITCHES
    led = ledger()
    print(f"subnet_soft | device {DEV} | switches {switches}\nledger {TSV}\n", flush=True)
    if a.part in ("all", "test"):
        part_test(led, switches)
    if a.part == "regimes":
        part_test(led, switches, regime="budget")
    if a.part in ("all", "report"):
        part_report(led, regime=a.regime)


if __name__ == "__main__":
    main()
