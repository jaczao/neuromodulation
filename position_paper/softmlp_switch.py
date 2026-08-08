"""SOFT_MLP WITH A TRUE -> INFERRED TASK-ID SWITCH — THESIS-PLAN B, own variant.

User-requested. pt6's `soft_mlp` trains its gate table through the TRUE task id (`train_gate` ->
`table.oracle(tids)`) and only meets the inference net at EVAL. That train/eval mismatch is the
proposal's target: train on true ids while the selector is still bad, then switch to the INFERRED id
for the LAST task, so the gate table gets to learn under the routing it will actually face.

WHAT MAKES THIS WORTH RUNNING RATHER THAN A RESTATEMENT OF pt6. The binding law from pt3-iter8, pt6
and pt8 is `pred ~= oracle x infer`: a task-DIFFERENTIATED gate has ~zero tolerance for misrouting,
because a wrong task's row is actively wrong rather than merely unhelpful. Training on true ids is
what MAXIMISES that differentiation — each row gets a clean unmixed gradient — so it is also what
makes misrouting maximally expensive. Training under inferred ids should trade differentiation for
robustness. The question is whether the PRODUCT improves, and it decomposes cleanly:

    oracle   should FALL   (rows get mixed gradients, so they differentiate less)
    infer    unchanged     (the selector's own training is untouched)
    pred     ???           (the whole point: does robustness beat differentiation?)

Reporting all three is what makes this readable rather than a single number. pt6-followup-D2 already
ran the extreme — making the TRAIN driver the soft posterior for the WHOLE run — and it LOST
(er-own null, buf-own -0.103, oracle down in every cell), with the stated cause being that a blend
smears every sample across all rows so they differentiate less. This study asks whether a LATE,
PARTIAL switch avoids that: by task 4 the selector is at ~0.88, so the mixing is mostly correct.

VARIANTS (`--switch`):
    true       true ids throughout                          = pt6's soft_mlp, the reference
    last       true ids for tasks 0..T-2, INFERRED on the last task   = the requested mechanism
    half       switch at the midpoint, to test whether "when" matters or only "whether"
    always     inferred ids throughout                      = the pt6-followup-D2 extreme, as a bound

INFERRED means the soft posterior blend (`table.blend(posterior)`), not a hard argmax: the gate is
already a linear function of the row, so the blend is the differentiable, information-preserving
form and is what eval uses in `soft` mode. Hard routing is available as a resolution mode at eval.

CONTROLS. `dead` freezes the gate table at parity (gamma = 1) while everything else — the selector,
its replay training, the RNG it consumes — runs identically. That is the RNG-matched control (rule
#10) AND, because the gate is what the mechanism modifies, it is also the content-free control this
package now runs by default: two studies in a row found a content-free arm matching every real
driver, so it is built in rather than retrofitted.

ARMS: `erown` (main net + gate on the ER batch) and `bufown` (naive backbone; replay reaches the
gate and the selector only). class-IL, Adam, val-tuned main lr, 3 seeds.

Ledger `softmlp_switch_results.tsv`.
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
from neurocore.task_selection import SoftMLPSelector, routing_ceiling  # noqa: E402
from neurocore.tuned import tuned_main                             # noqa: E402
from prototype.data import SplitMNIST, make_sequence               # noqa: E402

DEV = p7.DEV
PROBLEM, METRIC, OPT = "splitmnist", "classil", "adam"

TSV = shard.ledger_path(Path(__file__).resolve().parent / "softmlp_switch_results.tsv")
KEYS = ["regime", "arm", "switch", "gate", "nlr", "seed", "split"]
METRICS = ["acc", "oracle", "hard", "infer", "forget", "gate_mag", "ceiling"]

SWITCHES = ("true", "last", "half", "always", "last_half", "last_plateau")
# The last two switch PART-WAY THROUGH the final task rather than at its boundary, which is what the
# original proposal actually asked for ("during the last task, when the inference net is good
# enough"). `last` switches at the start of task T-1; these two do not:
#   last_half     switch at the midpoint of the last task's optimisation steps — a fixed schedule.
#   last_plateau  switch when the selector's accuracy STOPS IMPROVING, which is the literal reading
#                 of "good enough". Measured on the training batches themselves (true task ids are
#                 available there for free), as a windowed running accuracy: switch the first time a
#                 window improves on the previous one by less than PLATEAU_EPS.
PLATEAU_WIN, PLATEAU_EPS = 50, 0.005      # fixed, unswept — this is a trigger, not a hyperparameter
GATES = ("live", "dead", "fixp")
# `fixp`: the gate table is FROZEN at a random init instead of learned. The gate is active, the
# selector still trains, but no gradient reaches the table — so it asks whether the switch schedule
# was ever about what the table LEARNS, or only about which rows a sample is routed through.
FIXP_SIGMA = 0.1
ARMS = ("erown", "bufown")
SEEDS = (42, 43, 44)
TUNE_SEED = 42
BUFFERS = {"normal": 1000, "budget": 200, "rfree": 0}
NEURO_GRID = (1e-4, 1e-3, 1e-2)
N_TASKS, GATEDIM = 5, 810                # 810 = h0 400 + h1 400 + out 10 (per-neuron gain)
VAL_SEQ = make_sequence(7)
VAL_FRAC = 0.1

# pt6, seed 42, lr 1e-3 / ep 5 / buffer 1000 — a NOISE BAND, not a bit-exact anchor: soft_mlp gates
# via P[tids], whose backward is an atomic scatter-add and is nondeterministic on MPS (~0.002-0.003
# of final accuracy). pt6's own code re-runs to 0.9885 against its logged 0.9913.
PT6_BAND = {"oracle": 0.9913, "soft": 0.8850}


def _label_to_task(seq):
    m = torch.full((10,), -1, dtype=torch.long)
    for t, pair in enumerate(seq):
        for c in pair:
            m[c] = t
    return m.to(DEV)


def _switch_task(kind):
    """First task index trained under INFERRED ids. N_TASKS => never (or an in-task trigger)."""
    return {"true": N_TASKS, "last": N_TASKS - 1, "half": N_TASKS // 2, "always": 0,
            "last_half": N_TASKS - 1, "last_plateau": N_TASKS - 1}[kind]


def _apply(net, x, gamma):
    """Per-neuron multiplicative gain on (h0, h1, logits) — pt6's gain-neuron target.

    Written out rather than reusing a frozen forward because the gate has to enter it; the split of
    `gamma` into the three layer blocks is the same 810 = 400 + 400 + 10 pt6 used.
    """
    g0, g1, go = gamma[:, :400], gamma[:, 400:800], gamma[:, 800:]
    h0 = F.relu(net.l0(x)) * (1.0 + g0)
    h1 = F.relu(net.l1(h0)) * (1.0 + g1)
    return net.l2(h1) * (1.0 + go)


def run(switch, gate, arm, seed, nlr, main_lr, epochs, buffer, split="test"):
    p7.seed_all(seed)
    seq = VAL_SEQ if split == "val" else p7.SEQ
    ds = SplitMNIST(sequence=seq, val_frac=VAL_FRAC if split == "val" else 0.0)
    loaders = [ds.get_task_loaders(t, 64) for t in range(N_TASKS)]
    evals = [ds.get_task_val_loader(t, 64) if split == "val" else loaders[t][1]
             for t in range(N_TASKS)]
    l2t = _label_to_task(seq)

    net = p7.Net().to(DEV)
    sel = SoftMLPSelector(n_tasks=N_TASKS, dim=GATEDIM).to(DEV)
    live = gate != "dead"
    if gate == "fixp":
        with torch.no_grad():
            sel.table.P.normal_(0.0, FIXP_SIGMA)
    # The gate table sits in the MAIN optimizer when live (it is a forward target, so the main loss
    # trains it); the selector always has its own, because its objective is task-CE, not the task
    # loss. `dead` keeps the table out of every optimizer, so gamma stays at parity but the
    # selector, its replay draws and its RNG consumption are IDENTICAL — that is what makes it the
    # RNG-matched control rather than merely a baseline.
    main_params = list(net.parameters()) + (sel.gate_params() if gate == "live" else [])
    opt = torch.optim.Adam(main_params, main_lr)
    inf_opt = torch.optim.Adam(sel.inf_params(), nlr)
    buf = p7.Reservoir(buffer) if buffer > 0 else None
    sw = _switch_task(switch)
    in_task = switch in ("last_half", "last_plateau")   # trigger fires DURING the last task
    steps_last = len(loaders[N_TASKS - 1][0]) * epochs  # for last_half's midpoint
    step_in_task = 0
    inferred_on = False                                 # latched once the trigger fires
    win_hits = win_n = 0
    prev_win = None
    switch_step = -1                                    # recorded for the report
    A = np.full((N_TASKS, N_TASKS), np.nan)

    for t in range(N_TASKS):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                xf = x.view(x.size(0), -1)
                rep = buf.sample_any(64) if buf is not None else None
                if rep is not None and arm == "erown":
                    Xm = torch.cat([xf, rep[0].to(DEV)]); Ym = torch.cat([y, rep[1].to(DEV)])
                else:
                    Xm, Ym = xf, y
                tids = l2t[Ym]

                use_inferred = (t >= sw) and (inferred_on or not in_task)
                if use_inferred:                   # INFERRED: the soft posterior blend
                    with torch.no_grad():
                        post = sel.inf.posterior(Xm)
                    gamma = sel.table.blend(post)
                else:                              # TRUE ids: each row gets an unmixed gradient
                    gamma = sel.train_gate(Xm, tids)
                if not live:
                    gamma = torch.zeros_like(gamma)

                loss = p7.CE(_apply(net, Xm, gamma), Ym)
                opt.zero_grad(); loss.backward(); opt.step()

                # the selector trains on the buffer in BOTH arms — pt6-followup-B found that with no
                # buffer `infer` collapses to chance (0.198) while the oracle stays at 0.933, i.e.
                # replay is what makes the selector work, not the gate
                sx, sy = (Xm, tids)
                if arm == "bufown" and rep is not None:
                    sx = torch.cat([xf, rep[0].to(DEV)])
                    sy = l2t[torch.cat([y, rep[1].to(DEV)])]
                inf_opt.zero_grad()
                F.cross_entropy(sel.task_logits(sx), sy).backward()
                inf_opt.step()

                if buf is not None:
                    buf.add(x, y)

                # ---- in-task switch triggers (last task only, latched once fired) ----
                if in_task and t == sw and not inferred_on:
                    step_in_task += 1
                    if switch == "last_half":
                        if step_in_task >= steps_last // 2:
                            inferred_on, switch_step = True, step_in_task
                    else:                          # last_plateau
                        with torch.no_grad():
                            win_hits += (sel.task_logits(Xm).argmax(1) == tids).sum().item()
                            win_n += len(tids)
                        if win_n >= PLATEAU_WIN * 64:
                            acc_win = win_hits / win_n
                            if prev_win is not None and acc_win - prev_win < PLATEAU_EPS:
                                inferred_on, switch_step = True, step_in_task
                            prev_win, win_hits, win_n = acc_win, 0, 0
        for i in range(t + 1):
            A[t, i] = _eval(net, sel, evals[i], "soft", live, l2t, i)

    last = N_TASKS - 1
    acc = float(np.nanmean(A[last, :]))
    modes = {m: float(np.mean([_eval(net, sel, evals[i], m, live, l2t, i) for i in range(N_TASKS)]))
             for m in ("oracle", "hard")}
    infer = _infer_acc(sel, evals, l2t)
    if in_task and not inferred_on:
        switch_step = -1                           # trigger never fired: this cell IS `true`
    return dict(acc=acc, oracle=modes["oracle"], hard=modes["hard"], infer=infer,
                switch_step=switch_step,
                forget=float(np.mean([max([A[k, i] for k in range(i, N_TASKS)]) - A[last, i]
                                      for i in range(N_TASKS)])),
                gate_mag=float(sel.table.rows().detach().abs().mean()),
                ceiling=routing_ceiling(modes["oracle"], infer),
                cost=Cost(backbone_params=count_params(net),
                          extra_params=count_params(sel),
                          buffer_bytes=0 if buf is None else
                          buf.X.element_size() * buf.X.nelement()
                          + buf.Y.element_size() * buf.Y.nelement(),
                          fwd_train=2.0, bwd_train=2.0,
                          # the gate IS in the forward here, so unlike wd_modulation this mechanism
                          # costs at inference: one selector forward per test batch
                          fwd_infer=2.0, bwd_infer=0.0))


@torch.no_grad()
def _eval(net, sel, loader, mode, live, l2t, task_i):
    net.eval()
    c = tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        xf = x.view(x.size(0), -1)
        if not live:
            gamma = torch.zeros(xf.size(0), GATEDIM, device=DEV)
        elif mode == "oracle":
            gamma = sel.eval_gate(xf, "oracle",
                                  tids=torch.full((xf.size(0),), task_i, device=DEV))
        else:
            gamma = sel.eval_gate(xf, mode)
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


def run_cell(led, regime, arm, switch, gate, nlr, seed, split="test"):
    key = dict(regime=regime, arm=arm, switch=switch, gate=gate, nlr=f"{nlr:g}", seed=seed,
               split=split)
    if led.is_done(**key):
        return float(where(led.rows(), **key)[0]["acc"])
    base = "er" if arm == "erown" else "naive"
    p = tuned_main(PROBLEM, METRIC, base, OPT)
    r = run(switch, gate, arm, seed, nlr, p["lr"], p["epochs_per_task"], BUFFERS[regime],
            split=split)
    led.append(key, {k: r[k] for k in METRICS}, cost=r["cost"])
    print(f"  {regime:6s} {arm:6s} {switch:6s} {gate:4s} nlr={nlr:<7g} s{seed} {split:4s} "
          f"soft={r['acc']:.4f} oracle={r['oracle']:.4f} hard={r['hard']:.4f} "
          f"infer={r['infer']:.4f} |P|={r['gate_mag']:.4f} sw_step={r['switch_step']}",
          flush=True)
    return r["acc"]


def part_tune(led, arms):
    print(f"TUNE selector lr on VAL — grid {NEURO_GRID}\n", flush=True)
    for arm in arms:
        scores = {n: run_cell(led, "normal", arm, "true", "live", n, TUNE_SEED, split="val")
                  for n in NEURO_GRID}
        best = max(scores, key=lambda k: scores[k])
        span = max(scores.values()) - min(scores.values())
        note = "  !! span < noise floor — UNRESOLVED at 1 seed" if span < NOISE_FLOOR else ""
        print(f"  >>> {arm}: selector lr = {best:g} (val {scores[best]:.4f}, span {span:.4f})"
              f"{note}", flush=True)


def tuned_nlr(led, arm):
    rows = where(led.rows(), arm=arm, switch="true", gate="live", split="val", seed=TUNE_SEED,
                 regime="normal")
    if not rows:
        raise KeyError(f"no val rows for {arm} — run --part tune first")
    return float(max(rows, key=lambda r: float(r["acc"]))["nlr"])


def part_test(led, arms, switches, regime="normal"):
    for arm in arms:
        nlr = tuned_nlr(led, arm)
        for sw in switches:
            for g in GATES:
                for s in SEEDS:
                    run_cell(led, regime, arm, sw, g, nlr, s)


def part_regimes(led, arms, switches):
    for regime in ("budget", "rfree"):
        part_test(led, arms, switches, regime=regime)


def part_report(led, arms, regime="normal"):
    rows = led.rows()
    print("\n" + "=" * 104)
    print(f"SOFT_MLP TRUE -> INFERRED SWITCH   |   class-IL / Adam   |   regime={regime}")
    print(f"pt6 reference (noise band, MPS scatter-add): oracle {PT6_BAND['oracle']:.4f}  "
          f"soft {PT6_BAND['soft']:.4f}")
    print("=" * 104)
    for arm in arms:
        try:
            nlr = tuned_nlr(led, arm)
        except KeyError:
            continue
        print(f"\n--- {arm} ---")
        print(f"  {'switch':8s} {'gate':5s} {'soft':>8s} {'sd':>7s} {'d-dead':>9s} {'pos':>5s} "
              f"{'oracle':>8s} {'hard':>7s} {'infer':>7s} {'o x i':>7s} {'|P|':>7s}")
        for sw in SWITCHES:
            live = _seeded(rows, arm, sw, "live", nlr, regime)
            dead = _seeded(rows, arm, sw, "dead", nlr, regime)
            both = sorted(set(live) & set(dead))
            if not both:
                continue
            d = [live[s] - dead[s] for s in both]
            ex = _one(rows, arm, sw, "live", nlr, regime)
            flag = " " if abs(np.mean(d)) >= NOISE_FLOOR else "~"
            print(f"  {sw:8s} {'live':5s} {np.mean([live[s] for s in sorted(live)]):>8.4f} "
                  f"{np.std([live[s] for s in sorted(live)]):>7.4f} {np.mean(d):>+9.4f}{flag}"
                  f"{sum(x > 0 for x in d)}/{len(d):<3d} {float(ex['oracle']):>8.4f} "
                  f"{float(ex['hard']):>7.4f} {float(ex['infer']):>7.4f} "
                  f"{float(ex['ceiling']):>7.4f} {float(ex['gate_mag']):>7.4f}")
    print(f"\n  ~ = |d-dead| < {NOISE_FLOOR}. `soft` is the oracle-FREE number and the headline.")
    print("  DECOMPOSITION: switching to inferred ids should LOWER `oracle` (rows get mixed")
    print("  gradients) while leaving `infer` alone. Compare `soft` against `o x i` — pt3-iter8/pt6/")
    print("  pt8 all found pred ~= oracle x infer, so a switch only wins if it breaks that law by")
    print("  making the gate TOLERANT of misrouting, not by improving either factor.")


def _seeded(rows, arm, sw, gate, nlr, regime):
    return {int(r["seed"]): float(r["acc"])
            for r in where(rows, arm=arm, switch=sw, gate=gate, nlr=f"{nlr:g}", split="test",
                           regime=regime)}


def _one(rows, arm, sw, gate, nlr, regime):
    return where(rows, arm=arm, switch=sw, gate=gate, nlr=f"{nlr:g}", split="test",
                 regime=regime)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "tune", "test", "regimes", "report"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--switch", default=None, help="comma filter (a good shard axis)")
    a = ap.parse_args()
    arms = tuple(a.arm.split(",")) if a.arm else ARMS
    switches = tuple(a.switch.split(",")) if a.switch else SWITCHES
    led = ledger()
    print(f"soft_mlp switch | device {DEV} | arms {arms} | switches {switches}\nledger {TSV}\n",
          flush=True)
    if a.part in ("all", "tune"):
        part_tune(led, arms)
    if a.part in ("all", "test"):
        part_test(led, arms, switches)
    if a.part == "regimes":
        part_regimes(led, arms, switches)
    if a.part in ("all", "report"):
        part_report(led, arms)


if __name__ == "__main__":
    main()
