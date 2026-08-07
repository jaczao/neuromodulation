"""TEST-SET ORDER SENSITIVITY of the stateful / GRU gating cells (user-requested).

QUESTION. Every frozen study evaluates on the test set in ONE fixed order: task blocks 0->4 in sequence
order, and WITHIN a block the raw MNIST test-file order (`get_task_loaders` builds its test loader with
`shuffle=False`). That order is not neutral -- `results/pt7_driver_traces` measured, on raw pixels with no
model, that mean ||x|| RISES monotonically within every task block (Spearman rho +0.40..+0.76, mean +0.64).
A cell whose gate depends on anything beyond the individual sample is therefore being read along a
systematic ink ramp that resets five times.

This study re-evaluates every ORDER-SENSITIVE cell of four studies under a FULLY SHUFFLED test set (all
10 000 test images pooled, shuffled, batched into 64 -- so no task blocks and each batch is a random mix of
all 10 classes), from the SAME trained weights, with NO new tuning. Nothing about training changes.

WHAT MAKES A CELL ORDER-SENSITIVE. Two independent mechanisms, both requiring the gate to be in the
FORWARD pass (a plasticity-target gate multiplies gradients and is immune by construction):
  (a) PERSISTENT STATE advancing on the test stream -- running EMAs / standardisation stats (`update=True`,
      pt7_stateful's `running` mode, signalnet_traces' `live` mode) and a GRU hidden that is written at
      eval (`GRUOnVec.forward` DEFAULTS to `update_state=True`, so this fires in cells labelled "frozen").
  (b) BATCH-STATISTIC dependence with no state at all -- both GRU wrappers compute their candidate hidden
      from the BATCH MEAN (`cell(p.mean(0, keepdim=True), hidden)`), so the gate for sample i depends on
      which other samples share its batch even when nothing is written back.
NOTE (b) is why `pt7_stateful`'s `frozen|gru1` cells are order-sensitive: in frozen mode `update_state` is
False so the hidden does NOT advance, but `h_new` is still recomputed from each batch's mean. Persistence
is sufficient but not necessary.

DESIGN. One training run per trained net; the test pass is then run 2x (ordered, shuffled) -- and 4x for
pt7_stateful, which crosses order with its own frozen/running axis. The ORDERED pass is the anchor: it must
reproduce the frozen ledger bit-exact, which is what licenses reading the shuffled number as a real effect
rather than a porting artefact.

METRIC. The frozen studies all report POOLED accuracy (`c/tot` over all 10 000 test samples) while the
project's primary CL metric and the frozen BASELINES macro-average the five per-task accuracies -- a
systematic ~0.0015 gap (see CLAUDE.md). A fully shuffled test set destroys the task blocks and so gives
pooled by construction. Both are reported here: `pooled` is what anchors against the frozen ledgers, and
`macro` is recovered under BOTH orders by binning each sample by its own label (class c -> task c//2), so
the two orders stay comparable under either statistic.

FREE CORRECTNESS CONTROL. `nerisez|gru0|frozen` (MLP predictor, stats frozen) is per-sample by
construction, so its accuracy MUST come out bit-identical under a shuffle. It is run and reported for
exactly that reason -- if it moves, the shuffled loader is wrong. The same holds for any cell whose
|g| is identically 0.

NOT RE-RUN HERE: case 4 (the `true` diagnostic column of `results/pt7_neuromodulators.py`, order-sensitive
via `DA`'s within-batch `ell.std()`), which the user scoped out of this run.

Device MPS is mandatory for the anchors -- every frozen ledger here was produced on MPS and device changes
numerics (CLAUDE.md: the same ER cell reads 0.7337 on MPS and 0.7284 on CPU).

Run: uv run python test_order/test_order.py --part {stateful,traces,signalnet,capacity,all} [--resume]
     uv run python test_order/test_order.py --part table
"""
import argparse
import copy
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "results"))
import pt7_neuromodulators as p7                                     # noqa: E402
import pt7_signalnet as psn                                          # noqa: E402
import pt7_stateful as pst                                           # noqa: E402
sys.path.insert(0, str(REPO / "signalnet_capacity"))
import signalnet_capacity as snc                                     # noqa: E402
sys.path.insert(0, str(REPO / "prototype"))
from data import SplitMNIST                                          # noqa: E402

DEV = p7.DEV
CE = nn.CrossEntropyLoss()
TSV = HERE / "test_order_results.tsv"

SHUF_SEED = 1234                    # fixes the shuffled test permutation across all cells
LR, EPOCHS, BUFFER = 1e-3, 5, 1000
LABEL2TASK = {c: t for t, pair in enumerate(p7.SEQ) for c in pair}

# ---- anchors: the ORDERED pass must reproduce these (pooled), from the frozen ledgers ----
ANCHOR = {
    # results/pt7_stateful_results.tsv
    "stateful": None,                                                # read from the ledger file
    # results/pt7_signalnet_results.tsv (`eng` rows)
    "gru-all4|neuron|er-own|adam|eng": 0.8789,
    "signalnet-gru|neuron|K4|std1|er-own|adam|eng": 0.8657,
    "signalnet-gru|neuron|K16|std1|er-own|adam|eng": 0.8799,
    # driver_traces/signalnet_traces.md
    "sn|predH|frozen": 0.5215, "sn|actualH|frozen": 0.7137,
    "sngru|predH|frozen": 0.8657, "sngru|actualH|frozen": 0.8845,
    "sn|predH|live": 0.6023, "sn|actualH|live": 0.3447,
    "sngru|predH|live": 0.7141, "sngru|actualH|live": 0.8521,
}


# ------------------------------- RNG hygiene -------------------------------
class rng_frozen:
    """Snapshot/restore torch+numpy+random around a block.

    A DataLoader iterator draws a `_base_seed` from the default generator whenever `loader.generator is
    None` -- REGARDLESS of shuffle/sampler/workers -- so merely ITERATING an extra loader is a write to the
    global RNG stream (the trap that cost `pt5_taskil/plast_drivers.py` an anchor failure). Every eval pass
    added by this study is wrapped, so the added passes cannot move any cell off its reference trajectory.
    """

    def __enter__(self):
        import numpy as np
        self.s = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        return self

    def __exit__(self, *a):
        import numpy as np
        torch.set_rng_state(self.s[0]); np.random.set_state(self.s[1]); random.setstate(self.s[2])
        return False


# ------------------------------- the two test orders -------------------------------
def ordered_streams(loaders):
    """The frozen protocol: five task blocks in sequence order, each in raw MNIST test-file order."""
    return [loaders[i][1] for i in range(5)]


def shuffled_streams(loaders, seed=SHUF_SEED):
    """One stream over ALL 10 000 test images, shuffled, batched into 64.

    No task blocks; each batch is a random mix of all 10 classes. The generator is explicit so the
    permutation is fixed across cells and draws nothing from the global stream.
    """
    ds = ConcatDataset([loaders[i][1].dataset for i in range(5)])
    g = torch.Generator().manual_seed(seed)
    return [DataLoader(ds, batch_size=64, shuffle=True, generator=g)]


class Score:
    """Pooled + macro accuracy from one pass, valid under ANY sample order.

    macro bins by the sample's own label (class c -> task c//2), which is what makes the two orders
    comparable under the project's primary metric even though a shuffled pass has no task blocks.
    """

    def __init__(self):
        self.c = self.tot = 0
        self.pc = [0] * 5
        self.pt = [0] * 5
        self.mags = {"h0": 0.0, "h1": 0.0, "out": 0.0}

    def add(self, pred, y, pl=None, b=None):
        ok = (pred == y)
        self.c += ok.sum().item(); self.tot += y.numel()
        for cls, hit in zip(y.tolist(), ok.tolist()):
            t = LABEL2TASK[cls]
            self.pt[t] += 1; self.pc[t] += int(hit)
        if pl is not None:
            for k in self.mags:
                self.mags[k] += pl[k] * (b if b is not None else y.numel())

    def out(self):
        pooled = self.c / self.tot
        macro = sum(c / t for c, t in zip(self.pc, self.pt)) / 5
        return {"pooled": pooled, "macro": macro,
                **{k: v / self.tot for k, v in self.mags.items()}}


# ------------------------------- pt7_stateful -------------------------------
def train_stateful(mech, gru, arm, opt_kind, seed=42, epochs=EPOCHS, log=print):
    """Copy-forward of `pt7_stateful.run_stateful`'s TRAINING half (eval split out so one trained net can
    serve all four eval combos). `eval_mode` never entered training, so this is the same net the frozen
    study trained twice."""
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV)
    drv = pst.StatefulDriver(mech, gru).to(DEV)
    gate = p7.NeuronGate(1, None).to(DEV)
    opt = p7._opt(opt_kind, list(net.parameters()) + gate.params(), LR)
    head_opt = torch.optim.Adam(drv.parameters(), LR)
    buf = p7.Reservoir(BUFFER) if arm == "er-own" else None
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                if arm == "er-own":
                    Xs, Ys = [x.view(x.size(0), -1)], [y]
                    r = buf.sample_any(64)
                    if r is not None:
                        Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                    Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                else:
                    Xm, Ym = x.view(x.size(0), -1), y
                m = drv.driver(Xm, update_state=True, update_stats=(mech == "ach")).detach()
                logits = gate(net, m, Xm)
                loss = CE(logits, Ym) if arm == "er-own" else p7.masked_ce(logits, Ym)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    Hact = p7.entropy(net.plain(Xm)[0]).unsqueeze(1)
                if mech == "nerisez":
                    drv.upd_actual(Hact)
                hloss = F.mse_loss(drv.predictH(Xm, update_state=False), Hact)
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                if arm == "er-own":
                    buf.add(x, y)
        log(f"      task {t} done")
    return net, drv, gate, loaders


def snap_stateful(drv):
    return {"hidden": (drv.hidden.clone() if drv.gru else None), "emaH": drv.emaH,
            "varH": drv.varH, "rm": (None if drv.rm is None else drv.rm.clone()),
            "rv": (None if drv.rv is None else drv.rv.clone())}


def restore_stateful(drv, s):
    if drv.gru:
        drv.hidden = s["hidden"].clone()
    drv.emaH = s["emaH"]; drv.varH = s["varH"]
    drv.rm = None if s["rm"] is None else s["rm"].clone()
    drv.rv = None if s["rv"] is None else s["rv"].clone()


@torch.no_grad()
def eval_stateful(net, drv, gate, streams, upd):
    """Copy-forward of `run_stateful`'s eval half, generalised over the stream list."""
    net.eval(); sc = Score()
    for stream in streams:
        for x, y in stream:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            m = drv.driver(x.view(b, -1), update_state=upd, update_stats=upd)
            pred = gate(net, m, x).argmax(1)
            sc.add(pred, y, gate.per_layer_mag(m), b)
    return sc.out()


def part_stateful(done, log):
    """All 24 cells x 2 orders, from 12 trainings (frozen/running share a net).

    The 4 `nerisez|gru0|frozen` cells are the order-IMMUNE control: per-sample by construction, so their
    shuffled pooled accuracy must be bit-identical.
    """
    ledger_ref = {ln.split("\t")[0]: float(ln.split("\t")[1])
                  for ln in (REPO / "results" / "pt7_stateful_results.tsv").read_text().splitlines()
                  if ln.strip()}
    for mech, gru in pst.MECHS:
        for arm in ("er-own", "nobuf"):
            for opt in ("sgd", "adam"):
                base = f"{mech}|gru{int(gru)}|{arm}|{opt}"
                tags = [f"{base}|{em}|{od}" for em in ("frozen", "running")
                        for od in ("ordered", "shuffled")]
                if all(t in done for t in tags):
                    continue
                log(f"  training {base} ...")
                net, drv, gate, loaders = train_stateful(mech, gru, arm, opt, log=log)
                pristine = snap_stateful(drv)
                with rng_frozen():
                    streams = {"ordered": ordered_streams(loaders),
                               "shuffled": shuffled_streams(loaders)}
                    for em, upd in (("frozen", False), ("running", True)):
                        res = {}
                        for od in ("ordered", "shuffled"):
                            restore_stateful(drv, pristine)      # every pass from the same state
                            res[od] = eval_stateful(net, drv, gate, streams[od], upd)
                        ref = ledger_ref.get(f"{mech}|gru{int(gru)}|{em}|{arm}|{opt}")
                        emit(f"{base}|{em}", res, ref, log)
    return


# ------------------------------- signalnet family (shared) -------------------------------
def snap_sn(feat, gru):
    return (copy.deepcopy(feat), (gru.hidden.clone() if gru is not None else None))


def restore_sn(snap, gru):
    feat = copy.deepcopy(snap[0])
    if gru is not None:
        gru.hidden = snap[1].clone()
    return feat


@torch.no_grad()
def eval_signalnet(net, gate, feat, snet, gru, fheads, streams, update):
    """Copy-forward of `psn._eval_signalnet` / `signalnet_traces.evaluate`, generalised over streams."""
    net.eval(); sc = Score()
    for stream in streams:
        for x, y in stream:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            f = feat.build(net, fheads, x, y=None, update=update)
            code = snet(f)
            m = gru(code) if gru else code
            pred = gate(net, m, x).argmax(1)
            sc.add(pred, y, gate.per_layer_mag(m), b)
    return sc.out()


def train_signalnet(use_gru, engage, actual_h, K, seed=42, epochs=EPOCHS, log=print):
    """Copy-forward of `pt7_signalnet.run_signalnet` (gran=neuron, std1, adam). Construction order is
    byte-identical to the frozen study (Net -> gate -> SignalHeads -> SignalFeatures -> SignalNet ->
    GRUOnVec), which is what makes the anchors reproduce; the shim holds no parameters."""
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    net = p7.Net().to(DEV)
    gate = psn._mk("neuron", K)
    heads = psn.SignalHeads().to(DEV)
    feat = psn.SignalFeatures(True)
    snet = psn.SignalNet(K, engage=engage).to(DEV)
    gru = psn.GRUOnVec(K, K, engage=engage).to(DEV) if use_gru else None
    fheads = snc.ActualHShim(heads, net) if actual_h else heads
    params = list(net.parameters()) + gate.params() + list(snet.parameters()) \
        + (list(gru.parameters()) if gru else [])
    main_opt = p7._opt("adam", params, LR)
    head_opt = torch.optim.Adam(heads.parameters(), LR)
    buf = p7.Reservoir(BUFFER)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                f = feat.build(net, fheads, Xm, Ym, update=True).detach()
                code = snet(f)
                m = gru(code) if gru else code
                loss = CE(gate(net, m, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                T = feat.targets(net, Xm, Ym)
                hloss = F.mse_loss(heads(Xm), T)                 # the RAW head, never the shim
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)
        log(f"      task {t} done")
    return net, gate, heads, feat, snet, gru, fheads, loaders


# ------------------------------- gru-all4 -------------------------------
@torch.no_grad()
def eval_gru_all4(net, gate, heads, gru, streams):
    """Copy-forward of `psn._eval_gru`, generalised over streams."""
    net.eval(); sc = Score()
    for stream in streams:
        for x, y in stream:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            a = heads(x.view(b, -1))
            m = gru(a)
            pred = gate(net, m, x).argmax(1)
            sc.add(pred, y, gate.per_layer_mag(m), b)
    return sc.out()


def train_gru_all4(seed=42, epochs=EPOCHS, log=print):
    """Copy-forward of `pt7_signalnet.run_gru_all4` (gran=neuron, engage=True, adam)."""
    p7.seed_all(seed)
    ds = SplitMNIST(sequence=p7.SEQ)
    loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
    drivers = ["DA", "ACh", "NE", "5HT"]; K = 4
    net = p7.Net().to(DEV); gate = psn._mk("neuron", K)
    heads = p7.Heads(K).to(DEV); sig = p7.Signals(drivers, standardize=True)
    gru = psn.GRUOnVec(K, K, engage=True).to(DEV)
    main_opt = p7._opt("adam", list(net.parameters()) + gate.params() + list(gru.parameters()), LR)
    head_opt = torch.optim.Adam(heads.parameters(), LR)
    buf = p7.Reservoir(BUFFER)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                Xs, Ys = [x.view(x.size(0), -1)], [y]
                r = buf.sample_any(64)
                if r is not None:
                    Xs.append(r[0].to(DEV)); Ys.append(r[1].to(DEV))
                Xm, Ym = torch.cat(Xs), torch.cat(Ys)
                a = heads(Xm).detach()
                m = gru(a)
                loss = CE(gate(net, m, Xm), Ym)
                main_opt.zero_grad(); loss.backward(); main_opt.step()
                T = sig.targets(net, Xm, Ym)
                hloss = F.mse_loss(heads(Xm), T)
                head_opt.zero_grad(); hloss.backward(); head_opt.step()
                buf.add(x, y)
        log(f"      task {t} done")
    return net, gate, heads, gru, loaders


def part_signalnet(done, log):
    """The 3 ENGAGED GRU cells of results/pt7_signalnet.py. The 6 non-`eng` GRU rows are skipped: they log
    |g| = 0.0000 (the double-zero-init saddle), so gamma == 1 exactly and the advancing hidden state cannot
    reach the forward -- provably order-immune, nothing to measure."""
    cells = [("gru-all4|neuron|er-own|adam|eng", None),
             ("signalnet-gru|neuron|K4|std1|er-own|adam|eng", 4),
             ("signalnet-gru|neuron|K16|std1|er-own|adam|eng", 16)]
    for cid, K in cells:
        if all(f"{cid}|{od}" in done for od in ("ordered", "shuffled")):
            continue
        log(f"  training {cid} ...")
        if K is None:
            net, gate, heads, gru, loaders = train_gru_all4(log=log)
            hidden0 = gru.hidden.clone()
            with rng_frozen():
                streams = {"ordered": ordered_streams(loaders),
                           "shuffled": shuffled_streams(loaders)}
                res = {}
                for od in ("ordered", "shuffled"):
                    gru.hidden = hidden0.clone()
                    res[od] = eval_gru_all4(net, gate, heads, gru, streams[od])
        else:
            net, gate, heads, feat, snet, gru, fheads, loaders = \
                train_signalnet(True, True, False, K, log=log)
            snap = snap_sn(feat, gru)
            with rng_frozen():
                streams = {"ordered": ordered_streams(loaders),
                           "shuffled": shuffled_streams(loaders)}
                res = {}
                for od in ("ordered", "shuffled"):
                    feat = restore_sn(snap, gru)
                    res[od] = eval_signalnet(net, gate, feat, snet, gru, fheads, streams[od], False)
        emit(cid, res, ANCHOR.get(cid), log)


def part_traces(done, log):
    """The 4 driver_traces/signalnet_traces.py cells x {frozen, live} x {ordered, shuffled} = 16 evals,
    loaded from the checkpoints that study already wrote -- NO retraining."""
    for kind in ("sn", "sngru"):
        for actual_h in (True, False):
            cid = f"{kind}|{'actualH' if actual_h else 'predH'}"
            if all(f"{cid}|{em}|{od}" in done for em in ("frozen", "live")
                   for od in ("ordered", "shuffled")):
                continue
            ck = REPO / "driver_traces" / \
                f"ckpt_sn_{kind}_{'actualH' if actual_h else 'predH'}.pt"
            if not ck.exists():
                log(f"  SKIP {cid}: no checkpoint {ck.name}")
                continue
            log(f"  loading {cid} from {ck.name} ...")
            use_gru = kind == "sngru"
            K = 4
            d = torch.load(ck, weights_only=False, map_location=DEV)
            p7.seed_all(42)                                   # only to build modules; weights overwritten
            ds = SplitMNIST(sequence=p7.SEQ)
            loaders = [ds.get_task_loaders(t, 64) for t in range(5)]
            net = p7.Net().to(DEV); net.load_state_dict(d["net"])
            gate = psn._mk("neuron", K); gate.load_state_dict(d["gate"])
            heads = psn.SignalHeads().to(DEV); heads.load_state_dict(d["heads"])
            snet = psn.SignalNet(K, engage=True).to(DEV); snet.load_state_dict(d["snet"])
            gru = None
            if use_gru:
                gru = psn.GRUOnVec(K, K, engage=True).to(DEV)
                gru.load_state_dict(d["gru"]); gru.hidden = d["gru_hidden"].to(DEV)
            feat0 = d["feat"]
            fheads = snc.ActualHShim(heads, net) if actual_h else heads
            snap = (feat0, (gru.hidden.clone() if gru is not None else None))
            with rng_frozen():
                streams = {"ordered": ordered_streams(loaders),
                           "shuffled": shuffled_streams(loaders)}
                for em, upd in (("frozen", False), ("live", True)):
                    res = {}
                    for od in ("ordered", "shuffled"):
                        feat = restore_sn(snap, gru)
                        res[od] = eval_signalnet(net, gate, feat, snet, gru, fheads,
                                                 streams[od], upd)
                    emit(f"{cid}|{em}", res, ANCHOR.get(f"{cid}|{em}"), log)


def part_capacity(done, log):
    """signalnet_capacity's 9 `sngru` cells: H in {400,10,5} x seeds {42,43,44}, actual-H, frozen protocol.
    `sngru-dead` / `sn` / `er` are skipped -- dead gate (|g|=0), no GRU, or no gate at all."""
    ref = {}
    for ln in (REPO / "signalnet_capacity" /
               "signalnet_capacity_results.tsv").read_text().splitlines():
        f = ln.split("\t")
        if len(f) >= 4 and f[2] == "sngru":
            ref[(int(f[0]), int(f[1]))] = float(f[3])
    for H in snc.WIDTHS:
        for seed in snc.SEEDS:
            cid = f"sngru|H{H}|seed{seed}"
            if all(f"{cid}|{od}" in done for od in ("ordered", "shuffled")):
                continue
            log(f"  training {cid} ...")
            with snc.width(H):
                net, gate, heads, feat, snet, gru, fheads, loaders = \
                    train_signalnet(True, True, True, snc.K, seed=seed, log=log)
                snap = snap_sn(feat, gru)
                with rng_frozen():
                    streams = {"ordered": ordered_streams(loaders),
                               "shuffled": shuffled_streams(loaders)}
                    res = {}
                    for od in ("ordered", "shuffled"):
                        feat = restore_sn(snap, gru)
                        res[od] = eval_signalnet(net, gate, feat, snet, gru, fheads,
                                                 streams[od], False)
            emit(cid, res, ref.get((H, seed)), log)


# ------------------------------- ledger + reporting -------------------------------
def emit(cid, res, anchor, log):
    """Append both orders, print the anchor delta and the order delta.

    Anchor tolerance is 1e-6: the frozen ledgers store "%.6f", so a bit-exact run still differs from the
    ledger-read value by pure rounding (CLAUDE.md -- match the tolerance to how the value was produced).
    """
    o, s = res["ordered"], res["shuffled"]
    for od in ("ordered", "shuffled"):
        r = res[od]
        row = "\t".join([f"{cid}|{od}", f"{r['pooled']:.6f}", f"{r['macro']:.6f}",
                         f"{r['h0']:.4f}", f"{r['h1']:.4f}", f"{r['out']:.4f}"])
        with TSV.open("a") as fh:
            fh.write(row + "\n")
    tag = ""
    if anchor is not None:
        d = o["pooled"] - anchor
        tag = f"  [anchor {anchor:.4f} d={d:+.6f} {'OK' if abs(d) < 1e-6 else 'MISMATCH'}]"
    log(f"  {cid:52s} ordered {o['pooled']:.4f} (macro {o['macro']:.4f})  "
        f"shuffled {s['pooled']:.4f} (macro {s['macro']:.4f})  "
        f"d-shuf {s['pooled'] - o['pooled']:+.4f}{tag}")


def load_done():
    if not TSV.exists():
        return set()
    return {ln.split("\t", 1)[0] for ln in TSV.read_text().splitlines() if ln.strip()}


def table(log=print):
    rows = {}
    for ln in TSV.read_text().splitlines():
        if not ln.strip():
            continue
        f = ln.split("\t")
        rows[f[0]] = dict(pooled=float(f[1]), macro=float(f[2]),
                          h0=float(f[3]), h1=float(f[4]), out=float(f[5]))
    base = sorted({k.rsplit("|", 1)[0] for k in rows})
    log(f"{'cell':52s}{'ord pooled':>12s}{'shuf pooled':>12s}{'d-shuf':>10s}"
        f"{'ord macro':>11s}{'shuf macro':>11s}{'d-macro':>10s}{'|g| out':>10s}")
    for b in base:
        o, s = rows.get(f"{b}|ordered"), rows.get(f"{b}|shuffled")
        if not (o and s):
            continue
        log(f"{b:52s}{o['pooled']:12.4f}{s['pooled']:12.4f}{s['pooled'] - o['pooled']:+10.4f}"
            f"{o['macro']:11.4f}{s['macro']:11.4f}{s['macro'] - o['macro']:+10.4f}"
            f"{o['out']:10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "stateful", "traces", "signalnet", "capacity", "table"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tsv", default=None,
                    help="ledger shard path; parts run concurrently write their own, merged after "
                         "(Ledger append is not guaranteed atomic across processes)")
    args = ap.parse_args()
    if args.tsv:
        global TSV
        TSV = Path(args.tsv)
    if args.part == "table":
        table(); return
    done = load_done() if args.resume else set()
    log = lambda s: print(s, flush=True)                              # noqa: E731
    log(f"device={DEV}  shuffled-test-order re-check  (shuf seed {SHUF_SEED}, "
        f"lr {LR}, ep {EPOCHS}, buffer {BUFFER}; no new tuning)\n")
    parts = {"stateful": part_stateful, "traces": part_traces,
             "signalnet": part_signalnet, "capacity": part_capacity}
    order = ["traces", "signalnet", "stateful", "capacity"] if args.part == "all" else [args.part]
    for p in order:
        log(f"=== part {p} ===")
        parts[p](done, log)
    log("DONE")


if __name__ == "__main__":
    main()
