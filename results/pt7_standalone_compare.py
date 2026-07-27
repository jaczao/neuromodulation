"""Standalone (NO buffer) neuromod on class-IL Split MNIST, gain-SYNAPSE, masked loss.

User-requested comparison: run the standalone (nobuf) arm of three mechanisms at the
NAIVE-TUNED operating point (naive's own best lr per optimizer, from pt3_retry:
sgd lr=1e-3/ep5, adam lr=1e-5/ep5), 1 seed (42), class-IL, masked training loss,
gain-SYNAPSE gate:

  - soft_mlp       (pt6 task-inference selector; oracle-free eval = soft/hard)
  - all4  (std1)   (pt7 DA/ACh/NE/5HT drivers, standardized)
  - ACh_ema (std0) (pt7 tonic entropy EMA, UN-standardized per the tonic rule)

Baselines (naive, er, ewc+er) are already saved in pt3_retry_results.tsv; standalone
EWC is run separately via train.py. Nothing here is re-run if already in the ledger.

Self-contained: reuses pt7_neuromodulators (p7) for all4/ACh_ema nobuf, and
pt6_synapse (p6s) primitives for a soft_mlp nobuf loop (buf-own minus the buffer).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pt7_neuromodulators as p7      # noqa: E402
import pt6_synapse as p6s             # noqa: E402

DEV = p7.DEV
SEED = 42
EP = 5
NAIVE_LR = {"sgd": 1e-3, "adam": 1e-5}          # naive's OWN tuned lr (pt3_retry)
LEDGER = HERE / "pt7_standalone_compare_results.tsv"


def _load_done():
    done = {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                parts = line.split("\t")
                done[parts[0]] = parts
    return done


def _write(tag, fields):
    with LEDGER.open("a") as f:
        f.write(tag + "\t" + "\t".join(f"{v:.4f}" for v in fields) + "\n")


# ---------------- pt7 all4 / ACh_ema : nobuf gain-synapse ----------------
def run_p7_nobuf(name, standardize, opt_kind):
    """Standalone nobuf gain-synapse via p7.train_nobuf at the naive-tuned lr. -> pred/true/probe."""
    loaders, net, gate, heads, sig, is_free, is_const = p7.build(
        name, "synapse", seed=SEED, standardize=standardize)
    p7.net_loaders = loaders
    p7.train_nobuf(name, "synapse", net, gate, heads, sig, is_free, is_const,
                   opt_kind, lr=NAIVE_LR[opt_kind], epochs=EP)
    r = p7.eval_cell(name, "synapse", net, gate, heads, sig, is_const, loaders)
    return r


# ---------------- pt6 soft_mlp : nobuf gain-synapse (no buffer anywhere) ----------------
def train_softmlp_nobuf(mech, net, loaders, opt_kind, lr):
    """buf-own minus the buffer: naive masked-CE main + selector g(x) + gate meta,
    ALL on the current task only (no replay). Selector is expected to forget -> chance."""
    inf_opt = torch.optim.Adam(mech.inf_params(), lr=lr)
    main_opt = p6s._opt(opt_kind, net.parameters(), lr)
    gate_opt = torch.optim.Adam(mech.gate_params(), lr=lr)
    for t in range(5):
        for _ in range(EP):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV)
                tids = torch.full((x.size(0),), t, device=DEV)
                loss = p6s.masked_ce(p6s.fwd_grouped(net, mech, x, tids), y)
                net.zero_grad(); mech.zero_grad(); loss.backward(); main_opt.step()
                ce = p6s.CE(mech.task_logits(x), tids)                 # selector, current task only
                inf_opt.zero_grad(); ce.backward(); inf_opt.step()
                meta = p6s.masked_ce(p6s.fwd_grouped(net, mech, x, tids), y)  # gate meta, current only
                net.zero_grad(); mech.zero_grad(); meta.backward(); gate_opt.step()


def run_softmlp_nobuf(opt_kind):
    loaders, net, mech = p6s.build(seed=SEED)
    train_softmlp_nobuf(mech, net, loaders, opt_kind, NAIVE_LR[opt_kind])
    return p6s.evaluate(mech, net, loaders)


def main():
    done = _load_done()
    print(f"device={DEV}  standalone nobuf gain-SYNAPSE  masked  class-IL  seed={SEED}", flush=True)
    print(f"naive-tuned lr: sgd={NAIVE_LR['sgd']:g} adam={NAIVE_LR['adam']:g}  ep={EP}\n", flush=True)

    for opt in ("sgd", "adam"):
        # all4 std1
        tag = f"all4|synapse|nobuf|{opt}|std1|lr{NAIVE_LR[opt]:g}|ep{EP}"
        if tag not in done:
            r = run_p7_nobuf("all4", True, opt)
            _write(tag, [r["pred"], r["true"], r["probe"]])
            print(f"  all4    std1 {opt:4s} | pred={r['pred']:.4f} true={r['true']:.4f} probe={r['probe']:.3f}", flush=True)
        else:
            print(f"  all4    std1 {opt:4s} | (cached) pred={done[tag][1]}", flush=True)

        # ACh_ema std0
        tag = f"ACh_ema|synapse|nobuf|{opt}|std0|lr{NAIVE_LR[opt]:g}|ep{EP}"
        if tag not in done:
            r = run_p7_nobuf("ACh_ema", False, opt)
            _write(tag, [r["pred"], r["true"], r["probe"]])
            print(f"  ACh_ema std0 {opt:4s} | pred={r['pred']:.4f} true={r['true']:.4f} probe={r['probe']:.3f}", flush=True)
        else:
            print(f"  ACh_ema std0 {opt:4s} | (cached) pred={done[tag][1]}", flush=True)

        # soft_mlp
        tag = f"soft_mlp|synapse|nobuf|{opt}|-|lr{NAIVE_LR[opt]:g}|ep{EP}"
        if tag not in done:
            r = run_softmlp_nobuf(opt)
            _write(tag, [r["soft"], r["hard"], r["oracle"], r["infer"]])
            print(f"  soft_mlp     {opt:4s} | soft={r['soft']:.4f} hard={r['hard']:.4f} "
                  f"oracle={r['oracle']:.4f} infer={r['infer']:.4f}", flush=True)
        else:
            print(f"  soft_mlp     {opt:4s} | (cached) soft={done[tag][1]}", flush=True)


if __name__ == "__main__":
    main()
