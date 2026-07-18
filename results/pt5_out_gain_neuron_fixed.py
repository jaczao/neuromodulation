"""pt5 OUT-LAYER modulation for gain per-neuron under the FIXED projections (iter 1 disjoint,
iter 2 shared) — the fixed-P counterpart of results/pt5_out_bias.py's gain-neuron cells (learned P).

User scope: gain per-NEURON only, {sgd, adam}, {standalone (non-ER), er}, class-IL, both fixed
projections. For per-neuron gain the bias variants ("biases", "out+biases") are N/A: the activation
gate gamma multiplies (Wx+b), so gamma=0 already zeroes a neuron's bias contribution AND its bias
gradient (implicit freeze) — there is no separate bias projection wired to GainDriverModulator. So the
only meaningful stages are hid (gate h0,h1) and out (gate h0,h1 AND the 10 output logits).

THE POINT: under the LEARNED projection (pt5_out_bias.log) the out-logit gate was the study's win
(zero-init P_out => gamma=1 parity, trained per-task row; +0.209 on gain-neuron SGD er). Under a FIXED
projection the logit gate P_out = build_disjoint/shared_proj(n_tasks, 10) is a RANDOM balanced column
partition, NOT the true Split-MNIST class->task map (task t = classes {2t, 2t+1}). So task t's eval
gate keeps 2 RANDOM class columns, almost never t's real classes -> the correct logits are zeroed ->
collapse to chance. This sweep confirms that across {disjoint, shared} x {sgd, adam} x {naive, er}.
gain_form is inert under a fixed P (all forms -> raw {0,1}); left at the default.

Config matches the fixed-P studies: er_task_id=True (default), buffer 1000, lr 1e-3, ep 5, seed 42.
Standalone = naive + masked loss ('loss'); er = 'none'. Watch the per-task "seen tasks: [...]" line in
the log for the out cells — the broken pattern is ~[x, x, 0, 0, 0] (only the task whose random gate
accidentally overlaps a true class scores).

Run: uv run python results/pt5_out_gain_neuron_fixed.py   (redirect to results/pt5_out_gain_neuron_fixed.log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000
OPTIMIZERS = ["sgd", "adam"]
PROJECTIONS = ["disjoint", "shared"]          # iter 1, iter 2
ARMS = [("standalone", "naive", False), ("er-own", "er", True)]
STAGES = [("hid", "0,2"), ("out", "0,2,4")]   # per-neuron: no bias variants (N/A)

# soft comparison points (config may differ slightly on the er_task_id default; not asserted).
# disjoint hid: naive+gain 0.6225 (iter-1) ; er+gain 0.8163 (er_task_id ON) / 0.8264 (OFF)
# shared   hid: naive+gain 0.6752 SGD / 0.6827 Adam ; er+gain 0.8709 SGD / 0.9728 Adam (iter-2, er OFF)
BASE_REF = {("sgd", "naive"): 0.6296, ("sgd", "er"): 0.7226,
            ("adam", "naive"): 0.3894, ("adam", "er"): 0.9053}


def _cfg(optimizer, replay, **kw) -> CLConfig:
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                    output_masking=("none" if replay else "loss"),
                    er_buffer_size=BUFFER, **kw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:52s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


baselines = {}
results = {}
for optimizer in OPTIMIZERS:
    print(f"\n################ class-IL  optimizer={optimizer} ################", flush=True)
    baselines[(optimizer, "naive")] = run(f"[{optimizer}] naive (baseline)",
                                          _cfg(optimizer, replay=False), "naive")
    baselines[(optimizer, "er")] = run(f"[{optimizer}] er (baseline)",
                                       _cfg(optimizer, replay=True), "er")
    for projection in PROJECTIONS:
        for arm, method, replay in ARMS:
            for stage, layers in STAGES:
                cfg = _cfg(optimizer, replay,
                           use_neuromod=True, neuromod_drivers="task_id=onehot",
                           neuromod_context="none", neuromod_target="activation",
                           neuromod_granularity="neuron", neuromod_gain_layers=layers,
                           neuromod_projection=projection, neuromod_er_task_id=True)
                results[(optimizer, projection, arm, stage)] = run(
                    f"[{optimizer}] {projection} gain-neuron {arm} {stage} ({layers})", cfg, method)


print("\n\n" + "=" * 92)
print("pt5 OUT-LAYER gain per-neuron, FIXED projections (iter1 disjoint, iter2 shared) — class-IL")
print("1 seed (42), lr=1e-3, ep=5, buffer=1000. hid = gate (h0,h1) | out = + output logits (P_out)")
print("=" * 92)
for optimizer in OPTIMIZERS:
    nb = baselines[(optimizer, "naive")][0]; eb = baselines[(optimizer, "er")][0]
    print(f"\n--- optimizer={optimizer.upper()} ---   baselines: naive={nb:.4f}  er={eb:.4f}")
    for projection in PROJECTIONS:
        print(f"  [{projection}]")
        print(f"  {'arm':12s} {'hid':>8s} {'f':>7s} {'out':>8s} {'f':>7s} {'d-out':>8s} {'vs-base':>8s}")
        for arm, _, replay in ARMS:
            base = eb if replay else nb
            h, hf = results[(optimizer, projection, arm, "hid")]
            o, of = results[(optimizer, projection, arm, "out")]
            print(f"  {arm:12s} {h:>8.4f} {hf:>7.4f} {o:>8.4f} {of:>7.4f} {o - h:>+8.4f} "
                  f"{h - base:>+8.4f}")
        print(f"  (vs-base uses the HID cell vs same-arm baseline; d-out = out-hid, expected < 0 "
              f"since fixed P_out is random)")

print("\n--- BASELINE REPRODUCTION CHECK ---")
for (opt, m), ref in BASE_REF.items():
    got = baselines[(opt, m)][0]
    ok = abs(got - ref) < 5e-4
    print(f"  {opt:4s} {m:5s} got={got:.4f} ref={ref:.4f} {'MATCH' if ok else '*** MISMATCH ***'}")

print("\nContrast with the LEARNED projection (results/pt5_out_bias.log gain-neuron): there d-out was "
      "+0.055 (SGD standalone) / +0.209 (SGD er) / +0.215 (Adam standalone) because P_out is zero-init "
      "(gamma=1 parity) and trained per-task row. Under a FIXED P the logit gate is a random column "
      "partition, so d-out should be strongly NEGATIVE (collapse toward chance). Bias variants are N/A "
      "for per-neuron gain (gamma=0 freezes biases implicitly). Caveats: ORACLE task id; 1 seed.")
