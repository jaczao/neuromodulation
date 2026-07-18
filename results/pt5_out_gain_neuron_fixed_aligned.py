"""pt5 OUT-LAYER gain per-neuron under the FIXED projections — LABEL-ALIGNED output gate (redo).

Supersedes the `out` cells of results/pt5_out_gain_neuron_fixed.py, which used the BROKEN random
P_out (a random column partition -> zeroed the wrong logits -> chance: disjoint out ~0.06/0.19).
The output gate is now built label-aligned in GainDriverModulator: for a FIXED projection, task t's
output gate keeps EXACTLY task t's own class columns (== task-IL gating), threaded from the sequence
via _build_pt5_model. See the aligned build in prototype/neuromod.py and test_gain_output_gate_label_aligned.

Per the user's instruction, the `hid` (0,2) cells are NOT re-run — they never touch P_out, so the
values from results/pt5_out_gain_neuron_fixed.log are still valid and are merged in as constants
(SGD hid + both baselines). This script runs only what the alignment fix changes or what was missing:
  - 8 aligned `out` cells : {disjoint, shared} x {sgd, adam} x {standalone, er}
  - 4 Adam `hid` cells    : (the old job was killed before it produced them)
  - Adam baselines (naive, er)

Config identical: gain per-neuron, class-IL, er_task_id=True (default), buffer 1000, lr 1e-3, ep 5,
seed 42. Standalone = naive + masked loss ('loss'); er = 'none'. gain_form inert under fixed P.

CONTRAST with the LEARNED projection (results/pt5_out_bias.log gain-neuron): there d-out was +0.055
(SGD std) / +0.209 (SGD er) / +0.215 (Adam std). Now the fixed projections have a CORRECT
(label-aligned) output gate too, so this asks: does adding the task-IL logit gate help the fixed
disjoint/shared subnetwork mechanism? NOTE the aligned output gate makes gain-neuron+out essentially
task-IL at eval (each task's own 2 class columns kept, the other 8 zeroed), so forgetting -> ~0 and
these numbers are task-IL-style on the class-IL metric (oracle, as ever).

Run: uv run python results/pt5_out_gain_neuron_fixed_aligned.py  (redirect to .log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000
PROJECTIONS = ["disjoint", "shared"]
ARMS = [("standalone", "naive", False), ("er-own", "er", True)]

# valid HID cells + baselines from results/pt5_out_gain_neuron_fixed.log (never touched P_out).
# (opt, proj, arm) -> (acc, forget). Adam hid is re-run below (the old job was killed first).
HID_CONST = {
    ("sgd", "disjoint", "standalone"): (0.6225, 0.0071),
    ("sgd", "disjoint", "er-own"): (0.8163, 0.0012),
    ("sgd", "shared", "standalone"): (0.6597, 0.1448),
    ("sgd", "shared", "er-own"): (0.9658, 0.0009),
}
BASE_CONST = {("sgd", "naive"): (0.6296, 0.1245), ("sgd", "er"): (0.7226, 0.2385)}
BASE_REF = {("sgd", "naive"): 0.6296, ("sgd", "er"): 0.7226,
            ("adam", "naive"): 0.3894, ("adam", "er"): 0.9053}


def _cfg(optimizer, replay, **kw) -> CLConfig:
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                    output_masking=("none" if replay else "loss"),
                    er_buffer_size=BUFFER, **kw)


def _gain_cfg(optimizer, replay, projection, layers):
    return _cfg(optimizer, replay, use_neuromod=True, neuromod_drivers="task_id=onehot",
                neuromod_context="none", neuromod_target="activation", neuromod_granularity="neuron",
                neuromod_gain_layers=layers, neuromod_projection=projection, neuromod_er_task_id=True)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:54s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


baselines = dict(BASE_CONST)
hid = dict(HID_CONST)
out = {}
for optimizer in ["sgd", "adam"]:
    print(f"\n################ class-IL  optimizer={optimizer} ################", flush=True)
    if optimizer == "adam":                       # SGD baselines are constants; re-run Adam's
        baselines[("adam", "naive")] = run("[adam] naive (baseline)", _cfg("adam", False), "naive")
        baselines[("adam", "er")] = run("[adam] er (baseline)", _cfg("adam", True), "er")
    for projection in PROJECTIONS:
        for arm, method, replay in ARMS:
            if optimizer == "adam":               # SGD hid are constants; run Adam hid
                hid[(optimizer, projection, arm)] = run(
                    f"[adam] {projection} gain-neuron {arm} hid (0,2)",
                    _gain_cfg("adam", replay, projection, "0,2"), method)
            out[(optimizer, projection, arm)] = run(
                f"[{optimizer}] {projection} gain-neuron {arm} out (0,2,4) ALIGNED",
                _gain_cfg(optimizer, replay, projection, "0,2,4"), method)


print("\n\n" + "=" * 94)
print("pt5 OUT-LAYER gain per-neuron, FIXED projections, LABEL-ALIGNED P_out — class-IL")
print("1 seed (42), lr=1e-3, ep=5, buffer=1000. hid = gate (h0,h1) | out = + label-aligned logit gate")
print("SGD hid + baselines are constants from results/pt5_out_gain_neuron_fixed.log (unaffected by the fix)")
print("=" * 94)
for optimizer in ["sgd", "adam"]:
    nb = baselines[(optimizer, "naive")][0]; eb = baselines[(optimizer, "er")][0]
    print(f"\n--- optimizer={optimizer.upper()} ---   baselines: naive={nb:.4f}  er={eb:.4f}")
    for projection in PROJECTIONS:
        print(f"  [{projection}]")
        print(f"  {'arm':12s} {'hid':>8s} {'f':>7s} {'out(aligned)':>13s} {'f':>7s} "
              f"{'d-out':>8s} {'out vs-base':>11s}")
        for arm, _, replay in ARMS:
            base = eb if replay else nb
            h, hf = hid[(optimizer, projection, arm)]
            o, of = out[(optimizer, projection, arm)]
            print(f"  {arm:12s} {h:>8.4f} {hf:>7.4f} {o:>13.4f} {of:>7.4f} "
                  f"{o - h:>+8.4f} {o - base:>+11.4f}")

print("\n--- BASELINE REPRODUCTION CHECK ---")
for (opt, m), ref in BASE_REF.items():
    got = baselines[(opt, m)][0]
    ok = abs(got - ref) < 5e-4
    print(f"  {opt:4s} {m:5s} got={got:.4f} ref={ref:.4f} {'MATCH' if ok else '*** MISMATCH ***'}")

print("\nd-out = aligned out minus hid (same arm). The aligned output gate keeps each task's own 2 "
      "class columns and zeros the other 8 (== task-IL gating), so gain-neuron+out is task-IL-style on "
      "the class-IL metric (oracle). Bias variants are N/A for per-neuron gain (gamma=0 freezes biases "
      "implicitly). Broken-random out (pre-fix, for contrast): disjoint 0.0647/0.1937, shared "
      "0.4212/0.5908 (see results/pt5_out_gain_neuron_fixed.log). 1 seed.")
