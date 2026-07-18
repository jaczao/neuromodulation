"""pt5 OUT-LAYER + BIAS modulation (user-requested) — extend the learned-projection winning arms to
the OUTPUT layer (net.4 / logits), then to the BIASES of the modulated layers.

Grid: {gain, plast} x {per-neuron, per-synapse} x {sgd, adam} x class-IL x two arms:
  - buf-meta-own : STANDALONE with the modulator-only replay buffer (--neuromod-meta-replay +
    --neuromod-er-task-id). gain: per-task meta gate trains ONLY P (followup-3); plast: lookahead
    meta-loss on current+buffer (followup-2; er_task_id is inert there — replay-gated for the main
    batch, meta_task_id read only by the gain meta-loop). method=naive, masking='loss'.
  - er-own       : +ER with each replayed sample under its OWN P[j] (--neuromod-er-task-id, the
    now-default). method=er, masking='none'.

STAGES per mechanism (columns of the table):
  - hid  : hidden-only layer set (0,2) — the prior-study configuration.
  - out  : + output layer. gain-neuron: gain_layers 0,2,4 (LEARNED per-class logit gain — NOT the
    broken fixed-P random partition; P_out is zero-init => gamma=1 parity, trained per task row).
    per-synapse gain/plast: mask_layers 0,2,4 (explicit head-synapse gate — the thing that fought
    replay under the fixed P, here ablated under er-own's correct routing).
    plast-neuron: 0,2,4 is the long-standing DEFAULT (head cols via the a1 out-side coupling), so
    its OUT cell is the existing reference and the NEW run is the reverse ablation hid=(0,2).
  - bias : out + --neuromod-modulate-bias (independent learned P_bias per listed layer, incl. the
    HEAD bias — the known residual leak the weight gates cannot reach). Per-SYNAPSE mechanisms only:
    per-neuron gain freezes biases implicitly (gamma=0 => zero bias grad) and per-neuron plasticity
    already gates biases on the `in` side; the toggle is not wired to them (N/A, not a degenerate 0).

Config held at the prior studies' values: projection=learned, gain_form=unbounded, plast init=0.5,
seed 42, lr=1e-3, ep=5, buffer=1000. REFS below come from results/pt5_gain_forms_buffer.log (gain
hid cells) and results/pt5_plast_init.log init=0.5 (plast cells; its standalone numbers match
results/pt5_iter3_metareplay.log, confirming er_task_id's inertness there).

CAVEATS carried: ORACLE task id at train+eval (task-IL-style on the class-IL metric); the meta arms
USE the buffer (modulator-only replay) so they are not apples-to-apples with no-buffer naive;
beating ER is not beating ER (no oracle there); 1 seed.

Run: uv run python results/pt5_out_bias.py   (redirect to results/pt5_out_bias.log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED = 42
LR = 1e-3
EP = 5
BUFFER = 1000
PROJECTION = "learned"
OPTIMIZERS = ["sgd", "adam"]

# arm -> (method, replay, meta_replay). er_task_id=True everywhere (the default; explicit for clarity)
ARMS = [
    ("buf-meta-own", "naive", False, True),
    ("er-own", "er", True, False),
]

# mechanism -> target, granularity kwargs, layer-set kwarg name, (hid, out) layer strings, bias?
MECHS = [
    ("gain-neuron", "activation", dict(neuromod_granularity="neuron"),
     "neuromod_gain_layers", ("0,2", "0,2,4"), False),
    ("gain-synapse", "activation", dict(neuromod_granularity="synapse"),
     "neuromod_mask_layers", ("0,2", "0,2,4"), True),
    ("plast-neuron", "plasticity", dict(neuromod_granularity="neuron", neuromod_plasticity_scope="both"),
     "neuromod_plasticity_layers", ("0,2", "0,2,4"), False),
    ("plast-synapse", "plasticity", dict(neuromod_granularity="synapse"),
     "neuromod_mask_layers", ("0,2", "0,2,4"), True),
]

# (opt, arm, mech, stage) -> (acc, forget) from the prior logs (same seed/lr/ep/buffer/config).
# gain hid: pt5_gain_forms_buffer.log (unbounded). plast: pt5_plast_init.log init=0.5.
# plast-neuron's REF is its OUT cell (0,2,4 is the default it has always run with).
REFS = {
    ("sgd", "buf-meta-own", "gain-neuron", "hid"): (0.9074, 0.0205),
    ("adam", "buf-meta-own", "gain-neuron", "hid"): (0.5075, 0.3713),
    ("sgd", "er-own", "gain-neuron", "hid"): (0.7376, 0.2241),
    ("adam", "er-own", "gain-neuron", "hid"): (0.9887, 0.0082),
    ("sgd", "buf-meta-own", "gain-synapse", "hid"): (0.9871, 0.0015),
    ("adam", "buf-meta-own", "gain-synapse", "hid"): (0.6304, 0.3238),
    ("sgd", "er-own", "gain-synapse", "hid"): (0.7282, 0.2332),
    ("adam", "er-own", "gain-synapse", "hid"): (0.9900, 0.0067),
    ("sgd", "buf-meta-own", "plast-neuron", "out"): (0.6454, 0.1273),
    ("adam", "buf-meta-own", "plast-neuron", "out"): (0.4244, 0.5059),
    ("sgd", "er-own", "plast-neuron", "out"): (0.6598, 0.3006),
    ("adam", "er-own", "plast-neuron", "out"): (0.8936, 0.1001),
    ("sgd", "buf-meta-own", "plast-synapse", "hid"): (0.6417, 0.1106),
    ("adam", "buf-meta-own", "plast-synapse", "hid"): (0.3861, 0.5280),
    ("sgd", "er-own", "plast-synapse", "hid"): (0.6110, 0.3468),
    ("adam", "er-own", "plast-synapse", "hid"): (0.8935, 0.0991),
}
# baseline reproduction targets (bit-consistent across iter3 / gain_forms_buffer / plast_init)
BASE_REF = {("sgd", "naive"): 0.6296, ("sgd", "er"): 0.7226,
            ("adam", "naive"): 0.3894, ("adam", "er"): 0.9053}


def _cfg(optimizer, replay, **kw) -> CLConfig:
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer,
                    output_masking=("none" if replay else "loss"),
                    er_buffer_size=BUFFER, **kw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:58s} acc={acc:.4f}  forget={forget:.4f}", flush=True)
    return acc, forget


results = dict(REFS)   # merged: refs fill the already-run cells, runs fill the rest
baselines = {}
for optimizer in OPTIMIZERS:
    print(f"\n################ class-IL  optimizer={optimizer} ################", flush=True)
    baselines[(optimizer, "naive")] = run(f"[{optimizer}] naive (baseline)",
                                          _cfg(optimizer, replay=False), "naive")
    baselines[(optimizer, "er")] = run(f"[{optimizer}] er (baseline)",
                                       _cfg(optimizer, replay=True), "er")
    for arm, method, replay, meta in ARMS:
        for mech, target, gkw, layer_key, (hid, out), has_bias in MECHS:
            stages = [(s, ls, False) for s, ls in (("hid", hid), ("out", out))]
            if has_bias:
                stages.append(("bias", out, True))
            for stage, layer_set, mod_bias in stages:
                if (optimizer, arm, mech, stage) in results:
                    continue                     # already run in a prior study (REFS)
                cfg = _cfg(optimizer, replay,
                           use_neuromod=True, neuromod_drivers="task_id=onehot",
                           neuromod_context="none", neuromod_target=target,
                           neuromod_projection=PROJECTION,
                           neuromod_meta_replay=meta, neuromod_er_task_id=True,
                           neuromod_modulate_bias=mod_bias,
                           **{layer_key: layer_set}, **gkw)
                results[(optimizer, arm, mech, stage)] = run(
                    f"[{optimizer}] {mech} {arm} {stage} ({layer_set}"
                    f"{'+bias' if mod_bias else ''})", cfg, method)


print("\n\n" + "=" * 106)
print("pt5 OUT-LAYER + BIAS (projection=learned, class-IL) — 1 seed (42), lr=1e-3, ep=5, buffer=1000")
print("hid = layers (0,2) | out = (0,2,4) | bias = (0,2,4) + --neuromod-modulate-bias (per-synapse only)")
print("plast-neuron: 0,2,4 is its default => 'out' is the prior ref and 'hid' is the NEW reverse ablation")
print("=" * 106)
for optimizer in OPTIMIZERS:
    nb = baselines[(optimizer, "naive")][0]
    eb = baselines[(optimizer, "er")][0]
    print(f"\n--- optimizer={optimizer.upper()} ---   baselines: naive={nb:.4f}  er={eb:.4f}")
    for arm, _, replay, _ in ARMS:
        base = eb if replay else nb
        base_name = "er" if replay else "naive"
        print(f"  [{arm}]  (deltas vs {base_name}={base:.4f}; d-out = out-hid, d-bias = bias-out)")
        print(f"  {'mechanism':14s} {'hid':>7s} {'f':>7s} {'out':>7s} {'f':>7s} {'d-out':>8s} "
              f"{'bias':>7s} {'f':>7s} {'d-bias':>8s} {'best-vs-base':>13s}")
        for mech, _, _, _, _, has_bias in MECHS:
            h, hf = results[(optimizer, arm, mech, "hid")]
            o, of = results[(optimizer, arm, mech, "out")]
            if has_bias:
                b, bf = results[(optimizer, arm, mech, "bias")]
                bias_s = f"{b:>7.4f} {bf:>7.4f} {b - o:>+8.4f}"
                best = max(h, o, b)
            else:
                bias_s = f"{'n/a':>7s} {'':>7s} {'':>8s}"
                best = max(h, o)
            print(f"  {mech:14s} {h:>7.4f} {hf:>7.4f} {o:>7.4f} {of:>7.4f} {o - h:>+8.4f} "
                  f"{bias_s} {best - base:>+13.4f}")

print("\n--- BASELINE REPRODUCTION CHECK ---")
for (opt, m), ref in BASE_REF.items():
    got = baselines[(opt, m)][0]
    ok = abs(got - ref) < 5e-4
    print(f"  {opt:4s} {m:5s} got={got:.4f} ref={ref:.4f} {'MATCH' if ok else '*** MISMATCH ***'}")

print("\nN/A cells: per-neuron bias modulation is not wired by design (gain-neuron freezes biases "
      "implicitly via gamma=0; plast-neuron gates them on the `in` side; the HEAD bias has no "
      "output-neuron alpha to attach to). Caveats: ORACLE task id at train+eval; the meta arms use "
      "the buffer (modulator-only replay); 1 seed; sparsity_lambda=0; plast init=0.5 (the shipped "
      "default — known suboptimal for the +ER arm under SGD, see the init-sweep gotcha).")
