"""pt5 OUT-LAYER + BIAS follow-up: the hid+bias cell (layers 0,2 + --neuromod-modulate-bias).

results/pt5_out_bias.py ran bias modulation only on the OUT stage (0,2,4 + bias), so its head bias
was always included. This adds the missing hid+bias cell — layers (0,2) + bias — which modulates ONLY
the two HIDDEN-layer biases and NOT the head bias, isolating hidden-bias modulation from the head-bias
leak. Per-SYNAPSE mechanisms only (the toggle is not wired to per-neuron by design).

8 cells: {gain-synapse, plast-synapse} x {sgd, adam} x {buf-meta-own standalone, er-own}, all layers
(0,2) + bias. Config identical to pt5_out_bias.py (learned P, gain_form=unbounded, plast init 0.5,
seed 42, lr 1e-3, ep 5, buffer 1000). Compare against that study's hid (no bias) and out+bias cells.

REFERENCE cells (from results/pt5_out_bias.log; the columns this slots between):
                        hid (0,2)        out+bias (0,2,4+bias)
  gain-syn  SGD  meta   0.9871           0.9892
  gain-syn  SGD  er     0.7282           0.7348
  gain-syn  Adam meta   0.6304           0.5203
  gain-syn  Adam er     0.9900           0.9923
  plast-syn SGD  meta   0.6417           0.6440
  plast-syn SGD  er     0.6110           0.5837
  plast-syn Adam meta   0.3861           0.4008
  plast-syn Adam er     0.8935           0.9018

Caveats carry: ORACLE task id at train+eval; the meta arm uses the buffer (modulator-only replay);
1 seed; sparsity_lambda=0; plast init 0.5.

Run: uv run python results/pt5_out_bias_hid.py   (redirect to results/pt5_out_bias_hid.log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000
OPTIMIZERS = ["sgd", "adam"]

# arm -> (method, replay, meta_replay). er_task_id=True everywhere (the default).
ARMS = [("buf-meta-own", "naive", False, True), ("er-own", "er", True, False)]

MECHS = [
    ("gain-synapse", "activation", dict(neuromod_granularity="synapse")),
    ("plast-synapse", "plasticity", dict(neuromod_granularity="synapse")),
]

# comparison cells from results/pt5_out_bias.log: (opt, arm, mech) -> (hid, out+bias)
REF = {
    ("sgd", "buf-meta-own", "gain-synapse"): (0.9871, 0.9892),
    ("sgd", "er-own", "gain-synapse"): (0.7282, 0.7348),
    ("adam", "buf-meta-own", "gain-synapse"): (0.6304, 0.5203),
    ("adam", "er-own", "gain-synapse"): (0.9900, 0.9923),
    ("sgd", "buf-meta-own", "plast-synapse"): (0.6417, 0.6440),
    ("sgd", "er-own", "plast-synapse"): (0.6110, 0.5837),
    ("adam", "buf-meta-own", "plast-synapse"): (0.3861, 0.4008),
    ("adam", "er-own", "plast-synapse"): (0.8935, 0.9018),
}
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


results = {}
baselines = {}
for optimizer in OPTIMIZERS:
    print(f"\n################ class-IL  optimizer={optimizer} ################", flush=True)
    baselines[(optimizer, "naive")] = run(f"[{optimizer}] naive (baseline)",
                                          _cfg(optimizer, replay=False), "naive")
    baselines[(optimizer, "er")] = run(f"[{optimizer}] er (baseline)",
                                       _cfg(optimizer, replay=True), "er")
    for arm, method, replay, meta in ARMS:
        for mech, target, gkw in MECHS:
            cfg = _cfg(optimizer, replay,
                       use_neuromod=True, neuromod_drivers="task_id=onehot",
                       neuromod_context="none", neuromod_target=target,
                       neuromod_projection="learned", neuromod_meta_replay=meta,
                       neuromod_er_task_id=True, neuromod_modulate_bias=True,
                       neuromod_mask_layers="0,2", **gkw)
            results[(optimizer, arm, mech)] = run(
                f"[{optimizer}] {mech} {arm} hid+bias (0,2+bias)", cfg, method)


print("\n\n" + "=" * 96)
print("pt5 OUT-LAYER + BIAS follow-up: hid+bias (layers 0,2 + --neuromod-modulate-bias, per-synapse)")
print("hidden-layer biases modulated, HEAD bias NOT. 1 seed (42), lr=1e-3, ep=5, buffer=1000")
print("=" * 96)
for optimizer in OPTIMIZERS:
    nb = baselines[(optimizer, "naive")][0]; eb = baselines[(optimizer, "er")][0]
    print(f"\n--- optimizer={optimizer.upper()} ---   baselines: naive={nb:.4f}  er={eb:.4f}")
    for arm, _, replay, _ in ARMS:
        base = eb if replay else nb; bname = "er" if replay else "naive"
        print(f"  [{arm}]  vs {bname}={base:.4f}   (hid, out+bias from results/pt5_out_bias.log)")
        print(f"  {'mechanism':14s} {'hid':>8s} {'hid+bias':>9s} {'d-hidbias':>10s} "
              f"{'out+bias':>9s} {'vs-base':>8s}")
        for mech, _, _ in MECHS:
            hb, hbf = results[(optimizer, arm, mech)]
            hid, outbias = REF[(optimizer, arm, mech)]
            print(f"  {mech:14s} {hid:>8.4f} {hb:>9.4f} {hb - hid:>+10.4f} "
                  f"{outbias:>9.4f} {hb - base:>+8.4f}")

print("\n--- BASELINE REPRODUCTION CHECK ---")
for (opt, m), ref in BASE_REF.items():
    got = baselines[(opt, m)][0]
    ok = abs(got - ref) < 5e-4
    print(f"  {opt:4s} {m:5s} got={got:.4f} ref={ref:.4f} {'MATCH' if ok else '*** MISMATCH ***'}")

print("\nd-hidbias = (0,2)+bias minus (0,2) no-bias: the effect of modulating ONLY the hidden biases "
      "(head bias excluded). Caveats: ORACLE task id at train+eval; the meta arm uses the buffer; "
      "1 seed; plast init 0.5.")
