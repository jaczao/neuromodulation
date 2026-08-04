"""Gate statistics for EVERY plasticity mechanism cell (task-IL, tuned), seed 42.

Covers {nobuf, bufcur, ercur, erown} x {neuron, synapse} at each cell's own tuned point. `bufown` is
omitted only because it is byte-identical to `bufcur` (plast_taskil.ARMS) — same gate object, so it
would measure nothing new.

The study says the mechanism ties its dead-gate control. That is an accuracy claim; it does not say
WHY. Two very different states produce it, and they are distinguishable by looking at the gate:
  (a) the gate never engaged   — alpha stayed at its parity init 0.5, so the run was a uniform
      half-LR rescale of naive and there was no mechanism to test (pt7's double-zero-init saddle, and
      the `free` dead-gate control, are both this);
  (b) the gate engaged and was absorbed — alpha moved, differentiated per task, and still bought
      nothing (the pt5/pt6/pt7 scale-degeneracy story, and what pt7_capacity found under scarcity:
      engagement rises, benefit does not).
This reads the learned alpha out and says which.

HOW. `prototype/train.py` already calls `_pt5_dump_overlap(model, T, config, plast_mod=...)` after
training, guarded by env PT5_DUMP_OVERLAP, and it handles BOTH plasticity granularities. Rather than
parse its printed summary, this patches that symbol with a collector that keeps the raw per-task gate
rows, then computes the statistics here. The hook fires after the last eval, so nothing about training
changes (default OFF = parity, and we only replace what the env var already gates).

alpha_t = sigmoid(init_bias + P[t]) in [0,1]; with plasticity_init=0.5 the parity gate is alpha=0.5
and P=0 (zero init), so `alpha - 0.5` IS the learned task-specific modulation. alpha multiplies the
per-parameter LR: 0 freezes, 1 is full plasticity, 0.5 is half.

REPORTED PER LAYER, never as one mean — pt6-followup F/G: a single mean over all entries hid a 10-wide
out-gate that was 50-100x the hidden gate and was the load-bearing part. plast-neuron's alpha lives on
the hidden neurons (h0, h1); plast-synapse's on the gated weight matrices (net0, net2).

  |a-parity|  mean deviation from the parity gate — the ENGAGEMENT measure. ~0 means case (a).
  a mean/sd/min/max, and the fraction of units below 0.25 (near-frozen) or above 0.75 (near-free).
  cos(dev)    mean off-diagonal cosine of the per-task deviation rows: ~0 orthogonal (tasks allocate
              disjointly), ~1 identical (every task learns the same gate, so it carries no task
              information and is just a global LR rescale). THE key number for "is this allocation".
  cos(gate)   the same on the applied alpha itself — the apples-to-apples view against iter-1's fixed
              {0,1} disjoint gate.
  frac        mean peak-relative engaged fraction (|dev| > 0.25 * that task's peak |dev|).

Run (only after plast_taskil.py has finished): uv run python pt5_taskil/gate_stats.py
"""
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402

import plast_taskil as S                                 # noqa: E402
import prototype.train as train_mod                      # noqa: E402
from prototype.train import cl_train                     # noqa: E402
from trajectories import lookup_only                     # noqa: E402

SEED = 42
# Every mechanism cell of plast_taskil, not just one arm: "did the gate engage, and did each task
# learn a DIFFERENT gate?" is a per-mechanism question, and the arms differ in exactly the inputs
# that could change the answer (whether P sees a retention signal, and whether replayed samples are
# gated by their own task). `bufown` is excluded because it is byte-identical to `bufcur` by
# construction (see plast_taskil.ARMS) — its gate is the same object, so running it measures nothing.
ARMS = [a for a in ("nobuf", "bufcur", "ercur", "erown")]
COLLECTED: dict = {}


def collector(model, n_tasks, config, plast_mod=None):
    """Replacement for _pt5_dump_overlap: keep the per-task gate rows instead of printing a summary.
    Mirrors that function's own gate reconstruction for the two plasticity modulators."""
    ib = float(plast_mod.init_bias)
    parity = float(torch.sigmoid(torch.tensor(ib)))
    layers = {}
    if hasattr(plast_mod, "n_hidden_layers"):            # PlasticityDriverModulator (per-neuron)
        for l in range(plast_mod.n_hidden_layers):
            P = getattr(plast_mod, f"P_{l}").detach()
            layers[f"h{l}"] = torch.sigmoid(ib + P).cpu().numpy()
    else:                                                # SynapsePlasticityDriverModulator
        for idx in plast_mod.layer_indices:
            P = getattr(plast_mod, f"P_{idx}").detach()
            layers[f"net{idx}"] = torch.sigmoid(ib + P).cpu().numpy()
    COLLECTED["layers"] = layers
    COLLECTED["parity"] = parity
    COLLECTED["fixed"] = plast_mod.fixed


def _offdiag_cos(D):
    """Mean off-diagonal cosine between the T task rows of D (rows flattened)."""
    D = D.reshape(D.shape[0], -1)
    n = np.linalg.norm(D, axis=1, keepdims=True)
    C = (D / np.clip(n, 1e-12, None)) @ (D / np.clip(n, 1e-12, None)).T
    T = D.shape[0]
    return float((C.sum() - np.trace(C)) / (T * (T - 1))) if T > 1 else float("nan")


def _engaged_frac(dev, rel=0.25):
    fr = []
    for t in range(dev.shape[0]):
        a = np.abs(dev[t]).ravel()
        peak = a.max()
        fr.append(float((a > rel * peak).mean()) if peak > 0 else 0.0)
    return float(np.mean(fr))


def run(arm, gran, lr, ep, nlr):
    COLLECTED.clear()
    cell = S.cell_name("plast", arm, gran)
    config, method = S.build_config(cell, lr, ep, None, nlr, SEED)
    os.environ["PT5_DUMP_OVERLAP"] = "1"
    original = train_mod._pt5_dump_overlap
    train_mod._pt5_dump_overlap = collector
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            acc, forget = cl_train(config, method, no_wandb=True, sequence=None, eval_split="test")
    finally:
        train_mod._pt5_dump_overlap = original
        os.environ.pop("PT5_DUMP_OVERLAP", None)
    return cell, acc, forget, dict(COLLECTED)


def report(cell, acc, forget, data, nlr, ledger, lr, ep):
    parity = data["parity"]
    key = S.key_of("test", cell, lr, ep, None, nlr, SEED)
    ref = ledger.get(key)
    # 1e-6 = the ledger's storage precision ("%.6f"), not machine epsilon: `acc` is full precision
    # here while `ref` was rounded on write, so a tighter tolerance flags identical runs as mismatched.
    flag = "" if ref is None else ("  [matches ledger]" if abs(ref[0] - acc) < 1e-6
                                   else f"  !! MISMATCH vs ledger {ref[0]:.6f}")
    print(f"\n{'=' * 96}")
    print(f"{cell}  |  test acc {acc:.4f}  forget {forget:.4f}  |  neuro_lr {nlr:g}  "
          f"parity alpha {parity:.4f}{flag}")
    print("=" * 96)
    print(f"{'layer':7s} {'n':>9s} {'|a-par|':>9s} {'a mean':>8s} {'a sd':>8s} {'a min':>8s} "
          f"{'a max':>8s} {'<0.25':>7s} {'>0.75':>7s} {'>0.95':>7s} {'>0.99':>7s} "
          f"{'cos(dev)':>9s} {'cos(a)':>8s} {'frac':>6s}")
    for name, A in data["layers"].items():               # A: (T, ...) per-task gate rows
        dev = A - parity
        print(f"{name:7s} {A[0].size:>9d} {np.abs(dev).mean():>9.5f} {A.mean():>8.4f} "
              f"{A.std():>8.4f} {A.min():>8.4f} {A.max():>8.4f} "
              f"{(A < 0.25).mean():>7.3f} {(A > 0.75).mean():>7.3f} "
              f"{(A > 0.95).mean():>7.3f} {(A > 0.99).mean():>7.3f} "
              f"{_offdiag_cos(dev):>9.4f} {_offdiag_cos(A):>8.4f} {_engaged_frac(dev):>6.3f}")
    print("  per-task mean alpha:")
    for name, A in data["layers"].items():
        print(f"    {name:7s} " + "  ".join(f"T{t}={A[t].mean():.4f}" for t in range(A.shape[0])))


def main():
    S.run_cell = lookup_only                             # never train to fill a tuning gap
    ledger = S.load_ledger()
    buf = io.StringIO()
    with redirect_stdout(buf):
        tuned_main = S.tune_main(ledger)
        tuned_nlr = S.tune_neuro(ledger, tuned_main)
    S.run_cell = None                                    # trajectories/gate runs go via cl_train

    for arm in ARMS:
        for gran in S.GRANS:
            lr, ep = S.main_point(S.cell_name("plast", arm, gran), tuned_main)
            nlr = tuned_nlr[(arm, gran)]
            cell, acc, forget, data = run(arm, gran, lr, ep, nlr)
            report(cell, acc, forget, data, nlr, ledger, lr, ep)
            # Persist the raw per-task alpha rows: any further threshold/quantile question is then a
            # numpy load rather than a re-run (asking for alpha>0.95 after the fact cost one).
            np.savez(Path(__file__).resolve().parent / f"gate_alpha_{arm}_{gran}.npz",
                     parity=data["parity"], **data["layers"])

    print("\nReading: |a-par| ~ 0 => the gate never left parity (no mechanism was exercised, the run "
          "is a uniform LR rescale of naive). |a-par| > 0 with cos(dev) ~ 1 => the gate engaged but "
          "every task learned the SAME modulation, i.e. a global LR knob carrying no task "
          "information. cos(dev) ~ 0 with a low frac => genuine per-task allocation.")


if __name__ == "__main__":
    main()
