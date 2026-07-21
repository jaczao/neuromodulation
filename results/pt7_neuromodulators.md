# pt7 — the four classic neuromodulators as pre-forward gate drivers (findings)

**Setup.** `results/pt7_neuromodulators.py`, class-IL Split MNIST, gain target on (h0,h1,out), seed 42,
lr 1e-3, ep 5, buffer 1000, **1 seed**. Driver → head `m_k(x)` (784→32→K, trained WITH REPLAY to regress a
per-sample biological signal `τ_k`) → **rank-K linear gate** `Γ_i = 1 + Σ_k m_ik P_k` (K+1 matmuls,
synapse-tractable). Arms: **nobuf** (standalone, no buffer), **buf-own** (naive main + per-task replay
meta-loss on P), **er-own** (main+P joint on the ER batch). Controls: **free** (K=4 heads, no bio target),
**5ht-const** (constant gate = scale-degeneracy null). Eval: **pred** (heads, oracle-free = THE number),
**true** (2-pass, uses labels = diagnostic upper bound), **probe** (task-decodability of `m(x)`, chance 0.20),
per-layer |gate|. Baselines: naive-sgd 0.6287 / adam 0.3900, er-sgd 0.7234 / adam 0.8946.

## Headline (confirms the pre-registered prediction)
**No biological driver beats ER on class-IL — every `er-own` cell is within ±0.02 of the same-optimizer ER
baseline, none positive beyond noise — and none adds anything standalone (`nobuf` ≈ naive).** The four
neuromodulators encode difficulty / novelty / uncertainty / reward, **not task identity**, so per the
pt5/pt6 `oracle × infer` factorization their gate is task-agnostic and cannot touch the class-IL
head-competition bottleneck. This is the SPEC's promised deliverable: the **decomposition + controls**, not
a win.

| er-own (pred / Δ vs ER) | sgd | adam |
|---|---|---|
| DA        | 0.7227 (−0.001) | 0.8787 (−0.016) |
| ACh       | 0.7132 (−0.010) | 0.8841 (−0.011) |
| NE        | 0.7210 (−0.002) | 0.8910 (−0.004) |
| NE_emb    | 0.7180 (−0.005) | 0.8896 (−0.005) |
| 5HT       | 0.7217 (−0.002) | 0.8968 (+0.002) |
| all4      | 0.7055 (−0.018) | 0.8816 (−0.013) |
| free      | 0.7264 (+0.003) | 0.8760 (−0.019) |
| DA synapse   | 0.7234 (+0.000) | 0.8986 (+0.004) |
| all4 synapse | 0.7258 (+0.002) | 0.8919 (−0.003) |

Standalone `nobuf` (best cells): DA-sgd 0.6334, all4-sgd 0.6352, NE-sgd 0.6283 — all **≈ naive 0.6287**,
i.e. the gate does nothing without replay. (Contrast pt6 `soft_mlp` buf-own 0.856: a *task-selective* gate
did work standalone; a *difficulty* gate does not.)

## Why (four controls close the interpretation)
1. **Probe ≪ pt6.** Task-decodability of `m(x)` is 0.21–0.52 (all4/K=4 highest at ~0.46–0.52) vs pt6's
   selector `infer ≈ 0.88`. The modulatory code is only weakly task-decodable → the gate is ~task-agnostic
   → no class-IL lever. This is the mechanistic reason for "≈ ER".
2. **`free` → gate 0.** The no-bio-target control drives |g| to exactly 0.000 and reproduces the baseline
   bit-for-bit (er-sgd 0.7264, er-adam 0.8760). So the ≈-baseline biological cells are **not** riding hidden
   gate capacity — there is none to ride; the heads (bio or free) correctly learn the gate is useless.
3. **`true` ≤ `pred`.** Feeding the REAL per-sample signal (diagnostic, uses labels) is **no better and
   often worse** than the head's smoothed prediction: NE er-own adam true 0.638 vs pred 0.891; all4 true
   0.691 vs pred 0.882; DA true 0.691 vs pred 0.879. So there is no "if only the head were more accurate"
   ceiling to chase — the raw signal is noisier and the head's regularization is what keeps pred at ER. For
   the low-noise signals (ACh entropy, 5HT reward, NE_emb) true ≈ pred (heads regress them near-perfectly),
   and it still ≈ ER.
4. **`5ht-const` null** behaves as a null (er-adam 0.886 ≈ ER; er-sgd 0.755, +0.032 is the usual constant-
   gate reparam noise, cf. pt6-followup-(E)).

## Per-layer emergence: the ARM, not the neuromodulator, picks the layer
The theory-predicted specialization (ACh→h0 bottom-up, NE/DA→out gain) does **not** emerge; the CL structure
dominates (as in pt6-followup-(F/G)):
- **er-own → the gate lives in the OUT layer** for every driver (e.g. ACh sgd 0.002/0.002/**0.083**; 5HT sgd
  →**0.050**; all4 sgd →**0.138**) — replay handles the features, the gate does a per-task logit nudge.
- **buf-own (standalone) → the gate moves to HIDDEN** (ACh sgd **0.567/0.615**/0.379; all4 sgd
  **0.898/1.026**/0.802) — nothing else refreshes the backbone.
- **NE_emb** is out-only by construction (0.000/0.000/0.085) and behaves exactly like the other out-gates.
Always read |gate| per layer — the single mean hides this.

## Synapse (rank-K linearity validated)
The K+1-matmul pre-forward gate runs at synapse granularity and **matches neuron** in the working regimes:
all4-synapse er-own adam 0.8919 ≈ neuron 0.8816; DA-synapse er-own adam 0.8986 ≈ neuron 0.8787. `P` is
`(K, n_syn)` ≈ 1.9M (K=4), NOT the 374M content-projection blow-up pt5 flagged — the neuromodulator framing
fits gain-synapse. all4-synapse buf-own adam 0.562 (+0.172 over naive-adam) is the one standalone bump, but
still ≪ ER and buf-own-noisy (sgd companion 0.535).

## Tonic / scalar variants are degenerate (and numerically collapse under SGD)
`ACh_vol` (√ema volatility) and `NE_rise` (relu(emafast−emaslow)) are **scalar / constant-per-step** → after
per-driver standardization their per-sample variance → 0, so `τ/√var` explodes: |g| blows to 10–17 and
accuracy collapses (ACh_vol er-adam 0.184, er-sgd 0.098; NE_rise er-sgd 0.098, nan |g|). Under Adam NE_rise
survives at 0.857 (moment normalization absorbs the blow-up). This is the "degenerate at eval" family from
the SPEC, failing loudly: a signal with no per-sample content cannot drive a per-sample gate. `DA_step`
(per-sample) is fine and ≈ ER (er-adam 0.887), unlike its tonic cousins.

## Verdict
pt7 is a controlled **negative**: the four neuromodulators, as pre-forward drivers of a per-sample gain gate
(neuron AND synapse), reproduce ER but never beat it on class-IL, and add nothing standalone — because
difficulty/novelty/reward is not task-identity (probe ≪ pt6). The mechanism is sound and synapse-tractable;
the *signal* carries no retention/selection information the class-IL bottleneck needs. This is consistent
with the whole pt2→pt6 arc: replay (or a task-selective gate, pt6) is the lever; a modulatory difficulty
code is not. **Caveats:** 1 seed; buf-own high-variance (the scattered >0.02 buf-own cells are naive-backbone
noise, all ≪ ER); oracle-FREE by construction (a genuine improvement in honesty over pt5, no new class-IL
win); the reportable class-IL headline across the project stays pt6 (ER-parity oracle-free selector) / pt5
(disjoint gain+ER under the oracle).
