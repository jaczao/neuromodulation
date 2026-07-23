# pt7 signal-net + GRU drivers — findings

Four user-requested mechanisms on top of pt7. All **class-IL Split MNIST, gain (h0,h1,out), er-own,
ADAM, seed 42, lr 1e-3, ep 5, buffer 1000** (Adam-ER is operating-point-insensitive: untuned 0.8946 ≈
tuned 0.8975, so this is comparable to the tuned point). Baselines: **er-adam 0.8946**, naive-adam 0.390,
pt7 all4 neuron er-own adam 0.8816. Ledger `pt7_signalnet_results.tsv`; module `pt7_signalnet.py`.

**Headline: nothing beats ER (0.8946).** Every mechanism is either INERT (gate stays at parity → = ER)
or, once forced to engage, ACTIVELY WORSE than ER. Consistent with the whole pt7 controlled-negative:
a difficulty / uncertainty / novelty gate is not task identity, and replay is the only class-IL lever.

## Task 1 — neuromodulator-net RESET at each task switch (all4, 3 seeds, standardised)
Reset the neuromodulator net (heads `m_k(x)` + gate `P`) to its start-of-training weights + fresh
optimizer state at every task boundary (t>0); main net NOT reset, buffer persists (task-2 training still
sees task-1 replay under the freshly-reset gate); inference uses the last (end-of-task-5) net.

| seed | reset-ON pred | \|g\|(h0/h1/out) | probe |
|---|---|---|---|
| 42 | 0.8906 | 0.000/0.001/0.001 | 0.337 |
| 43 | 0.8923 | 0.000/0.001/0.003 | 0.310 |
| 44 | 0.9048 | 0.000/0.001/0.001 | 0.290 |
| **mean±std** | **0.8959 ± 0.0078** | ~0 | — |

vs ER-adam **0.8946**, and no-reset all4 (seed42) 0.8816 (\|g\| 0.016/0.036/0.045, probe 0.46).
**Reset ≈ ER.** Mechanism: resetting `P` every task stops it accumulating magnitude, so \|g\|→~0.0006
(≈60× smaller than no-reset) and the gate is effectively parity — the run is just ER. The reset even
nudges seed-42 up (0.8816→0.8906): the small nonzero gate the no-reset all4 builds was itself very
mildly hurting, and killing it returns to clean ER. The probe falls toward chance (0.46→0.29–0.34): a
per-task-reset gate is less task-decodable. Sanity: reset-OFF reproduces pt7 all4 er-own adam **0.8816
bit-exact** (harness faithful; splitting `P` into its own same-lr Adam is identical to keeping it in
`main_opt`).

## Tasks 2–4 — GRU-on-all4, signal net, signal-net-as-GRU (zero-init) = the DEAD-SADDLE
`gru-all4` (GRU on the predicted all4 vector), `signalnet` (23-signal MLP → low-D K → upproject), and
`signalnet-gru` (signal net → GRU → gate) **all pin \|g\| to exactly 0.000 → = ER** across
{neuron, synapse} × K{4,16} × std{on,off}:

| mechanism | cells | pred range |
|---|---|---|
| gru-all4 | neuron, synapse | 0.8856 |
| signalnet | neuron/synapse × K{4,16} × std{0,1} | 0.8908–0.8943 |
| signalnet-gru | neuron × K{4,16} × std{0,1} | 0.8947–0.8985 |

**Why exactly parity: a double-zero-init saddle.** Stacking a zero-init module (GRU-out / signal-net-out,
zero-init to give γ=1 parity) BEFORE the zero-init gate `P` makes `dL/dP ∝ m = 0` and
`dL/d(module) ∝ P = 0` — neither can bootstrap off zero, so the gate is frozen at parity for the whole
run. This is exactly pt7's `free` control ("→ gate 0, \|g\|=0.000, baseline bit-exact"): the signal net
is trained end-to-end like `free`. Plain all4 escapes this only because its heads have an MSE target that
forces `m≠0`. So the zero-init cells do NOT test whether the signals help — the gate never engages.
Standardization, K, and granularity are all irrelevant while inert.

## ENGAGE re-run — break the saddle (module output normal-init, `P` still zero-init → parity at step 0)
Give the module's OUTPUT layer normal init so `m≠0` (P still zero-init, so γ=1 at step 0 is preserved,
but `P` now bootstraps). The gate engages — and **every cell is now WORSE than ER**:

| mechanism | pred | \|g\|(h0/h1/out) | vs ER |
|---|---|---|---|
| gru-all4 eng | 0.8789 | 0.017/0.042/0.048 | −0.016 |
| signalnet-gru K4 eng | 0.8657 | 1.52/1.32/1.94 | −0.029 |
| signalnet-gru K16 eng | 0.8799 | 1.59/1.83/2.92 | −0.015 |
| signalnet K16 eng | 0.8063 | 1.97/2.78/2.90 | −0.088 |
| signalnet K4 eng | **0.5215** | 0.47/0.53/0.80 | **−0.373** |

When the signal net actually drives the gate it over-modulates / injects noise and drops below plain ER;
the raw K=4 signalnet collapses catastrophically (0.52). More gate magnitude → more harm. The GRU's
temporal smoothing partly stabilises a large gate (signalnet-gru > signalnet at matched K) but still
< ER. So the signal net is net-negative once it does anything.

## Task 5 — h1-gate: a sibling net gates the main net's h1 (`--part h1gate`)
A second net with the main net's architecture up to h1 (784→400→400), same input `x`, output squashed to
[0,1] by sigmoid; its 400-d output gates the main net's h1 by element-wise multiply
(`logits = l2(h1 ⊙ σ(g1(relu(g0(x))))))`). Trained JOINTLY with the main net by the ER loss (no separate
target). 1 seed, adam, er-own.

**pred 0.8956, mean h1-gate value 0.281 ⇒ ≈ ER (0.8946, +0.001, noise).** Unlike the dead-saddle cells this
gate is genuinely ACTIVE (mean 0.28, far from parity 1.0 — it strongly suppresses h1), yet accuracy is still
ER: the jointly-trained backbone simply absorbs the uniform [0,1] rescale (Adam handles it — cf. the
`bounded01` gotcha: sigmoid(0)=0.5 halves activations but Adam's per-parameter normalisation compensates),
so the learned h1 gate adds nothing over replay. The pt6-followup-(E) scale-degeneracy again: a
jointly-trained multiplicative gate is reabsorbed by the weights it multiplies — gate ≠ memory.

## Task 6 — h1-gate in the STANDARD regime (`--part h1gate-std`)
Same sibling-net h1 gate, but full-MNIST single-task 10-way CE (project goal #2: neuromod must not hurt
plain accuracy). seed42, adam, ep5, untuned. **vanilla 0.9766 vs h1-gate 0.9759 (Δ −0.0007 = neutral)**;
the gate is genuinely active (mean h1-gate 0.639) but standard accuracy is preserved — the h1 gate is
HARMLESS in the single-task regime, same as every other neuromod mechanism in pt4/pt7-standard (goal #2
holds). (Untuned matched budget; a proper standard sweep would lift both ~0.98, cf. pt7 std-tuned.)

## Task 7 — all4 with a FIXED RANDOM projection (`--part all4fixed`, 3 ways × 2 seeds)
pt7 all4 er-own, but the rank-K projection `P` is **FIXED RANDOM (frozen, never gets a gradient)** instead
of learned. The heads `m_k(x)` still regress the standardized bio τ (with replay); only the per-sample
coefficients `m` and the backbone adapt — the modulation rides random-but-fixed directions. adam, class-IL.

| way | seed42 | seed43 | mean | mean \|g\| |
|---|---|---|---|---|
| neuron gaussian scale 0.1 | 0.8857 | 0.8972 | 0.8915 | ~0.04 |
| neuron gaussian scale 0.3 | 0.8883 | 0.8963 | 0.8923 | ~0.09 |
| synapse gaussian scale 0.1 | 0.8919 | 0.8891 | 0.8905 | ~0.003 |

vs ER-adam **0.8946** and LEARNED all4 er-own adam 0.8816 (1 seed). **All three ways ≈ ER** (−0.002 to
−0.004 on the means), across scale (0.1/0.3) and granularity (neuron/synapse). The random gate is genuinely
active (nonzero \|g\|, scaling with the projection scale as expected) yet neither helps nor hurts.
**Crucially the fixed-random projection (0.891) ≈ / marginally ABOVE the learned projection (0.8816)** — so
the learned `P` was never the lever: injecting the bio signals along random frozen directions does exactly
as well as learning the directions, and both = ER. This is the tightest isolation yet that the projection
*structure* carries nothing on class-IL — the backbone absorbs whatever direction the gate points in, and
replay is the only lever.

## Verdict
Every mechanism lands in the pt7 controlled-negative: **inert → = ER, engaged/active → ≤ ER, never > ER.**
The rich 23-dim difficulty/uncertainty/novelty feature set, the low-D bottleneck, the extra gate capacity
(K=16), statefulness (GRU), per-task neuromodulator resetting, a direct active h1 gate, AND a fixed-random
(vs learned) projection each add **nothing** over replay on class-IL Split MNIST — the same conclusion as
the neuromodulator drivers, the tonic variants, and the UNIFY-12 composite: a difficulty/novelty code is
not task identity, the gate's projection structure is not the lever, and replay is the only lever. In the
STANDARD regime the h1 gate is harmless (goal #2). Project class-IL headline unchanged. 1 seed except reset
+ all4fixed (2–3 seeds); oracle-free by construction.
