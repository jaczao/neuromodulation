# pt7 follow-up — neuromodulators on a PLASTICITY target + two new gate FORMS (temp / slope)

User-requested. class-IL Split MNIST, **er-own**, seed 42, lr 1e-3, ep 5, buffer 1000, **1 seed**,
**non-standardised** drivers. Self-contained: `results/pt7_plast_tempslope.py` (reuses the pt7
`Net`/`Reservoir`/`Signals`/`Heads`/`NEDriver`/`StatefulDriver` primitives). Ledger
`pt7_plast_tempslope_results.tsv`, log `pt7_plast_tempslope.log`.

Baselines (this harness, matched): naive 0.6287 / 0.3900, **er 0.7234 (SGD) / 0.8946 (Adam)**. For the
tuned reference: CLAUDE.md's val-tuned ER-SGD = **0.9034** (lr 0.03), ER-Adam ≈ 0.8975.

Both parts land squarely in the **pt7 controlled-negative**: nothing beats the best replay baseline; every
apparent "win" is an untuned-SGD LR artifact, and every gate that engages a large non-standardised driver
collapses. Set 1 was run BOTH untuned (lr=1e-3, the pt7 default) and **tuned (main SGD lr=0.03)** — the untuned
+0.13 "boost" over ER-SGD 0.7234 DISSOLVES to ≈0 at the tuned operating point (all cells within ±0.007 of tuned
ER-SGD 0.9034), which is the decisive proof that the boost was pure LR-compensation, not plasticity.

---

## SET 1 — plasticity target (SGD main net, lookahead-trained P)

Drivers `da_fast, ach_ema, ach_gru, 5ht_ema` (all K=1, head-predicted, trained WITH REPLAY) drive a
per-neuron / per-synapse / **global (scalar→scalar)** learning-rate gate `α = exp(mbar·P)` (P zero-init ⇒
α=1 parity). All three mechanisms train P by the **pt5 lookahead meta-loss** (the gate is on the gradient, so
P gets no grad from the main loss under in-place SGD gating): `W_fast = W.detach() − lr·(α⊙g)`, meta-CE on the
same ER batch (replay ⇒ retention meta-loss) → Adam on P, then commit the real gated SGD step. Eval is the
**plain** net (the gate only shaped LEARNING) ⇒ pred = plain acc.

| driver  | neuron | synapse | global | probe | `|α−1|` h0/h1/out (neuron) |
|---------|:------:|:-------:|:------:|:-----:|:--------------------------|
| da_fast | 0.7154 | 0.7139  | 0.7152 | 0.28  | 0.05 / 0.05 / 0.08 |
| ach_ema | **0.8516** | 0.8331 | **0.8497** | 0.28 | 1.74 / 1.32 / 0.75 |
| ach_gru | **0.8491** | 0.8274 | **0.8448** | 0.26 | 1.88 / 1.42 / 0.76 |
| 5ht_ema | 0.7665 | 0.7416  | 0.7676 | 0.27  | 0.18 / 0.16 / 0.11 |

**The apparent win (ach +0.11–0.13 over ER-SGD) is a GLOBAL LR-BOOST ARTIFACT, not structured plasticity —
four independent tells:**

1. **`global` (one scalar) ≈ `neuron`, and both ≥ `synapse`, for EVERY driver** (ach_ema 0.8497 ≈ 0.8516 >
   0.8331; 5ht 0.7676 ≈ 0.7665 > 0.7416; da_fast 0.7152 ≈ 0.7154 ≈ 0.7139). Per-neuron/per-synapse *structure*
   adds nothing (per-synapse mildly HURTS) — a single scalar LR multiplier reproduces the whole effect.
2. **probe ≈ chance (0.25–0.28 vs 1/5=0.20)** — the gate is NOT task-decodable, so it is not doing
   task-selective retention; it is a roughly task-agnostic LR knob.
3. **accuracy tracks the driver's MEAN MAGNITUDE, not its information content.** The tonic ACh drivers
   (ema-entropy, sustained positive) push `α≈exp(1.6)≈2.6×` the LR → big boost; `5ht_ema` (moderate) → mild;
   the phasic `da_fast` (mean ≈ 0) cannot sustain a boost → `α≈1` → **≈ ER-SGD**. The lookahead simply learns
   "lr=1e-3 is too small under SGD, crank it up," which only a nonzero-mean driver can express.
4. **All below the tuned/Adam ceiling:** ach 0.85 < ER-Adam 0.8946 < tuned ER-SGD 0.9034. The boost only
   *partially* closes the untuned-SGD gap (2.6× lr vs the 30× the tune wanted), exactly like every prior pt7
   "SGD +ER boost" that dissolves once ER-SGD is tuned (see CLAUDE.md pt7_tuned_syn).

So neuromodulator-gated plasticity is the same controlled negative: the lever is the LR (which tuning ER
supplies directly), not a biologically-patterned plastic subset. Corroborates pt5's whole plasticity arc
(learned-P plasticity never beat replay) and pt7's "capacity/LR closing SGD under-fit, not a class-IL lever."

### Set 1 TUNED confirmation (`--part plast-tuned`) — the boost DISSOLVES, proving the artifact
Re-ran Set 1 with the MAIN SGD net at the val-tuned **lr=0.03/ep5** (the pt7_tuned_syn ER-SGD operating point),
neuromod net (gate P + head) left at the default neuro_lr=1e-3 (pt7_tuned_neuro methodology; main/neuro
decoupled). Anchor reproduces bit-exact: **er sgd-tuned = 0.9034** (= pt7_tuned_syn). naive sgd-tuned = 0.5548
(higher lr forgets harder, per CLAUDE.md).

| driver  | neuron | synapse | global | Δ vs ER-SGD 0.9034 | `|α−1|` h0/h1/out (neuron) |
|---------|:------:|:-------:|:------:|:-------------------|:--------------------------|
| da_fast | 0.9049 | 0.9040  | 0.9029 | +0.0015 / +0.0006 / −0.0005 | 0.06 / 0.06 / 0.06 |
| ach_ema | 0.9017 | 0.9010  | 0.9019 | −0.0017 / −0.0024 / −0.0015 | 0.16 / 0.09 / 0.03 |
| ach_gru | 0.8967 | 0.8964  | 0.8967 | −0.0067 (all three)         | 0.09 / 0.05 / 0.02 |
| 5ht_ema | 0.9014 | 0.9015  | 0.9014 | −0.0020 (all three)         | 0.01 / 0.01 / 0.00 |

**Every cell is within [−0.0067, +0.0015] of tuned ER-SGD — the +0.11–0.13 "win" is entirely GONE.** And the
gate magnitude COLLAPSED (ach_ema neuron `|α−1|` 1.74/1.32/0.75 → 0.16/0.09/0.03): with the main lr already at
0.03 the lookahead no longer wants to amplify it, so the gate falls back to ≈parity and its effect vanishes —
the direct mechanistic proof that the untuned `α≈2.6×` gate was pure LR-compensation. At a properly-tuned SGD
operating point, neuromodulator-gated plasticity = ER (best +0.0015, worst −0.0067, all inside the ~±0.007
1-seed noise floor). This is the honest test the untuned run could not give.

---

## SET 2 — two gate FORMS (Adam main net, P joint via the ER loss)

Drivers `ne, ne_emb_all, ne_vecproj, nerisez, nerisez_gru` under two positive-scalar gate forms, `exp(m·p)`,
p zero-init (parity), trained jointly by the main ER loss (in-forward):
- **temp** — `logits_out *= exp(m·p_out)` (softmax temperature, out only). A uniform positive scale is
  **argmax-invariant at eval**, so temp's only effect is via TRAINING dynamics.
- **slope** — `h0 *= exp(m·p_h0); h1 *= exp(m·p_h1)` (per-hidden-layer ReLU-slope gain; affects eval too).

| driver       | temp | slope | probe | note |
|--------------|:----:|:-----:|:-----:|------|
| ne           | 0.3874 | **0.1037** | 0.37 | large non-std NE ⇒ exp gate blows up (`|α−1|≈1.0`) |
| ne_emb_all   | **0.1009** | **0.1006** | 0.29 | unbounded `‖h1−mean‖` ⇒ full collapse both forms |
| ne_vecproj   | 0.8837 | 0.8860 | 0.69 | bounded 32-d proj ⇒ moderate gate ⇒ **≈ ER-Adam** |
| nerisez      | 0.8905 | 0.8869 | 0.24 | intrinsic z-score ⇒ small gate ≈ parity ⇒ **≈ ER** |
| nerisez_gru  | 0.8929 | 0.8880 | 0.39 | dead-ish gate ⇒ **≈ ER** |

**Two clean results, no win:**

1. **Bounded/small drivers → gate ≈ parity → ≈ ER-Adam** (ne_vecproj, nerisez, nerisez_gru all 0.886–0.893 vs
   0.8946). No gate form beats replay; the healthy cells simply *are* ER.
2. **Large unbounded NON-STANDARDISED drivers → the `exp` gate blows up → COLLAPSE** (ne, ne_emb_all → chance).
   This is the pt7 "standardize or the gate blows up" rule, now for the temp/slope forms: the user's explicit
   *non-standardised* choice makes the failure direct. **temp collapses LESS than slope for `ne`**
   (0.39 vs 0.10) — exactly because temperature is argmax-invariant at eval, so a blown-up temperature only
   corrupts *training* (per-sample loss reweighting), while a blown-up slope also corrupts the eval forward
   (destroys the hidden representation). `ne_emb_all` collapses fully both ways (its magnitude is large enough
   to wreck training regardless of the eval-time invariance).

The temp/slope distinction is thus interpretable: **temperature is a training-time per-sample loss weighting
(no eval-time lever), slope is a genuine forward gain.** Neither adds anything over replay; the notable effect
is a failure mode (collapse), and the safe cells reproduce ER.

---

## Verdict
Neither the plasticity target nor the temp/slope gate forms produce a class-IL lever. Plasticity's "boost" is
a global LR-scaling artifact of the untuned SGD baseline (global scalar reproduces it; probe at chance; below
tuned/Adam ER); the gate forms either tie ER (bounded drivers) or collapse (large non-standardised drivers,
temp<slope by argmax-invariance). Project class-IL headline unchanged: **replay is the only lever.** 1 seed;
oracle-free by construction.
