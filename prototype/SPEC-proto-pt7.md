# SPEC-proto-pt7 — the four classic neuromodulators as PRE-FORWARD gate drivers

## Context (why pt7)
pt5/pt6 fixed the axes as **driver → mechanism → projection → target** with the driver = `task_id`.
That driver is an ORACLE (the true task index), which is why the reportable pt5 win (er+gain ≈ 0.99) and
even pt6's oracle-free selector all revolve around *task identity*. pt7 changes the **driver**: instead of
task identity, feed the gate the **four classic neuromodulators** — **DA** (dopamine), **ACh**
(acetylcholine), **NE** (norepinephrine), **5-HT** (serotonin) — as difficulty / uncertainty / novelty /
reward signals.

**The pt7 constraint (the whole point of this part).** pt7 restricts to mechanisms whose **projected gate
signal is computable BEFORE the main forward pass**, so that **both** gain granularities work:
`gain-neuron` AND `gain-synapse`. A signal that needs the current sample's own logits/loss is *not*
pre-forward; pt7 obtains every signal either (a) from a small **head on `x`** (predicted), or (b) from a
partial forward whose result gates a *later* layer (NE-emb → out gate). This is what lets pt7 reuse pt6's
per-sample gate at synapse granularity without the `(B, d_out, d_in)` weight expansion (see "rank-K gate").

**Pre-registered prediction (do NOT post-hoc this).** pt5/pt6 established that accuracy factorises as
≈ `oracle_acc × infer_acc` because a helpful gate must be **task-DIFFERENTIATED** with near-zero
misrouting tolerance. DA/ACh/NE/5-HT encode *difficulty / novelty / reward*, **not task identity**, so the
expectation is: their gate is roughly task-agnostic → pt7 lands **≈ ER at best**, NOT a new class-IL lever.
The deliverable is therefore the **decomposition** (how much of any effect is a genuine low-D modulatory
code vs. plain extra gate capacity), the **biological-plausibility** framing, and the **per-layer
emergence** question (does NE/DA land on the `out` gate, ACh on `h0`, as theory predicts?) — not a headline
number. If a cell *does* beat ER, that is a surprise and needs ≥3 seeds before it is believed.

**Relationship to pt2-Iteration-3 (surprise/uncertainty/activation_stats, all rejected).** That rejection
is CONFOUNDED with the wrong target: it gated one hidden layer via `weight_mask` *before* we knew the
shared head is the class-IL bottleneck (van de Ven & Tolias). pt7 retries loss/uncertainty-derived drivers
on the **gain target over (h0,h1,out)** — which reaches the head — so it is a legitimate retry, not a
re-run. Say so in the findings.

## Non-negotiable inheritance from pt5/pt6 (carry verbatim)
- Target: **gain** on layers **(h0, h1, out) = (0,2,4)**; granularities **{neuron, synapse}**.
- Arms: **buf-own** (naive main net + per-task replay meta-loss on the gate `P`), **er-own** (main net
  + gate `P` joint on the ER batch, own-task gating), and **nobuf** (standalone, NO buffer anywhere: naive
  masked-CE main + gate jointly on the current task; heads regress `τ` on the current task only — the
  pt6-followup-(B) stress test for whether a head-driven gate collapses without replay). Baselines
  **naive**, **er** (no gate).
- **{sgd, adam}**, class-IL Split MNIST, seed 42, lr 1e-3, ep 5, buffer 1000, **1 seed** (≥3 if a buf-own
  cell becomes a headline — buf-own is high-variance, pt6).
- **Heads MUST be trained WITH REPLAY** (pt6-followup-(B): a selector/head trained without the buffer
  collapses to chance 0.198 — it forgets old-task statistics; the same failure mode applies to any head
  learned across the task sequence).
- **Report gate magnitude PER LAYER, never a single mean** (pt6-followup-(F/G): a single mean hides the
  load-bearing out-gate).
- The oracle caveat language carries, BUT note that pt7 drivers are **oracle-free BY CONSTRUCTION** (the
  gate is a function of `x` via the heads, or of a partial forward; no task id is consumed at eval).

## The rank-K linear gate (why pre-forward ⇒ synapse-tractable)
Gate applied per sample: **`Γ_i = 1 + Σ_k m_ik · P_k`**, i.e. `(Γ_i ⊙ W) x_i + b`.
- **`i`** = sample index in the batch (the gate is per-sample).
- **`k`** = which neuromodulator, `k ∈ {DA, ACh, NE, 5-HT}` (K ≤ 4; K=1 for a single driver, K=4 for `all4`).
- **`m_ik`** = scalar output of modulator `k` for sample `i` (its novelty / difficulty / reward value).
- **`P_k`** = a learned gate pattern over the target's elements (neuron: 810; synapse: `n_syn`). One per `k`.

Because the gate is **linear in `m`**:
`(Γ_i ⊙ W) x_i + b = (W x_i + b) + Σ_k m_ik · [(P_k ⊙ W) x_i]` — **K+1 matmuls per layer, exact,
differentiable in both `m` and `P`, and NO per-sample weight expansion** (same linearity as pt6's
soft-blend, but parity `1` is explicit so it does not need `Σ_k p = 1`). Consequences:
- Per-synapse `P` is `(K, n_syn)` ≈ **1.9M** at K=4 — *smaller* than pt6's `(T=5, n_syn)` task lookup — and
  the content projection is `784 → K → n_syn`, which **eliminates the 374M-param blow-up** that made pt5's
  content drivers per-synapse-infeasible. The neuromodulator framing fits gain-synapse *better* than task_id
  did. A K-dim gate is the biologically-motivated version of pt6's hand-waved rank-64 low-rank projection.

## Axis 1 — DRIVERS (the four neuromodulators) and their signals
Each driver defines a per-sample **true signal** `τ_k(x)` (computed at train from a **plain, unmodulated,
detached** forward — never the gated forward, which would be circular) plus running EMA state. A small
**head** `m_k(x): 784→h→1` is trained (with replay) to regress `τ_k`; the head drives the gate at train AND
eval (consistent, per-sample, oracle-free, pre-forward). EMA rates: fast β=0.1, slow β=0.02.

Let `ℓ_i` = per-sample masked CE (the loss the net optimises), `H_i` = full-softmax predictive entropy,
`h1_i` = last hidden. State: `ema_fast, ema_slow` (loss), `ema_sq` (→ `ach_vol = √ema_sq`), `ema_reward`,
`prev_loss` (last step's batch-mean loss), `mean_h1`.

- **DA — reward prediction error (phasic).** Native to the `out`/logit gate (β) and to plasticity.
  - `DA` (default): `τ = (ℓ_i − ema_slow) / (std_batch + ε)` — normalised loss deviation.
  - `DA_step` (ablation): `τ = (ℓ_i − prev_loss) / (std_batch + ε)` — one-step change.
  - Pre-forward: **at eval there is no loss → predict it** via `m_DA(x)`.
- **ACh — expected uncertainty.** Native to `h0` (up-weights bottom-up sensory drive).
  - `ACh` (default): `τ = H_i` — per-sample predictive entropy (running baseline `ema_H`).
  - `ACh_vol` (ablation): `τ = √(ema_slow((ℓ̄ − ema_fast)²)) = ach_vol` — expected volatility; **scalar /
    tonic** (constant per step) → no per-sample variation → expected to behave like the null.
- **NE — unexpected uncertainty / adaptive gain.** Native to the `out` gate (gain β; Aston-Jones & Cohen).
  - `NE` (default): `τ = relu((|DA_i| − ach_vol) / (ach_vol + ε))` — surprise in excess of expected noise
    (Yu & Dayan). Per-sample (uses per-sample `DA_i`, scalar `ach_vol`).
  - `NE_rise` (ablation): `τ = relu(ema_fast − ema_slow)` — loss rising = boundary/change detector;
    **scalar / tonic** → null-like.
  - `NE_emb` (default, distinct mechanism): `τ = ‖h1_i − mean_h1‖` — embedding novelty of the current
    sample. Because the embedding is the **last hidden**, this gates **what comes after it = the `out`
    layer only** (a *within-forward* signal: run up to `h1`, measure novelty, gate the logits). No head
    needed at eval (it is computed during the forward); a head-predicted variant is optional.
- **5-HT — tonic average reward / time horizon.** Native to global/tonic → the **degenerate-null neighbour**.
  - `5HT` (default): `τ = −ℓ_i` — per-sample reward; the tonic level is `ema_reward` (a critic `V(x)`
    predicts it at eval). Predict reward at eval when the loss is unavailable.
  - Its pure-tonic form (scalar `ema_reward` broadcast) is the `5ht-const` null below.

## Axis 2 — SIGNAL AVAILABILITY (how the pre-forward value is obtained)
- **predicted-head (family B, DEFAULT).** `m_k(x)` regresses `τ_k` (with replay); drives the gate at train
  & eval. Oracle-free, one forward for the gate + one detached plain forward for the target.
- **two-pass / true-at-train (family C, ablation).** Pass-1 unmodulated → true `τ_k` → pass-2 gated by the
  true signal; eval still needs the head (no labels). Most faithful signal; 2× forward. Held as a follow-up.
- **tonic-constant (null).** A scalar EMA broadcast / a single learned constant gate — degenerate at eval.

## Axis 3 — TARGET (unchanged): gain (h0,h1,out), {neuron, synapse}.

## Controls (MANDATORY — without them a pt7 number is uninterpretable)
- **`free`.** Identical K=4 bottleneck and gate `Γ = 1 + Σ m_k(x) P_k`, but the heads have **no biological
  target** — trained end-to-end by the main/meta loss only. Separates "the neuromodulator *definition*
  carries information" from "4 extra dims of learned gate capacity." If `free` matches the biological
  drivers, the biology added nothing.
- **`5ht-const`.** A single **learned constant** gate vector (`m ≡ 1`, `Γ = 1 + P`), no `x`-dependence —
  the pt6-followup-(E) **scale-degeneracy null** (the jointly-trained backbone just absorbs it). The floor
  any "extra constant gate" gives.
- **task-discriminability probe.** A linear probe from `m(x) ∈ ℝ^K` to task id on the test set; report its
  accuracy next to pt6's selector `infer ≈ 0.88`. This is the diagnostic that **explains** the result: if
  the modulatory code is not task-decodable, the pre-registered "≈ ER" prediction is confirmed mechanistically.

## Grid (1 seed; class-IL; seed 42, lr 1e-3, ep 5, buffer 1000)
- **Baselines:** naive, er × {sgd, adam}.
- **Main (neuron):** drivers {DA, ACh, NE, NE_emb, 5HT, all4, free} × {buf-own, er-own} × {sgd, adam}.
- **Nulls/tonic (neuron, er-own × {sgd, adam}):** {DA_step, ACh_vol, NE_rise, 5ht-const}.
- **Synapse:** {DA, all4, free} × er-own × {sgd, adam} (+ all4 buf-own) — demonstrate the rank-K synapse
  path runs and matches neuron in the working regimes.

## Eval / metrics (per cell)
- **`pred`** — gate from the heads `m(x)`. **Oracle-free; THE reportable number.** class-IL 10-way argmax.
- **`true`** — gate from the true per-sample signal via a two-pass eval (uses test labels). **Diagnostic
  upper bound** ("if the head were perfect"); clearly flagged as label-using.
- **`probe`** — task-decodability of `m(x)` (above).
- **per-layer |gate|** — mean |applied multiplicative deviation| at `h0`, `h1`, `out` (neuron) /
  `net0, net2, net4` (synapse). Never a single mean.
- Report against naive/er baselines and note `pred − er` (the honest, same-metric delta).

## Promotion policy (REQUIRED — supersedes pt5/pt6 "winners only")
After pt7, the migration into `neuromod.py`'s `--neuromod-drivers` registry promotes **ALL** explored
mechanisms — **winners AND non-winners** — from pt5, pt6, and pt7, not just the reportable wins:
- pt5/pt6 driver mechanisms: `onehot`, `mean_image` ({lin, mlp} × ±center), `soft_mlp`, `embedding`.
- pt7 neuromodulator drivers: `DA`, `ACh`, `NE`, `NE_emb`, `5HT` (+ the `DA_step`/`ACh_vol`/`NE_rise`
  tonic variants) and the controls `all4`, `free`, `5ht-const`.
Rationale: the thesis needs the full ablation set **live and reproducible**, not reconstructed from the
`results/` study scripts; a non-winning mechanism is a documented negative result and must be runnable via
the same one-flag path as a winner. (Promotion is still deferred to the post-iteration migration; pt7 itself
delivers the self-contained study, as pt5/pt6 did.)

## Deliverable
`results/pt7_neuromodulators.py` (self-contained, repo-relative, runnable) + `.log` + `.md` findings, plus a
CLAUDE.md "Known gotchas" bullet and a `prototype/iteration-notes.md` "pt7" section. 1 seed; oracle-free by
construction; buf-own high-variance (report ≥3 seeds if a buf-own cell headlines).

## Plan of record (execute in order)
1. Base module: heads (`784→h→K`), rank-K gate for **neuron** (`raw = m @ P`) and **synapse** (the K+1-matmul
   linearity), the signal/EMA computation, controls (`free`, `5ht-const`), the probe, per-layer |gate|.
2. Smoke-test each code path (1 epoch).
3. Run the grid — baselines + neuron main first (fast), nulls, then synapse (heavier) — 1 seed, log to
   `results/pt7_neuromodulators.log`.
4. Write `.md` findings + CLAUDE.md gotcha + iteration-notes; commit + push **when asked**.
