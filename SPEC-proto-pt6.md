# SPEC-proto-pt6 — content & inference-net mechanisms for the task_id driver + an eval-resolution axis

## Context (why pt6)
pt5 fixed the axes as **driver → mechanism → projection → target** with `neuromod_drivers="task_id=onehot"`
(driver = `task_id`, mechanism = `onehot` = independent per-task gate rows). The reportable pt5 win
(er+gain ≈ 0.99) is **oracle-dependent** (task id at eval). The pt5 driver-representation study
(`results/pt5_driver_repr.py` + `.md`, CLAUDE.md gotcha) showed a **centered mean-image** mechanism
MATCHES onehot **under the oracle**, but every oracle-free eval (per-image, hard nearest) falls **below
plain ER** — the mechanism is task-IL, capped by ~76% nearest-prototype task inference. It never tried
the **soft** resolution, nor a **learned** task inference.

pt6 (a) promotes the mean-image code into the driver/mechanism framing, (b) adds the soft eval-resolution
the study lacked, and (c) adds two mechanisms that **learn** the task selection (a soft task-inference MLP,
and its hidden embedding). Standalone study module extending pt5_driver_repr.py (same pattern as pt5
studies); promoting winners into `neuromod.py`'s `--neuromod-drivers` is deferred, as in pt5.

## Axis 1 — MECHANISMS of the `task_id` driver (how task_id → per-task gate)
- `onehot` (existing, REFERENCE): `raw_t = P[t]`, independent per-task rows.
- `mean_image` (NEW): `raw_t = proj(μ_t)`, `μ_t` = task-mean image, ± centering (`μ_t − mean_t μ_t`;
  inter-task cos 0.82 → −0.24). `proj ∈ {lin, mlp}` (784→gate, and 784→128→gate).
- `soft_mlp` (NEW): gate table = learned per-task `P` (onehot-style rows) **plus** a separate
  task-inference net `g(x): 784→128→T` trained WITH REPLAY (cross-entropy on the reservoir, so it does
  not forget old-task inference). Train uses the true task's row `P[task]`; eval blends (Axis-2 soft).
- `embedding` (NEW): take a HIDDEN layer `e(x)` of that same inference net `g` (a learned continuous task
  embedding) and gate = `proj(e(x))`, `proj ∈ {lin, mlp}` (128→gate). Per-image, **inherently oracle-free**
  (no discrete task blending; the embedding replaces the prototype as the driver value).

## Axis 2 — EVAL RESOLUTION (how the driver value is chosen at inference; training unchanged, ONE training
run produces ALL modes). Applies to `mean_image` (and `onehot` has only `oracle`).
- `oracle`: gate = `raw_true-task` (needs the task id) — REFERENCE.
- `per-image`: driver = the test image; gate = `proj(x − center)`.
- `nearest`: hard nearest-prototype task inference `nn = argmin_t ‖(x−center) − μ_t‖`; gate = `raw_nn`.
- `soft-nearest(τ)`: `p(t|x) = softmax(−‖(x−center) − μ_t‖² / τ)`; gate = `Σ_t p_t · raw_t`.
`soft_mlp` defines its own resolution `p(t|x)=softmax(g(x))` → `Σ_t p_t·P[t]` (report oracle + this).
`embedding` is per-image continuous (report its per-image number; oracle N/A by construction).

## Axis 3 — TARGET (unchanged from pt5): gain, gate layers **(h0, h1, out) = (0,2,4)**.
- granularity `neuron` (per-unit: 400+400+10 = 810 gains; gate the two hidden activations AND the 10 logits).
- granularity `synapse` (per-weight). NB the per-synapse **content** projection (784 → n_syn) is huge
  (net0+net2+net4 = 477 600 synapses ⇒ lin proj ≈ 374M params, 780× the net — pt5's "does not scale to
  per-synapse"). pt6 keeps it runnable with a **low-rank** content projection (rank 64) for
  `mean_image`/`embedding` synapse cells, and full lookup `P` (T×n_syn) for onehot/soft_mlp. Documented;
  a caveat, not a clean comparison to neuron.

## Grid (1 seed; class-IL Split MNIST; seed 42, lr 1e-3, ep 5, buffer 1000; ORACLE caveat carries)
- optimizers `{sgd, adam}`.
- arms: **buf-own** (standalone; main net naive on current task + per-task replay META-loss on the gate)
  and **er-own** (main net + gate joint on the ER batch, own-task gating); + baselines **naive**, **er**.
- granularity `{neuron, synapse}`; projection `{lin, mlp}` (for mean_image/embedding).
- Report acc per resolution mode vs the naive/er baselines and the onehot/oracle reference; keep the
  overlap-on-`dev` and buf-own-variance caveats from pt5.

## Deliverable
`results/pt6_driver_mechanisms.py` (self-contained, repo-relative, runnable) + `.log` + `.md` findings,
plus a CLAUDE.md "Known gotchas" bullet and a `prototype/iteration-notes.md` section. 1 seed; oracle
caveat; buf-own high-variance (report ≥3 seeds if a buf-own cell becomes a headline).

## Plan of record (execute in order)
1. Extend the base module: gate layers → (h0,h1,out); add `soft-nearest(τ)` eval; add `soft_mlp` (P +
   replay-trained `g`) and `embedding` (proj over `g`'s hidden) mechanisms; add synapse (low-rank content).
2. Smoke-test each code path (1 epoch).
3. Run the grid (neuron first — fast; synapse after — heavy), 1 seed, log to `results/pt6_*.log`.
4. Write findings `.md` + CLAUDE.md gotcha + iteration-notes; commit + push.
