# pt6 — content & inference-net mechanisms + eval-resolution axis (neuron grid)

Companion to `pt6_driver_mechanisms.py` (`SPEC-proto-pt6.md`). Gain target on **(h0,h1,out)**,
gain-**neuron**, class-IL Split MNIST, seed 42, lr 1e-3, ep 5, buffer 1000. **1 seed; ORACLE caveat
applies only to `oracle`/`onehot` — the point of pt6 is the oracle-FREE columns.** Baselines reproduce
the pt5 harness (`naive-sgd 0.629 / naive-adam 0.390 / er-sgd 0.723 / er-adam 0.895`).

## Headline
pt5 concluded "no oracle-free eval beats/matches ER; the oracle carried the ~0.99, and nearest-prototype
inference caps at ~0.76." **pt6 overturns the "well below ER" half:** a **learned** task selection
(trained with replay) reaches **~0.88 oracle-free — ER parity** — because learned inference (~0.88
task-acc) far exceeds the fixed nearest-prototype (0.76). It **matches** ER (doesn't clearly beat it
under Adam), so the gate is not a new lever *over* replay, but it removes the eval-oracle dependency
that dominated pt5.

| mechanism (best cell) | oracle-free acc | vs ER (same opt) | note |
|---|---|---|---|
| mean_image + nearest (pt5-style) | 0.755 | ≈/below | prototype-capped (0.76 infer) |
| mean_image + **soft-nearest(τ=0.03)** | 0.819 | er-sgd +0.10 | soft > hard, still prototype-capped |
| **soft_mlp** (learned soft inference) | **0.885** (adam), 0.856 (sgd) | adam ≈, **sgd +0.13** | infer 0.88 ≫ 0.76 |
| **embedding** (learned per-image, oracle-free) | **0.889** (sgd), 0.879 (adam) | **sgd +0.17**, adam ≈ | cleanest oracle-free |

## Selected numbers (er-own arm; oracle is the reference, rest are oracle-free)
```
onehot        er-own sgd  oracle 0.949            adam oracle 0.989
mean_image/lin/cen  er-own sgd  oracle 0.980  per-img 0.748  nearest 0.749  soft@.03 0.767
mean_image/lin/cen  er-own adam oracle 0.991  per-img 0.520  nearest 0.755  soft@.03 0.743
mean_image/lin      er-own sgd  oracle 0.973  per-img 0.836  nearest 0.775  soft@.03 0.819
soft_mlp            er-own sgd  oracle 0.948  soft-mlp 0.856 (infer 0.884)
soft_mlp            er-own adam oracle 0.991  soft-mlp 0.885 (infer 0.884)
embedding/lin       er-own sgd  per-img 0.889     embedding/mlp er-own sgd per-img 0.881
embedding/lin       er-own adam per-img 0.879     embedding/mlp er-own adam per-img 0.869
soft_mlp            buf-own sgd oracle 0.939  soft-mlp 0.856 (infer 0.865)   <- STANDALONE backbone
```

## Readings
1. **Learned inference beats prototype inference.** soft_mlp's `g(x)` reaches 0.86–0.88 task-accuracy
   vs nearest-prototype's fixed 0.759. Since the gate has ~zero misrouting tolerance
   (`nearest ≈ oracle×infer`), lifting infer 0.76→0.88 is exactly what lifts the oracle-free number
   ~0.75→~0.88. **soft-nearest(τ)** confirms softness helps (sharp τ=0.03 beats hard nearest,
   0.819 vs 0.775) but is bounded by the prototype's 0.76 — the *learned* selector is the real lever.
2. **Same-optimizer verdict.** Under **SGD** the learned mechanisms clearly beat ER (embedding 0.889 /
   soft_mlp 0.856 vs er-sgd 0.723, +0.13..+0.17). Under **Adam** they **match** ER (0.88 vs 0.895, ≈).
   So they let SGD reach Adam-ER-level retention **without the oracle** — parity, not a new lever.
3. **`embedding` is the cleanest result** — per-image continuous, oracle-free *by construction* (no
   discrete task inference at all), 0.889 (er-own/sgd). `lin ≈ mlp` for embedding (~0.88).
4. **Standalone (buf-own) also works for the learned mechs:** soft_mlp buf-own/sgd 0.856 oracle-free
   with a naive backbone (replay only on the gate + inference net) — well above naive 0.629 and
   er-sgd 0.723. buf-own is high-variance (pt5: report ≥3 seeds before headlining).
5. **mean_image reproduces pt5** — centered lin er-own matches onehot under oracle (0.980/0.991), the
   buf-own/adam and uncentered-lin cells collapse; `mlp ≤ lin` (relu re-correlation).

## Verdict
The task-inference dependency pt5 flagged is real, but **not** a hard wall: a **replay-trained soft
inference** (soft_mlp) or a **learned per-image embedding** reaches **ER parity oracle-free** (~0.88),
vs pt5's prototype-capped ~0.75. It matches rather than beats replay, so the reportable class-IL lever
is still ER — but pt6 shows the gate mechanism can be made **genuinely oracle-free** at ER-level
accuracy, which the pt5 driver-representation study could not achieve. Synapse granularity deferred (see
below). 1 seed; buf-own high-variance.

## Why synapse granularity was deferred (corrected rationale)

The original one-liner ("per-sample gates conflict with per-image/soft resolution + 374M-param content
projection") welded two independent blockers together and over-applied both. Per mechanism:

| mechanism | gate params at synapse (n_syn = 477 600) | param blocker? |
|---|---|---|
| `onehot` | `P: (T, n_syn)` = 2.4M lookup | **no** — pt5 already ran per-synapse onehot |
| `soft_mlp` | same 2.4M lookup + `g` (784→128→5, ~100k) | **no** |
| `embedding`/lin | `W: (128, n_syn)` = 61M | yes, 128× the net |
| `mean_image`/lin | `W: (784, n_syn)` = 374M | yes, 780× the net |

So the projection blow-up is a `mean_image`/`embedding` problem only; the lookup mechanisms are cheap.

The *other* blocker is the **per-sample Γ expansion**: with a per-task gate all samples in a batch share
`Γ`, so you form `(Γ⊙W)` once and do a normal matmul, but a per-sample `Γ` needs `(B, d_out, d_in)` —
~160 MB for layer 0 alone at B=128. That blocks **training** for `embedding` only, whose `train_gate` *is*
`gate_per_sample`. `soft_mlp` trains on **true task ids** (`P[tids]`), so a batch — even the mixed
current+replay `er-own` batch — holds at most `T=5` distinct gates and groups into ≤5 masked matmuls,
exactly the pt5 `er_task_id` path; only its soft-blend **eval** needs a per-sample `Γ`, and that chunks
down freely under `no_grad` (bit-identical, since eval is per-sample independent — you pay wall-clock,
not correctness).

**Net: per-synapse `soft_mlp` was runnable and simply not run** (the oracle-only synapse cells were
skipped as redundant with pt5). The deferral is genuinely justified only for `mean_image` and
`embedding`. The SPEC's rank-64 low-rank content projection would cut the *parameter* count but not the
per-sample expansion (it becomes 64 masked matmuls per layer per batch).
