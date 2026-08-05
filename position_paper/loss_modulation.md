# Loss modulation — `L = Σ_T c_T · L_T`

THESIS-PLAN direction B, position-paper mechanism 1. Split MNIST class-IL, ER, Adam at the
val-tuned operating point (lr 3e-4 / ep 5 / buffer 1000), 3 seeds.

**Verdict.** The requested mechanism — coefficients from the replay-trained task-inference posterior
— is a **null** (−0.0024, 0/3 seeds). Its across-step variants do help, but so does a **5-parameter
content-free vector that never sees an input** (+0.0107), which is the best cell in the study. The
effect is task loss RE-BALANCING, a known CL trick; the inference signal contributes nothing.

Second study in a row where a content-free control matches or beats every real driver.

---

## Why `soft` had to be null, and was

Plain batch-mean CE **already is** a task-weighted sum, with the weights being the true composition:

```
mean CE  =  Σ_T (n_T / N) · L_T
```

So `c_T = n_T/N` is not one choice among many — it is *exactly* plain ER, which makes `truefrac` an
algebraic parity control. And `soft` (`c_T = mean_i p(T|x_i)`) is an **estimator of that same
vector**: a perfect selector gives `mean_i p(T|x_i) = n_T/N` identically.

**`soft` can therefore differ from ER only by inference error** — it has no other channel. That was
pre-registered in the module docstring before the run, and the ledger confirms both halves: `soft`
lands at −0.0024 (null), and its `c_err` (distance from the true composition) is 0.023, the smallest
of any mechanism form. A better selector would make it *more* exactly ER, not better than it.

---

## Results (d vs `truefrac` = plain ER, 3 seeds)

| coef | acc | d | seeds | forget | c_err | Σc |
|---|---|---|---|---|---|---|
| `truefrac` (= plain ER) | 0.9009 | — | — | 0.0918 | 0.000 | 1 |
| `soft` (posterior) | 0.8986 | −0.0024 ~ | 0/3 | 0.0946 | 0.023 | 1 |
| `ema` | 0.9071 | +0.0061 ~ | 3/3 | 0.0914 | 0.180 | 1 |
| `dev` (soft − ema + 1) | 0.9093 | **+0.0084** | 3/3 | 0.0875 | 1.011 | 5 |
| `dev_norm` | 0.9115 | **+0.0106** | 3/3 | 0.0803 | 1.011 | 1 |
| `uniform` (1/T, content-free) | 0.9088 | **+0.0079** | 3/3 | 0.0859 | 0.423 | 1 |
| **`learned`** (content-free) | **0.9117** | **+0.0107** | 3/3 | 0.0798 | 1.062 | 1 |

Selector task accuracy 0.8845 throughout — matching pt6's 0.86–0.88, so the promoted
`TaskInferenceNet` is working as advertised. It simply isn't the thing producing the gain.

### The three readings

**1. Benefit tracks DEPARTURE from the true composition, not signal quality.** `c_err` orders the
results almost perfectly: 0.023 → null, 0.180 → +0.006, 0.42 → +0.008, ~1.0 → +0.011. What helps is
weighting old tasks above their batch frequency, and in an ER batch the true composition is heavily
skewed toward the current task (64 current vs 64 replay spread over up to 5 tasks). This is task
loss balancing / class-balanced replay, arrived at from the position paper's direction.

**2. The scale confound was real but pointed the other way.** `Σ dev = T = 5`, so `dev` runs a ~5×
larger loss — an LR change in mechanism's clothing. `dev_norm` (renormalised) scores **higher**
(+0.0106 vs +0.0084), so the 5× scale was mildly *harmful* and the mechanism lives in the shape of
`c`, not its magnitude. Worth recording because the reflex — from pt7's ach_ema and wd_modulation's
per-step cells — is to assume a scale confound is inflating a result. Here it deflated one. **Run
the normalised twin either way; the sign is not predictable.**

**3. `learned` wins, and it is 5 parameters that never see an input.** `c = softmax(free_param)`,
trained by the main loss, no dependence on `x`, the task, or the loss landscape. It equals
`dev_norm` (+0.0107 vs +0.0106) and beats every inference-driven form. `uniform` — content-free
*and* untrained — already captures +0.0079 of it. So the ordering is:

```
inference posterior (0)  <  fixed 1/T (+0.008)  ≈  learned 5-vector (+0.011)
```

The entire effect is available without any signal at all.

---

## Regimes (rule #12)

**`budget`** (buffer 200, `truefrac` = ER falls to 0.7716). Every effect roughly doubles, and the
ordering *changes*:

| coef | acc | d | seeds | c_err |
|---|---|---|---|---|
| `soft` | 0.7785 | +0.0069 ~ | 3/3 | 0.017 |
| `ema` | 0.7635 | −0.0081 | 1/3 | 0.179 |
| `dev` | 0.7891 | +0.0175 | 3/3 | 1.016 |
| `dev_norm` | 0.7879 | +0.0163 | 3/3 | 1.016 |
| **`uniform`** | **0.7985** | **+0.0269** | 3/3 | 0.422 |
| `learned` | 0.7899 | +0.0183 | 3/3 | 1.066 |

`soft` stays a null (+0.0069, right at the floor) even though the selector degrades sharply with the
smaller buffer (`infer` 0.8845 → 0.7300) — consistent with the algebra: a *worse* estimator of the
true composition does not become a better weighting.

**`uniform` — the simplest content-free option, with no learning at all — is now the outright
winner**, beating `learned` by +0.009. Note it also has the *smaller* `c_err` (0.42 vs 1.07), so the
"benefit tracks departure" reading from the normal regime does not extend indefinitely: there is an
optimum departure, and equal-per-task weighting sits nearer it than anything the free vector finds.
Under memory pressure, over-departing costs.

This is the opposite of what `wd_modulation` found at `budget`, where task-informative drivers pulled
*ahead* of the content-free control. The two mechanisms differ in what they can express: a decay gate
has per-parameter freedom that can encode task structure, while these coefficients are a T-vector
whose best setting is close to a constant. Worth keeping in mind before generalising either result.

**`rfree`** (buffer 0): every coefficient form returns 0.198 ± 0.0006 with all deltas ~0, and `infer`
is 0.1983 = chance. The predicted degeneracy, confirmed rather than assumed — one task per batch, one
nonzero `L_T`, so the mechanism is a scalar loss rescale and there is nothing task-differentiated
left to modulate. (The chance-level selector independently reproduces pt6-followup-B: with no buffer,
inference collapses.)

## Limits

3 seeds, one operating point, one dataset. The gain is ~1 pt over tuned ER and the best cell's sd is
0.005, so this is a small effect measured just above its noise floor. The inference-net lr grid
spanned 0.0061 < the noise floor (UNRESOLVED at 1 seed), though `infer` did separate cleanly (0.884
at 1e-3 vs 0.390 at 1e-2, where the selector collapses). Regimes (`budget`, `rfree`) not yet run —
note `rfree` is structurally degenerate here: with no replay a batch holds one task, one `L_T` is
nonzero, and the mechanism collapses to a scalar loss rescale, i.e. a learning-rate knob.
