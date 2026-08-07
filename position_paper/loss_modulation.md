# Loss modulation — `L = Σ_T c_T · L_T`

THESIS-PLAN direction B, position-paper mechanism 1. Split MNIST class-IL, ER, Adam at the
val-tuned operating point (lr 3e-4 / ep 5 / buffer 1000), 3 seeds.

**Verdict — it depends entirely on how the loss is split, and that is the study's main result.**
Two formulations were run and both are kept:

- **v1, split by SAMPLE** (`L = Σ_T c_T · L_T`, `L_T` = mean 10-way CE over task-T samples): the
  requested mechanism is a **null** (−0.0024, 0/3), and a 5-parameter content-free vector wins
  (+0.0107). `soft` is capped at ER *by algebra* — see below.
- **v2, split by CLASS** (each per-class term in the CE scaled by its task's `c_T`, i.e. the logit
  adjustment `z_c ← z_c + log c_task(c)`): the same mechanism reaches **+0.0232 (3/3)**, and the
  **content-free control FAILS** (`learned` −0.0009). At buffer 200 it grows to **+0.0575**.

So loss modulation is real, and v1 was measuring a formulation that could not express it. v2 is also
the **first cell in Phase B where a content-free control loses** — the useful adjustment is per
batch, tracking which tasks the replay draw contains, and a constant vector cannot track it.

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

## v2 — the per-CLASS formulation (user-corrected), and it changes the answer

v1 split the loss by **sample**: `L = Σ_T c_T · L_T` with `L_T` the mean 10-way CE over task-T
samples. The intended reading splits by **class**: the CE's normaliser has one term per class, and
each is scaled by that class's task coefficient —

```
L = -log( c_y·exp(z_y) / Σ_c c_task(c)·exp(z_c) )   implemented as   z_c ← z_c + log c_task(c)
```

Kept in a separate ledger (`loss_modulation_logit_results.tsv`); v1 stands.

**The parity control MOVES, and that is the whole difference.** In v1 `truefrac` (`c = n_T/N`) was
plain ER by algebra, which capped `soft` — an estimator of that same vector — at ER. In v2 a
*constant* `c` cancels in the softmax, so **`uniform` is the parity line and `truefrac` becomes a
real mechanism**: logit adjustment by observed task frequency, i.e. balanced softmax.

**normal** (buffer 1000), 3 seeds, d vs `uniform` = 0.9021:

| coef | acc | d | seeds | c_err |
|---|---|---|---|---|
| `truefrac` | **0.9269** | **+0.0247** | 3/3 | 0.000 |
| `ema` | 0.9258 | +0.0237 | 3/3 | 0.180 |
| `soft` | 0.9253 | +0.0232 | 3/3 | 0.023 |
| `dev` | 0.9038 | +0.0016 ~ | 2/3 | 1.011 |
| `dev_norm` | 0.9008 | −0.0014 ~ | 1/3 | 1.011 |
| **`learned`** | 0.9013 | **−0.0009 ~** | 2/3 | 1.011 |

**budget** (buffer 200), d vs `uniform` = 0.7689: `ema` **+0.0629**, `soft` **+0.0575**, `truefrac`
+0.0545 (all 3/3); `learned` +0.0052 ~, `dev`/`dev_norm` ~null. The effect roughly doubles under
memory pressure, and `ema` overtakes `truefrac` — with a small buffer the per-batch composition is
noisier, so a smoothed estimate beats the instantaneous truth.

### Two things this establishes

**`soft` goes from −0.0024 to +0.0232** with the same selector, posterior and operating point. v1's
null was the formulation, not the mechanism.

**The content-free control FAILS here — the first time in Phase B.** `learned` is −0.0009 at normal
and +0.0052 at budget, both null, while the signal-driven forms reach +0.023/+0.063. The reason is
structural: the useful adjustment is *per batch*, tracking which tasks the replay draw actually
contains, and a free constant vector cannot track a varying quantity. That is also why `dev`/
`dev_norm` are null — `soft − ema + 1` is nearly uniform, so it sits at parity.

Ordering is principled and `c_err` predicts it: exact composition > smoothed > per-batch estimate.
**The inference net recovers ~94% of the oracle version** (+0.0232 vs +0.0247), and nothing here uses
a task id at eval — the coefficients only shape the training loss.

### A caveat on the parity control, and what `rfree` exposed

`uniform` is `1/(tasks **present** in the batch)`, so it equals plain CE only when all T are present.
The degeneracy check makes this visible: at buffer 0 a batch holds one task, so `uniform` and
`truefrac` both become a one-hot `c`, the adjustment sends every other class to `log(1e-12)`, and the
loss becomes **exactly per-task masked CE** — study 4's `curr`. Both read 0.5359, against naive 0.198
and study 4's `curr` 0.5990/0.5430. So the two studies meet: a one-hot logit adjustment *is* the
masked loss. A stricter parity control would use `c ≡ 1/T` over all tasks regardless of presence;
the anchor passed at normal (0.9005 vs ER 0.9019) because a filled buffer almost always contains all
five.

Per the regime policy, that `rfree` column is a degeneracy check, not a rehearsal-free result.

## v3 — split by SAMPLE-WEIGHT (`L = (1/N) Σᵢ c_task(i)·CEᵢ`)

The third reading: scale each sample's *whole* CE by its own task's coefficient. Expanding,
`= Σ_T c_T·(n_T/N)·L_T` — the v1 family with `c` composed with the true fraction, **not** a logit
adjustment. The gradient is `c_y·(p − onehot(y))`: the ordinary CE gradient times a scalar, so only
that sample's effective weight moves.

Parity here is **`ones`** (`c ≡ 1`), not `uniform` — the other forms sum to 1, so `c = 1/T` makes the
loss 5× smaller, an LR change in disguise. Verified `ones` ≡ plain CE to float precision, and the
anchor confirms it end to end (0.9038 vs ER 0.9019). `w_mean` (mean per-sample weight) is ledgered so
scale is measured rather than inferred.

3 seeds, d vs `ones`, selector lr 1e-3:

| coef | normal | d | budget | d | w_mean |
|---|---|---|---|---|---|
| `ones` (parity) | 0.8994 | — | **0.7783** | — | 1.000 |
| `learned` | **0.9027** | +0.0034 ~ | 0.7699 | −0.0085 | 0.191 |
| `dev` | 0.9007 | +0.0013 ~ | 0.7737 | −0.0047 ~ | 1.035 |
| `dev_norm` | 0.8997 | +0.0003 ~ | 0.7764 | −0.0020 ~ | 0.207 |
| `uniform` | 0.8986 | −0.0008 ~ | 0.7747 | −0.0036 ~ | 0.464 |
| `ema` | 0.8937 | −0.0057 ~ | 0.7464 | −0.0320 | 0.518 |
| `soft` | 0.8852 | −0.0142 | 0.7494 | −0.0290 | 0.553 |
| `truefrac` | 0.8824 | **−0.0169** | 0.7402 | **−0.0381** | 0.554 |

**REJECT — and `truefrac` is the WORST cell, an exact reversal of v2 where it was the best.** The
same vector `c = n_T/N` helps in v2 (+0.0247) and hurts here (−0.0169), because the two formulations
give it opposite meanings:

- **v3**: a small `c_T` DOWN-WEIGHTS that task's samples. Old tasks are rare in an ER batch, so
  weighting by observed frequency starves exactly the tasks that are being forgotten.
- **v2**: a small `c_T` shifts that task's class logits down *during training*, so the network must
  learn larger weights to fit them — and at test, where no adjustment is applied, that surfaces as a
  boost. It is a prior correction, not a weighting.

So "weight each task by how likely the inference net thinks it is" is only useful when it lands in
logit space. In sample space the same estimate is actively harmful, and the best thing to do is
nothing (`ones`).

**Not a scale artifact.** `w_mean` does not order the results: `dev_norm` (0.207) and `uniform`
(0.464) are both null while `soft` (0.553) and `truefrac` (0.554) are clearly negative. A 5× loss
rescale (`dev_norm`) is ~neutral here, so what hurts is the SHAPE of `c`, not its magnitude.

**A val-selection artifact worth recording.** Tuning picked selector lr 1e-2, where task inference
collapses to 0.28, over 1e-3 where it reaches 0.87 — which read as "v3 prefers a worse selector". On
3-seed TEST that does not hold: `soft` is 0.8867 at 1e-2 and 0.8852 at 1e-3, inside noise, and the
forms that ignore the selector (`truefrac`, `uniform`, `ones`) are bit-identical across the two. The
one real difference is `learned`, which collapses to 0.7954 at 1e-2 because its free vector shares
that optimizer and over-trains (w_mean 0.149). **A 1-seed val preference is not evidence about a
mechanism; running both points is what showed it was noise.**

## Limits

3 seeds, one operating point, one dataset. The gain is ~1 pt over tuned ER and the best cell's sd is
0.005, so this is a small effect measured just above its noise floor. The inference-net lr grid
spanned 0.0061 < the noise floor (UNRESOLVED at 1 seed), though `infer` did separate cleanly (0.884
at 1e-3 vs 0.390 at 1e-2, where the selector collapses). Regimes (`budget`, `rfree`) not yet run —
note `rfree` is structurally degenerate here: with no replay a batch holds one task, one `L_T` is
nonzero, and the mechanism collapses to a scalar loss rescale, i.e. a learning-rate knob.
