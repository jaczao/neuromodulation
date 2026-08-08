# Loss modulation — `L = Σ_T c_T · L_T`

THESIS-PLAN direction B, position-paper mechanism 1. Split MNIST class-IL, ER, Adam at the
val-tuned operating point (lr 3e-4 / ep 5 / buffer 1000), 3 seeds.

**Verdict — it depends entirely on how the loss is split, and that is the study's main result.**
Two formulations were run and both are kept:

- **v1, split by SAMPLE** (`L = Σ_T c_T · L_T`, `L_T` = mean 10-way CE over task-T samples): the
  requested mechanism is a **null** (−0.0009, 1/3), and content-free coefficients win — a 5-parameter
  free vector +0.0122, a *random* posterior +0.0136. `soft` is capped at ER *by algebra* — see below.
- **v2, split by CLASS** (each per-class term in the CE scaled by its task's `c_T`, i.e. the logit
  adjustment `z_c ← z_c + log c_task(c)`): the same mechanism reaches **+0.0236 (3/3)**, and the
  **content-free control FAILS** (`learned` −0.0005, random posterior −0.0141). At buffer 200 it
  grows to **+0.0575**.
- **v3, split by SAMPLE-WEIGHT** (`L = (1/N) Σᵢ c_task(i)·CEᵢ`): **reject** — the same coefficients
  that help in v2 hurt here, and the best action is nothing.

So loss modulation is real, and v1 was measuring a formulation that could not express it. v2 is also
the **first cell in Phase B where a content-free control loses** — the useful adjustment is per
batch, tracking which tasks the replay draw contains, and a constant vector cannot track it.

The **random-posterior control** (last section) applies to every mechanism form and is what makes
that split airtight: it is *better* than the real posterior in v1 and v3 and clearly worse in v2.

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
| `truefrac` (= plain ER) | 0.8994 | — | — | 0.0886 | 0.000 | 1 |
| `soft` (posterior) | 0.8986 | −0.0009 ~ | 1/3 | 0.0946 | 0.023 | 1 |
| `ema` | 0.9071 | +0.0076 | 3/3 | 0.0914 | 0.180 | 1 |
| `dev` (soft − ema + 1) | 0.9093 | **+0.0099** | 3/3 | 0.0875 | 1.011 | 5 |
| `dev_norm` | 0.9115 | **+0.0121** | 3/3 | 0.0803 | 1.011 | 1 |
| `uniform` (1/T, content-free) | 0.9088 | **+0.0094** | 3/3 | 0.0859 | 0.423 | 1 |
| **`learned`** (content-free) | **0.9117** | **+0.0122** | 3/3 | 0.0798 | 1.062 | 1 |
| **`randproj`** (random posterior) | **0.9131** | **+0.0136** | 3/3 | 0.0836 | 1.455 | 1 |

(The `truefrac` row was re-run on CPU — it had been produced on MPS while the rest of the table came
from the sharded CPU runner, which inflated it by 0.0045 and deflated every `d` by ~0.0015. See
"A mixed-device ledger" below. `randproj` is the section after next.)

Selector task accuracy 0.8845 throughout — matching pt6's 0.86–0.88, so the promoted
`TaskInferenceNet` is working as advertised. It simply isn't the thing producing the gain.

### The three readings

**1. Benefit tracks DEPARTURE from the true composition, not signal quality.** `c_err` orders the
results almost perfectly: 0.023 → null, 0.180 → +0.008, 0.42 → +0.009, ~1.0 → +0.012, 1.46 → +0.014.
What helps is
weighting old tasks above their batch frequency, and in an ER batch the true composition is heavily
skewed toward the current task (64 current vs 64 replay spread over up to 5 tasks). This is task
loss balancing / class-balanced replay, arrived at from the position paper's direction.

**2. The scale confound was real but pointed the other way.** `Σ dev = T = 5`, so `dev` runs a ~5×
larger loss — an LR change in mechanism's clothing. `dev_norm` (renormalised) scores **higher**
(+0.0121 vs +0.0099), so the 5× scale was mildly *harmful* and the mechanism lives in the shape of
`c`, not its magnitude. Worth recording because the reflex — from pt7's ach_ema and wd_modulation's
per-step cells — is to assume a scale confound is inflating a result. Here it deflated one. **Run
the normalised twin either way; the sign is not predictable.**

**3. `learned` beats every inference-driven form, and it is 5 parameters that never see an input.**
`c = softmax(free_param)`, trained by the main loss, no dependence on `x`, the task, or the loss
landscape. It equals `dev_norm` (+0.0122 vs +0.0121). `uniform` — content-free *and* untrained —
already captures +0.0094 of it, and a *random* posterior (`randproj`) tops the table at +0.0136. So
the ordering is:

```
inference posterior (0)  <  fixed 1/T (+0.009)  ≈  learned 5-vector (+0.012)  ≲  random posterior (+0.014)
```

The entire effect is available without any signal at all — and the better the signal, the smaller
the effect.

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
| `randproj` | 0.7959 | +0.0243 | 3/3 | 1.474 |
| `randproj_ema` | 0.7981 | +0.0265 | 3/3 | 1.461 |

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

**normal** (buffer 1000), 3 seeds, d vs `uniform` = 0.9018:

| coef | acc | d | seeds | c_err |
|---|---|---|---|---|
| `truefrac` | **0.9269** | **+0.0251** | 3/3 | 0.000 |
| `ema` | 0.9258 | +0.0240 | 3/3 | 0.180 |
| `soft` | 0.9253 | +0.0236 | 3/3 | 0.023 |
| `dev` | 0.9038 | +0.0020 ~ | 2/3 | 1.011 |
| `dev_norm` | 0.9008 | −0.0010 ~ | 1/3 | 1.011 |
| **`learned`** | 0.9013 | **−0.0005 ~** | 2/3 | 1.011 |
| **`randproj`** | 0.8877 | **−0.0141** | 0/3 | 1.455 |

**budget** (buffer 200), d vs `uniform` = 0.7689: `ema` **+0.0629**, `soft` **+0.0575**, `truefrac`
+0.0545 (all 3/3); `learned` +0.0052 ~, `dev`/`dev_norm` ~null. The effect roughly doubles under
memory pressure, and `ema` overtakes `truefrac` — with a small buffer the per-batch composition is
noisier, so a smoothed estimate beats the instantaneous truth.

### Two things this establishes

**`soft` goes from −0.0009 to +0.0236** with the same selector, posterior and operating point. v1's
null was the formulation, not the mechanism.

**The content-free control FAILS here — the first time in Phase B.** `learned` is −0.0005 at normal
and +0.0052 at budget, both null, while the signal-driven forms reach +0.024/+0.063. The reason is
structural: the useful adjustment is *per batch*, tracking which tasks the replay draw actually
contains, and a free constant vector cannot track a varying quantity. That is also why `dev`/
`dev_norm` are null — `soft − ema + 1` is nearly uniform, so it sits at parity. (The `randproj`
control below sharpens this: a coefficient vector that *does* vary per batch but carries no task
content fails harder still — though it varies at a smaller amplitude, so it bounds the "constant"
reading rather than replacing it.)

Ordering is principled and `c_err` predicts it: exact composition > smoothed > per-batch estimate.
**The inference net recovers ~94% of the oracle version** (+0.0236 vs +0.0251), and nothing here uses
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

## The random-posterior control, applied to every mechanism form

`randproj_<form>` is the form computed exactly as before, except the posterior comes from a **frozen
random 784→5 projection** (`softmax(x @ R)`, `R ~ N(0, 0.1²)`, drawn once, never trained) instead of
the replay-trained inference net. The EMA, the deviation and the renormalisation are all computed
from that random posterior, so each form keeps its own arithmetic and only the *content* of `p(T|x)`
is destroyed. `randproj` (no suffix) is `randproj_soft`.

It was originally run on `soft` alone. Extending it to every form is what makes it decisive, because
the forms differ in what they do with the posterior and there was no reason the answer would
transfer — and it does not.

**`d = randproj − real`, 3 seeds, `neg` = seeds where the real posterior won:**

| formulation | regime | `soft` | `ema` | `dev` | `dev_norm` |
|---|---|---|---|---|---|
| v1 `group` | normal | **+0.0145** (0/3) | +0.0053 ~ (0/3) | +0.0005 ~ | −0.0024 ~ |
| v1 `group` | budget | **+0.0174** (0/3) | **+0.0346** (0/3) | −0.0008 ~ | −0.0002 ~ |
| v2 `logit` | normal | **−0.0377** (3/3) | **−0.0377** (3/3) | −0.0023 ~ | −0.0018 ~ |
| v2 `logit` | budget | **−0.0735** (3/3) | **−0.0766** (3/3) | +0.0046 ~ | +0.0007 ~ |
| v3 `sample` | normal | **+0.0172** (0/3) | **+0.0162** (0/3) | −0.0004 ~ | −0.0040 ~ |
| v3 `sample` | budget | **+0.0407** (0/3) | **+0.0323** (0/3) | +0.0025 ~ | −0.0005 ~ |

### 1. The control separates the three formulations exactly along the line their mechanisms predict

**v2 is the only formulation that needs a real posterior**, and it needs it badly: −0.038 at normal
and −0.074 at budget, negative in every seed. That is the strongest form of v2's content result. The
previous evidence was `learned` (content-free *and* constant) failing, which admitted the reading
"the coefficients just have to vary per batch". `randproj` **does** vary per batch — the projection
is frozen, but it is applied to the batch, so `c` is a fixed function of the images and differs from
one batch to the next — and against parity it scores −0.0141, *worse* than the constant `learned`'s
−0.0005.

**But read that with the amplitudes, which the ledger records.** `c_sd` (std of each coefficient
across steps, averaged over T) at v2/normal/seed 42:

| | `truefrac` | `soft` | `ema` | `uniform` | `randproj` | `dev_norm` | `learned` |
|---|---|---|---|---|---|---|---|
| `c_sd` | 0.245 | 0.245 | 0.228 | 0.157 | 0.029 | 0.016 | 0.006 |

`randproj` varies ~5× more than `learned` but ~8× *less* than the real posterior. That is structural
rather than incidental: the real posterior's batch mean tracks a composition that genuinely swings
(64 current + 64 replay spread over up to 5 tasks), while a task-blind posterior's batch mean is the
sampling noise of a mean over 128 images, which concentrates. So the control is "varies a little,
tracks nothing", and it **does not cleanly separate varying from tracking** — the two are confounded
with amplitude here.

What survives is the ordering variable, unchanged: `c_err` (tracking) predicts every result, and a
per-batch-varying vector is *not rescued* by its variation. Note also that `uniform` is not constant
either (`c_sd` 0.157, since `1/(tasks present)` changes as the sequence accumulates tasks), so
"content-free" and "constant" were never the same axis. Separating amplitude from tracking would need
a control that varies at the real posterior's amplitude while carrying no task information — e.g.
`R` rescaled until `c_sd` matches, or a shuffled posterior. Not run.

**v1 and v3 are better off without it.** In v1 the random posterior is the top cell in the whole
study (+0.0136 vs parity, above `learned`'s +0.0122); in v3 it is the only coefficient form that does
not lose to `ones` (+0.0046 ~, while the real posterior is −0.0127). Both follow from what those
formulations do with `c`: v1's benefit comes from *departing* from the true composition (a good
posterior estimates that composition, hence estimates plain ER), and v3's `c` down-weights whole
samples, so an accurate posterior starves exactly the rare old tasks. In both, accuracy of the
signal is the thing that hurts.

### 2. `c_err` remains the single ordering variable, and `randproj` extends its range

Sorted by distance from the true composition — `truefrac` 0.000, `soft` 0.023, `ema` 0.180,
`uniform` 0.423, `dev`/`dev_norm` 1.011, `learned` 1.062, **`randproj` 1.455** — the three
formulations order *the same vectors* in three different ways:

- **v1**: benefit rises monotonically along `c_err`, all the way to the new maximum.
- **v2**: benefit falls monotonically along it (+0.025 → +0.024 → +0.024 → 0 → ~0 → −0.014).
- **v3**: benefit rises toward parity and stops there; nothing beats `ones`.

`randproj` was the missing point at the far end, and in all three cases it lands where the trend
says it should. This is the cleanest statement of the study's main result: **the same coefficient
vector is good, useless or harmful depending only on where in the loss it is applied.**

### 3. NEW — `dev` and `dev_norm` are content-blind *by construction*

Randomising their posterior changes nothing in any of the six cells (|d| ≤ 0.0046, all inside the
noise floor), which is not true of any other form. The reason is visible in `c_err`: `dev = soft −
ema + 1` has `c_err` ≈ 1.011 with the real posterior and 1.032 with a random one — the constant `1`
dominates `soft − ema`, so the vector is nearly uniform either way.

Their nulls were previously read as "this signal does not help". The control upgrades that to **"this
form discards the signal"** — a different and more useful statement, since it says the deviation
parameterisation is the problem rather than the deviation *idea*. Any future form of the shape
`signal − baseline + const` should be checked for the same swamping before it is run.

### 4. Two checks that make the pairing readable

**`R` is drawn from a private generator, never the global stream.** Constructing it from the global
stream (as the first version did) consumes RNG before training and shifts every replay draw — worth
~0.002 at width 400, the same order as these effects — so `randproj_X` would have differed from `X`
by that shift as well as by the mechanism. `--part rngcheck` verifies the fix rather than asserting
it: `randproj_truefrac` (whose coefficients ignore the posterior entirely, so only the projection's
existence differs) is **bit-identical to `truefrac`**, matching on accuracy, forgetting and selector
accuracy alike.

**`randproj` is invariant to the inference-net lr, and that was measured.** v3's ledger is keyed at
the tuned selector lr 1e-2 while its headline table is at 1e-3. Since a `randproj` cell never reads
the inference net, trains it only in a separate optimizer and draws no RNG doing so, the two should
be identical — and they are, to six decimals, for both `randproj` and `randproj_dev`. So the v3
`randproj` numbers are directly comparable to the 1e-3 table, and the check doubles as confirmation
that the control really does bypass the selector.

### Cost note

The frozen projection is 3,920 parameters and replaces a 101k-parameter inference net. The ledger's
`extra_params` column still counts the inference net, because these runs keep training it to report
`infer` — a deployed `randproj` would not have one at all.

## A mixed-device ledger, found by a control that should have been trivial

`--part rngcheck` was written to test one thing (that the private generator is RNG-neutral) and
immediately reported SHIFTED for a pair that is identical by construction. The generator was fine;
the *stored* row was not. The `anchor` and `tune` parts of this study had been run directly — on MPS
— while `test` and `regimes` went through the sharded CPU runner, so each ledger's parity control
had one MPS row at seed 42 sitting in an otherwise-CPU table.

Confirmed by re-running the stored cells on each device: MPS reproduces `group`/`truefrac` **exactly**
(0.903172) and `sample`/`ones` **exactly** (0.903759), while the sharded `group`/`soft` row does not
(0.901268 vs 0.900617 stored). So v1 and v2 were device-mixed and v3 is uniformly MPS.

Fixed by re-running the parity rows on CPU: `truefrac` 0.903172 → 0.898666, `uniform` 0.900508 →
0.899401. That deflated the parity line and raised every v1 `d` by ~0.0015 and every v2 `d` by
~0.0004; the tables above are the corrected numbers, and no conclusion changes. The new v1 and v2
`randproj` rows were run on CPU and the v3 ones on MPS, each matching its own ledger.

The `val` rows are still MPS and were left alone deliberately: re-running them could move the
selected inference lr and orphan every test row keyed to it, and this study is not being re-tuned.

Two things worth carrying forward: the per-cell device gap is **not constant** (0.0045 for one cell,
0.0011 for another, 0.0007 for a third), so it cannot be corrected for on paper; and a control whose
answer is known by construction is the cheapest possible detector for this class of defect — it found
a problem it was not written to look for.

## Limits

3 seeds, one operating point, one dataset. The gain is ~1 pt over tuned ER and the best cell's sd is
0.005, so this is a small effect measured just above its noise floor. The inference-net lr grid
spanned 0.0061 < the noise floor (UNRESOLVED at 1 seed), though `infer` did separate cleanly (0.884
at 1e-3 vs 0.390 at 1e-2, where the selector collapses). `rfree` is structurally degenerate here:
with no replay a batch holds one task, one `L_T` is nonzero, and the mechanism collapses to a scalar
loss rescale, i.e. a learning-rate knob.

For the `randproj` control specifically: one projection scale (σ = 0.1) and one draw per seed, so
"a random posterior" is characterised by a single point in a two-parameter family. Its `c_err` is
1.455 in every cell, which is *higher* than any other form's — the control moves along the `c_err`
axis and the "random" part is what puts it there, so the two are not separable from these runs
alone. `rfree` was not run for it (degenerate for every form). v1/v2 rows are CPU and v3 rows MPS,
each matching its own ledger; the two are not comparable across formulations at the ~0.005 level.
