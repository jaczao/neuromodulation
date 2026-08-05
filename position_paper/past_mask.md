# Past-only output masking

THESIS-PLAN direction B, own variant. Split MNIST class-IL, Adam, **each variant at its own
val-tuned lr**, 3 seeds.

**Verdict — `past` REJECT, but for a reason worth having.** It does protect old classes exactly as
intended (forgetting 0.1545, matching `curr`'s 0.1531), and it still lands at 0.2842 against `curr`'s
0.5990 — because protecting the past costs it the future. Each task suppresses every not-yet-seen
class, so by the time a task arrives its logits have already been pushed down four times. The
half-prediction that failed is the informative half.

---

## The 2×2

Which logits a sample competes against, defined per sample by its own task:

| mask | competes with | |
|---|---|---|
| `none` | all 10 | plain naive |
| `curr` | own 2 | pt3's lever B |
| `past` | all but *older* tasks | requested |
| `future` | all but *newer* tasks | the complement, as control |

**Anchors pass**: `none` 0.1986 vs the project's long-standing naive class-IL 0.198, and `curr`
0.5990 vs pt3_retry's 0.5990 — to four decimals, which is what validates the loop.

## Results (each at its own tuned lr)

**naive arm**

| mask | lr | acc | d-none | forget | allowed |
|---|---|---|---|---|---|
| `none` | 1e-6 | 0.3016 | — | 0.6506 | 10.00 |
| `curr` | 1e-5 | **0.5990** | +0.2974 | 0.1531 | 2.00 |
| `past` | 3e-4 | 0.2842 | −0.0174 | **0.1545** | 6.05 |
| `future` | 3e-4 | 0.1988 | −0.1028 | 0.7983 | 5.95 |

**er arm** — every mask hurts: `none` 0.9078, `curr` 0.6623, `past` 0.2080, `future` 0.2136.

### Prediction 1 (`past ≈ curr`) FAILED, and the failure is the mechanism

`past` reproduces `curr`'s *forgetting* almost exactly (0.1545 vs 0.1531) while reaching less than
half its accuracy. That combination — **low forgetting with low accuracy** — is the over-suppression
signature CLAUDE.md already records: under-learning, not retention. Here the cause is specific and
new. `past` is asymmetric in time: task *t* never pushes down classes of tasks < *t*, but it does
push down every class of tasks > *t*. So task 4's logits have been suppressed by tasks 0–3 before
task 4 is ever trained, and each task's classes arrive pre-buried. Old classes are protected; new
ones are sabotaged in advance.

`curr` avoids this by masking symmetrically — a sample sees only its own two logits, so it suppresses
neither past nor future.

### Prediction 2 (`future ≈ none`) CONFIRMED, at matched lr

`future` keeps the harmful suppression of old classes and removes only the harmless future
competition, so it should behave like `none`. At the matched lr 3e-4 it does, to four decimals:
**0.1988 vs `none`'s 0.1986**. The table's −0.1028 gap is not a mechanism difference — it is that
`none` tuned down to 1e-6 and gained 0.10 from the smaller steps, while `future`'s grid found no such
benefit. A reminder that with per-variant tuning the d-column mixes mechanism with operating point;
the matched-lr comparison is the one that answers the question.

### `past` is capped by construction

At task 0 nothing is past, so `past` *is* `none`; at the last task everything but the current pair is
past, so it *is* `curr`. It interpolates between the two arms pt3 already measured (`allowed` shows
it directly: 10 → 2, mean 6.05) and cannot beat `curr` by more than the early-task slack. It does not
come close, because the interpolation costs more than the slack is worth.

---

## Limits

3 seeds; class-IL, Adam, one dataset. `none`'s tuned lr sits at the grid floor (1e-6) — a genuine
truncation warning, and extending downward would likely lift `none` further, which would make `past`
look worse, not better. The er-arm numbers are all far below plain ER and were not tuned beyond the
shared grid. `past`'s asymmetry suggests an untested symmetric variant — mask past *and* future,
which is exactly `curr` — so the space between them is empty by construction.
