# ER replay buffer as uint8 — normal and memory-budgeted regimes

No neuromodulation. Class-IL Split MNIST, Adam at the val-tuned ER point (lr 3e-4, ep 5) from
`neurocore.tuned`, macro metric, 3 seeds (42/43/44), MPS, 20 cells (14 + a 6-cell `fp32big`
follow-up), 9.6 min sharded. Ledger `buffer_dtype_results.tsv`.

## The codec is bit-exact lossless

MNIST is uint8 upstream and the transform is affine (`ToTensor` -> `/255`, `Normalize` ->
`(x-mu)/sigma`), so

    round((x*sigma + mu)*255) -> uint8 -> ((k/255) - mu)/sigma

reproduces the stored float32 **bit-for-bit** (`torch.equal` True on the training set and on a real
task batch). That is why "does quantisation cost accuracy?" is not a 6-run experiment here — it is
an assertion plus a 2-run integration check. The only open question is what the saved bytes buy.

## Correctness gates (both passed before any result was read)

| gate | expected | got |
|---|---|---|
| **anchor** `fp32`/normal/seed42 vs frozen `pt7_tuned_syn` ER-adam | 0.897549 | **0.897549**, \|d\|=0 |
| **parity** `u8count` vs `fp32`, cap 1000 / cap 200 (2 cells) | bit-identical | \|d\|=0 both |
| **parity** `u8bytes` vs `fp32big`, cap 3969 / cap 793 (6 cells) | bit-identical | \|d\|=0 all six |

Eight matched-cap parity cells across two caps and three seeds, every one at `|d| = 0.00e+00`.

The anchor also proves the added forgetting matrix (25 eval passes per run) stayed RNG-neutral —
iterating a DataLoader draws from the global torch RNG even at `shuffle=False`, so without the
`rng_frozen()` guard those evals would have moved the run off its reference trajectory.

The parity arms are bit-exact **by construction**, not by luck: the reservoir's RNG consumption is
unchanged (one `random.randint` per evicted sample, one `torch.randint` per draw) and the codec draws
none. They are therefore a test of the *integration*, not of the codec — a mismatch would have meant
a wiring bug, never "uint8 costs accuracy".

Free-standing confirmation: `fp32`/normal over 3 seeds is **0.9029 ± 0.0042**, reproducing the
project's tuned ER figure (0.9029 ± 0.0042) to four decimals.

## Result

| regime | arm | cap | bytes | acc | sd | forget | d-fp32 |
|---|---|---:|---:|---:|---:|---:|---:|
| normal | fp32 | 1000 | 3,144,000 | 0.9029 | 0.0042 | 0.0919 | — |
| normal | u8count | 1000 | 792,000 | 0.8975 | — | 0.0972 | +0.0000 |
| normal | **u8bytes** | **3969** | 3,143,448 | **0.9493** | 0.0044 | 0.0438 | **+0.0464** |
| normal | fp32big | 3969 | 12,478,536 | 0.9493 | 0.0044 | 0.0438 | +0.0464 |
| budget | fp32 | 200 | 628,800 | 0.7674 | 0.0006 | 0.2288 | — |
| budget | u8count | 200 | 158,400 | 0.7679 | — | 0.2287 | +0.0000 |
| budget | **u8bytes** | **793** | 628,056 | **0.8896** | 0.0033 | 0.1058 | **+0.1223** |
| budget | fp32big | 793 | 2,493,192 | 0.8896 | 0.0033 | 0.1058 | +0.1223 |

`fp32big` prices the result. It is bit-identical to `u8bytes` and **costs 3.97x the memory to be so**
— 12.5 MB against 3.1 MB at normal, 2.5 MB against 628 KB at budget. So this is not a lossy trade
with a favourable exchange rate; **uint8 delivers exactly the buffer you would otherwise have to pay
4x for, and the accuracy column is not an approximation of `fp32big`'s, it is the same number.**

Two ways to spend the 3.97x (not 4x — int64 labels stay 8 B/sample):

- **Keep the samples, drop the bytes.** `u8count` is the same accuracy to the last bit at **1/3.97
  the memory**. Free.
- **Keep the bytes, take the samples.** +0.0464 at normal, **+0.1223 at budget**, with forgetting
  roughly halved in both (0.0919 -> 0.0438, 0.2288 -> 0.1058).

**The budget regime is where it matters.** Going 1000 -> 200 samples costs ER 0.1355 (0.9029 ->
0.7674). Storing those same bytes as uint8 recovers **0.1223 of that 0.1355** — nine tenths of the
entire memory-budget penalty, from a lossless change of container. Put the other way: uint8 at the
*budget* byte budget (0.8896, 628 KB) is within 0.0133 of fp32 at the *normal* byte budget (0.9029,
3.14 MB) — near-normal accuracy at a fifth of the memory.

## What this does to the project's budget-regime rows

This is a **cost-column result, not a mechanism result** — "4x more replay samples helps" is not
surprising. What is worth flagging is the size relative to everything the neuromodulation work has
measured at buffer 200:

- `wd_modulation` boundary decay: +0.0354 over its in-harness budget ER (0.7913)
- `state_drivers` `w_absmax` at budget: +0.0231 over `const`; `act_pr` +0.0135
- `loss_modulation` v2 at budget: +0.0629 (best cell)

A **pure storage change is worth +0.1223** in the same regime, i.e. roughly 2-5x the best mechanism
win the project has found under memory pressure, and it requires no gate, no driver, no head and no
extra parameter (`extra_params` 0, `param_ratio` 0 in every row).

The sharper reading is that rule #12's budget regime has been defined by **sample count** while its
accounting column is **bytes**. Under a byte-defined budget, ER at "budget" should have been running
793 samples, not 200 — so every budget-regime baseline in the project is ~0.12 too low, and the
mechanism deltas measured against it are deltas against a byte-inefficient reference. Cross-study
comparison carries the usual caveat (different harnesses, different RNG streams — `wd_modulation`'s
in-harness budget ER is 0.7913 vs this study's 0.7674), but at this effect size the direction is not
in doubt.

## Limits

- **Not re-tuned per cap.** Both arms run the ER-tuned point selected at buffer 1000 (rule #3,
  identical budget). A 3969-sample buffer may want a different lr/epochs, so the `u8bytes` numbers
  are "at the incumbent operating point", not at its own optimum.
- Split MNIST, class-IL, Adam, one operating point, 3 seeds.
- `rfree` excluded as structurally degenerate: ER at buffer 0 is naive, a degeneracy check rather
  than a rehearsal-free result (rule #12). A real rehearsal-free row needs a rehearsal-free base.
- Losslessness relies on the pipeline being **affine over uint8-native inputs**. It holds for any
  standard image benchmark and for per-channel normalisation (still affine), and for geometric
  augmentation applied before storage. It would NOT hold for a buffer of post-encoder features or of
  mixup/noise-augmented floats — those need a real quantisation study.
- Labels are still int64. Storing them as uint8 gives cap 4005 / 800 instead of 3969 / 793 — an
  untested ~1% more.
