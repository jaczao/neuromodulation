# Test-set ORDER sensitivity of the stateful / GRU gating cells

User-requested. Re-evaluates every order-sensitive cell of four studies under a **fully shuffled test set**
(all 10 000 test images pooled, shuffled, batched into 64 — no task blocks, each batch a random mix of all
10 classes), from the **same trained weights**, with **no new tuning**. Ledger `test_order_results.tsv`
(44 cells x 2 orders = 88 rows), driver `test_order.py`, MPS, seed 42 unless stated.

**ANCHOR: all 44 ordered passes reproduce their frozen ledger BIT-EXACT (`d = +0.000000`, 0 mismatches)**
across four source ledgers — `pt7_stateful_results.tsv` (24), `pt7_signalnet_results.tsv` (3),
`signalnet_capacity_results.tsv` (9) and `signalnet_traces.md` (8). That is what licenses reading the
shuffled column as a real effect rather than a porting artefact.

## Headline

**41 of 44 cells are nulls; 22 are EXACTLY 0.000000. Only 5 cells clear the 0.007 noise floor, all of them
`live`/`running` protocol cells, and all POSITIVE — shuffling never hurts beyond noise.**

| | cells | max \|d-shuf\| |
|---|---|---|
| exactly 0.000000 | 22 | 0.0000 |
| \|d\| <= 0.0007 | 31 | 0.0007 |
| \|d\| > 0.007 | 5 | 0.1295 |

The five: `sngru|predH|live` **+0.1295**, `sn|predH|live` +0.0724, `sn|actualH|live` +0.0596,
`sngru|actualH|live` +0.0264, `nerisez|gru1|er-own|sgd|running` +0.0197.

Only 3 cells are negative beyond 0.0007, all inside the floor: `ach|gru1|er-own|adam|running` −0.0045,
`nerisez|gru1|er-own|adam|running` −0.0024, `ach|gru1|er-own|adam|frozen` −0.0020.

## (1) THE DISCRIMINATOR IS THE VARIABILITY OF THE GATE'S INPUT, NOT THE GATE'S MAGNITUDE

Pre-registered guess (wrong): order would matter where `|g|` is order-1. Measured, it is uncorrelated:

| cell | \|g\| out | d-shuf | GRU input |
|---|---|---|---|
| signalnet-gru\|K16\|eng | **2.9220** | **+0.0000** | signal-net code (near-constant) |
| signalnet-gru\|K4\|eng | 1.9400 | +0.0000 | signal-net code |
| sngru\|H10\|seed43 | 3.3385 | +0.0000 | signal-net code |
| gru-all4\|neuron\|eng | **0.0478** | **−0.0001** | `heads(x)` (per-sample) |
| nerisez\|gru1\|er-own\|sgd\|running | 0.7317 | +0.0197 | `relu(projx(x))` (per-sample) + live stats |

The largest gate in the study (2.92) moves by exactly zero while a gate 60x smaller (0.048) is the only
one of its three siblings to move at all. **A large gate that is CONSTANT is order-invariant.** The
mechanism is legible from `signalnet_traces.md`: the signal net emits a near-constant code at inference
(across-batch sd ~1e-4 against a magnitude of 4–1300, because 13 of its 23 features are broadcast running
scalars frozen at eval and 6 more are head predictions of scalar targets), so `cell(a.mean(0), hidden)`
receives the same vector in every batch **whatever the batch composition**. Given a varying input, `|g|`
then scales the size of a nonzero effect (`nerisez|gru0` running: |g| 0.0147 -> +0.0002, 0.0812 -> +0.0011).

**So: input variability decides WHETHER there is an effect; gate magnitude only scales one that exists.**

## (2) CORRECTION — the `sngru` FROZEN cells are order-INVARIANT in practice

`GRUOnVec.forward` defaults to `update_state=True`, so the GRU hidden advances at eval even in cells
labelled "frozen" (a real inconsistency in the frozen protocol). It was reasonable to expect those cells to
be order-sensitive. They are not: **every frozen-protocol signalnet-family cell reads identically to all
six decimal places under both orders** (16 cells: 2 traces-frozen-sngru + 2 pt7_signalnet + 9 capacity +
3 others), pooled AND macro. The state advances; the stream it advances on carries no variation. The
dependence is structurally real and numerically void.

Separately, `pt7_stateful`'s `frozen|gru1` cells are order-sensitive by a DIFFERENT mechanism than
persistence: in frozen mode `update_state=False`, so the hidden does not advance, but `h_new` is still
recomputed from each batch's mean (`cell(p.mean(0,keepdim=True), hidden)`). Persistence is sufficient, not
necessary — a batch statistic suffices.

## (3) HOW MUCH OF THE "LIVE UPDATING HURTS" RESULT IS THE ORDER? 16%–85%, NOT A CONSTANT

For each cell that live/running updating penalised, the share of the penalty recovered by shuffling:

| cell | frozen | live, ordered | live, shuffled | order-share |
|---|---|---|---|---|
| sngru\|predH\|live | 0.8657 | 0.7141 (−0.1516) | 0.8436 (−0.0221) | **85.4%** |
| nerisez\|gru1\|er-own\|sgd\|running | 0.6434 | 0.6200 (−0.0234) | 0.6397 (−0.0037) | **84.2%** |
| sngru\|actualH\|live | 0.8845 | 0.8521 (−0.0324) | 0.8785 (−0.0060) | **81.5%** |
| sn\|actualH\|live | 0.7137 | 0.3447 (−0.3690) | 0.4043 (−0.3094) | **16.2%** |
| sn\|predH\|live | 0.5215 | 0.6023 (+0.0808) | 0.6747 (+0.1532) | (no penalty) |

**Three of four recover ~85%, but the LARGEST collapse (`sn|actualH`, −0.369) recovers only 16%** — that
one is genuinely live updating destroying the cell, not a protocol artefact. Do not generalise the 85% to
the whole family; it is specifically the GRU cells and the small penalty. NOTE `sn|predH|live` had no
penalty to recover (live *improves* it, 0.52 -> 0.60, which `signalnet_traces.md` reads as "less collapsed
rather than good") and shuffling improves it further to 0.6747 — still below ER 0.8946.

## (4) `pt7_stateful`'s recorded "FROZEN ~= RUNNING (+-0.003)" IS AN UNDERSTATEMENT — and shuffling FIXES it

Recomputed from that study's own ledger, under its own protocol, **4 of 12 frozen/running pairs exceed
+-0.003, max 0.0234** (~8x the stated bound). Under a shuffled test order the spread TIGHTENS:

| | pairs > +-0.003 | max \|gap\| |
|---|---|---|
| ordered (the frozen protocol) | 4/12 | 0.0234 |
| shuffled | 2/12 | 0.0064 |

So the study's qualitative conclusion — inference-time statefulness buys nothing — SURVIVES and is
strengthened (no running cell beats its frozen twin by more than +0.0013 under either order); what was
wrong was the error bar, and its worst outlier was an artefact of the blocked/ramped test stream.

## (5) The metric choice does not drive any of this

Both statistics are reported. The frozen studies pool (`c/tot` over all 10 000 samples) while the frozen
BASELINES macro-average five per-task accuracies — a systematic ~0.0015 gap. Macro is recovered under both
orders by binning each sample by its own label (class c -> task c//2). **`d-macro` tracks `d-pooled` to
within 0.0037 in every cell** (and within 0.0007 in all but the largest), so no conclusion here depends on
which statistic is used. The absolute pooled/macro gap is visible as expected, e.g. the diverged cells read
pooled 0.0980 / macro 0.0927 — the exact divergence signature (980/2115 on task 0, zero elsewhere, /5).

## Correctness controls (both designed in, both passed)

1. **The immune control.** `nerisez|gru0|frozen` is per-sample by construction (MLP predictor, frozen
   stats), so its shuffled accuracy MUST be bit-identical. It is, at a healthy 0.8892 (not a degenerate
   operating point) — confirming the shuffled loader re-partitions the same 10 000 samples and nothing else.
2. **Cross-study training parity.** `sngru|H400|seed42` was trained fresh here and reached 0.8845, matching
   both the capacity ledger AND the `signalnet_traces` checkpoint loaded in the same run.

## What was not run

- **Case 4**: the `true` diagnostic column of `results/pt7_neuromodulators.py` (26 cells, order-sensitive
  via `DA`'s within-batch `ell.std()`), scoped out by the user.
- **The provably-inert cells**: the 6 non-`eng` GRU rows of `pt7_signalnet` and `sngru-dead` in the
  capacity ledger all log `|g| = 0.0000` (the double-zero-init saddle), so gamma == 1 exactly and no order
  effect can exist.

## Limits

One shuffled permutation (seed 1234) per cell — a permutation ENSEMBLE would give an error bar on the
shuffled number itself, which this does not. 1 seed except the 9 capacity cells (3 seeds x 3 widths).
6 of the 44 cells sit at the 0.0980/0.1009 divergence signature or at H<=10 collapse, where order cannot
express anything. No retuning (by request), so the operating points are inherited.
