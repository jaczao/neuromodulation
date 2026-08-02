# Signal-net code traces — what the signal net actually emits, per dimension

Mechanism run (not observer-only): the signal net has no target of its own, so its code exists only when
the gate is applied and the main CE trains it end-to-end. This runs the real `pt7_signalnet` mechanism and
traces the K-dim vectors inside it.

Script: `signalnet_traces.py` · outputs `signalnet_traces.npz`, `figs_signalnet/*.png`, `.log`
Cells: `{signalnet, signalnet-gru} × {actual-H, pred-H}`, **engage=True**, K=4, neuron, std1, er-own,
class-IL, Adam, lr 1e-3 / 5 ep per task / buffer 1000, seed 42. **1 seed.**

`engage=True` throughout because the canonical config zero-inits both the signal net's output layer and
the gate `P` — the double-zero saddle, where the code stays exactly 0 for the whole run. Tracing that
would give four flat zero lines.

## The one deviation

Feature column 8, "predicted entropy of the current sample", takes the **actual** entropy from an extra
unmodulated forward instead of the head's prediction. Entropy needs no labels, so it is exactly
computable at inference for the price of that forward — the head was never necessary for it.

Applied via a shim on `SignalHeads` rather than by editing `SignalFeatures.build`: `build` reads col 8 as
`pred[:, 0:1]`, so replacing head output column 0 changes exactly that one feature, and does it *before*
the 23-dim standardisation — the only place it can go without desynchronising the running stats from the
values they normalise. The head's own MSE step still sees the unwrapped head.

Everything else is identical to the frozen study.

## The eval protocol is an axis, not a choice

Each trained cell is evaluated **twice**, from the same weights:

- **`frozen`** — `update=False`, the frozen study's protocol and the ledger anchor.
- **`live`** — `update=True`, nothing frozen (`live_traces.py`'s premise).

Neither pass is side-effect-free, so both are snapshotted and restored between passes: `feat` holds the
running scalars that `update=True` mutates, and `GRUOnVec.forward` defaults to `update_state=True`, so
the **GRU hidden advances at eval in BOTH modes** — an inconsistency already present in the frozen study,
whose running scalars are frozen at test while its GRU hidden is not. `frozen` runs first because
`update=False` leaves `feat` untouched, which keeps the anchor exact (verified: it does).

The live pass needs no labels. `SignalFeatures.build` already supports `update=True` with `y=None`, where
it falls back to the **predicted** loss for the actual-loss running state. Worth holding onto when
reading the results: during training those same EMAs are driven by the *true* masked CE, so `live` is not
"the same statistic kept current" — partway through, the statistic switches to a different input signal.

## Anchor

The `pred-H` cells are the frozen configuration, and tracing is read-only, so they must reproduce
`pt7_signalnet_results.tsv`. Both do, **bit-exact on accuracy and on all three gate magnitudes**:

| cell | pred | Δ vs ledger | \|g\| h0/h1/out | Δ\|g\| |
|---|---|---|---|---|
| signalnet \| pred-H | 0.5215 | +0.0000 | 0.4726 / 0.5269 / 0.8037 | +0.0000 / +0.0000 / +0.0000 |
| signalnet-gru \| pred-H | 0.8657 | +0.0000 | 1.5248 / 1.3156 / 1.9400 | −0.0000 / +0.0000 / −0.0000 |

## Findings

### 1. At inference the signal net's output is a CONSTANT

This is the headline, and it is not marginal. Every traced dimension, every cell, at test under the
**frozen** protocol (finding 3 shows what the live protocol does to it — it does not rescue this):

| cell / dim | test \|mean\| | across-batch sd | within-batch sd |
|---|---|---|---|
| sngru \| actualH code1 | 1303 | 8.2e-03 | 2.7e-02 |
| sngru \| predH code3 | 1163 | 4.3e-02 | 1.0e-01 |
| sn \| actualH code2 | 111.4 | 4.3e-03 | 1.1e-02 |
| sngru \| actualH gruout1 | 73.29 | 7.1e-04 | 2.9e-03 |
| sngru \| predH gruout0 | 119.1 | 4.6e-03 | 1.0e-02 |
| *(all 24 dims)* | 4.4 … 1303 | 1.1e-04 … 4.3e-02 | 3.5e-04 … 1.0e-01 |

Both variation scales sit **3 to 5 orders of magnitude below the magnitude**. The code does not vary
across samples within a batch, and it does not vary across test batches either — the test panels are flat
horizontal lines. To four or five significant figures, the vector the gate consumes at eval is a fixed
learned constant with no dependence on the input at all.

So the 23-dim signal vector, the three hidden layers, the low-D bottleneck and the GRU together collapse,
at inference, to **a constant gain vector**. That is exactly pt7's `5ht-const` control — a learned
constant gate with no x-dependence, whose whole point was to be the scale-degeneracy null. The engaged
signal net is not a different mechanism from that control; it is an expensive way to arrive at it.

**Why it happens** is legible from the feature construction. 13 of the 23 inputs are broadcast running
scalars (`sc(v)` → `full((B,1), v)`), and they are frozen at eval, so they are literally identical on
every test batch. Six more (cols 15–17, 21–23) are head predictions of *scalar* targets — the head is
trained to regress a constant, so it emits one. Only four inputs are genuinely per-sample: cols 8
(entropy), 9 (‖h1−mean_h1‖), 10 (‖x−mean_x‖) and 11 (predicted loss). Their contribution is swamped by
the DC offset the other nineteen produce.

### 2. Using the actual entropy helps, and not by restoring per-sample information

| cell | pred-H | actual-H | Δ |
|---|---|---|---|
| signalnet | 0.5215 | **0.7137** | **+0.192** |
| signalnet-gru | 0.8657 | **0.8845** | +0.019 |

Both deltas are against a bit-exact anchor, so they are real, and the large one rescues the catastrophic
K4 collapse most of the way. Neither reaches ER (0.8946).

But the traces rule out the obvious explanation. The relative dispersion `sd/|mean|` is unchanged by the
substitution (~1e-4 in both arms) — the code is just as constant with the actual entropy as with the
predicted one. Whatever the substitution fixed, it is not that the gate became more input-sensitive.

I can say what it is *not*, and I will not guess at what it is. It is also not a simple magnitude story:
the code's DC magnitude falls for `signalnet` (train mean|·| ≈ 263 → 176), which would fit "smaller
constant offset does less damage", but the *applied* gate magnitude moves the opposite way
(|g| 0.47/0.53/0.80 → 2.05/3.18/2.93) while accuracy improves. `P` grew as `m` shrank. Establishing the
mechanism would need a further run, not these traces.

### 3. Unfreezing the running scalars makes the constant WANDER — it does not make it per-sample

The same structural result as `ACh_ema` in `live_traces.md`, arriving through a completely different
mechanism, which is what makes it worth stating as a rule.

| cell / dim | phase | \|mean\| | across-batch sd | within-batch sd |
|---|---|---|---|---|
| sn \| actualH code0 | frozen | 4.4 | 3.7e-03 | 7.4e-03 |
| sn \| actualH code0 | **live** | 104.6 | **1.2e+02** | 7.0e-03 |
| sn \| predH code0 | frozen | 19.0 | 9.3e-03 | 2.6e-02 |
| sn \| predH code0 | **live** | 572.9 | **5.4e+02** | 1.1e-02 |
| sngru \| actualH code0 | frozen | 8.8 | 1.7e-02 | 4.8e-02 |
| sngru \| actualH code0 | **live** | 2995 | **1.3e+03** | 3.1e-02 |
| sngru \| predH gruout0 | frozen | 119.1 | 4.6e-03 | 1.0e-02 |
| sngru \| predH gruout0 | **live** | 360.5 | **2.3e+02** | 4.9e-03 |

Across-batch SD rises by **four to five orders of magnitude** (1e-2 → 1e2…1.8e3). Within-batch SD does
not move at all (7.4e-03 → 7.0e-03; 4.8e-02 → 3.1e-02). And because the magnitude inflates ~20× at the
same time, the ratio `within/|mean|` gets *worse*, from ~1e-3 to ~1e-5.

So unfreezing converts a fixed global gain into a **wandering global gain**. The gate becomes
time-varying, and remains exactly as blind to the individual sample as before. Live statistics can only
ever restore *between-batch* variation; per-sample variation has to be present in the features, and here
19 of 23 columns cannot supply it under either protocol.

### 4. Live eval inflates the gate everywhere and mostly hurts

| cell | frozen | live | Δ | \|g\| out frozen → live |
|---|---|---|---|---|
| signalnet \| actual-H | 0.7137 | 0.3447 | **−0.369** | 2.93 → 5.21 |
| signalnet \| pred-H | 0.5215 | 0.6023 | +0.081 | 0.80 → **17.21** |
| signalnet-gru \| actual-H | **0.8845** | 0.8521 | −0.032 | 1.57 → 6.72 |
| signalnet-gru \| pred-H | 0.8657 | 0.7141 | −0.152 | 1.94 → 7.11 |

Gate magnitude inflates in **all four** cells, by 2× to 20× — that part is universal and follows directly
from finding 3's magnitude blow-up. Accuracy is not universal: three cells degrade, one improves, and the
one that improves is the already-collapsed `signalnet | pred-H` (0.52 → 0.60), i.e. less-collapsed rather
than good. The best cell overall is still frozen `signalnet-gru | actual-H` at 0.8845.

The non-stateful net takes by far the worst damage (−0.369 vs −0.032 for its stateful counterpart at
matched entropy source), which is finding 5 again: the GRU's amplitude compression is what absorbs it.

I am not claiming a mechanism for the *sign* of the accuracy change. With 1 seed, a protocol that also
switches which quantity drives the loss EMAs (see above), and a gate that is over-modulating in every
cell, the sign is not something these runs can settle.

### 5. The GRU is a magnitude compressor

At train, `signalnet-gru`'s pre-GRU code is *larger* than plain `signalnet`'s (mean|·| ≈ 800 vs 263), but
its post-GRU output is ~9× smaller (≈ 85). Relative dispersion is unchanged — the GRU compresses a
constant into a smaller constant.

That matches pt7's accuracy ordering (signalnet-gru 0.8657 > signalnet 0.5215 at matched K) and its
reading that "the GRU's temporal smoothing partly stabilises a large gate". The traces make it concrete:
the stabilisation is amplitude reduction, not the addition of temporal structure — statefulness buys no
per-sample variation, because there is none in the input to carry.

### 6. Nothing beats replay

Best cell 0.8845 (frozen eval) against ER 0.8946. Consistent with the pt7 controlled-negative and with the project's
class-IL headline.

## Caveats

- **1 seed**, engage variant only, K=4, neuron, std1. The canonical (non-engage) config is the dead
  saddle and is not traced here — that result already exists.
- Cells are now checkpointed (`ckpt_sn_*.pt`, gitignored), so a further eval-side change costs seconds.
  The first version of this study had no checkpointing, which is why adding the live-eval axis required
  a full retrain — the trap CLAUDE.md records for `pt7_driver_traces`.
- Code dimensions have no semantic identity; they are exchangeable. Their panels answer whether the code
  is degenerate, not what any one dimension "means".
- The actual-H arm costs one extra `net.plain` forward per call. At train `build` already computes the
  actual entropy internally, so the forward is redundant there; at eval it is the substitution itself.
- Accuracy deltas are 1-seed. The trace conclusions (constancy by 3–5 orders of magnitude) are far
  outside any seed noise, but the +0.192 would need 3 seeds to report as a number.

## Reproduce

```bash
uv run python driver_traces/signalnet_traces.py               # 4 cells, ~4 min each on MPS
uv run python driver_traces/signalnet_traces.py --plot-only
uv run python driver_traces/signalnet_traces.py --only 'sn|predH' --force   # just the anchor
# every cell is evaluated twice (frozen + live); phases in the npz are train / test / testlive
```
