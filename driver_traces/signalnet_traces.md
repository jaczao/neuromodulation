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

Everything else is identical to the frozen study, **including `update=False` at eval** (running scalars
frozen at test). This study deliberately does not inherit `live_traces.py`'s live-stats premise.

## Anchor

The `pred-H` cells are the frozen configuration, and tracing is read-only, so they must reproduce
`pt7_signalnet_results.tsv`. Both do, **bit-exact on accuracy and on all three gate magnitudes**:

| cell | pred | Δ vs ledger | \|g\| h0/h1/out | Δ\|g\| |
|---|---|---|---|---|
| signalnet \| pred-H | 0.5215 | +0.0000 | 0.4726 / 0.5269 / 0.8037 | +0.0000 / +0.0000 / +0.0000 |
| signalnet-gru \| pred-H | 0.8657 | +0.0000 | 1.5248 / 1.3156 / 1.9400 | −0.0000 / +0.0000 / −0.0000 |

## Findings

### 1. At inference the signal net's output is a CONSTANT

This is the headline, and it is not marginal. Every traced dimension, every cell, at test:

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

### 3. The GRU is a magnitude compressor

At train, `signalnet-gru`'s pre-GRU code is *larger* than plain `signalnet`'s (mean|·| ≈ 800 vs 263), but
its post-GRU output is ~9× smaller (≈ 85). Relative dispersion is unchanged — the GRU compresses a
constant into a smaller constant.

That matches pt7's accuracy ordering (signalnet-gru 0.8657 > signalnet 0.5215 at matched K) and its
reading that "the GRU's temporal smoothing partly stabilises a large gate". The traces make it concrete:
the stabilisation is amplitude reduction, not the addition of temporal structure — statefulness buys no
per-sample variation, because there is none in the input to carry.

### 4. Nothing beats replay

Best cell 0.8845 against ER 0.8946. Consistent with the pt7 controlled-negative and with the project's
class-IL headline.

## Caveats

- **1 seed**, engage variant only, K=4, neuron, std1. The canonical (non-engage) config is the dead
  saddle and is not traced here — that result already exists.
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
```
