# Capacity ablation — signal net and signal net + GRU (actual entropy)

Does the signal-net gating null survive a shrinking backbone? `results/pt7_capacity.py` asked this for the
canonical pt7 `all4` gain-neuron gate and found the delta never emerged, down to a collapsed H=5 net. This
runs the same sweep for the two richest mechanisms in the project — the 23-dim **signal net** and the
**signal net + GRU** — in their ACTUAL-ENTROPY variant, at H ∈ {400, 10, 5}.

Class-IL Split MNIST, Adam, lr 1e-3 / 5 ep per task / buffer 1000, gain-neuron K=4, std1, er-own,
3 seeds {42,43,44}, frozen eval protocol (`update=False`). 45 cells.

**The answer is not "no benefit" — it is active, escalating harm.** Both mechanisms are worse than their
own dead-gate control at every width, and the damage grows sharply as capacity shrinks: at H ≤ 10 both
collapse to chance while the identical architecture with the gate switched off still reaches 0.50–0.81.

## Design

Two deviations from `results/pt7_signalnet.py`, both mandatory:

1. **`engage=True`.** The canonical config zero-inits both the signal net's output layer and the gate `P` —
   the double-zero-init saddle (`dL/dP ∝ m = 0`, `dL/d(snet) ∝ P = 0`), so neither bootstraps and the code
   stays exactly 0. That cell *is* pt7's `free` control, not a test of the mechanism. `engage=True` gives
   the output layer a normal init while keeping `P` zero-init, so γ = 1 at step 0 but the code can move.
2. **Actual entropy.** Feature column 8 ("predicted entropy") is replaced by the real entropy from an extra
   unmodulated forward, via a shim on `SignalHeads` — entropy needs no labels, so it is exactly computable
   at inference. This is `driver_traces/signalnet_traces.py`'s variant.

The **dead-gate controls are the load-bearing part of the design.** `sn-dead` / `sngru-dead` are the same
mechanisms with `engage=False`: |g| is pinned at exactly 0, so they are numerically ER, but they construct
the identical modules and consume the identical torch RNG. `pt7_capacity` established that this matters
~0.002 at H=400 and ~0.06 at H=5, so plain `er` is *not* a valid reference at these widths. `sngru-dead`
is separate from `sn-dead` because the GRUCell consumes extra RNG — they are different draws.

## Anchor

All five H=400 / seed-42 references reproduce **bit-exact**, against two independently frozen studies:

| cell | got | expected | source |
|---|---|---|---|
| `er` | 0.8946 | 0.8946 | pt7 ledger |
| `sn` pred-H | 0.5215 | 0.5215 | `pt7_signalnet_results.tsv` (`eng` row) |
| `sngru` pred-H | 0.8657 | 0.8657 | `pt7_signalnet_results.tsv` (`eng` row) |
| `sn` actual-H | 0.7137 | 0.7137 | `driver_traces/signalnet_traces.md` |
| `sngru` actual-H | 0.8845 | 0.8845 | `driver_traces/signalnet_traces.md` |

This validates the copy-forwarded loop, the `width(H)` global-rebinding, and the entropy shim at once, so
what the sweep reports at H=10/5 is attributable to width, not harness drift.

## Results

```
  H          net  n            er       sn-dead            sn    sngru-dead         sngru
  400    478,410  3  0.8907±0.008  0.8979±0.004  0.7616±0.056  0.8951±0.007  0.8746±0.007
  10       8,070  3  0.8083±0.021  0.7746±0.056  0.1451±0.032  0.8063±0.011  0.2948±0.103
  5        4,015  3  0.5539±0.149  0.5826±0.027  0.0998±0.002  0.5013±0.084  0.1016±0.004

  H      mod/net       sn d-dead        sn d-er    sngru d-dead     sngru d-er
  400      0.08x  -0.1363±0.041  -0.1291±0.037   -0.0205±0.006  -0.0161±0.006
  10       4.41x  -0.6295±0.022  -0.6632±0.022   -0.5115±0.078  -0.5135±0.080
  5        8.86x  -0.4829±0.020  -0.4542±0.105   -0.3997±0.060  -0.4523±0.106
```

### 1. Harm at every width, escalating with scarcity

`d-dead` is negative in all six cells and far outside the ±0.007 noise floor. At H=400 the GRU variant is
only mildly negative (−0.021) and plain signalnet clearly negative (−0.136) — consistent with the frozen
study's ordering (0.8845 > 0.7137). Shrink the backbone and both fall off a cliff: −0.63 / −0.51 at H=10,
−0.48 / −0.40 at H=5. The H=5 deltas are *smaller in magnitude* than H=10 only because the dead control has
itself degraded to ~0.5 — there is less left to destroy, not less destruction.

### 2. At H ≤ 10 the engaged gate collapses to chance

Per-seed, `sn` at H=5 reads 0.1009 / 0.1009 / 0.0975 and `sngru` 0.1066 / 0.1009 / 0.0974 — that is 1/10,
pure chance, the same signature pt7 logged repeatedly as "the gate blows up → 0.098". At H=10, `sn` gives
0.1764 / 0.1581 / 0.1009. Meanwhile `sn-dead` at the same widths reaches 0.78 and 0.58. **The backbone is
perfectly capable of learning; the gate destroys it.** This is qualitatively different from `pt7_capacity`,
where the `all4` gate remained absorbed and harmless at every width.

### 3. Engagement stays high while the backbone vanishes

|g| does not shrink with H — it stays in the 0.9–2.5 band at every width for both mechanisms (`sn` at H=5:
1.19 / 1.10 / 1.84). The multiplicative deviation is order-1 or larger against an H=5 backbone of 4,015
parameters. A gate this large is not a modulation of the representation, it is a replacement of it, and
the collapse to chance follows directly.

The contrast with `pt7_capacity` locates the cause. There, `all4`'s |g| *rose* ~15× as capacity shrank
(0.014 → 0.207) and accuracy still did not move: a small learned gate driven by a per-sample bio signal is
absorbed. Here |g| starts an order of magnitude higher and the run collapses. The difference is not the
scarcity — it is that the signal net's code is a large, essentially input-independent quantity.

### 4. Why: the signal net emits a constant, and a constant of this size is fatal

`driver_traces/signalnet_traces.md` measured what the engaged signal net actually emits at inference:
every one of its 24 traced dimensions has |mean| between 4.4 and 1303 with within-batch SD of 3.5e-4 to
1.0e-1 — **three to five orders of magnitude below the magnitude.** 13 of the 23 input features are
broadcast running scalars frozen at eval, 6 more are head predictions of scalar targets; only 4 are
genuinely per-sample, and their contribution is swamped by the DC offset of the other 19.

So the mechanism reduces at inference to a *fixed learned gain vector* — pt7's `5ht-const` scale
degeneracy. At H=400 an over-parameterized backbone absorbs that rescale (the `bounded01` / h1-gate story),
which is why the frozen study read ≈ ER. At H=5 there is no spare capacity to absorb an order-1 constant
multiplicative distortion, so the run dies. **The capacity ablation converts a benign scale degeneracy into
a catastrophic one, without ever making the gate informative.**

### 5. The RNG-matching rule bites exactly as predicted

At H=400 the controls sit within 0.008 of plain `er`. At H=5 they span 0.5013 (`sngru-dead`) to 0.5826
(`sn-dead`) against `er` 0.5539 — a ±0.05 swing from provably inert gates that consume torch RNG and shift
the replay draws. `sn d-er` at H=5 (−0.454 ± 0.105) has more than five times the seed spread of the
RNG-matched `sn d-dead` (−0.483 ± 0.020). Reading against plain ER at these widths measures instability.

## Verdict

The pt5/6/7 gating null is confirmed **not** to be an over-parameterization artifact for these mechanisms
either — but the failure mode is the opposite of gentle. `pt7_capacity` found `all4` absorbed and harmless
under scarcity; the signal net and signal net + GRU are absorbed and harmless only *while the backbone is
large enough to absorb them*, and are catastrophic the moment it is not. Nothing here approaches ER at any
width. The project's class-IL headline is unchanged: replay is the only lever.

### Limits

- **The modulator dwarfs the backbone at small H** (`mod/net` 0.08× → 4.41× → 8.86× for signalnet, and
  0.11× → 6.11× → 12.28× with the GRU). Per the standing rule, this must be stated alongside any small-net
  result. Here it cuts *against* the mechanism rather than flattering it — a modulator 9–12× the size of
  what it modulates still produces chance accuracy — so it does not threaten the negative conclusion, but
  it does mean the H≤10 rows are not a clean test of "gain control under scarcity" in isolation.
- **Not re-tuned per width** (same caveat as `pt7_capacity`): lr/epochs are the standard pt7 point at every
  H, so absolute accuracies at small H are pessimistic. Every arm at a given H gets the identical budget,
  so the matched-H delta is unaffected.
- 3 seeds; `er` at H=5 is genuinely unstable (±0.149).
- Only the actual-entropy variant was swept, per the request. pred-H was run at H=400 for the anchor only.

## Files

- `signalnet_capacity.py` — study (`--part smoke|anchor|sweep|table|plot`, `--resume`)
- `signalnet_capacity_results.tsv` — ledger, 45 rows
- `signalnet_capacity.log`, `anchor.log`
- `figs/signalnet_capacity.png`, `figs/gate_engagement.png`
