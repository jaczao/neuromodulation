# er-own plasticity driven by pt7 neuromodulator signals — task-IL, SGD, tuned

Five pt7 content drivers (not the task id) × three granularities {global, neuron, synapse}, on the
er-own plasticity arm, task-IL, main lr 0.1 / ep 5 (`plast_taskil`'s val-tuned ER-SGD point),
neuro_lr 1e-3, buffer 1000, 3 seeds. Code `plast_drivers.py`, log `plast_drivers.log`, ledger
`plast_drivers_results.tsv`.

**Verdict: REJECT all 14 cells** (d-dead from −0.0046 to +0.0006, nothing positive beyond noise) —
and the study produces one result that *sharpens* pt7's negative rather than repeating it.

## Anchor

The loop is a copy-forward of `results/pt7_plast_tempslope.run_plast`; `--part anchor` reproduces
that frozen ledger's `ach_ema × {neuron, synapse, global}` cells **exactly** (0.9017 / 0.9010 /
0.9019). The dead-control granularity check also passes: at a fixed driver the neuro_lr=0 control is
byte-identical across all three granularities (`ach` 0.993482, `vecproj` 0.994133), confirming it
depends only on the driver (through `Heads(K)` RNG) and not on the gate shape.

## Results (3 seeds, TEST)

| driver | gran | acc | d-dead | pos | probe | \|α−1\| h0/h1/out |
|---|---|---|---|---|---|---|
| ER (plain) | — | 0.9934±0.0006 | — | | | |
| ach | DEAD | 0.9932±0.0006 | — | | | |
| ach | global | 0.9933 | +0.0001 | 3/3 | 0.384 | 0.031 |
| ach | neuron | 0.9933 | +0.0000 | 1/3 | 0.382 | 0.075/0.054/0.029 |
| ach | synapse | 0.9932 | −0.0001 | 1/3 | 0.385 | 0.026/0.019/0.028 |
| ach_ema | global | 0.9938 | **+0.0006** | 2/3 | 0.237 | 0.805 |
| ach_ema | neuron | 0.9933 | +0.0000 | 1/3 | 0.244 | **1.579**/0.464/0.037 |
| ach_ema | synapse | 0.9932 | −0.0000 | 1/3 | 0.235 | 0.286/0.119/0.077 |
| nerisez | global | 0.9935 | +0.0002 | 2/3 | 0.329 | 0.087 |
| nerisez | neuron | 0.9938 | **+0.0006** | 2/3 | 0.313 | 0.329/0.113/0.026 |
| nerisez | synapse | 0.9933 | +0.0001 | 2/3 | 0.330 | 0.053/0.021/0.019 |
| vec_x | global | 0.9927 | −0.0007 | 0/3 | **0.934** | inf (see below) |
| vec_x | neuron | 0.9888 | **−0.0046** | 0/3 | **0.934** | inf |
| vec_x | synapse | NOT RUN — K=784 ⇒ P is 3.7e8 params (~500× the backbone) | | | | |
| vecproj | global | 0.9936 | +0.0002 | 3/3 | 0.685 | 0.183 |
| vecproj | neuron | 0.9935 | +0.0001 | 1/3 | 0.685 | 0.273/0.210/0.105 |
| vecproj | synapse | 0.9934 | −0.0000 | 1/3 | 0.685 | 0.053/0.053/0.101 |

Reference (`plast_taskil`, prototype harness): naive 0.9784, EWC 0.9821, ER 0.9946. This harness's
ER is 0.9934 — the two agree to 0.0012, and every dead control sits on it.

## The result worth keeping: task-decodability is NOT the binding constraint

pt7 explained its neuromodulator negative mechanistically: the gate's task-decodability probe was
0.21–0.52 vs pt6's learned selector at 0.88, i.e. *difficulty/novelty is not task identity*, so the
gate could not allocate per task.

**This study breaks that explanation for the plasticity target.** `vec_x` has **probe 0.934** —
more task-decodable than pt6's oracle-free selector (0.884), the most task-informative content
driver measured anywhere in this project — and it is the **worst cell in the table** (−0.0046,
negative in all 3 seeds). Meanwhile `ach_ema`, the *least* decodable (probe 0.24), is tied for best.
Across the 14 cells there is no relationship between probe and d-dead, and if anything it is
inverted.

So for gating gradients, "the driver doesn't know the task" was never the binding constraint. A
driver that *does* know the task still buys nothing. The constraint is the lever itself — the same
conclusion the task-id-driven studies in this package reached from the opposite direction (there the
driver was maximally task-informative *by construction* and the gate still collapsed to a global LR
knob).

## Engagement without benefit, again — and pt7's ach_ema artifact dissolving on cue

`ach_ema`/neuron drives |α−1| to **1.58** on h0 (α ≈ 2.6× the base LR on the first layer) for
**d-dead = +0.0000**. `ach_ema`/global reaches |α−1| = 0.805, a large uniform LR rescale, for
+0.0006. That is precisely pt7's `ach_ema` "global LR boost" — worth +0.11 there at an *untuned* SGD
lr — reappearing at a tuned lr with its effect gone, now replicated in task-IL. Consistent with
`pt7_capacity` and `meta_schedule`: engagement rises, benefit does not.

**Granularity is inert.** global ≈ neuron ≈ synapse for every driver (spread ≤ 0.0006 within a
driver), replicating pt7's SET-1 finding that one scalar matches per-neuron and per-synapse
structure. Note this is partly structural: the driver is mean-pooled over the batch, so for a
plasticity target every granularity is equally constant along the *sample* axis and only the
parameter axis differs.

## "No freezes at inference": implemented, and provably inert here

Requested and implemented (running stats / predictor state keep updating through the test pass, per
`driver_traces/live_traces.py`). For a plasticity target it **cannot** move accuracy — the gate
multiplies gradients, never the forward — and the run prints frozen-vs-live `pred` as a check rather
than an assertion: they are **identical in all 42 cells**, to full printed precision.

Two further structural facts, worth stating because they make the request partly vacuous by
construction rather than by measurement: of the five drivers only `vec_x`/`vecproj` (running input
mean) and `nerisez` (actual-entropy stats, advanced live here) have *any* inference-time state;
`ach`/`ach_ema` are head predictions m(x) = heads(x), pure functions of the image, so frozen and live
are the same computation.

## `vec_x` overflows: standardising a driver with constant dimensions

`vec_x`'s |α−1| reads `inf`. Measured cause: **212 of its 784 dimensions have `run_var == 0`** —
MNIST border pixels are constant — so standardisation divides by ~EPS. During training the numerator
is ~0 there too, so values stay bounded (max |m| ≈ 80, already large); at test a single nonzero
border pixel gives |m| ≈ 2×10⁶, and `exp(m@P)` overflows.

- The `inf` is diagnostic-only *here* (the gate is not in the forward, so accuracy is unaffected — a
  forward-gain target would have collapsed to chance, the pt7 "standardize or the gate blows up"
  signature).
- The training-time magnitude ~80 is the plausible reason `vec_x`/neuron is the one cell that
  actively *hurts*.
- This is the **per-dimension analogue of the tonic-driver rule**: CLAUDE.md says "standardize
  per-sample drivers, never a tonic one" — `vec_x` is per-sample *overall* while 27% of its
  dimensions are individually tonic. `vecproj` escapes it because the random projection mixes all
  pixels, so all 32 dims have real variance — a second reason (beyond parameter count) that it is
  the usable form of this driver.

## Limits

- `neuro_lr` fixed at 1e-3 for every cell (identical budget, rule #3) rather than swept — stated as a
  limit. Four consecutive `neuro_lr` sweeps in this package came back unresolved inside the noise
  floor, so 75 more val runs would buy little, but a per-cell optimum cannot be ruled out.
- `vec_x`×synapse not run (parameter count above; by `pt7_capacity`'s rule a modulator ~500× its
  backbone makes any result a capacity confound).
- The er-own arm has very little headroom: the ER backbone is already at 0.9934 here, so the whole
  distance to `plast_taskil`'s ER is 0.001 and to a perfect score 0.007. A null in this regime is
  weaker evidence than the same null in a regime with room to move.
- 3 seeds; task-IL, SGD, one operating point.
