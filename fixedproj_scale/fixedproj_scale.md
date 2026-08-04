# Fixed-random projection: does a BIGGER gate change pt7's negative?

**No — and the offset axis is the more damaging one, which was the opposite of the prediction.**

`results/pt7_signalnet.py --part all4fixed` froze the rank-K projection to a zero-mean gaussian and
found it tied ER, but only ever at sigma in {0.1, 0.3}, always zero-mean, always on `all4`. This
study extends that to sigma in {1, 10} and to a NON-ZERO MEAN, on three oracle-free head-free drivers
at a val-tuned operating point.

Mechanism: `Gamma_i = 1 + sum_k m_ik P_k`, `P_kj ~ N(pmean, pstd^2)` FROZEN (in no optimizer), gain
target on (h0, h1, out), er-own arm, class-IL, buffer 1000, seed 42, **1 seed**. lr/epochs read from
`neurocore.tuned` and NOT retuned (sgd 3e-2/ep5, adam 3e-4/ep5). Drivers UNSTANDARDISED as requested;
`ach` and `nerisez` use the ACTUAL-value convention (one extra unmodulated forward, no head).

Ledger `fixedproj_scale_results.tsv`, table via `--part table`.

## Result: 36 of 36 cells are negative, and accuracy is MONOTONE in gate magnitude

| driver / gran / opt | (mu=0,sd=1) | (1,1) | (10,1) | (0,10) | (1,10) | (10,10) |
|---|---|---|---|---|---|---|
| ach / syn / **sgd**     | 0.8882 | 0.8707 | *0.0927* | *0.0927* | *0.0927* | *0.0927* |
| ach / syn / **adam**    | **0.8972** | 0.8839 | 0.3582 | 0.8396 | 0.8211 | 0.7612 |
| nerisez / syn / **sgd** | *0.0927* | *0.0927* | *0.0927* | *0.0927* | *0.0927* | *0.0927* |
| nerisez / syn / **adam**| 0.8736 | 0.8748 | 0.7623 | 0.8496 | 0.8289 | 0.7900 |
| vecproj / neu / **sgd** | *0.0927* | *0.0927* | *0.0927* | *0.0927* | *0.0927* | *0.0927* |
| vecproj / neu / **adam**| 0.5807 | 0.6206 | 0.3317 | 0.2465 | 0.2480 | 0.2008 |

Dead control = plain ER = **0.9034** (sgd) / **0.8975** (adam). `d-dead` spans **-0.0003 to -0.8107**;
nothing is positive. *Italic* = the divergence signature (see below).

**The best cell in the entire grid is the one with the smallest gate** (ach/adam, mu=0 sd=1,
|g| 0.038/0.051/0.061), and it lands at -0.0003 = exactly the dead control. There is no interior
optimum on either axis: every step away from parity costs accuracy, monotonically.

## The prediction that was wrong: a non-zero MEAN is WORSE, not weaker

Pre-registered (in the module docstring, before running): `pmean` would be the *weaker* axis per unit
of |g|, because a common-mode term is a per-sample GLOBAL gain — absorbable on a ReLU layer and
argmax-invariant on the logits. **Measured: it is substantially more damaging at matched |g|.**

The clean matched pair is nerisez/adam, where two cells have nearly identical h0 gate magnitude:

| cell | \|g\| h0/h1/out | acc |
|---|---|---|
| mu=0, sd=10 | 0.080 / 0.114 / 0.125 | 0.8496 |
| mu=10, sd=1 | 0.072 / 0.097 / 0.188 | **0.7623** |

Same perturbation size, **-0.087** for putting it in the common mode. ach/adam shows it larger and
in the same direction (mu=0/sd=10 -> 0.8396 at |g|h0 0.35; mu=10/sd=1 -> 0.3582 at |g|h0 0.44).

**Why the argmax-invariance argument fails.** It was right about the out layer *in isolation* and
irrelevant, because the gate is applied at h0, h1 AND out. ReLU is positively homogeneous, so a
common-mode factor `1 + mu*sum_k m_ik` at each of three gated layers multiplies the logits by roughly
its CUBE, whereas a zero-mean P gives each unit an independent signed coefficient whose effect on a
sum over 400 units largely averages out. With `ach` raw entropy reaching m ~ 2.3, mu=10 is a logit
rescale of order `(1+23)^3 ~ 1.4e4`. That distorts the softmax temperature and hence the TRAINING
signal — uniform positive scaling is argmax-invariant at eval, so all of the damage is done during
training, and by then the weights are wrong. **Offset compounds across depth; spread does not.**

## Three other things the grid shows

**1. `0.0927` is a precise divergence signature, not "chance".** Under the macro metric a NaN network
argmaxes to class 0 everywhere, giving task-0 accuracy 980/2115 = 0.4634 and zero elsewhere:
0.4634/5 = 0.0927 exactly. (pt7's pooled metric reports the same state as ~0.098.) Every italic cell
above is a fully diverged net, not a degraded one.

**2. SGD at its tuned lr has almost no tolerance for a fixed gate; Adam degrades gracefully.** At
lr 3e-2 only `ach` survives at all (and only at sd=1), because its raw driver is smallest
(|m| 0.175 vs nerisez 0.69). Adam's per-parameter normalisation absorbs an order-1 multiplicative
perturbation and yields a readable dose-response curve instead of a cliff. This is the same
optimizer asymmetry pt7 recorded for un-standardised drivers, now with the projection as the cause
rather than the driver.

**3. The most task-decodable driver is the worst one — again.** `vecproj` probes at **0.685**, far
above the entropy family's 0.21-0.30, and it is the only driver that fails at EVERY sigma under BOTH
optimizers (best cell 0.6206, i.e. -0.277). Because it is computed from the input alone, its probe
and |m| are **identical in every cell** (0.685 / 0.689) even where the network has been destroyed —
the driver is perfectly intact while the net is rubble. That is `pt5_taskil/plast_drivers.py`'s
finding (probe 0.934 = worst cell) reproduced on a FORWARD gain target: task-decodability of the
driver does not predict benefit, and here is anti-correlated with it.

**4. `nerisez` under SGD self-destructs in the way `live_traces.md` predicted.** |m| is **exactly 0**
in all six cells: once the net diverges, entropy is constant, so `H - ema_H <= 0` and the relu clamps
to zero forever. The rectifier makes the driver a one-way door — the same collapse mechanism
`driver_traces/live_traces.md` measured for live statistics, arriving here through divergence.

## What this does and does not close

**Closes:** the projection-scale explanation for the pt5/6/7 gating null. Combined with the frozen
studies, the magnitude axis is now mapped end to end at 1 seed —

| \|g\| | source | result |
|---|---|---|
| 0.006-0.08 | learned all4 | = ER |
| 0.33-0.43 | fixed-random sigma 0.1-0.3 (`pt7_capacity`) | = ER |
| 0.04-0.7 | **this study, adam** | **-0.000 to -0.54, monotone down** |
| 0.9-2.5 | signalnet engaged (`signalnet_capacity`) | = ER at H=400, chance at H<=10 |
| 3-52 | **this study, sgd / vecproj** | **divergence** |

Nowhere on that curve is there a gain. "The drivers had too little influence" is not why
neuromodulation fails here: given more influence they do strictly more harm.

**Does not close:** 1 seed (though -0.02 to -0.81 is far outside the 0.007 noise floor; only the
ach/adam sd=1 cells are inside it). One arm (er-own), one metric (class-IL), one composite-free
driver set, gaussian only — no heavy-tailed or bimodal +-c projection, which is the remaining
distributional variant and the one that would come closest to the hard {0,1} freeze that actually
worked in pt5 iter-1. No cost columns (rule #12): this is an ablation inside pt7's existing arm, not
a new direction.

## Methodology notes

- **Anchors.** ER at the tuned point reproduces the frozen `pt7_tuned_syn` ledger bit-exact
  (sgd 0.9034, adam 0.8975).
- **The dead control came out EXACTLY equal to plain ER** for all three drivers and both optimizers.
  That is unusual for this project (rule #10 normally makes them differ by ~0.002) and is a property
  of head-free drivers: `ActualEntropy` builds no modules, `NEDriver` draws its projection from a
  private generator, and a zero-init gate draws nothing — verified by probing the global RNG state
  after each construction. It does NOT transfer to a head-based driver, where `Heads(K)` init does
  consume RNG.
- **Accuracy is the mean of the five per-task accuracies, for every arm.** The frozen pt7 eval POOLS
  (`c/tot`) while the frozen pt7 baselines macro-average, a systematic ~+0.0015 in the mechanism's
  favour. Measured here directly: the same dead-gate cell reads 0.9049 pooled and 0.9034 macro.
