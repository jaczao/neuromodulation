# Meta-schedule variants — standalone buf-cur plasticity, task-IL, SGD

Two variants of the modulator's training SCHEDULE, at `plast_init.py`'s best operating point
(init 0.5, ep 5, main lr 3e-3 SGD, buffer 1000, task-IL, learned P, 3 seeds, `neuro_lr` re-tuned on
val per variant). Code `meta_schedule.py`, log `meta_schedule.log`, ledger `meta_schedule_results.tsv`,
gate rows `gate_alpha_{multi5,phase}_{neuron,synapse}.npz`.

**Verdict: REJECT both.** Neither schedule changes the conclusion, and together they turn the
"under-optimised modulator" hypothesis from unrefuted into refuted — the better the modulator's
optimisation, the MORE completely its gate collapses onto a uniform global learning-rate knob.

## Why these two

Everything before this varied WHAT the gate is (granularity, init, `neuro_lr`) and left the schedule
alone: one meta update per main step, interleaved, on the same batch. `gate_stats.py` had explained
the null two different ways, and both point at the modulator's optimisation rather than its form —
the per-neuron gate engages hard but every task learns the same deviation (cos(dev) 0.97–0.99), and
the per-synapse gate never leaves parity at all. So:

- **`multi5`** — 5 sequential modulator updates per main step, main net held fixed, gate re-projected
  between passes (it follows the meta-loss curvature, not just a 5× lr) with a fresh buffer draw per
  pass. Tests "the modulator is under-optimised per step".
- **`phase`** — per task, phase 1 trains only the modulator (the main gradient is computed to build
  the lookahead but never applied), phase 2 trains only the main net under the now-frozen gate, on a
  **disjoint** half of the data. Tests "interleaving is the problem": the gate normally chases a
  backbone moving under it, so give it a stationary target first.

Each phase gets 2×ep epochs on half the data, so both nets take the same number of updates as base.
`multi5` is the one arm spending more meta updates (25N vs 5N) — that is the variable under test.

## Copy-forward parity (the precondition)

Both variants restructure `cl_train`'s inner loop, and `prototype/` is frozen (rule #9), so the pt5
plasticity branch was copy-forwarded here. `--part anchor` runs the `base` schedule (1 pass, no
phases) and **all 12 cells reproduce `plast_init_results.tsv` bit-exact** — both granularities × live
and dead × 3 seeds. Nothing below is trustworthy without that, and it holds.

## Results (3 seeds, TEST set)

| variant | gran | nlr | live | dead (α≡0.5) | d-dead | per seed | pos | d-forget |
|---|---|---|---|---|---|---|---|---|
| base | neuron | 1e-2 | 0.9778±0.0019 | 0.9754±0.0028 | +0.0025 | +0.0013 +0.0035 +0.0027 | 3/3 | |
| base | synapse | 1e-4 | 0.9767±0.0017 | 0.9767±0.0017 | +0.0000 | +0.0000 ×3 | 0/3 | |
| **multi5** | neuron | 3e-3 | 0.9780±0.0018 | 0.9754±0.0028 | **+0.0026** | +0.0014 +0.0039 +0.0027 | 3/3 | +0.0017 |
| **multi5** | synapse | 3e-4 | 0.9768±0.0015 | 0.9767±0.0017 | **+0.0001** | +0.0001 +0.0007 −0.0004 | 2/3 | +0.0001 |
| **phase** | neuron | 1e-2 | 0.9787±0.0015 | 0.9751±0.0026 | **+0.0036** | +0.0024 +0.0051 +0.0032 | 3/3 | +0.0016 |
| **phase** | synapse | 3e-4 | 0.9771±0.0017 | 0.9767±0.0019 | **+0.0004** | +0.0002 +0.0007 +0.0003 | 3/3 | −0.0003 |

Reference (`plast_taskil`, same point): naive 0.9784, EWC 0.9821, **ER 0.9946**.

`multi5` ≈ base exactly (+0.0026 vs +0.0025): **5× the meta-optimisation buys nothing.** `phase` is
the largest effect in the package (+0.0036, 3/3 seeds) — and the next section shows it is not
retention.

## The +0.0036 is LR restoration, not retention — three independent tells

1. **Forgetting goes UP as accuracy goes up.** Both neuron cells gain accuracy while forgetting rises
   (+0.0016/+0.0017 vs their dead controls). A plasticity gate is supposed to protect old weights;
   this is better ACQUISITION, the opposite signature.
2. **It lands exactly on the full-LR baseline, per seed.** `plast_taskil` has a fixed α=0.9999 dead
   gate (numerically naive at full LR): 0.980286 / 0.976374 / 0.978583. `phase`/neuron live:
   0.980173 / 0.976668 / 0.979153 — mean difference **+0.00025**, per-seed −0.0001/+0.0003/+0.0006.
   The entire d-dead is recovering the half-LR handicap the α=0.5 init imposes, and stops there.
3. **The gate says so.** More modulator optimisation ⇒ a MORE uniform gate, not a more allocated one:

   | | \|α−0.5\| h0 | α mean h0 | cos(dev) h0 |
   |---|---|---|---|
   | base | 0.433 | 0.933 | 0.9909 |
   | multi5 | 0.451 | 0.951 | 0.9937 |
   | phase | **0.491** | **0.991** | **0.9993** |

   `phase`/neuron per-task mean α on h0 is 0.983 0.993 0.992 0.994 0.993 — five tasks, one gate.
   Given a stationary target the modulator SOLVES its problem, and the solution is "open the gate".

**So the hypothesis both variants were built to test is refuted, not merely unsupported.** The
modulator was never under-optimised; optimising it better makes it converge harder onto the global LR
knob, because that is what the meta-loss actually rewards.

## Per-synapse still never engages

Under both variants the per-synapse gate stays at parity: |α−0.5| = 0.009 (`multi5`) / 0.006
(`phase`), no unit above 0.75, α max 0.63–0.75. This is nonetheless a **better-founded** negative than
base's, whose tuned `neuro_lr` (1e-4) was the grid floor = the gate's off-switch and whose live run was
byte-identical to its dead control. Here the gate is 8–13× more engaged than base's and still moves
nothing. Its cos(dev) 0.41–0.51 is noise around zero at |dev|≈0.006 — read cosine with magnitude.

## Context: how small this all is

`trajectories.py` showed task-IL at a tuned point has almost nothing to win — naive's total forgetting
is 0.0061, so a PERFECT retention mechanism tops out at 0.9845, still 0.010 below ER. The best cell
here (+0.0036) is about the size of EWC's +0.0037 and ~4.4× smaller than ER's +0.016, and it is not
retention. Every mechanism hyperparameter remains inert: **all four `neuro_lr` grids span 0.0006–0.0015,
below the 0.007 noise floor** — the axis is unresolved at 1 seed, not optimised.

## Limits

- `phase` trains its backbone on half of each task's unique images (no reuse between phases), so its
  absolute accuracy is comparable only to its own dead control, which carries the identical handicap.
  Read `d-dead`, not the live column against 0.9784.
- 1 seed for tuning, 3 seeds for every reported number. Main net SGD; the pt5 plasticity
  meta-optimizer is Adam by construction in the frozen loop.
- `M=5` and a 50/50 phase split are single points on two axes that were not swept.
- Only the committed-gate policy of the frozen loop was used (the main step commits the PRE-update
  gate). Committing the POST-update gate is a second axis, not taken, so that `multi5` differs from
  base in exactly one thing (rule #8).
