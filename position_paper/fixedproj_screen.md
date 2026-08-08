# Fixed-random projection screen — all Phase-B studies

User-requested, no tuning. Every study's LEARNED projection is replaced by a FROZEN RANDOM one and
re-run, to ask whether the learning was ever doing work. σ = 0.1 throughout (0.4 for study 5's
uniform), matching `pt7_signalnet.run_all4_fixedproj`; unswept, and it is a real knob
(`fixedproj_scale/` found accuracy monotone-decreasing in gate magnitude).

Encoded as new values in existing key columns — coef `randproj` (study 1), driver `<name>_fixp`
(study 2), gate `fixp` (studies 3, 5) — so no ledger changed schema. Study 4 has no projection and
no learned component at all, so it is N/A rather than skipped.

**Headline: freezing the projection changes almost nothing — except in the two places where it
changes everything, and both are informative.**

---

## Study 1 — frozen random selector (1 seed, d vs each formulation's parity)

| formulation | `randproj` | `soft` (trained selector) |
|---|---|---|
| v1 `group`, normal / budget | **+0.0180 / +0.0433** | −0.0024 / +0.0069 |
| v2 `logit`, normal / budget | **−0.0261 / −0.0286** | **+0.0232 / +0.0575** |
| v3 `sample`, normal / budget | **+0.0094 / +0.0195** | −0.0134 / −0.0337 |

Absolute: v1 0.9190 / 0.8149, v2 0.8760 / 0.7403, v3 0.9088 / 0.7979.

**A double dissociation.** In the two WEIGHTING formulations a frozen random projection *beats the
trained selector* — and in v1 it beats every previous cell, `learned` (+0.0107) and `uniform`
(+0.0269 at budget) included. In the LOGIT formulation it is the only arm that decisively LOSES,
while the trained selector wins.

This sharpens both earlier readings. v1's benefit tracked `c_err` (departure from the true batch
composition), and a random projection is simply a large arbitrary departure — so v1 is not merely
content-free but **content-AVERSE**: the better the estimate of the composition, the worse the
result. v2 is confirmed as the one formulation where the signal is load-bearing, now from two
directions: `learned` (content-free, constant) fails, `randproj` (content-free, varying) fails, and
only the replay-trained posterior wins.

---

## Study 2 — learned `P` vs frozen random `P`, boundary / class-IL / er-own, 3 seeds

| driver | neuron learned | neuron **fixed** | synapse learned | synapse **fixed** |
|---|---|---|---|---|
| `vecproj` | 0.9228 (+0.0219) | **0.9232 (+0.0223)** | 0.9244 (+0.0235) | 0.9198 (+0.0189) |
| `vec_x_ns` | — | **0.9232 (+0.0223)** | — | **0.9224 (+0.0215)** |
| `vecproj_ns` | — | **0.9232 (+0.0224)** | — | **0.9244 (+0.0235)** |
| `ach_act` | 0.9215 (+0.0206) | 0.9155 (+0.0146) | 0.9209 (+0.0200) | 0.9091 (+0.0082) |
| `nerisez_act` | 0.9184 (+0.0176) | 0.8766 (−0.0243) | 0.9185 (+0.0177) | 0.8537 (−0.0472) |
| `taskid` | 0.9241 (+0.0232) | 0.8688 (−0.0321) | 0.9258 (+0.0249) | 0.6946 (−0.2063) |
| `const` | 0.9209 (+0.0204) | **0.0927 (−0.8082)** | 0.9201 (+0.0197) | **0.0927** |
| `vec_x` (raw vector) | — | **0.0927** | — | skipped (374M params) |

**For a well-conditioned PER-SAMPLE driver the learning is unnecessary**: `vecproj` and both `_ns`
forms reach +0.022 with a frozen random `P`, matching the learned version to within noise.

**But `const` collapses to chance, and that reverses part of an earlier conclusion.** `const` was the
control that showed the boundary effect is ~85% content-free. It only works *because its projection
is learned*: with `m(x) ≡ 1` and a frozen `P`, every task applies the SAME random per-parameter decay,
compounding across all five boundaries with nothing able to correct it. `taskid` degrades for the
milder version of the same reason (one row per task, applied once each, so less compounding but no
correction either). So the honest statement is now:

> The effect needs EITHER a learned projection OR a per-sample driver — not neither. A content-free
> driver survives only if its projection can adapt; a frozen projection survives only if the driver
> varies per sample.

The two-combo restriction on the novelty family also pays off directly: the RAW 784-d vector goes to
chance under a frozen `P`, while both standardised-NORM forms are the best fixed-`P` cells in the
table.

---

## Studies 3 and 5 — freezing changes nothing

**Study 3** (gate table learned vs frozen random), er-own, 3 seeds, dead = 0.8993:

| switch | learned | fixp | learned oracle | fixp oracle |
|---|---|---|---|---|
| `true` | 0.8923 | 0.8913 | 0.9849 | 0.9529 |
| `last` | 0.8929 | 0.8902 | 0.9424 | 0.9159 |
| `half` | 0.8983 | 0.8974 | 0.9236 | 0.9104 |
| `always` | 0.8983 | 0.8985 | 0.9198 | 0.9074 |
| `last_half` | 0.8792 | 0.8807 | 0.9709 | 0.9363 |
| `last_plateau` | 0.8848 | 0.8855 | 0.9627 | 0.9286 |

`fixp` tracks learned within 0.003 at every switch, and both sit below the dead control. The `oracle`
column is uniformly lower for `fixp` (a random table differentiates less than a trained one), which
confirms the arm is doing what it should — it just does not matter.

**Study 5** (disjoint partition vs frozen random DENSE allocation, matched mean gate 0.2), 3 seeds,
dead = 0.8993:

| switch | disjoint | random dense | disjoint oracle | random oracle |
|---|---|---|---|---|
| `true` | 0.8898 | 0.8847 | 0.9929 | 0.9939 |
| `last` | 0.8761 | 0.8814 | 0.9886 | 0.9896 |
| `last_half` | 0.8740 | 0.8791 | 0.9909 | 0.9922 |
| `last_plateau` | 0.8710 | 0.8782 | 0.9903 | 0.9922 |
| `always` | 0.8776 | 0.8855 | 0.9880 | 0.9832 |

Within 0.008 everywhere, both below dead. Notably the random dense allocation reaches the SAME oracle
(0.9939 vs 0.9929), so pt5 iter-1's near-perfect oracle number does not require a disjoint partition
either — any per-task gate with the same mean mass gets there, and the oracle is still doing all the
work.

---

## Limits

Study 1 is 1 seed (its deltas are 0.018–0.043, well outside the 0.007 floor, but its parity rows are
n=3 and the comparison is unpaired). One σ, unswept. Studies 2/3/5 are 3 seeds. `vec_x` × synapse
remains unrunnable frozen or learned (374M-param projection).

**Infrastructure note.** `SKIP` keyed on the bare driver name, so `vec_x_fixp` × synapse was not
caught and the worker was killed allocating that projection — the runner correctly blocked the merge
rather than writing a partial ledger. Fixed with `_base_driver()`, which strips the suffix before the
skip check. Also: for a `_fixp` driver the `nlr` key column carries σ rather than a learning rate
(there is nothing to rate), so σ = 0 is an all-zero `P` and remains exactly the dead control — the
same column-overloading convention as the `fixed` schedule, and the same trap.
