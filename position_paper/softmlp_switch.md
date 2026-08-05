# soft_mlp with a true → inferred task-id switch

THESIS-PLAN direction B, own variant. Split MNIST class-IL, Adam at the val-tuned point, 3 seeds,
every delta against the RNG-matched dead gate.

**Verdict — REJECT, and unusually informatively.** The switch does *exactly* what it was designed to
do, monotonically and measurably: it trades gate differentiation for tolerance to misrouting, and it
bends the `pred ≈ oracle × infer` law that has capped every task-conditioned mechanism in this
project. But **the optimum of that trade is at zero differentiation, which is the dead gate.** The
mechanism's best move is to turn itself off.

---

## The decomposition, pre-registered and confirmed

Predicted before running: `oracle` should FALL (rows get mixed gradients), `infer` should be
UNCHANGED (the selector's training is untouched), `pred` was the open question. er-own, 3 seeds:

| switch | soft (oracle-free) | d-dead | oracle | infer | o×i | soft − o×i |
|---|---|---|---|---|---|---|
| `true` (pt6) | 0.8923 | −0.0070 | **0.9828** | 0.8882 | 0.8729 | +0.019 |
| `last` (requested) | 0.8929 | −0.0064 ~ | 0.9441 | 0.8882 | 0.8386 | +0.054 |
| `half` | 0.8983 | −0.0010 ~ | 0.9259 | 0.8882 | 0.8224 | +0.076 |
| `always` | 0.8983 | −0.0010 ~ | **0.9179** | 0.8882 | 0.8152 | +0.083 |

Every prediction holds, and monotonically in how early the switch happens:

- **`oracle` falls** 0.9828 → 0.9179 as the switch moves earlier. Mixed gradients, less
  differentiation, exactly as pt6-followup-D2 diagnosed.
- **`infer` is constant** at 0.8882 to four decimals across all four variants — the selector really
  is untouched, which is what makes the other two columns interpretable.
- **`soft` rises**, 0.8923 → 0.8983.
- **`soft − o×i` grows** +0.019 → +0.083. This is the interesting one: the routing law is being
  broken, progressively, in the intended direction. The gate genuinely becomes tolerant of
  misrouting rather than either factor improving.

## Why it is still a reject

`d-dead` is negative everywhere, and it approaches zero exactly as the gate approaches no gate. Read
the two columns together: `soft` improves by +0.006 across the sweep while `oracle` gives up 0.065,
and the endpoint of the improvement (`always`, d-dead −0.0010) is statistically the parity gate. The
switch is not adding robustness *on top of* a useful gate — it is removing a gate that was costing
0.007, and it stops helping precisely when there is nothing left to remove.

So the honest statement is: **the trade is real and measurable, and it is not worth making.** At this
tuned Adam operating point a parity gate beats a trained one, so there is no differentiation worth
being tolerant about. This is a sharper version of pt6-followup-D2's finding (which measured only the
`always` extreme and read it as a loss); the intermediate points show it is a smooth trade with its
optimum at the boundary.

---

## The `bufown` arm is degenerate — and is NOT pt6's buf-own

Every `bufown` cell returns 0.198 (chance) with `oracle` also 0.1990, while `infer` is a healthy
0.8882. The selector is fine; the backbone is gone. **This arm is weaker than pt6's `buf-own` and the
difference is deliberate to state rather than paper over:** pt6 trained its gate on a modulator-only
replay META-loss, so replay reached the gate even though the backbone stayed naive (reaching 0.856
oracle-free). Here the gate is trained by the MAIN loss on the current task only, so nothing in the
run has a retention signal and the result is plain naive class-IL. A per-neuron gain cannot restore
features the backbone has already overwritten — which is why `oracle` collapses too.

That is a limitation of this study's arm, not a refutation of pt6. Reproducing pt6's buf-own would
need the meta-replay loop, and the switch question would then be worth re-asking there.

---

## Limits

3 seeds, one operating point, class-IL only. The er-own deltas span 0.006 against a noise floor of
0.007, so the *ordering* is trustworthy (it is monotone in all four columns and `infer` is pinned)
while any individual pair is not. `PT6_BAND` is a noise band, not a bit-exact anchor: `soft_mlp`
gates through `P[tids]`, whose backward is an atomic scatter-add and is nondeterministic on MPS
(~0.002–0.003 of final accuracy); `SoftMLPSelector(deterministic=True)` would make it reproducible at
the cost of one `(B, T)` matmul. Regimes not run — and note `rfree` would be doubly degenerate here,
since pt6-followup-B already showed a bufferless selector collapses to chance inference.
