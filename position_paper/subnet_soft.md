# Study 5 — subnets allocated in proportion to the soft_mlp posterior

THESIS-PLAN direction B, own variant. Split MNIST class-IL, ER, Adam at the val-tuned point
(lr 3e-4 / ep 5 / buffer 1000), er-own, 3 seeds, delta against the RNG-matched dead gate.

**Verdict — REJECT, and it settles what pt5 iter-1's win actually was.** The mechanism reproduces
iter-1 faithfully under the oracle (**0.9939** vs its 0.9901), and that number survives every
variant. But **oracle-free it lands at 0.87–0.89, below plain ER (~0.902)**, every cell is below its
own dead control, and `soft ≈ oracle × infer` holds to within ±0.007 in all five rows. Proportional
allocation does **not** buy tolerance to misrouting.

---

## The mechanism

`γ(x)_j = p_{owner(j)}(x)` over a **fixed disjoint** partition of the 800 hidden units (160 per
subnet, from the same `build_disjoint_proj` primitive iter-1 used). A posterior of
(0.4, 0.4, 0.1, 0, 0.1) scales subnets 0 and 1 by 0.4, subnet 2 by 0.1, and switches 3–4 off —
verified directly (γ takes exactly the values {0.4, 0.1, 0.0}). Under a one-hot posterior this
reduces to iter-1's {0, 1} gate identically, which is what makes `true` a reproduction and every
other row a measured departure.

Head left ungated: pt5 established that a label-aligned output gate *is* task-IL masking, which would
reintroduce the confound this study exists to remove. Selector lr reused from `softmlp_switch` (1e-4).

## Results

| switch | soft (oracle-free) | d-dead | seeds | oracle | hard | infer | o×i | γ̄ |
|---|---|---|---|---|---|---|---|---|
| `true` (= iter-1) | **0.8898** | −0.0095 | 0/3 | **0.9939** | 0.8831 | 0.8882 | 0.8828 | 0.200 |
| `last` | 0.8761 | −0.0233 | 0/3 | 0.9899 | 0.8819 | 0.8882 | 0.8792 | 0.200 |
| `last_half` | 0.8740 | −0.0253 | 0/3 | 0.9916 | 0.8825 | 0.8882 | 0.8808 | 0.200 |
| `last_plateau` | 0.8710 | −0.0284 | 0/3 | 0.9909 | 0.8823 | 0.8882 | 0.8802 | 0.200 |
| `always` | 0.8776 | −0.0217 | 0/3 | 0.9893 | 0.8833 | 0.8882 | 0.8787 | 0.200 |

### 1. The routing law holds tightly — proportional allocation buys nothing

`soft − oracle×infer` = +0.007, −0.003, −0.007, −0.009, −0.001 across the five rows. That is the
tightest confirmation of `pred ≈ oracle × infer` in the project, and it was the study's open
question: a misrouted sample *does* still put mass on the correct subnet, so the degradation ought to
be graceful — and it simply is not. Contrast study 3, where a learned gain table's `soft − o×i` grew
to +0.083 once it was trained under inferred ids. **A fixed disjoint partition cannot become tolerant
of misrouting, because there is nothing about it to soften.**

`hard` ≈ `soft` everywhere (0.882–0.883 vs 0.871–0.890), replicating pt6's finding that for a
confident learned selector the blend and the argmax coincide.

### 2. The schedule backfires, for a reason specific to a FIXED partition

In study 3 the softer schedules *improved* the oracle-free number, because training the gate table
under inferred ids made its rows less differentiated and therefore more forgiving. Here the opposite
happens: `true` is the **best** variant and every softer schedule is worse (−0.014 to −0.019 relative
to it).

The asymmetry is the point. Study 3's gate was **learned**, so exposure to inferred ids changed what
it learned. Here the partition is **fixed** — training under the posterior does not reallocate a
single unit, it only scales the backbone's updates by a noisy factor. There is no "learning to use
multiple subnets" available, because subnet membership is not a learnable quantity in this design.
The schedule adds gradient noise and nothing else. Confirmed by the `oracle` column, which barely
moves (0.9939 → 0.9893) where study 3's fell 0.9828 → 0.9179.

**If the intent is for tasks to learn to share subnets, the partition itself has to be learnable** —
that is a different mechanism (pt5's `learned` projection, which was measured and lost to `disjoint`).

### 3. What this settles about pt5 iter-1

iter-1 is the project's one large win over replay, always reported with an oracle caveat. This
quantifies the caveat: the same mechanism, same partition, same arm, with the task id replaced by a
replay-trained selector at 0.8882 task accuracy, gives **0.8898 against plain ER's ~0.902**. The
oracle was worth essentially all of it, and the oracle-free ceiling is `oracle × infer` =
0.9939 × 0.8882 = **0.883** — which the mechanism reaches, and cannot exceed.

### 4. A diagnostic that turned out uninformative

`γ̄ = 0.200` in every row, including the soft ones. That is forced: each unit belongs to exactly one
subnet and the posterior sums to 1, so the mean of `p_{owner(j)}` over units is `1/T` regardless of
how the posterior is distributed. It therefore cannot distinguish confident from diffuse routing —
per-subnet mass or posterior entropy would be the informative statistic. Recorded so the column is
not misread as "the gate did not change".

---

## Limits

3 seeds, class-IL, er-own, one operating point. One fixed partition seed per run (the partition is
re-drawn per seed, so partition variance is folded into seed variance rather than isolated). The
selector lr was reused rather than re-swept. The `bufown` arm was not run — study 3 showed it is
degenerate under this harness. Regimes not run: the mechanism is below its control at buffer 1000,
so a budget sweep would be measuring how a losing arm loses.
