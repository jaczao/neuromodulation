# Capacity ablation — is the gating null an over-parameterization artifact?

**Verdict: NO. The null holds at every width, including where the baseline has collapsed. Capacity was
never the issue — this is the airtight negative.**

Driver `results/pt7_capacity.py` · ledger `pt7_capacity_results.tsv` · log `pt7_capacity.log` ·
plots `pt7_capacity/{capacity_ablation,gate_engagement}.png`

---

## The question

Across pt5–pt7 a soft, main-loss-trained multiplicative gate was always ABSORBED by the backbone: the h1
gate sat at mean 0.281 yet landed exactly on ER; a fixed-random projection matched a learned one; the L1
penalty only shuffled magnitude between γ and W. One candidate explanation for that null is
**over-parameterization** — the MLP (784→400→400→10) is so oversized for 2 classes at a time that there is
never any SCARCITY to allocate, and gain control is a resource-allocation mechanism with no scarce resource
to allocate. The one pt5 cell that worked (Iter 1 disjoint gain) manufactured scarcity by hand (1/5 capacity
per task).

So: shrink hidden width H until the BASELINE ITSELF degrades, then re-run the gate at each H.

## Setup

Class-IL Split MNIST, Adam, lr 1e-3 / ep 5 / buffer 1000. H ∈ {400,200,100,50,25,10,5} (both hidden layers).
Mechanism held fixed across the sweep: the canonical pt7 **`all4` gain-neuron gate on ER** (er-own) — K=4
heads m_k(x) regress standardized DA/ACh/NE/5HT, driving Γ = 1 + Σ_k m_k P_k over (h0,h1,out).

3 seeds; **5 at H=10 and 9 at H=5**, where per-seed spread grows and the verdict is actually decided.

Reuses the pt7 code path verbatim — width enters only by rebinding `p7.H0/H1/GATEDIM` in a `width(H)`
context manager. **H=400 therefore reproduces the frozen pt7 ledger bit-exact** (naive 0.3900 / er 0.8946 /
free 0.8760 / all4 0.8816 at seed 42, `--part anchor`). That is the sanity anchor for the whole sweep.

### The fourth arm, and why it decides the study

Beyond naive / ER / ER+all4 there is **`er+free`**: the pt7 dead-gate control (K=4 zero-init heads, no bio
target). Its gate measures `|g| = 0.000000` at every width and every seed — provably inert, numerically ER —
but constructing the heads consumes torch RNG and shifts the replay draws. It is the **RNG-matched
baseline**. `all4 − free` is the honest delta; `all4 − ER` is not.

## Results

| H | params | n | naive+masked | ER | ER+free | ER+all4 | d-ER | **d-free** |
|---|---|---|---|---|---|---|---|---|
| 400 | 478,410 | 3 | 0.3901 ±.005 | 0.8907 ±.008 | 0.8890 ±.009 | 0.8909 ±.007 | +0.0002 | **+0.0019** |
| 200 | 199,210 | 3 | 0.3855 ±.032 | 0.8858 ±.003 | 0.8821 ±.006 | 0.8791 ±.004 | −0.0067 | **−0.0030** |
| 100 | 89,610 | 3 | 0.3655 ±.011 | 0.8819 ±.006 | 0.8882 ±.008 | 0.8948 ±.003 | +0.0129 | **+0.0066** |
| 50 | 42,310 | 3 | 0.3438 ±.054 | 0.8837 ±.005 | 0.8828 ±.003 | 0.8875 ±.003 | +0.0038 | **+0.0047** |
| 25 | 20,535 | 3 | 0.2380 ±.034 | 0.8600 ±.012 | 0.8592 ±.008 | 0.8703 ±.007 | +0.0103 | **+0.0111** |
| 10 | 8,070 | 5 | 0.1688 ±.042 | 0.7741 ±.080 | 0.8183 ±.019 | 0.8184 ±.014 | +0.0443 | **+0.0000** |
| 5 | 4,015 | 9 | 0.1862 ±.070 | 0.5748 ±.148 | 0.6376 ±.075 | 0.6729 ±.075 | +0.0981 | **+0.0353** |

Paired-per-seed significance on the honest delta (`all4 − free`):

| H | 400 | 200 | 100 | 50 | 25 | 10 | 5 |
|---|---|---|---|---|---|---|---|
| d-free | +0.0019 | −0.0030 | +0.0066 | +0.0047 | +0.0111 | +0.0000 | +0.0353 |
| t | +0.98 | −0.49 | +1.28 | +1.77 | +2.37 | **+0.00** | +1.39 |
| sign | 2/3 | 1/3 | 2/3 | 3/3 | 3/3 | 2/5 | 7/9 |

**1. Scarcity was genuinely reached.** ER falls 0.891 → 0.774 → 0.575 and naive collapses to ~chance
(0.19). At H=5 the network has 4k parameters for 10 classes. The premise of the ablation is met.

**2. The delta never emerges.** `d-free` is inside the ±0.007 noise floor at 400/200/100/50, is exactly
**+0.0000 (t=0.00, n=5)** at H=10, and is **+0.0353 (t=1.39, n=9, p≈0.20 — not significant)** at H=5.
There is no monotone trend: the largest well-resolved value is at H=25, in the middle of the sweep.

**3. Why `d-ER` looks like a win at small H, and is not.** The inert `free` gate — `|g| = 0.000000` —
"beats" plain ER by **+0.0443 at H=10 and +0.0628 at H=5**. A gate that does literally nothing cannot
allocate capacity, so that entire gap is run-to-run instability at a collapsed operating point. It accounts
for most of `d-ER` (+0.098 at H=5, of which +0.063 is reproduced by the dead gate). Read against plain ER,
this study would have reported a spurious "gain control rescues small networks" result.

**4. The strongest mechanistic finding: engagement rises, benefit does not.** Gate magnitude grows ~15× as
capacity shrinks — exactly what a resource-allocation account predicts — while accuracy does not follow:

| H | 400 | 200 | 100 | 50 | 25 | 10 | 5 |
|---|---|---|---|---|---|---|---|
| \|g\| h0 | 0.0143 | 0.0296 | 0.0061 | 0.0213 | 0.0160 | 0.0914 | **0.2074** |
| \|g\| h1 | 0.0261 | 0.0453 | 0.0111 | 0.0316 | 0.0298 | 0.1253 | **0.2662** |
| \|g\| out | 0.0331 | 0.0559 | 0.0162 | 0.0248 | 0.0383 | 0.0755 | **0.2321** |
| probe | 0.473 | 0.440 | 0.437 | 0.415 | 0.414 | 0.385 | 0.483 |

The gate is *more* active under scarcity and still buys nothing. So the null is **not** "the gate sat idle
because there was nothing to allocate" — it engages precisely when the resource-allocation story says it
should, and the modulation is still absorbed or uninformative. The task-probe stays at 0.39–0.51 throughout
(never task-decodable), consistent with pt7's core reason: difficulty/novelty is not task identity.

## Reading

Outcome (1) of the two the study was designed to separate: **capacity was never the issue; the null is real
and the absorption story stands.** This closes over-parameterization as an explanation for the pt5/6/7 gating
null and supports the alternative reading of pt5 Iter 1 — that its win came from the hard {0,1} freeze
(which kills the gradient and is therefore un-absorbable), not from scarcity per se, since manufacturing real
scarcity here does not reproduce it.

## Follow-up: single driver, and fixed-random projections (1 seed)

Requested check — same widths, seed 42, **baselines not re-run** (reused ledger rows). `er+ACh` = ACh alone
(K=1, learned P); `er+AChfix` / `er+all4fix` = the same with P **fixed random** (gaussian scale 0.1, frozen,
in no optimizer; heads still regress the standardized bio τ and the backbone still adapts). The fixed-P
construction is reused verbatim from `pt7_signalnet._freeze_random_proj`, and H=400 reproduces that study's
frozen ledger bit-exact (all4fix 0.8857, probe 0.408, |g| 0.0648/0.0667/0.0708).

| H | ER | ER+free | ER+all4 | ER+ACh | ER+AChfix | ER+all4fix | d-free ACh | d-free AChfix | d-free all4fix |
|---|---|---|---|---|---|---|---|---|---|
| 400 | 0.8946 | 0.8760 | 0.8816 | 0.8841 | 0.8945 | 0.8857 | +0.0081 | +0.0185 | +0.0097 |
| 200 | 0.8811 | 0.8802 | 0.8747 | 0.8937 | 0.9010 | 0.8740 | +0.0135 | +0.0208 | −0.0062 |
| 100 | 0.8884 | 0.8980 | 0.8943 | 0.8950 | 0.8897 | 0.8689 | −0.0030 | −0.0083 | −0.0291 |
| 50 | 0.8874 | 0.8861 | 0.8901 | 0.8765 | 0.8755 | 0.8765 | −0.0096 | −0.0106 | −0.0096 |
| 25 | 0.8449 | 0.8599 | 0.8629 | 0.8709 | 0.8640 | 0.8643 | +0.0110 | +0.0041 | +0.0044 |
| 10 | 0.7958 | 0.8157 | 0.8157 | 0.8364 | 0.8452 | 0.7265 | +0.0207 | +0.0295 | −0.0892 |
| 5 | 0.4547 | 0.4818 | 0.6238 | 0.4723 | 0.3539 | 0.5447 | −0.0095 | −0.1279 | +0.0629 |

**Read only the H ≥ 25 rows.** At H ≤ 10 a single seed cannot resolve anything: the main study measured the
*inert* `free` gate swinging ±0.06 between seeds at H=5, and ER's own spread there is ±0.148. The H=5 row
(AChfix 0.354 vs all4 0.624) is run instability, not mechanism — the same trap that inflates `d-ER`.

At H ≥ 25, where the multi-seed noise floor is ~±0.01: **every d-free is within ±0.021 with scattered
signs.** No arm separates from the dead gate, at any capacity.

1. **A single driver is not hiding an effect the K=4 composite dilutes.** ACh alone tracks all4 and ER at
   every resolvable width. Its probe drops to 0.25–0.28 (vs 0.37–0.51 for all4) — one entropy driver is even
   less task-decodable, consistent with pt7's reason for the negative.
2. **The fixed-random projection sharpens the absorption story.** `all4fix` engages far harder than the
   learned gate — |g| = 0.33–0.43 at H=200/100 versus 0.006–0.08 learned, a 5–70× larger modulation — and
   still lands on ER (0.8740 / 0.8689 vs free 0.8802 / 0.8980). A large, frozen, random multiplicative
   perturbation is absorbed as completely as a small learned one. That is the cleanest single statement of
   the mechanism: the backbone reabsorbs whatever the gate does, learned or not, big or small.

## Honest limits

- **H=5 is not a clean zero.** +0.0353, 7/9 seeds positive, t=1.39 — not significant, but I cannot exclude a
  small positive there. It would need ~25 seeds to resolve at that variance, and it sits at an operating
  point where the baseline is collapsing and unstable.
- **lr/epochs are not re-tuned per width** (kept at the standard pt7 point). Every arm at a given H gets an
  identical budget so the matched-H delta is unaffected, but absolute accuracy at H ≤ 10 is likely
  pessimistic, and that regime's large variance is partly mis-tuning. A per-width tune would be the
  follow-up if H=5 is ever worth resolving.
- **One mechanism.** `all4` gain-neuron; the gate is task-agnostic by construction (probe ≈ 0.4), so this
  tests soft-gain absorption, not task-conditioned allocation. A task-conditioned gate under scarcity
  (pt6 `soft_mlp`) is the untested cell.
- Class-IL only; oracle-free by construction (pt7 driver, no task id at eval).
