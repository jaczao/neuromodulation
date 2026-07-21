# pt7 VARIANTS — standard regime, new drivers, NE multidim, standardization + mean-mode ablations

Follow-up to `pt7_neuromodulators.md` (user-requested). `results/pt7_variants.py` (+ `.log`,
`pt7_variants_results.tsv` ledger, `pt7_variants_make_table.py`). 152 cells, class-IL Split MNIST unless
STANDARD; gain (h0,h1,out) neuron; seed 42, lr 1e-3, ep 5, buffer 1000, 1 seed. Baselines (from
`pt7_results.tsv`): naive 0.6287/0.3900, er 0.7234/0.8946 (sgd/adam).

## A. STANDARD regime (full MNIST, single task, 10-way CE): the gate does NOT hurt
all4-gate vs vanilla MLP, matched budget (5 ep, lr 1e-3, UNtuned — rule #2 still needs a separate standard
sweep for a reportable number): sgd 0.8849 vs 0.8802 (+0.0047), adam 0.9804 vs 0.9766 (+0.0038). The
neuromodulation gate preserves (marginally improves) standard accuracy — project goal #2 ✓. (sgd numbers are
low because 5 ep of SGD@1e-3 under-trains, not a gate effect; adam is the meaningful row.)

## B. New head drivers ≈ baselines (as pt7)
`DA_fast`=(loss−ema_fast)/ema_fast and `ACh_vol_ps`=|loss−ema_fast| (PER-SAMPLE) are stable and ≈ ER/naive
everywhere (er-own: DA_fast 0.722/0.882, ACh_vol_ps 0.725/0.891). `ACh_ema`=ema(entropy) and
`5HT_ema`=ema_slow(−loss) are TONIC SCALARS → see the standardization ablation (they only work std-OFF).

## C/D. STANDARDIZATION ablation (the clean mechanistic result)
- **Tonic scalar drivers COLLAPSE under SGD *with* standardization, are fine *without* it.** `ACh_ema`,
  `5HT_ema` (and the pt7 `ACh_vol`, `NE_rise`): std-ON er-sgd → **0.098** (`nan` |g|) because standardizing a
  near-constant divides by ~0 and blows the gate up; std-OFF er-sgd → ≈ ER (ACh_ema 0.759, 5HT_ema 0.723).
- **`NE_rise` std-OFF is INERT** (|g|=0.000, pred = baselines): without the divide-by-~0, the head just
  learns a ≈0 gate. So standardization is exactly what turns "inert" into "catastrophic" for a tonic driver
  — a signal with no per-sample content cannot drive a per-sample gate either way.
- **Per-sample / multidim** drivers: std-ON ≈ std-OFF (no collapse either way). **Rule: standardize
  per-sample drivers; never standardize a tonic/scalar one.**

## E. NE double-forward / multidim novelty — the one notable effect (still a NON-WIN)
Head-free novelty drivers (computed directly, gating ALL layers): `emb_all` (scalar ‖h1−mean_h1‖, double
forward), `vec_h1` (400-d h1 diff), `vec_h1proj` (h1→32), `vec_x` (784-d input diff, pre-forward),
`vecproj` (input→32).

**Under SGD + er-own they lift ER-sgd 0.723 → 0.79–0.86 (+0.07..+0.14):** vec_x 0.864 (+0.140), vec_h1
0.863 (+0.139), emb_all(std-OFF) 0.845 (+0.122), NE_emb(std-OFF, out-only) 0.832 (+0.108), vec_h1proj 0.818
(+0.095), vecproj 0.795 (+0.071). **Three caveats make it a non-win:**
1. **Below the best baseline.** The ceiling (~0.86) is under Adam-ER (0.895), and **under Adam the same
   gates add nothing** (all −0.01..−0.06). It is the pt5/pt6 SGD-underfitting-closure pattern: SGD+ER leaves
   the head marginal in 5 ep and a rich gate closes the gap; Adam already gets there ungated.
2. **Capacity, not novelty.** INPUT novelty (`vec_x`) helps as much as EMBEDDING novelty (`vec_h1`), and
   more dims → more boost (784/400 ≈ +0.14 > 32-proj ≈ +0.07). So the driver is providing extra per-sample
   gate capacity SGD+replay exploits, not novelty semantics.
3. **Without replay it hurts** — nobuf/buf-own mostly below naive; multidim collapses under Adam (vec_h1
   nobuf-adam 0.19). (One high-variance buf-own outlier: vecproj buf-own-adam 0.619, still ≪ ER.)
`emb_all` (head-free) ≈ `NE_emb` (head-based) → for a NOVELTY driver the head is a design choice (single-pass
prediction) not a necessity; only loss/entropy drivers genuinely need a head at eval.

## F. Cumulative mean vs EMA (disambiguates capacity-vs-novelty)
- **The SGD+ER boost is NOT an EMA artifact** — it survives the cumulative (true) mean: vec_h1 sgd
  0.863→0.861, vec_x sgd 0.859→0.866.
- **Cumulative mean is fragile precisely for the DRIFTING h1**: `vec_h1` cumulative + Adam **collapses to
  0.101** (chance) — the cumulative `mean_h1` averages over early undertrained representations and lags the
  drift, so `h1−mean_h1` grows biased and destabilises Adam. For the STATIONARY input, `vec_x` cumulative ≈
  EMA (0.836→0.811). **So EMA is required for embedding novelty (must track the drifting representation);
  input novelty is mean-mode-agnostic** — reinforcing the capacity (not novelty) reading.

## G. Split optimizer (main net = Adam, neuromodulator gate P + head = SGD), standardized
Isolates whether the SGD+ER boost came from the MAIN net being under-fit or from the gate being trained with
SGD. `vecproj` (headless) and `NE_emb` (head-based, out-only), er-own/nobuf/buf-own, main=Adam. **The boost
DISAPPEARS**: er-own vecproj 0.8914, NE_emb 0.8886 — both ≈ ER-adam (0.895), NOT the +0.07..+0.11 seen under
full-SGD (vecproj 0.795 / NE_emb 0.832). Three-way: full-SGD 0.795/0.832 (+boost) vs full-Adam 0.883/0.878
vs **main-Adam/gate-SGD 0.891/0.889** (≈ ER). So the boost was **entirely a MAIN-net-optimizer artifact** —
SGD under-fits the head in 5 ep and the gate closes that gap; once the main net is Adam it already reaches
~0.89 (no gap), and the gate (Adam OR SGD) adds nothing (|g|~0.002–0.04). The gate optimizer is irrelevant;
this confirms "capacity closing SGD's under-fit, not a class-IL lever" as tightly as possible.
nobuf/buf-own ≈ naive-adam (gate inert without replay).

## Verdict
Nothing beats the best baseline (Adam-ER 0.895). The standard-regime gate is harmless; the tonic drivers are
degenerate (and standardization is what makes them catastrophic vs merely inert); the NE-novelty/multidim
gates give a real but explained SGD-only boost (capacity closing SGD's under-fit, not a class-IL lever, and
harmful without replay). The `vec_h1`/`vec_x` SGD+ER cells (+0.14) are the strongest pt7 numbers and would
need **3 seeds** before being trusted. Reportable class-IL headline across the project is unchanged (pt6
oracle-free selector / pt5 disjoint gain+ER). 1 seed; buf-own high-variance.
