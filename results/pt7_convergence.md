# pt7 follow-up — does neuromodulation (all4, Adam) help CONVERGENCE / EFFICIENCY?

User-requested. The pt7 headline already showed all4 ties the baseline at the FINAL accuracy plateau
(er-adam ≈ vanilla). This asks the orthogonal question: even when the endpoint ties, is the **learning curve**
faster/steeper — does the gate reach a given accuracy in fewer epochs? all4 = the canonical pt7 gate (K=4 heads
regress the standardized bio signals DA/ACh/NE/5HT, trained WITH REPLAY, gain-NEURON over (h0,h1,out)), Adam,
seed 42, lr 1e-3, buffer 1000, **1 seed**. Self-contained: `results/pt7_convergence.py`, log/ledger alongside.

**Answer: NO — all4 improves neither convergence speed nor epoch-efficiency in either regime. It changes
neither the plateau nor the path to it; if anything it costs a small early warm-up.** All deltas below are
within the ±0.007–0.016 MPS 1-seed noise floor (CLAUDE.md pt6-followups).

---

## Standard (full MNIST, single-task, Adam) — per-epoch test-accuracy learning curve (E=15)

| driver  | e1 | e2 | e3 | e5 | e10 | e15 | →0.97 | →0.98 |
|---------|----|----|----|----|-----|-----|:-----:|:-----:|
| vanilla | 0.9678 | 0.9741 | 0.9728 | 0.9748 | 0.9794 | 0.9816 | ep2 | ep11 |
| all4    | 0.9549 | 0.9699 | 0.9719 | 0.9763 | 0.9774 | 0.9844 | ep3 | ep9  |

**all4 is SLOWER early, then ties.** e1 0.955 vs 0.968 and 0.97-threshold at ep3 vs ep2 — the zero-init
heads/gate have to warm up (heads start at 0 → gate ≈ parity → the first epoch is just vanilla plus untrained
extra params, a small drag). By ep10–15 the curves converge (~0.98; the ep9-vs-ep11 0.98 crossing and the
0.9844-vs-0.9816 endpoint are inside seed noise). No convergence benefit; a small early cost.

## CL class-IL (Split MNIST, Adam, er-own) — EPOCHS-PER-TASK efficiency sweep

| ep/task | 1 | 2 | 3 | 5 | 8 |
|---------|---|---|---|---|---|
| er      | **0.9058** | 0.8980 | 0.8871 | 0.8892 | 0.8872 |
| er+all4 | 0.8889 | **0.9061** | 0.9043 | 0.8829 | 0.8815 |

**No efficiency gain — ER is if anything more epoch-efficient.** Both methods PEAK at a small budget and DECAY
with more epochs/task (more passes over the current task before moving on ⇒ more catastrophic overwriting
before the next task; replay only partly offsets it — this is the dominant effect and it is NOT neuromod).
Best-over-budget ties (~0.906), but **ER hits its peak at ep=1 while all4 needs ep=2** — the gate does not shift
the efficiency frontier earlier; it reaches the same peak at a *larger* budget. Per-budget deltas
(−0.017/+0.008/+0.017/−0.006/−0.006) alternate sign ⇒ 1-seed noise, no trend.

## CL class-IL trajectory (er-own, ep=5) — avg class-IL acc at each end-of-task

| checkpoint | t0 | t1 | t2 | t3 | t4 (final) |
|------------|----|----|----|----|-----|
| er      | 0.200 | 0.396 | 0.576 | 0.750 | 0.8892 |
| er+all4 | 0.200 | 0.396 | 0.578 | 0.751 | 0.8829 |

**Identical trajectories** (Δ ≤ 0.006 at every checkpoint). The average accuracy climbs the same way through
the sequence (each new task's 2 classes add ~0.18–0.19 to the 10-way average); the gate does not accelerate
the accumulation. Sanity: cl-traj finals reproduce the cl-sweep ep=5 cells (er 0.8892, all4 0.8829).

---

## Verdict
Neuromodulation (all4, Adam) is a convergence/efficiency **null** in both regimes, mirroring its final-accuracy
null: it does not reach a given accuracy in fewer epochs (standard: small early warm-up lag then tie; CL:
identical trajectory, same-or-worse epoch-efficiency). The gate changes neither the plateau nor the rate of
approach to it — replay (CL) / plain Adam (standard) already sets both. Consistent with the whole pt7
controlled-negative: difficulty/novelty modulation is not a lever, on accuracy OR on speed. 1 seed; deltas
within MPS noise. (For standard, goal #2's "preserve accuracy" also holds here — all4 does not degrade the
curve, it just doesn't speed it.)
