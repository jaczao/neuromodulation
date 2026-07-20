# pt6 follow-ups — seven isolated probes (A–G)

Companion to `pt6_followups.py` / `pt6_followups2.py` (+ `.log`s). Each probe is **isolated** (its own
small cell set, no cross-product). soft_mlp unless stated; gain-neuron on (h0,h1,out); class-IL Split
MNIST, seed 42, lr 1e-3, ep 5, buffer 1000, **1 seed**.
Refs: `naive-sgd 0.629`, `er-sgd 0.723`, `er-adam 0.895`; standard soft_mlp soft-blend
`er-own/adam 0.886`, `er-own/sgd 0.856`, `buf-own/sgd 0.856`.

> **Noise floor.** Identical configs re-run differ by **~0.007–0.016** (MPS nondeterminism; e.g.
> parity λ=10 gave oracle 0.962 then 0.978). **At 1 seed treat anything under ~0.02 as null.**

---

## A. mean_image/mlp/centered — finer soft-nearest(τ)
```
              @.003  @.01  @.03  @.05  @.1   hard
er-own sgd    .757  .767  .777  .765  .692  .751
er-own adam   .751  .750  .739  .710  .617  .752
buf-own sgd   .519  .522  .534  .543  .519  .517
```
τ≈0.03–0.05 is a **genuine interior peak**, and τ→0 converges to hard-nearest exactly as theory says
(0.757→0.751 = hard) — which also validates the implementation. Soft beats hard **only under SGD**
(+0.026); under Adam it just converges to hard. **Nothing below τ=0.03 to gain.**

## B. soft_mlp standalone variants
```
                        oracle  soft   infer
no-buffer sgd            .933   .463   .198
no-buffer adam           .432   .324   .198
buf-cur (wrong-task) sgd .720   .690   .865
ref buf-own/own sgd      .939   .856   .865
```
**Replay is what makes the selector work:** with no buffer, `infer` collapses to **0.198 ≈ chance
(1/5)** and the oracle-free number dies (0.46) even though the oracle stays 0.93. Separately,
**own-task gating matters**: `buf-cur` costs −0.17 soft / −0.22 oracle.

## C. hard-mlp vs soft-mlp (matched training)
```
                soft   hard    Δ
buf-own sgd     .856   .839   +.017 soft
buf-own adam    .661   .675   +.014 hard
er-own sgd      .856   .863   +.008 hard
er-own adam     .886   .881   +.005 soft
```
**A wash** (±0.017, no consistent direction, all inside noise). A well-trained selector is
*confident*, so its softmax ≈ one-hot and the blend ≈ the argmax. **Softness only pays when the
posterior is diffuse** — i.e. the prototype case (A), not a learned selector.

## D. selector trained on the main net's OUT LAYER (pseudo-labels)
```
                    soft   infer   (vs true-label)
er-own sgd  inf<-out .854   .882    (.856 / .884)
er-own adam inf<-out .864   .861    (.886 / .884)
```
**Pseudo-labels ≈ true task ids.** The selector does not need the true task label — the main net's own
predictions suffice. (Caveat: this does *not* make the method label-free; the gate table and backbone
are still trained with true task/class labels. It only removes the true task id from the *selector's*
signal.)

## D2. DRIVER AT TRAIN: true task id vs the inference net's SOFT posterior
```
                soft            oracle
er-own sgd     .856 → .850     .948 → .936
er-own adam    .886 → .893     .989 → .957
buf-own sgd    .856 → .753     .939 → .799
```
**Removing the train/eval mismatch does NOT help.** er-own deltas are inside the noise floor (null);
buf-own clearly **hurts** (−0.103); and the **oracle drops in every cell** (up to −0.14). Training on
`P[t_true]` gives each row a **clean, unmixed gradient** from its own task — the same "independent
rows" property that makes one-hot work. Training on a blend smears every sample's gradient across all
rows, so the rows differentiate less. buf-own suffers most because its standalone gate depends
entirely on that per-task meta signal. **The mismatch was never the problem.**

## E. SPARSITY sweep  λ·mean|1+P|  (gates → off), er-own/adam
```
  λ      soft    mean|P|
  0     .886     .059
  0.1   .887     .97
  1     .882    1.00
  10    .879    1.00
```
**Ineffective.** It moves the gate enormously (mean|P| 0.06→1.0) while accuracy stays flat — almost
certainly a **scale degeneracy**: the jointly-trained backbone simply rescales its weights to absorb
the gate's scale, so an L1 on γ shifts magnitude between γ and W without changing the function.
(Mean-penalties also only start to bite near λ≈1 under Adam, per pt5.)

## F/G. PARITY sweep λ·mean|P| + the PER-LAYER gate study
Where the modulation actually lives (mean |P| per gated layer):
```
                h0     h1    out
er-own sgd     .001   .002   .107
er-own adam    .033   .083   .133
buf-own sgd    .516   .545   .251
```
Parity sweep (er-own/adam):
```
  λ     oracle   soft   |P|out
  0     .991    .888    .134
  0.1   .989    .886    .109
  1     .980    .884    .064
  10    .978    .887    .023
```
1. **CORRECTION to an earlier reading.** `mean|P| ≈ 0.003` for er-own/sgd looked like "the gate is ≈
   parity", but that averages 800 hidden entries and **hides the 10-wide out gate**: the hidden gates
   *are* ≈ parity (0.001–0.002) while **out is 50–100× larger (0.107)**. So the er-own gate is
   essentially a **pure per-task LOGIT adjustment** — precisely the class-IL bottleneck — and that
   explains the er-own/sgd gap (0.856 vs er-sgd 0.723) that `mean|P|` could not.
2. **The two arms use different layers.** With ER the gate lives in the **out** layer (replay handles
   the features, the gate only recalibrates logits); **standalone (buf-own) it lives in the hidden
   layers** (0.52/0.55 vs out 0.25), because with no replay refreshing the backbone the gate must
   modulate features itself.
3. **The out gate is load-bearing.** The parity penalty crushes hidden ~100× (0.102→0.001) but the out
   gate resists, only ~6× (0.134→0.023) — the last thing the loss gives up. Even fully crushed the
   oracle loses only 0.012 and the oracle-free number is flat, i.e. most of the gate *magnitude* is
   dispensable under er-own/adam.

---

## Net take
The pt6 headline (learned selection → ~0.88 oracle-free ≈ ER) survives, and these probes localise
*why*: the win needs (i) **replay for the selector** (B: without it inference → chance), (ii)
**per-task (unmixed) training of the gate rows** (D2, B's buf-cur), and (iii) the modulation is a
**per-task out-layer logit adjustment** under ER (G). What does **not** matter: soft-vs-hard blending
(C), the selector's label source (D), sparsity regularisation (E), and — under er-own/adam — most of
the gate's magnitude (F). 1 seed throughout; ≤0.02 differences are noise.
