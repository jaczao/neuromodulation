# pt6 — gain-SYNAPSE for soft_mlp (soft-blend and hard-argmax)

Companion to `pt6_synapse.py` / `.log`. Gated layers net0(400×784) + net2(400×400) + net4(10×400),
n_syn = 477 600; gate table `P:(T, n_syn)` = 2.4M (a LOOKUP — the 374M blow-up is mean_image/embedding
only). Γ = 1+P, class-IL, seed 42, lr 1e-3, ep 5, buffer 1000, **1 seed**.
Refs: `er-sgd 0.723`, `er-adam 0.895`; NEURON soft_mlp soft `er-own/adam 0.886`, `er-own/sgd 0.856`,
`buf-own/sgd 0.856`.

## The deferral was over-stated — no per-sample Γ is needed
The remaining blocker on record was "the soft-blend eval needs a per-sample Γ `(B,d_out,d_in)`, which
must be chunked under `no_grad`". It doesn't. Since `Γ_i = Σ_t p_it Γ_t` and `(Γ⊙W)x` is **linear in
Γ**:

```
(Γ_i⊙W)x_i + b  =  Σ_t p_it · [ (Γ_t⊙W)x_i + b ]      (uses Σ_t p_it = 1 for the bias)
```

so a blended layer is just the p-weighted sum of the T per-task gated outputs — **T=5 matmuls per
layer, no expansion, and exact** (verified: `max|grouped − mixed(one-hot)| = 2.1e-07`, float noise).
hard-argmax is cheaper still (each sample has one task → ≤5 masked matmuls, the training path).

## Results
```
                oracle   soft    hard   infer
er-own  sgd     .748    .745    .746    .892
er-own  adam    .991    .890    .888    .892
buf-own sgd     .989    .887    .878    .883
buf-own adam    .649    .598    .605    .883
```
Per-synapse gate magnitude (mean |P| per gated layer):
```
                net0      net2      net4
er-own  sgd    2.2e-05   5.0e-05   6.6e-04
er-own  adam   2.1e-02   1.4e-02   7.9e-02
buf-own sgd    1.2e-01   1.7e-01   6.2e-02
buf-own adam   3.8e-02   2.7e-02   1.8e-02
```

## Readings
1. **Synapse matches or beats neuron in the working regimes.** er-own/adam soft **0.890** ≈ neuron
   0.886 (inside the ~0.02 noise floor). buf-own/sgd soft **0.887** vs neuron 0.856 (**+0.031**) and
   oracle **0.989** vs 0.939 (**+0.050**) — per-synapse is the better *standalone* mechanism.
2. **soft ≈ hard replicates at synapse** (0.890/0.888, 0.887/0.878, 0.745/0.746). Same cause as the
   neuron probe C: a confident learned selector makes softmax ≈ one-hot, so blending buys nothing.
   This now holds across BOTH granularities.
3. **er-own/sgd is INERT, and the reason is an optimizer-routing asymmetry** — not a property of
   synapse gating. In `er-own` the gate rides the **main optimizer** (SGD here), and SGD at lr 1e-3
   cannot move 2.4M gate params in 5 epochs: |P| stays ~1e-5..7e-4, so there is effectively no
   modulation and every mode collapses to the same number (.748/.745/.746 ≈ plain er-sgd .723). In
   `buf-own` the gate has its **own Adam optimizer**, which is exactly why buf-own/sgd trains fine
   (|P| 0.12–0.17) and lands at 0.887. **If a per-synapse gate looks inert, check which optimizer
   owns it before concluding the mechanism failed.**
4. **The per-layer localisation rule replicates.** With ER the gate concentrates in the **out** layer
   (er-own/adam net4 7.9e-2 vs hidden 1.4–2.1e-2, ~4–6×); **standalone it moves to the hidden**
   layers (buf-own/sgd net2 1.7e-1 > net0 1.2e-1 > net4 6.2e-2). Same split found at neuron
   granularity — replay handles features so the gate only recalibrates logits; without replay the
   gate must modulate features itself.
5. buf-own/adam collapses (0.598) as it did at neuron granularity (0.661) — that arm/optimizer pairing
   is simply bad, independent of granularity.

**Net:** per-synapse soft_mlp is fully runnable and is the strongest *standalone* pt6 cell
(buf-own/sgd 0.887 oracle-free, oracle 0.989), while under ER it ties neuron. It adds no new
oracle-free ceiling — still ≈ ER parity (0.89) — consistent with the pt6 verdict. 1 seed; <0.02 is noise.
