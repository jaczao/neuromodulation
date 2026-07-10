# SPEC: Prototype Iteration pt5 — the generalized driver system, first driver = task-id (one-hot)

## Context
pt2 (Iter 1-4) and pt3 (Iter 5-10) closed the continual-learning front: no neuromod variant
beats Naive by >=5pts on class-IL Split MNIST nor complements ER by >=2pts; replay is the only
lever and the modulator, fed only current-task signals, adds ~0 on top (see
`prototype/iteration-notes.md` "pt3 SUMMARY"). pt4 covered the standard regime (all Group-R
mechanisms preserve vanilla accuracy).

pt5 opens a NEW mechanism front: a **generalized driver system**. Today the driver axis is a
single string (`surprise|uncertainty|activation_stats`) consumed ONLY by the `weight_mask` target
(plus the special `logit+recency`). pt5 replaces that with a **driver -> bottleneck -> target**
architecture in which zero-or-more named drivers of varying dimension feed a bottleneck that is
up-projected to ANY target. This is a real new mechanism, so rule 8 (one axis at a time) is
honoured strictly: pt5 builds the infrastructure and then explores exactly **one driver
(`task_id`, one-hot) with one projection variant per iteration**. Further drivers (dopamine,
acetylcholine, norepinephrine, serotonin) and mechanisms are deferred to a later SPEC.

`task_id` is an **oracle** (the true task index is fed to the modulator at train AND test). This
is accepted: pt5 results are only claimed in scenarios where a task-IL-style privileged task
signal is acceptable. The task id is NOT inferred (that was pt3 Iter 8, rejected). Eval remains
class-IL 10-way for the reported metric (see Methodology 7); the oracle lives in the modulator
input, not in eval-time output masking.

## What pt5 builds (the generalized bottleneck, minimal slice)
Every modulator is split into two stages: **input assembly -> bottleneck**, then **target head
(up-projection)**.
- Input = `[optional image context] ++ [zero-or-more driver values]`. 0 drivers is legal only if
  the image context is present; drivers-only is legal with >=1 driver.
- If drivers are the only input, the bottleneck `z` IS the concatenation of driver values (no
  down-projection). If an image context is present it is down-projected to `k_img` and the drivers
  are concatenated raw into the code: `z = concat(signal_net(image), driver_values)`, width
  `k = k_img + sum(d_i)`.
- The target head up-projects `z` to the target's modulation (this is the existing per-target
  projection, made width-agnostic: it accepts `k` instead of a hardcoded width).

For pt5: `context = none`, single driver `task_id` with mechanism `onehot` (dim = `n_tasks` = 5),
so the bottleneck is the one-hot `z = e_t in {0,1}^T`, and the "projection" P: T -> D is the whole
modulator. D = `hidden_dim` (gain / plasticity, per gated layer) or `d_out*d_in` (weight_mask on
the masked layer).

## The task-id driver and its three projections (the three iterations)
`e_t` is the one-hot for the current task t. A projection `P (T x D)` maps `e_t` to a per-element
gate `gamma = (e_t @ P)` over the target's D elements. The three variants are the three iterations:

| iter | projection | P construction | semantics |
|------|-----------|----------------|-----------|
| 1 | `disjoint` | fixed binary; each column has a 1 in exactly one task row (seeded even partition) | fully disjoint per-task subnetworks, no shared capacity (extreme HAT with an oracle id) |
| 2 | `shared` | fixed binary; a fraction `shared_frac` (default 0.5) of columns are all-ones (1 for every task), the rest disjointly assigned as in iter 1 | shared backbone + task-private capacity |
| 3 | `learned` | real P, sigmoid-gated to (0,1), trained by a **modulator-only replay meta-loss** over tasks seen so far (buffer of past examples; the meta-gradient updates ONLY P, the main net is detached) | learn the allocation; "ER for the neuromod net only" |

Gate application per target. Let `raw = (e_t @ P)`. For the fixed projections (iter 1, 2) `raw` is
binary in {0,1} and is used as the gate DIRECTLY (no squashing, no shift): every target gets a hard
on/off gate in {0,1}. For the learned projection (iter 3) `raw` is real and squashed per target.
Every target uses an **SGD main net** in pt5 (see Methodology 6):
- **gain** (`activation` target), applied as `h_l <- h_l * gamma`, run in BOTH forms:
  - **bounded-[0,1]**: `gamma = sigmoid(raw)` (learned) or `raw in {0,1}` (fixed). Range [0,1],
    suppress-only (unassigned units are turned off for task t).
  - **unbounded**: `gamma = 1 + raw` (learned; range (-inf, +inf), init 1.0 at zero-init P, so
    gain can amplify above 1 and invert below 0) or `raw in {0,1}` (fixed).
  The two forms differ ONLY for the learned projection; under a fixed binary P both collapse to the
  same {0,1} gate, so gain is run once in iter 1/2 and twice (both forms) in iter 3.
- **plasticity**: per-neuron `alpha = gamma`, range [0,1] (`sigmoid(raw)` learned, `raw in {0,1}`
  fixed = frozen vs fully plastic); unassigned units are frozen for task t.
- **weight_mask**: per-synapse `M = gamma`, range [0,1] (`sigmoid(raw)` learned, `raw in {0,1}`
  fixed = synapse off vs on) on **multiple** layers at once, and the layer set depends on the
  retention lever in play:
  - **with masked loss** (the `naive`/`neurom` standalone conditions): mask **both hidden layers**
    (`net.0` and `net.2`). Lever B (loss masking) already handles the shared output head, so the
    modulator gates the hidden capacity.
  - **without masked loss** (the `er`/`neurom+er` conditions): mask **both hidden layers plus the
    output head** (`net.0`, `net.2`, `net.4`), so the task-conditioned mask also reaches the
    class-IL logit bottleneck that no lever B is covering.
  The modulator emits one per-synapse mask per masked layer (each layer has its own projection
  `P_l: T -> d_out_l * d_in_l` for the fixed variants, or its own learned `P_l` for iter 3).

Key structural fact: for the fixed projections (iter 1, 2) the modulator is **parameter-free** (P
is a fixed buffer, nothing to train), so the main net simply trains under a fixed per-task gate.
Only iter 3 has trainable projection params, and they are trained by the meta-loss, never by the
main loss.

## Goal
At 1 seed (screening), for each `target x projection` cell, answer BOTH:
- **standalone**: does the oracle-task-conditioned modulator (method=naive, masked-loss ON) beat
  the same-optimizer `naive + masked-loss` baseline?
- **+ER**: does the modulator stacked on ER (masked-loss OFF) beat plain ER by >=2pts?

The mechanism under test is task-conditioned capacity allocation (disjoint vs shared-backbone vs
learned) driven by a privileged task id. The scientific question: with the shared output head
NOT restricted at eval, can hidden/synapse allocation alone move class-IL forgetting, standalone
and on top of replay?

## Methodology — non-negotiable (carried from pt2/pt3, plus pt5 additions)
1. **Config-selected, one-flag.** The whole driver system is behind `--neuromod-drivers` (mapping
   string). `--use-neuromod` OFF and an empty driver string must reproduce vanilla numerically.
2. **Keep the legacy driver code and flags intact.** The old `--neuromod-driver
   {surprise,uncertainty,activation_stats}` path (`_weight_mask_driver_train_task`) stays as a
   parallel route, unchanged, for a later old-vs-new comparison. The new path is selected only
   when `--neuromod-drivers` is non-empty. Do NOT delete or rewrite legacy behaviour.
3. **1 seed (screening).** This intentionally relaxes rule 5 (3 seeds) for the pt5 screen only.
   Any cell that clears an accept bar gets a 3-seed confirm before it is a reported "final"
   number; that confirm is deferred (not part of this SPEC's runs).
4. **Masked loss per condition.** `output_masking='loss'` (MaskedCE, per-sample label->task-pair)
   is ON for the two non-replay conditions (`naive`, `neurom`) and OFF (`none`) for the two
   replay conditions (`er`, `neurom+er`). ER supplies its own retention; masked loss is lever B
   for the non-replay conditions only.
5. **Report BOTH comparisons** per `target x projection` cell (carried pt3 rule): `neurom vs
   naive+masked-loss` AND `neurom+ER vs ER`.
6. **SGD main net throughout (no optimizer confound).** ALL three targets (`gain`, `plasticity`,
   `weight_mask`) and ALL four conditions (`naive`, `neurom`, `er`, `neurom+er`) use a plain SGD
   main net in pt5, so every comparison is same-optimizer by construction. This removes the
   Adam/SGD masked-loss confound (naive+masked-loss = 0.39 Adam vs 0.63 SGD) and matches the
   "masked-loss + SGD is the lever" finding; it also sidesteps the plasticity Adam-moments caveat
   (gating the gradient before Adam is not LR scaling). Baselines are `naive-SGD+masked-loss` (the
   standalone bar for every target) and `ER-SGD` (the `+ER` bar). Never introduce an Adam run.
7. **task-id oracle, class-IL eval.** The true task id is fed to the modulator at train and eval;
   it is an accepted privileged input, not inferred. The reported avg_final_acc is class-IL
   (10-way, no eval output masking), so it is directly comparable to `naive+masked-loss` and `ER`.
   (Restricting the eval output to the task's classes, i.e. a true task-IL number, is a one-flag
   change `output_masking='taskil'`, noted but not the pt5 default.)
8. **No hardcoded hyperparameters** in training code; route through `configs.py` (rule 7).
9. **One driver, one mechanism per iteration.** Across iter 1-3 ONLY the projection changes; the
   driver (`task_id=onehot`), targets, and everything else are held fixed.
10. **No em dashes** in any output (commas, parentheses, separate sentences, `->`).

## Iteration order
Each iteration runs the same 3 targets (`gain`, `plasticity`, `weight_mask`) and the same 2
neuromod conditions (`neurom`, `neurom+er`) against the shared baselines.

### Iteration 1 — disjoint per-task subnetworks (`projection=disjoint`)
Fixed binary P, each target element assigned to exactly one task (seeded even partition). Each
task uses a private, non-overlapping slice of the gated capacity.
**Hypothesis.** Even with perfect task ids, gating hidden units (gain, plasticity, and weight_mask
in the standalone/masked-loss cells) cannot touch the shared output head, so those standalone
numbers are expected near `naive+masked-loss` (the head ceiling from the pt2/pt3 diagnosis). The
cells that can move are `weight_mask+ER` (which also masks `net.4`) and, generally, any `+ER` cell
where private capacity reduces interference. Report the delta; a disjoint allocation that does NOT
move the number is itself the result.

### Iteration 2 — shared backbone + private capacity (`projection=shared`, `shared_frac=0.5`)
Like iter 1 but ~50% of elements are shared by all tasks (all-ones columns), the rest disjoint.
**Hypothesis.** A shared backbone should transfer common features while private capacity limits
interference; tests whether partial sharing beats the full-disjoint extreme of iter 1 at equal
oracle information.

### Iteration 3 — learned allocation via modulator-only replay (`projection=learned`)
P is learned, trained by a meta-loss that replays past-task examples through the (detached) main
net with the modulator conditioned on each sample's task id, updating ONLY P. The main net is
never updated by replay.
**Hypothesis (and the point of the iteration).** This is exactly the "meta-loss variant" we
predict is weaker than ER: the buffer shapes the modulator, not the shared weights/head, so the
head still drifts to current classes. Predict `neurom+ER` <= `ER` and standalone near
`naive+masked-loss`. Iter 3 tests that reasoning empirically. If it beats ER, that is a genuine
surprise worth a 3-seed confirm.

## Experimental matrix (1 seed)
Baselines (computed once, reused as reference columns; both SGD, see Methodology 6):
- `naive-SGD + masked-loss`: the standalone bar for every target.
- `ER-SGD` (no masked-loss): the `+ER` bar.

Target-configs: `plasticity`, `weight_mask`, and `gain`. Gain has two forms (`gain-unbounded`,
`gain-bounded01`) that are distinct only under the learned projection; under a fixed binary P both
collapse to the {0,1} gate, so gain is a single config in iter 1/2 and two configs in iter 3. Per
target-config x `projection in {disjoint, shared, learned}`:
- `neurom` = neuromod + masked-loss, method=naive, SGD. weight_mask masks `net.0`+`net.2`.
- `neurom+ER` = neuromod + ER, masked-loss OFF, SGD. weight_mask masks `net.0`+`net.2`+`net.4`.

So iter 1 and iter 2 = 3 target-configs x 2 neuromod runs = 6 runs each; iter 3 = 4 x 2 = 8 runs;
total 20 neuromod runs, plus the 2 shared baselines. Report a table per iteration with columns:
`target-config | naive-SGD+masked-loss | neurom (delta) | ER-SGD | neurom+ER (delta)`.

## Config / CLI (mapping-string format)
New `CLConfig` fields (and matching `argparse` overrides), all defaulting to the OFF/legacy path:
- `neuromod_drivers: str = ""`: comma-separated `name=mechanism` pairs, e.g. `"task_id=onehot"`.
  Non-empty selects the new driver path; empty keeps legacy/off. Presence of a key activates the
  driver; the value picks its mechanism. pt5 implements only `task_id=onehot`; any other pair
  raises `NotImplementedError` (stub for later drivers).
- `neuromod_context: str = "image"`: `image | none`. pt5 runs use `none` (drivers-only
  bottleneck). `none` with an empty driver set is a config error.
- `neuromod_projection: str = "disjoint"`: `disjoint | shared | learned` (the three iterations).
- `neuromod_shared_frac: float = 0.5`: fraction of all-task shared elements for `shared`.
- `neuromod_proj_seed: int = 0`: layout seed for the fixed binary projections.
- `neuromod_gain_form: str = "unbounded"`: `unbounded` (`h*(1+raw)`, range (-inf,+inf), init 1.0) |
  `bounded01` (`h*sigmoid(raw)`), the learned-projection gain form; inert under a fixed binary P
  (the gate is {0,1} either way). The gain target is run in both forms for the learned projection.
- `neuromod_mask_layers: str = "2"`: comma-separated `net` linear indices the weight_mask target
  masks together; the pt5 runner sets `"0,2"` for the masked-loss conditions and `"0,2,4"` for the
  ER conditions. Supersedes the single `neuromod_mask_layer` for the driver path.
- learned projection reuses `neuromod_lr` (the meta-optimizer LR) and `er_buffer_size` (the
  modulator-only replay buffer size).
- All pt5 runs pass `--optimizer sgd` (Methodology 6); the CL branch already honours it.

Target is the existing `--neuromod-target` (`activation` for gain, `plasticity`, `weight_mask`).
When `--neuromod-drivers` is set the driver bottleneck + `neuromod_projection` REPLACES the
target's default (image-driven) signal path; the target's APPLICATION to the net (how gain/alpha/
mask multiplies the forward/gradient) is unchanged.

Example runs:
```
# iter 1, gain (bounded[0,1]), standalone (masked loss), SGD
uv run python prototype/train.py --method naive --optimizer sgd --output-masking loss \
  --use-neuromod --neuromod-drivers "task_id=onehot" --neuromod-context none \
  --neuromod-target activation --neuromod-gain-form bounded01 \
  --neuromod-projection disjoint --seed 42 --no-wandb
# iter 3, weight_mask on hidden+output, +ER (no masked loss), SGD
uv run python prototype/train.py --method er --optimizer sgd --er-buffer-size 1000 \
  --use-neuromod --neuromod-drivers "task_id=onehot" --neuromod-context none \
  --neuromod-target weight_mask --neuromod-mask-layers "0,2,4" \
  --neuromod-projection learned --seed 42 --no-wandb
```

## Implementation
Keep every change behind the new flags; legacy untouched.
- `neuromod.py`: add a minimal `Driver` base + `TaskIdOneHot` driver + `DriverBank` (concat,
  detached, `set_driver` on the modulator base); three projection builders
  (`build_disjoint_proj`, `build_shared_proj`, and a learned `nn.Parameter` P with sigmoid); a
  pt5 driver-modulator that maps `e_t -> gamma` for each target and exposes the per-target
  application hooks, with the gain hook supporting both `neuromod_gain_form` values. Lift
  `driver_dim` / `current_driver` / `set_driver` into the `Modulator` base so
  `gain`/`plasticity`/`weight_mask` all inherit driver support. Extend the weight_mask wrapper to a
  **multi-layer** form: wrap every `net` linear in `neuromod_mask_layers` as a `ModulatedLinear`
  and have the modulator emit one per-synapse mask per wrapped layer (each with its own `P_l`).
- `configs.py`: the fields above.
- `train.py`: a new CL path taken when `config.neuromod_drivers` is non-empty. It builds an
  **SGD** main net (Methodology 6), the bank + projection(s), sets the current task's one-hot each
  step and at eval, composes with `naive` and `er`, applies masked loss per condition, sets the
  weight_mask layer set per condition (`0,2` with masked loss, `0,2,4` without), and for
  `projection=learned` runs the modulator-only meta-replay update (replay buffer trains ONLY P,
  main net detached). Existing branches stay.
- `tests/test_data.py` (or a new `tests/test_neuromod_pt5.py`): (a) OFF / empty-driver parity vs
  vanilla; (b) `disjoint` P columns are disjoint and cover all elements; (c) `shared` P has
  ~`shared_frac` all-ones columns; (d) `task_id=onehot` driver dim == n_tasks; (e) an unknown
  driver pair raises; (f) gain `unbounded` vs `bounded01` apply the intended form; (g) multi-layer
  weight_mask wraps every listed layer and each unmasked-parity forward matches `nn.Linear`. Re-run
  `pytest tests/` (rule 6) after any `data.py` touch.
- Runner `results/pt5_taskid.py` (mirrors `results/iterN_*.py`): loops target-configs
  (`plasticity`, `weight_mask`, `gain`; gain runs both forms only for the learned projection) x
  `projection` x `{neurom, neurom+er}` at 1 seed, all SGD, setting the weight_mask layer set per
  condition; prints the per-iteration table, logs to `results/pt5_taskid.log`.

## Accept / reject (per cell, screening)
- **standalone**: accept-for-confirm if `neurom` beats the same-optimizer `naive+masked-loss` by a
  clear margin (screening threshold, in the spirit of the pt3 bar: beat `naive+masked-loss`, not
  just Naive). Otherwise reject-at-1-seed.
- **+ER**: accept-for-confirm if `neurom+ER` beats `ER` by >=2pts (carried pt3 rule).
- Any accept-for-confirm cell is re-run at 3 seeds before it is a reported result.

## Honest caveats (state these in the writeup)
- **Oracle.** Every pt5 number uses a privileged task id in the modulator input; it is a task-IL-
  style result, not a class-IL claim, even though the metric is class-IL 10-way accuracy.
- **The shared head is the bottleneck, and only some cells reach it.** class-IL forgetting is
  dominated by output-logit competition (van de Ven & Tolias; the pt2/pt3 diagnosis). `gain` and
  `plasticity` gate hidden units only, so they never touch the head and a strong standalone effect
  is NOT expected from them even with a perfect id. `weight_mask` DOES reach the head in the `+ER`
  conditions (it masks `net.4`), and in the standalone conditions the head is instead covered by
  lever B (masked loss) while the mask gates the two hidden layers; those are the cells most likely
  to move.
- **Iter 3 is the meta-loss-vs-ER test.** Spending the buffer on the modulator (not the shared
  weights) is predicted to underperform plain ER; if confirmed it is direct evidence for the
  "replay is the lever" conclusion, not a failure of implementation.
- **SGD throughout.** Every condition uses an SGD main net so all comparisons are same-optimizer
  and the Adam/SGD masked-loss confound cannot arise (Methodology 6).

## Out of scope (pt5)
- Any driver other than `task_id` (dopamine / acetylcholine / norepinephrine / serotonin), the
  learned-neuromodulator-net variant, and per-driver mechanism menus (later SPEC).
- Image-context + driver bottlenecks (pt5 is drivers-only); phase-split pre/post-forward driver
  updates; the fused next-image lookahead.
- 3-seed confirmation runs, wider projection sweeps, output-head weight_mask as its own iteration.
- Standard regime, Permuted MNIST, GRU/CNN/ViT, repo scaffolding (unchanged, post-iteration).
