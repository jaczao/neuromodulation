# Chapter 3 Notes: Experiments and Results

Raw notes for the MLP prototype phase. All numbers are mean +- std over 3 seeds (42, 43, 44) unless stated otherwise.
Two runs exist: an initial Phase 7 run with default hyperparameters (commit 4b6c4f1, numbers logged in
`results/phase7_log.txt`) and a re-run with Phase 6 best hyperparameters (commit edc1379, numbers in
`results/standard_mnist_table.csv`, `results/split_mnist_table.csv`). The re-run numbers are the final
authoritative results.

---

## 3.1 Architecture

**Base MLP**
- Input: 784 (flattened 28x28 MNIST pixel, normalized with mean=0.1307, std=0.3081)
- Hidden: 400 -> ReLU -> 400 -> ReLU
- Output: 10 logits (shared head, no task-id at inference)
- No batchnorm, no dropout
- Source: `prototype/model.py`

**Neuromodulated MLP (GainModulator, variant=gain, target=hidden)**
- Sidecar wrapper `ModulatedMLP` around the base MLP; base MLP code unchanged
- Signal net: Linear(784->64) -> ReLU -> Linear(64->k=8); final linear zero-initialized
- Per-layer broadcast: mod_l = signal @ P_l, P_l in R^(k x 400), fixed random buffer (randn/sqrt(k)), one per hidden layer
- Modulation: h_l <- (1 + mod_l) * h_l applied post-ReLU on both hidden layers
- Zero-init of final linear ensures gain=1.0 at initialization; --use-neuromod off reproduces vanilla exactly
- Optional --neuromod-learned-projection flag makes P_l a learnable parameter (not used in final eval)
- Source: `prototype/neuromod.py`, `prototype/train.py`

---

## 3.2 Datasets

**Standard MNIST**
- Full MNIST, all 10 classes, standard train (60k) / test (10k) split
- Validation split: last 10k of the 60k training set; official test set not touched during tuning
- Normalization: whole-dataset stats (mean=0.1307, std=0.3081), never per-task
- Source: `prototype/data.py`, `get_standard_loaders()`

**Split MNIST (Continual Learning)**
- 5 tasks, 2 classes each: (0,1), (2,3), (4,5), (6,7), (8,9)
- Class-IL (class-incremental): shared 10-class head, no task-id at inference
- Test sequence: class-pair order shuffled with seed=42 (fixed for all final evaluations)
- Validation sequence: shuffled with seed=7 (used only during Phase 6 hyperparameter sweeps, never at eval)
- Source: `prototype/data.py`, `SplitMNIST`, `make_sequence()`

---

## 3.3 Continual Learning Methods

Source: `prototype/methods.py`

**Naive**
Sequential fine-tuning, no CL mechanism. Each task overwrites the previous.

**Joint (oracle upper bound)**
Single training pass over all tasks simultaneously (full data access). Not a real CL method; provides ceiling.

**EWC (Elastic Weight Consolidation)**
- Fisher diagonal computed after each task from ~200 per-task training samples using per-sample gradients (not batch-mean)
- Loss: task_loss + (lambda/2) * sum_i F_i * (theta - theta_i*)^2, accumulated across tasks
- Note: EWC is known to not beat Naive on class-IL Split MNIST (shared head, no task-id); van de Ven and Tolias (2019) confirm this. Output logit competition between tasks makes weight regularization ineffective.

**ER (Experience Replay)**
- Reservoir buffer of fixed size M
- Per batch: update buffer with current batch first (before gradient step), then sample buffer batch (size B, with replacement), concatenate with current-task batch, single gradient step on combined loss
- Buffer fill: reservoir sampling

**Neuromod (standalone)**
GainModulator applied to hidden activations, Naive CL method (no replay, no regularization).

**Neuromod+ER**
GainModulator combined with ER (best baseline). Same ER buffer and procedure; modulator applied on top.

---

## 3.4 Optimizer and General Training Setup

- Optimizer: Adam
- Batch size: 64 throughout
- Loss: CrossEntropyLoss
- Seeds: torch, numpy, and python random all seeded from --seed; torch.cuda.manual_seed_all also called (no-op on MPS, portable to CUDA)
- Source: `prototype/configs.py`, `prototype/train.py`

---

## 3.5 Hyperparameter Sweeps (Phase 6)

Sweeps run on validation data only. Grid is small by design (1.5-day sprint budget).

### 3.5.1 Standard MNIST Sweep (Phase 6a)

Grid: lr in {3e-4, 1e-3} x epochs in {10, 20}, 1 seed per cell.
Selection criterion: val_acc on held-out 10k of MNIST train split.
Run for both vanilla MLP and neuromod MLP with identical budget.

Best configs selected:
- Vanilla: lr=3e-4, epochs=20 (BEST_STANDARD_VANILLA)
- Neuromod: lr=3e-4, epochs=20 (BEST_STANDARD_NEUROMOD)

Source: `prototype/configs.py` (BEST_STANDARD_VANILLA, BEST_STANDARD_NEUROMOD)

### 3.5.2 CL Sweep (Phase 6b)

Grid: lr in {3e-4, 1e-3} x epochs_per_task in {5, 10}, plus method-specific knobs.
Selection criterion: avg_final_acc on validation sequence (seed=7).
1 seed per trial.

Best configs selected:
- Naive:    lr=1e-3, epochs_per_task=5  (BEST_CL_NAIVE)
- Joint:    lr=1e-3, epochs_per_task=10 (BEST_CL_JOINT)
- EWC:      lr=3e-4, epochs_per_task=5, ewc_lambda=1e4  (BEST_CL_EWC)
  Note: default ewc_lambda was 1e5; the sweep selected 1e4 for the val seq.
  Note: ewc_samples=200, per-sample Fisher.
- ER:       lr=3e-4, epochs_per_task=5, er_buffer_size=1000  (BEST_CL_ER)
  Note: default buffer was 200; 1000 is the Phase 6 best.
- Neuromod: lr=1e-3, epochs_per_task=10, use_neuromod=True  (BEST_CL_NEUROMOD)
  Note: neuromod+er inherits the ER best config.

Source: `prototype/configs.py` (BEST_CL_* constants)

---

## 3.6 Results

### 3.6.1 Standard MNIST (Phase 7 re-run, commit edc1379)

Hyperparameters: lr=3e-4, epochs=20, batch_size=64 (same for both).
Evaluation: official MNIST test set (10k). 3 seeds: 42, 43, 44.

| Method       | Test Acc mean | Test Acc std |
|--------------|---------------|--------------|
| vanilla MLP  | 0.9796        | 0.0008       |
| neuromod MLP | 0.9806        | 0.0006       |

Raw CSV values: vanilla=0.9795666..., neuromod=0.9805666...
Source: `results/standard_mnist_table.csv`, `results/standard_mnist_table.md`

**Note on initial run vs re-run:** The initial Phase 7 run (commit 4b6c4f1, `results/phase7_log.txt`)
used default hyperparameters (lr=1e-3, epochs=10) and produced vanilla=0.9789+-0.0012 and
neuromod=0.9750+-0.0036. The re-run (edc1379) with Phase 6 best configs (lr=3e-4, epochs=20) produced
the final numbers above. The log file documents the initial run; the CSV/MD files document the re-run.

Per-seed test accuracy from initial run (default lr=1e-3, epochs=10), from `results/phase7_log.txt`:
- vanilla: seed42=0.9794, seed43=0.9800, seed44=0.9773
- neuromod: seed42=0.9700, seed43=0.9783, seed44=0.9766

Per-seed test accuracy from re-run is not separately logged; only the mean and std are in the CSV/MD.

### 3.6.2 Split MNIST CL (Phase 7 re-run, commit edc1379)

Test sequence: seed=42 (class order fixed). 3 seeds: 42, 43, 44.
Each method uses its Phase 6b best hyperparameters (listed in Section 3.5.2).

| Method       | Avg Final Acc mean | Avg Final Acc std | Forgetting mean | Forgetting std |
|--------------|--------------------|-------------------|-----------------|----------------|
| naive        | 0.1979             | 0.0003            | 0.7979          | 0.0004         |
| joint        | 0.9804             | 0.0014            | 0.0000          | 0.0000         |
| ewc          | 0.2014             | 0.0026            | 0.7948          | 0.0021         |
| er           | 0.9023             | 0.0039            | 0.0919          | 0.0044         |
| neuromod     | 0.1983             | 0.0002            | 0.7982          | 0.0001         |
| neuromod+er  | 0.9000             | 0.0063            | 0.0941          | 0.0069         |

Source: `results/split_mnist_table.csv`, `results/split_mnist_table.md`

Raw CSV values (full precision from `results/split_mnist_table.csv`):
- naive:       avg_acc=0.19791561606992772, std=0.0002647170238363385, forget=0.7978823781268901, fstd=0.0004179692494008245
- joint:       avg_acc=0.9803501908689344,  std=0.0014060607627670648, forget=0.0,               fstd=0.0
- ewc:         avg_acc=0.2013999863720458,  std=0.0026416938468712104, forget=0.7947942153118223, fstd=0.0020577841082365144
- er:          avg_acc=0.90232556493088,    std=0.0039028320050778437, forget=0.09185201791398086, fstd=0.004351952182856115
- neuromod:    avg_acc=0.19828542612203734, std=0.00021787664139882076, forget=0.798206903516852,  fstd=8.852313090328835e-05
- neuromod+er: avg_acc=0.9000118073475022, std=0.0063242837656132655, forget=0.09414276115166377, fstd=0.006870896095418008

Per-seed avg_final_acc from `results/phase7_log.txt` (initial run, default configs, ER buf=200):
- naive:       seed42=0.1977, seed43=0.1978, seed44=0.1983
- joint:       seed42=0.9748, seed43=0.9761, seed44=0.9771
- ewc:         seed42=0.1983, seed43=0.1980, seed44=0.2011
- er:          seed42=0.7337, seed43=0.7306, seed44=0.7377
- neuromod:    seed42=0.1968, seed43=0.1984, seed44=0.1975
- neuromod+er: seed42=0.7412, seed43=0.7435, seed44=0.6897

**Observation from log:** Joint summary in log shows acc=0.9760+-0.0010 (initial run), re-run shows 0.9804+-0.0014 (lr=1e-3, 10 epochs vs lr=1e-3, 10 epochs -- same config, but the per-seed accs in the log are 0.9748, 0.9761, 0.9771). The re-run CSV shows 0.9804; this discrepancy may reflect a different random seed state or a code difference between commits.

---

## 3.7 Summary of Key Findings

Source for interpretation: `results/phase7_log.txt` lines 269-279, commit message for edc1379.

1. **Standard MNIST:** neuromod MLP (0.9806+-0.0006) is within noise of vanilla MLP (0.9796+-0.0008).
   Difference is +0.001, well within one standard deviation. Neuromodulation does not hurt standard accuracy.

2. **CL - Naive vs neuromod (standalone):** neuromod (0.1983+-0.0002) is statistically indistinguishable
   from naive (0.1979+-0.0003). Forgetting is similarly near-total for both (~0.798). Gain modulation
   alone does not reduce catastrophic forgetting.

3. **CL - EWC:** 0.2014+-0.0026, near-identical to naive. Confirms known result: EWC does not help on
   class-IL with shared head (van de Ven and Tolias 2019 result). No task-id means output logit
   competition dominates weight regularization.

4. **CL - ER:** 0.9023+-0.0039 with buf=1000. Strong improvement over naive. Buffer size matters: default
   buf=200 gave 0.7340+-0.0029 (initial run, log); Phase 6 best buf=1000 gave 0.9023.

5. **CL - Neuromod+ER:** 0.9000+-0.0063. Within noise of ER alone (0.9023+-0.0039). Neuromodulation
   does not add to ER. Not complementary at this scale.

6. **Joint (oracle):** 0.9804+-0.0014. Upper bound. ER with buf=1000 reaches 0.9023, which is ~8 points
   below joint.

---

## 3.8 Config Reference (final eval values)

All from `prototype/configs.py`:

```
StandardConfig defaults:    lr=1e-3, epochs=10, batch_size=64
BEST_STANDARD_VANILLA:      lr=3e-4, epochs=20
BEST_STANDARD_NEUROMOD:     lr=3e-4, epochs=20, use_neuromod=True

CLConfig defaults:          lr=1e-3, epochs_per_task=5, batch_size=64,
                            ewc_lambda=1e5, ewc_samples=200, er_buffer_size=200

BEST_CL_NAIVE:              lr=1e-3, epochs_per_task=5
BEST_CL_JOINT:              lr=1e-3, epochs_per_task=10
BEST_CL_EWC:                lr=3e-4, epochs_per_task=5, ewc_lambda=1e4
BEST_CL_ER:                 lr=3e-4, epochs_per_task=5, er_buffer_size=1000
BEST_CL_NEUROMOD:           lr=1e-3, epochs_per_task=10, use_neuromod=True
```

GainModulator params (not config-tuned, fixed in code):
- k=8 (signal dimension)
- hidden_dim=400
- signal net: Linear(784->64)->ReLU->Linear(64->8), final linear zero-initialized
- projection P_l: randn(8, 400)/sqrt(8), registered as buffer (not learnable by default)

Source: `prototype/neuromod.py:37-58`
