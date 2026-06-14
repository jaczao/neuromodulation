# Standard MNIST Results

Hyperparameters (Phase 6a sweep): vanilla lr=0.0003, epochs=20; neuromod lr=0.0003, epochs=20; batch_size=64

| Method | Test Acc (mean ± std) |
|--------|----------------------|
| vanilla MLP | 0.9796 ± 0.0008 |
| neuromod MLP | 0.9806 ± 0.0006 |

## pt4: every neuromod mechanism in the standard regime (3 seeds 42/43/44, same config)

Group R (runnable in single-task standard). Group N (drivers, stateful, task_route, logit+recency,
consolidation) are N/A by construction; see `prototype/SPEC-proto-pt4.md` / `iteration-notes.md`.

| Group | Mechanism (iter) | Target | Test Acc (mean ± std) | Verdict |
|-------|------------------|--------|-----------------------|---------|
| - | vanilla (Adam) | - | 0.9796 ± 0.0008 | reference |
| R1 | activation gain | activation | 0.9806 ± 0.0006 | preserve (slight +) |
| R2 | weight mask (2) | weight_mask | 0.9805 ± 0.0013 | preserve |
| R3 | logit calibration (6) | logit | 0.9811 ± 0.0006 | marginal improve |
| R5 | importance gating (7) | importance | 0.9791 ± 0.0011 | preserve |
| - | vanilla (SGD ref) | - | 0.8879 ± 0.0011 | R4 reference |
| R4 | plasticity / meta-LR (1) | plasticity | 0.8863 ± 0.0011 | preserve (vs SGD ref) |

Conclusion: every definable mechanism preserves vanilla MNIST accuracy (R3 marginally improves,
none degrades). Neuromodulation imposes no standard-accuracy tax. R4 uses an SGD main net (Adam
caveat), so it is compared to the SGD-vanilla reference, not the Adam vanilla.
