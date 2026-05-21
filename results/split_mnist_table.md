# Split MNIST CL Results

Hyperparameters: lr=1e-3, epochs_per_task=5, batch_size=64, ewc_lambda=1e5, er_buffer=200 (default configs; Phase 6 skipped)

Seeds: 42, 43, 44 (test sequence class order: seed=42)

| Method | Avg Final Acc (mean ± std) | Forgetting (mean ± std) |
|--------|---------------------------|------------------------|
| naive | 0.1979 ± 0.0003 | 0.7979 ± 0.0004 |
| joint | 0.9760 ± 0.0010 | 0.0000 ± 0.0000 |
| ewc | 0.1991 ± 0.0014 | 0.7965 ± 0.0011 |
| er | 0.7340 ± 0.0029 | 0.2621 ± 0.0029 |
| neuromod | 0.1975 ± 0.0007 | 0.7978 ± 0.0006 |
| neuromod+er | 0.7248 ± 0.0248 | 0.2707 ± 0.0255 |
