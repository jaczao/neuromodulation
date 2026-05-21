# Split MNIST CL Results

Hyperparameters (Phase 6b sweep, val-seq seed=7): naive lr=0.001 ept=5; joint lr=0.001 ept=10; ewc lr=0.0003 ept=5 λ=1e+04; er lr=0.0003 ept=5 buf=1000; neuromod lr=0.001 ept=10

Seeds: 42, 43, 44 (test sequence class order: seed=42)

| Method | Avg Final Acc (mean ± std) | Forgetting (mean ± std) |
|--------|---------------------------|------------------------|
| naive | 0.1979 ± 0.0003 | 0.7979 ± 0.0004 |
| joint | 0.9804 ± 0.0014 | 0.0000 ± 0.0000 |
| ewc | 0.2014 ± 0.0026 | 0.7948 ± 0.0021 |
| er | 0.9023 ± 0.0039 | 0.0919 ± 0.0044 |
| neuromod | 0.1983 ± 0.0002 | 0.7982 ± 0.0001 |
| neuromod+er | 0.9000 ± 0.0063 | 0.0941 ± 0.0069 |
