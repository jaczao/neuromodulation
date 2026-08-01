"""Device, seeding and shared numerical constants.

Extracted verbatim from results/pt7_neuromodulators.py. `seed_all` seeds python/numpy/torch
together (torch.manual_seed covers MPS; the cuda call is a no-op on this machine but kept for
CUDA portability).
"""
import random

import numpy as np
import torch

DEV = torch.device("mps" if torch.backends.mps.is_available()
                   else ("cuda" if torch.cuda.is_available() else "cpu"))

EPS = 1e-6
BF, BS = 0.1, 0.02                               # fast / slow EMA rates


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
