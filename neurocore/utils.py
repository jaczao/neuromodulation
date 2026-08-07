"""Device, seeding and shared numerical constants.

Extracted verbatim from results/pt7_neuromodulators.py. `seed_all` seeds python/numpy/torch
together (torch.manual_seed covers MPS; the cuda call is a no-op on this machine but kept for
CUDA portability).

`rng_frozen` is promoted here on its SECOND use (driver_traces/live_traces.py was the first, around
observer construction; signals.dataset_mean is the second, around an extra pass over the data).
"""
import contextlib
import random

import numpy as np
import torch

DEV = torch.device("mps" if torch.backends.mps.is_available()
                   else ("cuda" if torch.cuda.is_available() else "cpu"))

EPS = 1e-6
BF, BS = 0.1, 0.02                               # fast / slow EMA rates


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@contextlib.contextmanager
def rng_frozen():
    """Snapshot/restore torch+numpy+random around a block that must be RNG-NEUTRAL.

    Anything added to a training run has to leave the RNG stream where it found it, and "it only
    reads" is NOT sufficient: `_BaseDataLoaderIter.__init__` draws a `_base_seed` from the default
    generator whenever `loader.generator is None`, regardless of sampler, workers or shuffling. So
    merely ITERATING a loader for a diagnostic — or for an extra pass to compute a dataset mean —
    silently moves the run off its reference trajectory. Constructing a module (a head, a predictor)
    does the same through its weight init.

    Wrap the added work in this and the anchors keep reproducing bit-exact.
    """
    t, n, r = torch.get_rng_state(), np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(t); np.random.set_state(n); random.setstate(r)
