"""Reservoir replay buffer.

Extracted verbatim from results/pt7_neuromodulators.py. Not on the brief's four-item extraction
list, but pulled in as a judged addition: every direction is to be reported in three memory regimes
(normal / rehearsal-free / memory-budgeted), and `buffer_bytes` is a first-class ledger column, so
the one object that defines what "stored" means belongs in the shared core rather than being
re-implemented per problem package.

Buffer discipline (pt1): update BEFORE the gradient step, sample WITH replacement, reservoir
sampling for the fill.
"""
import random

import torch


class Reservoir:
    def __init__(self, cap):
        self.cap = cap; self.X = torch.zeros(cap, 784); self.Y = torch.zeros(cap, dtype=torch.long)
        self.n = 0; self.filled = 0

    def add(self, x, y):
        x = x.view(x.size(0), -1).cpu(); y = y.cpu()
        for i in range(len(x)):
            if self.filled < self.cap:
                self.X[self.filled] = x[i]; self.Y[self.filled] = y[i]; self.filled += 1
            else:
                j = random.randint(0, self.n)
                if j < self.cap:
                    self.X[j] = x[i]; self.Y[j] = y[i]
            self.n += 1

    def sample_task(self, j, b):
        idx = (torch.div(self.Y[:self.filled], 2, rounding_mode="floor") == j).nonzero().squeeze(1)
        if len(idx) == 0:
            return None
        p = idx[torch.randint(0, len(idx), (b,))]
        return self.X[p], self.Y[p]

    def sample_any(self, b):
        if self.filled == 0:
            return None
        idx = torch.randint(0, self.filled, (b,))
        return self.X[idx], self.Y[idx]

    def nbytes(self):
        """Resident bytes of stored samples (the memory-regime accounting number)."""
        return self.X.element_size() * self.X.numel() + self.Y.element_size() * self.Y.numel()
