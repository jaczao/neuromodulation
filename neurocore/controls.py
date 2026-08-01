"""Controls and diagnostics — the highest-value reusable asset, previously duplicated across a dozen
pt7_*.py study files.

Every one of these exists because reading a result WITHOUT it produced a wrong conclusion at least
once in pt5-pt7. They are not optional extras.

THE `free` CONTROL / RNG-MATCHED BASELINE (mandatory).
  `free` is K zero-init heads with NO biological target, trained end-to-end by the task loss. Stacked
  behind a zero-init P it is provably inert (|g| == 0.000000 at every step) so it is NUMERICALLY the
  plain baseline -- but CONSTRUCTING the unused heads consumes torch RNG before training, which
  shifts the replay-sampling draws. So `free` is the honest baseline for any gated cell, and the
  plain baseline is NOT. This is worth ~0.002 at hidden width 400 and ~0.06 at width 5: the
  necessity GROWS as the baseline destabilises. Reading a small-net result against plain ER would
  have reported a spurious "gain control rescues small networks" win (+0.098) that is mostly
  instability.

THE DOUBLE-ZERO-INIT SADDLE.
  Stacking a zero-init module BEFORE the zero-init gate P makes dL/dP proportional to m = 0 AND
  dL/dmodule proportional to P = 0, so neither bootstraps off zero and the gate is frozen at parity
  for the entire run. Such a cell does not test the mechanism -- it trivially reproduces the
  baseline. `assert_dead_gate` / `assert_live_gate` turn that into a checked precondition instead of
  a silently-passing null. To ENGAGE a stacked gate, break the symmetry with `break_symmetry`: give
  the module's OUTPUT layer a normal init while keeping P zero-init (parity at step 0 preserved,
  but m != 0 so P can bootstrap).

THE PROBE.
  Task-decodability of m(x): can a linear map recover the task id from the modulatory code? This is
  the mechanistic "why" behind the pt7 negative -- the bio drivers probe at 0.21-0.52 versus pt6's
  learned task selector at 0.88, i.e. difficulty/novelty is not task identity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import EPS


# name -> (drivers, is_free, is_const)
def cell_spec(name):
    """Driver list + control flags for a cell name.

    `free`      : K=4 heads, NO bio target, trained end-to-end -> the RNG-matched dead-gate baseline.
    `5ht-const` : learned constant gate, no x-dependence -> the scale-degeneracy null.
    `all4`      : the canonical DA/ACh/NE/5HT composite.
    anything else: a single named driver.
    """
    if name == "free":       return (["free"] * 4, True, False)
    if name == "5ht-const":  return (["const"], False, True)
    if name == "all4":       return (["DA", "ACh", "NE", "5HT"], False, False)
    return ([name], False, False)


def probe(M, T, K, steps=300, lr=0.05):
    """Linear probe task-acc from m(x) in R^K (diagnostic: is the modulatory code task-decodable?)."""
    M = (M - M.mean(0)) / (M.std(0) + EPS)
    clf = nn.Linear(K, int(T.max().item()) + 1)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); F.cross_entropy(clf(M), T).backward(); opt.step()
    return (clf(M).argmax(1) == T).float().mean().item()


# ------------------------------- saddle guards -------------------------------
def gate_magnitude(per_layer):
    """Total |gate deviation| summed over layers -- the scalar the saddle guards key off."""
    return sum(abs(v) for v in per_layer.values())


def assert_dead_gate(per_layer, tol=1e-9, label="free"):
    """A control that is SUPPOSED to be inert must actually be inert.

    Fails loudly if the `free`-style control has somehow engaged, which would mean it is no longer a
    valid RNG-matched baseline. Note a FIXED RANDOM P breaks the saddle on the P side, so a
    fixed-projection `free` variant IS live -- do not assert deadness on that arm.
    """
    g = gate_magnitude(per_layer)
    if g > tol:
        raise AssertionError(
            f"{label}: expected an inert gate (double-zero-init saddle) but |g|={g:.3g}. "
            "This arm is no longer the RNG-matched baseline.")
    return g


def assert_live_gate(per_layer, tol=1e-9, label="mechanism"):
    """A cell under test must actually engage its gate.

    Guards the commonest false null in pt5-pt7: a stacked zero-init modulator pinned at |g| = 0 by
    the saddle reports exactly the baseline and looks like a clean negative, when in fact the
    mechanism never ran. If this fires, break the symmetry (see break_symmetry) and re-run before
    concluding anything.
    """
    g = gate_magnitude(per_layer)
    if g <= tol:
        raise AssertionError(
            f"{label}: gate is pinned at |g|={g:.3g} (double-zero-init saddle). The mechanism never "
            "engaged, so this result is the baseline, not a null. Break the symmetry and re-run.")
    return g


def break_symmetry(module, gain=1.0):
    """Restore a normal init on a zero-init module OUTPUT layer so it can bootstrap off the saddle.

    Use on the module that feeds the gate (heads / signal net / GRU), never on P itself: keeping P
    zero-init is what preserves gamma = 1 parity at step 0.
    """
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("break_symmetry: no nn.Linear found to re-initialise")
    nn.init.xavier_uniform_(last.weight, gain=gain)
    if last.bias is not None:
        nn.init.zeros_(last.bias)
    return last
