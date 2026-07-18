"""pt5 COMPUTE + MEMORY cost of each retention lever, vs the cheapest gain cell (user-requested).

Baseline = gain per-neuron, SGD, standalone (no buffer). Measures, at ep=1 (1 seed), the marginal cost
of each axis: (a) Adam instead of SGD, (b) adding a buffer (ER, or the standalone meta-loss), (c) gain
per-synapse instead of per-neuron, (d) EWC (a regularisation retention mechanism, per-task Fisher+theta*).

TWO cost axes reported:
  - wall-time at ep=1 (indicative; MPS/CPU, 1 run).
  - TOTAL resident memory in float32 counts = model params (base MLP + modulator P) + optimizer state
    (SGD ~0; Adam = 2x the params it trains) + buffer (er_buffer_size x 784 image floats) + EWC state
    (per-task Fisher + theta* snapshot, EACH = 2x the base params; this repo's EWC is PER-TASK, so the
    state GROWS with the task count: (T-1)x2xP during the last task's training, Tx2xP after the final
    consolidation). Base MLP [784->400->400->10] = 478,410 params; buffer sample = 784 floats.

Key findings (see the printed table): Adam is ~free on time but ~3x the memory (2x moments); the buffer
costs compute + ~784k floats, not params; gain-synapse costs params (~590x the per-neuron P), not time;
EWC is the memory-heaviest AND task-count-scaling (~9-11x base here) AND fails class-IL.

Run: uv run python results/pt5_compute_memory.py   (redirect to results/pt5_compute_memory.log)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from prototype.configs import CLConfig
from prototype.model import MLP
from prototype.train import cl_train, _build_model, _pt5_gain_modulator_params

SEED, LR, EP, BUFFER = 42, 1e-3, 1, 1000
T = 5
IMAGE_FLOATS = 784                          # one MNIST sample stored in the buffer
BASE_PARAMS = sum(p.numel() for p in MLP().parameters())
SEQ = [(2 * t, 2 * t + 1) for t in range(T)]


def _gain_cfg(optimizer, masking, buffer, **kw):
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer=optimizer, output_masking=masking,
                    er_buffer_size=buffer, use_neuromod=True, neuromod_drivers="task_id=onehot",
                    neuromod_context="none", neuromod_target="activation", neuromod_projection="learned",
                    **kw)


def p_params(cfg):
    m = _build_model(cfg, torch.device("cpu"), n_tasks=T, sequence=SEQ)
    return sum(p.numel() for p in _pt5_gain_modulator_params(m))


def timeit(cfg, method):
    t = time.time()
    acc, _ = cl_train(cfg, method, no_wandb=True, sequence=None)
    return acc, time.time() - t


# each row: (label, cfg, method, P params, adam-trained params, buffer samples, ewc?)
GAIN_NEU = dict(neuromod_granularity="neuron", neuromod_gain_layers="0,2")
GAIN_SYN = dict(neuromod_granularity="synapse", neuromod_mask_layers="0,2", neuromod_gain_layers="0,2")

ROWS = []
# baseline: gain-neuron SGD standalone, no buffer (learned P trained by the MAIN SGD optimizer)
c = _gain_cfg("sgd", "loss", BUFFER, **GAIN_NEU)
P = p_params(c)
ROWS.append(("baseline gain-neu SGD standalone", c, "naive", P, 0, 0, False))
# +Adam: main optimizer is Adam over (base + P)
c = _gain_cfg("adam", "loss", BUFFER, **GAIN_NEU)
ROWS.append(("+Adam", c, "naive", P, BASE_PARAMS + P, 0, False))
# +buffer, standalone meta-loss: main SGD trains base; a SEPARATE Adam meta-opt trains P; buffer used
c = _gain_cfg("sgd", "loss", BUFFER, neuromod_meta_replay=True, **GAIN_NEU)
ROWS.append(("+buffer standalone meta (SGD)", c, "naive", P, P, BUFFER, False))
# +buffer, ER: main SGD trains base+P; buffer used for replay
c = _gain_cfg("sgd", "none", BUFFER, **GAIN_NEU)
ROWS.append(("+buffer ER (SGD)", c, "er", P, 0, BUFFER, False))
# gain-synapse SGD standalone, no buffer
c = _gain_cfg("sgd", "loss", BUFFER, **GAIN_SYN)
Psyn = p_params(c)
ROWS.append(("gain-synapse SGD standalone", c, "naive", Psyn, 0, 0, False))
# EWC SGD standalone (plain MLP, no neuromod, no buffer)
c = CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer="sgd")
ROWS.append(("EWC SGD standalone", c, "ewc", 0, 0, 0, True))


def total_floats(P, adam_params, buf_samples, ewc):
    model = BASE_PARAMS + P
    optim = 2 * adam_params
    buffer = buf_samples * IMAGE_FLOATS
    if ewc:                                    # per-task Fisher + theta*, each 2x base; peak at end = T
        ewc_state = T * 2 * BASE_PARAMS
        return model + optim + buffer + ewc_state, ewc_state
    return model + optim + buffer, 0


print(f"base MLP params = {BASE_PARAMS:,} | buffer sample = {IMAGE_FLOATS} floats | T = {T} tasks\n")
measured = []
for label, cfg, method, P, adam_p, buf, ewc in ROWS:
    acc, secs = timeit(cfg, method)
    tot, ewc_state = total_floats(P, adam_p, buf, ewc)
    measured.append((label, acc, secs, P, adam_p, buf, ewc, tot, ewc_state))
    print(f">>> {label:34s} acc={acc:.4f}  time={secs:5.1f}s  Pparams={P:>9,}  total_floats={tot:>10,}",
          flush=True)

base_secs = measured[0][2]
base_tot = measured[0][7]
print("\n\n" + "=" * 104)
print("pt5 COMPUTE + MEMORY (ep=1, 1 seed) — vs baseline gain-neuron SGD standalone no-buffer")
print("=" * 104)
print(f"  {'config':34s} {'wall':>7s} {'xtime':>6s} | {'model+P':>10s} {'adam st':>9s} "
      f"{'buffer':>9s} {'ewc st':>9s} {'TOTAL':>11s} {'xmem':>6s}")
for label, acc, secs, P, adam_p, buf, ewc, tot, ewc_state in measured:
    print(f"  {label:34s} {secs:6.1f}s {secs/base_secs:5.2f}x | {BASE_PARAMS + P:>10,} "
          f"{2*adam_p:>9,} {buf*IMAGE_FLOATS:>9,} {ewc_state:>9,} {tot:>11,} {tot/base_tot:5.1f}x")
print("\nMemory in float32 counts (x4 bytes for MB). Adam state = 2x the params it trains. Buffer = "
      "er_buffer_size x 784. EWC (PER-TASK) state = T x 2 x base params at the final consolidation "
      "(grows LINEARLY with task count; (T-1)x2xP during the last task). Reading: Adam ~free on time / "
      "~3x memory; buffer costs compute + ~784k floats (not params); gain-synapse costs params (~590x P) "
      "not time; EWC is the memory-heaviest and the only task-count-scaling row, and fails class-IL. "
      "1 seed, ep=1; wall-times indicative.")
