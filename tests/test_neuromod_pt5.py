"""pt5 (generalized driver system, Iteration 1 = disjoint task-id projection) unit tests.

Covers the SPEC's Implementation test list: OFF/empty-driver parity, disjoint coverage/disjointness,
shared frac, task_id one-hot dim, unknown-driver raise, gain form application, and the multi-layer
task-driven weight mask (wraps every listed layer; all-ones mask matches the base net = parity).
"""
import copy

import pytest
import torch
import torch.nn as nn

from prototype.model import MLP, ModulatedLinear
from prototype.neuromod import (
    DriverBank,
    GainDriverModulator,
    PlasticityDriverModulator,
    TaskIdOneHot,
    TaskWeightMaskMLP,
    build_disjoint_proj,
    build_shared_proj,
    gain_gamma,
    parse_drivers,
)

N_TASKS = 5


# --- drivers ---------------------------------------------------------------
def test_taskid_onehot_dim_and_value():
    """task_id=onehot driver dim == n_tasks and value() is the current one-hot (detached)."""
    d = TaskIdOneHot(N_TASKS)
    assert d.dim == N_TASKS
    d.set_task(2)
    v = d.value()
    assert v.shape == (N_TASKS,)
    assert torch.equal(v, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert not v.requires_grad


def test_driver_bank_dim_and_onehot():
    bank = DriverBank("task_id=onehot", N_TASKS)
    assert bank.dim == N_TASKS
    bank.set_task(3)
    assert torch.equal(bank.value(), torch.eye(N_TASKS)[3])


def test_empty_driver_spec_raises():
    """Empty driver string is not a valid bank (the OFF/legacy path is selected upstream instead)."""
    with pytest.raises(ValueError):
        DriverBank("", N_TASKS)


def test_unknown_driver_pair_raises():
    """Any pair other than task_id=onehot is a NotImplementedError stub (later-SPEC drivers)."""
    with pytest.raises(NotImplementedError):
        DriverBank("dopamine=phasic", N_TASKS)
    with pytest.raises(NotImplementedError):
        DriverBank("task_id=learned_embed", N_TASKS)


def test_parse_drivers():
    assert parse_drivers("task_id=onehot") == [("task_id", "onehot")]
    assert parse_drivers("") == []
    assert parse_drivers(" task_id = onehot ") == [("task_id", "onehot")]
    with pytest.raises(ValueError):
        parse_drivers("task_id")  # missing '='


# --- fixed projections -----------------------------------------------------
def test_disjoint_columns_disjoint_and_cover():
    """Every column has exactly one 1 (assigned to one task); columns cover all D elements."""
    D = 400
    P = build_disjoint_proj(N_TASKS, D, seed=0)
    assert P.shape == (N_TASKS, D)
    assert set(P.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(P.sum(dim=0), torch.ones(D))       # exactly one task per element
    # even partition: each task gets ~D/T elements (differ by at most 1)
    counts = P.sum(dim=1)
    assert counts.max() - counts.min() <= 1
    # per-task gates are disjoint and union to full coverage
    gates = [torch.eye(N_TASKS)[t] @ P for t in range(N_TASKS)]
    assert torch.equal(torch.stack(gates).sum(dim=0), torch.ones(D))


def test_disjoint_gate_is_binary_row_select():
    """z = e_t selects row t: the gate for task t is exactly P[t] and is binary."""
    D = 50
    P = build_disjoint_proj(N_TASKS, D, seed=1)
    for t in range(N_TASKS):
        gate = torch.eye(N_TASKS)[t] @ P
        assert torch.equal(gate, P[t])
        assert set(gate.unique().tolist()) <= {0.0, 1.0}


def test_shared_has_shared_frac_allones_columns():
    """~shared_frac of columns are all-ones (shared by every task); the rest are disjoint (sum 1)."""
    D = 400
    frac = 0.5
    P = build_shared_proj(N_TASKS, D, shared_frac=frac, seed=0)
    col_sums = P.sum(dim=0)
    n_shared = int((col_sums == N_TASKS).sum())
    n_private = int((col_sums == 1).sum())
    assert n_shared == round(D * frac)
    assert n_shared + n_private == D                       # every column is all-ones or disjoint
    assert set(P.unique().tolist()) <= {0.0, 1.0}


def test_shared_frac_extremes():
    D = 100
    all_shared = build_shared_proj(N_TASKS, D, shared_frac=1.0, seed=0)
    assert torch.equal(all_shared, torch.ones(N_TASKS, D))
    none_shared = build_shared_proj(N_TASKS, D, shared_frac=0.0, seed=0)
    assert torch.equal(none_shared.sum(dim=0), torch.ones(D))  # reduces to disjoint


# --- gain form -------------------------------------------------------------
def test_gain_form_fixed_and_learned():
    """Fixed P uses raw directly (binary gate); learned bounded01 -> sigmoid, unbounded -> 1+raw."""
    raw = torch.tensor([0.0, 1.0, 0.0, 1.0])
    # fixed: gate == raw regardless of nominal form
    assert torch.equal(gain_gamma(raw, fixed=True, form="unbounded"), raw)
    assert torch.equal(gain_gamma(raw, fixed=True, form="bounded01"), raw)
    # learned forms differ
    real = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.allclose(gain_gamma(real, fixed=False, form="unbounded"), 1.0 + real)
    assert torch.allclose(gain_gamma(real, fixed=False, form="bounded01"), torch.sigmoid(real))
    # unbounded gain is exactly 1.0 at raw=0 (vanilla init point)
    assert torch.allclose(gain_gamma(torch.zeros(3), fixed=False, form="unbounded"), torch.ones(3))


def test_gain_modulator_disjoint_zeroes_other_task_units():
    """Under a disjoint fixed gate, gain suppresses (zeros) the units not assigned to task t."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = GainDriverModulator(bank, hidden_dim=400, projection="disjoint", seed=0)
    mod.set_task(0)
    h = torch.ones(4, 400)
    out = mod.modulate(h, torch.zeros(4, 784), layer_idx=0)
    gate = torch.eye(N_TASKS)[0] @ mod.P_0
    assert torch.equal(out, gate.unsqueeze(0) * h)
    assert (out == 0).any() and (out == 1).any()           # some units off, some on


def test_plasticity_alphas_binary_and_partition():
    """Plasticity per-neuron alpha is a binary {0,1} gate (frozen vs plastic) under a fixed P."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = PlasticityDriverModulator(bank, hidden_dim=400, projection="disjoint", seed=0)
    mod.set_task(1)
    alphas = mod.compute_alphas()
    assert set(alphas.keys()) == {0, 1}
    for a in alphas.values():
        assert a.shape == (400,)
        assert set(a.unique().tolist()) <= {0.0, 1.0}
    # param_factors reuse maps per-neuron alpha to per-parameter multipliers for the [784,400,400,10] MLP
    factors = mod.param_factors(alphas)
    assert set(factors) == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight"}


# --- multi-layer task weight mask -----------------------------------------
def _make_task_wm(layers, projection="disjoint"):
    model = MLP()
    layer_dims = {i: (model.net[i].out_features, model.net[i].in_features) for i in layers}
    bank = DriverBank("task_id=onehot", N_TASKS)
    return model, TaskWeightMaskMLP(model, layer_dims, bank, projection=projection, seed=0)


def test_task_wm_wraps_only_listed_layers():
    """Every listed linear (incl. output head net.4) becomes ModulatedLinear; ReLUs untouched."""
    _, m = _make_task_wm([0, 2, 4])
    net = m.base.net
    for idx in (0, 2, 4):
        assert isinstance(net[idx], ModulatedLinear), f"net.{idx} not wrapped"
    assert isinstance(net[1], nn.ReLU) and isinstance(net[3], nn.ReLU)


def test_task_wm_hidden_only_layer_set():
    """The masked-loss condition masks net.0 + net.2 only; the head net.4 stays a plain Linear."""
    _, m = _make_task_wm([0, 2])
    net = m.base.net
    assert isinstance(net[0], ModulatedLinear) and isinstance(net[2], ModulatedLinear)
    assert isinstance(net[4], nn.Linear) and not isinstance(net[4], ModulatedLinear)


def test_task_wm_forward_shape_and_runs():
    _, m = _make_task_wm([0, 2, 4])
    m.set_task(0)
    out = m(torch.randn(16, 1, 28, 28))
    assert out.shape == (16, 10)
    assert torch.isfinite(out).all()


def test_task_wm_allones_mask_parity_with_base():
    """With an all-ones mask, M⊙W = W, so the wrapped net matches the (unwrapped) base MLP exactly."""
    ref = MLP()
    model = copy.deepcopy(ref)                      # wrapping mutates net in place; keep a clean ref
    layers = [0, 2, 4]
    layer_dims = {i: (model.net[i].out_features, model.net[i].in_features) for i in layers}
    bank = DriverBank("task_id=onehot", N_TASKS)
    m = TaskWeightMaskMLP(model, layer_dims, bank, projection="disjoint", seed=0)
    for idx in layers:                              # force every mask to all-ones
        getattr(m, f"P_{idx}").fill_(1.0)
    m.set_task(0)
    x = torch.randn(8, 1, 28, 28)
    assert torch.allclose(m(x), ref(x), atol=1e-6)


def test_task_wm_gate_freezes_other_task_synapses():
    """A disjoint mask gates the gradient at W: synapses not assigned to task t get zero grad."""
    _, m = _make_task_wm([2])
    m.set_task(0)
    x = torch.randn(8, 1, 28, 28)
    loss = m(x).sum()
    loss.backward()
    wgrad = m.base.net[2].weight.grad
    mask = (torch.eye(N_TASKS)[0] @ m.P_2).view(400, 400)
    # gradient is nonzero only where the mask is 1 (task-0 synapses)
    assert torch.equal((wgrad != 0), (wgrad != 0) & (mask == 1))


def test_learned_projection_not_implemented():
    """Iteration 3 (learned P) is deferred: constructing it must raise, not silently mis-run."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    with pytest.raises(NotImplementedError):
        GainDriverModulator(bank, projection="learned")
