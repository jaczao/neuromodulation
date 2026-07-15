"""pt5 (generalized driver system, Iteration 1 = disjoint task-id projection) unit tests.

Covers the SPEC's Implementation test list: OFF/empty-driver parity, disjoint coverage/disjointness,
shared frac, task_id one-hot dim, unknown-driver raise, gain form application, and the multi-layer
task-driven weight mask (wraps every listed layer; all-ones mask matches the base net = parity).
"""
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype.model import MLP, ModulatedLinear
from prototype.neuromod import (
    SOFTPLUS_PARITY_BIAS,
    DriverBank,
    GainDriverModulator,
    ModulatedMLP,
    PlasticityDriverModulator,
    SynapsePlasticityDriverModulator,
    TaskIdOneHot,
    TaskWeightMaskMLP,
    build_disjoint_proj,
    build_proj,
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
    # fixed: gain == raw regardless of nominal form
    assert torch.equal(gain_gamma(raw, fixed=True, form="unbounded"), raw)
    assert torch.equal(gain_gamma(raw, fixed=True, form="bounded01"), raw)
    assert torch.equal(gain_gamma(raw, fixed=True, form="positive"), raw)
    # learned forms differ
    real = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.allclose(gain_gamma(real, fixed=False, form="unbounded"), 1.0 + real)
    assert torch.allclose(gain_gamma(real, fixed=False, form="bounded01"), torch.sigmoid(real))
    # unbounded gain is exactly 1.0 at raw=0 (vanilla init point)
    assert torch.allclose(gain_gamma(torch.zeros(3), fixed=False, form="unbounded"), torch.ones(3))


def test_gain_form_positive():
    """`positive` = softplus(raw + ln(e-1)): strictly positive (never inverts), unbounded above,
    and offset so it shares `unbounded`'s neutral 1.0 init at raw=0 (zero-init learned P = parity)."""
    real = torch.tensor([-6.0, -2.0, 0.0, 2.0, 6.0])
    gamma = gain_gamma(real, fixed=False, form="positive")
    assert torch.allclose(gamma, F.softplus(real + SOFTPLUS_PARITY_BIAS))
    assert (gamma > 0).all()                                  # never inverts, unlike unbounded
    assert torch.allclose(gain_gamma(torch.zeros(3), fixed=False, form="positive"), torch.ones(3))
    assert gamma[-1] > 1.0                                    # amplifies above 1, unlike bounded01
    # asymptotic-only zero: it cannot hard-freeze a unit the way a fixed {0,1} gain does
    assert 0.0 < gain_gamma(torch.tensor([-30.0]), fixed=False, form="positive").item() < 1e-6


def test_gain_modulator_disjoint_zeroes_other_task_units():
    """Under a disjoint fixed gate, gain suppresses (zeros) the units not assigned to task t."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = GainDriverModulator(bank, hidden_dim=400, projection="disjoint", seed=0)
    mod.set_task(0)
    h = torch.ones(4, 400)
    out = mod.modulate(h, torch.zeros(4, 784), layer_idx=0)
    gate = torch.eye(N_TASKS)[0] @ mod.P_h0
    assert torch.equal(out, gate.unsqueeze(0) * h)
    assert (out == 0).any() and (out == 1).any()           # some units off, some on


def test_gain_modulator_gate_layers_selective():
    """gate_layers selects which activations are gated; unselected ones pass through unchanged."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = GainDriverModulator(bank, gate_layers=(2,), projection="disjoint", seed=0)  # h1 only
    mod.set_task(0)
    h = torch.ones(4, 400)
    assert torch.equal(mod.modulate(h, torch.zeros(4, 784), layer_idx=0), h)     # h0 untouched
    assert not torch.equal(mod.modulate(h, torch.zeros(4, 784), layer_idx=1), h)  # h1 gated
    assert torch.equal(mod.modulate_logits(torch.ones(4, 10), torch.zeros(4, 784)), torch.ones(4, 10))  # no output gate


def test_gain_modulator_output_gate():
    """gate_layers with 4 gates the 10 output logits (per-class gain); disjoint P keeps task-t classes."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = GainDriverModulator(bank, gate_layers=(0, 2, 4), projection="disjoint", seed=0)
    mod.set_task(0)
    logits = torch.ones(4, 10)
    out = mod.modulate_logits(logits, torch.zeros(4, 784))
    gate = torch.eye(N_TASKS)[0] @ mod.P_out
    assert torch.equal(out, gate.unsqueeze(0) * logits)
    assert (out == 0).any()                                 # non-task classes suppressed under disjoint P


def test_gain_bad_gate_layers_raise():
    bank = DriverBank("task_id=onehot", N_TASKS)
    with pytest.raises(ValueError):
        GainDriverModulator(bank, gate_layers=(0, 3))       # 3 is not a valid net linear index


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


# --- learned projection (Iteration 3): P is a trainable nn.Parameter, gate differentiable --------
def test_build_proj_fixed_vs_learned():
    """build_proj returns a binary buffer (learned=False) for fixed projections and a zero-init real
    tensor (learned=True) for the learned projection; an unknown projection raises."""
    P_fix, learned = build_proj("disjoint", N_TASKS, 40, 0.5, seed=0)
    assert learned is False
    assert set(P_fix.unique().tolist()) <= {0.0, 1.0}
    P_learn, learned = build_proj("learned", N_TASKS, 40, 0.5, seed=0)
    assert learned is True
    assert torch.equal(P_learn, torch.zeros(N_TASKS, 40))     # zero-init -> raw=0 -> neutral gate
    with pytest.raises(ValueError):
        build_proj("bogus", N_TASKS, 40, 0.5, seed=0)


def test_gain_learned_projection_trainable_and_parity_at_init():
    """Learned gain: P_h* are trainable Parameters (in .parameters()); the unbounded form is exactly
    parity (gain 1.0) at zero-init, and the gradient reaches only the current task's row."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = GainDriverModulator(bank, hidden_dim=8, projection="learned", gain_form="unbounded")
    assert isinstance(mod.P_h0, nn.Parameter) and mod.P_h0.requires_grad
    assert any(p is mod.P_h0 for p in mod.parameters())
    mod.set_task(2)
    h = torch.randn(4, 8)
    out = mod.modulate(h, torch.zeros(4, 784), layer_idx=0)
    assert torch.allclose(out, h, atol=1e-6)                  # gain = 1 + 0 = 1 -> identity (parity)
    out.sum().backward()
    grad = mod.P_h0.grad
    assert grad[2].abs().sum() > 0                            # task-2 row got a gradient
    assert grad[torch.arange(N_TASKS) != 2].abs().sum() == 0  # every other task row is untouched


def test_gain_learned_bounded01_init_is_half():
    """Learned bounded01 gain starts at sigmoid(0) = 0.5 (neutral suppress-only gate)."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = GainDriverModulator(bank, hidden_dim=8, projection="learned", gain_form="bounded01")
    mod.set_task(0)
    out = mod.modulate(torch.ones(2, 8), torch.zeros(2, 784), layer_idx=0)
    assert torch.allclose(out, torch.full((2, 8), 0.5), atol=1e-6)


def test_plasticity_learned_alphas_in_unit_interval():
    """Learned per-neuron plasticity: P_l trainable; alpha = sigmoid(raw) in (0,1), 0.5 at init."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = PlasticityDriverModulator(bank, hidden_dim=8, projection="learned")
    assert isinstance(mod.P_0, nn.Parameter)
    mod.set_task(1)
    a = mod.compute_alphas()[0]
    assert a.shape == (8,) and a.requires_grad                # differentiable in P
    assert torch.allclose(a, torch.full((8,), 0.5), atol=1e-6)  # sigmoid(0)


def test_synapse_plasticity_learned_gate_differentiable():
    """Learned per-synapse plasticity: weight/bias gates are sigmoid (not binary) and reach P by
    autograd (so a lookahead/meta loop can train it), on only the current task's row."""
    model = MLP()
    layer_dims = {2: (model.net[2].out_features, model.net[2].in_features)}
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = SynapsePlasticityDriverModulator(bank, layer_dims, projection="learned", seed=0, modulate_bias=True)
    assert isinstance(mod.P_2, nn.Parameter) and isinstance(mod.P_bias_2, nn.Parameter)
    mod.set_task(0)
    g = mod.weight_grad_masks()["net.2.weight"]
    assert g.requires_grad
    assert not (set(g.detach().unique().tolist()) <= {0.0, 1.0})   # squashed, not {0,1}
    assert torch.allclose(g.detach(), torch.full_like(g.detach(), 0.5), atol=1e-6)  # sigmoid(0)
    g.sum().backward()                                             # a gate-only (meta) loss trains P
    assert mod.P_2.grad[0].abs().sum() > 0
    assert mod.P_2.grad[torch.arange(N_TASKS) != 0].abs().sum() == 0


def test_task_wm_learned_mask_trains_via_forward():
    """Learned weight_mask: P_l trainable; mask = sigmoid(raw) in (0,1); the MAIN loss puts a gradient
    on P through the forward (M⊙W), on only the current task's row."""
    base = MLP()
    layer_dims = {2: (base.net[2].out_features, base.net[2].in_features)}
    bank = DriverBank("task_id=onehot", N_TASKS)
    m = TaskWeightMaskMLP(base, layer_dims, bank, projection="learned", seed=0, gain_form="bounded01")
    assert isinstance(m.P_2, nn.Parameter) and any(p is m.P_2 for p in m.parameters())
    m.set_task(3)
    m(torch.randn(8, 1, 28, 28)).sum().backward()
    assert m.P_2.grad[3].abs().sum() > 0                          # task-3 row trained by the main loss
    assert m.P_2.grad[torch.arange(N_TASKS) != 3].abs().sum() == 0


def test_plasticity_learned_init_gate_bias():
    """pt5 iter3 init-bias: init_gate sets the LEARNED plasticity gate at zero-init P. 0.5 ->
    sigmoid(0)=0.5 (parity, reproduces iter3); 0.95 -> ~0.95 via a logit bias. Fixed P ignores it."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod0 = PlasticityDriverModulator(bank, hidden_dim=8, projection="learned", init_gate=0.5)
    mod0.set_task(0)
    assert torch.allclose(mod0.compute_alphas()[0], torch.full((8,), 0.5), atol=1e-6)
    mod = PlasticityDriverModulator(bank, hidden_dim=8, projection="learned", init_gate=0.95)
    mod.set_task(0)
    assert torch.allclose(mod.compute_alphas()[0], torch.full((8,), 0.95), atol=1e-4)
    fx = PlasticityDriverModulator(bank, hidden_dim=8, projection="disjoint", init_gate=0.95)
    fx.set_task(0)
    assert set(fx.compute_alphas()[0].unique().tolist()) <= {0.0, 1.0}   # fixed stays binary


def test_synapse_plasticity_learned_init_gate_bias():
    """init_gate also sets the per-synapse learned plasticity gate (0.95 -> ~0.95 at zero-init)."""
    model = MLP()
    layer_dims = {2: (model.net[2].out_features, model.net[2].in_features)}
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = SynapsePlasticityDriverModulator(bank, layer_dims, projection="learned", seed=0, init_gate=0.95)
    mod.set_task(0)
    g = mod.weight_grad_masks()["net.2.weight"].detach()
    assert torch.allclose(g, torch.full_like(g, 0.95), atol=1e-4)


def test_gate_l1_differentiable_gain_plasticity_synapse():
    """pt5 iter3 sparsity reg: gate_l1() is a scalar differentiable in P (only the current task's
    row), so a L1 penalty trains P toward sparse gates. Covers gain and both plasticity variants."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    g = GainDriverModulator(bank, hidden_dim=8, projection="learned", gain_form="unbounded")
    g.set_task(0)
    lg = g.gate_l1()
    assert lg.ndim == 0 and lg.requires_grad
    lg.backward()
    assert g.P_h0.grad[0].abs().sum() > 0
    assert g.P_h0.grad[torch.arange(N_TASKS) != 0].abs().sum() == 0     # only current task's row
    p = PlasticityDriverModulator(bank, hidden_dim=8, projection="learned")
    p.set_task(1)
    lp = p.gate_l1()
    assert lp.ndim == 0 and lp.requires_grad
    lp.backward()
    assert p.P_1.grad[1].abs().sum() > 0
    model = MLP()
    sp = SynapsePlasticityDriverModulator(
        bank, {2: (model.net[2].out_features, model.net[2].in_features)}, projection="learned", seed=0)
    sp.set_task(2)
    assert sp.gate_l1().requires_grad


def test_modulated_mlp_gate_l1_delegates():
    """ModulatedMLP.gate_l1 delegates to its gain modulator (used to add the sparsity penalty to the
    MAIN loss for the forward gain target)."""
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = GainDriverModulator(bank, hidden_dim=400, projection="learned", gain_form="unbounded")
    mm = ModulatedMLP(MLP(), mod)
    mm.set_task(0)
    assert torch.allclose(mm.gate_l1(), mod.gate_l1())


def test_task_wm_learned_gain_unbounded_parity_at_init():
    """Learned per-synapse gain (unbounded) is parity at init: gain = 1 + 0 = 1 -> M⊙W = W."""
    ref = MLP()
    base = copy.deepcopy(ref)
    layer_dims = {i: (base.net[i].out_features, base.net[i].in_features) for i in (0, 2)}
    bank = DriverBank("task_id=onehot", N_TASKS)
    m = TaskWeightMaskMLP(base, layer_dims, bank, projection="learned", seed=0, gain_form="unbounded")
    m.set_task(0)
    x = torch.randn(8, 1, 28, 28)
    assert torch.allclose(m(x), ref(x), atol=1e-6)


# --- per-neuron plasticity scope (item 1) ----------------------------------
def _plast_mod(hidden_dim=4):
    bank = DriverBank("task_id=onehot", N_TASKS)
    return PlasticityDriverModulator(bank, hidden_dim=hidden_dim, projection="disjoint", seed=0)


# hand-built alphas so the checks are independent of the projection layout
_A0 = torch.tensor([1.0, 0.0, 1.0, 0.0])
_A1 = torch.tensor([0.0, 1.0, 1.0, 0.0])
_ALPHAS = {0: _A0, 1: _A1}


def test_param_factors_scope_both_is_in_union_out():
    """scope='both' (default) gates each hidden unit's incoming AND outgoing weights (legacy coupling)."""
    f = _plast_mod().param_factors(_ALPHAS, scope="both")
    assert set(f) == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight"}
    assert torch.equal(f["net.0.weight"], _A0.unsqueeze(1))
    assert torch.equal(f["net.2.weight"], _A1.unsqueeze(1) * _A0.unsqueeze(0))   # rows·cols
    assert torch.equal(f["net.4.weight"], _A1.unsqueeze(0))
    # default arg == explicit 'both'
    assert set(_plast_mod().param_factors(_ALPHAS)) == set(f)


def test_param_factors_scope_in_only_incoming():
    """scope='in' gates only incoming weights/biases; the output head net.4 is left fully plastic."""
    f = _plast_mod().param_factors(_ALPHAS, scope="in")
    assert set(f) == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"}
    assert "net.4.weight" not in f                              # h1 outgoing untouched
    assert torch.equal(f["net.0.weight"], _A0.unsqueeze(1))     # h0 incoming rows
    assert torch.equal(f["net.2.weight"], _A1.unsqueeze(1))     # h1 incoming rows only (no a0 cols)
    assert torch.equal(f["net.2.bias"], _A1)


def test_param_factors_scope_out_only_outgoing():
    """scope='out' gates only outgoing weights; net.0 and the hidden biases stay fully plastic."""
    f = _plast_mod().param_factors(_ALPHAS, scope="out")
    assert set(f) == {"net.2.weight", "net.4.weight"}
    assert "net.0.weight" not in f and "net.2.bias" not in f    # incoming side untouched
    assert torch.equal(f["net.2.weight"], _A0.unsqueeze(0))     # h0 outgoing cols only
    assert torch.equal(f["net.4.weight"], _A1.unsqueeze(0))     # h1 outgoing cols


def test_param_factors_unknown_scope_raises():
    with pytest.raises(ValueError):
        _plast_mod().param_factors(_ALPHAS, scope="sideways")


def test_param_factors_layers_subset_out():
    """Layer selection: out-scope on {2,4} gates only h0/h1 outgoing weights; net.0 is left alone."""
    f = _plast_mod().param_factors(_ALPHAS, scope="out", layers=(2, 4))
    assert set(f) == {"net.2.weight", "net.4.weight"}
    assert torch.equal(f["net.2.weight"], _A0.unsqueeze(0))
    assert torch.equal(f["net.4.weight"], _A1.unsqueeze(0))


def test_param_factors_single_layer_out_only_head():
    """out-scope on {4} alone gates just the output head's incoming-from-h1 columns."""
    f = _plast_mod().param_factors(_ALPHAS, scope="out", layers=(4,))
    assert set(f) == {"net.4.weight"}
    assert torch.equal(f["net.4.weight"], _A1.unsqueeze(0))


def test_param_factors_invalid_side_is_noop():
    """A (layer, side) with no α contributes nothing: net.0 has no 'out', net.4 has no 'in'."""
    assert _plast_mod().param_factors(_ALPHAS, scope="out", layers=(0,)) == {}   # net.0 out: no input α
    assert _plast_mod().param_factors(_ALPHAS, scope="in", layers=(4,)) == {}    # net.4 in: no output α


def test_param_factors_in_on_zero_out_on_two():
    """A global scope can't mix sides across layers: 'both' on {0,2} still adds net.2 rows (in)."""
    f = _plast_mod().param_factors(_ALPHAS, scope="both", layers=(0, 2))
    assert set(f) == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"}
    assert torch.equal(f["net.2.weight"], _A1.unsqueeze(1) * _A0.unsqueeze(0))    # net.2 gets both sides


# --- per-synapse plasticity (item 2) ---------------------------------------
def _syn_plast(layers=(0, 2, 4)):
    model = MLP()
    layer_dims = {i: (model.net[i].out_features, model.net[i].in_features) for i in layers}
    bank = DriverBank("task_id=onehot", N_TASKS)
    return SynapsePlasticityDriverModulator(bank, layer_dims, projection="disjoint", seed=0)


def test_synapse_plasticity_masks_shape_binary_and_keys():
    mod = _syn_plast([0, 2, 4])
    mod.set_task(0)
    masks = mod.weight_grad_masks()
    assert set(masks) == {"net.0.weight", "net.2.weight", "net.4.weight"}   # weights only, no biases
    assert masks["net.0.weight"].shape == (400, 784)
    assert masks["net.2.weight"].shape == (400, 400)
    assert masks["net.4.weight"].shape == (10, 400)
    for m in masks.values():
        assert set(m.unique().tolist()) <= {0.0, 1.0}


def test_synapse_plasticity_disjoint_partitions_synapses():
    """Per-synapse gates are disjoint across tasks and cover every synapse (union of {0,1} gates = all-ones)."""
    mod = _syn_plast([4])
    gates = []
    for t in range(N_TASKS):
        mod.set_task(t)
        gates.append(mod.weight_grad_masks()["net.4.weight"])
    assert torch.equal(torch.stack(gates).sum(dim=0), torch.ones(10, 400))


def test_synapse_plasticity_requires_layers():
    bank = DriverBank("task_id=onehot", N_TASKS)
    with pytest.raises(ValueError):
        SynapsePlasticityDriverModulator(bank, {}, projection="disjoint", seed=0)


# --- per-synapse gain (item 3) ---------------------------------------------
def _wm(base, layers, gain_form="unbounded"):
    layer_dims = {i: (base.net[i].out_features, base.net[i].in_features) for i in layers}
    bank = DriverBank("task_id=onehot", N_TASKS)
    return TaskWeightMaskMLP(
        base, layer_dims, bank, projection="disjoint", seed=0, gain_form=gain_form
    )


def test_synapse_gain_equals_weight_mask_under_fixed_P():
    """Per-synapse gain coincides with weight_mask (gain_form='bounded01') under a fixed binary P.

    gain_gamma(raw, fixed=True) returns raw ({0,1}) for EVERY form, so with the same seed and base
    weights the forward paths are numerically identical. They diverge only under a learned P (Iter 3).
    """
    ref = MLP()
    m_mask = _wm(copy.deepcopy(ref), [0, 2], gain_form="bounded01")
    m_gain = _wm(copy.deepcopy(ref), [0, 2], gain_form="unbounded")
    m_pos = _wm(copy.deepcopy(ref), [0, 2], gain_form="positive")
    for m in (m_mask, m_gain, m_pos):
        m.set_task(1)
    x = torch.randn(8, 1, 28, 28)
    assert torch.allclose(m_mask(x), m_gain(x), atol=1e-6)
    assert torch.allclose(m_mask(x), m_pos(x), atol=1e-6)


def test_synapse_gain_forward_runs_and_freezes_grad():
    """Per-synapse gain runs and, like the mask, gates the gradient at W (task-t synapses only)."""
    m = _wm(MLP(), [2])
    m.set_task(0)
    x = torch.randn(8, 1, 28, 28)
    out = m(x)
    assert out.shape == (8, 10) and torch.isfinite(out).all()
    out.sum().backward()
    wgrad = m.base.net[2].weight.grad
    gain = (torch.eye(N_TASKS)[0] @ m.P_2).view(400, 400)
    assert torch.equal((wgrad != 0), (wgrad != 0) & (gain == 1))


def test_bad_gain_form_raises():
    """Rejected at construction, even under a fixed P where gain_gamma short-circuits on `fixed`."""
    with pytest.raises(ValueError):
        _wm(MLP(), [2], gain_form="scale")
    with pytest.raises(ValueError):
        GainDriverModulator(DriverBank("task_id=onehot", N_TASKS), hidden_dim=8, gain_form="scale")


# --- optional per-neuron bias modulation (toggle) --------------------------
def test_modulated_linear_bias_mask():
    """bias_mask=None -> identical to nn.Linear (parity); a per-neuron bias_mask gates the bias."""
    lin = ModulatedLinear(5, 3)
    x = torch.randn(4, 5)
    assert torch.allclose(lin(x), nn.functional.linear(x, lin.weight, lin.bias), atol=1e-6)  # parity
    zero = torch.zeros(3)
    assert torch.allclose(lin(x, bias_mask=zero), nn.functional.linear(x, lin.weight, None), atol=1e-6)
    bm = torch.tensor([1.0, 0.0, 1.0])
    assert torch.allclose(lin(x, bias_mask=bm), nn.functional.linear(x, lin.weight, bm * lin.bias), atol=1e-6)


def test_synapse_plasticity_bias_masks_when_enabled():
    """modulate_bias=True adds per-neuron {net.<l>.bias: (d_out,)} grad gates, disjoint across tasks."""
    model = MLP()
    layer_dims = {i: (model.net[i].out_features, model.net[i].in_features) for i in (0, 2, 4)}
    bank = DriverBank("task_id=onehot", N_TASKS)
    mod = SynapsePlasticityDriverModulator(bank, layer_dims, projection="disjoint", seed=0, modulate_bias=True)
    mod.set_task(0)
    masks = mod.weight_grad_masks()
    assert set(masks) == {"net.0.weight", "net.0.bias", "net.2.weight",
                          "net.2.bias", "net.4.weight", "net.4.bias"}
    assert masks["net.0.bias"].shape == (400,) and masks["net.4.bias"].shape == (10,)
    for k in ("net.0.bias", "net.2.bias", "net.4.bias"):
        assert set(masks[k].unique().tolist()) <= {0.0, 1.0}
    # bias projection is independent of the weight projection (own seed namespace), not aliased
    assert not torch.equal(mod.P_bias_4, mod.P_4[:, :10])
    gates = []
    for t in range(N_TASKS):
        mod.set_task(t)
        gates.append(mod.weight_grad_masks()["net.4.bias"])
    assert torch.equal(torch.stack(gates).sum(dim=0), torch.ones(10))   # disjoint partition of neurons


def test_task_wm_no_bias_proj_by_default():
    """Default (modulate_bias off): no per-neuron bias projection; biases stay unmodulated (parity)."""
    _, m = _make_task_wm([0, 2, 4])
    assert m.modulate_bias is False
    for idx in (0, 2, 4):
        assert not hasattr(m, f"P_bias_{idx}")


def test_task_wm_modulate_bias_gates_and_freezes_bias():
    """modulate_bias=True registers a per-neuron P_bias per layer and gates the bias in the forward,
    so (like the weight gate) a non-owned neuron's bias is both suppressed and frozen (zero grad)."""
    base = MLP()
    layer_dims = {i: (base.net[i].out_features, base.net[i].in_features) for i in (4,)}
    bank = DriverBank("task_id=onehot", N_TASKS)
    m = TaskWeightMaskMLP(base, layer_dims, bank, projection="disjoint", seed=0, modulate_bias=True)
    assert hasattr(m, "P_bias_4") and m.P_bias_4.shape == (N_TASKS, 10)
    m.set_task(0)
    x = torch.randn(8, 1, 28, 28)
    m(x).sum().backward()
    bgrad = m.base.net[4].bias.grad
    bgate = torch.eye(N_TASKS)[0] @ m.P_bias_4              # (10,) binary task-0 bias gate
    assert (bgate == 0).any()                              # some biases are gated off (disjoint / 5 tasks)
    assert torch.equal((bgrad != 0), (bgrad != 0) & (bgate == 1))   # frozen where β=0
