"""Tests for the extracted cross-cutting core (neurocore/).

Fast by construction — no training. The expensive check that the extraction is FAITHFUL to the
frozen ledger lives in neurocore/verify_anchors.py (minutes, run manually), not here.
"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurocore import cost as C
from neurocore import ledger as L
from neurocore import projections as P
from neurocore import tuned as T
from neurocore.controls import (assert_dead_gate, assert_live_gate, break_symmetry, cell_spec,
                                gate_magnitude)
from neurocore.gates import (DEFAULT_DIMS, GAIN_FORMS, GateDims, Heads, ModulatorHead, NeuronGate,
                             SynapseGate, check_gain_form, gain_gamma, gate_K, gate_l1, make_gate)
from neurocore.signals import (NEDriver, Signals, dataset_mean, entropy, per_sample_ce_plain,
                               per_sample_masked_ce)
from neurocore.utils import DEV, rng_frozen


# ------------------------------------------------------------------ projections
def test_disjoint_projection_is_a_partition():
    Pm = P.build_disjoint_proj(5, 810, seed=0)
    assert Pm.shape == (5, 810)
    assert torch.all(Pm.sum(0) == 1)                      # every column owned by exactly one task
    counts = Pm.sum(1)
    assert counts.max() - counts.min() <= 1               # balanced to within one element
    assert set(Pm.unique().tolist()) <= {0.0, 1.0}


def test_shared_projection_has_the_requested_shared_fraction():
    Pm = P.build_shared_proj(5, 800, shared_frac=0.5, seed=0)
    shared = (Pm.sum(0) == 5).sum().item()
    assert shared == 400
    private = (Pm.sum(0) == 1).sum().item()
    assert shared + private == 800


def test_learned_projection_is_zero_init_and_trainable():
    Pm, learned = P.build_proj("learned", 5, 32, 0.5, 0)
    assert learned and torch.all(Pm == 0)                 # zero-init => neutral gate at step 0
    fixed, is_learned = P.build_proj("disjoint", 5, 32, 0.5, 0)
    assert not is_learned and set(fixed.unique().tolist()) <= {0.0, 1.0}


def test_register_proj_buffer_vs_parameter():
    m = nn.Module()
    P.register_proj(m, "P_fixed", "disjoint", 5, 16, 0.5, 0)
    P.register_proj(m, "P_learned", "learned", 5, 16, 0.5, 0)
    names = dict(m.named_parameters())
    assert "P_learned" in names and "P_fixed" not in names
    assert "P_fixed" in dict(m.named_buffers())


def test_build_fixed_proj_rejects_learned_and_unknown():
    with pytest.raises(NotImplementedError):
        P.build_fixed_proj("learned", 5, 8, 0.5, 0)
    with pytest.raises(ValueError):
        P.build_fixed_proj("nonsense", 5, 8, 0.5, 0)


def test_task_id_at_eval_label_travels_with_the_fixed_projections():
    # The label the brief requires stay attached: legitimate in task-IL (XdG), an ORACLE in class-IL.
    assert P.requires_task_id_at_eval("disjoint")
    assert P.requires_task_id_at_eval("shared")
    assert not P.requires_task_id_at_eval("learned")


def test_gate_logit_bias_inverts_sigmoid():
    for v in (0.1, 0.5, 0.99):
        assert math.isclose(1 / (1 + math.exp(-P._gate_logit_bias(v))), v, rel_tol=1e-6)


# ------------------------------------------------------------------ gain forms
def test_init_parity_semantics_of_each_gain_form():
    raw = torch.zeros(4)
    # unbounded and positive are EXACTLY 1.0 at raw=0 -> zero-init P starts at parity.
    assert torch.allclose(gain_gamma(raw, fixed=False, form="unbounded"), torch.ones(4))
    assert torch.allclose(gain_gamma(raw, fixed=False, form="positive"), torch.ones(4), atol=1e-6)
    # bounded01 is 0.5 -> NOT parity; it halves every gated activation from step one.
    assert torch.allclose(gain_gamma(raw, fixed=False, form="bounded01"), torch.full((4,), 0.5))


def test_fixed_projection_collapses_every_gain_form_to_raw():
    raw = torch.tensor([0.0, 1.0, 0.0, 1.0])
    for form in GAIN_FORMS:
        assert torch.allclose(gain_gamma(raw, fixed=True, form=form), raw)


def test_positive_form_never_inverts_but_unbounded_does():
    raw = torch.tensor([-5.0])
    assert gain_gamma(raw, fixed=False, form="positive").item() > 0
    assert gain_gamma(raw, fixed=False, form="unbounded").item() < 0


def test_check_gain_form_rejects_typos():
    for f in GAIN_FORMS:
        assert check_gain_form(f) == f
    with pytest.raises(ValueError):
        check_gain_form("unbounded ")


def test_gate_l1_is_mean_abs():
    g = torch.tensor([[1.0, -3.0], [0.0, 2.0]])
    assert math.isclose(gate_l1(g).item(), 1.5)


# ------------------------------------------------------------------ gates
class _TinyNet(nn.Module):
    """Satisfies the backbone contract plain(x) -> (logits, features)."""
    def __init__(self, h0=400, h1=400, out=10):
        super().__init__()
        self.l0 = nn.Linear(784, h0); self.l1 = nn.Linear(h0, h1); self.l2 = nn.Linear(h1, out)

    def plain(self, x):
        x = x.view(x.size(0), -1)
        a = F.relu(self.l0(x)); b = F.relu(self.l1(a))
        return self.l2(b), b


def test_neuron_gate_is_parity_at_zero_init():
    torch.manual_seed(0)
    net, g = _TinyNet(), NeuronGate(4, None)
    x = torch.randn(8, 784); m = torch.randn(8, 4)
    assert torch.allclose(g(net, m, x), net.plain(x)[0], atol=1e-6)   # P=0 -> Gamma=1


def test_neuron_gate_layer_mask_restricts_the_gate():
    torch.manual_seed(0)
    g = NeuronGate(2, ("out",))
    with torch.no_grad():
        g.P.normal_()
    m = torch.randn(5, 2)
    pl = g.per_layer_mag(m)
    assert pl["h0"] == 0.0 and pl["h1"] == 0.0 and pl["out"] > 0


def test_synapse_gate_rank_k_identity_matches_explicit_gamma_times_w():
    """The K+1-matmul form must equal an explicit per-sample (Gamma . W)x. This identity is what
    makes per-synapse granularity tractable at all."""
    torch.manual_seed(0)
    net = _TinyNet(h0=6, h1=5, out=3)
    g = SynapseGate(2, None, GateDims(in_dim=784, h0=6, h1=5, out=3))
    with torch.no_grad():
        g.P0.normal_(std=0.01); g.P1.normal_(std=0.01); g.P2.normal_(std=0.01)
    x = torch.randn(4, 784); m = torch.randn(4, 2)

    got = g(net, m, x)

    # explicit reference: per sample, build Gamma = 1 + sum_k m_k P_k and apply (Gamma . W)x + b
    ref = []
    for i in range(4):
        def lin(inp, layer, Pk):
            Gamma = 1.0 + torch.einsum("k,kod->od", m[i], Pk)
            return F.linear(inp, Gamma * layer.weight, layer.bias)
        h = F.relu(lin(x[i:i + 1], net.l0, g.P0))
        h = F.relu(lin(h, net.l1, g.P1))
        ref.append(lin(h, net.l2, g.P2))
    assert torch.allclose(got, torch.cat(ref), atol=1e-5)


def test_gate_K_reads_rank_at_both_granularities():
    assert gate_K(NeuronGate(7, None), "neuron") == 7
    assert gate_K(SynapseGate(3, ("out",)), "synapse") == 3


def test_heads_are_zero_init_on_the_output_layer():
    h = Heads(4)
    assert torch.all(h.f2.weight == 0) and torch.all(h.f2.bias == 0)
    assert torch.all(h(torch.randn(6, 784)) == 0)         # half of the double-zero-init saddle


def test_head_satisfies_the_modulator_head_contract():
    """Any x -> (B,K) module drives the gate, so a recurrent head is a drop-in for the MLP one."""
    h = Heads(3)
    assert isinstance(h, ModulatorHead)
    assert h(torch.randn(5, 784)).shape == (5, 3)


# ------------------------------------------------------------------ gate dims (no module globals)
def test_gate_dims_default_to_the_split_mnist_mlp():
    assert DEFAULT_DIMS == GateDims(784, 400, 400, 10)
    assert DEFAULT_DIMS.total == 810                       # the historical GATEDIM


def test_gate_width_is_a_constructor_arg_not_a_module_global():
    """This is what replaces pt7_capacity's width(H) context manager rebinding p7.H0/H1/GATEDIM:
    two gates of different widths must coexist, which module globals cannot express."""
    small = NeuronGate(2, None, GateDims(784, 10, 10, 10))
    big = NeuronGate(2, None, DEFAULT_DIMS)
    assert small.P.shape == (2, 30) and big.P.shape == (2, 810)
    assert small.dims.h0 == 10 and big.dims.h0 == 400      # neither disturbed the other


def test_narrow_gate_runs_end_to_end_at_its_own_width():
    torch.manual_seed(0)
    dims = GateDims(784, 12, 8, 4)
    net, g = _TinyNet(h0=12, h1=8, out=4), NeuronGate(3, None, dims)
    with torch.no_grad():
        g.P.normal_()
    x, m = torch.randn(6, 784), torch.randn(6, 3)
    assert g(net, m, x).shape == (6, 4)
    pl = g.per_layer_mag(m)
    assert all(v > 0 for v in pl.values())


def test_per_layer_slices_follow_the_dims():
    g = NeuronGate(1, ("h1",), GateDims(784, 3, 5, 2))
    assert g.lm[:3].sum() == 0 and g.lm[3:8].sum() == 5 and g.lm[8:].sum() == 0


def test_make_gate_threads_dims_to_both_granularities():
    dims = GateDims(784, 7, 6, 3)
    assert make_gate("neuron", 2, None, dims).P.shape == (2, 16)
    syn = make_gate("synapse", 2, None, dims)
    assert syn.P0.shape == (2, 7, 784) and syn.P2.shape == (2, 3, 6)


# ------------------------------------------------------------------ signals
def test_entropy_matches_hand_computation():
    logits = torch.tensor([[0.0, 0.0], [100.0, -100.0]])
    e = entropy(logits)
    assert math.isclose(e[0].item(), math.log(2), rel_tol=1e-5)
    assert e[1].item() < 1e-6                             # confident -> ~0


def test_masked_ce_only_scores_the_label_pair():
    logits = torch.zeros(1, 10); logits[0, 7] = 5.0       # class 7 -> task 3 -> pair (6,7)
    y = torch.tensor([7])
    assert per_sample_masked_ce(logits, y).item() < per_sample_ce_plain(logits, y).item()


def test_ach_is_invariant_to_the_loss_fn_hook_but_5ht_is_not():
    """The correct internal check on the loss_fn hook: entropy does not depend on the loss, so ACh is
    byte-identical across hooks, while a directly loss-proportional driver rescales."""
    torch.manual_seed(0)
    net = _TinyNet()
    x, y = torch.randn(16, 784), torch.randint(0, 10, (16,))
    vals = {}
    for tag, fn in (("masked", per_sample_masked_ce), ("plain", per_sample_ce_plain)):
        s = Signals(["ACh", "5HT"], standardize=False, loss_fn=fn)
        vals[tag] = s.targets(net, x, y, update=False)
    assert torch.equal(vals["masked"][:, 0], vals["plain"][:, 0])          # ACh identical
    assert not torch.allclose(vals["masked"][:, 1], vals["plain"][:, 1])   # 5HT rescales


def test_standardization_is_off_by_flag_and_leaves_raw_scale():
    torch.manual_seed(0)
    net = _TinyNet()
    x, y = torch.randn(32, 784), torch.randint(0, 10, (32,))
    raw = Signals(["5HT"], standardize=False).targets(net, x, y)
    assert raw.abs().max() > 0


def test_tonic_driver_has_no_per_sample_variation():
    """The mechanistic reason tonic drivers must NOT be standardized: their per-batch variance is ~0,
    so dividing by sqrt(run_var) divides by ~0."""
    torch.manual_seed(0)
    net = _TinyNet()
    x, y = torch.randn(32, 784), torch.randint(0, 10, (32,))
    t = Signals(["ACh_ema"], standardize=False).targets(net, x, y)
    assert t.var(0).item() == pytest.approx(0.0, abs=1e-12)


def test_signals_rejects_unknown_driver():
    torch.manual_seed(0)
    with pytest.raises(ValueError):
        Signals(["NOT_A_DRIVER"], standardize=False).targets(
            _TinyNet(), torch.randn(4, 784), torch.randint(0, 10, (4,)))


def test_nedriver_widths():
    assert NEDriver("emb_all", False).K() == 1
    assert NEDriver("vec_x", False).K() == 784
    assert NEDriver("vecproj", False).K() == 32


def test_nedriver_feature_width_is_a_constructor_arg():
    d = NEDriver("vec_h1", False, feat_dim=64)
    assert d.K() == 64
    assert NEDriver("vec_h1proj", False, feat_dim=64).R.shape == (64, 32)


def test_input_novelty_driver_needs_no_forward():
    """vec_x / vecproj are pre-forward: computable with no model at all (why they are the only
    drivers well-conditioned at test time)."""
    d = NEDriver("vec_x", standardize=False)
    v = d.value(None, torch.randn(8, 784))
    assert v.shape == (8, 784)


# ------------------------------------------------------------------ novelty driver: norm axis
def test_norm_axis_collapses_the_driver_to_one_scalar():
    """norm=True is a rank axis: K -> 1, and with it the gate projection shrinks by the same factor."""
    assert NEDriver("vec_x", False, norm=True).K() == 1
    assert NEDriver("vecproj", False, norm=True).K() == 1
    assert NEDriver("vec_h1", False, norm=True).K() == 1
    assert NEDriver("vec_x", False).K() == 784                # default is unchanged
    x = torch.randn(8, 784)
    assert NEDriver("vec_x", False, norm=True).value(None, x).shape == (8, 1)


def test_norm_is_the_l2_norm_of_the_unnormed_driver():
    x = torch.randn(6, 784).to(DEV)                           # vecproj's R lives on DEV
    vec = NEDriver("vecproj", standardize=False).value(None, x)
    nrm = NEDriver("vecproj", standardize=False, norm=True).value(None, x)
    assert torch.allclose(nrm.squeeze(1), vec.norm(dim=1), atol=1e-6)


def test_vec_h1_with_norm_reproduces_emb_all():
    """emb_all IS the norm of the vec_h1 difference, so the two must agree exactly — and norm=True is
    therefore a no-op on emb_all itself."""
    net = _TinyNet(h0=16, h1=16); x = torch.randn(6, 784)
    a = NEDriver("vec_h1", standardize=False, norm=True, feat_dim=16).value(net, x)
    b = NEDriver("emb_all", standardize=False, feat_dim=16).value(net, x)
    c = NEDriver("emb_all", standardize=False, norm=True, feat_dim=16).value(net, x)
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(b, c, atol=1e-6)


def test_norm_is_taken_before_standardization():
    """Standardising per-dimension and THEN taking the norm concentrates it at sqrt(K); the driver
    must z-score the scalar instead, so a standardised norm straddles zero."""
    d = NEDriver("vecproj", standardize=True, norm=True)
    v = None
    for _ in range(30):
        v = d.value(None, torch.randn(64, 784).to(DEV))
    assert v.shape == (64, 1)
    assert abs(float(v.mean())) < 2.0                         # z-scored, not pinned near sqrt(32)=5.7


# ------------------------------------------------------------------ novelty driver: reference mean
def test_mean_modes_are_validated_at_construction():
    with pytest.raises(ValueError):
        NEDriver("vec_x", False, mean_mode="running")


def test_trueavg_uses_the_installed_mean_and_never_drifts():
    mu = torch.arange(784, dtype=torch.float32)
    d = NEDriver("vec_x", standardize=False, mean_mode="trueavg").set_true_mean(mu)
    x = torch.randn(4, 784)
    for _ in range(5):                                        # updates must not move the reference
        d.value(None, torch.randn(64, 784))
    assert torch.allclose(d.value(None, x), x - mu, atol=1e-5)


def test_missing_true_mean_raises_rather_than_falling_back_to_the_ema():
    with pytest.raises(RuntimeError):
        NEDriver("vec_x", False, mean_mode="trueavg").value(None, torch.randn(4, 784))
    d = NEDriver("vec_x", False, mean_mode="ema+trueavg")
    d.value(None, torch.randn(4, 784))                        # training phase is fine (uses the ema)
    with pytest.raises(RuntimeError):
        d.value(None, torch.randn(4, 784), update=False)      # inference without a true mean is not


def test_ema_plus_trueavg_switches_reference_between_phases():
    """The one mode whose two phases disagree: ema while training, exact mean at inference."""
    mu = torch.full((784,), 3.0)
    hyb = NEDriver("vec_x", standardize=False, mean_mode="ema+trueavg").set_true_mean(mu)
    ema = NEDriver("vec_x", standardize=False, mean_mode="ema")
    xs = [torch.randn(32, 784) for _ in range(6)]
    for x in xs:
        assert torch.allclose(hyb.value(None, x), ema.value(None, x), atol=1e-6)   # train: identical
    x = torch.randn(4, 784)
    assert torch.allclose(hyb.value(None, x, update=False), x - mu, atol=1e-5)
    assert not torch.allclose(ema.value(None, x, update=False), x - mu, atol=1e-3)


def test_inference_switch_can_be_forced_for_a_mid_training_readout():
    """update=False during TRAINING (a meta-loss batch) must be able to keep the training reference."""
    d = NEDriver("vec_x", standardize=False, mean_mode="ema+trueavg").set_true_mean(torch.zeros(784))
    d.value(None, torch.randn(32, 784))
    x = torch.randn(4, 784)
    train_read = d.value(None, x, update=False, inference=False)
    infer_read = d.value(None, x, update=False)
    assert not torch.allclose(train_read, infer_read)
    assert torch.allclose(infer_read, x, atol=1e-6)                        # true mean is all-zeros


def test_cumulative_mean_is_the_exact_running_arithmetic_mean():
    d = NEDriver("vec_x", standardize=False, mean_mode="cumulative")
    xs = [torch.randn(16, 784) for _ in range(4)]
    for x in xs:
        d.value(None, x)
    assert torch.allclose(d.mx, torch.cat(xs).mean(0), atol=1e-5)


def test_default_construction_is_unchanged():
    """Both new axes default to the historical behaviour, so every frozen number still reproduces."""
    d = NEDriver("vec_x", standardize=False)
    assert d.mean_mode == "ema" and d.norm is False and d.K() == 784


def test_dataset_mean_is_exact_and_rng_neutral():
    xs = torch.randn(50, 784)
    loader = [(xs[i:i + 8], torch.zeros(len(xs[i:i + 8]), dtype=torch.long)) for i in range(0, 50, 8)]
    m = dataset_mean(loader)
    assert torch.allclose(m.cpu(), xs.mean(0), atol=1e-5)
    with pytest.raises(ValueError):
        dataset_mean(loader, space="h1")                      # h1 space needs the net
    before = torch.get_rng_state()
    dataset_mean(loader)
    assert torch.equal(torch.get_rng_state(), before)


def test_rng_frozen_restores_every_stream():
    before = torch.get_rng_state()
    with rng_frozen():
        torch.randn(100)
    assert torch.equal(torch.get_rng_state(), before)


# ------------------------------------------------------------------ controls
def test_cell_spec_flags_the_two_controls():
    assert cell_spec("free") == (["free"] * 4, True, False)
    assert cell_spec("5ht-const") == (["const"], False, True)
    assert cell_spec("all4") == (["DA", "ACh", "NE", "5HT"], False, False)
    assert cell_spec("ACh") == (["ACh"], False, False)


def test_dead_gate_guard_accepts_inert_and_rejects_engaged():
    assert_dead_gate({"h0": 0.0, "h1": 0.0, "out": 0.0})
    with pytest.raises(AssertionError, match="RNG-matched baseline"):
        assert_dead_gate({"h0": 0.0, "h1": 0.3, "out": 0.0})


def test_live_gate_guard_catches_the_double_zero_init_saddle():
    with pytest.raises(AssertionError, match="saddle"):
        assert_live_gate({"h0": 0.0, "h1": 0.0, "out": 0.0})
    assert assert_live_gate({"h0": 0.2, "h1": 0.0, "out": 0.0}) == pytest.approx(0.2)


def test_break_symmetry_makes_a_zero_init_head_produce_nonzero_m():
    torch.manual_seed(0)
    h = Heads(4)
    x = torch.randn(8, 784)
    assert torch.all(h(x) == 0)
    break_symmetry(h)
    assert h(x).abs().sum() > 0                           # can now bootstrap off the saddle


def test_gate_magnitude_sums_layers():
    assert gate_magnitude({"h0": 0.1, "h1": 0.2, "out": 0.3}) == pytest.approx(0.6)


# ------------------------------------------------------------------ ledger
def _ledger(tmp_path, with_cost=False):
    return L.Ledger(tmp_path / "l.tsv", keys=["regime", "arm", "seed"],
                    metrics=["acc"], with_cost=with_cost)


def test_ledger_roundtrip_and_resume(tmp_path):
    led = _ledger(tmp_path)
    assert led.rows() == []
    led.append(dict(regime="normal", arm="er", seed=42), dict(acc=0.8946))
    assert led.is_done(regime="normal", arm="er", seed=42)
    assert not led.is_done(regime="normal", arm="er", seed=43)
    r = led.rows()[0]
    assert r["arm"] == "er" and r["seed"] == 42 and r["acc"] == pytest.approx(0.8946)


def test_ledger_rejects_schema_drift(tmp_path):
    _ledger(tmp_path)
    with pytest.raises(ValueError, match="schema drift"):
        L.Ledger(tmp_path / "l.tsv", keys=["regime", "arm"], metrics=["acc"])


def test_ledger_requires_every_key_column(tmp_path):
    led = _ledger(tmp_path)
    with pytest.raises(KeyError):
        led.append(dict(regime="normal", arm="er"), dict(acc=0.5))


def test_ledger_with_cost_requires_a_cost(tmp_path):
    led = _ledger(tmp_path, with_cost=True)
    with pytest.raises(ValueError, match="with_cost"):
        led.append(dict(regime="normal", arm="er", seed=42), dict(acc=0.5))
    led.append(dict(regime="normal", arm="er", seed=42), dict(acc=0.5),
               cost=C.Cost(backbone_params=478410, extra_params=25252, buffer_bytes=3_136_000))
    row = led.rows()[0]
    assert row["buffer_bytes"] == 3_136_000
    assert row["param_ratio"] == pytest.approx(25252 / 478410, rel=1e-3)


def test_paired_delta_takes_its_seeds_from_the_ledger(tmp_path):
    """The bug this guards: a width carrying extra seeds must pair on ALL of them, not on the first
    few from some constant."""
    led = _ledger(tmp_path)
    for s, (a, b) in {42: (0.90, 0.88), 43: (0.91, 0.89), 44: (0.80, 0.79), 45: (0.70, 0.60)}.items():
        led.append(dict(regime="normal", arm="mech", seed=s), dict(acc=a))
        led.append(dict(regime="normal", arm="free", seed=s), dict(acc=b))
    d = L.paired_delta(led.rows(), "mech", "free")
    assert d[2] == 4                                       # all four seeds paired, not three
    assert d[0] == pytest.approx((0.02 + 0.02 + 0.01 + 0.10) / 4)


def test_paired_delta_is_none_without_a_shared_pair(tmp_path):
    led = _ledger(tmp_path)
    led.append(dict(regime="normal", arm="mech", seed=42), dict(acc=0.9))
    led.append(dict(regime="normal", arm="free", seed=43), dict(acc=0.8))
    assert L.paired_delta(led.rows(), "mech", "free") is None


def test_delta_table_warns_when_the_rng_matched_control_is_missing(tmp_path):
    led = _ledger(tmp_path)
    for s in (42, 43):
        led.append(dict(regime="normal", arm="mech", seed=s), dict(acc=0.90))
        led.append(dict(regime="normal", arm="er", seed=s), dict(acc=0.88))
    out = L.delta_table(led.rows(), ["mech", "er"], baseline="er")
    assert "no `free` control" in out
    led.append(dict(regime="normal", arm="free", seed=42), dict(acc=0.885))
    led.append(dict(regime="normal", arm="free", seed=43), dict(acc=0.885))
    out2 = L.delta_table(led.rows(), ["mech", "er", "free"], baseline="er")
    assert "no `free` control" not in out2 and "d-free" in out2


def test_delta_table_marks_deltas_inside_the_noise_floor(tmp_path):
    led = _ledger(tmp_path)
    for s in (42, 43):
        led.append(dict(regime="normal", arm="mech", seed=s), dict(acc=0.8820))
        led.append(dict(regime="normal", arm="er", seed=s), dict(acc=0.8816))
        led.append(dict(regime="normal", arm="free", seed=s), dict(acc=0.8816))
    out = L.delta_table(led.rows(), ["mech"], baseline="er")
    assert "~" in out                                      # +0.0004 is null at the noise floor


def test_where_filters_across_types(tmp_path):
    led = _ledger(tmp_path)
    led.append(dict(regime="normal", arm="er", seed=42), dict(acc=0.9))
    led.append(dict(regime="rehearsal-free", arm="er", seed=42), dict(acc=0.7))
    assert len(L.where(led.rows(), regime="normal")) == 1
    assert len(L.where(led.rows(), seed=42)) == 2


# ------------------------------------------------------------------ cost
def test_param_ratio_and_capacity_confound_flag():
    big_backbone = C.Cost(backbone_params=478410, extra_params=25252)
    assert not big_backbone.capacity_confound()
    assert big_backbone.warn_if_confounded() is None
    # the pt7 capacity case: a 25k head over a 4k backbone
    tiny_backbone = C.Cost(backbone_params=4015, extra_params=25252)
    assert tiny_backbone.capacity_confound()
    assert "capacity confound" in tiny_backbone.warn_if_confounded("er+freefix")


def test_count_params_skips_none():
    assert C.count_params(nn.Linear(2, 3), None) == 9


def test_buffer_bytes_zero_for_rehearsal_free():
    from neurocore.buffers import Reservoir
    assert C.buffer_bytes(None) == 0
    b = Reservoir(1000)
    assert C.buffer_bytes(b) == 1000 * 784 * 4 + 1000 * 8


def test_forward_counter_catches_an_extra_pass():
    net = _TinyNet()
    x = torch.randn(2, 784)
    with C.counted(net) as c:
        net.plain(x)
        net.plain(x)                                       # e.g. a driver's observer forward
    assert c.count(net) == 2


def test_regimes_are_the_three_reported():
    assert C.REGIMES == ("normal", "rehearsal-free", "memory-budgeted")


# ------------------------------------------------------------------ tuned points
def test_missing_tuned_main_key_raises_by_design():
    with pytest.raises(KeyError, match="Tune it"):
        T.tuned_main("fashionmnist", "classil", "er", "adam")
    with pytest.raises(KeyError):
        T.tuned_main("splitmnist", "classil", "naive", "sgd")     # deliberately not swept


def test_tuned_main_values_survived_the_problem_key_migration():
    assert T.tuned_main("splitmnist", "classil", "er", "adam") == dict(lr=3e-4, epochs_per_task=5)
    assert T.tuned_main("splitmnist", "classil", "er", "sgd") == dict(lr=3e-2, epochs_per_task=5)
    assert T.tuned_main("splitmnist", "taskil", "er", "adam") == dict(lr=3e-4, epochs_per_task=10)


def test_neuro_lr_falls_back_at_the_right_optimizer_scale():
    assert T.tuned_neuro_lr("splitmnist", "classil", "er", "adam", "all4", "neuron") == 1e-3
    # un-swept combination -> optimizer-scale default, never a cross-optimizer value
    assert T.tuned_neuro_lr("splitmnist", "classil", "er", "sgd", "gain", "neuron") == 3e-3
    assert T.tuned_neuro_lr("fashionmnist", "classil", "er", "adam", "all4", "neuron") == 1e-3


def test_select_tuned_prefers_the_cheapest_cell_inside_the_noise_floor():
    cells = [dict(lr=1e-3, epochs_per_task=20, acc=0.9050),
             dict(lr=1e-3, epochs_per_task=5,  acc=0.9020),   # within noise, far cheaper -> pick
             dict(lr=3e-3, epochs_per_task=10, acc=0.8600)]
    pick, info = T.select_tuned(cells)
    assert pick["epochs_per_task"] == 5 and info["n_tied"] == 2


def test_select_tuned_takes_a_clear_winner():
    cells = [dict(lr=1e-3, epochs_per_task=20, acc=0.95),
             dict(lr=1e-3, epochs_per_task=5,  acc=0.80)]
    pick, _ = T.select_tuned(cells)
    assert pick["epochs_per_task"] == 20


def test_grid_boundary_selection_is_flagged_as_a_truncated_grid():
    grid = [3e-4, 1e-3, 3e-3]
    assert T.at_grid_boundary(dict(lr=3e-4), grid, "lr")      # floor -> extend downward
    assert not T.at_grid_boundary(dict(lr=1e-3), grid, "lr")  # interior -> a real optimum


# ------------------------------------------------------------------ task selectors (pt6)
from neurocore.task_selection import (PROTOTYPE_TAUS, EmbeddingSelector, GateTable, NearestPrototype,  # noqa: E402
                                 SoftMLPSelector, TaskInferenceNet, grouped_synapse,
                                 routing_ceiling, soft_blend_synapse, synapse_mats)


def test_inference_net_returns_a_task_posterior():
    g = TaskInferenceNet(n_tasks=5)
    p = g.posterior(torch.randn(7, 784))
    assert p.shape == (7, 5)
    assert torch.allclose(p.sum(1), torch.ones(7), atol=1e-6)
    assert g.embed(torch.randn(7, 784)).shape == (7, 128)


def test_gate_table_is_zero_init_so_every_task_starts_at_parity():
    tab = GateTable(5, 810)
    assert torch.all(tab.rows() == 0)
    assert torch.all(tab.oracle(torch.tensor([0, 3, 4])) == 0)


def test_gate_table_blend_is_the_posterior_weighted_sum():
    tab = GateTable(3, 4)
    with torch.no_grad():
        tab.P.copy_(torch.tensor([[1.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 3.0, 0]]))
    post = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    got = tab.blend(post)
    assert torch.allclose(got[0], torch.tensor([0.5, 1.0, 0.0, 0.0]))
    assert torch.allclose(got[1], torch.tensor([0.0, 0.0, 3.0, 0.0]))


def test_soft_resolution_equals_hard_when_the_posterior_is_confident():
    """Why soft ~= hard for a LEARNED selector: a confident softmax is effectively one-hot. Softness
    only pays on a DIFFUSE posterior (the prototype case)."""
    sel = SoftMLPSelector(n_tasks=3, dim=6)
    with torch.no_grad():
        sel.table.P.normal_()
        sel.inf.go.weight.mul_(0); sel.inf.go.bias.copy_(torch.tensor([50.0, 0.0, 0.0]))
    x = torch.randn(4, 784)
    assert torch.allclose(sel.eval_gate(x, "soft"), sel.eval_gate(x, "hard"), atol=1e-5)


def test_soft_mlp_train_gate_uses_true_ids_not_the_posterior():
    """Training on the blend HURTS: a one-hot gives each row a clean unmixed gradient."""
    sel = SoftMLPSelector(n_tasks=3, dim=6)
    with torch.no_grad():
        sel.table.P.normal_()
    tids = torch.tensor([0, 2])
    assert torch.equal(sel.train_gate(torch.randn(2, 784), tids), sel.table.P[tids])


def test_soft_mlp_rejects_per_image_resolution():
    with pytest.raises(ValueError, match="oracle"):
        SoftMLPSelector(n_tasks=3, dim=6).eval_gate(torch.randn(2, 784), "per-image")


def test_embedding_selector_is_oracle_free_by_construction():
    sel = EmbeddingSelector(n_tasks=5, dim=810, proj="lin")
    x = torch.randn(3, 784)
    # no tids anywhere in the signature path
    assert sel.train_gate(x).shape == (3, 810)
    assert sel.eval_gate(x, "per-image").shape == (3, 810)
    assert torch.all(sel.eval_gate(x, "per-image") == 0)          # zero-init W -> parity at start


def test_embedding_selector_mlp_projection_and_bad_args():
    assert EmbeddingSelector(dim=810, proj="mlp").gate_per_sample(torch.randn(2, 784)).shape == (2, 810)
    with pytest.raises(ValueError, match="lin"):
        EmbeddingSelector(proj="conv")
    with pytest.raises(ValueError, match="per-image only"):
        EmbeddingSelector().eval_gate(torch.randn(2, 784), "soft")


def test_selector_param_groups_are_disjoint():
    """The gate and the inference net are trained by DIFFERENT losses (main/meta vs task-CE), so
    their parameter groups must not overlap or one optimizer silently steps the other's params."""
    for sel in (SoftMLPSelector(dim=32), EmbeddingSelector(dim=32)):
        gate_ids = {id(p) for p in sel.gate_params()}
        inf_ids = {id(p) for p in sel.inf_params()}
        assert gate_ids and inf_ids and not (gate_ids & inf_ids)


def test_nearest_prototype_hard_assignment_and_diffuse_posterior():
    mus = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    np_ = NearestPrototype(mus)
    x = torch.tensor([[0.1, 0.0], [9.0, 10.0]])
    assert torch.equal(np_.hard(x), torch.tensor([0, 1]))
    p = np_.posterior(x, tau=1000.0)                               # large tau -> diffuse
    assert p[0, 0] > p[0, 1] and abs(p[0, 0] - p[0, 1]) < 0.5


def test_prototype_taus_start_at_the_interior_peak():
    assert min(PROTOTYPE_TAUS) == 0.03                             # tau->0 converges to hard nearest


# ---- the per-synapse resolution identities ----
def _syn_fixture():
    torch.manual_seed(0)
    layers = ((6, 5), (4, 6), (3, 4))
    net = _TinyNet(h0=6, h1=4, out=3)
    net.l0 = nn.Linear(5, 6); net.l1 = nn.Linear(6, 4); net.l2 = nn.Linear(4, 3)
    wb = [(net.l0.weight, net.l0.bias), (net.l1.weight, net.l1.bias), (net.l2.weight, net.l2.bias)]
    P = torch.randn(3, sum(a * b for a, b in layers)) * 0.05
    return wb, synapse_mats(P, layers), torch.randn(6, 5)


def test_synapse_mats_partitions_the_flat_table():
    P = torch.arange(2 * (6 * 5 + 4 * 6)).float().view(2, -1)
    m = synapse_mats(P, ((6, 5), (4, 6)))
    assert m[0].shape == (2, 6, 5) and m[1].shape == (2, 4, 6)
    assert torch.equal(m[0][0].flatten(), P[0, :30])


def test_soft_blend_equals_grouped_routing_under_a_one_hot_posterior():
    """The blend must degenerate exactly to hard routing — that is what makes one training run serve
    both resolution modes."""
    wb, mats, X = _syn_fixture()
    tids = torch.tensor([0, 1, 2, 0, 1, 2])
    post = F.one_hot(tids, num_classes=3).float()
    assert torch.allclose(soft_blend_synapse(wb, mats, X, post),
                          grouped_synapse(wb, mats, X, tids), atol=1e-5)


def test_soft_blend_is_exact_against_an_explicit_per_sample_gamma():
    """Gamma_i = sum_t p_it Gamma_t and (Gamma . W)x is linear in Gamma, so the T-matmul form is
    EXACT — no per-sample (B, d_out, d_in) expansion and no chunking is needed."""
    wb, mats, X = _syn_fixture()
    post = F.softmax(torch.randn(6, 3), dim=1)
    got = soft_blend_synapse(wb, mats, X, post)

    ref = []
    for i in range(X.size(0)):
        h = X[i:i + 1]
        for li, (W, b) in enumerate(wb):
            Gamma = 1.0 + torch.einsum("t,tod->od", post[i], mats[li])
            h = F.linear(h, Gamma * W, b)
            if li < len(wb) - 1:
                h = F.relu(h)
        ref.append(h)
    assert torch.allclose(got, torch.cat(ref), atol=1e-5)


def test_routing_ceiling_is_the_multiplicative_law():
    assert routing_ceiling(0.9972, 0.8878) == pytest.approx(0.8853, abs=1e-4)


def test_gate_table_deterministic_path_is_numerically_identical():
    """one_hot(tids) @ P == P[tids], but with a matmul backward instead of an atomic scatter-add.
    The scatter-add is nondeterministic on MPS (measured 3.8e-6/step vs exactly 0 for matmul), which
    is why pt6's soft_mlp does not reproduce bit-exact even from its own code."""
    tab = GateTable(5, 12)
    with torch.no_grad():
        tab.P.normal_()
    tids = torch.tensor([0, 4, 2, 2])
    assert torch.allclose(tab.oracle(tids, deterministic=False),
                          tab.oracle(tids, deterministic=True), atol=1e-6)


def test_deterministic_flag_threads_through_the_selector():
    sel = SoftMLPSelector(n_tasks=5, dim=12, deterministic=True)
    with torch.no_grad():
        sel.table.P.normal_()
    tids = torch.tensor([1, 3])
    ref = SoftMLPSelector(n_tasks=5, dim=12, deterministic=False)
    ref.table.P.data.copy_(sel.table.P.data)
    assert torch.allclose(sel.train_gate(torch.randn(2, 784), tids),
                          ref.train_gate(torch.randn(2, 784), tids), atol=1e-6)
