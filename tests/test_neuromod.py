import torch
import torch.nn as nn

from prototype.model import MLP, ModulatedLinear
from prototype.neuromod import MultiWeightMaskMLP, parse_layer_list


def _make_multi(layers, rank=0):
    """Build a MultiWeightMaskMLP masking the given net.<idx> linears of a fresh MLP."""
    model = MLP()
    layer_dims = {idx: (model.net[idx].out_features, model.net[idx].in_features) for idx in layers}
    return MultiWeightMaskMLP(model, layer_dims, rank=rank)


def test_parse_layer_list():
    assert parse_layer_list("0,2,4") == [0, 2, 4]
    assert parse_layer_list("") == []
    assert parse_layer_list("  4 , 0 ,2 ") == [0, 2, 4]  # spaces + reordering
    assert parse_layer_list("2,2,0") == [0, 2]            # dedup


def test_modulated_linear_parity_without_mask():
    """ModulatedLinear with no mask must be numerically identical to nn.Linear (rule 4)."""
    torch.manual_seed(0)
    ref = nn.Linear(400, 10)
    ml = ModulatedLinear(400, 10)
    with torch.no_grad():
        ml.weight.copy_(ref.weight)
        ml.bias.copy_(ref.bias)
    x = torch.randn(8, 400)
    assert torch.allclose(ml(x), ref(x), atol=1e-6)


def test_multi_wraps_only_listed_layers_including_output():
    """Every listed linear (incl. the output head net.4) becomes ModulatedLinear; others unchanged."""
    m = _make_multi([0, 2, 4])
    net = m.base.net
    for idx in (0, 2, 4):
        assert isinstance(net[idx], ModulatedLinear), f"net.{idx} not wrapped"
    # ReLUs (odd indices) are untouched, still plain nn.ReLU
    assert isinstance(net[1], nn.ReLU) and isinstance(net[3], nn.ReLU)


def test_multi_forward_shape_and_runs():
    """Forward through a multi-masked net (hidden + output) produces (B, 10) logits."""
    m = _make_multi([0, 2, 4])
    x = torch.randn(16, 1, 28, 28)
    out = m(x)
    assert out.shape == (16, 10)
    assert torch.isfinite(out).all()


def test_output_head_only_masking():
    """Masking just the output head (net.4) is allowed and wraps only that layer."""
    m = _make_multi([4])
    net = m.base.net
    assert isinstance(net[4], ModulatedLinear)
    assert isinstance(net[0], nn.Linear) and not isinstance(net[0], ModulatedLinear)
    assert m(torch.randn(4, 1, 28, 28)).shape == (4, 10)


def test_all_mask_params_are_trainable_submodules():
    """One optimizer over parameters() must see the shared signal net and every per-layer head."""
    m = _make_multi([0, 2, 4])
    names = {n for n, _ in m.named_parameters()}
    assert any(n.startswith("signal_net.") for n in names), "shared signal net params missing"
    for idx in (0, 2, 4):
        assert any(n.startswith(f"heads.{idx}.") for n in names), f"layer {idx} head params missing"


def test_signal_net_is_shared_across_layers():
    """Exactly ONE signal net feeds all layers; each layer has its own head (projection)."""
    m = _make_multi([0, 2, 4])
    signal_nets = [mod for name, mod in m.named_modules() if name == "signal_net"]
    assert len(signal_nets) == 1, "expected a single shared signal net"
    # one distinct head per masked layer
    assert set(m.heads.keys()) == {"0", "2", "4"}
    assert len({id(h) for h in m.heads.values()}) == 3


def test_reject_non_linear_layer():
    """Selecting a ReLU index (odd) must raise, not silently mask a non-linear."""
    model = MLP()
    try:
        MultiWeightMaskMLP(model, {1: (400, 400)})
    except ValueError:
        return
    assert False, "expected ValueError for a non-Linear layer index"
