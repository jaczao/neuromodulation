"""EWCER = ER replay + EWC Fisher penalty, stacked on the same step.

Baseline-composition test: EWCER must exercise BOTH mechanisms — the ER reservoir
buffer AND the EWC Fisher/snapshot bookkeeping — and must reduce to plain ER when
ewc_lambda=0 (the penalty term vanishes). It is a plain CL method (no neuromod), so
it flows through cl_train's generic make_cl_method path.
"""
import random

import torch

from prototype.configs import CLConfig
from prototype.methods import ER, EWC, EWCER, make_cl_method
from prototype.model import MLP


def _tiny_config(**kw) -> CLConfig:
    return CLConfig(epochs_per_task=1, er_buffer_size=8, ewc_samples=8, **kw)


def _two_tasks():
    """Two disjoint 2-class tasks of 784-dim fake samples, labels {0,1} then {2,3}."""
    g = torch.Generator().manual_seed(0)
    for base in (0, 2):
        x = torch.randn(16, 784, generator=g)
        y = torch.randint(base, base + 2, (16,), generator=g)
        yield [(x, y)]  # single-batch "loader"


def test_registry_and_mro():
    m = make_cl_method("ewc_er")
    assert isinstance(m, EWCER)
    # composes both parents, so inherited buffer (ER) + Fisher (EWC) machinery are present
    assert isinstance(m, ER) and isinstance(m, EWC)


def test_both_mechanisms_are_live():
    """After two tasks, the ER buffer must be populated AND one Fisher/snapshot stored."""
    torch.manual_seed(0)
    model = MLP()
    method = EWCER()
    config = _tiny_config()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    loaders = list(_two_tasks())
    for t, loader in enumerate(loaders):
        method.train_task(t, model, loader, opt, crit, device, config)
        method.on_task_end(t, model, loader, device, config)

    assert len(method.buf_x) > 0, "ER buffer never filled — replay is not live"
    assert len(method.fishers) == 2 and len(method.snapshots) == 2, "EWC bookkeeping missing"


def test_lambda_zero_matches_pure_er():
    """ewc_lambda=0 kills the penalty term, so EWCER must step identically to ER.

    Same seed/model/data/optimizer: with no penalty the two loops issue the same
    gradient steps, so the final parameters match exactly.
    """
    device = torch.device("cpu")
    crit = torch.nn.CrossEntropyLoss()

    def run(method):
        torch.manual_seed(0)
        random.seed(0)  # ER's reservoir buffer uses python's random; seed for run-to-run determinism
        model = MLP()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        config = _tiny_config(ewc_lambda=0.0)
        # identical data each run (generator reseeded inside _two_tasks)
        for t, loader in enumerate(_two_tasks()):
            method.train_task(t, model, loader, opt, crit, device, config)
            method.on_task_end(t, model, loader, device, config)
        return [p.detach().clone() for p in model.parameters()]

    er_params = run(ER())
    ewcer_params = run(EWCER())
    for a, b in zip(er_params, ewcer_params):
        assert torch.allclose(a, b), "ewc_lambda=0 EWCER diverged from pure ER"
