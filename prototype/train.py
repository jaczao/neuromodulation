import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from prototype.configs import CLConfig, StandardConfig
from prototype.data import SplitMNIST, get_standard_loaders, make_sequence
from prototype.methods import make_cl_method
from prototype.model import MLP
from prototype.neuromod import (
    DriverBank,
    GainDriverModulator,
    LogitModulatedMLP,
    ModulatedMLP,
    MultiWeightMaskMLP,
    PlasticityDriverModulator,
    PlasticityModulator,
    SynapsePlasticityDriverModulator,
    TaskInferenceNet,
    TaskWeightMaskMLP,
    WeightMaskMLP,
    activation_stats,
    make_modulator,
    parse_layer_list,
    predictive_entropy,
)

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _is_plasticity(config) -> bool:
    # pt5's plasticity target has its OWN (fixed-projection, SGD grad-gating) path; keep it out of
    # the legacy lookahead branch so the pt5 driver dispatch owns it (see _is_pt5).
    return config.use_neuromod and config.neuromod_target == "plasticity" and not _is_pt5(config)


def _is_weight_mask_driver(config) -> bool:
    return (
        config.use_neuromod
        and config.neuromod_target == "weight_mask"
        and config.neuromod_driver != "none"
    )


def _is_importance(config) -> bool:
    return config.use_neuromod and config.neuromod_target == "importance"


def _is_task_route(config) -> bool:
    return config.use_neuromod and config.neuromod_target == "task_route"


def _is_logit_recency(config) -> bool:
    return (config.use_neuromod and config.neuromod_target == "logit"
            and config.neuromod_driver == "recency")


def _is_consolidation(config) -> bool:
    return config.use_neuromod and config.neuromod_target == "consolidation"


def _is_pt5(config) -> bool:
    """pt5 generalized driver path: selected when --neuromod-drivers is non-empty (rule 1)."""
    return config.use_neuromod and bool(getattr(config, "neuromod_drivers", "").strip())


def _build_pt5_model(config, model: nn.Module, n_tasks: int, device: torch.device) -> nn.Module:
    """pt5 (driver system) model builder. context=none, driver=task_id one-hot, projection selects
    the iteration (disjoint = Iteration 1). gain/weight_mask return a task-driven wrapper; plasticity
    keeps the base MLP unwrapped (its per-neuron LR gate is applied to gradients in the pt5 loop).
    """
    if config.neuromod_context != "none":
        raise NotImplementedError(
            "pt5 is drivers-only: pass --neuromod-context none (image-context bottleneck is a later SPEC)"
        )
    target = config.neuromod_target
    proj, sfrac, pseed = config.neuromod_projection, config.neuromod_shared_frac, config.neuromod_proj_seed
    gran = getattr(config, "neuromod_granularity", "neuron")
    if gran not in ("neuron", "synapse"):
        raise ValueError(f"unknown neuromod granularity {gran!r}; known: neuron | synapse")
    if target in ("activation", "hidden"):  # gain
        if gran == "synapse":
            # Per-synapse gain: forward gate (Γ⊙W)x on the listed linears (gain form), not per-neuron.
            # Under a fixed binary P this coincides numerically with weight_mask; they diverge only
            # under the learned projection (pt5 Iter 3), where gain uses 1+raw / sigmoid(raw).
            layers = parse_layer_list(getattr(config, "neuromod_mask_layers", ""))
            if not layers:
                raise ValueError(
                    "pt5 per-synapse gain (--neuromod-granularity synapse) requires --neuromod-mask-layers (e.g. '0,2')"
                )
            layer_dims = {l: (model.net[l].out_features, model.net[l].in_features) for l in layers}
            bank = DriverBank(config.neuromod_drivers, n_tasks)
            return TaskWeightMaskMLP(
                model, layer_dims, bank, projection=proj, shared_frac=sfrac, seed=pseed,
                gate="gain", gain_form=config.neuromod_gain_form,
                modulate_bias=getattr(config, "neuromod_modulate_bias", False),
            ).to(device)
        bank = DriverBank(config.neuromod_drivers, n_tasks)
        gate_layers = tuple(parse_layer_list(getattr(config, "neuromod_gain_layers", "0,2"))) or (0, 2)
        mod = GainDriverModulator(
            bank, gate_layers=gate_layers, projection=proj, shared_frac=sfrac, seed=pseed,
            gain_form=config.neuromod_gain_form,
        )
        return ModulatedMLP(model, mod).to(device)
    if target == "weight_mask":
        layers = parse_layer_list(getattr(config, "neuromod_mask_layers", ""))
        if not layers:
            raise ValueError("pt5 weight_mask requires --neuromod-mask-layers (e.g. '0,2' or '0,2,4')")
        layer_dims = {l: (model.net[l].out_features, model.net[l].in_features) for l in layers}
        bank = DriverBank(config.neuromod_drivers, n_tasks)
        return TaskWeightMaskMLP(
            model, layer_dims, bank, projection=proj, shared_frac=sfrac, seed=pseed,
            modulate_bias=getattr(config, "neuromod_modulate_bias", False),
        ).to(device)
    if target == "plasticity":
        return model  # plain MLP; the external plasticity modulator is built in the pt5 loop
    raise ValueError(f"pt5 supports targets activation|plasticity|weight_mask, got {target!r}")


def _pt5_gain_modulator_params(model: nn.Module) -> list:
    """The learned projection P params of a pt5 FORWARD-gain model (ModulatedMLP -> its modulator;
    TaskWeightMaskMLP -> its P_* projections). Used to train P by a modulator-only replay meta-loss
    (--neuromod-meta-replay) with a SEPARATE optimizer, so the main net is not trained on the buffer."""
    if isinstance(model, ModulatedMLP):
        return list(model.modulator.parameters())
    if isinstance(model, TaskWeightMaskMLP):
        return [p for n, p in model.named_parameters() if n.split(".")[0].startswith("P_")]
    return []


def _install_importance_gates(model: nn.Module, lam: float) -> dict:
    """Iteration 7: importance-gated plasticity, via per-parameter grad hooks.

    Maintains an online per-parameter importance omega (running sum of raw grad^2, never reset
    across tasks). Each backward, the hook scales that parameter's gradient by
    alpha_p = 1/(1 + lam*omega_p) BEFORE the optimizer sees it, so parameters important to past
    tasks (large omega) are frozen (alpha->0) and protected. omega starts at 0 (alpha=1 =
    vanilla), so it ramps in as training proceeds. Composes with any loop (naive, ER) since the
    hooks fire during backward regardless of the method.
    """
    state: dict = {"omega": {}, "handles": [], "gate_sum": 0.0, "gate_min": 1.0, "n": 0}
    for name, p in model.named_parameters():
        state["omega"][name] = torch.zeros_like(p)

    def make_hook(nm: str):
        def hook(g: torch.Tensor) -> torch.Tensor:
            om = state["omega"][nm]
            gate = 1.0 / (1.0 + lam * om)           # gate by importance accumulated SO FAR
            state["omega"][nm] = om + g.detach() ** 2  # then accumulate this batch
            state["gate_sum"] += float(gate.mean())
            state["gate_min"] = min(state["gate_min"], float(gate.min()))
            state["n"] += 1
            return g * gate
        return hook

    for name, p in model.named_parameters():
        state["handles"].append(p.register_hook(make_hook(name)))
    return state


def _build_model(config, device: torch.device, n_tasks: int | None = None) -> nn.Module:
    """Create vanilla MLP or ModulatedMLP depending on config.

    Plasticity target keeps the base MLP unwrapped (forward untouched); its
    modulator lives outside the model and is handled in the training loop.
    """
    model = MLP().to(device)
    if _is_pt5(config):
        # pt5 (driver system) needs the task count; only reachable from cl_train (CL-only front).
        if n_tasks is None:
            raise ValueError("pt5 driver path requires n_tasks (CL only)")
        return _build_pt5_model(config, model, n_tasks, device)
    if (not config.use_neuromod or _is_plasticity(config) or _is_importance(config)
            or _is_task_route(config) or _is_consolidation(config)):
        return model  # plain MLP; importance/task-router/consolidation are handled in cl_train
    if config.neuromod_target == "weight_mask":
        layers = parse_layer_list(getattr(config, "neuromod_mask_layers", ""))
        if layers:
            # Multi-layer form: mask several linears at once (incl. the output head net.4).
            # ONE shared signal net feeds a per-layer mask head each. Composes with the standard
            # method loops via model(x).
            if config.neuromod_driver != "none":
                raise NotImplementedError(
                    "multi-layer weight_mask (--neuromod-mask-layers) does not support the legacy "
                    "--neuromod-driver path; use a single layer or driver='none'"
                )
            layer_dims = {
                layer: (model.net[layer].out_features, model.net[layer].in_features)
                for layer in layers
            }
            return MultiWeightMaskMLP(
                model, layer_dims,
                rank=config.neuromod_mask_rank,
                mask_init=config.neuromod_mask_init,
            ).to(device)
        layer = config.neuromod_mask_layer
        lin = model.net[layer]
        mod = make_modulator(
            "weight_mask",
            variant=config.neuromod_variant,
            mask_dims=(lin.out_features, lin.in_features),
            mask_rank=config.neuromod_mask_rank,
            mask_init=config.neuromod_mask_init,
            driver=config.neuromod_driver,
            stateful_hidden=config.neuromod_stateful_hidden,
        )
        return WeightMaskMLP(model, mod, layer_idx=layer).to(device)
    if config.neuromod_target == "logit":
        mod = make_modulator("logit", variant=config.neuromod_variant, driver=config.neuromod_driver)
        return LogitModulatedMLP(model, mod).to(device)
    if config.neuromod_target == "direct_gain":
        mod = make_modulator(
            "direct_gain", variant=config.neuromod_variant, gain_gate=config.neuromod_gain_gate,
            gain_bounded=config.neuromod_gain_bounded,
        )
        return ModulatedMLP(model, mod).to(device)
    mod = make_modulator(
        config.neuromod_target,
        variant=config.neuromod_variant,
        learned_projection=config.neuromod_learned_projection,
    )
    return ModulatedMLP(model, mod).to(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on MPS/CPU, needed for CUDA portability


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _logit_mask(n_classes: int, allowed: list[int], device: torch.device) -> torch.Tensor:
    """Additive mask: 0 for allowed class indices, -inf elsewhere (task-IL output masking)."""
    mask = torch.full((n_classes,), float("-inf"), device=device)
    mask[allowed] = 0.0
    return mask


class MaskedCE:
    """Cross-entropy with optional per-sample output masking (pt3 lever B / task-IL).

    With `pairs` set (the list of class-pairs), each sample's logits are masked to the
    task-pair that contains its own label before the softmax, so the loss and its gradient
    only touch that sample's task classes. This is correct for replay too: a buffered
    old-task sample is masked to its own task, not the current one (a single current-task
    mask would send replayed samples' true logits to -inf). For a single-task batch (naive)
    per-sample masking is identical to a per-task mask. With `pairs=None` it is plain CE.
    """

    def __init__(self) -> None:
        self.base = nn.CrossEntropyLoss()
        self.pairs: list[tuple[int, int]] | None = None
        self._table: torch.Tensor | None = None  # (C, C) bool: allowed cols per label

    def _allowed_table(self, n_classes: int, device: torch.device) -> torch.Tensor:
        if self._table is None or self._table.size(0) != n_classes or self._table.device != device:
            table = torch.zeros(n_classes, n_classes, dtype=torch.bool, device=device)
            for a, b in self.pairs:
                table[a, a] = table[a, b] = True
                table[b, a] = table[b, b] = True
            self._table = table
        return self._table

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.pairs is not None:
            allowed = self._allowed_table(logits.size(1), logits.device)[y]  # (B, C) bool
            add = torch.zeros_like(logits)
            add[~allowed] = float("-inf")
            logits = logits + add
        return self.base(logits, y)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, allowed: list[int] | None = None
) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if allowed is not None:
                logits = logits + _logit_mask(logits.size(1), allowed, logits.device)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def _allowed_table(sequence: list, n_classes: int, device: torch.device) -> torch.Tensor:
    """(T, C) bool table: row t marks the class indices belonging to task t."""
    table = torch.zeros(len(sequence), n_classes, dtype=torch.bool, device=device)
    for t, pair in enumerate(sequence):
        for c in pair:
            table[t, c] = True
    return table


def evaluate_routed(
    model: nn.Module, g: nn.Module, loader: DataLoader, device: torch.device,
    sequence: list, true_task: int,
) -> tuple[float, float]:
    """Eval with task-inferred output routing (Iteration 8). Returns (acc, routing_acc).

    For each input: infer task t_hat = argmax g(x), mask the output logits to task t_hat's
    classes, then argmax. routing_acc = fraction routed to the correct (true) task.
    """
    model.eval(); g.eval()
    table = _allowed_table(sequence, 10, device)  # (T, C) bool
    correct = routed_right = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t_hat = g(x).argmax(dim=1)                       # (B,)
            allowed = table[t_hat]                            # (B, C) bool
            logits = model(x).clone()
            logits[~allowed] = float("-inf")
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            routed_right += (t_hat == true_task).sum().item()
            total += len(y)
    return correct / total, routed_right / total


def _plasticity_train_standard(
    model: nn.Module,
    modulator: PlasticityModulator,
    train_loader: DataLoader,
    mod_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: StandardConfig,
) -> None:
    """One epoch of standard (single-task) plasticity training (pt4 R4).

    Same lookahead / first-order meta-gradient as `_plasticity_train_task` (CL), but over the
    standard full-MNIST loader: alpha = modulator(batch); g = grad(loss).detach();
    W_fast = W.detach() - lr*(alpha*g) [differentiable in alpha]; L_meta on functional_call;
    step the modulator; commit W <- W_fast. Main net uses plain SGD (Adam-moments caveat), so
    pt4 compares this against an SGD-vanilla reference, not the Adam vanilla.
    """
    model.train()
    names = [n for n, _ in model.named_parameters()]
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        alphas = modulator.compute_alphas(x)
        factors = modulator.param_factors(alphas)
        params = list(model.parameters())
        loss = criterion(model(x), y)
        grads = [g.detach() for g in torch.autograd.grad(loss, params)]
        fast = {}
        for n, p, g in zip(names, params, grads):
            step = config.lr * (factors[n] * g) if n in factors else config.lr * g
            fast[n] = p.detach() - step
        meta_loss = criterion(torch.func.functional_call(model, fast, (x,)), y)
        mod_optimizer.zero_grad()
        meta_loss.backward()
        mod_optimizer.step()
        with torch.no_grad():
            for n, p in zip(names, params):
                p.copy_(fast[n].detach())


def train_standard(config: StandardConfig, no_wandb: bool = False) -> tuple[float, float]:
    """Train vanilla MLP on full MNIST. Returns (val_acc, test_acc)."""
    device = _device()
    seed_everything(config.seed)

    train_loader, val_loader, test_loader = get_standard_loaders(
        batch_size=config.batch_size,
        val_size=config.val_size,
    )
    model = _build_model(config, device)
    criterion = nn.CrossEntropyLoss()

    # pt4 R4: plasticity is a meta-LR mechanism with its own (SGD main + Adam modulator) loop.
    plasticity = _is_plasticity(config)
    plast_mod = plast_opt = None
    if plasticity:
        plast_mod = make_modulator(
            "plasticity", variant=config.neuromod_variant,
            learned_projection=config.neuromod_learned_projection,
            alpha_init=config.neuromod_alpha_init,
        ).to(device)
        plast_opt = torch.optim.Adam(plast_mod.parameters(), lr=config.neuromod_lr)

    # pt4 R5: importance gating installs per-parameter grad hooks (online omega), any optimizer.
    importance_state = (
        _install_importance_gates(model, config.neuromod_importance_lambda)
        if _is_importance(config) else None
    )

    if getattr(config, "optimizer", "adam") == "sgd" or plasticity:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    use_wandb = not no_wandb and _WANDB_AVAILABLE
    if use_wandb:
        _wandb.init(
            project="neuromod-cl-prototype",
            config={"lr": config.lr, "epochs": config.epochs, "batch_size": config.batch_size, "seed": config.seed},
            tags=[
                "method=standard",
                "dataset=standard_mnist",
                f"seed={config.seed}",
                f"use_neuromod={config.use_neuromod}",
                f"neuromod_variant={config.neuromod_variant if config.use_neuromod else 'none'}",
                f"neuromod_target={config.neuromod_target if config.use_neuromod else 'none'}",
            ],
        )

    val_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        if plasticity:
            _plasticity_train_standard(
                model, plast_mod, train_loader, plast_opt, criterion, device, config
            )
        else:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:>2}/{config.epochs} | val_acc={val_acc:.4f}")
        if use_wandb:
            _wandb.log({"val_acc": val_acc, "epoch": epoch})

    if importance_state is not None:
        ns = max(importance_state["n"], 1)
        print(f"[importance debug] mean gate = {importance_state['gate_sum'] / ns:.4f}, "
              f"min gate = {importance_state['gate_min']:.4f}")
        for h in importance_state["handles"]:
            h.remove()

    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    if use_wandb:
        _wandb.log({"test_acc": test_acc})
        _wandb.finish()
    return val_acc, test_acc


def _train_joint(
    model: nn.Module,
    split_mnist: SplitMNIST,
    config: CLConfig,
    device: torch.device,
    criterion: nn.Module,
) -> None:
    """Train on the union of all task data for config.epochs_per_task epochs."""
    all_datasets = [
        split_mnist.get_task_loaders(t, config.batch_size)[0].dataset
        for t in range(split_mnist.n_tasks)
    ]
    combined_loader = DataLoader(
        ConcatDataset(all_datasets), batch_size=config.batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(1, config.epochs_per_task + 1):
        model.train()
        for x, y in combined_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"  Joint epoch {epoch}/{config.epochs_per_task}")


def _plasticity_train_task(
    model: nn.Module,
    modulator: PlasticityModulator,
    train_loader: DataLoader,
    mod_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: CLConfig,
    debug: dict | None = None,
) -> None:
    """Sequential fine-tuning (naive) with per-neuron plasticity gating.

    Lookahead / first-order meta-gradient (see neuromod.PlasticityModulator):
      1. α = modulator(batch)                         [differentiable in modulator params]
      2. g = autograd.grad(task_loss, params).detach()  [first-order: g is constant]
      3. W_fast = W.detach() - lr·(α⊙g)               [differentiable in α]
      4. L_meta = CE(functional_call(model, W_fast), batch); backward → train modulator
      5. commit W ← W_fast.detach()                   [the real gated-SGD step]
    Main net uses plain SGD (Adam caveat option (a)); inner step is linear in α.
    """
    model.train()
    names = [n for n, _ in model.named_parameters()]
    for _ in range(config.epochs_per_task):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            alphas = modulator.compute_alphas(x)          # {layer: α}, requires grad
            factors = modulator.param_factors(alphas)     # {param_name: multiplier}

            params = list(model.parameters())
            loss = criterion(model(x), y)
            grads = torch.autograd.grad(loss, params)
            grads = [g.detach() for g in grads]

            fast = {}
            for n, p, g in zip(names, params, grads):
                step = config.lr * (factors[n] * g) if n in factors else config.lr * g
                fast[n] = p.detach() - step           # differentiable in α via factors[n]

            meta_loss = criterion(torch.func.functional_call(model, fast, (x,)), y)
            mod_optimizer.zero_grad()
            meta_loss.backward()
            mod_optimizer.step()

            # Commit the real gated update: W ← W_fast (detached).
            with torch.no_grad():
                for n, p in zip(names, params):
                    p.copy_(fast[n].detach())

            if debug is not None:
                with torch.no_grad():
                    a_min = min(a.min().item() for a in alphas.values())
                    a_max = max(a.max().item() for a in alphas.values())
                    a_mean = sum(a.mean().item() for a in alphas.values()) / len(alphas)
                gnorm = sum(
                    p.grad.norm().item() ** 2 for p in modulator.parameters() if p.grad is not None
                ) ** 0.5
                debug["alpha_min"] = min(debug.get("alpha_min", 1.0), a_min)
                debug["alpha_max"] = max(debug.get("alpha_max", 0.0), a_max)
                debug["alpha_mean_sum"] = debug.get("alpha_mean_sum", 0.0) + a_mean
                debug["mod_gradnorm_sum"] = debug.get("mod_gradnorm_sum", 0.0) + gnorm
                debug["n_steps"] = debug.get("n_steps", 0) + 1


def _weight_mask_driver_train_task(
    model: WeightMaskMLP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: CLConfig,
    state: dict,
    acts: dict,
    debug: dict | None = None,
) -> None:
    """Naive fine-tuning for weight_mask with a detached driver (Iteration 3).

    The driver is computed from each step's loss/logits/activations (all detached) and
    fed to the modulator for the NEXT step (lag-1), so it never sits on the main-loss
    backprop path. `state` carries the surprise EMA across steps and tasks (never reset);
    `acts` is filled by forward hooks on the hidden ReLUs (for activation_stats).
    """
    driver = config.neuromod_driver
    beta = 0.99
    model.train()
    for _ in range(config.epochs_per_task):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)                 # mask uses modulator.current_driver (prev step)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if driver == "surprise":
                    ld = loss.detach()
                    if state.get("ema") is None:
                        state["ema"] = ld.clone()
                    else:
                        state["ema"].mul_(beta).add_(ld, alpha=1 - beta)
                    d = (ld - state["ema"]).view(1)
                elif driver == "uncertainty":
                    d = predictive_entropy(logits)
                elif driver == "activation_stats":
                    d = activation_stats([acts["h1"], acts["h2"]])
                else:
                    d = None
                if d is not None:
                    model.modulator.set_driver(d.to(device))
                    if debug is not None:
                        debug["driver_abs_sum"] = debug.get("driver_abs_sum", 0.0) + float(d.abs().mean())
                        debug["n_steps"] = debug.get("n_steps", 0) + 1


def cl_train(
    config: CLConfig,
    method_name: str,
    no_wandb: bool = False,
    sequence: list | None = None,
    eval_split: str = "test",
) -> tuple[float, float]:
    """CL training loop. Returns (avg_final_acc, forgetting).

    sequence: optional task class-pair order (e.g. make_sequence(7) for the
              validation sequence). None → default test sequence.
    eval_split: 'test' (report on the official MNIST test set) or 'val' (tune on a
              held-out split carved from each task's TRAIN set; never touches test).
              In 'val' mode the train set is reduced by config.val_frac; in 'test'
              mode no split is carved (train is unchanged from the historical default).
    """
    device = _device()
    seed_everything(config.seed)

    # Only carve a validation split in val (tuning) mode; report runs use the full train set.
    effective_val_frac = config.val_frac if eval_split == "val" else 0.0
    split_mnist = SplitMNIST(sequence=sequence, val_frac=effective_val_frac)
    T = split_mnist.n_tasks

    def eval_loader_for(i: int) -> DataLoader:
        """Per-task eval loader: val split when tuning, else the task test set."""
        if eval_split == "val":
            return split_mnist.get_task_val_loader(i, config.batch_size)
        return split_mnist.get_task_loaders(i, config.batch_size)[1]
    # A[t, i] = accuracy on task i after training on task t; NaN = not yet evaluated
    A = np.full((T, T), np.nan)
    output_masking = getattr(config, "output_masking", "none")
    criterion = MaskedCE() if output_masking != "none" else nn.CrossEntropyLoss()
    model = _build_model(config, device, n_tasks=T)
    importance_state = (
        _install_importance_gates(model, config.neuromod_importance_lambda)
        if _is_importance(config) else None
    )

    use_wandb = not no_wandb and _WANDB_AVAILABLE
    if use_wandb:
        _wandb.init(
            project="neuromod-cl-prototype",
            config={
                "lr": config.lr,
                "epochs_per_task": config.epochs_per_task,
                "batch_size": config.batch_size,
                "seed": config.seed,
                "method": method_name,
            },
            tags=[
                f"method={method_name}",
                "dataset=split_mnist",
                f"seed={config.seed}",
                f"use_neuromod={config.use_neuromod}",
                f"neuromod_variant={config.neuromod_variant if config.use_neuromod else 'none'}",
                f"neuromod_target={config.neuromod_target if config.use_neuromod else 'none'}",
                f"neuromod_driver={config.neuromod_driver if config.use_neuromod else 'none'}",
                f"neuromod_granularity={config.neuromod_granularity if config.use_neuromod else 'none'}",
                f"neuromod_scope={config.neuromod_plasticity_scope if config.use_neuromod else 'none'}",
                f"neuromod_modulate_bias={config.neuromod_modulate_bias if config.use_neuromod else 'none'}",
            ],
        )

    if method_name == "joint":
        _train_joint(model, split_mnist, config, device, criterion)
        t = T - 1
        for i in range(T):
            test_loader_i = eval_loader_for(i)
            A[t, i] = evaluate(model, test_loader_i, device)
            if use_wandb:
                _wandb.log({f"acc/task_{i}": A[t, i]})
        print(f"Joint | per-task accs: [{', '.join(f'{A[t,i]:.3f}' for i in range(T))}]")
    elif _is_plasticity(config):
        # Iteration 1: plasticity target composes with the naive (sequential) loop only.
        if method_name != "naive":
            raise NotImplementedError(
                f"plasticity target composes with method=naive in Iteration 1, got {method_name!r}"
            )
        modulator = make_modulator(
            "plasticity",
            variant=config.neuromod_variant,
            learned_projection=config.neuromod_learned_projection,
            alpha_init=config.neuromod_alpha_init,
        ).to(device)
        mod_optimizer = torch.optim.Adam(modulator.parameters(), lr=config.neuromod_lr)
        if isinstance(criterion, MaskedCE):
            criterion.pairs = list(split_mnist.sequence)  # masked-loss (lever B) for plasticity
        debug: dict = {}
        for t in range(T):
            train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
            _plasticity_train_task(
                model, modulator, train_loader, mod_optimizer, criterion, device, config, debug
            )
            for i in range(t + 1):
                test_loader_i = eval_loader_for(i)
                A[t, i] = evaluate(model, test_loader_i, device)
                if use_wandb:
                    _wandb.log({f"acc/task_{i}": A[t, i], "after_task": t})
            seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
            print(f"After task {t + 1}/{T} | seen tasks: [{seen}]")
        n = max(debug.get("n_steps", 1), 1)
        print(
            f"[plasticity debug] alpha range [{debug.get('alpha_min', float('nan')):.3f}, "
            f"{debug.get('alpha_max', float('nan')):.3f}], "
            f"alpha_mean={debug.get('alpha_mean_sum', 0.0) / n:.3f}, "
            f"mod_gradnorm_mean={debug.get('mod_gradnorm_sum', 0.0) / n:.4e}"
        )
        if use_wandb:
            _wandb.log({
                "plasticity/alpha_min": debug.get("alpha_min"),
                "plasticity/alpha_max": debug.get("alpha_max"),
                "plasticity/alpha_mean": debug.get("alpha_mean_sum", 0.0) / n,
                "plasticity/mod_gradnorm_mean": debug.get("mod_gradnorm_sum", 0.0) / n,
            })
    elif _is_weight_mask_driver(config):
        # Iteration 3: weight_mask + detached driver, naive (sequential) loop only.
        if method_name != "naive":
            raise NotImplementedError(
                f"weight_mask drivers compose with method=naive in Iteration 3, got {method_name!r}"
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        if isinstance(criterion, MaskedCE):
            criterion.pairs = list(split_mnist.sequence)  # masked-loss (lever B) for weight_mask drivers
        state: dict = {"ema": None}     # surprise EMA, persists across tasks
        acts: dict = {}
        handles = []
        if config.neuromod_driver == "activation_stats":
            handles.append(model.base.net[1].register_forward_hook(
                lambda m, i, o: acts.__setitem__("h1", o.detach())))
            handles.append(model.base.net[3].register_forward_hook(
                lambda m, i, o: acts.__setitem__("h2", o.detach())))
        debug = {}
        try:
            for t in range(T):
                train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
                _weight_mask_driver_train_task(
                    model, train_loader, optimizer, criterion, device, config, state, acts, debug
                )
                for i in range(t + 1):
                    test_loader_i = eval_loader_for(i)
                    A[t, i] = evaluate(model, test_loader_i, device)
                    if use_wandb:
                        _wandb.log({f"acc/task_{i}": A[t, i], "after_task": t})
                seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
                print(f"After task {t + 1}/{T} | seen tasks: [{seen}]")
        finally:
            for h in handles:
                h.remove()
        n = max(debug.get("n_steps", 1), 1)
        print(f"[{config.neuromod_driver} debug] mean |driver| = {debug.get('driver_abs_sum', 0.0) / n:.4e}")
    elif _is_task_route(config):
        # Iteration 8 (simplified HAT / lever C): masked-loss main net + a task-inference net g,
        # used to route the output at eval. method=naive (g trained sequentially, no replay) or
        # method=er (a shared reservoir buffer trains BOTH the main net and g, so g need not forget).
        if method_name not in ("naive", "er"):
            raise NotImplementedError(f"task_route composes with naive/er, got {method_name!r}")
        use_replay = method_name == "er"
        g = TaskInferenceNet(T).to(device)
        main_opt = torch.optim.Adam(model.parameters(), lr=config.lr)
        g_opt = torch.optim.Adam(g.parameters(), lr=config.lr)
        main_crit = MaskedCE()
        main_crit.pairs = list(split_mnist.sequence)  # masked loss for the main net
        g_crit = nn.CrossEntropyLoss()
        # label -> task index (for deriving g's targets, incl. for replayed buffer samples)
        label2task = torch.zeros(10, dtype=torch.long, device=device)
        for ti, pair in enumerate(split_mnist.sequence):
            for c in pair:
                label2task[c] = ti
        buf_x: list = []; buf_y: list = []; n_seen = 0
        final_route_acc = 0.0
        for t in range(T):
            train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
            model.train(); g.train()
            for _ in range(config.epochs_per_task):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    if use_replay:
                        for xi, yi in zip(x.cpu(), y.cpu()):  # reservoir update before the step
                            n_seen += 1
                            if len(buf_x) < config.er_buffer_size:
                                buf_x.append(xi); buf_y.append(yi)
                            else:
                                j = random.randrange(n_seen)
                                if j < config.er_buffer_size:
                                    buf_x[j] = xi; buf_y[j] = yi
                        idx = random.choices(range(len(buf_x)), k=len(x))
                        bx = torch.stack([buf_x[j] for j in idx]).to(device)
                        by = torch.stack([buf_y[j] for j in idx]).to(device)
                        cx, cy = torch.cat([x, bx]), torch.cat([y, by])
                    else:
                        cx, cy = x, y
                    main_opt.zero_grad(); main_crit(model(cx), cy).backward(); main_opt.step()
                    g_opt.zero_grad(); g_crit(g(cx), label2task[cy]).backward(); g_opt.step()
            route_accs = []
            for i in range(t + 1):
                test_loader_i = eval_loader_for(i)
                acc, racc = evaluate_routed(model, g, test_loader_i, device, split_mnist.sequence, true_task=i)
                A[t, i] = acc
                route_accs.append(racc)
            final_route_acc = float(np.mean(route_accs))
            seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
            print(f"After task {t + 1}/{T} | routed accs: [{seen}] | mean routing acc={final_route_acc:.3f}")
        print(f"[task_route debug] final mean task-inference (routing) accuracy = {final_route_acc:.4f}")
    elif _is_logit_recency(config):
        # Iteration 9: logit calibrator (Iter 6) + per-class recency driver (the retention signal
        # it lacked). presence EMA over classes seen, fed to the modulator; naive or er (replay).
        if method_name not in ("naive", "er"):
            raise NotImplementedError(f"logit+recency composes with naive/er, got {method_name!r}")
        use_replay = method_name == "er"
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
        presence = torch.zeros(10, device=device)
        beta = 0.95
        buf_x: list = []; buf_y: list = []; n_seen = 0
        for t in range(T):
            present = torch.zeros(10, device=device)
            for c in split_mnist.sequence[t]:
                present[c] = 1.0
            train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
            model.train()
            for _ in range(config.epochs_per_task):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    model.modulator.set_driver(presence)        # history up to now
                    if use_replay:
                        for xi, yi in zip(x.cpu(), y.cpu()):
                            n_seen += 1
                            if len(buf_x) < config.er_buffer_size:
                                buf_x.append(xi); buf_y.append(yi)
                            else:
                                j = random.randrange(n_seen)
                                if j < config.er_buffer_size:
                                    buf_x[j] = xi; buf_y[j] = yi
                        idx = random.choices(range(len(buf_x)), k=len(x))
                        bx = torch.stack([buf_x[j] for j in idx]).to(device)
                        by = torch.stack([buf_y[j] for j in idx]).to(device)
                        cx, cy = torch.cat([x, bx]), torch.cat([y, by])
                    else:
                        cx, cy = x, y
                    opt.zero_grad(); criterion(model(cx), cy).backward(); opt.step()
                    presence = beta * presence + (1 - beta) * present
            model.modulator.set_driver(presence)
            for i in range(t + 1):
                test_loader_i = eval_loader_for(i)
                A[t, i] = evaluate(model, test_loader_i, device)
            seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
            print(f"After task {t + 1}/{T} | seen tasks: [{seen}]")
        print(f"[logit+recency debug] final presence = {presence.tolist()}")
    elif _is_consolidation(config):
        # Iteration 10: stateful boundary detector (running surprise) triggers EWC-style
        # consolidation (snapshot + importance anchor) at DETECTED boundaries, no task ID.
        # naive or er. Reports how many boundaries were detected (4 true internal boundaries).
        if method_name not in ("naive", "er"):
            raise NotImplementedError(f"consolidation composes with naive/er, got {method_name!r}")
        use_replay = method_name == "er"
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
        lam = config.neuromod_importance_lambda
        names = [n for n, _ in model.named_parameters()]
        anchors: list = []  # (theta_star, omega) EWC anchors at detected boundaries
        omega = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        ema_loss = None
        steps = last_boundary = n_boundaries = 0
        min_gap = 150       # cooldown so a single task-boundary spike triggers once
        spike = 2.0         # boundary when current loss > spike * ema_loss
        buf_x: list = []; buf_y: list = []; n_seen = 0
        for t in range(T):
            train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
            model.train()
            for _ in range(config.epochs_per_task):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    if use_replay:
                        for xi, yi in zip(x.cpu(), y.cpu()):
                            n_seen += 1
                            if len(buf_x) < config.er_buffer_size:
                                buf_x.append(xi); buf_y.append(yi)
                            else:
                                j = random.randrange(n_seen)
                                if j < config.er_buffer_size:
                                    buf_x[j] = xi; buf_y[j] = yi
                        idx = random.choices(range(len(buf_x)), k=len(x))
                        bx = torch.stack([buf_x[j] for j in idx]).to(device)
                        by = torch.stack([buf_y[j] for j in idx]).to(device)
                        cx, cy = torch.cat([x, bx]), torch.cat([y, by])
                    else:
                        cx, cy = x, y
                    task_loss = criterion(model(cx), cy)
                    pen = task_loss.new_zeros(())
                    for ts, om in anchors:
                        for n, p in model.named_parameters():
                            pen = pen + (om[n] * (p - ts[n]) ** 2).sum()
                    opt.zero_grad()
                    (task_loss + 0.5 * lam * pen).backward()
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            omega[n] += p.grad.detach() ** 2
                    opt.step()
                    ld = float(task_loss.detach())
                    ema_loss = ld if ema_loss is None else 0.99 * ema_loss + 0.01 * ld
                    steps += 1
                    if steps - last_boundary > min_gap and ema_loss > 0 and ld > spike * ema_loss:
                        anchors.append((
                            {n: p.detach().clone() for n, p in model.named_parameters()},
                            {n: omega[n].clone() for n in names},
                        ))
                        omega = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
                        last_boundary = steps; n_boundaries += 1
            for i in range(t + 1):
                test_loader_i = eval_loader_for(i)
                A[t, i] = evaluate(model, test_loader_i, device)
            seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
            print(f"After task {t + 1}/{T} | seen tasks: [{seen}]")
        print(f"[consolidation debug] boundaries detected = {n_boundaries} (4 true internal)")
    elif _is_pt5(config):
        # pt5: task-id-driven capacity allocation (disjoint/shared/learned) via the generalized
        # driver system. SGD main net throughout (Methodology 6). Composes with naive (masked-loss
        # ON) and er (masked-loss OFF). The task one-hot is set per task at train AND per task at
        # eval (oracle). gain/weight_mask gate the forward; plasticity gates the gradient.
        if method_name not in ("naive", "er"):
            raise NotImplementedError(f"pt5 driver path composes with naive/er, got {method_name!r}")
        use_replay = method_name == "er"
        target = config.neuromod_target
        # Gain modulator-only replay meta-loss (--neuromod-meta-replay): train the LEARNED gain P on a
        # buffer (per-task meta-loss) via a SEPARATE optimizer, main net stays naive. FORWARD gain only
        # (P sits in model.parameters()); standalone only (+ER already replays); learned only.
        gain_meta_replay_on = (
            target in ("activation", "hidden") and config.neuromod_projection == "learned"
            and getattr(config, "neuromod_meta_replay", False) and not use_replay
        )
        # Default SGD (Methodology 6); --optimizer adam is allowed for the FORWARD targets
        # (gain/weight_mask). NOTE: plasticity gates grads before .step(), so Adam here re-triggers
        # the Adam-moments caveat (scaled grad feeds Adam's moments) — only opt into Adam for gain/wm.
        gain_modopt = None
        if gain_meta_replay_on:
            mod_ids = {id(p) for p in _pt5_gain_modulator_params(model)}
            base_params = [p for p in model.parameters() if id(p) not in mod_ids]
            mod_params = [p for p in model.parameters() if id(p) in mod_ids]
            OptCls = torch.optim.Adam if getattr(config, "optimizer", "sgd") == "adam" else torch.optim.SGD
            optimizer = OptCls(base_params, lr=config.lr)               # main net only (P excluded)
            gain_modopt = torch.optim.Adam(mod_params, lr=config.neuromod_lr)  # trains ONLY P
        elif getattr(config, "optimizer", "sgd") == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        if isinstance(criterion, MaskedCE):
            criterion.pairs = list(split_mnist.sequence)  # per-sample masked loss (lever B), correct for ER
        # label -> task index (for the per-task gain meta-loss: forward each buffered sample's task
        # under ITS OWN gate P[j], since gain gates the FORWARD, unlike plasticity's grad gate).
        label_to_task = {c: j for j, pair in enumerate(split_mnist.sequence) for c in pair}

        # plasticity uses an external (parameter-free, fixed-projection) modulator on the raw net.
        # granularity=neuron: per-neuron alpha (scope in/out/both); granularity=synapse: per-synapse
        # gate on WEIGHT gradients only (forward untouched), layer set = --neuromod-mask-layers.
        gran = getattr(config, "neuromod_granularity", "neuron")
        sparsity_lambda = getattr(config, "neuromod_sparsity_lambda", 0.0)  # pt5 iter3 gate L1 reg
        plast_mod = None
        plast_modopt = None
        if target == "plasticity":
            bank = DriverBank(config.neuromod_drivers, T)
            if gran == "synapse":
                layers = parse_layer_list(getattr(config, "neuromod_mask_layers", ""))
                if not layers:
                    raise ValueError(
                        "pt5 per-synapse plasticity requires --neuromod-mask-layers (e.g. '0,2' or '0,2,4')"
                    )
                layer_dims = {l: (model.net[l].out_features, model.net[l].in_features) for l in layers}
                plast_mod = SynapsePlasticityDriverModulator(
                    bank, layer_dims, projection=config.neuromod_projection,
                    shared_frac=config.neuromod_shared_frac, seed=config.neuromod_proj_seed,
                    modulate_bias=getattr(config, "neuromod_modulate_bias", False),
                    init_gate=getattr(config, "neuromod_plasticity_init", 0.5),
                ).to(device)
            else:
                plast_mod = PlasticityDriverModulator(
                    bank, projection=config.neuromod_projection,
                    shared_frac=config.neuromod_shared_frac, seed=config.neuromod_proj_seed,
                    init_gate=getattr(config, "neuromod_plasticity_init", 0.5),
                ).to(device)
            # Learned projection (pt5 Iter 3): the plasticity gate is applied to grads IN PLACE below,
            # which gives a learned P no gradient. Train P by a lookahead / first-order meta-gradient
            # (per-batch, keeps the gate in the autograd graph); the fixed projections (disjoint/shared)
            # are parameter-free buffers with nothing to train, so no meta-optimizer is built for them.
            if not plast_mod.fixed:
                plast_modopt = torch.optim.Adam(plast_mod.parameters(), lr=config.neuromod_lr)

        def set_task(t: int) -> None:
            (plast_mod if target == "plasticity" else model).set_task(t)

        # Standalone modulator-only replay (SPEC iter-3): train the LEARNED plasticity P on a buffer
        # of past examples (a retention signal for the meta-loss) while the MAIN net stays naive.
        # Only when not already replaying (ER) and P is trainable (learned, not fixed).
        meta_replay_on = (
            target == "plasticity" and plast_mod is not None and not plast_mod.fixed
            and getattr(config, "neuromod_meta_replay", False) and not use_replay
        )
        need_buffer = use_replay or meta_replay_on or gain_meta_replay_on

        buf_x: list = []; buf_y: list = []; n_seen = 0
        for t in range(T):
            set_task(t)
            train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
            model.train()
            for _ in range(config.epochs_per_task):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    if need_buffer:
                        for xi, yi in zip(x.cpu(), y.cpu()):   # reservoir update before the step
                            n_seen += 1
                            if len(buf_x) < config.er_buffer_size:
                                buf_x.append(xi); buf_y.append(yi)
                            else:
                                j = random.randrange(n_seen)
                                if j < config.er_buffer_size:
                                    buf_x[j] = xi; buf_y[j] = yi
                    if use_replay:
                        idx = random.choices(range(len(buf_x)), k=len(x))
                        bx = torch.stack([buf_x[j] for j in idx]).to(device)
                        by = torch.stack([buf_y[j] for j in idx]).to(device)
                        cx, cy = torch.cat([x, bx]), torch.cat([y, by])
                    else:
                        cx, cy = x, y

                    optimizer.zero_grad()
                    loss = criterion(model(cx), cy)
                    # pt5 iter3 sparsity reg (FORWARD targets, learned P only): L1 on the projected
                    # gate, pushing each task toward a sparse active subset (toward the disjoint {0,1}).
                    # P sits in model.parameters(), so this term trains it via the main loss. Under
                    # gain meta-replay P is excluded from `optimizer`, so the L1 moves to the meta-loss.
                    if (sparsity_lambda > 0 and config.neuromod_projection == "learned"
                            and target != "plasticity" and hasattr(model, "gate_l1")
                            and not gain_meta_replay_on):
                        loss = loss + sparsity_lambda * model.gate_l1()
                    loss.backward()
                    if target == "plasticity":
                        # Gate grads by the per-task alpha; under SGD this is exact per-parameter LR
                        # scaling (no Adam-moments caveat). synapse: per-weight gate on the listed
                        # layers; neuron: per-neuron alpha broadcast (scope in/out/both). For a fixed
                        # projection the gate is a binary {0,1} buffer (no grad); for the learned
                        # projection it is sigmoid(P), differentiable in P.
                        if gran == "synapse":
                            factors = plast_mod.weight_grad_masks()
                        else:
                            factors = plast_mod.param_factors(
                                plast_mod.compute_alphas(),
                                scope=config.neuromod_plasticity_scope,
                                layers=tuple(parse_layer_list(config.neuromod_plasticity_layers)),
                            )
                        if plast_modopt is not None:
                            # Learned P (pt5 Iter 3): the in-place grad-gate below severs P's autograd
                            # edge, so train P by a lookahead / first-order meta-gradient (mirrors the
                            # legacy PlasticityModulator lookahead). W_fast = W - lr*(gate⊙g) with g
                            # detached (differentiable in P via the gate); a meta-loss on the SAME
                            # (replay-augmented for ER) batch trains ONLY P (main net detached inside
                            # functional_call). For neurom+ER, cx/cy already carry replayed past-task
                            # samples, so the meta-loss is the SPEC's modulator-only replay meta-loss.
                            raw_g = {n: p.grad.detach() for n, p in model.named_parameters()
                                     if p.grad is not None}
                            fast = {}
                            for n, p in model.named_parameters():
                                if n in factors and n in raw_g:
                                    fast[n] = p.detach() - config.lr * (factors[n] * raw_g[n])
                                else:
                                    fast[n] = p.detach()
                            # meta-loss batch: for +ER, cx already carries replay; for standalone with
                            # meta_replay_on, augment the current batch with a buffer sample so the
                            # meta-loss (which trains ONLY P) gets a retention signal on past tasks
                            # while the main step below stays naive (SPEC modulator-only replay).
                            if meta_replay_on and buf_x:
                                mi = random.choices(range(len(buf_x)), k=len(x))
                                mbx = torch.stack([buf_x[j] for j in mi]).to(device)
                                mby = torch.stack([buf_y[j] for j in mi]).to(device)
                                meta_x, meta_y = torch.cat([x, mbx]), torch.cat([y, mby])
                            else:
                                meta_x, meta_y = cx, cy
                            meta_loss = criterion(torch.func.functional_call(model, fast, (meta_x,)), meta_y)
                            # pt5 iter3 sparsity reg: L1 on the per-task gate, added to the meta-loss
                            # (the loss that trains P), pushing alphas toward a sparse plastic subset.
                            if sparsity_lambda > 0:
                                meta_loss = meta_loss + sparsity_lambda * plast_mod.gate_l1()
                            plast_modopt.zero_grad()
                            meta_loss.backward()
                            plast_modopt.step()
                            # commit the real gated step with the SAME (pre-update) gate, detached
                            factors = {n: v.detach() for n, v in factors.items()}
                        with torch.no_grad():
                            for n, p in model.named_parameters():
                                if p.grad is not None and n in factors:
                                    p.grad.mul_(factors[n])
                    optimizer.step()   # main net (P excluded when gain_meta_replay_on)

                    if gain_meta_replay_on:
                        # Modulator-only replay meta-loss for FORWARD gain: train ONLY P on a per-task
                        # meta-loss, main net untouched (its step above stayed naive). Each seen task
                        # j is forwarded under ITS OWN gate P[j] (gain gates the forward, so a buffered
                        # task-j sample MUST use P[j], not P[t]), so the past-task rows get a retention
                        # signal; current task t uses the fresh batch. Only P[j] gets a gradient (the
                        # one-hot zeroes the other rows).
                        gain_modopt.zero_grad()
                        meta_loss = 0.0; n_j = 0
                        for j in range(t + 1):
                            if j == t:
                                mbx, mby = x, y                       # fresh current-task batch
                            else:
                                js = [i for i in range(len(buf_x))
                                      if label_to_task[int(buf_y[i])] == j]
                                if not js:
                                    continue
                                mi = random.choices(js, k=len(x))
                                mbx = torch.stack([buf_x[i] for i in mi]).to(device)
                                mby = torch.stack([buf_y[i] for i in mi]).to(device)
                            model.set_task(j)
                            meta_loss = meta_loss + criterion(model(mbx), mby)
                            n_j += 1
                        meta_loss = meta_loss / max(n_j, 1)
                        model.set_task(t)
                        if sparsity_lambda > 0:
                            meta_loss = meta_loss + sparsity_lambda * model.gate_l1()
                        meta_loss.backward()                          # trains only P (in gain_modopt)
                        gain_modopt.step()
                        model.set_task(t)                             # restore for the next main step

            for i in range(t + 1):
                set_task(i)                          # oracle: each task evaluated under its own gate
                test_loader_i = eval_loader_for(i)
                # taskil: also mask eval to task i's classes (2-way), matching the non-pt5 branch;
                # loss/none: allowed=None -> class-IL 10-way eval (prior pt5 default, unchanged).
                allowed_i = list(split_mnist.sequence[i]) if output_masking == "taskil" else None
                A[t, i] = evaluate(model, test_loader_i, device, allowed=allowed_i)
                if use_wandb:
                    _wandb.log({f"acc/task_{i}": A[t, i], "after_task": t})
            seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
            print(f"After task {t + 1}/{T} | seen tasks: [{seen}]")
        gran_dbg = getattr(config, "neuromod_granularity", "neuron")
        neuron_plast = target == "plasticity" and gran_dbg == "neuron"
        scope_dbg = config.neuromod_plasticity_scope if neuron_plast else "-"
        if neuron_plast:
            layers_dbg = config.neuromod_plasticity_layers
        elif target in ("activation", "hidden") and gran_dbg == "neuron":
            layers_dbg = config.neuromod_gain_layers
        else:
            layers_dbg = config.neuromod_mask_layers
        plast_init_dbg = config.neuromod_plasticity_init if target == "plasticity" else "-"
        print(f"[pt5 debug] target={target} granularity={gran_dbg} scope={scope_dbg} "
              f"layers={layers_dbg} projection={config.neuromod_projection} "
              f"modulate_bias={getattr(config, 'neuromod_modulate_bias', False)} "
              f"plast_init={plast_init_dbg} sparsity_lambda={sparsity_lambda} "
              f"meta_replay={meta_replay_on or gain_meta_replay_on} "
              f"optimizer={getattr(config, 'optimizer', 'sgd')} "
              f"driver={config.neuromod_drivers} method={method_name} masking={output_masking}")
    else:
        method = make_cl_method(method_name)
        if getattr(config, "optimizer", "adam") == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        if isinstance(criterion, MaskedCE):
            # Per-sample masking by label->task-pair (correct for naive and for ER replay).
            criterion.pairs = list(split_mnist.sequence)
        for t in range(T):
            train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
            method.train_task(t, model, train_loader, optimizer, criterion, device, config)
            method.on_task_end(t, model, train_loader, device, config)
            for i in range(t + 1):
                test_loader_i = eval_loader_for(i)
                # task-IL: also mask eval to task i's classes; loss/none: class-IL eval over all 10.
                allowed_i = list(split_mnist.sequence[i]) if output_masking == "taskil" else None
                A[t, i] = evaluate(model, test_loader_i, device, allowed=allowed_i)
                if use_wandb:
                    _wandb.log({f"acc/task_{i}": A[t, i], "after_task": t})
            seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
            print(f"After task {t + 1}/{T} | seen tasks: [{seen}]")

    if importance_state is not None:
        ns = max(importance_state["n"], 1)
        print(f"[importance debug] mean gate = {importance_state['gate_sum'] / ns:.4f}, "
              f"min gate = {importance_state['gate_min']:.4f}")
        for h in importance_state["handles"]:
            h.remove()

    # avg_final_acc = mean over all tasks of final row
    avg_final_acc = float(np.nanmean(A[T - 1, :]))

    # forgetting = mean over all tasks of (peak acc seen - final acc)
    # Per spec: mean over i < T; last task always contributes 0
    forget_vals = []
    for i in range(T):
        col = [A[t, i] for t in range(i, T) if not np.isnan(A[t, i])]
        if col:
            forget_vals.append(max(col) - A[T - 1, i])
    forgetting = float(np.mean(forget_vals)) if forget_vals else 0.0

    print(f"\navg_final_acc={avg_final_acc:.4f} | forgetting={forgetting:.4f}")
    if use_wandb:
        _wandb.log({"avg_final_acc": avg_final_acc, "forgetting": forgetting})
        _wandb.finish()

    return avg_final_acc, forgetting


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuromodulation prototype training")
    parser.add_argument("--standard", action="store_true", help="Run standard MNIST training")
    parser.add_argument("--method", choices=["naive", "joint", "ewc", "er"], default="naive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--val", action="store_true",
                        help="CL tuning mode: eval on a held-out val split (carved from TRAIN) using the "
                             "validation task order (make_sequence(val_sequence_seed)); never touches the test set")
    # Hyperparameter overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epochs-per-task", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "sgd"])
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--er-buffer-size", type=int, default=None)
    parser.add_argument("--output-masking", type=str, default=None, choices=["none", "loss", "taskil"])
    # Neuromod flags
    parser.add_argument("--use-neuromod", action="store_true")
    parser.add_argument("--neuromod-variant", type=str, default=None)
    parser.add_argument("--neuromod-target", type=str, default=None)
    parser.add_argument("--neuromod-driver", type=str, default=None)
    parser.add_argument("--neuromod-lr", type=float, default=None)
    parser.add_argument("--neuromod-alpha-init", type=float, default=None)
    parser.add_argument("--neuromod-mask-layer", type=int, default=None)
    parser.add_argument("--neuromod-mask-layers", type=str, default=None,
                        help="weight_mask: comma-sep net linear indices to mask together, e.g. '0,2,4' "
                             "(incl. output head net.4). Empty/omitted = single --neuromod-mask-layer.")
    parser.add_argument("--neuromod-mask-rank", type=int, default=None)
    parser.add_argument("--neuromod-mask-init", type=float, default=None)
    parser.add_argument("--neuromod-stateful-hidden", type=int, default=None)
    parser.add_argument("--neuromod-importance-lambda", type=float, default=None)
    parser.add_argument("--neuromod-gain-gate", type=str, default=None,
                        choices=["last_hidden", "two_hidden", "last_hidden_output", "two_hidden_output"])
    parser.add_argument("--neuromod-gain-bounded", action="store_true")
    parser.add_argument("--neuromod-learned-projection", action="store_true")
    # pt5 generalized driver system (CL only). Non-empty --neuromod-drivers selects the new path.
    parser.add_argument("--neuromod-drivers", type=str, default=None,
                        help="pt5: comma-sep name=mechanism, e.g. 'task_id=onehot'. Empty = legacy/off path.")
    parser.add_argument("--neuromod-context", type=str, default=None, choices=["image", "none"])
    parser.add_argument("--neuromod-projection", type=str, default=None,
                        choices=["disjoint", "shared", "learned"])
    parser.add_argument("--neuromod-shared-frac", type=float, default=None)
    parser.add_argument("--neuromod-proj-seed", type=int, default=None)
    parser.add_argument("--neuromod-gain-form", type=str, default=None, choices=["unbounded", "bounded01"])
    parser.add_argument("--neuromod-granularity", type=str, default=None, choices=["neuron", "synapse"],
                        help="pt5 activation/plasticity: neuron (per-unit) | synapse (per-weight). weight_mask is always synapse.")
    parser.add_argument("--neuromod-plasticity-scope", type=str, default=None, choices=["both", "in", "out"],
                        help="per-neuron plasticity only: gate a unit's incoming (in), outgoing (out), or both weight sets.")
    parser.add_argument("--neuromod-plasticity-layers", type=str, default=None,
                        help="per-neuron plasticity: comma-sep net.<idx> weight layers to gate, e.g. '2,4'. scope picks the side per layer.")
    parser.add_argument("--neuromod-gain-layers", type=str, default=None,
                        help="per-neuron gain: comma-sep activations to gate (0=h0, 2=h1, 4=output logits), e.g. '0,2,4'.")
    parser.add_argument("--neuromod-modulate-bias", action="store_true",
                        help="pt5 per-synapse gain/weight_mask/plasticity: also gate per-neuron biases "
                             "(independent P_bias per layer). Default off = biases fully plastic (parity).")
    parser.add_argument("--neuromod-plasticity-init", type=float, default=None,
                        help="pt5 iter3 LEARNED plasticity: initial per-side gate alpha (0.5 = iter3 "
                             "default; higher starts more units plastic via a logit bias).")
    parser.add_argument("--neuromod-sparsity-lambda", type=float, default=None,
                        help="pt5 iter3 LEARNED projections: L1 penalty on the projected gate "
                             "(lambda*mean|gate|), toward a sparse per-task active subset. 0 = off.")
    parser.add_argument("--neuromod-meta-replay", action="store_true",
                        help="pt5 iter3 LEARNED plasticity STANDALONE: train P on a modulator-only "
                             "replay buffer (retention meta-loss); the main net stays naive.")
    args = parser.parse_args()

    if args.standard:
        config = StandardConfig(seed=args.seed)
        if args.lr is not None:
            config.lr = args.lr
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.optimizer is not None:
            config.optimizer = args.optimizer
        if args.use_neuromod:
            config.use_neuromod = True
        if args.neuromod_variant is not None:
            config.neuromod_variant = args.neuromod_variant
        if args.neuromod_target is not None:
            config.neuromod_target = args.neuromod_target
        if args.neuromod_driver is not None:
            config.neuromod_driver = args.neuromod_driver
        if args.neuromod_lr is not None:
            config.neuromod_lr = args.neuromod_lr
        if args.neuromod_alpha_init is not None:
            config.neuromod_alpha_init = args.neuromod_alpha_init
        if args.neuromod_mask_layer is not None:
            config.neuromod_mask_layer = args.neuromod_mask_layer
        if args.neuromod_mask_layers is not None:
            config.neuromod_mask_layers = args.neuromod_mask_layers
        if args.neuromod_mask_rank is not None:
            config.neuromod_mask_rank = args.neuromod_mask_rank
        if args.neuromod_mask_init is not None:
            config.neuromod_mask_init = args.neuromod_mask_init
        if args.neuromod_stateful_hidden is not None:
            config.neuromod_stateful_hidden = args.neuromod_stateful_hidden
        if args.neuromod_importance_lambda is not None:
            config.neuromod_importance_lambda = args.neuromod_importance_lambda
        if args.neuromod_gain_gate is not None:
            config.neuromod_gain_gate = args.neuromod_gain_gate
        if args.neuromod_gain_bounded:
            config.neuromod_gain_bounded = True
        if args.neuromod_learned_projection:
            config.neuromod_learned_projection = True
        if args.neuromod_granularity is not None:
            config.neuromod_granularity = args.neuromod_granularity
        if args.neuromod_plasticity_scope is not None:
            config.neuromod_plasticity_scope = args.neuromod_plasticity_scope
        if args.neuromod_plasticity_layers is not None:
            config.neuromod_plasticity_layers = args.neuromod_plasticity_layers
        if args.neuromod_gain_layers is not None:
            config.neuromod_gain_layers = args.neuromod_gain_layers
        if args.neuromod_modulate_bias:
            config.neuromod_modulate_bias = True
        if args.neuromod_plasticity_init is not None:
            config.neuromod_plasticity_init = args.neuromod_plasticity_init
        if args.neuromod_sparsity_lambda is not None:
            config.neuromod_sparsity_lambda = args.neuromod_sparsity_lambda
        if args.neuromod_meta_replay:
            config.neuromod_meta_replay = True
        train_standard(config, no_wandb=args.no_wandb)
    else:
        config = CLConfig(seed=args.seed)
        if args.lr is not None:
            config.lr = args.lr
        if args.epochs_per_task is not None:
            config.epochs_per_task = args.epochs_per_task
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.optimizer is not None:
            config.optimizer = args.optimizer
        if args.ewc_lambda is not None:
            config.ewc_lambda = args.ewc_lambda
        if args.er_buffer_size is not None:
            config.er_buffer_size = args.er_buffer_size
        if args.output_masking is not None:
            config.output_masking = args.output_masking
        if args.use_neuromod:
            config.use_neuromod = True
        if args.neuromod_variant is not None:
            config.neuromod_variant = args.neuromod_variant
        if args.neuromod_target is not None:
            config.neuromod_target = args.neuromod_target
        if args.neuromod_driver is not None:
            config.neuromod_driver = args.neuromod_driver
        if args.neuromod_lr is not None:
            config.neuromod_lr = args.neuromod_lr
        if args.neuromod_alpha_init is not None:
            config.neuromod_alpha_init = args.neuromod_alpha_init
        if args.neuromod_mask_layer is not None:
            config.neuromod_mask_layer = args.neuromod_mask_layer
        if args.neuromod_mask_layers is not None:
            config.neuromod_mask_layers = args.neuromod_mask_layers
        if args.neuromod_mask_rank is not None:
            config.neuromod_mask_rank = args.neuromod_mask_rank
        if args.neuromod_mask_init is not None:
            config.neuromod_mask_init = args.neuromod_mask_init
        if args.neuromod_stateful_hidden is not None:
            config.neuromod_stateful_hidden = args.neuromod_stateful_hidden
        if args.neuromod_importance_lambda is not None:
            config.neuromod_importance_lambda = args.neuromod_importance_lambda
        if args.neuromod_gain_gate is not None:
            config.neuromod_gain_gate = args.neuromod_gain_gate
        if args.neuromod_gain_bounded:
            config.neuromod_gain_bounded = True
        if args.neuromod_learned_projection:
            config.neuromod_learned_projection = True
        # pt5 driver-system overrides (CL only)
        if args.neuromod_drivers is not None:
            config.neuromod_drivers = args.neuromod_drivers
        if args.neuromod_context is not None:
            config.neuromod_context = args.neuromod_context
        if args.neuromod_projection is not None:
            config.neuromod_projection = args.neuromod_projection
        if args.neuromod_shared_frac is not None:
            config.neuromod_shared_frac = args.neuromod_shared_frac
        if args.neuromod_proj_seed is not None:
            config.neuromod_proj_seed = args.neuromod_proj_seed
        if args.neuromod_gain_form is not None:
            config.neuromod_gain_form = args.neuromod_gain_form
        if args.neuromod_granularity is not None:
            config.neuromod_granularity = args.neuromod_granularity
        if args.neuromod_plasticity_scope is not None:
            config.neuromod_plasticity_scope = args.neuromod_plasticity_scope
        if args.neuromod_plasticity_layers is not None:
            config.neuromod_plasticity_layers = args.neuromod_plasticity_layers
        if args.neuromod_gain_layers is not None:
            config.neuromod_gain_layers = args.neuromod_gain_layers
        if args.neuromod_modulate_bias:
            config.neuromod_modulate_bias = True
        if args.neuromod_plasticity_init is not None:
            config.neuromod_plasticity_init = args.neuromod_plasticity_init
        if args.neuromod_sparsity_lambda is not None:
            config.neuromod_sparsity_lambda = args.neuromod_sparsity_lambda
        if args.neuromod_meta_replay:
            config.neuromod_meta_replay = True
        if args.val:
            # Tuning: validation task order + held-out val split. Report runs (no --val)
            # use the default task order and the official test set.
            eval_split = "val"
            sequence = make_sequence(config.val_sequence_seed)
        else:
            eval_split = "test"
            sequence = None
        cl_train(config, args.method, no_wandb=args.no_wandb, sequence=sequence, eval_split=eval_split)


if __name__ == "__main__":
    main()
