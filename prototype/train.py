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
from prototype.data import SplitMNIST, get_standard_loaders
from prototype.methods import make_cl_method
from prototype.model import MLP
from prototype.neuromod import (
    LogitModulatedMLP,
    ModulatedMLP,
    PlasticityModulator,
    WeightMaskMLP,
    activation_stats,
    make_modulator,
    predictive_entropy,
)

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _is_plasticity(config) -> bool:
    return config.use_neuromod and config.neuromod_target == "plasticity"


def _is_weight_mask_driver(config) -> bool:
    return (
        config.use_neuromod
        and config.neuromod_target == "weight_mask"
        and config.neuromod_driver != "none"
    )


def _build_model(config, device: torch.device) -> nn.Module:
    """Create vanilla MLP or ModulatedMLP depending on config.

    Plasticity target keeps the base MLP unwrapped (forward untouched); its
    modulator lives outside the model and is handled in the training loop.
    """
    model = MLP().to(device)
    if not config.use_neuromod or _is_plasticity(config):
        return model
    if config.neuromod_target == "weight_mask":
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
        mod = make_modulator("logit", variant=config.neuromod_variant)
        return LogitModulatedMLP(model, mod).to(device)
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


def train_standard(config: StandardConfig, no_wandb: bool = False) -> tuple[float, float]:
    """Train vanilla MLP on full MNIST. Returns (val_acc, test_acc)."""
    device = _device()
    seed_everything(config.seed)

    train_loader, val_loader, test_loader = get_standard_loaders(
        batch_size=config.batch_size,
        val_size=config.val_size,
    )
    model = _build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

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
) -> tuple[float, float]:
    """CL training loop. Returns (avg_final_acc, forgetting).

    sequence: optional task class-pair order (e.g. make_sequence(7) for the
              validation sequence). None → default test sequence.
    """
    device = _device()
    seed_everything(config.seed)

    split_mnist = SplitMNIST(sequence=sequence)
    T = split_mnist.n_tasks
    # A[t, i] = accuracy on task i after training on task t; NaN = not yet evaluated
    A = np.full((T, T), np.nan)
    output_masking = getattr(config, "output_masking", "none")
    criterion = MaskedCE() if output_masking != "none" else nn.CrossEntropyLoss()
    model = _build_model(config, device)

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
            ],
        )

    if method_name == "joint":
        _train_joint(model, split_mnist, config, device, criterion)
        t = T - 1
        for i in range(T):
            _, test_loader_i = split_mnist.get_task_loaders(i, config.batch_size)
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
        debug: dict = {}
        for t in range(T):
            train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
            _plasticity_train_task(
                model, modulator, train_loader, mod_optimizer, criterion, device, config, debug
            )
            for i in range(t + 1):
                _, test_loader_i = split_mnist.get_task_loaders(i, config.batch_size)
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
                    _, test_loader_i = split_mnist.get_task_loaders(i, config.batch_size)
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
                _, test_loader_i = split_mnist.get_task_loaders(i, config.batch_size)
                # task-IL: also mask eval to task i's classes; loss/none: class-IL eval over all 10.
                allowed_i = list(split_mnist.sequence[i]) if output_masking == "taskil" else None
                A[t, i] = evaluate(model, test_loader_i, device, allowed=allowed_i)
                if use_wandb:
                    _wandb.log({f"acc/task_{i}": A[t, i], "after_task": t})
            seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
            print(f"After task {t + 1}/{T} | seen tasks: [{seen}]")

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
    # Hyperparameter overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epochs-per-task", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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
    parser.add_argument("--neuromod-mask-rank", type=int, default=None)
    parser.add_argument("--neuromod-mask-init", type=float, default=None)
    parser.add_argument("--neuromod-stateful-hidden", type=int, default=None)
    parser.add_argument("--neuromod-learned-projection", action="store_true")
    args = parser.parse_args()

    if args.standard:
        config = StandardConfig(seed=args.seed)
        if args.lr is not None:
            config.lr = args.lr
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
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
        if args.neuromod_mask_rank is not None:
            config.neuromod_mask_rank = args.neuromod_mask_rank
        if args.neuromod_mask_init is not None:
            config.neuromod_mask_init = args.neuromod_mask_init
        if args.neuromod_stateful_hidden is not None:
            config.neuromod_stateful_hidden = args.neuromod_stateful_hidden
        if args.neuromod_learned_projection:
            config.neuromod_learned_projection = True
        train_standard(config, no_wandb=args.no_wandb)
    else:
        config = CLConfig(seed=args.seed)
        if args.lr is not None:
            config.lr = args.lr
        if args.epochs_per_task is not None:
            config.epochs_per_task = args.epochs_per_task
        if args.batch_size is not None:
            config.batch_size = args.batch_size
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
        if args.neuromod_mask_rank is not None:
            config.neuromod_mask_rank = args.neuromod_mask_rank
        if args.neuromod_mask_init is not None:
            config.neuromod_mask_init = args.neuromod_mask_init
        if args.neuromod_stateful_hidden is not None:
            config.neuromod_stateful_hidden = args.neuromod_stateful_hidden
        if args.neuromod_learned_projection:
            config.neuromod_learned_projection = True
        cl_train(config, args.method, no_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
