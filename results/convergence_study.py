"""Convergence-speed study (user-requested, 1 seed).

Question: does neuromodulation change *convergence speed* (not just final accuracy) vs a naive
baseline, in the standard and CL regimes, for the pt1->iter3 mechanisms — and does it matter
whether the modulator is bottlenecked (with a random or a learned projection) or non-bottlenecked?

For each variant we log, per epoch, train loss/acc and val loss/acc (standard) or current-task
train loss/acc + current-task test acc (CL), plus a per-step train-loss trace, so the convergence
curve is fully reconstructable. Each mechanism also logs a probe that contrasts the modulated and
the un-modulated (naive) quantity:
  - gain        : the gain vector g (naive = 1): mean, mean|g-1|, std
  - plasticity  : per-neuron alpha and the effective LR base_lr*alpha (naive = base_lr, alpha=1)
  - weight_mask : the mask M and M⊙W vs the raw W (mean M, frac masked, ||M⊙W||/||W||, mean|M⊙W-W|)

Mechanisms x projection axis (the bottleneck/projection axis is only fully defined for gain;
plasticity and weight_mask get a non-bottleneck form built for this study, see neuromod.py):
  gain        : bottleneck+random proj | bottleneck+learned proj | non-bottleneck (direct_gain)
  plasticity  : bottleneck+random proj | bottleneck+learned proj | non-bottleneck (direct_plasticity)
  weight_mask : bottleneck (learned head) | non-bottleneck (direct_weight_mask)
  drivers     : weight_mask(bottleneck) + surprise|uncertainty|activation_stats  [CL only; drivers
                are N/A in single-task standard, see pt4 notes]

Budget (1 seed): standard epochs=10 lr=1e-3; CL epochs_per_task=5 lr=1e-3 (naive). Writes a
readable log to stdout and a machine-readable convergence_study.json for plotting.

Usage: uv run python results/convergence_study.py [--regime standard|cl|both] [--variants a,b,...]
       [--standard-epochs N] [--cl-epochs N] [--max-train-batches N] [--seed N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from prototype.data import SplitMNIST, get_standard_loaders
from prototype.model import MLP
from prototype.neuromod import (
    DirectGainModulator,
    DirectPlasticityModulator,
    DirectWeightMaskModulator,
    GainModulator,
    ModulatedMLP,
    PlasticityModulator,
    WeightMaskMLP,
    WeightMaskModulator,
    activation_stats,
    driver_dim,
    predictive_entropy,
)
from prototype.train import _device, seed_everything

MASK_LAYER = 2          # iter2 masked net.2 (the 400x400 second linear)
HIDDEN = 400
SEED = 42


# ---------------------------------------------------------------------------
# Variant catalogue. family decides the loop; build() returns (model, extra).
# 'extra' holds anything the loop/probe needs (plast modulator, base lr, etc.).
# ---------------------------------------------------------------------------
def make_base() -> MLP:
    return MLP()


def build_variant(name: str, device: torch.device):
    """Return dict: model, family, optimizer, lr, base_lr, probe (callable or None), driver."""
    base = make_base()
    if name == "naive_adam":
        return {"model": base, "family": "plain", "opt": "adam", "lr": 1e-3, "probe": None, "driver": None}
    if name == "naive_sgd":
        return {"model": base, "family": "plain", "opt": "sgd", "lr": 1e-3, "probe": None, "driver": None}

    # ---- gain (activation target) ----
    if name in ("gain_random", "gain_learned", "gain_direct", "gain_direct_bounded"):
        if name in ("gain_direct", "gain_direct_bounded"):
            # bounded=True: g = 1 + tanh(m) in [0,2] (vs unbounded 1+m for gain_direct).
            mod = DirectGainModulator(gate_hidden=(0, 1), gate_output=False,
                                      bounded=(name == "gain_direct_bounded"))
        else:
            mod = GainModulator(learned_projection=(name == "gain_learned"))
        model = ModulatedMLP(base, mod)

        def probe(x, _m=mod):
            with torch.no_grad():
                h = torch.ones(x.size(0), HIDDEN, device=x.device)
                xf = x.view(x.size(0), -1)
                g = torch.cat([_m.modulate(h, xf, 0), _m.modulate(h, xf, 1)], dim=1)
                return {"gain_mean": g.mean().item(), "gain_abs_dev_from_1": (g - 1).abs().mean().item(),
                        "gain_std": g.std().item()}
        return {"model": model, "family": "plain", "opt": "adam", "lr": 1e-3, "probe": probe, "driver": None}

    # ---- plasticity (per-neuron LR gate, lookahead loop, SGD main net) ----
    if name in ("plast_random", "plast_learned", "plast_direct"):
        if name == "plast_direct":
            mod = DirectPlasticityModulator(hidden_dim=HIDDEN)
        else:
            mod = PlasticityModulator(learned_projection=(name == "plast_learned"))
        mod = mod.to(device)

        def probe(x, _m=mod, _lr=1e-3):
            with torch.no_grad():
                a = _m.compute_alphas(x.view(x.size(0), -1))
                av = torch.cat([a[0], a[1]])
                return {"alpha_mean": av.mean().item(), "alpha_min": av.min().item(),
                        "alpha_max": av.max().item(), "eff_lr_mean": _lr * av.mean().item(),
                        "base_lr": _lr}
        return {"model": base, "family": "plasticity", "opt": "sgd", "lr": 1e-3,
                "plast_mod": mod, "mod_lr": 1e-3, "probe": probe, "driver": None}

    # ---- weight_mask (per-synapse mask, forward-graph, ordinary backprop) ----
    if name in ("wm_bottleneck", "wm_direct", "wm_random"):
        d_out, d_in = base.net[MASK_LAYER].out_features, base.net[MASK_LAYER].in_features
        if name == "wm_direct":
            mod = DirectWeightMaskModulator(d_out=d_out, d_in=d_in)
        elif name == "wm_random":
            mod = WeightMaskModulator(d_out=d_out, d_in=d_in, learned_projection=False)
        else:
            mod = WeightMaskModulator(d_out=d_out, d_in=d_in)  # learned head (default)
        model = WeightMaskMLP(base, mod, layer_idx=MASK_LAYER)

        def probe(x, _m=mod, _model=model):
            with torch.no_grad():
                M = _m.compute_mask(x.view(x.size(0), -1))
                W = _model.base.net[MASK_LAYER].weight
                MW = M * W
                return {"mask_mean": M.mean().item(), "mask_frac_below_0.5": (M < 0.5).float().mean().item(),
                        "masked_over_raw_norm": (MW.norm() / W.norm()).item(),
                        "mean_abs_weight_change": (MW - W).abs().mean().item()}
        return {"model": model, "family": "plain", "opt": "adam", "lr": 1e-3, "probe": probe, "driver": None}

    # ---- drivers (CL iter3): weight_mask bottleneck + a detached lag-1 driver ----
    if name in ("wm_surprise", "wm_uncertainty", "wm_activation_stats"):
        drv = name.split("wm_")[1]
        d_out, d_in = base.net[MASK_LAYER].out_features, base.net[MASK_LAYER].in_features
        mod = WeightMaskModulator(d_out=d_out, d_in=d_in, driver_dim=driver_dim(drv))
        model = WeightMaskMLP(base, mod, layer_idx=MASK_LAYER)

        def probe(x, _m=mod, _model=model):
            with torch.no_grad():
                M = _m.compute_mask(x.view(x.size(0), -1))
                W = _model.base.net[MASK_LAYER].weight
                MW = M * W
                return {"mask_mean": M.mean().item(), "mask_frac_below_0.5": (M < 0.5).float().mean().item(),
                        "masked_over_raw_norm": (MW.norm() / W.norm()).item()}
        return {"model": model, "family": "plain", "opt": "adam", "lr": 1e-3, "probe": probe, "driver": drv}

    raise ValueError(f"unknown variant {name!r}")


ALL_VARIANTS_STANDARD = [
    "naive_adam", "naive_sgd",
    "gain_random", "gain_learned", "gain_direct", "gain_direct_bounded",
    "plast_random", "plast_learned", "plast_direct",
    "wm_bottleneck", "wm_random", "wm_direct",
]
ALL_VARIANTS_CL = ALL_VARIANTS_STANDARD + ["wm_surprise", "wm_uncertainty", "wm_activation_stats"]


# ---------------------------------------------------------------------------
# Eval helper + driver helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_loss_acc(model: nn.Module, loader, device) -> tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    ls = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ls += ce(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return ls / n, correct / n


def seen_test_loader(split, t, batch_size=256):
    """Pooled test set of tasks 0..t (the digits seen so far), for class-IL seen-acc per epoch."""
    ds = ConcatDataset([split.get_task_loaders(i, 64)[1].dataset for i in range(t + 1)])
    return DataLoader(ds, batch_size=batch_size)


def _attach_act_hooks(model, acts):
    base = model.base if hasattr(model, "base") else model
    h1 = base.net[1].register_forward_hook(lambda m, i, o: acts.__setitem__("h1", o.detach()))
    h2 = base.net[3].register_forward_hook(lambda m, i, o: acts.__setitem__("h2", o.detach()))
    return [h1, h2]


def _update_driver(model, driver, loss, logits, state, acts):
    """Set the lag-1 detached driver on the modulator for the next step (iter3 logic)."""
    with torch.no_grad():
        if driver == "surprise":
            ld = loss.detach()
            state["ema"] = ld.clone() if state.get("ema") is None else state["ema"].mul_(0.99).add_(ld, alpha=0.01)
            d = (ld - state["ema"]).view(1)
        elif driver == "uncertainty":
            d = predictive_entropy(logits)
        elif driver == "activation_stats":
            d = activation_stats([acts["h1"], acts["h2"]])
        else:
            return
        model.modulator.set_driver(d.to(loss.device))
        state["driver_abs_sum"] = state.get("driver_abs_sum", 0.0) + float(d.abs().mean())
        state["n"] = state.get("n", 0) + 1


# ---------------------------------------------------------------------------
# Standard loops
# ---------------------------------------------------------------------------
def run_standard(name, spec, loaders, probe_x, device, epochs, max_batches):
    train_loader, val_loader, test_loader = loaders
    model = spec["model"].to(device)
    if spec["family"] == "plasticity":
        return run_standard_plasticity(name, spec, loaders, probe_x, device, epochs, max_batches)

    opt = (torch.optim.SGD if spec["opt"] == "sgd" else torch.optim.Adam)(model.parameters(), lr=spec["lr"])
    ce = nn.CrossEntropyLoss()
    driver, state, acts, hooks = spec["driver"], {"ema": None}, {}, []
    if driver == "activation_stats":
        hooks = _attach_act_hooks(model, acts)
    epochs_log, step_loss = [], []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ls = correct = n = 0
        for bi, (x, y) in enumerate(train_loader):
            if max_batches and bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            step_loss.append(loss.item())
            ls += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); n += len(y)
            if driver:
                _update_driver(model, driver, loss, logits, state, acts)
        tr_loss, tr_acc = ls / n, correct / n
        val_loss, val_acc = eval_loss_acc(model, val_loader, device)
        probe = spec["probe"](probe_x) if spec["probe"] else {}
        epochs_log.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                           "val_loss": val_loss, "val_acc": val_acc, "probe": probe})
        print(f"  [{name}] ep{ep:>2}/{epochs} tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
              + (f" | {probe}" if probe else ""))
    for h in hooks:
        h.remove()
    test_loss, test_acc = eval_loss_acc(model, test_loader, device)
    print(f"  [{name}] test_acc={test_acc:.4f}  ({time.time()-t0:.1f}s)")
    return {"epochs": epochs_log, "step_loss": step_loss, "test_acc": test_acc,
            "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "seconds": time.time() - t0}


def run_standard_plasticity(name, spec, loaders, probe_x, device, epochs, max_batches):
    """Lookahead meta-gradient (SGD main net, Adam modulator), with per-epoch/step logging."""
    train_loader, val_loader, test_loader = loaders
    model, mod = spec["model"].to(device), spec["plast_mod"]
    mod_opt = torch.optim.Adam(mod.parameters(), lr=spec["mod_lr"])
    ce = nn.CrossEntropyLoss()
    names = [n for n, _ in model.named_parameters()]
    epochs_log, step_loss = [], []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ls = correct = n = 0
        for bi, (x, y) in enumerate(train_loader):
            if max_batches and bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            alphas = mod.compute_alphas(x)
            factors = mod.param_factors(alphas)
            params = list(model.parameters())
            logits = model(x)
            loss = ce(logits, y)
            grads = [g.detach() for g in torch.autograd.grad(loss, params)]
            fast = {}
            for nm, p, g in zip(names, params, grads):
                step = spec["lr"] * (factors[nm] * g) if nm in factors else spec["lr"] * g
                fast[nm] = p.detach() - step
            meta_loss = ce(torch.func.functional_call(model, fast, (x,)), y)
            mod_opt.zero_grad(); meta_loss.backward(); mod_opt.step()
            with torch.no_grad():
                for nm, p in zip(names, params):
                    p.copy_(fast[nm].detach())
            step_loss.append(loss.item())
            ls += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); n += len(y)
        tr_loss, tr_acc = ls / n, correct / n
        val_loss, val_acc = eval_loss_acc(model, val_loader, device)
        probe = spec["probe"](probe_x) if spec["probe"] else {}
        epochs_log.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                           "val_loss": val_loss, "val_acc": val_acc, "probe": probe})
        print(f"  [{name}] ep{ep:>2}/{epochs} tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {probe}")
    test_loss, test_acc = eval_loss_acc(model, test_loader, device)
    print(f"  [{name}] test_acc={test_acc:.4f}  ({time.time()-t0:.1f}s)")
    return {"epochs": epochs_log, "step_loss": step_loss, "test_acc": test_acc,
            "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)
                           + sum(p.numel() for p in mod.parameters()),
            "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# CL loops (naive / sequential)
# ---------------------------------------------------------------------------
def run_cl(name, spec, device, epochs_per_task, max_batches, batch_size=64):
    split = SplitMNIST()
    T = split.n_tasks
    if spec["family"] == "plasticity":
        return run_cl_plasticity(name, spec, split, device, epochs_per_task, max_batches, batch_size)

    model = spec["model"].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    ce = nn.CrossEntropyLoss()
    driver, state, acts, hooks = spec["driver"], {"ema": None}, {}, []
    if driver == "activation_stats":
        hooks = _attach_act_hooks(model, acts)
    # probe batch = task-0 test batch (fixed across all tasks for comparability)
    probe_x = next(iter(split.get_task_loaders(0, batch_size)[1]))[0].to(device)
    A = np.full((T, T), np.nan)
    per_task_log, step_loss = [], []
    t0 = time.time()
    for t in range(T):
        train_loader, test_loader_t = split.get_task_loaders(t, batch_size)
        seen_loader = seen_test_loader(split, t)
        for ep in range(1, epochs_per_task + 1):
            model.train()
            ls = correct = n = 0
            for bi, (x, y) in enumerate(train_loader):
                if max_batches and bi >= max_batches:
                    break
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = ce(logits, y)
                loss.backward()
                opt.step()
                step_loss.append(loss.item())
                ls += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); n += len(y)
                if driver:
                    _update_driver(model, driver, loss, logits, state, acts)
            cur_loss, cur_acc = eval_loss_acc(model, test_loader_t, device)
            seen_loss, seen_acc = eval_loss_acc(model, seen_loader, device)  # tasks 0..t, class-IL
            probe = spec["probe"](probe_x) if spec["probe"] else {}
            per_task_log.append({"task": t, "epoch": ep, "train_loss": ls / n, "train_acc": correct / n,
                                 "curtask_test_loss": cur_loss, "curtask_test_acc": cur_acc,
                                 "seen_loss": seen_loss, "seen_acc": seen_acc, "probe": probe})
            print(f"  [{name}] task{t} ep{ep}/{epochs_per_task} tr_acc={correct/n:.4f} "
                  f"curtask_test_acc={cur_acc:.4f} seen_acc={seen_acc:.4f}")
        for i in range(t + 1):
            _, test_i = split.get_task_loaders(i, batch_size)
            A[t, i] = eval_loss_acc(model, test_i, device)[1]
    for h in hooks:
        h.remove()
    return _finish_cl(name, A, T, per_task_log, step_loss, model, t0,
                      extra_state=state if driver else None)


def run_cl_plasticity(name, spec, split, device, epochs_per_task, max_batches, batch_size):
    T = split.n_tasks
    model, mod = spec["model"].to(device), spec["plast_mod"]
    mod_opt = torch.optim.Adam(mod.parameters(), lr=spec["mod_lr"])
    ce = nn.CrossEntropyLoss()
    names = [n for n, _ in model.named_parameters()]
    probe_x = next(iter(split.get_task_loaders(0, batch_size)[1]))[0].to(device)
    A = np.full((T, T), np.nan)
    per_task_log, step_loss = [], []
    t0 = time.time()
    for t in range(T):
        train_loader, test_loader_t = split.get_task_loaders(t, batch_size)
        seen_loader = seen_test_loader(split, t)
        for ep in range(1, epochs_per_task + 1):
            model.train()
            ls = correct = n = 0
            for bi, (x, y) in enumerate(train_loader):
                if max_batches and bi >= max_batches:
                    break
                x, y = x.to(device), y.to(device)
                alphas = mod.compute_alphas(x)
                factors = mod.param_factors(alphas)
                params = list(model.parameters())
                logits = model(x)
                loss = ce(logits, y)
                grads = [g.detach() for g in torch.autograd.grad(loss, params)]
                fast = {}
                for nm, p, g in zip(names, params, grads):
                    step = spec["lr"] * (factors[nm] * g) if nm in factors else spec["lr"] * g
                    fast[nm] = p.detach() - step
                meta_loss = ce(torch.func.functional_call(model, fast, (x,)), y)
                mod_opt.zero_grad(); meta_loss.backward(); mod_opt.step()
                with torch.no_grad():
                    for nm, p in zip(names, params):
                        p.copy_(fast[nm].detach())
                step_loss.append(loss.item())
                ls += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); n += len(y)
            cur_loss, cur_acc = eval_loss_acc(model, test_loader_t, device)
            seen_loss, seen_acc = eval_loss_acc(model, seen_loader, device)  # tasks 0..t, class-IL
            probe = spec["probe"](probe_x)
            per_task_log.append({"task": t, "epoch": ep, "train_loss": ls / n, "train_acc": correct / n,
                                 "curtask_test_loss": cur_loss, "curtask_test_acc": cur_acc,
                                 "seen_loss": seen_loss, "seen_acc": seen_acc, "probe": probe})
            print(f"  [{name}] task{t} ep{ep}/{epochs_per_task} tr_acc={correct/n:.4f} "
                  f"curtask_test_acc={cur_acc:.4f} seen_acc={seen_acc:.4f}")
        for i in range(t + 1):
            _, test_i = split.get_task_loaders(i, batch_size)
            A[t, i] = eval_loss_acc(model, test_i, device)[1]
    n_params = (sum(p.numel() for p in model.parameters() if p.requires_grad)
                + sum(p.numel() for p in mod.parameters()))
    return _finish_cl(name, A, T, per_task_log, step_loss, model, t0, n_params=n_params)


def _finish_cl(name, A, T, per_task_log, step_loss, model, t0, extra_state=None, n_params=None):
    avg_final = float(np.nanmean(A[T - 1, :]))
    forget_vals = []
    for i in range(T):
        col = [A[t, i] for t in range(i, T) if not np.isnan(A[t, i])]
        if col:
            forget_vals.append(max(col) - A[T - 1, i])
    forgetting = float(np.mean(forget_vals)) if forget_vals else 0.0
    print(f"  [{name}] avg_final_acc={avg_final:.4f} forgetting={forgetting:.4f}  ({time.time()-t0:.1f}s)")
    out = {"per_task": per_task_log, "step_loss": step_loss, "final_per_task": A[T - 1, :].tolist(),
           "avg_final_acc": avg_final, "forgetting": forgetting, "seconds": time.time() - t0,
           "n_trainable": n_params if n_params is not None
                          else sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if extra_state and extra_state.get("n"):
        out["driver_abs_mean"] = extra_state["driver_abs_sum"] / extra_state["n"]
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["standard", "cl", "both"], default="both")
    ap.add_argument("--variants", type=str, default=None, help="comma-separated subset")
    ap.add_argument("--standard-epochs", type=int, default=10)
    ap.add_argument("--cl-epochs", type=int, default=5)
    ap.add_argument("--max-train-batches", type=int, default=0, help="0 = full epoch")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "convergence_study.json"))
    args = ap.parse_args()
    max_batches = args.max_train_batches or None
    device = _device()
    print(f"device={device} seed={args.seed} standard_epochs={args.standard_epochs} "
          f"cl_epochs={args.cl_epochs} max_train_batches={max_batches}")

    results = {"meta": {"seed": args.seed, "standard_epochs": args.standard_epochs,
                        "cl_epochs": args.cl_epochs, "max_train_batches": max_batches,
                        "mask_layer": MASK_LAYER, "lr_standard": 1e-3, "lr_cl": 1e-3},
               "standard": {}, "cl": {}}

    if args.regime in ("standard", "both"):
        print("\n######## STANDARD (full MNIST) ########")
        variants = args.variants.split(",") if args.variants else ALL_VARIANTS_STANDARD
        for name in variants:
            seed_everything(args.seed)  # identical init/data order per variant
            loaders = get_standard_loaders(batch_size=64, val_size=10_000)
            probe_x = next(iter(loaders[1]))[0].to(device)
            spec = build_variant(name, device)
            results["standard"][name] = run_standard(
                name, spec, loaders, probe_x, device, args.standard_epochs, max_batches)
            json.dump(results, open(args.out, "w"), indent=1)  # checkpoint after each variant

    if args.regime in ("cl", "both"):
        print("\n######## CL Split MNIST (naive / sequential) ########")
        variants = args.variants.split(",") if args.variants else ALL_VARIANTS_CL
        for name in variants:
            seed_everything(args.seed)
            spec = build_variant(name, device)
            results["cl"][name] = run_cl(name, spec, device, args.cl_epochs, max_batches)
            json.dump(results, open(args.out, "w"), indent=1)

    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
