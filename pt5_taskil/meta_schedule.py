"""Standalone buf-cur plasticity, task-IL/SGD: two META-SCHEDULE variants at the tuned operating point.

User-requested follow-up to `plast_init.py`. Everything so far varied WHAT the gate is (granularity,
init, neuro_lr) and left the SCHEDULE alone: one meta update per main step, interleaved, on the same
batch. `gate_stats.py` then explained the null two different ways — the per-NEURON gate engages hard
(alpha 0.5 -> 0.93) but every task learns the SAME deviation (cos(dev) 0.97-0.99), i.e. it spends
itself as a global LR knob undoing its own 0.5 init; the per-SYNAPSE gate never leaves parity at all
(|alpha-0.5| = 0.0007 at a tuned neuro_lr that sits on the grid floor = the gate's off-switch). Both
readings are about the modulator not getting enough, or not the right kind of, optimisation. So vary
the SCHEDULE:

  V1 `multi5`  — 5 meta passes per main step. The main net is held fixed while the modulator takes 5
                 sequential Adam steps on the meta-loss, then the main net takes its one gated step.
                 Tests "the modulator is under-optimised per step": if the gate is merely too slow to
                 track the backbone, more inner steps should help. NOTE this is NOT just a 5x larger
                 neuro_lr — the gate is RE-PROJECTED between passes (each pass sees the lookahead
                 W_fast built from the CURRENT alpha), so it follows the meta-loss curvature; and
                 neuro_lr is re-tuned here anyway, so the 5x-lr reading is controlled for.
  V2 `phase`   — two phases per task over DISJOINT sample sets. Phase 1: only the modulator learns
                 (lookahead meta-loss over half the task's data + the buffer; the main net's gradient
                 is computed to build the lookahead but never applied). Phase 2: only the main net
                 learns, gated by the now-frozen modulator, over the OTHER half. Tests "interleaving
                 is the problem": the gate normally chases a backbone that moves under it every step,
                 so give it a stationary target first, then let the backbone move under a fixed gate.
                 The no-reuse constraint is the user's; it also removes the obvious confound that the
                 gate was tuned on exactly the samples the main net then trains on.

BUDGET MATCHING (rule #3). Base = 5 epochs x N batches = 5N main steps and 5N meta steps. `phase`
gives each phase 2*ep = 10 epochs over half the data, so it is 5N meta steps then 5N main steps —
the same count of each, only the ORDER and the sample disjointness differ. `multi5` is the one arm
that spends more meta updates (25N); that is the variable under test, and its dead control spends
exactly as many, so `d-dead` is still a clean read of what the extra optimisation bought.

WHAT `phase` COSTS, STATED UP FRONT: with no sample reuse, its backbone sees only HALF of each task's
unique images (10 passes over half, rather than 5 over all), and its modulator likewise. So its
ABSOLUTE accuracy is not comparable to base/naive — only to its own dead control, which carries the
identical handicap. This is exactly the case rule #10's matched control exists for; read `d-dead`,
and do not read the live column against 0.9784.

CONFIG = the best cell of `plast_init.py`: init 0.5, ep 5, main lr 3e-3 (naive's tuned point), SGD,
buffer 1000, taskil. That study selected init 0.5 for BOTH granularities on val, and it is the only
init where anything was ever positive (neuron d-dead +0.0025, 3/3 seeds). `neuro_lr` IS re-tuned per
(variant, granularity) on val: both variants change how much meta optimisation happens per unit of
backbone movement, so the inherited value is not transferable — and for synapse the inherited value
(1e-4, the grid floor) is the gate's off-switch, so reusing it would guarantee a null by construction.
Same 5-point grid, same budget, same tie-break as before.

DEAD-GATE CONTROL, per variant and per granularity (rule #10): the identical schedule with
neuromod_lr = 0, so P stays at its zero init and alpha is pinned at exactly 0.5 — a uniform half-LR
rescale. The modulator is still built, the buffer still filled and sampled, every meta pass still run.
That is the RNG-matched baseline; `naive` is not. Report `d-dead`.

WHY THIS IS A COPY-FORWARD, NOT A FLAG. Both variants restructure `cl_train`'s inner loop (a second
meta iteration; a per-task phase split with two loaders), and `prototype/` is FROZEN (rule #9). So the
pt5 plasticity branch is copied here and edited, per the extraction rule (COPY-FORWARD, never cut).
The copy is proved faithful by the `--part anchor` stage: the `base` variant (1 meta pass, no phases)
must reproduce `plast_init_results.tsv` bit-exact at the same config, for both granularities and both
the live and the dead gate. Do not trust any number below until that stage prints [MATCHES].

Ledger pt5_taskil/meta_schedule_results.tsv; `--resume` skips done rows; `--part` chunks the run
(a Bash 600s timeout DETACHES rather than kills, so chunk deliberately).

Run: uv run python pt5_taskil/meta_schedule.py --part all --resume  (redirect to meta_schedule.log)
"""
import argparse
import io
import os
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_stats as G                                   # noqa: E402
import plast_taskil as S                                 # noqa: E402
import prototype.train as train_mod                      # noqa: E402
from prototype.configs import CLConfig                    # noqa: E402
from prototype.data import SplitMNIST                     # noqa: E402
from prototype.neuromod import (                          # noqa: E402
    DriverBank,
    PlasticityDriverModulator,
    SynapsePlasticityDriverModulator,
    parse_layer_list,
)
from prototype.train import _build_model, _device, evaluate, seed_everything  # noqa: E402

TSV = Path(__file__).resolve().parent / "meta_schedule_results.tsv"
COLS = ["stage", "variant", "gran", "nlr", "seed", "acc", "forget"]

# --- the tuned operating point inherited from plast_init.py (its best cell, both granularities) ---
MAIN_LR = 3e-3                  # naive's tuned point; the standalone arm's backbone is naive
EPOCHS = 5                      # per task (base); `phase` uses 2*EPOCHS per phase on half the data
INIT = 0.5                      # plast_init's val-selected init for BOTH granularities
BUFFER = 1000
NEURO_LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
INHERITED_NLR = {"neuron": 1e-2, "synapse": 1e-4}         # plast_init's selections (the anchor cells)
SEEDS = (42, 43, 44)
TUNE_SEED = 42
NOISE_FLOOR = 0.007
DEAD_NLR = 0.0
META_PASSES = 5                 # V1: meta updates per main step
PHASE_SPLIT_SEED = 987          # A/B halves keyed by class pair only => identical across model seeds

VARIANTS = ("multi5", "phase")
ALL_VARIANTS = ("base",) + VARIANTS

# plast_init anchors: (variant=base) live/dead at the inherited neuro_lr, seeds 42/43/44.
ANCHORS = {
    ("neuron", 1e-2): (0.979672, 0.975181, 0.978690),
    ("neuron", 0.0): (0.978401, 0.971657, 0.976019),
    ("synapse", 1e-4): (0.978673, 0.974492, 0.976898),
    ("synapse", 0.0): (0.978673, 0.974492, 0.976898),
}


# ============================================================== the copy-forward training loop
def _factors(plast_mod, config, gran):
    """The per-parameter LR gate, differentiable in P. Mirrors cl_train's two plasticity branches."""
    if gran == "synapse":
        return plast_mod.weight_grad_masks()
    return plast_mod.param_factors(
        plast_mod.compute_alphas(),
        scope=config.neuromod_plasticity_scope,
        layers=tuple(parse_layer_list(config.neuromod_plasticity_layers)),
    )


def _meta_batch(x, y, buf_x, buf_y, device):
    """The modulator-only replay meta-batch: the current batch + an equal buffer draw (past tasks)."""
    if not buf_x:
        return x, y
    mi = random.choices(range(len(buf_x)), k=len(x))
    mbx = torch.stack([buf_x[j] for j in mi]).to(device)
    mby = torch.stack([buf_y[j] for j in mi]).to(device)
    return torch.cat([x, mbx]), torch.cat([y, mby])


def _meta_update(model, plast_mod, plast_modopt, criterion, config, gran, raw_g, meta_x, meta_y):
    """One lookahead meta step: W_fast = W - lr*(gate(P) ⊙ g) with g DETACHED, so the meta-loss is
    differentiable in P only; Adam then moves P. Returns the gate it used (still attached)."""
    factors = _factors(plast_mod, config, gran)
    fast = {}
    for n, p in model.named_parameters():
        if n in factors and n in raw_g:
            fast[n] = p.detach() - config.lr * (factors[n] * raw_g[n])
        else:
            fast[n] = p.detach()
    meta_loss = criterion(torch.func.functional_call(model, fast, (meta_x,)), meta_y)
    plast_modopt.zero_grad()
    meta_loss.backward()
    plast_modopt.step()
    return factors


def _reservoir(x, y, buf_x, buf_y, n_seen, size):
    for xi, yi in zip(x.cpu(), y.cpu()):
        n_seen += 1
        if len(buf_x) < size:
            buf_x.append(xi); buf_y.append(yi)
        else:
            j = random.randrange(n_seen)
            if j < size:
                buf_x[j] = xi; buf_y[j] = yi
    return n_seen


def _phase_loaders(split_mnist, t, batch_size):
    """Task t's train indices split into two DISJOINT halves (no sample is reused between phases).

    Keyed by the class pair only — like the val split in data.py — so the halves are the same images
    for every model seed, making the phase composition a fixed property of the study rather than a
    per-seed nuisance. Respects val_frac (the indices come from the same accessor cl_train uses).
    """
    train_idx, _ = split_mnist._task_train_val_idx(t)
    classes = sorted(split_mnist.sequence[t])
    perm = np.random.default_rng([PHASE_SPLIT_SEED, classes[0], classes[1]]).permutation(len(train_idx))
    arr = np.asarray(train_idx)
    half = len(arr) // 2
    mk = lambda idx: DataLoader(Subset(split_mnist._train_ds, idx.tolist()),  # noqa: E731
                                batch_size=batch_size, shuffle=True)
    return mk(arr[perm[:half]]), mk(arr[perm[half:]])


def run_cl(config, variant, sequence=None, eval_split="test", dump_gate=False):
    """pt5 standalone buf-cur plasticity (naive backbone + modulator-only replay meta-loss), task-IL.

    COPY-FORWARD of cl_train's `_is_pt5` branch, narrowed to this one arm (method=naive, learned P,
    target=plasticity, er_task_id off, no sparsity, no moment reset) and extended with the two
    schedule variants. variant='base' is the unmodified schedule and is anchor-tested against the
    frozen path. Returns (avg_final_acc, forgetting), computed exactly as cl_train computes them.
    """
    device = _device()
    seed_everything(config.seed)
    effective_val_frac = config.val_frac if eval_split == "val" else 0.0
    split_mnist = SplitMNIST(sequence=sequence, val_frac=effective_val_frac)
    T = split_mnist.n_tasks

    def eval_loader_for(i):
        if eval_split == "val":
            return split_mnist.get_task_val_loader(i, config.batch_size)
        return split_mnist.get_task_loaders(i, config.batch_size)[1]

    A = np.full((T, T), np.nan)
    criterion = train_mod.MaskedCE()
    model = _build_model(config, device, n_tasks=T, sequence=list(split_mnist.sequence))
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    criterion.pairs = list(split_mnist.sequence)          # per-sample masked loss (taskil convention)

    gran = config.neuromod_granularity
    bank = DriverBank(config.neuromod_drivers, T)
    if gran == "synapse":
        layers = parse_layer_list(config.neuromod_mask_layers)
        layer_dims = {l: (model.net[l].out_features, model.net[l].in_features) for l in layers}
        plast_mod = SynapsePlasticityDriverModulator(
            bank, layer_dims, projection=config.neuromod_projection,
            shared_frac=config.neuromod_shared_frac, seed=config.neuromod_proj_seed,
            modulate_bias=config.neuromod_modulate_bias,
            init_gate=config.neuromod_plasticity_init,
        ).to(device)
    else:
        plast_mod = PlasticityDriverModulator(
            bank, projection=config.neuromod_projection,
            shared_frac=config.neuromod_shared_frac, seed=config.neuromod_proj_seed,
            init_gate=config.neuromod_plasticity_init,
        ).to(device)
    plast_modopt = torch.optim.Adam(plast_mod.parameters(), lr=config.neuromod_lr)

    buf_x: list = []; buf_y: list = []; n_seen = 0
    n_meta = n_main = 0

    def main_step(x, y, do_meta, meta_passes):
        """One batch: reservoir, main fwd/bwd, `meta_passes` modulator updates, then (optionally) the
        gated main step. The committed gate is the PRE-update one, exactly as in the frozen loop —
        so with meta_passes=1 this is byte-identical to cl_train, and the number of meta updates is
        the ONLY thing that changes at meta_passes=5 (rule #8: one axis)."""
        nonlocal n_seen, n_meta
        n_seen = _reservoir(x, y, buf_x, buf_y, n_seen, config.er_buffer_size)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        raw_g = {n: p.grad.detach() for n, p in model.named_parameters() if p.grad is not None}
        factors = _factors(plast_mod, config, gran)        # the gate the main step will commit
        if do_meta:
            for _ in range(meta_passes):
                meta_x, meta_y = _meta_batch(x, y, buf_x, buf_y, device)
                # pass 0 re-derives the same gate as `factors`; later passes re-project P after its
                # previous update, so the lookahead follows the meta-loss rather than replaying one
                # gradient 5 times — and each pass draws a fresh buffer sample, likewise.
                _meta_update(model, plast_mod, plast_modopt, criterion, config, gran,
                             raw_g, meta_x, meta_y)
                n_meta += 1
        factors = {n: v.detach() for n, v in factors.items()}
        return factors

    def commit(factors):
        nonlocal n_main
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None and n in factors:
                    p.grad.mul_(factors[n])
        optimizer.step()
        n_main += 1

    for t in range(T):
        plast_mod.set_task(t)
        model.train()
        if variant == "phase":
            # Phase 1: ONLY the modulator learns (meta updates, no optimizer.step()); phase 2: ONLY
            # the main net learns, under the gate phase 1 left behind. Disjoint halves, 2*EPOCHS each,
            # so each net takes the same number of updates as the base schedule.
            loader_a, loader_b = _phase_loaders(split_mnist, t, config.batch_size)
            for _ in range(2 * config.epochs_per_task):
                for x, y in loader_a:
                    main_step(x.to(device), y.to(device), do_meta=True, meta_passes=1)
            optimizer.zero_grad()
            for _ in range(2 * config.epochs_per_task):
                for x, y in loader_b:
                    commit(main_step(x.to(device), y.to(device), do_meta=False, meta_passes=0))
        else:
            passes = META_PASSES if variant == "multi5" else 1
            train_loader, _ = split_mnist.get_task_loaders(t, config.batch_size)
            for _ in range(config.epochs_per_task):
                for x, y in train_loader:
                    commit(main_step(x.to(device), y.to(device), do_meta=True, meta_passes=passes))

        for i in range(t + 1):
            plast_mod.set_task(i)
            A[t, i] = evaluate(model, eval_loader_for(i), device,
                              allowed=list(split_mnist.sequence[i]))   # taskil: 2-way masked eval
        seen = ", ".join(f"{A[t, i]:.3f}" for i in range(t + 1))
        print(f"After task {t + 1}/{T} | seen tasks: [{seen}]")
        plast_mod.set_task(t)

    print(f"[meta_schedule] variant={variant} gran={gran} nlr={config.neuromod_lr:g} "
          f"init={config.neuromod_plasticity_init} main_steps={n_main} meta_steps={n_meta}")
    if dump_gate:
        train_mod._pt5_dump_overlap(model, T, config, plast_mod=plast_mod)

    avg_final_acc = float(np.nanmean(A[T - 1, :]))
    forget_vals = []
    for i in range(T):
        col = [A[t, i] for t in range(i, T) if not np.isnan(A[t, i])]
        if col:
            forget_vals.append(max(col) - A[T - 1, i])
    forgetting = float(np.mean(forget_vals)) if forget_vals else 0.0
    print(f"\navg_final_acc={avg_final_acc:.4f} | forgetting={forgetting:.4f}")
    return avg_final_acc, forgetting


# ============================================================================== ledger + cells
def load_ledger():
    if not TSV.exists():
        return {}
    rows = {}
    for line in TSV.read_text().splitlines()[1:]:
        if line.strip():
            f = line.split("\t")
            rows[tuple(f[:5])] = (float(f[5]), float(f[6]))
    return rows


def append(key, acc, forget):
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with TSV.open("a") as fh:
        fh.write("\t".join(list(key) + [f"{acc:.6f}", f"{forget:.6f}"]) + "\n")


def key_of(stage, variant, gran, nlr, seed):
    return (stage, variant, gran, f"{nlr:g}", str(seed))


def build(gran, nlr, seed):
    return CLConfig(
        seed=seed, lr=MAIN_LR, epochs_per_task=EPOCHS, optimizer="sgd",
        output_masking="taskil", er_buffer_size=BUFFER,
        use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
        neuromod_target="plasticity", neuromod_projection="learned",
        neuromod_meta_replay=True, neuromod_er_task_id=False,
        neuromod_lr=nlr, neuromod_plasticity_init=INIT,
        **S.MECH_KW[gran],
    )


def run_cell(stage, variant, gran, nlr, seed, ledger, split):
    key = key_of(stage, variant, gran, nlr, seed)
    if key in ledger:
        acc, forget = ledger[key]
        print(f"[skip] {'|'.join(key)} acc={acc:.4f}", flush=True)
        return acc, forget
    config = build(gran, nlr, seed)
    config.val_frac = 0.1
    buf = io.StringIO()
    with redirect_stdout(buf):
        acc, forget = run_cl(config, variant, sequence=S.VAL_SEQ if split == "val" else None,
                             eval_split=split)
    append(key, acc, forget)
    ledger[key] = (acc, forget)
    print(f"[run ] {'|'.join(key)} acc={acc:.4f} forget={forget:.4f}", flush=True)
    return acc, forget


# ==================================================================================== stages
def anchor(ledger):
    """Parity: variant='base' must reproduce plast_init_results.tsv bit-exact, live AND dead.

    1e-6 = the ledger's storage precision ("%.6f"), not machine epsilon (CLAUDE.md: matching the
    tolerance to how the value was produced — a tighter one flags identical runs as mismatched).
    """
    print("\n" + "=" * 96)
    print("ANCHOR — copy-forward parity vs plast_init.py (base schedule, init 0.5, ep 5, lr 3e-3)")
    print("=" * 96)
    ok = True
    for gran in S.GRANS:
        for nlr in (INHERITED_NLR[gran], DEAD_NLR):
            for seed, ref in zip(SEEDS, ANCHORS[(gran, nlr)]):
                acc, _ = run_cell("anchor", "base", gran, nlr, seed, ledger, "test")
                match = abs(acc - ref) < 1e-6
                ok &= match
                print(f"  {gran:8s} nlr={nlr:<7g} seed {seed}: {acc:.6f} vs {ref:.6f}  "
                      f"{'[MATCHES]' if match else '!! MISMATCH'}", flush=True)
    print(f"\n  copy-forward parity: {'CONFIRMED' if ok else 'FAILED — do not trust the rest'}")
    return ok


def tune(ledger):
    """neuro_lr per (variant, gran) on VAL, seed 42: best val acc, exact ties -> the LARGER nlr.

    NOT the "smallest value inside the noise floor" tie-break plast_init used for EPOCHS. That rule is
    right for a cost axis (a cheaper budget for the same val acc is strictly better) and wrong for the
    mechanism's own axis: neuro_lr = 0 IS the dead gate, so "prefer the smaller value when everything
    ties" walks the selection toward the off-switch — here it picked 1e-4, simultaneously the LOWEST
    val cell in the grid and the least-engaged gate, which would have made d-dead ~ 0 by construction
    rather than by measurement. So: argmax, and on an exact tie prefer MORE engagement, so the
    mechanism is actually exercised at the reported cell.

    When the whole grid spans less than the noise floor the sweep has not selected anything (CLAUDE.md:
    a 1-seed val sweep that is inert cannot resolve the mechanism, only bracket it) — that is flagged,
    and the gate read-out, not this number, is what says whether the gate did anything.
    """
    best = {}
    for variant in VARIANTS:
        for gran in S.GRANS:
            res = [(run_cell("tune", variant, gran, nlr, TUNE_SEED, ledger, "val")[0], nlr)
                   for nlr in NEURO_LRS]
            acc, nlr = max(res)                            # (acc, nlr): ties -> the larger nlr
            best[(variant, gran)] = (nlr, acc)
            span = max(r[0] for r in res) - min(r[0] for r in res)
            print(f"\n>>> TUNED [{variant} {gran}] nlr={nlr:g} (val {acc:.4f})", flush=True)
            if span < NOISE_FLOOR:
                print(f"    !! the WHOLE grid spans {span:.4f} < the {NOISE_FLOOR} noise floor — "
                      f"this axis is unresolved at 1 seed, not optimised", flush=True)
            elif nlr in (NEURO_LRS[0], NEURO_LRS[-1]):
                print(f"    !! nlr={nlr:g} at a GRID EDGE with a resolvable span ({span:.4f}) — "
                      f"a truncated grid; extend it", flush=True)
            print(f"    val by nlr: " + "  ".join(f"{n:g}={a:.4f}" for a, n in res), flush=True)
    return best


def test(ledger, best):
    table = {}
    for variant in VARIANTS:
        for gran in S.GRANS:
            nlr = best[(variant, gran)][0]
            for kind, use_nlr in (("live", nlr), ("dead", DEAD_NLR)):
                accs, forgets = [], []
                for s in SEEDS:
                    a, f = run_cell("test", variant, gran, use_nlr, s, ledger, "test")
                    accs.append(a); forgets.append(f)
                table[(variant, gran, kind)] = dict(
                    accs=accs, mean=float(np.mean(accs)), std=float(np.std(accs)),
                    forget=float(np.mean(forgets)), nlr=use_nlr)
    return table


def gate(best):
    """Read the learned alpha at each selected cell (seed 42): did the schedule change change WHAT
    the gate learned, even where accuracy did not move? |a-par| ~ 0 = never engaged; cos(dev) ~ 1 =
    engaged but every task learned the same deviation (a global LR knob, carrying no task info)."""
    for variant in VARIANTS:
        for gran in S.GRANS:
            nlr = best[(variant, gran)][0]
            G.COLLECTED.clear()
            original = train_mod._pt5_dump_overlap
            train_mod._pt5_dump_overlap = G.collector
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    acc, forget = run_cl(build(gran, nlr, TUNE_SEED), variant, sequence=None,
                                         eval_split="test", dump_gate=True)
            finally:
                train_mod._pt5_dump_overlap = original
            data = dict(G.COLLECTED)
            parity = data["parity"]
            print(f"\n{'=' * 104}")
            print(f"GATE  {variant} {gran}  nlr={nlr:g}  |  test acc {acc:.4f} forget {forget:.4f}"
                  f"  |  parity alpha {parity:.4f}")
            print("=" * 104)
            print(f"{'layer':7s} {'n':>9s} {'|a-par|':>9s} {'a mean':>8s} {'a sd':>8s} "
                  f"{'a min':>8s} {'a max':>8s} {'>0.75':>7s} {'>0.95':>7s} "
                  f"{'cos(dev)':>9s} {'cos(a)':>8s} {'frac':>6s}")
            for name, Aa in data["layers"].items():
                dev = Aa - parity
                print(f"{name:7s} {Aa[0].size:>9d} {np.abs(dev).mean():>9.5f} {Aa.mean():>8.4f} "
                      f"{Aa.std():>8.4f} {Aa.min():>8.4f} {Aa.max():>8.4f} "
                      f"{(Aa > 0.75).mean():>7.3f} {(Aa > 0.95).mean():>7.3f} "
                      f"{G._offdiag_cos(dev):>9.4f} {G._offdiag_cos(Aa):>8.4f} "
                      f"{G._engaged_frac(dev):>6.3f}")
            print("  per-task mean alpha: " + "  ".join(
                f"{name}:[" + " ".join(f"{Aa[t].mean():.4f}" for t in range(Aa.shape[0])) + "]"
                for name, Aa in data["layers"].items()))
            np.savez(Path(__file__).resolve().parent / f"gate_alpha_{variant}_{gran}.npz",
                     parity=parity, **data["layers"])


def report(table, best):
    print("\n" + "=" * 112)
    print("META-SCHEDULE VARIANTS — standalone buf-cur plasticity, task-IL, SGD, init 0.5, "
          "3 seeds, TEST set")
    print("=" * 112)
    print(f"{'variant':9s} {'gran':8s} {'nlr':>8s} {'live acc':>17s} {'dead acc':>17s} "
          f"{'d-dead':>8s} {'per seed':>28s} {'pos':>4s} {'forget':>8s}")
    for gran in S.GRANS:                                  # the base schedule, from the anchors
        live, dead = ANCHORS[(gran, INHERITED_NLR[gran])], ANCHORS[(gran, DEAD_NLR)]
        ps = [a - b for a, b in zip(live, dead)]
        print(f"{'base':9s} {gran:8s} {INHERITED_NLR[gran]:>8g} "
              f"{np.mean(live):>10.4f}±{np.std(live):.4f} {np.mean(dead):>10.4f}±{np.std(dead):.4f} "
              f"{np.mean(live) - np.mean(dead):>+8.4f} "
              f"{', '.join(f'{v:+.4f}' for v in ps):>28s} "
              f"{sum(1 for v in ps if v > 0)}/3 {'(plast_init)':>8s}")
    for variant in VARIANTS:
        for gran in S.GRANS:
            L, D = table[(variant, gran, "live")], table[(variant, gran, "dead")]
            ps = [a - b for a, b in zip(L["accs"], D["accs"])]
            print(f"{variant:9s} {gran:8s} {L['nlr']:>8g} "
                  f"{L['mean']:>10.4f}±{L['std']:.4f} {D['mean']:>10.4f}±{D['std']:.4f} "
                  f"{L['mean'] - D['mean']:>+8.4f} "
                  f"{', '.join(f'{v:+.4f}' for v in ps):>28s} "
                  f"{sum(1 for v in ps if v > 0)}/3 {L['forget']:>8.4f}")
    print("\n  reference (plast_taskil, ep=5, main lr 3e-3): naive 0.9784  er 0.9946  ewc 0.9821")
    print("  d-dead is the mechanism's effect (RNG-matched, same schedule, alpha pinned at 0.5).")
    print(f"  1-seed noise floor {NOISE_FLOOR}; |d-dead| below ~0.002 is inside 3-seed spread.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "anchor", "tune", "test", "gate", "report"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"meta-schedule variants {VARIANTS} (M={META_PASSES}) | init {INIT} ep {EPOCHS} "
          f"main lr {MAIN_LR:g} sgd taskil buffer {BUFFER}\n", flush=True)

    ledger = load_ledger() if args.resume else {}
    if args.part in ("all", "anchor"):
        anchor(ledger)
        if args.part == "anchor":
            return
    best = tune(ledger)
    if args.part == "tune":
        return
    if args.part == "gate":
        gate(best)
        return
    table = test(ledger, best)
    if args.part == "all":
        gate(best)
    report(table, best)


if __name__ == "__main__":
    main()
