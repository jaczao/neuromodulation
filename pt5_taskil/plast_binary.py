"""SELECTIVE PLASTICITY (THESIS-PLAN B): a HARD {0,1} learned plasticity mask via a clipped STE.

User-requested. THESIS-PLAN.md's position-paper mechanism B is
    w_{t+1}_ij = w_t_ij - sigma * 1[(i,j) in p] * grad_{w_ij} L
i.e. an INDICATOR over parameters, not a graded gate. Every plasticity study in this project so far
used a gate in [0,1] (sigmoid), and every one of them died the same death: the gate collapses onto a
UNIFORM GLOBAL LR KNOB. pt7's `ach_ema` boost was a 2.6x lr rescale that dissolved at a tuned lr;
`gate_stats.py` measured cos(dev) = 0.97-0.99 across tasks (every task learns the SAME deviation);
`meta_schedule.py` showed that the better the modulator is optimised the MORE uniform it gets
(cos(dev) -> 0.9993, per-task mean alpha 0.983 0.993 0.992 0.994 0.993).

WHY A BINARY GATE IS THE RIGHT NEXT TEST, and not just another variant: a binary gate CANNOT express
a graded global rescale. A uniform binary gate is either alpha == 1 (exactly vanilla) or alpha == 0
(no learning at all). So the entire failure mode that has explained away every previous plasticity
result here is unavailable to it BY CONSTRUCTION, and any effect it produces has to come from WHICH
elements it freezes — genuine allocation. It is also the one structural property CLAUDE.md credits for
the only pt5 mechanism that ever worked: iter-1's win came from a HARD {0,1} freeze (zero gradient =>
un-absorbable), and the reason the learned gates never matched it is that "iter-1 gamma is bimodal
{0,1} (FREEZES units), learned gamma is unimodal ~1 (NEVER freezes)". This mechanism is exactly the
missing cell: LEARNED allocation with a HARD freeze.

THE ESTIMATOR (as specified): forward = step, backward = the clipped/saturating STE.
    alpha = 1[z >= 0]          forward (hard, non-differentiable)
    d alpha / dz := 1[|z| <= 1]   backward (straight-through, saturating outside the unit window)
`_BinarySTE` below. The clipping is what keeps it stable: once a unit is driven far from the
threshold its gradient is zero, so it stops accumulating drift and the mask stays put instead of
oscillating.

PARITY AT INIT IS alpha == 1 (all plastic), because P is zero-init => z = 0 => 1[z>=0] = 1. That is a
deliberate improvement on the sigmoid studies, where init 0.5 meant a half-LR handicap and
`meta_schedule.py` showed the gate spent its whole capacity undoing its own init (and that recovering
it was the entire measured effect). Here the dead control is numerically NAIVE, with nothing to undo,
so d-dead measures allocation and only allocation.

ARM: standalone buf-cur (as requested) — naive backbone, learned P trained by the modulator-only
replay meta-loss (`--neuromod-meta-replay`), gate applied to the CURRENT task's row. Task-IL, SGD.

GRANULARITIES {global, neuron, synapse}. `global` is included because THESIS-PLAN B explicitly
requires it ("include the `global` scalar control, or a nonzero-mean driver will just recover the
missing tune"). For a BINARY gate that control is degenerate — one scalar that is either 1 (vanilla)
or 0 (frozen solid) — and that degeneracy IS the cleanest statement of the immunity argued above, so
it is run and reported rather than skipped.

CONTROLS
  dead-<gran>  neuro_lr = 0 => P stays 0 => alpha == 1 everywhere => numerically naive, but with the
               modulator constructed, the reservoir filled and sampled and the lookahead run (rule
               #10 RNG matching).
  Reported against `plast_taskil`'s tuned task-IL baselines: naive 0.9784, EWC 0.9821, ER 0.9946.

DIAGNOSTIC (the point of the study): the FROZEN-SET STRUCTURE. For each task, what fraction of
elements is frozen (alpha = 0), and how much do different tasks' frozen sets overlap (IoU + the
per-task-pair Jaccard)? A learned hard gate that reproduces iter-1's disjoint subnetworks should show
frozen fractions ~ (T-1)/T with low IoU; a gate that has merely learned "freeze nothing" shows
fraction ~ 0 (and is then numerically naive, which the dead control already is).

ANCHOR: `--part anchor` runs this file's copy-forwarded loop with the ORIGINAL sigmoid modulators and
must reproduce `plast_init_results.tsv` (neuron nlr 1e-2 = 0.979672, dead = 0.978401, seed 42). That
isolates the loop from the gate: after it passes, the ONLY difference in the mechanism cells is the
estimator.

PROTOCOL: main lr 3e-3 (naive's val-tuned point — THESIS-PLAN B insists on a tuned main lr from the
start, since an untuned one lets any nonzero-mean gate masquerade as a mechanism), ep 5, buffer 1000,
neuro_lr tuned per granularity on VAL (the STE passes gradients of magnitude 1 inside the window,
vs sigmoid's <= 0.25, so the inherited value does not transfer), 3 seeds on test.

Ledger pt5_taskil/plast_binary_results.tsv; `--resume`; `--part` chunks the run.

Run: uv run python pt5_taskil/plast_binary.py --part all --resume  (redirect to plast_binary.log)
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import meta_schedule as MS                                        # noqa: E402  (loop primitives)
import plast_taskil as S                                          # noqa: E402
from prototype.configs import CLConfig                            # noqa: E402
from prototype.data import SplitMNIST                             # noqa: E402
from prototype.neuromod import (                                  # noqa: E402
    DriverBank,
    DriverModulator,
    PlasticityDriverModulator,
    SynapsePlasticityDriverModulator,
    parse_layer_list,
)
from prototype.train import _build_model, _device, evaluate, seed_everything  # noqa: E402
import prototype.train as train_mod                               # noqa: E402

TSV = Path(__file__).resolve().parent / "plast_binary_results.tsv"
COLS = ["stage", "gate", "gran", "nlr", "seed", "acc", "forget"]

MAIN_LR = 3e-3          # naive's val-tuned point (plast_taskil); the standalone arm's backbone
EPOCHS = 5
BUFFER = 1000
NEURO_LRS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)   # shifted down vs the sigmoid grid: the STE passes |g|=1
SEEDS = (42, 43, 44)
TUNE_SEED = 42
NOISE_FLOOR = 0.007
DEAD_NLR = 0.0
GRANS = ("global", "neuron", "synapse")
# anchor: the sigmoid gate through THIS loop must reproduce plast_init (init 0.5, ep5, lr 3e-3)
ANCHOR = {("neuron", 1e-2): 0.979672, ("neuron", 0.0): 0.978401}


# ==================================================================== the clipped/saturating STE
class _BinarySTE(torch.autograd.Function):
    """alpha = 1[z >= 0] forward; d alpha/dz := 1[|z| <= 1] backward (clipped straight-through).

    Outside the unit window the gradient is exactly zero, so an element that has been driven well
    past the threshold stops accumulating drift — that saturation is what stops the mask oscillating
    between steps, and it is why this is the clipped STE rather than the plain identity one.
    """

    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return (z >= 0).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (z,) = ctx.saved_tensors
        return grad_out * (z.abs() <= 1).to(grad_out.dtype)


def binary_ste(z):
    return _BinarySTE.apply(z)


# ============================================================================ binary modulators
# ---------------------------------------------------------------- the threshold offset `b`
# z = b + P[t]. `b` is a FIXED (never trained, seeded) per-element offset that decides where each
# element starts relative to the step threshold, and it is the difference between a mechanism that
# runs and one that self-disables:
#   open   (b = 0)      every element starts EXACTLY on the threshold, alpha == 1 (all plastic).
#                       MEASURED to be a no-op: the meta-loss pushes z uniformly UP (a one-step
#                       lookahead always prefers a bigger step), every element leaves the |z| <= 1
#                       window within ~50 steps, and the clipped STE then gives EXACTLY zero gradient
#                       forever — the gate welds at alpha == 1 and the run is its own dead control.
#                       Kept as a reported cell because that mechanism is the finding.
#   spread (b ~ N(0,s)) elements start spread ACROSS the threshold, so ~half are frozen and nearly all
#                       sit INSIDE the window where the STE passes gradient. This is the init that
#                       actually exercises selective plasticity, and it makes the dead control much
#                       sharper: neuro_lr = 0 freezes the SAME random half forever, so d-dead answers
#                       "does LEARNING which elements to freeze beat freezing a random half of the
#                       same density?" rather than the weaker "does it beat naive?".
SPREAD_SIGMA = 0.5           # ~95% of elements start inside the +/-1 STE window, ~50% frozen
MASK_INITS = ("open", "spread")


def _offset(shape, mask_init, seed, device):
    if mask_init == "open":
        return torch.zeros(shape, device=device)
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * SPREAD_SIGMA).to(device)


class BinaryNeuronPlast(PlasticityDriverModulator):
    """Per-neuron hard mask. Only the squashing changes: sigmoid(init_bias + raw) -> 1[b + raw >= 0]."""

    def set_offsets(self, mask_init, seed, device):
        for l in range(self.n_hidden_layers):
            P = getattr(self, f"P_{l}")
            self.register_buffer(f"b_{l}", _offset(P.shape[1], mask_init, seed + l, device))

    def compute_alphas(self, context=None):
        return {l: binary_ste(getattr(self, f"b_{l}") + self._raw(getattr(self, f"P_{l}")))
                for l in range(self.n_hidden_layers)}


class BinarySynapsePlast(SynapsePlasticityDriverModulator):
    """Per-synapse hard mask — the literal THESIS-PLAN B indicator 1[(i,j) in p]."""

    def set_offsets(self, mask_init, seed, device):
        self._off = {}
        for idx in self.layer_indices:
            P = getattr(self, f"P_{idx}")
            self._off[idx] = _offset(P.shape[1], mask_init, seed + idx, device)
        self._cur = None

    def _gate(self, raw):
        return binary_ste(self._cur + raw) if self._cur is not None else binary_ste(raw)

    def weight_grad_masks(self):
        z = self.bank.value()
        masks = {}
        for idx in self.layer_indices:
            self._cur = self._off[idx]
            masks[f"net.{idx}.weight"] = self._gate(z @ getattr(self, f"P_{idx}")).view(*self._dims[idx])
        self._cur = None
        return masks


class BinaryGlobalPlast(DriverModulator):
    """One scalar per task: alpha = 1[z_t >= 0] applied to EVERY parameter.

    THESIS-PLAN B's required scalar control. For a binary gate it can only say "train normally" or
    "do not train at all", so unlike the continuous global gate it cannot produce a graded LR-scaling
    artifact — which is the point of running it here.
    """

    def __init__(self, bank, projection="learned", shared_frac=0.5, seed=0):
        super().__init__(bank, projection, shared_frac, seed)
        self._register_proj("P_g", 1, 0)
        self.b = 0.0

    def set_offsets(self, mask_init, seed, device):
        # A per-element offset is meaningless for a single scalar (a random b would just decide, once
        # and for all, whether this run trains at all), so `global` stays at the threshold under both
        # inits. Its role here is THESIS-PLAN B's required scalar control, and the degeneracy is the
        # point: a binary global gate can only say "train" or "do not train".
        self.b = 0.0

    def alpha(self):
        return binary_ste(self.b + self._raw(self.P_g))[0]


# ================================================================================ the training loop
def _factors(plast_mod, config, gran, model):
    """Per-parameter LR gate. global broadcasts one scalar to every parameter; neuron/synapse reuse
    the frozen modulators' own broadcasting (so only the SQUASHING differs from the sigmoid runs)."""
    if gran == "global":
        a = plast_mod.alpha()
        return {n: a for n, _ in model.named_parameters()}
    if gran == "synapse":
        return plast_mod.weight_grad_masks()
    return plast_mod.param_factors(
        plast_mod.compute_alphas(),
        scope=config.neuromod_plasticity_scope,
        layers=tuple(parse_layer_list(config.neuromod_plasticity_layers)),
    )


def build_mod(gran, config, model, T, device, mask_init="open"):
    bank = DriverBank(config.neuromod_drivers, T)
    if gran == "global":
        mod = BinaryGlobalPlast(bank, projection=config.neuromod_projection,
                                shared_frac=config.neuromod_shared_frac,
                                seed=config.neuromod_proj_seed).to(device)
    elif gran == "synapse":
        layers = parse_layer_list(config.neuromod_mask_layers)
        dims = {l: (model.net[l].out_features, model.net[l].in_features) for l in layers}
        mod = BinarySynapsePlast(bank, dims, projection=config.neuromod_projection,
                                 shared_frac=config.neuromod_shared_frac,
                                 seed=config.neuromod_proj_seed,
                                 modulate_bias=config.neuromod_modulate_bias,
                                 init_gate=config.neuromod_plasticity_init).to(device)
    else:
        mod = BinaryNeuronPlast(bank, projection=config.neuromod_projection,
                                shared_frac=config.neuromod_shared_frac,
                                seed=config.neuromod_proj_seed,
                                init_gate=config.neuromod_plasticity_init).to(device)
    # offsets are drawn from their OWN generator, so they consume no global RNG and the live and
    # dead arms stay RNG-matched (rule #10) as well as mask-matched.
    mod.set_offsets(mask_init, config.neuromod_proj_seed + 7717, device)
    return mod


def build_sigmoid_mod(gran, config, model, T, device):
    """The ORIGINAL (sigmoid) modulators, for the anchor stage."""
    bank = DriverBank(config.neuromod_drivers, T)
    if gran == "synapse":
        layers = parse_layer_list(config.neuromod_mask_layers)
        dims = {l: (model.net[l].out_features, model.net[l].in_features) for l in layers}
        return SynapsePlasticityDriverModulator(
            bank, dims, projection=config.neuromod_projection,
            shared_frac=config.neuromod_shared_frac, seed=config.neuromod_proj_seed,
            modulate_bias=config.neuromod_modulate_bias,
            init_gate=config.neuromod_plasticity_init).to(device)
    return PlasticityDriverModulator(
        bank, projection=config.neuromod_projection, shared_frac=config.neuromod_shared_frac,
        seed=config.neuromod_proj_seed, init_gate=config.neuromod_plasticity_init).to(device)


def run_cl(config, gran, gate="binary", sequence=None, eval_split="test", mask_init="open",
           trace=None):
    """Standalone buf-cur plasticity with a HARD (or, for the anchor, sigmoid) gate. Task-IL.

    Copy-forward of `meta_schedule.run_cl`'s base schedule (itself anchored bit-exact against the
    frozen pt5 path), with the modulator swapped. One meta update per main step; the committed gate
    is the pre-update one, exactly as in the frozen loop.
    """
    device = _device()
    seed_everything(config.seed)
    split = SplitMNIST(sequence=sequence,
                       val_frac=config.val_frac if eval_split == "val" else 0.0)
    T = split.n_tasks

    def eval_loader_for(i):
        return (split.get_task_val_loader(i, config.batch_size) if eval_split == "val"
                else split.get_task_loaders(i, config.batch_size)[1])

    A = np.full((T, T), np.nan)
    criterion = train_mod.MaskedCE()
    model = _build_model(config, device, n_tasks=T, sequence=list(split.sequence))
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    criterion.pairs = list(split.sequence)
    plast_mod = (build_mod(gran, config, model, T, device, mask_init) if gate == "binary"
                 else build_sigmoid_mod(gran, config, model, T, device))
    plast_modopt = torch.optim.Adam(plast_mod.parameters(), lr=config.neuromod_lr)

    buf_x: list = []; buf_y: list = []; n_seen = 0
    for t in range(T):
        plast_mod.set_task(t)
        model.train()
        train_loader, _ = split.get_task_loaders(t, config.batch_size)
        for _ in range(config.epochs_per_task):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                n_seen = MS._reservoir(x, y, buf_x, buf_y, n_seen, config.er_buffer_size)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                raw_g = {n: p.grad.detach() for n, p in model.named_parameters()
                         if p.grad is not None}
                factors = _factors(plast_mod, config, gran, model)
                meta_x, meta_y = MS._meta_batch(x, y, buf_x, buf_y, device)
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
                factors = {n: v.detach() for n, v in factors.items()}
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in factors:
                            p.grad.mul_(factors[n])
                optimizer.step()
        for i in range(t + 1):
            plast_mod.set_task(i)
            A[t, i] = evaluate(model, eval_loader_for(i), device, allowed=list(split.sequence[i]))
        plast_mod.set_task(t)
        if trace is not None and gate == "binary":
            # frozen fraction + the share of elements whose |z| > 1 (STE gradient exactly zero, i.e.
            # permanently stuck). Recorded per task: the end state alone cannot tell a gate that
            # never moved from one that moved and came back.
            fr, dead = [], []
            for _, get in _mask_getters(plast_mod, gran):
                a = get().detach()
                fr.append(float(1.0 - a.mean()))
            for z in _z_values(plast_mod, gran):
                dead.append(float((z.abs() > 1).to(torch.float32).mean()))
            trace.append((t, float(np.mean(fr)), float(np.mean(dead))))

    acc = float(np.nanmean(A[T - 1, :]))
    forget = float(np.mean([max([A[k, i] for k in range(i, T)]) - A[T - 1, i] for i in range(T)]))
    return acc, forget, plast_mod


# ======================================================================== gate structure read-out
@torch.no_grad()
def gate_structure(plast_mod, gran, T):
    """Per-task hard masks -> frozen fraction and cross-task overlap of the FROZEN sets.

    The study's actual question: does a LEARNED hard gate build iter-1-style disjoint subnetworks
    (frozen fraction ~ (T-1)/T, low IoU between tasks) or does it just learn 'freeze nothing'
    (fraction ~ 0, i.e. numerically the dead control)?
    """
    out = {}
    for name, get in _mask_getters(plast_mod, gran):
        rows = []
        for t in range(T):
            plast_mod.set_task(t)
            rows.append(get().detach().cpu().numpy().ravel())
        Aa = np.stack(rows)                                   # (T, D) in {0,1}, 1 = plastic
        frozen = 1.0 - Aa                                     # 1 = frozen
        ious = []
        for a in range(T):
            for b in range(a + 1, T):
                inter = float((frozen[a] * frozen[b]).sum())
                union = float(((frozen[a] + frozen[b]) > 0).sum())
                ious.append(inter / union if union > 0 else float("nan"))
        out[name] = dict(frozen_frac=frozen.mean(axis=1), iou=float(np.nanmean(ious)) if ious else
                         float("nan"), n=Aa.shape[1])
    return out


@torch.no_grad()
def _z_values(plast_mod, gran):
    """The pre-threshold z = b + P[t] for the current task, per gated layer."""
    if gran == "global":
        return [(plast_mod.b + plast_mod._raw(plast_mod.P_g)).detach()]
    if gran == "synapse":
        return [(plast_mod._off[i] + plast_mod._raw(getattr(plast_mod, f"P_{i}"))).detach()
                for i in plast_mod.layer_indices]
    return [(getattr(plast_mod, f"b_{l}") + plast_mod._raw(getattr(plast_mod, f"P_{l}"))).detach()
            for l in range(plast_mod.n_hidden_layers)]


def _mask_getters(plast_mod, gran):
    if gran == "global":
        return [("global", lambda: plast_mod.alpha().reshape(1))]
    if gran == "synapse":
        names = [f"net.{i}.weight" for i in plast_mod.layer_indices]
        return [(n, (lambda n=n: plast_mod.weight_grad_masks()[n])) for n in names]
    return [(f"h{l}", (lambda l=l: plast_mod.compute_alphas()[l]))
            for l in range(plast_mod.n_hidden_layers)]


# ==================================================================================== ledger/cells
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


def key_of(stage, gate, gran, nlr, seed):
    return (stage, gate, gran, f"{nlr:g}", str(seed))


def build(gran, nlr, seed):
    kw = S.MECH_KW["synapse" if gran == "synapse" else "neuron"]
    return CLConfig(
        seed=seed, lr=MAIN_LR, epochs_per_task=EPOCHS, optimizer="sgd",
        output_masking="taskil", er_buffer_size=BUFFER,
        use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
        neuromod_target="plasticity", neuromod_projection="learned",
        neuromod_meta_replay=True, neuromod_er_task_id=False,
        neuromod_lr=nlr, neuromod_plasticity_init=0.5, **kw,
    )


def run_cell(stage, label, gran, nlr, seed, ledger, split, trace=None):
    """`label` names the ARM in the ledger ('binary-open', 'binary-spread', 'dead-spread',
    'sigmoid'); the MODULATOR and the init are derived from it. Keeping these separate is the fix
    for the first crash: passing the arm label straight through as the modulator selector built a
    sigmoid modulator for the dead control and then called .alpha() on it. A dead control must be
    the SAME mechanism with neuro_lr = 0, never a different one."""
    key = key_of(stage, label, gran, nlr, seed)
    if key in ledger:
        print(f"[skip] {'|'.join(key)} acc={ledger[key][0]:.4f}", flush=True)
        return ledger[key]
    gate = "sigmoid" if label == "sigmoid" else "binary"
    mask_init = "spread" if label.endswith("spread") else "open"
    config = build(gran, nlr, seed)
    config.val_frac = 0.1
    buf = io.StringIO()
    with redirect_stdout(buf):
        acc, forget, _ = run_cl(config, gran, gate=gate, mask_init=mask_init, trace=trace,
                                sequence=S.VAL_SEQ if split == "val" else None, eval_split=split)
    append(key, acc, forget)
    ledger[key] = (acc, forget)
    extra = ""
    if trace:
        extra = ("  frozen/stuck by task: "
                 + " ".join(f"{f:.2f}/{d:.2f}" for _, f, d in trace))
    print(f"[run ] {'|'.join(key)} acc={acc:.4f} forget={forget:.4f}{extra}", flush=True)
    return acc, forget


# ========================================================================================== stages
def anchor(ledger):
    """The SIGMOID gate through this loop must reproduce plast_init — isolates loop from estimator."""
    print("\n" + "=" * 96)
    print("ANCHOR — this loop + the ORIGINAL sigmoid modulator vs plast_init_results.tsv")
    print("=" * 96)
    ok = True
    for (gran, nlr), ref in ANCHOR.items():
        acc, _ = run_cell("anchor", "sigmoid", gran, nlr, TUNE_SEED, ledger, "test")
        match = abs(acc - ref) < 1e-6
        ok &= match
        print(f"  {gran:8s} nlr={nlr:<6g}: {acc:.6f} vs {ref:.6f}  "
              f"{'[MATCHES]' if match else '!! MISMATCH'}", flush=True)
    print(f"\n  loop parity: {'CONFIRMED' if ok else 'FAILED — do not trust the rest'}")
    return ok


def openness(ledger):
    """The `open` init, 1 seed per granularity + a per-task frozen/stuck trace.

    Not given the full 3-seed x tuned treatment because it is provably degenerate: measured, the
    meta-gradient pushes z uniformly UP, every element leaves the |z| <= 1 window within ~50 steps
    (P.grad norm -> EXACTLY 0), so alpha == 1 for the whole run and the cell IS its own dead control
    at any neuro_lr — which is why its val sweep returned the same number to 4 dp across 1e-5..1e-1.
    Reported as the mechanism it is, with the trace as the evidence."""
    print("\n" + "=" * 96)
    print("OPEN INIT (b = 0, all elements start ON the threshold) — expect a self-disabling no-op")
    print("=" * 96)
    for gran in GRANS:
        trace = []
        run_cell("open", "binary-open", gran, NEURO_LRS[-1], TUNE_SEED, ledger, "test", trace=trace)


def tune(ledger):
    """neuro_lr per granularity on VAL, `spread` init. argmax; exact ties -> the LARGER nlr (more
    engagement, so the mechanism is exercised at the reported cell) — the meta_schedule correction."""
    best = {}
    for gran in GRANS:
        res = [(run_cell("tune", "binary-spread", gran, nlr, TUNE_SEED, ledger, "val")[0], nlr)
               for nlr in NEURO_LRS]
        acc, nlr = max(res)
        best[gran] = (nlr, acc)
        span = max(r[0] for r in res) - min(r[0] for r in res)
        print(f"\n>>> TUNED [binary-spread {gran}] nlr={nlr:g} (val {acc:.4f})", flush=True)
        if span < NOISE_FLOOR:
            print(f"    !! whole grid spans {span:.4f} < the {NOISE_FLOOR} noise floor — "
                  f"unresolved at 1 seed, not optimised", flush=True)
        elif nlr in (NEURO_LRS[0], NEURO_LRS[-1]):
            print(f"    !! nlr={nlr:g} at a GRID EDGE with a resolvable span ({span:.4f}) — extend",
                  flush=True)
        print("    val by nlr: " + "  ".join(f"{n:g}={a:.4f}" for a, n in res), flush=True)
    return best


def test(ledger, best):
    """3 seeds at the tuned point. The dead control is the SAME binary mechanism and the SAME random
    mask, with neuro_lr = 0 — so d-dead isolates 'learning WHICH elements to freeze' from 'freezing
    a random half of that density', which is the sharper question."""
    table = {}
    for gran in GRANS:
        nlr = best[gran][0]
        for kind, label, use_nlr in (("binary", "binary-spread", nlr),
                                     ("dead", "dead-spread", DEAD_NLR)):
            accs, forgets = [], []
            for s in SEEDS:
                a, f = run_cell("test", label, gran, use_nlr, s, ledger, "test")
                accs.append(a); forgets.append(f)
            table[(gran, kind)] = dict(accs=accs, mean=float(np.mean(accs)),
                                       std=float(np.std(accs)),
                                       forget=float(np.mean(forgets)), nlr=use_nlr)
    return table


def gate_report(best):
    """Frozen-set structure at each selected cell (seed 42)."""
    for gran in GRANS:
        nlr = best[gran][0]
        buf = io.StringIO()
        with redirect_stdout(buf):
            acc, forget, mod = run_cl(build(gran, nlr, TUNE_SEED), gran, gate="binary",
                                      mask_init="spread", sequence=None, eval_split="test")
        st = gate_structure(mod, gran, 5)
        print(f"\n{'=' * 100}")
        print(f"GATE STRUCTURE  binary {gran}  nlr={nlr:g}  |  test acc {acc:.4f} forget {forget:.4f}")
        print("=" * 100)
        print(f"{'layer':12s} {'n':>9s} {'frozen frac per task':>44s} {'mean':>7s} {'IoU(frozen)':>12s}")
        for name, d in st.items():
            ff = d["frozen_frac"]
            print(f"{name:12s} {d['n']:>9d} {'  '.join(f'{v:.3f}' for v in ff):>44s} "
                  f"{ff.mean():>7.3f} {d['iou']:>12.4f}")
        print("  frozen frac ~0 => the gate learned to freeze NOTHING (numerically the dead control).")
        # T=5 disjoint subnets: each task freezes 4/5 of units; two tasks' frozen sets intersect in
        # 3/5 and their union is 5/5, so IoU = 0.6 (NOT 0.75, which an earlier version printed).
        print("  disjoint-subnet reference (pt5 iter-1, T=5): frozen frac 0.8 with IoU 0.60.")
        print("  shared-offset init starts every task at IoU 1.0, so differentiation shows as IoU < 1.")


def report(table, best):
    print("\n" + "=" * 108)
    print("SELECTIVE PLASTICITY — hard {0,1} mask via clipped STE, standalone buf-cur, task-IL, "
          "SGD, 3 seeds, TEST")
    print("=" * 108)
    print(f"{'gran':9s} {'nlr':>8s} {'binary acc':>17s} {'dead acc':>17s} {'d-dead':>8s} "
          f"{'per seed':>28s} {'pos':>4s} {'forget':>8s}")
    for gran in GRANS:
        B, D = table[(gran, "binary")], table[(gran, "dead")]
        ps = [a - b for a, b in zip(B["accs"], D["accs"])]
        print(f"{gran:9s} {B['nlr']:>8g} {B['mean']:>10.4f}±{B['std']:.4f} "
              f"{D['mean']:>10.4f}±{D['std']:.4f} {B['mean'] - D['mean']:>+8.4f} "
              f"{', '.join(f'{v:+.4f}' for v in ps):>28s} "
              f"{sum(1 for v in ps if v > 0)}/3 {B['forget']:>8.4f}")
    print("\n  reference (plast_taskil, tuned task-IL): naive 0.9784  EWC 0.9821  ER 0.9946")
    print("  dead = neuro_lr 0 => alpha == 1 everywhere => numerically naive, RNG-matched.")
    print("  A binary gate cannot express a graded global LR rescale, so any d-dead here is")
    print("  allocation, not the LR artifact that explained the sigmoid results.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "anchor", "open", "tune", "test", "gate", "report"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(f"selective plasticity (hard {{0,1}} STE) | grans {GRANS} | inits {MASK_INITS} | "
          f"standalone buf-cur task-IL sgd main_lr {MAIN_LR:g} ep {EPOCHS} buffer {BUFFER}\n",
          flush=True)
    ledger = load_ledger() if args.resume else {}
    if args.part in ("all", "anchor"):
        anchor(ledger)
        if args.part == "anchor":
            return
    if args.part in ("all", "open"):
        openness(ledger)
        if args.part == "open":
            return
    best = tune(ledger)
    if args.part == "tune":
        return
    if args.part == "gate":
        gate_report(best)
        return
    table = test(ledger, best)
    if args.part == "all":
        gate_report(best)
    report(table, best)


if __name__ == "__main__":
    main()
