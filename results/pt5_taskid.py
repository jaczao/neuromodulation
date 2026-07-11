"""pt5 runner: task-id oracle driver, generalized bottleneck. Iteration 1 = disjoint projection.

Screening (1 seed, SPEC Methodology 3). For each `target x projection` cell reports BOTH:
  - standalone : neurom (method=naive, masked-loss ON) vs the same-optimizer naive-SGD+masked-loss.
  - +ER        : neurom+ER (masked-loss OFF) vs plain ER-SGD.
SGD main net throughout (Methodology 6): no Adam/SGD confound, every comparison same-optimizer.
weight_mask masks net.0+net.2 with masked loss (lever B covers the head) and net.0+net.2+net.4
without (so the task-conditioned mask also reaches the class-IL logit bottleneck).

Iteration order: PROJECTIONS extends to shared (Iter 2) and learned (Iter 3) as those land; this
run is Iteration 1 (disjoint) only. class-IL eval (10-way, no eval masking); the oracle lives in
the modulator input, not the eval output (Methodology 7).

Run: uv run python results/pt5_taskid.py   (logs also to results/pt5_taskid.log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED = 42
LR = 1e-3            # SGD main net; matches the established naive-SGD+masked-loss ~0.63 reference
EP = 5              # epochs per task (SGD slowness is itself the masked-loss retention lever)
BUFFER = 1000       # ER replay buffer for the +ER conditions
PROJECTIONS = ["disjoint"]   # Iteration 1; add "shared" (Iter 2), "learned" (Iter 3) later

# target-config -> (neuromod_target, extra kwargs). gain has two forms only under a learned P;
# under a fixed binary P both collapse to the {0,1} gate, so it is a single config in Iter 1/2.
TARGET_CONFIGS = [
    ("plasticity", "plasticity", {}),
    ("weight_mask", "weight_mask", {}),
    ("gain", "activation", {"neuromod_gain_form": "unbounded"}),
]


def _base(**kw) -> CLConfig:
    return CLConfig(seed=SEED, lr=LR, epochs_per_task=EP, optimizer="sgd", **kw)


def _neurom(target, projection, masked, extra):
    """Build a neuromod CLConfig for a target x projection. masked=True -> naive+masked-loss;
    masked=False -> ER, masking off. weight_mask layer set depends on the condition."""
    kw = dict(
        use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
        neuromod_target=target, neuromod_projection=projection, **extra,
    )
    if target == "weight_mask":
        kw["neuromod_mask_layers"] = "0,2" if masked else "0,2,4"
    if masked:
        return _base(output_masking="loss", **kw)
    return _base(output_masking="none", er_buffer_size=BUFFER, **kw)


def run(tag, config, method):
    acc, forget = cl_train(config, method, no_wandb=True, sequence=None)
    print(f">>> {tag:34s} acc={acc:.4f}  forget={forget:.4f}\n")
    return acc, forget


print("==== pt5 shared baselines (SGD, 1 seed) ====")
naive_bar, naive_bar_f = run("naive-SGD + masked-loss", _base(output_masking="loss"), "naive")
er_bar, er_bar_f = run("ER-SGD (no masked-loss)", _base(output_masking="none", er_buffer_size=BUFFER), "er")

tables = {}
for projection in PROJECTIONS:
    print(f"\n############ pt5 projection = {projection} ############")
    rows = []
    for label, target, extra in TARGET_CONFIGS:
        s_acc, _ = run(f"{label} [{projection}] neurom", _neurom(target, projection, True, extra), "naive")
        e_acc, _ = run(f"{label} [{projection}] neurom+ER", _neurom(target, projection, False, extra), "er")
        rows.append((label, s_acc, e_acc))
    tables[projection] = rows

for projection in PROJECTIONS:
    print(f"\n=== pt5 ITERATION ({projection}) — SGD, seed={SEED}, class-IL eval ===")
    print(f"{'target-config':16s} {'naive+mask':>11s} {'neurom':>9s} {'(delta)':>9s} "
          f"{'ER':>9s} {'neurom+ER':>11s} {'(delta)':>9s}")
    print(f"{'baselines':16s} {naive_bar:>11.4f} {'-':>9s} {'-':>9s} {er_bar:>9.4f} {'-':>11s} {'-':>9s}")
    for label, s_acc, e_acc in tables[projection]:
        print(f"{label:16s} {naive_bar:>11.4f} {s_acc:>9.4f} {s_acc - naive_bar:>+9.4f} "
              f"{er_bar:>9.4f} {e_acc:>11.4f} {e_acc - er_bar:>+9.4f}")
    print("standalone accept-for-confirm: neurom clearly beats naive+masked-loss; "
          "+ER accept-for-confirm: neurom+ER beats ER by >=2pts.")
