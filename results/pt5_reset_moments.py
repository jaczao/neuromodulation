"""pt5 diagnostic (user-requested) — Adam with MOMENTS RESET at each task switch.

Motivation (the Adam-overwrite confound, CLAUDE.md): under masked loss on class-IL, Adam overwrites
old-task weights far faster than SGD (naive+masked 0.39 Adam vs 0.63 SGD) because its first/second-
moment estimates carry over across tasks. This clears the MAIN optimizer's Adam state at every task
switch (t>0) so each new task starts from fresh moments (like a freshly-built Adam), and asks: does
slowing that carryover-driven backbone drift help the learned-projection cells, where the soft gate
does NOT freeze the backbone and the standalone note says "Adam drifts fast so standalone gates lag"?

MAIN net only — the gain/plasticity modulator optimizers (which train the per-task P rows) are left
untouched, so this isolates main-net weight overwriting.

Grid: 4 mechs {gain,plast}x{per-neuron,per-synapse} x 2 arms x {reset OFF, reset ON}, all:
  class-IL, ADAM, LEARNED projection, seed 42, lr=1e-3, ep=5, buffer=1000, er_task_id=ON (default).
Arms:
  - "standalone buf": method=naive + --neuromod-meta-replay (modulator-only replay buffer trains ONLY
      P; main net trains naive on the current task), masked-loss ON ('loss'). Gating differs by target:
        * GAIN -> "own" (er_task_id=ON): the gain meta-loop is per-task, so each buffered task-j sample
          is forwarded under ITS OWN gate P[j].
        * PLASTICITY -> "cur" (er_task_id=OFF): the plasticity standalone meta-loop is NOT per-task
          split — buffered samples always run under the current task's gate P[t]. er_task_id is inert
          here (neither er_task_id_on nor the gain meta-loop reads it on this path), so "cur" is the
          only available/accurate label; the flag value does not change the result (verified inert).
  - "er own": method=er + --neuromod-er-task-id ON (each replayed sample under its own P[j]),
      masked-loss OFF ('none', the pt5 ER convention).
Learned projection for BOTH arms: it is required by the standalone meta arm, and it is the regime
where the backbone actually drifts (the disjoint freeze already zeroes old-task grads, so a moment
reset would be ~inert there). reset OFF reproduces the existing learned-projection numbers.

Oracle caveat carries (task id selects P[i] at eval -> task-IL-style on the class-IL metric). 1 seed.

Run: uv run python results/pt5_reset_moments.py   (redirect to .log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype.configs import CLConfig
from prototype.train import cl_train

SEED, LR, EP, BUFFER = 42, 1e-3, 5, 1000

# Mechanism -> the pt5 target/granularity/layer knobs (gain_form=unbounded throughout, as in the
# learned-projection gain studies).
MECH = {
    "gain-neuron":  dict(neuromod_target="activation", neuromod_gain_form="unbounded",
                         neuromod_granularity="neuron",  neuromod_gain_layers="0,2"),
    "gain-synapse": dict(neuromod_target="activation", neuromod_gain_form="unbounded",
                         neuromod_granularity="synapse", neuromod_mask_layers="0,2"),
    "plast-neuron":  dict(neuromod_target="plasticity", neuromod_granularity="neuron",
                          neuromod_plasticity_layers="0,2,4", neuromod_plasticity_scope="both"),
    "plast-synapse": dict(neuromod_target="plasticity", neuromod_granularity="synapse",
                          neuromod_mask_layers="0,2"),
}
MECHS = ["gain-neuron", "gain-synapse", "plast-neuron", "plast-synapse"]
ARMS = ["standalone-buf", "er-own"]


def gating_for(arm, mech):
    """(er_task_id, label). Gain standalone meta-loop is per-task -> 'own'; plasticity standalone
    meta-loop is not per-task -> 'cur' (er_task_id inert on that path). +ER is always 'own'."""
    if arm == "er-own":
        return True, "own"
    return (True, "own") if mech.startswith("gain") else (False, "cur")


def run(mech, arm, reset):
    method = "naive" if arm == "standalone-buf" else "er"
    # class-IL: naive uses masked-loss ON ('loss'); ER uses masked-loss OFF ('none') per pt5 convention.
    masking = "loss" if arm == "standalone-buf" else "none"
    er_task_id, gate_lbl = gating_for(arm, mech)
    c = CLConfig(
        seed=SEED, lr=LR, epochs_per_task=EP, optimizer="adam", output_masking=masking,
        use_neuromod=True, neuromod_drivers="task_id=onehot", neuromod_context="none",
        neuromod_projection="learned", er_buffer_size=BUFFER,
        neuromod_er_task_id=er_task_id,
        neuromod_meta_replay=(arm == "standalone-buf"),       # modulator-only replay buffer (standalone)
        neuromod_reset_moments=reset,
        **MECH[mech],
    )
    a, f = cl_train(c, method, no_wandb=True, sequence=None)
    print(f">>> [{arm:14s} {gate_lbl}] {mech:13s} reset={'ON ' if reset else 'OFF'}  "
          f"acc={a:.4f} forget={f:.4f}", flush=True)
    return a, f


RES = {}
for arm in ARMS:
    for mech in MECHS:
        for reset in (False, True):
            RES[(arm, mech, reset)] = run(mech, arm, reset)

print("\n\n" + "=" * 84)
print("pt5 ADAM MOMENTS-RESET-AT-TASK-SWITCH — does slowing Adam's backbone drift help?")
print("=" * 84)
print("class-IL, Adam, LEARNED projection, seed 42, lr=1e-3, ep=5, buffer=1000, er_task_id=ON.")
for arm in ARMS:
    print(f"\n--- {arm} ---")
    print(f"  {'mechanism':14s} {'gate':4s} {'OFF acc':>8s} {'ON acc':>8s} {'dAcc':>8s} "
          f"{'OFF forg':>9s} {'ON forg':>8s} {'dForg':>8s}")
    for mech in MECHS:
        _, gate_lbl = gating_for(arm, mech)
        oa, of = RES[(arm, mech, False)]; na, nf = RES[(arm, mech, True)]
        print(f"  {mech:14s} {gate_lbl:4s} {oa:>8.4f} {na:>8.4f} {na - oa:>+8.4f} "
              f"{of:>9.4f} {nf:>8.4f} {nf - of:>+8.4f}")
print("\nreset OFF = parity (Adam moments carry across tasks); reset ON = MAIN optimizer state cleared "
      "at every task switch (modulator optimizers untouched). Oracle (task id at eval) -> task-IL-style "
      "on the class-IL metric. 1 seed. gate: 'own' = each buffered sample under its own P[j] (gain "
      "standalone meta-loop + all +ER); 'cur' = current task P[t] (plasticity standalone meta-loop is "
      "not per-task split, so er_task_id is inert there -> 'cur' is the accurate label).")
