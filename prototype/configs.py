from dataclasses import dataclass, field


@dataclass
class StandardConfig:
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    seed: int = 42
    val_size: int = 10_000
    optimizer: str = "adam"                 # adam (default) | sgd (plasticity main net + its vanilla ref, pt4 R4)
    use_neuromod: bool = False
    neuromod_variant: str = "feedforward"   # feedforward | stateful (Iter 4)
    neuromod_target: str = "activation"     # activation | plasticity | weight_mask
    neuromod_driver: str = "none"           # none | surprise | uncertainty | activation_stats (Iter 3)
    neuromod_learned_projection: bool = False
    neuromod_lr: float = 1e-3               # modulator optimizer LR (own LR, see Iter1 checklist item 4)
    neuromod_alpha_init: float = 0.95       # plasticity: initial gate α (≈ full plasticity)
    neuromod_mask_layer: int = 2            # weight_mask: which net.<idx> linear to mask (default 2nd: 400×400)
    neuromod_mask_layers: str = ""          # weight_mask: comma-sep net.<idx> linears to mask together (multi-layer, incl. output net.4). Empty = single neuromod_mask_layer (legacy)
    neuromod_mask_rank: int = 0             # weight_mask: 0 = full-rank head; r>0 = low-rank sigmoid(A·diag(g)·Bᵀ)
    neuromod_mask_init: float = 0.99        # weight_mask: initial mask value M (≈1 → near-vanilla at init)
    neuromod_stateful_hidden: int = 64      # stateful variant: GRU hidden size (Iter 4)
    neuromod_importance_lambda: float = 100.0  # importance target: gate alpha=1/(1+lambda*omega) (Iter 7)
    neuromod_gain_gate: str = "two_hidden"  # direct_gain target: last_hidden|two_hidden|last_hidden_output|two_hidden_output (pt4/5)
    neuromod_gain_bounded: bool = False     # direct_gain: gain=1+tanh(m) in [0,2] (bounded) vs 1+m (unbounded)
    # pt5 generalized driver system (CL-only; standard ignores these). Empty neuromod_drivers = legacy/off.
    neuromod_drivers: str = ""              # comma-sep name=mechanism, e.g. "task_id=onehot"; empty = legacy/off path
    neuromod_context: str = "image"         # image | none. pt5 uses none (drivers-only bottleneck)
    neuromod_projection: str = "disjoint"   # disjoint | shared | learned (the three pt5 iterations)
    neuromod_shared_frac: float = 0.5       # shared projection: fraction of all-task (all-ones) shared elements
    neuromod_proj_seed: int = 0             # layout seed for the fixed binary projections (disjoint/shared)
    neuromod_gain_form: str = "unbounded"   # learned-proj gain form: unbounded (h*(1+raw), inverts <0) | bounded01 (h*sigmoid(raw), suppress-only) | positive (h*softplus(raw+ln(e-1)), init 1.0, never inverts/freezes); inert under fixed P; weight_mask pins bounded01
    neuromod_granularity: str = "neuron"    # pt5 activation/plasticity: neuron (per-unit) | synapse (per-weight). weight_mask is always synapse; per-synapse variants reuse neuromod_mask_layers
    neuromod_plasticity_scope: str = "both" # per-neuron plasticity only: both (in+out) | in (incoming weights) | out (outgoing weights)
    neuromod_plasticity_layers: str = "0,2,4"  # per-neuron plasticity: which net.<idx> weight layers to gate (0=in->h0, 2=h0->h1, 4=h1->out). scope picks side per layer
    neuromod_gain_layers: str = "0,2"       # per-neuron gain (activation): which activations to gate (0=h0, 2=h1, 4=output logits). Any combination
    neuromod_modulate_bias: bool = False    # pt5 per-SYNAPSE gain/weight_mask/plasticity: also gate per-neuron biases (indep. P_bias per layer). Off = biases fully plastic (parity)
    neuromod_plasticity_init: float = 0.5   # pt5 iter3 LEARNED plasticity: initial per-side gate alpha (0.5 = sigmoid(0), reproduces iter3; higher adds a logit bias so more units start plastic). Ignored for fixed P / gain
    neuromod_sparsity_lambda: float = 0.0   # pt5 iter3 LEARNED projections: L1 penalty on the projected GATE (lambda*mean|gate|), pushing each task to a sparse active subset (toward the disjoint {0,1}). 0 = off (parity)
    neuromod_meta_replay: bool = False      # pt5 iter3 LEARNED plasticity STANDALONE: train P on a modulator-only replay buffer (SPEC meta-loss retention signal) while the main net stays naive. Off = meta-loss on current batch only (iter3 default). No effect for +ER (already replays) or fixed P
    neuromod_er_task_id: bool = True         # pt5: gate each BUFFERED sample by its OWN task mask P[j], not the current task P[t]. Applies to BOTH buffer paths: +ER (split the batch by task; forward targets gate the forward per task, plasticity gates each task's gradient per task) AND the standalone gain meta-loss. DEFAULT ON = correct task ids whenever a buffer is used. False = legacy wrong-task P[t] arm (ablation / reproducing pre-flip numbers). Inert without a buffer.
    neuromod_reset_moments: bool = False     # pt5 diagnostic (Adam-overwrite confound): clear the MAIN optimizer's Adam moments at every task switch (t>0). No-op under SGD / on the modulator optimizers. Off = parity


@dataclass
class CLConfig:
    lr: float = 1e-3
    epochs_per_task: int = 5
    batch_size: int = 64
    seed: int = 42
    ewc_lambda: float = 1e5
    ewc_samples: int = 200
    er_buffer_size: int = 200
    optimizer: str = "adam"                 # adam (baselines) | sgd (matched plasticity control)
    output_masking: str = "none"            # none (class-IL) | loss (mask train loss) | taskil (mask train+eval)  [pt3 Iter 5]
    val_frac: float = 0.1                   # CL tuning: fraction of each task's TRAIN set held out for val; applied ONLY in --val mode
    val_sequence_seed: int = 7              # CL tuning: task-order seed for the validation sequence (make_sequence); report uses the default order
    use_neuromod: bool = False
    neuromod_variant: str = "feedforward"   # feedforward | stateful (Iter 4)
    neuromod_target: str = "activation"     # activation | plasticity | weight_mask
    neuromod_driver: str = "none"           # none | surprise | uncertainty | activation_stats (Iter 3)
    neuromod_learned_projection: bool = False
    neuromod_lr: float = 1e-3               # modulator optimizer LR (own LR, see Iter1 checklist item 4)
    neuromod_alpha_init: float = 0.95       # plasticity: initial gate α (≈ full plasticity)
    neuromod_mask_layer: int = 2            # weight_mask: which net.<idx> linear to mask (default 2nd: 400×400)
    neuromod_mask_layers: str = ""          # weight_mask: comma-sep net.<idx> linears to mask together (multi-layer, incl. output net.4). Empty = single neuromod_mask_layer (legacy)
    neuromod_mask_rank: int = 0             # weight_mask: 0 = full-rank head; r>0 = low-rank sigmoid(A·diag(g)·Bᵀ)
    neuromod_mask_init: float = 0.99        # weight_mask: initial mask value M (≈1 → near-vanilla at init)
    neuromod_stateful_hidden: int = 64      # stateful variant: GRU hidden size (Iter 4)
    neuromod_importance_lambda: float = 100.0  # importance target: gate alpha=1/(1+lambda*omega) (Iter 7)
    neuromod_gain_gate: str = "two_hidden"  # direct_gain target: last_hidden|two_hidden|last_hidden_output|two_hidden_output (pt4/5)
    neuromod_gain_bounded: bool = False     # direct_gain: gain=1+tanh(m) in [0,2] (bounded) vs 1+m (unbounded)
    # pt5 generalized driver system. Empty neuromod_drivers = legacy/off path (numerically vanilla).
    neuromod_drivers: str = ""              # comma-sep name=mechanism, e.g. "task_id=onehot"; empty = legacy/off path
    neuromod_context: str = "image"         # image | none. pt5 uses none (drivers-only bottleneck)
    neuromod_projection: str = "disjoint"   # disjoint | shared | learned (the three pt5 iterations)
    neuromod_shared_frac: float = 0.5       # shared projection: fraction of all-task (all-ones) shared elements
    neuromod_proj_seed: int = 0             # layout seed for the fixed binary projections (disjoint/shared)
    neuromod_gain_form: str = "unbounded"   # learned-proj gain form: unbounded (h*(1+raw), inverts <0) | bounded01 (h*sigmoid(raw), suppress-only) | positive (h*softplus(raw+ln(e-1)), init 1.0, never inverts/freezes); inert under fixed P; weight_mask pins bounded01
    neuromod_granularity: str = "neuron"    # pt5 activation/plasticity: neuron (per-unit) | synapse (per-weight). weight_mask is always synapse; per-synapse variants reuse neuromod_mask_layers
    neuromod_plasticity_scope: str = "both" # per-neuron plasticity only: both (in+out) | in (incoming weights) | out (outgoing weights)
    neuromod_plasticity_layers: str = "0,2,4"  # per-neuron plasticity: which net.<idx> weight layers to gate (0=in->h0, 2=h0->h1, 4=h1->out). scope picks side per layer
    neuromod_gain_layers: str = "0,2"       # per-neuron gain (activation): which activations to gate (0=h0, 2=h1, 4=output logits). Any combination
    neuromod_modulate_bias: bool = False    # pt5 per-SYNAPSE gain/weight_mask/plasticity: also gate per-neuron biases (indep. P_bias per layer). Off = biases fully plastic (parity)
    neuromod_plasticity_init: float = 0.5   # pt5 iter3 LEARNED plasticity: initial per-side gate alpha (0.5 = sigmoid(0), reproduces iter3; higher adds a logit bias so more units start plastic). Ignored for fixed P / gain
    neuromod_sparsity_lambda: float = 0.0   # pt5 iter3 LEARNED projections: L1 penalty on the projected GATE (lambda*mean|gate|), pushing each task to a sparse active subset (toward the disjoint {0,1}). 0 = off (parity)
    neuromod_meta_replay: bool = False      # pt5 iter3 LEARNED plasticity STANDALONE: train P on a modulator-only replay buffer (SPEC meta-loss retention signal) while the main net stays naive. Off = meta-loss on current batch only (iter3 default). No effect for +ER (already replays) or fixed P
    neuromod_er_task_id: bool = True         # pt5: gate each BUFFERED sample by its OWN task mask P[j], not the current task P[t]. Applies to BOTH buffer paths: +ER (split the batch by task; forward targets gate the forward per task, plasticity gates each task's gradient per task) AND the standalone gain meta-loss. DEFAULT ON = correct task ids whenever a buffer is used. False = legacy wrong-task P[t] arm (ablation / reproducing pre-flip numbers). Inert without a buffer.
    neuromod_reset_moments: bool = False     # pt5 diagnostic (Adam-overwrite confound): clear the MAIN optimizer's Adam moments at every task switch (t>0). No-op under SGD / on the modulator optimizers. Off = parity


# ---------------------------------------------------------------------------
# Phase 6 best configs — selected on validation data only (never the test set)
# ---------------------------------------------------------------------------
# (a) Standard-MNIST sweep: lr ∈ {3e-4, 1e-3} × epochs ∈ {10, 20}, 1 seed
#     Selected by val_acc on held-out 10k of MNIST training set.
BEST_STANDARD_VANILLA = StandardConfig(lr=3e-4, epochs=20)
BEST_STANDARD_NEUROMOD = StandardConfig(lr=3e-4, epochs=20, use_neuromod=True)

# (b) CL sweep: lr ∈ {3e-4, 1e-3} × epochs_per_task ∈ {5, 10},
#     plus method-specific knobs (EWC λ, ER buffer, neuromod learned_projection).
#     Validation sequence: make_sequence(7).  Selected by avg_final_acc.
BEST_CL_NAIVE = CLConfig(lr=1e-3, epochs_per_task=5)
BEST_CL_JOINT = CLConfig(lr=1e-3, epochs_per_task=10)
BEST_CL_EWC = CLConfig(lr=3e-4, epochs_per_task=5, ewc_lambda=1e4)
BEST_CL_ER = CLConfig(lr=3e-4, epochs_per_task=5, er_buffer_size=1000)
BEST_CL_NEUROMOD = CLConfig(lr=1e-3, epochs_per_task=10, use_neuromod=True)
