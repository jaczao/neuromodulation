"""Task selection — how a per-task gate is chosen WITHOUT a task id at eval.

Promoted from results/pt6_driver_mechanisms.py and results/pt6_synapse.py on second use (the
meta-learned-gate / task-free directions). Copy-forward: the pt6 scripts are unchanged and still
reproduce; `verify_anchors.py --part pt6` checks these primitives against their frozen numbers.

WHY THIS MATTERS. Every pt5 result carried an oracle caveat — a true task id selected the gate row at
eval, so a class-IL number was really a task-IL-style result. This is the only machinery in the
project that removed the oracle and still reached ER parity: soft_mlp 0.8850 and embedding 0.8888
against ER-adam 0.8946, genuinely oracle-free.

THE FACTORIZATION. pt6's real content is that two things are separable:
    INFERENCE   who decides which task a sample belongs to  -> a posterior over tasks
    GATE TABLE  what each task's gate actually is           -> rows P[t]
Resolution is then just how the posterior meets the table (oracle / hard / soft / per-image), and one
training run supports every mode. Keeping them separate is what lets a new problem swap the inference
stage without touching the gate, which is the whole reason this was worth promoting.

FINDINGS THAT CONSTRAIN USE (each cost a study; do not re-derive):

  pred ~= oracle x infer.  The routing law. A task-DIFFERENTIATED gate has ~zero tolerance for
  misrouting — a wrong task's gate is actively wrong, not merely unhelpful. It held for hard routing
  (pt3 iter-8, pt6) and again for a soft factorized posterior (pt8: 0.9972 x 0.8878 = 0.885 vs an
  observed 0.8875). So oracle-free accuracy is capped by inference accuracy, and improving the gate
  while inference stays put buys nothing.

  A LEARNED selector is the lever; a fixed one is not.  Replay-trained g(x) reaches 0.86-0.88 task
  accuracy against nearest-prototype's fixed 0.759, and that gap is the whole difference between
  ~0.75 (below ER) and ~0.88 (ER parity).

  REPLAY is what makes the selector work.  With no buffer, `infer` collapses to 0.198 ~= chance(1/5)
  and oracle-free accuracy dies (0.463) even though the ORACLE stays 0.933. The gate is fine; the
  SELECTOR forgot. Any rehearsal-free use of this needs a different way to keep g(x) alive.

  soft ~= hard for a LEARNED selector (within noise, no consistent direction): a well-trained
  selector is confident, so softmax ~= one-hot. Softness pays only when the posterior is DIFFUSE —
  the prototype case, where soft-nearest has a genuine interior peak at tau ~ 0.03-0.05 and beats
  hard nearest under SGD. Do NOT generalise "soft helps" from prototypes to a learned net.

  TRAIN THE GATE ROWS ON TRUE TASK IDS, not on the soft posterior.  Removing the train/eval mismatch
  by training on the blend HURTS (buf-own/sgd -0.103, and the oracle drops in every cell): a one-hot
  gives each row a clean UNMIXED gradient, while a blend smears every sample across all rows so they
  differentiate less. The mismatch was never the problem.

  The selector does not need TRUE task labels.  Pseudo-labels from the main net's own output layer
  (argmax(logits)//2, detached) match true-id training. NB this does NOT make the method label-free —
  the gate table and backbone still use real labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

RESOLUTION_MODES = ("oracle", "hard", "soft", "per-image")

# soft-nearest temperatures over MEAN squared distance (pt6 sweep). Interior peak ~0.03-0.05;
# tau -> 0 converges to hard nearest, so there is nothing below 0.03 to gain.
PROTOTYPE_TAUS = (0.03, 0.1, 0.3, 1.0)


class TaskInferenceNet(nn.Module):
    """g(x): in_dim -> hid -> T. The replay-trained selector, and the shared trunk of both mechanisms.

    Train it on task-CE over the reservoir (current + replay), NOT on the main loss. Without replay it
    forgets completely and the whole approach collapses to chance regardless of how good the gate is.
    """
    def __init__(self, n_tasks=5, hid=128, in_dim=784):
        super().__init__()
        self.gh = nn.Linear(in_dim, hid)
        self.go = nn.Linear(hid, n_tasks)

    def embed(self, x):
        return F.relu(self.gh(x.view(x.size(0), -1)))

    def task_logits(self, x):
        return self.go(self.embed(x))

    def posterior(self, x):
        return F.softmax(self.task_logits(x), dim=1)

    def params(self):
        return list(self.parameters())


class NearestPrototype:
    """FIXED inference from per-task mean images — the weak baseline the learned selector beats.

    Kept because it is the control that makes the learned selector's contribution legible (infer
    0.759 vs 0.88). `tau=None` gives a hard nearest assignment; a finite tau gives the diffuse
    posterior that soft resolution actually helps.
    """
    def __init__(self, mus, center=None):
        self.mus = mus                                   # (T, in_dim)
        self.center = center

    def _d(self, x):
        d = x.view(x.size(0), -1)
        return d if self.center is None else d - self.center

    def task_logits(self, x):
        return -(self._d(x)[:, None, :] - self.mus[None]).pow(2).mean(-1)

    def posterior(self, x, tau=0.1):
        return F.softmax(self.task_logits(x) / tau, dim=1)

    def hard(self, x):
        return (self._d(x)[:, None, :] - self.mus[None]).pow(2).sum(-1).argmin(1)


class GateTable(nn.Module):
    """Per-task gate rows P:(T, D). Zero-init, so every row starts at parity (gamma = 1).

    `oracle` and `blend` are the two ways to read it; both are differentiable in P. Train through
    `oracle(tids)` with TRUE ids so each row gets an unmixed gradient (see the module docstring).

    REPRODUCIBILITY. `P[tids]` is advanced indexing, and its backward is a SCATTER-ADD accumulated
    atomically — which is NONDETERMINISTIC on MPS (measured: max|d| 3.8e-6 across identical runs,
    versus exactly 0 for a matmul). Over a full run that compounds through Adam into ~3e-3 of weight
    drift and ~0.002-0.003 of final accuracy, which is why pt6's soft_mlp cells do not reproduce
    bit-exact even from their own unmodified code. Pass `deterministic=True` to route through
    `one_hot(tids) @ P` instead: mathematically identical, matmul backward, bit-reproducible, at the
    cost of a (B, T) matmul. Default False preserves pt6 parity.
    """
    def __init__(self, n_tasks, dim):
        super().__init__()
        self.n_tasks = n_tasks
        self.P = nn.Parameter(torch.zeros(n_tasks, dim))

    def rows(self):
        return self.P

    def oracle(self, tids, deterministic=False):
        if deterministic:
            return F.one_hot(tids, self.n_tasks).to(self.P.dtype) @ self.P
        return self.P[tids]

    def blend(self, posterior):
        return posterior @ self.P                        # sum_t p_t P[t]

    def hard(self, tids):
        return self.P[tids]

    def params(self):
        return [self.P]


class SoftMLPSelector(nn.Module):
    """Gate table + replay-trained inference net. Oracle-free at eval via the posterior.

    Construction order (table, then inference net) is load-bearing for reproducing pt6 bit-exact:
    the zero-init table consumes no RNG, the two Linears do.
    """
    has_inference = True

    def __init__(self, n_tasks=5, dim=810, hid=128, in_dim=784, deterministic=False):
        super().__init__()
        self.deterministic = deterministic            # see GateTable: P[tids] backward is nondet on MPS
        self.table = GateTable(n_tasks, dim)
        self.inf = TaskInferenceNet(n_tasks, hid, in_dim)

    def task_logits(self, x):
        return self.inf.task_logits(x)

    def train_gate(self, x, tids):
        return self.table.oracle(tids, self.deterministic)   # TRUE ids at train (unmixed rows)

    def eval_gate(self, x, mode="soft", tids=None):
        if mode == "oracle":
            return self.table.oracle(tids, self.deterministic)
        if mode == "soft":
            return self.table.blend(self.inf.posterior(x))
        if mode == "hard":
            return self.table.hard(self.task_logits(x).argmax(1))
        raise ValueError(f"{mode!r} not supported by SoftMLPSelector; use oracle | soft | hard")

    def gate_params(self):
        return self.table.params()

    def inf_params(self):
        return self.inf.params()


class EmbeddingSelector(nn.Module):
    """Per-image gate = proj(e(x)) off the inference net's hidden layer. Oracle-free BY CONSTRUCTION.

    The cleanest of the two: continuous, no discrete inference step, no gate table to route into — so
    the routing law does not apply to it in the same hard way. `lin` and `mlp` performed alike; `lin`
    is the parameter-cheap default.

    Construction order (inference net, then projection) mirrors pt6 for RNG parity.
    """
    has_inference = True

    def __init__(self, n_tasks=5, dim=810, hid=128, in_dim=784, proj="lin"):
        super().__init__()
        self.proj = proj
        self.inf = TaskInferenceNet(n_tasks, hid, in_dim)
        if proj == "lin":
            self.W = nn.Parameter(torch.zeros(hid, dim))
        elif proj == "mlp":
            self.pf1 = nn.Linear(hid, hid); self.pf2 = nn.Linear(hid, dim)
            nn.init.normal_(self.pf2.weight, std=1e-3); nn.init.zeros_(self.pf2.bias)
        else:
            raise ValueError(f"unknown proj {proj!r}; known: lin | mlp")

    def task_logits(self, x):
        return self.inf.task_logits(x)

    def gate_per_sample(self, x):
        e = self.inf.embed(x)
        return e @ self.W if self.proj == "lin" else self.pf2(F.relu(self.pf1(e)))

    def train_gate(self, x, tids=None):
        return self.gate_per_sample(x)                   # per-image, no oracle anywhere

    def eval_gate(self, x, mode="per-image", tids=None):
        if mode != "per-image":
            raise ValueError(f"EmbeddingSelector is per-image only, got {mode!r}")
        return self.gate_per_sample(x)

    def gate_params(self):
        return ([self.W] if self.proj == "lin"
                else list(self.pf1.parameters()) + list(self.pf2.parameters()))

    def inf_params(self):
        return self.inf.params()


# ------------------------------- per-synapse resolution -------------------------------
def synapse_mats(P, layers):
    """Slice a flat per-synapse gate table P:(T, n_syn) into per-layer (T, d_out, d_in) blocks."""
    out, off = [], 0
    for a, b in layers:
        out.append(P[:, off:off + a * b].view(P.size(0), a, b)); off += a * b
    return out


def grouped_synapse(wb, mats, X, tids):
    """Hard routing per-synapse: each sample carries ONE task, so group into <= T masked matmuls.

    This is the TRAINING path (true ids) and the hard-eval path. No per-sample Gamma anywhere.
    """
    X = X.view(X.size(0), -1)
    out = torch.zeros(X.size(0), wb[-1][0].size(0), device=X.device, dtype=X.dtype)
    for t in tids.unique():
        idx = (tids == t).nonzero().squeeze(1)
        h = X[idx]
        for li, (W, b) in enumerate(wb):
            h = F.linear(h, (1 + mats[li][t]) * W, b)
            if li < len(wb) - 1:
                h = F.relu(h)
        out = out.index_copy(0, idx, h)
    return out


def soft_blend_synapse(wb, mats, X, posterior):
    """EXACT soft blend in T matmuls per layer — no (B, d_out, d_in) expansion, no chunking.

    Because Gamma_i = sum_t p_it Gamma_t and (Gamma . W)x is LINEAR in Gamma:

        (Gamma_i . W) x_i + b  =  sum_t p_it [ (Gamma_t . W) x_i + b ]

    and the bias rides along because sum_t p_it = 1. Verified against the grouped path at
    max|delta| = 2.1e-07. This supersedes the earlier belief that a soft per-synapse blend needed a
    per-sample Gamma that had to be chunked under no_grad — it needs neither.
    """
    h = X.view(X.size(0), -1)
    T = posterior.size(1)
    for li, (W, b) in enumerate(wb):
        acc = 0
        for t in range(T):
            acc = acc + posterior[:, t:t + 1] * F.linear(h, (1 + mats[li][t]) * W, b)
        h = acc
        if li < len(wb) - 1:
            h = F.relu(h)
    return h


def routing_ceiling(oracle_acc, infer_acc):
    """pred ~= oracle x infer. Use it BEFORE running a cell: if this is below your baseline, a better
    gate cannot rescue it and the inference stage is where the work has to go."""
    return oracle_acc * infer_acc
