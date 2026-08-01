"""The TSV results ledger: append-only records, --resume, ledger-sourced paired deltas, tables.

Replaces three incompatible idioms that grew across the pt3-pt8 studies (a tag-string-keyed file, a
positional-column file, and one header'd file), each with its own hand-rolled resume and delta logic.
One schema now: a HEADER row, explicit KEY columns and METRIC columns, keys forming the resume
identity.

Two rules are baked in because breaking them produced wrong readings before:

  PAIRED DELTAS TAKE THEIR SEEDS FROM THE LEDGER, never from a SEEDS constant. A width carrying extra
  seeds otherwise pairs only the first few while the mean column averages all of them — a silent
  mismatch that is invisible in the output.

  A DELTA IS AGAINST THE RNG-MATCHED CONTROL. `delta_table` reports against the `free` dead-gate arm
  and warns when it is absent, because comparing to the plain baseline attributes the RNG shift from
  constructing the modulator to the mechanism. That error grows as the baseline destabilises.

The COST_METRICS block is appended to any ledger that opts in, so memory / parameter / compute
accounting is recorded per cell rather than reconstructed later.
"""
from pathlib import Path

import numpy as np

from .cost import Cost

# Accounting columns every direction reports (see cost.py). `regime` is a KEY, not a metric.
COST_METRICS = ["backbone_params", "extra_params", "param_ratio", "buffer_bytes",
                "fwd_train", "bwd_train", "fwd_infer", "bwd_infer"]

NOISE_FLOOR = 0.007          # 1-seed MPS run-to-run spread; treat |delta| below this as null


class Ledger:
    """Append-only TSV keyed by `keys`, carrying `metrics` (+ optional cost columns).

    >>> led = Ledger(path, keys=["regime", "mech", "arm", "opt", "seed"],
    ...              metrics=["acc", "forget", "probe"], with_cost=True)
    >>> if led.is_done(regime="normal", mech="all4", arm="er-own", opt="adam", seed=42): ...
    >>> led.append(dict(regime="normal", mech="all4", arm="er-own", opt="adam", seed=42),
    ...            dict(acc=0.8816, forget=0.02, probe=0.462), cost=cost)
    """

    def __init__(self, path, keys, metrics, with_cost=False):
        self.path = Path(path)
        self.keys = list(keys)
        self.metrics = list(metrics) + (COST_METRICS if with_cost else [])
        self.with_cost = with_cost
        self.columns = self.keys + self.metrics
        if self.path.exists():
            self._check_header()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("\t".join(self.columns) + "\n")

    # ------------------------- io -------------------------
    def _check_header(self):
        first = self.path.read_text().splitlines()[:1]
        if not first:
            self.path.write_text("\t".join(self.columns) + "\n")
            return
        have = first[0].split("\t")
        if have != self.columns:
            raise ValueError(
                f"ledger schema drift at {self.path.name}\n  on disk: {have}\n  expected: "
                f"{self.columns}\nA changed schema silently invalidates --resume; write a new ledger "
                f"file rather than mixing schemas.")

    def rows(self):
        """All records as dicts, numerics coerced. Empty if the ledger holds only its header."""
        lines = self.path.read_text().splitlines()
        out = []
        for ln in lines[1:]:
            if not ln.strip():
                continue
            vals = ln.split("\t")
            out.append({c: _coerce(v) for c, v in zip(self.columns, vals)})
        return out

    def append(self, key: dict, metrics: dict, cost: Cost | None = None):
        missing = [k for k in self.keys if k not in key]
        if missing:
            raise KeyError(f"ledger append missing key column(s): {missing}")
        row = dict(key)
        row.update(metrics)
        if cost is not None:
            row.update(cost.as_row())
        elif self.with_cost:
            raise ValueError("ledger was created with_cost=True but append() got cost=None")
        cells = [_fmt(row.get(c, "")) for c in self.columns]
        with open(self.path, "a") as f:
            f.write("\t".join(cells) + "\n")

    # ------------------------- resume -------------------------
    def done(self):
        """Set of key tuples already recorded."""
        return {tuple(str(r[k]) for k in self.keys) for r in self.rows()}

    def is_done(self, **key):
        return tuple(str(key[k]) for k in self.keys) in self.done()


# ------------------------- aggregation -------------------------
def where(rows, **conds):
    """Filter rows by exact field match (values compared as strings, so 42 matches '42')."""
    def ok(r):
        return all(str(r.get(k)) == str(v) for k, v in conds.items())
    return [r for r in rows if ok(r)]


def summarize(rows, metric="acc", group_by=("arm",)):
    """(group tuple) -> (mean, std, n) over whatever rows share that group."""
    out = {}
    for r in rows:
        g = tuple(r.get(k) for k in group_by)
        out.setdefault(g, []).append(float(r[metric]))
    return {g: (float(np.mean(v)), float(np.std(v)), len(v)) for g, v in out.items()}


def paired_delta(rows, arm, ref, metric="acc", arm_col="arm", pair_col="seed", **conds):
    """Per-`pair_col` paired delta (arm - ref) -> (mean, std, n, per-pair list), or None.

    The pairing values are read FROM THE ROWS, never from a constant: a cell carrying extra seeds
    must pair on all of them or the delta and the mean disagree without saying so.
    """
    sel = where(rows, **conds) if conds else rows
    pairs = sorted({r[pair_col] for r in sel}, key=lambda v: (str(type(v)), v))
    d = []
    for p in pairs:
        a = [float(r[metric]) for r in sel if r[arm_col] == arm and r[pair_col] == p]
        b = [float(r[metric]) for r in sel if r[arm_col] == ref and r[pair_col] == p]
        if a and b:
            d.append(a[0] - b[0])
    if not d:
        return None
    return float(np.mean(d)), float(np.std(d)), len(d), d


def delta_table(rows, arms, baseline, control="free", metric="acc",
                arm_col="arm", pair_col="seed", noise_floor=NOISE_FLOOR, **conds):
    """Formatted mean+-std per arm plus paired deltas against the baseline AND the RNG-matched control.

    Warns when `control` is missing: without it a delta credits the mechanism with the RNG shift that
    merely constructing the modulator causes.
    """
    sel = where(rows, **conds) if conds else rows
    have_control = any(r[arm_col] == control for r in sel)
    lines = []
    if not have_control:
        lines.append(f"  !! no `{control}` control in these rows — d-{baseline} attributes the "
                     f"modulator-construction RNG shift to the mechanism. Add it.")
    stats = summarize(sel, metric=metric, group_by=(arm_col,))
    head = f"  {'arm':<14s}{metric:>12s}{'sd':>9s}{'n':>4s}{'d-' + baseline:>12s}"
    if have_control:
        head += f"{'d-' + control:>12s}"
    lines.append(head)
    for arm in arms:
        if (arm,) not in stats:
            continue
        m, s, n = stats[(arm,)]
        row = f"  {arm:<14s}{m:>12.4f}{s:>9.4f}{n:>4d}"
        for ref in ([baseline, control] if have_control else [baseline]):
            d = paired_delta(sel, arm, ref, metric=metric, arm_col=arm_col, pair_col=pair_col)
            if d is None or arm == ref:
                row += f"{'-':>12s}"
            else:
                flag = " " if abs(d[0]) >= noise_floor else "~"      # ~ = inside the noise floor
                row += f"{d[0]:>+11.4f}{flag}"
        lines.append(row)
    lines.append(f"  (~ = |delta| < {noise_floor} 1-seed noise floor: read as null)")
    return "\n".join(lines)


# ------------------------- helpers -------------------------
def _coerce(v):
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)
