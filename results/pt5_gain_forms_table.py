"""Merge the two pt5 gain-form sweeps into ONE canonical table (no re-running, no transcription).

Parses:
  - results/pt5_gain_forms.log        (no buffer)  -> `no-buf` (neurom) and `er-cur` (neurom+er,
                                                      er_task_id defaulted OFF)
  - results/pt5_gain_forms_buffer.log (buffer)     -> `meta-own` / `meta-cur` / `er-own`

The two sweeps share seed/lr/epochs/buffer/config, and meta_replay is inert for +ER cells
(gain_meta_replay_on requires `not use_replay`), so the no-buffer sweep's neurom+er cells ARE the
er_task_id=OFF arm (`er-cur`) for this sweep's `er-own`. Hence er-cur was not re-run.

Arms per cell:
  no-buf   standalone, no buffer at all
  meta-cur standalone + modulator-only replay buffer, every meta batch under the CURRENT gate P[t]
  meta-own standalone + modulator-only replay buffer, each task j under its OWN gate P[j]
  er-cur   +ER, replayed sample under the current gate P[t]   (legacy/default)
  er-own   +ER, replayed sample under its own gate P[j]       (--neuromod-er-task-id)

Run: uv run python results/pt5_gain_forms_table.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
NOBUF_LOG = ROOT / "pt5_gain_forms.log"
BUF_LOG = ROOT / "pt5_gain_forms_buffer.log"

OPTS = ["sgd", "adam"]
MECHS = ["gain-neuron", "gain-synapse"]
FORMS = ["unbounded", "bounded01", "positive"]

LINE = re.compile(r">>> \[(\w+)\]\s+(.*?)\s+acc=([\d.]+)\s+forget=([\d.]+)")


def parse(path):
    """-> {(opt, rest_of_tag): (acc, forget)}"""
    out = {}
    if not path.exists():
        sys.exit(f"missing {path}")
    for line in path.read_text().splitlines():
        m = LINE.match(line.strip())
        if m:
            opt, tag, acc, forget = m.groups()
            out[(opt, tag.strip())] = (float(acc), float(forget))
    return out


nobuf = parse(NOBUF_LOG)
buf = parse(BUF_LOG)


def get(d, opt, tag):
    v = d.get((opt, tag))
    return v if v else (float("nan"), float("nan"))


print("=" * 118)
print("pt5 GAIN FORMS — canonical merged table (class-IL, projection=learned, seed 42, lr=1e-3, ep=5, buffer=1000)")
print("STANDALONE: main net naive, buffer trains ONLY the gain P  |  +ER: buffer trains the backbone")
print("=" * 118)

for opt in OPTS:
    nb = get(buf, opt, "naive (baseline)")[0]
    eb = get(buf, opt, "er (baseline)")[0]
    nb0 = get(nobuf, opt, "naive (baseline)")[0]
    eb0 = get(nobuf, opt, "er (baseline)")[0]
    agree = abs(nb - nb0) < 5e-4 and abs(eb - eb0) < 5e-4
    print(f"\n--- optimizer={opt.upper()} ---  baselines: naive={nb:.4f}  er={eb:.4f}   "
          f"(cross-sweep baseline agreement: {'OK' if agree else '*** DIFFER ***'})")
    print(f"  {'mechanism':13s} {'form':10s} | {'no-buf':>7s} {'meta-cur':>8s} {'meta-own':>8s} "
          f"{'d-meta':>7s} {'own-nv':>7s} | {'er-cur':>7s} {'er-own':>7s} {'d-er':>7s} {'own-er':>7s}")
    for mech in MECHS:
        for form in FORMS:
            nbf = get(nobuf, opt, f"{mech} {form} neurom")[0]
            ecur = get(nobuf, opt, f"{mech} {form} neurom+er")[0]
            mown = get(buf, opt, f"{mech} {form} buf-meta-own")[0]
            mcur = get(buf, opt, f"{mech} {form} buf-meta-cur")[0]
            eown = get(buf, opt, f"{mech} {form} er-own")[0]
            star = " *" if (eown - eb) >= 0.02 else "  "
            print(f"  {mech:13s} {form:10s} | {nbf:>7.4f} {mcur:>8.4f} {mown:>8.4f} "
                  f"{mown - mcur:>+7.4f} {mown - nb:>+7.4f} | {ecur:>7.4f} {eown:>7.4f} "
                  f"{eown - ecur:>+7.4f} {eown - eb:>+7.4f}{star}")

print("\nd-meta = per-task minus wrong-task META gate (standalone)")
print("d-er   = per-task minus wrong-task gate under ER (er-own vs er-cur) = the --neuromod-er-task-id effect")
print("own-nv = meta-own vs the naive baseline | own-er = er-own vs the ER baseline; * = clears the +2pt bar")
print("\nCaveats: ORACLE task id at train+eval => task-IL-style on the class-IL metric. The standalone")
print("meta arms USE the buffer (replay on the MODULATOR, not the backbone), so meta-own is NOT")
print("apples-to-apples with no-buf/naive, and clearing ER there is not beating ER. 1 seed, lambda=0.")
