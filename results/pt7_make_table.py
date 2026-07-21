"""Turn results/pt7_results.tsv (the resume ledger) into a readable findings table.

Groups cells by arm x optimizer, shows pred, the honest delta vs the same-optimizer baseline
(naive for nobuf/buf-own, er for er-own), the `true` diagnostic upper bound, the task-decodability
probe, and per-layer |gate|. Run after the grid: `uv run python results/pt7_make_table.py`.
"""
from pathlib import Path

TSV = Path(__file__).resolve().parent / "pt7_results.tsv"


def load():
    base, cells = {}, {}
    for ln in TSV.read_text().splitlines():
        if not ln.strip():
            continue
        f = ln.split("\t")
        tag = f[0]
        if len(f) == 2:                       # baseline
            base[tag] = float(f[1])
        else:
            cells[tag] = dict(pred=float(f[1]), true=float(f[2]), probe=float(f[3]),
                              h0=float(f[4]), h1=float(f[5]), out=float(f[6]))
    return base, cells


def main():
    base, cells = load()
    b = {k.split("|")[0] + "|" + k.split("|")[-1]: v for k, v in base.items()}   # naive|sgd -> acc
    print("baselines:")
    for opt in ("sgd", "adam"):
        print(f"  naive {opt}: {b.get('naive|'+opt, float('nan')):.4f}    "
              f"er {opt}: {b.get('er|'+opt, float('nan')):.4f}")
    print()
    hdr = f"  {'driver':10s} {'gran':7s} {'arm':7s} {'opt':4s} | {'pred':>7s} {'Δbase':>7s} " \
          f"{'true':>7s} {'probe':>6s} | {'|g| h0/h1/out':>18s}"
    order = ["DA", "ACh", "NE", "NE_emb", "5HT", "all4", "free",
             "DA_step", "ACh_vol", "NE_rise", "5ht-const"]

    def sortkey(tag):
        n, gran, arm, opt = tag.split("|")
        return (gran, order.index(n) if n in order else 99,
                ["nobuf", "buf-own", "er-own"].index(arm), opt)

    last = None
    print(hdr); print("  " + "-" * len(hdr))
    for tag in sorted(cells, key=sortkey):
        n, gran, arm, opt = tag.split("|")
        c = cells[tag]
        ref = b.get(("er|" if arm == "er-own" else "naive|") + opt, float("nan"))
        d = c["pred"] - ref
        if gran != last:
            print(f"  == {gran} ==")
            last = gran
        star = " *" if d > 0.02 else ""
        print(f"  {n:10s} {gran:7s} {arm:7s} {opt:4s} | {c['pred']:7.4f} {d:+7.4f} "
              f"{c['true']:7.4f} {c['probe']:6.3f} | {c['h0']:.3f}/{c['h1']:.3f}/{c['out']:.3f}{star}")
    print(f"\n  Δbase = pred - same-opt baseline (er for er-own, naive otherwise). "
          f"* = beats baseline by >0.02.\n  cells={len(cells)}  (probe chance = 0.20)")


if __name__ == "__main__":
    main()
