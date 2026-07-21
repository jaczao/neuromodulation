"""Table for results/pt7_variants_results.tsv. Groups by kind; for head/ne shows std-ON vs std-OFF
side by side and Δ vs the same-opt pt7 baseline (er for er-own, naive for nobuf). Standard shows all4 vs
vanilla. Baselines read from pt7_results.tsv. Run: uv run python results/pt7_variants_make_table.py
"""
from pathlib import Path

D = Path(__file__).resolve().parent
TSV = D / "pt7_variants_results.tsv"
BASE_TSV = D / "pt7_results.tsv"


def load(tsv):
    rows = {}
    for ln in tsv.read_text().splitlines():
        if not ln.strip():
            continue
        f = ln.split("\t")
        rows[f[0]] = f[1:]
    return rows


def baselines():
    b = {}
    for tag, v in load(BASE_TSV).items():
        if len(v) == 1:                                 # baseline row: name|-|-|opt
            n, _, _, opt = tag.split("|")
            b[f"{n}|{opt}"] = float(v[0])
    return b


def main():
    b = baselines()
    rows = load(TSV)
    print(f"pt7 baselines: naive {b.get('naive|sgd',0):.4f}/{b.get('naive|adam',0):.4f}  "
          f"er {b.get('er|sgd',0):.4f}/{b.get('er|adam',0):.4f}  (sgd/adam)\n")

    # ---- standard ----
    print("== A. STANDARD regime (full MNIST, test acc) ==")
    for opt in ("sgd", "adam"):
        a = rows.get(f"standard|all4|-|{opt}|std1")
        van = rows.get(f"standard|vanilla|-|{opt}|std1")
        if a and van:
            print(f"  {opt:4s}: vanilla {float(van[0]):.4f}   all4-gate {float(a[0]):.4f}   "
                  f"Δ {float(a[0])-float(van[0]):+.4f}")
    print()

    # ---- head + ne: std ON vs OFF ----
    def ref(arm, opt):
        return b.get(("er|" if arm == "er-own" else "naive|") + opt, float("nan"))

    for kind, names in (("head", ["DA_fast", "ACh_ema", "ACh_vol_ps", "5HT_ema", "ACh", "NE_emb", "DA_step"]),
                        ("ne", ["emb_all", "vec_h1", "vec_h1proj", "vec_x", "vecproj"])):
        print(f"== {'B/D. NEW+OLD HEAD drivers' if kind=='head' else 'C. NE double-forward / multidim'} "
              f"(pred; Δ vs same-opt baseline) ==")
        print(f"  {'name':11s} {'arm':7s} {'opt':4s} | {'std=ON':>16s} | {'std=OFF':>16s}")
        for n in names:
            for arm in ("er-own", "nobuf", "buf-own"):
                for opt in ("sgd", "adam"):
                    on = rows.get(f"{kind}|{n}|{arm}|{opt}|std1")
                    off = rows.get(f"{kind}|{n}|{arm}|{opt}|std0")
                    r = ref(arm, opt)

                    def cell(x):
                        if not x:
                            return f"{'--':>16s}"
                        return f"{float(x[0]):.4f} ({float(x[0])-r:+.3f})"
                    print(f"  {n:11s} {arm:7s} {opt:4s} | {cell(on):>16s} | {cell(off):>16s}")
        print()


if __name__ == "__main__":
    main()
