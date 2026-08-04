"""Run one study script's cells across several processes, one ledger SHARD per worker.

Measured on this machine (M1, 8 cores, 8 GB), a single Split-MNIST cell is SINGLE-CORE bound, not
GPU bound — the nets are small enough that MPS is pure kernel-launch overhead, so one run pins ~0.9
of a core and the GPU idles. Throughput for a 5-task x 5-epoch ER run:

    1 x MPS  52 s/run      2 x MPS  31 s/run     4 x MPS  25 s/run     8 x MPS  24 s/run
    1 x CPU  38 s/run      8 x CPU  16 s/run    12 x CPU  15 s/run     6 CPU + 3 MPS  15 s/run

i.e. ~3.4x at the plateau (memory bandwidth, not cores — RSS is only ~155 MB/process, so 8 GB is
never the limit). Serial MPS is the slowest thing you can do.

WHY THIS IS SAFE. Every study cell calls `seed_all(seed)` on entry and each worker is its own
process, so shards share no RNG and a parallel run reproduces the serial numbers EXACTLY. That is
not true of adding work INSIDE a loop — see CLAUDE.md on DataLoader iterators consuming global torch
RNG. Parallelism across processes is the one form of speedup this project's reproducibility contract
does not charge for.

TWO THINGS IT WILL NOT DO FOR YOU:

  DEVICE CHANGES NUMERICS. The same ER run gives 0.7337 on MPS and 0.7284 on CPU. Every frozen
  ledger in `results/` was produced on MPS, so anchor cells MUST run with `--device mps`; only a new
  ledger is free to choose. NEVER mix devices within one ledger — the runner records the device in a
  `<ledger>.runlog` sidecar so a mixed ledger is at least detectable after the fact.

  ONE LEDGER FILE PER WORKER, MERGED AT THE END. `Ledger.append` opens with "a", and a short line
  under O_APPEND is very probably atomic — but "very probably" is not a property an append-only
  archive should rest on, and `is_done()` re-reads the whole file, so concurrent workers would not
  see each other's rows anyway. Shards sidestep both. They are merged only after every worker exits
  0; a failed shard leaves its rows in the shard file for `--resume` rather than half-writing them
  into the master.

CONTRACT FOR A STUDY SCRIPT: read its ledger path through `ledger_path()` instead of hardcoding it.

    TSV = shard.ledger_path(Path(__file__).resolve().parent / "my_results.tsv")

That is the whole integration — no CLI flag, and the script keeps working unchanged when run
directly. Work is split with the filter flags the script ALREADY has (`--part`, `--drivers`, ...),
so the runner never needs to know what a cell is and survives a cell list changing under it.

Run:
    uv run python -m neurocore.shard --script pt5_taskil/plast_drivers.py \
        --ledger pt5_taskil/plast_drivers_results.tsv \
        --split drivers=ach,ach_ema,nerisez,vec_x,vecproj \
        --args "--part test --resume" --workers 6 --device cpu

    # cartesian product of two axes -> 15 shards over 6 workers
        --split drivers=ach,vec_x,vecproj --split part=tune,test,report
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Given to each worker so a CPU shard does not spawn 8 BLAS threads and fight the other shards.
CPU_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}

# `python -c` bootstrap: forces CPU by hiding MPS *before* the study imports torch, without editing
# the frozen `results/pt7_*` modules it pulls in (rule #9). sys.path[0] is cwd, which the runner
# pins to ROOT, so `import prototype` / `import pt7_*` resolve as they do for a direct run.
_BOOTSTRAP = """\
import sys, runpy, torch
if {force_cpu!r}:
    torch.backends.mps.is_available = lambda: False
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""


# ------------------------- the study-script side -------------------------
def ledger_path(default):
    """The ledger this process should write: the shard the runner assigned, else `default`.

    A study script calls this once at module level. Run directly it returns `default` unchanged, so
    nothing about a normal invocation changes.
    """
    return Path(os.environ.get("SHARD_LEDGER", str(default)))


def shard_tag():
    """This worker's shard name, or "" when running directly. For log lines only."""
    return os.environ.get("SHARD_TAG", "")


# ------------------------- planning -------------------------
def parse_split(spec):
    """"drivers=ach,vec_x" -> ("drivers", ["ach", "vec_x"])."""
    if "=" not in spec:
        raise ValueError(f"--split wants KEY=v1,v2,... , got {spec!r}")
    key, vals = spec.split("=", 1)
    key = key.strip().lstrip("-")
    values = [v.strip() for v in vals.split(",") if v.strip()]
    if not key or not values:
        raise ValueError(f"--split wants a non-empty key and values, got {spec!r}")
    return key, values


def plan(splits, base_args=()):
    """[(shard_name, argv_list)] — one shard per point of the cartesian product of the split axes.

    With no split axes this is a single shard, which is a legitimate way to run a study unchanged
    while still getting the log file and runlog provenance.
    """
    axes = [parse_split(s) for s in splits]
    base = list(base_args)
    if not axes:
        return [("all", base)]
    out = []
    for combo in product(*[[(k, v) for v in vals] for k, vals in axes]):
        name = "_".join(f"{k}-{v}" for k, v in combo)
        args = list(base)
        for k, v in combo:
            args += [f"--{k}", v]
        out.append((_safe(name), args))
    return out


def _safe(name):
    """Shard names become filenames, so keep them to a boring alphabet."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


# ------------------------- merging -------------------------
def merge(master, shard_files, keys=None):
    """Fold every shard's rows into `master`. Returns (rows_written, rows_dropped_as_duplicate).

    Shards are disjoint by construction, so duplicates only appear when a cell was re-run. With
    `keys` the LAST row for a key wins (a re-run supersedes); without it only byte-identical
    duplicate lines collapse, which is the conservative default for an append-only archive.
    """
    master = Path(master)
    header, rows = None, []
    if master.exists():
        lines = [ln for ln in master.read_text().splitlines() if ln.strip()]
        if lines:
            header, rows = lines[0], lines[1:]
    for f in shard_files:
        f = Path(f)
        if not f.exists():
            continue
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        if not lines:
            continue
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            raise ValueError(
                f"ledger schema drift: {f.name} header differs from the master ledger.\n"
                f"  master: {header}\n  shard : {lines[0]}\n"
                f"A merged mixed-schema ledger silently breaks --resume; write a new ledger instead.")
        rows += lines[1:]
    if header is None:
        return 0, 0

    cols = header.split("\t")
    if keys:
        idx = [cols.index(k) for k in keys]                  # raises if a key is not a column
        seen, kept = {}, []
        for r in rows:
            f = r.split("\t")
            seen[tuple(f[i] for i in idx)] = r
        for r in rows:                                       # preserve first-appearance order
            f = r.split("\t")
            k = tuple(f[i] for i in idx)
            if k in seen and seen[k] is not None:
                kept.append(seen[k])
                seen[k] = None
    else:
        seen, kept = set(), []
        for r in rows:
            if r not in seen:
                seen.add(r)
                kept.append(r)
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_text(header + "\n" + "".join(r + "\n" for r in kept))
    return len(kept), len(rows) - len(kept)


# ------------------------- running -------------------------
def _launch(script, args, shard_ledger, tag, log_path, device):
    env = dict(os.environ)
    env["SHARD_LEDGER"] = str(shard_ledger)
    env["SHARD_TAG"] = tag
    if device == "cpu":
        env.update(CPU_ENV)
    cmd = [sys.executable, "-c", _BOOTSTRAP.format(force_cpu=(device == "cpu")),
           str(script), *args]
    log = open(log_path, "w")
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, log


def run(script, ledger, splits=(), base_args=(), workers=6, device="cpu", logdir=None,
        keys=None, clean=False, dry_run=False):
    """Fan `script` out over `workers` processes, then merge the shard ledgers into `ledger`."""
    script, ledger = Path(script), Path(ledger)
    if not script.exists():
        raise FileNotFoundError(script)
    specs = plan(splits, base_args)
    logdir = Path(logdir) if logdir else ledger.parent / f"{ledger.stem}_shards"
    shard_files = [logdir / f"{name}.tsv" for name, _ in specs]

    print(f"script  {script}")
    print(f"ledger  {ledger}")
    print(f"device  {device}   workers {workers}   shards {len(specs)}")
    print(f"shards  {logdir}/")
    for (name, args), sf in zip(specs, shard_files):
        print(f"  {name:<28s} {' '.join(args)}")
    if dry_run:
        print("\n(dry run — nothing launched)")
        return []

    logdir.mkdir(parents=True, exist_ok=True)
    # Seed each shard file from the master so a study's own --resume sees everything already done.
    if ledger.exists():
        for sf in shard_files:
            if not sf.exists():
                sf.write_text(ledger.read_text())

    queue = list(zip(specs, shard_files))
    running, done, t0 = [], [], time.time()
    print(f"\nstarted {datetime.now():%H:%M:%S}\n", flush=True)
    try:
        while queue or running:
            while queue and len(running) < workers:
                (name, args), sf = queue.pop(0)
                log_path = logdir / f"{name}.log"
                p, log = _launch(script, args, sf, name, log_path, device)
                running.append((name, p, log, time.time()))
                print(f"  [{time.time() - t0:6.0f}s] start  {name}  -> {log_path.name}", flush=True)
            time.sleep(2)
            for entry in list(running):
                name, p, log, ts = entry
                if p.poll() is None:
                    continue
                running.remove(entry)
                log.close()
                done.append((name, p.returncode))
                mark = "ok  " if p.returncode == 0 else f"FAIL({p.returncode})"
                print(f"  [{time.time() - t0:6.0f}s] {mark}  {name}  "
                      f"({time.time() - ts:.0f}s, {len(queue)} queued, {len(running)} running)",
                      flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — terminating workers (shard ledgers keep their finished rows)")
        for _, p, log, _ in running:
            p.terminate()
            log.close()
        raise

    wall = time.time() - t0
    failed = [n for n, rc in done if rc != 0]
    print(f"\nall shards finished in {wall / 60:.1f} min")
    if failed:
        print(f"!! {len(failed)} shard(s) FAILED: {', '.join(failed)}")
        print(f"   master ledger NOT merged. Read {logdir}/<shard>.log, then re-run with --resume;")
        print(f"   finished rows are already in {logdir}/<shard>.tsv and will be skipped.")
        return done

    kept, dropped = merge(ledger, shard_files, keys=keys)
    print(f"merged -> {ledger}  ({kept} rows, {dropped} duplicate(s) collapsed)")
    _runlog(ledger, script, device, workers, specs, wall)
    if clean:
        for sf in shard_files:
            sf.unlink(missing_ok=True)
        print(f"removed {len(shard_files)} shard ledger(s)")
    return done


def _runlog(ledger, script, device, workers, specs, wall):
    """Provenance sidecar. The device a row was produced on is not a ledger column, and a ledger
    holding both CPU and MPS rows compares numbers that were never comparable — this is the record
    that makes that mistake findable afterwards."""
    line = (f"{datetime.now():%Y-%m-%d %H:%M:%S}\tscript={script}\tdevice={device}\t"
            f"workers={workers}\tshards={len(specs)}\twall_min={wall / 60:.1f}\t"
            f"names={','.join(n for n, _ in specs)}\n")
    with open(Path(str(ledger) + ".runlog"), "a") as f:
        f.write(line)


def main():
    ap = argparse.ArgumentParser(description="Shard one study script across processes.")
    ap.add_argument("--script", required=True, help="study script, e.g. pt5_taskil/plast_drivers.py")
    ap.add_argument("--ledger", required=True, help="master ledger the shards merge into")
    ap.add_argument("--split", action="append", default=[], metavar="KEY=v1,v2",
                    help="split axis, passed to the script as --KEY v; repeat for a product")
    ap.add_argument("--args", default="", help="args every shard gets, e.g. \"--part test --resume\"")
    ap.add_argument("--workers", type=int, default=6, help="max concurrent processes (default 6)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                    help="cpu is ~3.4x faster in aggregate; mps is REQUIRED for anchor cells")
    ap.add_argument("--keys", default=None,
                    help="comma-separated ledger key columns; last row per key wins on merge")
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--clean", action="store_true", help="delete shard ledgers after a clean merge")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.script, a.ledger, splits=a.split, base_args=a.args.split(), workers=a.workers,
        device=a.device, logdir=a.logdir, keys=a.keys.split(",") if a.keys else None,
        clean=a.clean, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
