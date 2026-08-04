"""Tests for the parallel shard runner (neurocore/shard.py).

Fast: the planning and merge logic is pure, and the one end-to-end test drives a throwaway study
script that writes ledger rows instead of training. The property that actually matters — that
sharded numbers equal serial numbers — is guaranteed by per-cell `seed_all` plus process isolation,
not by anything this file can assert cheaply.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neurocore import shard as S

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ planning
def test_parse_split_accepts_bare_and_dashed_keys():
    assert S.parse_split("drivers=ach,vec_x") == ("drivers", ["ach", "vec_x"])
    assert S.parse_split("--part=tune,test") == ("part", ["tune", "test"])


@pytest.mark.parametrize("bad", ["drivers", "=a,b", "drivers=", "drivers=,"])
def test_parse_split_rejects_malformed(bad):
    with pytest.raises(ValueError):
        S.parse_split(bad)


def test_plan_with_no_axes_is_one_shard():
    assert S.plan([], ["--part", "test"]) == [("all", ["--part", "test"])]


def test_plan_is_the_cartesian_product_and_passes_flags_through():
    got = S.plan(["drivers=ach,vec_x", "part=tune,test"], ["--resume"])
    assert len(got) == 4
    names = [n for n, _ in got]
    assert names == ["drivers-ach_part-tune", "drivers-ach_part-test",
                     "drivers-vec_x_part-tune", "drivers-vec_x_part-test"]
    assert got[0][1] == ["--resume", "--drivers", "ach", "--part", "tune"]


def test_shard_names_are_filename_safe():
    (name, _), = S.plan(["gain=1+m/2"])
    assert "/" not in name and name == "gain-1-m-2"


# ------------------------------------------------------------------ ledger_path contract
def test_ledger_path_defaults_when_unsharded(monkeypatch, tmp_path):
    monkeypatch.delenv("SHARD_LEDGER", raising=False)
    default = tmp_path / "study_results.tsv"
    assert S.ledger_path(default) == default


def test_ledger_path_follows_the_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("SHARD_LEDGER", str(tmp_path / "shard-3.tsv"))
    assert S.ledger_path(tmp_path / "study_results.tsv") == tmp_path / "shard-3.tsv"


# ------------------------------------------------------------------ merging
HEADER = "stage\tdriver\tseed\tacc"


def _write(p, rows, header=HEADER):
    p.write_text(header + "\n" + "".join(r + "\n" for r in rows))


def test_merge_concatenates_shards_into_a_new_master(tmp_path):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    _write(a, ["test\tach\t42\t0.9"])
    _write(b, ["test\tvec_x\t42\t0.8"])
    master = tmp_path / "m.tsv"
    kept, dropped = S.merge(master, [a, b])
    assert (kept, dropped) == (2, 0)
    assert master.read_text().splitlines() == [HEADER, "test\tach\t42\t0.9", "test\tvec_x\t42\t0.8"]


def test_merge_preserves_existing_master_rows(tmp_path):
    master, a = tmp_path / "m.tsv", tmp_path / "a.tsv"
    _write(master, ["test\told\t42\t0.5"])
    _write(a, ["test\tnew\t42\t0.6"])
    S.merge(master, [a])
    assert [ln.split("\t")[1] for ln in master.read_text().splitlines()[1:]] == ["old", "new"]


def test_merge_collapses_identical_lines_without_keys(tmp_path):
    """Shards are seeded from the master so --resume works; that duplicates its rows harmlessly."""
    master, a = tmp_path / "m.tsv", tmp_path / "a.tsv"
    _write(master, ["test\tach\t42\t0.9"])
    _write(a, ["test\tach\t42\t0.9", "test\tvec_x\t42\t0.8"])
    kept, dropped = S.merge(master, [a])
    assert (kept, dropped) == (2, 1)


def test_merge_with_keys_lets_a_rerun_supersede(tmp_path):
    master, a = tmp_path / "m.tsv", tmp_path / "a.tsv"
    _write(master, ["test\tach\t42\t0.9"])
    _write(a, ["test\tach\t42\t0.7"])                     # same key, re-run value
    kept, _ = S.merge(master, [a], keys=["stage", "driver", "seed"])
    assert kept == 1
    assert master.read_text().splitlines()[1].endswith("0.7")


def test_merge_refuses_a_schema_mismatch(tmp_path):
    """A merged mixed-schema ledger breaks --resume silently, so this must be loud."""
    master, a = tmp_path / "m.tsv", tmp_path / "a.tsv"
    _write(master, ["test\tach\t42\t0.9"])
    _write(a, ["test\tach\t42\t0.9\t0.1"], header=HEADER + "\tforget")
    with pytest.raises(ValueError, match="schema drift"):
        S.merge(master, [a])


def test_merge_of_nothing_is_a_noop(tmp_path):
    assert S.merge(tmp_path / "m.tsv", [tmp_path / "missing.tsv"]) == (0, 0)
    assert not (tmp_path / "m.tsv").exists()


# ------------------------------------------------------------------ end to end
FAKE_STUDY = '''\
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, r"{root}")
from neurocore import shard

TSV = shard.ledger_path(Path(__file__).resolve().parent / "fake_results.tsv")
COLS = ["stage", "driver", "seed", "acc"]

ap = argparse.ArgumentParser()
ap.add_argument("--drivers", default="ach")
ap.add_argument("--part", default="test")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

if not TSV.exists():
    TSV.write_text("\\t".join(COLS) + "\\n")
done = {{tuple(l.split("\\t")[:3]) for l in TSV.read_text().splitlines()[1:] if l.strip()}}
for seed in (42, 43):
    key = (a.part, a.drivers, str(seed))
    if a.resume and key in done:
        continue
    with TSV.open("a") as f:
        f.write("\\t".join(key) + "\\t0.5\\n")
print("shard", os.environ.get("SHARD_TAG"), "wrote", a.drivers, a.part)
'''


def test_end_to_end_shards_merge_and_log(tmp_path):
    script = tmp_path / "fake_study.py"
    script.write_text(FAKE_STUDY.format(root=ROOT))
    master = tmp_path / "fake_results.tsv"

    done = S.run(script, master, splits=["drivers=ach,vec_x,vecproj"], base_args=["--part", "test"],
                 workers=3, device="cpu", logdir=tmp_path / "shards")

    assert [rc for _, rc in done] == [0, 0, 0]
    rows = [l.split("\t") for l in master.read_text().splitlines()[1:]]
    assert len(rows) == 6                                  # 3 drivers x 2 seeds, nothing lost
    assert sorted({r[1] for r in rows}) == ["ach", "vec_x", "vecproj"]
    assert (tmp_path / "shards" / "drivers-ach.log").exists()

    runlog = Path(str(master) + ".runlog").read_text()
    assert "device=cpu" in runlog and "shards=3" in runlog


def test_end_to_end_resume_is_idempotent(tmp_path):
    """Re-running a finished study must add nothing — the shards are seeded from the master."""
    script = tmp_path / "fake_study.py"
    script.write_text(FAKE_STUDY.format(root=ROOT))
    master = tmp_path / "fake_results.tsv"
    kw = dict(splits=["drivers=ach,vec_x"], base_args=["--part", "test", "--resume"],
              workers=2, device="cpu", logdir=tmp_path / "shards")
    S.run(script, master, **kw)
    first = master.read_text()
    S.run(script, master, **kw)
    assert master.read_text() == first


def test_failed_shard_blocks_the_merge(tmp_path):
    """A half-finished run must not write into the archive; its rows wait in the shard file."""
    script = tmp_path / "boom.py"
    script.write_text("import sys; sys.exit(3)")
    master = tmp_path / "fake_results.tsv"
    done = S.run(script, master, workers=1, device="cpu", logdir=tmp_path / "shards")
    assert [rc for _, rc in done] == [3]
    assert not master.exists()


def test_cli_dry_run_launches_nothing(tmp_path):
    script = tmp_path / "fake_study.py"
    script.write_text(FAKE_STUDY.format(root=ROOT))
    master = tmp_path / "fake_results.tsv"
    out = subprocess.run(
        [sys.executable, "-m", "neurocore.shard", "--script", str(script), "--ledger", str(master),
         "--split", "drivers=ach,vec_x", "--args", "--part test", "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert out.returncode == 0, out.stderr
    assert "dry run" in out.stdout and "shards 2" in out.stdout
    assert not master.exists()
