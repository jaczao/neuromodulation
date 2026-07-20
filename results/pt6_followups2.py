"""pt6 follow-ups (2) — corrected D + the per-layer gate study.

D2 (corrected). Earlier "D" changed the inference net's TARGET (pseudo-labels). What was actually
    wanted: the DRIVER AT TRAINING should be the inference net's SOFT posterior — gate = sum_t p_t P[t]
    — the same driver used at eval, instead of the true task id P[t_true]. That removes the
    train/eval mismatch (train on the oracle row, eval on a blend). p is detached, so the main loss
    trains only P and g stays trained by its own replay task-CE.

G. Per-layer gate study. mean|P| averages 4050 entries but the OUT gate is only 50 of them (1.2%),
   and that is the part that touches class-IL logit competition — so a small mean|P| does NOT prove
   "the gate is off". Report mean|P| separately for h0 / h1 / out on the standard cells and across
   the parity sweep, to see WHERE the modulation lives and what the parity penalty actually kills.

Refs: er-adam 0.8946, er-sgd 0.7234, naive-sgd 0.6287; standard soft_mlp er-own/adam soft 0.8858,
er-own/sgd 0.8556, buf-own/sgd 0.8563. 1 seed, seed42, lr1e-3, ep5, buffer1000, class-IL.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pt6_driver_mechanisms import DEV  # noqa: E402
from pt6_followups import run_softmlp  # noqa: E402


def line(tag, r):
    print(f"  {tag:32s} oracle={r['oracle']:.4f}  soft={r['soft']:.4f}  hard={r['hard']:.4f}  "
          f"infer={r['infer']:.4f} | |P| h0={r['P_h0']:.3f} h1={r['P_h1']:.3f} out={r['P_out']:.3f}",
          flush=True)


def main():
    print(f"device={DEV}   (ref: er-adam 0.8946, er-sgd 0.7234, naive-sgd 0.6287)\n")

    print("D2. DRIVER AT TRAIN = true task id  vs  inference net's SOFT posterior")
    for arm, opt in (("er-own", "sgd"), ("er-own", "adam"), ("buf-own", "sgd")):
        line(f"{arm} {opt} driver=true", run_softmlp(arm, opt, train_driver="true"))
        line(f"{arm} {opt} driver=SOFT", run_softmlp(arm, opt, train_driver="soft"))

    print("\nG. per-layer gate magnitude across the PARITY sweep (soft_mlp, er-own/adam)")
    for lam in (0.0, 0.1, 1.0, 10.0):
        line(f"lam_parity={lam}", run_softmlp("er-own", "adam", lam_parity=lam))


if __name__ == "__main__":
    main()
