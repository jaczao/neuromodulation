"""pt6 follow-ups — six ISOLATED probes (each its own small table; no full cross-product).

A. mean_image/mlp/centered: finer soft-nearest(tau) sweep (is 0.03 really the peak?).
B. soft_mlp STANDALONE variants: no-buffer, and buf-"cur" (meta gates every sample with the CURRENT
   task's row instead of its own) -- the wrong-task ablation. sgd + adam.
C. hard-mlp: gate = P[argmax g(x)] instead of the soft blend sum_t p_t P[t]. Same trainings as the
   standard soft_mlp cells, so soft-vs-hard is matched.
D. soft_mlp whose inference net is trained on the MAIN NET'S OUT LAYER (pseudo-label
   task = argmax(logits)//2, detached) instead of the true task id.
E. small SPARSITY sweep on soft_mlp: lam * mean|gamma| = lam * mean|1+P|  (pushes gates toward 0/off).
F. small PARITY sweep on soft_mlp:  lam * mean|P|  (pushes gates toward parity gamma=1 = NO modulation;
   the tension the user flagged -- too much of it means no modulation at all).

Reference (from pt6_driver_mechanisms.log): er-adam 0.8946, er-sgd 0.7234, naive-sgd 0.6287;
soft_mlp standard soft-blend er-own/adam 0.8850 (infer 0.8843), er-own/sgd 0.8556, buf-own/sgd 0.8562.
1 seed, seed 42, lr 1e-3, ep 5, buffer 1000, class-IL, gain-neuron on (h0,h1,out).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pt6_driver_mechanisms import (  # noqa: E402
    CE, DEV, Net, Reservoir, SoftMLP, _opt, build, forward_gain, masked_ce, seed_all,
    train_bufown, train_erown,
)

FINE_TAUS = (0.003, 0.01, 0.03, 0.05, 0.1, 0.3)


# --------------------------------------------------------------------------- A
@torch.no_grad()
def soft_nearest_sweep(mech, net, center, loaders, taus=FINE_TAUS):
    G = mech.raw_table()
    cor = {t: 0 for t in taus}; near = tot = 0
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0); d = x.view(b, -1) - center
            dist = (d[:, None, :] - mech.mus[None]).pow(2).mean(-1)          # (B,5)
            for tau in taus:
                raw = F.softmax(-dist / tau, dim=1) @ G
                cor[tau] += (forward_gain(net, raw, x).argmax(1) == y).sum().item()
            near += (forward_gain(net, G[dist.argmin(1)], x).argmax(1) == y).sum().item()
            tot += b
    return {f"@{t}": cor[t] / tot for t in taus} | {"hard-nearest": near / tot}


# --------------------------------------------------------------------------- B/C/D/E/F
def train_softmlp(mech, net, loaders, arm, opt_kind, use_buffer=True, gate_mode="own",
                  inf_target="true", train_driver="true", lam_sparse=0.0, lam_parity=0.0,
                  lr=1e-3, epochs=5, buffer=1000):
    """Flexible soft_mlp trainer covering every follow-up variant.
    arm: 'buf-own' (naive main + per-task replay meta-loss on P) | 'er-own' (joint on the ER batch).
    gate_mode: 'own' (each sample gated by its own task row) | 'cur' (all gated by the current task).
    inf_target: 'true' (true task id) | 'out' (pseudo-label from the main net's out layer).
    train_driver: 'true' (gate = P[true task]) | 'soft' (gate = sum_t p_t P[t] from the INFERENCE NET,
        i.e. the SAME soft driver used at eval -> removes the train/eval mismatch). p is detached so
        the main loss trains only P; g stays trained by its own task-CE.
    lam_sparse: lam*mean|1+P| ; lam_parity: lam*mean|P|  (added to whichever loss trains P).
    """
    buf = Reservoir(buffer)
    inf_opt = torch.optim.Adam(mech.inf_params(), lr=lr)

    def gate_for(X, Tg):
        if train_driver == "soft":
            with torch.no_grad():
                p = F.softmax(mech.task_logits(X), dim=1)
            return p @ mech.P                    # differentiable in P, driver p detached
        return mech.train_gate(X, Tg)

    def reg():
        r = 0.0
        if lam_sparse:
            r = r + lam_sparse * (1 + mech.P).abs().mean()
        if lam_parity:
            r = r + lam_parity * mech.P.abs().mean()
        return r

    def inf_step(X, T_true, gate_for_pseudo=None):
        if inf_target == "true":
            tgt = T_true
        else:                                    # pseudo-label from the main net's OUT layer
            with torch.no_grad():
                logits = forward_gain(net, gate_for_pseudo, X)
                tgt = torch.div(logits.argmax(1), 2, rounding_mode="floor")
        ce = CE(mech.task_logits(X), tgt)
        inf_opt.zero_grad(); ce.backward(); inf_opt.step()

    if arm == "er-own":
        opt = _opt(opt_kind, list(net.parameters()) + list(mech.gate_params()), lr)
        for t in range(5):
            for _ in range(epochs):
                for x, y in loaders[t][0]:
                    x, y = x.to(DEV), y.to(DEV)
                    Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [torch.full((x.size(0),), t, device=DEV)]
                    if use_buffer:
                        r = buf.sample_any(64)
                        if r is not None:
                            xr, yr = r[0].to(DEV), r[1].to(DEV)
                            Xs.append(xr); Ys.append(yr)
                            Ts.append(torch.div(yr, 2, rounding_mode="floor"))
                    Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                    Tg = torch.full_like(Tm, t) if gate_mode == "cur" else Tm
                    loss = CE(forward_gain(net, gate_for(Xm, Tg), Xm), Ym) + reg()
                    opt.zero_grad(); inf_opt.zero_grad(); loss.backward(); opt.step()
                    inf_step(Xm, Tm, gate_for(Xm, Tg).detach())
                    buf.add(x, y)
        return

    # ---- buf-own (standalone backbone) ----
    main_opt = _opt(opt_kind, net.parameters(), lr)
    gate_opt = torch.optim.Adam(mech.gate_params(), lr=lr)
    for t in range(5):
        for _ in range(epochs):
            for x, y in loaders[t][0]:
                x, y = x.to(DEV), y.to(DEV); tids = torch.full((x.size(0),), t, device=DEV)
                gate = gate_for(x, tids).detach()
                loss = masked_ce(forward_gain(net, gate, x), y)          # main: naive, gate detached
                net.zero_grad(); mech.zero_grad(); loss.backward(); main_opt.step()
                buf.add(x, y)
                # inference net (replay only if use_buffer)
                Xi, Ti = [x.view(x.size(0), -1)], [tids]
                if use_buffer:
                    r = buf.sample_any(64)
                    if r is not None:
                        Xi.append(r[0].to(DEV)); Ti.append(torch.div(r[1].to(DEV), 2, rounding_mode="floor"))
                Xi_, Ti_ = torch.cat(Xi), torch.cat(Ti)
                inf_step(Xi_, Ti_, gate_for(Xi_, Ti_).detach())
                # meta step (per-task replay if use_buffer, else current batch only)
                Xs, Ys, Ts = [x.view(x.size(0), -1)], [y], [tids]
                if use_buffer:
                    for j in range(t):
                        s = buf.sample_task(j, 64)
                        if s is not None:
                            Xs.append(s[0].to(DEV)); Ys.append(s[1].to(DEV))
                            Ts.append(torch.full((s[0].size(0),), j, device=DEV))
                Xm, Ym, Tm = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
                Tg = torch.full_like(Tm, t) if gate_mode == "cur" else Tm
                meta = masked_ce(forward_gain(net, gate_for(Xm, Tg), Xm), Ym) + reg()
                net.zero_grad(); mech.zero_grad(); meta.backward(); gate_opt.step()


@torch.no_grad()
def eval_softmlp(mech, net, loaders):
    """oracle / soft-blend / hard-argmax / inference-acc, plus gate stats."""
    net.eval(); G = mech.raw_table()
    o = s = h = inf = tot = 0
    for i in range(5):
        for x, y in loaders[i][1]:
            x, y = x.to(DEV), y.to(DEV); b = x.size(0)
            p = F.softmax(mech.task_logits(x), dim=1)
            o += (forward_gain(net, G[i].unsqueeze(0).expand(b, -1), x).argmax(1) == y).sum().item()
            s += (forward_gain(net, p @ G, x).argmax(1) == y).sum().item()
            h += (forward_gain(net, G[p.argmax(1)], x).argmax(1) == y).sum().item()
            inf += (p.argmax(1) == i).sum().item(); tot += b
    P = mech.P.detach()
    return dict(oracle=o / tot, soft=s / tot, hard=h / tot, infer=inf / tot,
                mean_absP=float(P.abs().mean()), P_h0=float(P[:, :400].abs().mean()),
                P_h1=float(P[:, 400:800].abs().mean()), P_out=float(P[:, 800:].abs().mean()))


def run_softmlp(arm, opt_kind, **kw):
    loaders, net, mech, _ = build("soft_mlp", "-", False)
    train_softmlp(mech, net, loaders, arm, opt_kind, **kw)
    return eval_softmlp(mech, net, loaders)


def line(tag, r):
    print(f"  {tag:34s} oracle={r['oracle']:.4f}  soft={r['soft']:.4f}  hard={r['hard']:.4f}  "
          f"infer={r['infer']:.4f}  mean|P|={r['mean_absP']:.3f}", flush=True)


def main():
    print(f"device={DEV}   (ref: er-adam 0.8946, er-sgd 0.7234, naive-sgd 0.6287)\n")

    print("A. mean_image/mlp/centered — finer soft-nearest(tau) sweep")
    for arm, tr in (("er-own", train_erown), ("buf-own", train_bufown)):
        for opt in ("sgd", "adam"):
            if arm == "buf-own" and opt == "adam":
                continue                                     # that cell collapsed (0.09); skip
            loaders, net, mech, cvec = build("mean_image", "mlp", True)
            tr(mech, net, loaders, opt)
            r = soft_nearest_sweep(mech, net, cvec, loaders)
            print(f"  {arm:7s} {opt:4s} | " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)

    print("\nB. soft_mlp STANDALONE variants (buf-own arm)   [ref buf-own/own/sgd soft=0.8562]")
    for opt in ("sgd", "adam"):
        line(f"no-buffer {opt}", run_softmlp("buf-own", opt, use_buffer=False))
        line(f"buf-cur (wrong-task) {opt}", run_softmlp("buf-own", opt, use_buffer=True, gate_mode="cur"))

    print("\nC. hard-mlp vs soft-mlp (standard soft_mlp cells; same training)")
    for arm in ("buf-own", "er-own"):
        for opt in ("sgd", "adam"):
            line(f"{arm} {opt}", run_softmlp(arm, opt))

    print("\nD. soft_mlp inference net trained on the MAIN NET'S OUT LAYER (pseudo-label)")
    for opt in ("sgd", "adam"):
        line(f"er-own {opt} inf<-out", run_softmlp("er-own", opt, inf_target="out"))

    print("\nE. SPARSITY sweep  lam*mean|1+P|   (soft_mlp, er-own/adam)")
    for lam in (0.1, 1.0, 10.0):
        line(f"lam_sparse={lam}", run_softmlp("er-own", "adam", lam_sparse=lam))

    print("\nF. PARITY sweep  lam*mean|P|  (pulls gates toward gamma=1, i.e. no modulation)")
    for lam in (0.1, 1.0, 10.0):
        line(f"lam_parity={lam}", run_softmlp("er-own", "adam", lam_parity=lam))


if __name__ == "__main__":
    main()
