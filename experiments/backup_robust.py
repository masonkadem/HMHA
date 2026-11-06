"""Do the backup heads pay off? Train three arms on k* = 4 (upkeep 0.03 L_triv), then knock heads out
at test time and measure the loss.

  aware   heads fail with p = 0.2 in training, and the rule values heads under failure (keeps backups)
  blind   heads fail with p = 0.2 in training, but the rule values heads as if nothing failed
          (damage_blind = 1). Same training damage, no backups: separates the effect of the backups
          from the robustness that training under failure gives by itself.
  lean    no failures in training (keeps exactly k* heads)

Test damage: each open head fails independently with p_test, 400 random patterns per level. Two
losses per pattern, both in units of L_triv:
  fixed   the model as trained, failed heads switched off
  refit   failed heads switched off and the read-out re-fitted to the survivors (collateral flow);
          this is the loss the theory is about, L*(working heads)

  python experiments/backup_robust.py --seeds 0-9 --workers 3   ->  results/backup/robust_<arm>_s<seed>.pkl
"""
import argparse, os, pickle, sys, time
from dataclasses import replace
from multiprocessing import Pool

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "results", "backup")
ARMS = {"aware": dict(head_dropout=0.2), "blind": dict(head_dropout=0.2, damage_blind=1), "lean": {}}
P_TEST = (0.0, 0.1, 0.2, 0.3)
N_PATTERNS = 400


def run(job):
    arm, seed = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from hemo.config import Cfg
    from hemo.tasks import make_val, trivial_loss
    from hemo.train import train, evaluate
    path = os.path.join(OUT, f"robust_{arm}_s{seed}.pkl")
    if os.path.exists(path):
        return path
    t0 = time.time()
    cfg = replace(Cfg(), n_rel=4, seed=seed, supply="local", price_frac=0.03, taper=100,
                  budget_hold_frac=0.1, conserve=0, device="cpu", **ARMS[arm])
    if os.environ.get("SMOKE"):                       # quick check of the pipeline, not a result
        cfg = replace(cfg, steps=200, probe_masks=4)
        path = path.replace(".pkl", "_smoke.pkl")
    dev = torch.device("cpu")
    val = make_val(cfg, dev)
    model, hist = train(cfg, val, dev, hemo=True)
    model.eval()
    triv = trivial_loss(val)
    gate = model.gate().detach()
    open_ = torch.nonzero(gate > 0).flatten()
    X, Y, T = val[0], val[1], val[2]
    with torch.no_grad():
        out = model.heads(X, Y)[2]                                     # (b, H, n, dk)
    H, dk = out.size(1), out.size(3)
    Z = out.permute(0, 2, 1, 3).reshape(-1, H * dk).double()
    t = T.reshape(-1, T.size(-1)).double()
    Z, t = Z - Z.mean(0), t - t.mean(0)

    def refit_loss(working):
        if len(working) == 0:
            return float((t ** 2).mean()) / triv
        cols = torch.cat([torch.arange(h * dk, (h + 1) * dk) for h in working.tolist()])
        Zs = Z[:, cols]
        A = Zs.T @ Zs
        A += 1e-6 * torch.diagonal(A).mean() * torch.eye(len(cols), dtype=A.dtype)
        W = torch.linalg.solve(A, Zs.T @ t)
        return float(((t - Zs @ W) ** 2).mean()) / triv

    rng = np.random.default_rng(10_000 + seed)
    res = {}
    for p in P_TEST:
        fixed, refit = [], []
        for _ in range((20 if os.environ.get("SMOKE") else N_PATTERNS) if p > 0 else 1):
            alive = torch.tensor(rng.random(len(open_)) >= p)
            working = open_[alive]
            g = torch.zeros_like(gate)
            g[working] = gate[working]
            fixed.append(evaluate(model, val, gate=g) / triv)
            refit.append(refit_loss(working))
        res[p] = dict(fixed=np.array(fixed), refit=np.array(refit))
    rec = dict(arm=arm, seed=seed, kept=int(len(open_)), open=open_.tolist(), damage=res,
               final_loss=evaluate(model, val, gate=gate) / triv, hist_ledger=hist["ledger"],
               seconds=time.time() - t0)
    pickle.dump(rec, open(path, "wb"))
    print(f"{arm} s{seed}: kept {len(open_)}, loss under p=0.2 fixed "
          f"{res[0.2]['fixed'].mean():.3f} refit {res[0.2]['refit'].mean():.3f}  ({time.time() - t0:.0f}s)",
          flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    os.makedirs(OUT, exist_ok=True)
    jobs = [(arm, s) for s in range(lo, hi + 1) for arm in ARMS]
    with Pool(a.workers) as pool:
        for _ in pool.imap_unordered(run, jobs):
            pass
    print("all done")
