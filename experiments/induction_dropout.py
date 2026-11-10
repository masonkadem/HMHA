"""Pilot: do backup heads in the induction circuit grow with the head failure rate?
Predictions written first: results/induction_dropout/predictions.md

  python experiments/induction_dropout.py --p 0 0.1 0.2 0.3 --seeds 0 1 --workers 4
  ->  results/induction_dropout/ind_p{p}_s{seed}.pkl
"""
import argparse, os, pickle, sys, time
from dataclasses import replace
from multiprocessing import Pool

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "results", "induction_dropout")


def run(job):
    p, seed = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from hemo.induction import ICfg, train
    path = os.path.join(OUT, f"ind_p{p:g}_s{seed}.pkl")
    if os.path.exists(path):
        return path
    cfg = replace(ICfg(), rule="collateral", probe="logit", price_mode="bar", price_frac=0.05, steps=6000,
                  head_dropout=p, seed=seed, device="cuda")
    t0 = time.time()
    _, out = train(cfg)
    kept = np.asarray(out["kept"])
    out["seconds"] = time.time() - t0
    pickle.dump(out, open(path, "wb"))
    print(f"p={p:g} seed={seed}: kept {int(kept[:8].sum())} + {int(kept[8:].sum())}, "
          f"final loss {out['final_loss']:.3f} (bar {out['bar']:.3f}), acc {out['final_acc']:.3f}, "
          f"{out['seconds']:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(p, s) for s in a.seeds for p in a.p]
    with Pool(a.workers) as pool:
        for _ in pool.imap_unordered(run, jobs):
            pass
    print("all done")
