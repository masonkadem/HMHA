"""Where does qsa_cross have a cliff? (thesis proposal, experiment E2a)

Dense models with k heads trained from scratch, every k given the same 6000 steps, at
several head widths d_k. Finds the measured k* to test the loop against.

  python experiments/qsa_dense.py   ->  results/proposal/qsa_dense.pkl  (about 7 min on a GPU)
"""
import os, sys, time, pickle, math
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg
from hemo.tasks import make_val, trivial_loss
from hemo.train import train, pick_device

OUT = os.path.join(ROOT, "results", "proposal", "qsa_dense.pkl")
res = {}
t0 = time.time()
for dk in [2, 4, 8, 16]:
    cfg = Cfg(task="qsa_cross", d_k=dk, seed=0)
    dev = pick_device(cfg)
    val = make_val(cfg, dev)
    triv = trivial_loss(val)
    row = {}
    for k in [1, 2, 3, 4, 6, 8, 16, 32]:
        _, h = train(cfg, val, dev, hemo=False, num_heads=k, steps=6000)
        row[k] = h["val_loss"][-1]
        print(f"dk {dk:>2}  k {k:>2}  loss {row[k]:.5f}  loss/trivial {row[k] / triv:.4f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    res[dk] = dict(trivial=triv, loss=row)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(res, f)
print("formula 8/dk:", {dk: math.ceil(8 / dk) for dk in res},
      "  rank 16/dk:", {dk: math.ceil(16 / dk) for dk in res})
