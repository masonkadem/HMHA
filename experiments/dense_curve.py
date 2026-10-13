"""Reference loss curves for the compute comparison: a dense model with all H heads, and
one with exactly k* heads, which is what you would train if you already knew the answer.

  python experiments/dense_curve.py --n_rel 4 --heads 32 --seed 0
"""
import argparse, os, pickle, sys
from dataclasses import replace, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hemo.config import Cfg
from hemo.tasks import make_val, trivial_loss
from hemo.train import train, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--n_rel", type=int, default=4)
ap.add_argument("--heads", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "dense"))
a = ap.parse_args()

cfg = replace(Cfg(), n_rel=a.n_rel, seed=a.seed, device=a.device)
device = pick_device(cfg)
val = make_val(cfg, device)
_, hist = train(cfg, val, device, hemo=False, num_heads=a.heads)
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, f"dense_R{a.n_rel}_H{a.heads}_s{a.seed}.pkl")
with open(path, "wb") as f:
    pickle.dump({"cfg": asdict(cfg), "heads": a.heads, "trivial": trivial_loss(val),
                 "hist": {k: hist[k] for k in ("val_step", "val_loss")}}, f)
print(f"wrote {path}  final loss {hist['val_loss'][-1]:.6f}")
