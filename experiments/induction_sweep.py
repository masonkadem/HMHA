"""Summary of the induction price sweep (loss-based collateral values, probe=logit).

  python experiments/induction_sweep.py
"""
import glob, os, pickle
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "induction", "*.pkl"))]


def stayed_solved(r):
    h, start = r["hist"], r["cfg"]["start_frac"] * r["cfg"]["steps"]
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= start]
    first = next((i for i, l in enumerate(ls) if l <= r["bar"]), None)
    return first is not None and max(ls[first:]) <= r["bar"]


def roles_ok(r):
    h1 = r["sizes"][0]
    p = np.asarray(r["roles"]["prev"][0])[r["kept"][:h1]]
    i = np.asarray(r["roles"]["ind"][1])[r["kept"][h1:]]
    return bool(len(p) and p.max() >= 2 * r["chance"]["prev"] and len(i) and i.max() >= 2 * r["chance"]["ind"])


groups = {}
for r in RUNS:
    c = r["cfg"]
    if c["rule"] != "collateral":
        continue
    key = (c.get("probe", "mse"), c.get("price_mode", "absolute"), c["price_frac"], c["steps"])
    groups.setdefault(key, []).append(r)

print(f"{'probe':<6}{'mode':<10}{'price':>7}{'steps':>7}{'n':>4}  {'kept (layer1, layer2) per seed':<44}"
      f"{'1+1':>5}{'solved':>8}{'stays':>7}{'roles':>7}{'compute':>9}")
for key in sorted(groups):
    rs = sorted(groups[key], key=lambda r: r["cfg"]["seed"])
    h1 = rs[0]["sizes"][0]
    kp = [(int(r["kept"][:h1].sum()), int(r["kept"][h1:].sum())) for r in rs]
    print(f"{key[0]:<6}{key[1]:<10}{key[2]:>7g}{key[3]:>7}{len(rs):>4}  {str(kp):<44}"
          f"{sum(k == (1, 1) for k in kp):>5}{sum(r['final_loss'] <= r['bar'] for r in rs):>8}"
          f"{sum(stayed_solved(r) for r in rs):>7}{sum(roles_ok(r) for r in rs):>7}"
          f"{np.mean([r['compute'] for r in rs]):>9.2f}")
