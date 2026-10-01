"""Analysis of the two-layer induction batch (results/induction/*.pkl).

  python experiments/induction_analysis.py
"""
import glob, os, pickle
from math import comb
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "induction", "*.pkl"))]


def pick(rule, price=0.03, heads=(8, 8)):
    return sorted([r for r in RUNS if r["cfg"]["rule"] == rule and tuple(r["sizes"]) == heads
                   and (rule not in ("collateral", "ablate", "random") or r["cfg"]["price_frac"] == price)],
                  key=lambda r: r["cfg"]["seed"])


def kept_per_layer(r):
    h1 = r["sizes"][0]
    return int(r["kept"][:h1].sum()), int(r["kept"][h1:].sum())


def stayed_solved(r):
    h, start = r["hist"], r["cfg"]["start_frac"] * r["cfg"]["steps"]
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= start]
    first = next((i for i, l in enumerate(ls) if l <= r["bar"]), None)
    return first is not None and max(ls[first:]) <= r["bar"]


def roles_found(r):
    """A kept layer-1 head with previous-token attention at least twice chance, and a kept
    layer-2 head with induction attention at least twice chance."""
    h1 = r["sizes"][0]
    k1, k2 = r["kept"][:h1], r["kept"][h1:]
    prev = np.asarray(r["roles"]["prev"][0])[k1]
    ind = np.asarray(r["roles"]["ind"][1])[k2]
    return bool(len(prev) and prev.max() >= 2 * r["chance"]["prev"] and len(ind) and ind.max() >= 2 * r["chance"]["ind"])


def fisher(a, n1, b, n2):
    tot, k = n1 + n2, a + b
    pr = lambda x: comb(n1, x) * comb(n2, k - x) / comb(tot, k)
    obs = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pr(x) <= obs * (1 + 1e-9)))


print(f"{len(RUNS)} induction runs\n")
for name, heads in (("dense, 8 + 8 heads", (8, 8)), ("dense, 1 + 1 heads (known answer)", (1, 1))):
    rs = pick("dense", heads=heads)
    print(f"{name:<36} n={len(rs):>2}  final loss median {np.median([r['final_loss'] for r in rs]):.4f}"
          f"  acc {np.mean([r['final_acc'] for r in rs]):.3f}")
print()

rows = {}
print(f"{'rule':<14}{'n':>3}  {'kept (layer1, layer2) per seed':<60}{'1+1':>5}{'solved':>8}{'stays':>7}{'roles':>7}{'compute':>9}")
for rule in ("collateral", "ablate", "random", "prune"):
    rs = pick(rule)
    kp = [kept_per_layer(r) for r in rs]
    minimal = sum(k == (1, 1) for k in kp)
    solved = sum(r["final_loss"] <= r["bar"] for r in rs)
    stays = sum(stayed_solved(r) for r in rs)
    roles = sum(roles_found(r) for r in rs)
    rows[rule] = (minimal, stays, solved, roles, len(rs))
    print(f"{rule:<14}{len(rs):>3}  {str(kp):<60}{minimal:>5}{solved:>8}{stays:>7}{roles:>7}"
          f"{np.mean([r['compute'] for r in rs]):>9.2f}")
m0, s0, v0, _, n0 = rows["collateral"]
for rule in ("ablate", "random", "prune"):
    m, s, v, _, n = rows[rule]
    print(f"  collateral vs {rule:<8} minimal p = {fisher(m0, n0, m, n):.2g}   stays solved p = {fisher(s0, n0, s, n):.2g}"
          f"   solved p = {fisher(v0, n0, v, n):.2g}")

print("\nprice sweep, collateral rule: heads kept (layer1, layer2) per seed, and solved")
for price in (0.01, 0.03, 0.1, 0.3):
    rs = pick("collateral", price=price)
    print(f"  price {price:<5} n={len(rs):>2}  {[kept_per_layer(r) for r in rs]}  solved {sum(r['final_loss'] <= r['bar'] for r in rs)}/{len(rs)}"
          f"  roles {sum(roles_found(r) for r in rs)}/{len(rs)}")
