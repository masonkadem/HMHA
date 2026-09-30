"""Confirmation analysis: 10-seed collateral rule against its controls, planted copies, the
price sweep, and the damage (backup-head) predictions in results/confirm/predictions.md.

  python experiments/confirm_analysis.py
"""
import glob, os, pickle, sys
from math import comb
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

RUNS = []
for d in ("results", "results/proposal", "results/confirm"):
    for p in glob.glob(os.path.join(ROOT, d, "*.pkl")):
        r = pickle.load(open(p, "rb"))
        if isinstance(r, dict) and "hist" in r:
            RUNS.append(r)
DEFAULT = Cfg()
PINNED = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", leak=0.0,
              probe_every=25, prune_stop=1, conserve=1, local_value="refit", plant_copies=0,
              head_dropout=0.0, probe_masks=32, target_frac=0.02, budget_hold_frac=0.25,
              taper=0, price_frac=0.01)


def get(r, k):
    return r["cfg"].get(k, getattr(DEFAULT, k))


def runs(**want):
    want = {**PINNED, **want}
    return [r for r in RUNS if all(get(r, k) == v for k, v in want.items())]


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    start = int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on))
    return float(np.median(on[start:]))


def fell_back(r):
    """True if the loss went back above the solved bar after first reaching it."""
    h, bar = r["hist"], 0.02 * r["trivial"]
    start = get(r, "budget_hold_frac") * get(r, "steps")
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= start]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is None or max(ls[first:]) > bar


def compute(r):
    L = r["hist"]["ledger"]
    probes = r["hist"].get("n_probes", 0) * get(r, "probe_batch") / (3 * get(r, "batch_size"))
    return (L > 0).sum(1).mean() / L.shape[1] + probes / len(L)


def fisher(a, n1, b, n2):
    """Two-sided Fisher exact p-value for a/n1 successes against b/n2."""
    tot, k = n1 + n2, a + b
    pr = lambda x: comb(n1, x) * comb(n2, k - x) / comb(tot, k)
    obs = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pr(x) <= obs * (1 + 1e-9)))


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
print(f"{len(RUNS)} runs loaded\n")

# ------------------------------------------------------------------ A: count and stability
print("A. The collateral rule against its controls (no fixed total), heads kept per seed")
ARMS = {"collateral rule": RULE,
        "ordinary importance": {**RULE, "local_value": "ablate"},
        "random pick": {**RULE, "local_value": "random"},
        "standard pruning": dict(supply="prune")}
stats = {}
for name, kw in ARMS.items():
    exact = fell = n = 0
    for k in (2, 3, 4, 6, 8):
        rs = sorted(runs(**kw, n_rel=k), key=lambda r: get(r, "seed"))
        if not rs:
            continue
        B = [kept(r) for r in rs]
        exact += sum(b == k for b in B)
        fell += sum(fell_back(r) for r in rs)
        n += len(rs)
        print(f"  {name:<20} k*={k}  n={len(rs):>2}  kept {[int(b) if b == int(b) else b for b in B]}"
              f"  compute {np.mean([compute(r) for r in rs]):.2f}")
    stats[name] = (exact, fell, n)
print()
print(f"  {'':<20}{'exact count':>14}{'fell back above bar':>22}")
for name, (e, f, n) in stats.items():
    print(f"  {name:<20}{e:>8}/{n:<5}{f:>14}/{n}")
e0, f0, n0 = stats["collateral rule"]
for name in ("ordinary importance", "random pick", "standard pruning"):
    e, f, n = stats[name]
    print(f"  collateral vs {name:<20} exact count p = {fisher(e0, n0, e, n):.1e}"
          f"   fell back p = {fisher(f0, n0, f, n):.1e}")

# ------------------------------------------------------------------ planted copies
print("\nA5. Planted copies (k* = 4, no fixed total): heads kept, pairs of copies both kept")
for value, name in (("refit", "collateral"), ("ablate", "ordinary importance")):
    rs = sorted(runs(**RULE, n_rel=4, plant_copies=1, local_value=value), key=lambda r: get(r, "seed"))
    out = []
    for r in rs:
        on = r["hist"]["ledger"][-1] > 0
        out.append((int(on.sum()), sum(bool(on[h] and on[h + 16]) for h in range(16))))
    print(f"  {name:<20} n={len(rs)}  {out}")

# ------------------------------------------------------------------ price sweep
print("\nA6. Price sweep: heads kept (mean over seeds) at each price")
prices = (0.01, 0.02, 0.03, 0.05, 0.1)
print("  k*  " + "".join(f"{p:>8}" for p in prices))
for k in (2, 4, 8):
    row = []
    for p in prices:
        rs = runs(**{**RULE, "price_frac": p}, n_rel=k)
        row.append(f"{np.mean([kept(r) for r in rs]):>8.1f}" if rs else f"{'-':>8}")
    print(f"  {k:<4}" + "".join(row))

# ------------------------------------------------------------------ B: damage
def value(B, R, p):
    q = 1 - p
    return q * sum(comb(B - 1, a) * q ** a * p ** (B - 1 - a) for a in range(0, min(R - 1, B - 1) + 1)) / R


def predict(R, p, price):
    return max(B for B in range(1, 33) if value(B, R, p) >= price)


print("\nB. Backup heads under damage: heads kept against the prediction written beforehand")
print(f"  {'R':>2} {'price':>6} {'p':>5} {'predicted':>10} {'measured (per seed)':>22} {'mean':>6} {'final loss / trivial':>21}")
misses = []
for R, price in ((2, 0.03), (4, 0.03), (4, 0.01)):
    for p in (0.0, 0.05, 0.1, 0.2, 0.3):
        rs = sorted(runs(**{**RULE, "price_frac": price}, n_rel=R, head_dropout=p), key=lambda r: get(r, "seed"))
        if not rs:
            continue
        B = [kept(r) for r in rs]
        pred = predict(R, p, price)
        if p > 0:
            misses += [b - pred for b in B]
        print(f"  {R:>2} {price:>6} {p:>5} {pred:>10} {str([int(b) if b == int(b) else b for b in B]):>22}"
              f" {np.mean(B):>6.1f} {np.median([r['final_loss'] / r['trivial'] for r in rs]):>21.4f}")
if misses:
    print(f"\n  damaged runs: mean (measured - predicted) = {np.mean(misses):+.2f} heads, "
          f"mean |miss| = {np.mean(np.abs(misses)):.2f}, exact {sum(m == 0 for m in misses)}/{len(misses)}")
