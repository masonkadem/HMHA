import glob, os, pickle, sys, collections
import numpy as np
sys.path.insert(0, "src")
from math import comb
from hemo.config import Cfg
D = Cfg(); get = lambda r, k: r["cfg"].get(k, getattr(D, k))
def p_short(n, R, p):
    q = 1 - p
    return sum(comb(n, a) * q ** a * p ** (n - a) for a in range(0, min(R - 1, n) + 1))
value = lambda B, R, p: (1 - p) * p_short(B - 1, R, p) / R
predict = lambda R, p, c: max(B for B in range(1, 33) if value(B, R, p) >= c)
def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on)):]))
cells = collections.defaultdict(list)
for f in glob.glob("results/backup/multi_relation*.pkl"):
    r = pickle.load(open(f, "rb"))
    cells[(get(r, "n_rel"), get(r, "price_frac"), get(r, "head_dropout"))].append((get(r,"seed"), kept(r)))
tot = ex = 0; allm = []
for (R, c, p), v in sorted(cells.items()):
    pr = predict(R, p, c); ks = [k for _, k in sorted(v)]
    e = sum(k == pr for k in ks); tot += len(ks); ex += e
    allm += [k - pr for k in ks]
    print(f"k*={R} up={c} p={p}: pred {pr}  got {ks}  exact {e}/{len(ks)}  mean miss {np.mean([abs(k-pr) for k in ks]):.2f}")
print(f"TOTAL exact {ex}/{tot} = {ex/tot:.0%}; within 1: {sum(abs(m)<=1 for m in allm)}/{tot}; misses dist {collections.Counter(allm)}")
