"""Analysis of the three follow-up experiments:
  A  results/robust    do backup heads protect the model? (knock out each kept head)
  B  results/trial2 and results/induction/*tu0.5*   gentler trial closure
  C  results/l0        learned hard-concrete gates (Voita et al. 2019) against the collateral rule

  python experiments/extra_analysis.py
"""
import glob, os, pickle
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
load = lambda pattern: [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, pattern))]


def stayed_solved(r, bar=None, start=None):
    h = r["hist"]
    bar = 0.02 * r["trivial"] if bar is None else bar
    start = r["cfg"]["budget_hold_frac"] * r["cfg"]["steps"] if start is None else start
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= start]
    first = next((i for i, l in enumerate(ls) if l <= bar), None)
    return first is not None and max(ls[first:]) <= bar


def kept_main(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int(0.6 * len(on)):]))


# ------------------------------------------------------------------ A
print("A. Do backup heads protect the model? (k* = 4; knock out one kept head after training,")
print("   nobody re-fits; loss in units of the solved bar)")
print(f"{'p fail':>7} {'n':>3} {'heads kept':>16} {'intact':>8} {'worst knockout':>15} {'mean knockout':>14}")
A = load("results/robust/*.pkl")
for p in sorted({r["cfg"].get("head_dropout", 0.0) for r in A}):
    rs = sorted([r for r in A if r["cfg"].get("head_dropout", 0.0) == p], key=lambda r: r["cfg"]["seed"])
    bar = 0.02 * rs[0]["trivial"]
    kept = [len(r["knockout"]["per_head"]) for r in rs]
    intact = np.median([r["knockout"]["intact"] / bar for r in rs])
    worst = [max(r["knockout"]["per_head"].values()) / bar for r in rs]
    mean = [np.mean(list(r["knockout"]["per_head"].values())) / bar for r in rs]
    print(f"{p:>7} {len(rs):>3} {str(kept):>16} {intact:>8.2f} {np.median(worst):>15.2f} {np.median(mean):>14.2f}")

# ------------------------------------------------------------------ B
print("\nB. Gentler trial closure (undo at half the bar, at most 3 failures, none in the last quarter)")
ind = load("results/induction/*.pkl")


def ind_group(**w):
    return [r for r in ind if all(r["cfg"].get(k, d) == v for k, (v, d) in w.items())]


base = dict(rule=("collateral", None), probe=("logit_tone", "mse"), price_mode=("bar", "absolute"),
            price_frac=(0.02, None), steps=(6000, None), squeeze_at=(0.0, 0.0))
for name, extra in (("no trials", dict(trial=(0, 0))),
                    ("trials, first version", dict(trial=(1, 0), trial_undo=(1.0, 1.0))),
                    ("trials, gentle version", dict(trial=(1, 0), trial_undo=(0.5, 1.0)))):
    rs = ind_group(**base, **extra)
    if not rs:
        continue
    h1 = rs[0]["sizes"][0]
    kp = [(int(r["kept"][:h1].sum()), int(r["kept"][h1:].sum())) for r in rs]
    st = sum(stayed_solved(r, r["bar"], r["cfg"]["start_frac"] * r["cfg"]["steps"]) for r in rs)
    print(f"  induction, {name:<24} n={len(rs):>2}  exact 1+1 {sum(k == (1, 1) for k in kp):>2}/{len(rs)}"
          f"  solved {sum(r['final_loss'] <= r['bar'] for r in rs):>2}/{len(rs)}  stays solved {st:>2}/{len(rs)}  kept {kp}")
T2 = load("results/trial2/*.pkl")
for k in sorted({r["cfg"]["n_rel"] for r in T2}):
    rs = [r for r in T2 if r["cfg"]["n_rel"] == k]
    print(f"  main task k*={k}, gentle trials: n={len(rs)}  exact {sum(kept_main(r) == k for r in rs)}/{len(rs)}"
          f"  solved {sum(r['final_loss'] <= 0.02 * r['trivial'] for r in rs)}/{len(rs)}"
          f"  stays solved {sum(stayed_solved(r) for r in rs)}/{len(rs)}")

# ------------------------------------------------------------------ C
print("\nC. Learned hard-concrete gates (Voita et al. 2019) by penalty strength; heads kept per seed")
L0 = load("results/l0/*.pkl")
for lam in sorted({r["cfg"]["l0_lambda"] for r in L0}):
    row = []
    exact = stays = n = 0
    for k in (2, 4, 8):
        rs = sorted([r for r in L0 if r["cfg"]["l0_lambda"] == lam and r["cfg"]["n_rel"] == k],
                    key=lambda r: r["cfg"]["seed"])
        kp = [int(kept_main(r)) for r in rs]
        row.append(f"k*={k}: {kp}")
        exact += sum(x == k for x in kp)
        stays += sum(stayed_solved(r) for r in rs)
        n += len(rs)
    print(f"  lambda {lam:<5} {'  '.join(row):<60} exact {exact}/{n}  stays solved {stays}/{n}")
print("  (collateral rule for comparison: exact 47/50, stays solved 50/50)")
