"""Controls asked for by the novelty review (results/review), next to the matching runs
already on disk.

  A  prune AFTER training in one go (oneshot): at the hold step, close heads one at a time
     with the same collateral value and the same price, no fading, then fine-tune.
     Compared with the collateral rule run during training (hold 0.1, gradual).
  B  near-duplicate twins (copy_noise): every head starts as a noisy copy of another.
     Collateral value against ordinary importance, next to the exact-twin runs.

  python experiments/review_analysis.py   ->  prints the tables, writes results/review/analysis.txt
"""
import glob, io, os, pickle, sys
from contextlib import redirect_stdout
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

D = Cfg()
RUNS = []
for d in ("results", "results/proposal", "results/confirm", "results/review"):
    for p in glob.glob(os.path.join(ROOT, d, "*.pkl")):
        r = pickle.load(open(p, "rb"))
        if isinstance(r, dict) and "hist" in r and "cfg" in r and "ledger" in r["hist"]:
            RUNS.append(r)
get = lambda r, k: r["cfg"].get(k, getattr(D, k))
PIN = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", leak=0.0, probe_every=25,
           head_dropout=0.0, trial=0, target_frac=0.02, supply="local", price_frac=0.03, taper=100,
           conserve=0, local_value="refit", plant_copies=0, copy_noise=0.0, oneshot=0,
           budget_hold_frac=0.1)
runs = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PIN, **w}.items())]


def kept(r):                       # heads with supply, median over the last 40% of training
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int(0.6 * len(on)):]))


# Both checks start from the FIRST time the model is solved anywhere in training, not from
# the step pruning starts: one-shot pruning cuts a model that is already solved, and counting
# only from the cut would hide the jump it causes.
def after_first_solved(r):
    h, bar = r["hist"], 0.02 * r["trivial"]
    first = next((i for i, l in enumerate(h["val_loss"]) if l < bar), None)
    return None if first is None else h["val_loss"][first:]


def stayed_solved(r):              # never back above the solved bar after first reaching it
    ls = after_first_solved(r)
    return ls is not None and max(ls) <= 0.02 * r["trivial"]


def worst_after_solved(r):         # highest loss after first solved, as a fraction of trivial
    ls = after_first_solved(r)
    return float("nan") if ls is None else max(ls) / r["trivial"]


def compute(r):                    # head-steps used, as a fraction of the dense model's
    L = r["hist"]["ledger"]
    return (L > 0).sum(1).mean() / L.shape[1]


def twins(r):                      # pairs (h, h + 16) with both heads kept
    o = r["hist"]["ledger"][-1] > 0
    return sum(bool(o[h] and o[h + 16]) for h in range(16))


out = io.StringIO()
with redirect_stdout(out):
    print("A. Prune after training in one go, then fine-tune (same value, same price)")
    print(f"{'arm':<40}{'R':>3}{'runs':>6}{'exact':>8}{'stayed solved':>15}{'worst loss':>12}"
          f"{'final loss':>12}{'compute':>9}")
    arms = [("collateral rule, during training", dict()),
            ("one-shot at step 400 (before solved)", dict(oneshot=1, budget_hold_frac=0.1)),
            ("one-shot at step 1200 (after solved)", dict(oneshot=1, budget_hold_frac=0.3)),
            ("one-shot at step 2000", dict(oneshot=1, budget_hold_frac=0.5))]
    for name, w in arms:
        for R in (2, 4, 8):
            rs = runs(**w, n_rel=R)
            if not rs:
                continue
            print(f"{name:<40}{R:>3}{len(rs):>6}{sum(kept(r) == R for r in rs):>5}/{len(rs):<2}"
                  f"{sum(stayed_solved(r) for r in rs):>12}/{len(rs):<2}"
                  f"{np.median([worst_after_solved(r) for r in rs]):>12.3f}"
                  f"{np.median([r['final_loss'] for r in rs]):>12.1e}{np.mean([compute(r) for r in rs]):>9.2f}")
            if w:
                print(f"{'':<43}heads kept per run: {[int(kept(r)) for r in rs]}")
    print("  worst loss = highest validation loss after the model is first solved, as a fraction of the")
    print("  loss of a model that learned nothing (the solved bar is 0.02).")

    print("\nB. Every head starts with a twin: exact copies, and noisy copies that drift apart")
    print(f"{'twin':<14}{'value':<22}{'runs':>6}{'heads kept (per run)':>34}{'twin pairs both kept':>24}")
    for cn in (0.0, 0.1, 0.5):
        for lv, name in (("refit", "collateral"), ("ablate", "ordinary importance")):
            rs = runs(n_rel=4, plant_copies=1, copy_noise=cn, local_value=lv)
            if not rs:
                continue
            label = "exact" if cn == 0 else f"noise {cn}"
            print(f"{label:<14}{name:<22}{len(rs):>6}{str([int(kept(r)) for r in rs]):>34}"
                  f"{str([twins(r) for r in rs]):>24}")

text = out.getvalue()
print(text)
os.makedirs(os.path.join(ROOT, "results", "review"), exist_ok=True)
open(os.path.join(ROOT, "results", "review", "analysis.txt"), "w").write(text)
