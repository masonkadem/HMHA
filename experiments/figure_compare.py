"""Comparison figure: how each method scores a head, when it removes heads, and how it did, with every
number counted from the saved runs. One definition of 'never broke' for every row: once the model is
first solved, the loss never goes back above the solved bar (2% of a know-nothing model's loss).

  python experiments/figure_compare.py   ->  figures/fig_compare.png and figures/fig_compare.pdf
"""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

RED, GREY, K = "#b2182b", "#6b6b6b", "#111111"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "font.size": 7, "savefig.dpi": 450,
    "pdf.fonttype": 42})

# ---------------------------------------------------------------- count every method from the runs
D = Cfg()
RUNS = []
for d in ("results", "results/proposal", "results/confirm", "results/l0", "results/review", "results/equal"):
    for p in glob.glob(os.path.join(ROOT, d, "*.pkl")):
        r = pickle.load(open(p, "rb"))
        if isinstance(r, dict) and "hist" in r and "cfg" in r and "ledger" in r["hist"]:
            RUNS.append(r)
get = lambda r, k: r["cfg"].get(k, getattr(D, k))
PIN = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", leak=0.0, probe_every=25,
           prune_stop=1, conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0, trial=0,
           target_frac=0.02, budget_hold_frac=0.25, taper=0, price_frac=0.01, l0_lambda=0.01,
           copy_noise=0.0, oneshot=0)
runs_of = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PIN, **w}.items())]


def kept(r):                       # heads open, median over the last 40% of training
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int(0.6 * len(on)):]))


def never_broke(r):                # once first solved, never back above the solved bar
    ls, bar = r["hist"]["val_loss"], 0.02 * r["trivial"]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


def count(arms):                   # every method on the same 30 runs: k* = 2, 4, 8 with seeds 0 to 9
    rs = [r for kw, sizes in arms for k in sizes for r in runs_of(**kw, n_rel=k) if get(r, "seed") < 10]
    return (sum(kept(r) == get(r, "n_rel") for r in rs), sum(never_broke(r) for r in rs), len(rs))


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
EQUAL = (2, 4, 8)
ours = count([(RULE, EQUAL)])
michel = count([(dict(supply="prune"), EQUAL)])
l0_all = {lam: count([(dict(supply="l0", budget_hold_frac=0.1, l0_lambda=lam), EQUAL)]) for lam in (0.05, 0.2, 1.0)}
l0 = l0_all[max(l0_all, key=lambda lam: l0_all[lam][0] + l0_all[lam][1])]     # the best penalty for the baseline
oneshot = count([({**RULE, "oneshot": 1, "budget_hold_frac": 0.3}, EQUAL)])
importance = count([({**RULE, "local_value": "ablate"}, EQUAL)])
random_ = count([({**RULE, "local_value": "random"}, EQUAL)])
frac = lambda a, n: f"{a}/{n}"
l0_exact = f"{l0[0]}/{l0[2]}*"
l0_safe = f"{l0[1]}/{l0[2]}*"

# ---------------------------------------------------------------- the table
rows = [
    # method, type, how a head is scored, how heads are removed, exact, never broke, backups
    ("Collateral rule", "ours", r"$\min_W L_{-h}\, - \,\min_W L$" + "\nloss after removing $h$, read-out re-fitted",
     "during training, from scratch;\nprice, no target; gradual, reopens", frac(ours[0], ours[2]),
     frac(ours[1], ours[2]), "predicted"),
    ("Michel et al. 2019", "published", r"$|\,\partial L / \partial g_h\,|$" + "\ngradient of the loss w.r.t. head gate",
     "one at a time, while the loss\nstays below the solved bar", frac(michel[0], michel[2]), frac(michel[1], michel[2]), "no"),
    ("Voita et al. 2019", "published", r"learned gate $g_h$:  $L + \lambda \sum_h P(g_h \neq 0)$" + "\nhard-concrete (L0) penalty",
     "learned during training", l0_exact, l0_safe, "no"),
    ("Same score, all at once", "control", r"$\min_W L_{-h}\, - \,\min_W L$" + "\n(the OBS / ZipLM way)",
     "all at once after the task is solved,\nthen fine-tuned", frac(oneshot[0], oneshot[2]),
     frac(oneshot[1], oneshot[2]), "no"),
    ("Ordinary importance", "control", r"$L_{-h}(W) - L(W)$" + "\nloss after removing $h$, no re-fit",
     "our rule, this score instead", frac(importance[0], importance[2]), frac(importance[1], importance[2]), "no"),
    ("Random choice", "control", "head picked at random", "our rule, random head",
     frac(random_[0], random_[2]), frac(random_[1], random_[2]), "no"),
]
cols = [("Method", 0.0), ("Type", 0.155), ("How a head is scored", 0.235), ("How heads are removed", 0.49),
        ("Exact\ncount", 0.725), ("Never\nbroke", 0.81), ("Backups", 0.895)]

fig = plt.figure(figsize=(7.2, 3.55))
ax = fig.add_axes([0.01, 0.20, 0.98, 0.78])
ax.axis("off")
ax.set(xlim=(0, 1), ylim=(len(rows) + 0.9, -0.2))
for name, x in cols:
    ax.text(x, 0.15, name, fontsize=7, va="center", ha="left", color=K, fontweight="bold")
ax.plot([0, 1], [0.6, 0.6], color=K, lw=0.6)
for i, (method, kind, score, how, exact, safe, backups) in enumerate(rows):
    y = 1.15 + i
    c = RED if kind == "ours" else K
    ax.text(cols[0][1], y, method, color=c, va="center", fontsize=7)
    ax.text(cols[1][1], y, kind, color=c if kind == "ours" else GREY, va="center", fontsize=6.5)
    ax.text(cols[2][1], y, score, color=c, va="center", fontsize=6.3, linespacing=1.35)
    ax.text(cols[3][1], y, how, color=c, va="center", fontsize=6.3, linespacing=1.3)
    for (_, x), v in zip(cols[4:], (exact, safe, backups)):
        ax.text(x + 0.02, y, v, color=c, va="center", ha="left", fontsize=7)
    if i == 0:
        ax.plot([0, 1], [y + 0.5, y + 0.5], color=GREY, lw=0.4)
ax.plot([0, 1], [len(rows) + 0.65, len(rows) + 0.65], color=K, lw=0.6)

fig.text(0.01, 0.135,
         r"Our score in closed form:  $v_h = \mathrm{tr}(W_h^{\top}\, [M_{hh}]^{-1}\, W_h) / d$,   "
         r"$M = (Z^{\top} Z / n)^{-1}$  ($Z$: the heads' outputs).  This is the Optimal Brain Surgeon / ZipLM score "
         r"$\frac{1}{2}\, w_q^{\top} ([H^{-1}]_{qq})^{-1} w_q$", fontsize=6.5, color=K)
fig.text(0.01, 0.085,
         r"with $H = 2\,Z^{\top}Z/n$, the Hessian of the read-out. The score is not new; what is new is using it during "
         "training, with a price instead of a target size, gradually.", fontsize=6.5, color=K)
fig.text(0.01, 0.045,
         "Exact count: heads left = heads the task needs.  Never broke: once solved, the loss never went back above the "
         "solved bar.", fontsize=6, color=GREY)
fig.text(0.01, 0.012,
         "Backups: how many spare heads are kept when heads can fail, predicted before the runs.  *best of 3 penalty "
         "strengths.  Every method on the same 30 runs (tasks needing 2, 4 and 8 heads, 10 seeds each).", fontsize=6, color=GREY)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_compare.{ext}"), facecolor="white")
print("ours", ours, "michel", michel, "l0", l0_all, "oneshot", oneshot, "importance", importance, "random", random_)
print("wrote figures/fig_compare.png and figures/fig_compare.pdf")
