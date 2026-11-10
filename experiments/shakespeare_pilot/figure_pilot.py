"""Figure for the Shakespeare pilot. Every number is read from results/*.json.

  python experiments/shakespeare_pilot/figure_pilot.py  ->  figures/fig_shakespeare_pilot.png/.pdf
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, ROOT = os.path.join(HERE, "results"), os.path.join(HERE, "..", "..")
RED, K, G1, G2, G3 = "#b2182b", "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})
load = lambda pat: [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, pat)))]
r2, r3, r4, r5 = load("r2_none_*.json"), load("r3_none_*.json"), load("r4_none_*.json"), load("r5_none_*.json")
n = len(r2); assert len(r3) == len(r4) == len(r5) == n
rank = lambda x: np.argsort(np.argsort(x))
rho = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
title = lambda ax, l, t: ax.set_title(f"{l}   {t}", fontweight="bold")
STYLE = dict(zero=(K, "o", "-", "switch the head off"), mean=(G1, "s", "--", "replace by its average"),
             grad=(G2, "^", ":", "gradient score"), refit=(RED, "D", "-", "others in its layer cover (re-fit)"))

fig, axs = plt.subplots(2, 3, figsize=(12.5, 7.4))
fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.08, hspace=0.42, wspace=0.28)

# a: importance against adaptation budget
ax = axs[0, 0]; marks = np.array(r5[0]["marks"]); x = np.arange(len(marks))
sums = np.array([(np.array(r["heads"]) - np.array(r["ref"])[None]).sum(0) / g["gap"] for r, g in zip(r5, r2)])
for s in sums: ax.plot(x, s, color=G2, lw=0.8)
ax.plot(x, sums.mean(0), color=K, lw=1.8, marker="o", ms=4)
refit_sum = np.mean([np.sum(r["local"]) / r["gap"] for r in r2])
ax.axhline(refit_sum, color=RED, lw=1.2, ls=(0, (4, 3))); ax.text(x[-1], refit_sum * 1.12, f"re-fit score, no training: {refit_sum:.2f}", color=K, ha="right")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(marks)
ax.set_xlabel("training steps the network gets to adapt without the head"); ax.set_ylabel("importance of all heads, summed\n(fraction of what the model learned)")
ax.text(x[0] + 0.15, sums.mean(0)[0], f"{sums.mean(0)[0]:.2f}", va="center"); ax.text(x[-1], sums.mean(0)[-1] * 0.72, f"{sums.mean(0)[-1]:.2f}", ha="right")
title(ax, "a", f"Importance shrinks as the network adapts ({n} seeds)")

# b, c: per head, cheap score against the adapted truth
truth = [np.array(r["truth150"]) / r["gap"] for r in r2]
for ax, key, name, col, letter in ((axs[0, 1], "ablate", "switch the head off", K, "b"), (axs[0, 2], "local", "others in its layer cover", RED, "c")):
    sc = [np.array(r[key]) / r["gap"] for r in r2]
    for t, s_, mk in zip(truth, sc, "os^Dv"):
        ax.scatter(t, s_, s=22, color=col, marker=mk, edgecolor="white", linewidth=0.6, zorder=3)
    rs = [rho(s_, t) for s_, t in zip(sc, truth)]
    ax.set_xlabel("true importance: loss rise after the network adapts (150 steps)"); ax.set_ylabel(f"score: {name}")
    title(ax, letter, f"Score: {name}")
    ax.text(0.97, 0.05 if key == "local" else 0.95, "rank agreement per seed:\n" + ", ".join(f"{v:.2f}" for v in rs),
            transform=ax.transAxes, ha="right", va="bottom" if key == "local" else "top")
    ax.set_xlim(left=min(0, min(t.min() for t in truth)) - 0.001)

# d: rank agreement of each score with the truth
ax = axs[1, 0]
scores = dict(zero=[np.array(r["ablate"]) for r in r2], mean=[np.array(r["mean"]) for r in r3], refit=[np.array(r["local"]) for r in r2])
for i, (k, v) in enumerate(scores.items()):
    rs = [rho(a, np.array(r["truth150"])) for a, r in zip(v, r2)]
    ax.scatter([i] * n, rs, s=34, color=STYLE[k][0], marker=STYLE[k][1], edgecolor="white", linewidth=0.8, zorder=3)
    ax.plot([i - 0.22, i + 0.22], [np.mean(rs)] * 2, color=STYLE[k][0], lw=1.6)
    ax.text(i + 0.27, np.mean(rs), f"{np.mean(rs):.2f}", va="center")
ax.axhline(0, color=G2, lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(["switch off", "replace by\naverage", "others cover\n(re-fit)"])
ax.set_xlim(-0.5, 2.7); ax.set_ylim(-0.4, 1.05); ax.set_ylabel("rank agreement with true importance")
title(ax, "d", "Agreement with true importance")

# e: removal, loss after fine-tuning, against heads removed
ax = axs[1, 1]; ks = ("4", "8", "12")
arms = dict(zero=[[r["arm_ablate"]["tuned"][k] / r["gap"] for k in ks] for r in r2],
            mean=[[r["tuned"][k] / g["gap"] for k in ks] for r, g in zip(r3, r2)],
            refit=[[r["arm_local"]["tuned"][k] / r["gap"] for k in ks] for r in r2])
for k, v in arms.items():
    c, mk, ls, lab = STYLE[k]; v = np.array(v)
    ax.plot([4, 8, 12], v.mean(0), color=c, marker=mk, ls=ls, lw=1.6, ms=5, label=lab)
    for row in v: ax.scatter([4, 8, 12], row, s=8, color=c, alpha=0.5, zorder=2)
    ax.text(12.25, v.mean(0)[-1], f"{v.mean(0)[-1]:.3f}", va="center")
ax.set_xticks([4, 8, 12]); ax.set_xlim(3.4, 13.6); ax.set_xlabel("heads removed (of 16)"); ax.set_ylabel("loss rise after 150 fine-tuning steps\n(fraction of what the model learned)")
ax.legend(loc="upper left"); title(ax, "e", "Removing heads chosen by each score")

# f: 8 heads removed, against fine-tuning steps
ax = axs[1, 2]; ms_ = ("0", "150", "300", "600"); xs = np.arange(len(ms_))
for k, arm in (("zero", "ablate"), ("mean", "mean"), ("grad", "grad"), ("refit", "local")):
    c, mk, ls, lab = STYLE[k]
    v = np.array([[(r[arm]["log"][m] - r["ref"][m]) / r["gap"] for m in ms_] for r in r4])
    ax.plot(xs, v.mean(0), color=c, marker=mk, ls=ls, lw=1.6, ms=5, label=lab)
    if k != "grad": ax.text(xs[-1] + 0.08, v.mean(0)[-1] * (1.1 if k == "zero" else 1.0), f"{v.mean(0)[-1]:.3f}", va="center")
ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(ms_); ax.set_xlim(-0.2, 3.55)
ax.set_xlabel("fine-tuning steps after removing 8 of 16 heads"); ax.set_ylabel("loss rise against the full model\ntuned the same")
ax.legend(loc="upper right"); title(ax, "f", "8 heads removed: the gap narrows but remains")

out = os.path.join(ROOT, "figures", "fig_shakespeare_pilot")
fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); print("saved", out, "seeds", n)
