"""Aggregates results/*.pkl into figures and a summary table.

Every number in every figure, caption and table below is computed from the pickles at
render time. Nothing is typed in by hand. An earlier version of this work shipped a
summary box that contradicted the plots beside it, which is why this file exists.

  python experiments/aggregate.py --results results --out figures
"""
import argparse, glob, os, pickle, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from hemo.analysis import mean_ci, spearman

# validated palette (dataviz reference instance, light surface #fcfcfb).
# categorical slots in fixed order, never cycled; ordinal blue ramp for kappa.
SURFACE = "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
       "#e34948"]
ORD5 = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GOOD, CRIT = "#0ca30c", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK3, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "font.size": 9,
    "axes.titlesize": 10, "legend.frameon": False, "figure.dpi": 130,
})


def nlab(ns):
    return f"n = {ns[0]}" if len(set(ns)) == 1 else f"n = {min(ns)}-{max(ns)}"


def tidy(ax, title=None, xlabel=None, ylabel=None, grid="y"):
    if title:
        ax.set_title(title, loc="left", color=INK, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis=grid, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


# ---------------------------------------------------------------- loading

def load(results):
    runs = []
    for p in sorted(glob.glob(os.path.join(results, "*.pkl"))):
        with open(p, "rb") as f:
            r = pickle.load(f)
        r["_path"], r["_tag"] = p, os.path.basename(p)[:-4]
        runs.append(r)
    return runs


def sel(runs, **kw):
    """Select runs whose cfg matches every key. Values may be scalars or sets."""
    out = []
    for r in runs:
        c = r["cfg"]
        if all((c[k] in v if isinstance(v, (set, list, tuple)) else c[k] == v)
               for k, v in kw.items()):
            out.append(r)
    return out


def by(runs, key):
    d = defaultdict(list)
    for r in runs:
        d[r["cfg"][key]].append(r)
    return dict(sorted(d.items()))


def agg(rs, fn):
    """mean and 95% CI over seeds of fn(run)."""
    v = np.array([fn(r) for r in rs], dtype=float)
    m, ci = mean_ci(v.reshape(len(v), 1))
    return float(m[0]), float(ci[0])


def frac_at_kstar(r):
    """Fraction of the held final phase whose perfused set has exactly k* heads. For
    the emergent-B mechanisms this is the honest version of 'did it find k*': B is not
    imposed, so a run can pass through k* early and settle somewhere else."""
    led = r["hist"]["ledger"]
    if led is None or len(led) < 10:
        return float("nan")
    hold = int((r["cfg"]["budget_hold_frac"] + r["cfg"]["budget_anneal_frac"]) * len(led))
    a = (led[hold:] > r["cfg"]["leak"] + 1e-9).sum(axis=1)
    return float((a == r["kstar"]).mean()) if len(a) else float("nan")


def gate_flips(r):
    """Mean per-step changes in the perfused SET over the held final phase. The delay
    experiment's prediction is that too long a tau makes this rise (gate oscillation)."""
    led = r["hist"]["ledger"]
    if led is None or len(led) < 10:
        return float("nan")
    leak = r["cfg"]["leak"]
    act = led > leak + 1e-9
    hold = int((r["cfg"]["budget_hold_frac"] + r["cfg"]["budget_anneal_frac"])
               * len(led))
    a = act[hold:]
    return float((a[1:] != a[:-1]).sum(axis=1).mean()) if len(a) > 1 else float("nan")


# ---------------------------------------------------------------- figures

def fig_ground_truth(runs, out, rows):
    rs = [r for r in runs if "redundancy" in r]
    if not rs:
        return
    ks = rs[0]["redundancy"]["ks"]
    curves = np.array([r["redundancy"]["curve"] for r in rs])
    m, ci = mean_ci(curves)
    triv = float(np.mean([r["redundancy"]["trivial"] for r in rs]))
    th = float(np.mean([r["redundancy"]["thresh"] for r in rs]))
    kstars = [r["redundancy"]["kstar"] for r in rs]
    pred = rs[0]["redundancy"]["predicted"]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(ks, m, yerr=ci, color=CAT[0], lw=2, marker="o", ms=5, capsize=3,
                label=f"dense, k heads from scratch (n={len(rs)} seeds)")
    ax.axhline(triv, color=INK3, lw=1.2, ls=":")
    ax.axhline(th, color=CRIT, lw=1.2, ls="--")
    ax.annotate(f"trivial (predict-mean) {triv:.3f}", (ks[0], triv), color=INK2,
                fontsize=8, va="bottom", ha="left")
    ax.annotate(f"threshold {th:.4f}", (ks[0], th), color=CRIT, fontsize=8,
                va="bottom", ha="left")
    ax.axvline(pred, color=GOOD, lw=1.2, ls="-", alpha=0.7)
    ax.annotate(f"predicted k* = {pred}", (pred, m.max()), color=GOOD, fontsize=8,
                rotation=90, va="top", ha="right")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ks); ax.set_xticklabels(ks)
    tidy(ax, "Ground truth: loss vs dense head count", "heads k", "val MSE (log)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig0_ground_truth.png")); plt.close(fig)

    rows.append(("0 ground truth", f"measured k* per seed = {kstars}, predicted = {pred}; "
                 f"trivial = {triv:.4f}, threshold = {th:.5f}"))
    rows.append(("0 ground truth", "loss by k: " + ", ".join(
        f"k={k}: {v:.5f}" for k, v in zip(ks, m))))


def fig_emergent_B(runs, out, rows):
    rs = sel(runs, supply="threshold", demand="outnorm_ema", delay=0, pool_beta=0.0,
             leak=0.0)
    g = by(rs, "kappa_end")
    if len(g) < 2:
        return
    kstar = rs[0]["kstar"]
    kaps = list(g)
    B_m, B_c = zip(*[agg(v, lambda r: r["recovery"]["n_perfused"]) for v in g.values()])
    L_m, L_c = zip(*[agg(v, lambda r: r["final_loss"]) for v in g.values()])
    # B is emergent here, so record how much of the converged phase was actually spent
    # at k* rather than implying the run was ever held there.
    A_m = [agg(v, frac_at_kstar)[0] for v in g.values()]
    C_m, C_c = zip(*[agg(v, lambda r: r["recovery"]["role_coverage"]) for v in g.values()])
    ns = [len(v) for v in g.values()]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    ax.errorbar(kaps, B_m, yerr=B_c, color=CAT[0], lw=2, marker="o", ms=6, capsize=3,
                label="emergent perfused count B")
    ax.axhline(kstar, color=GOOD, lw=1.5, ls="--", label=f"true k* = {kstar}")
    for x, y, e in zip(kaps, B_m, B_c):
        ax.annotate(f"{y:.1f}", (x, y + e), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8, color=INK2)
    ax.set_xticks(kaps); ax.set_xticklabels([f"{k:g}" for k in kaps])
    tidy(ax, "Emergent B vs ischemia depth", "kappa_end", "perfused heads")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    ax.errorbar(kaps, L_m, yerr=L_c, color=CAT[1], lw=2, marker="s", ms=6, capsize=3,
                label="val MSE at the converged gate")
    ax.axhline(rs[0]["trivial"], color=INK3, lw=1.2, ls=":")
    ax.annotate(f"trivial {rs[0]['trivial']:.3f}", (kaps[0], rs[0]["trivial"]),
                color=INK2, fontsize=8, va="bottom", ha="left")
    ax.set_yscale("log")
    ax.set_xticks(kaps); ax.set_xticklabels([f"{k:g}" for k in kaps])
    tidy(ax, "Cost of over-starvation", "kappa_end", "val MSE (log)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    ax.errorbar(kaps, C_m, yerr=C_c, color=CAT[2], lw=2, marker="^", ms=6, capsize=3,
                label="role coverage at the final gate")
    ax.set_ylim(-0.05, 1.15)
    for x, y, e in zip(kaps, C_m, C_c):
        ax.annotate(f"{y:.2f}", (x, y + e), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8, color=INK2)
    ax.set_xticks(kaps); ax.set_xticklabels([f"{k:g}" for k in kaps])
    tidy(ax, "Role recovery", "kappa_end", "fraction of true offsets found")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"1. Emergent B vs true k*  (threshold supply, outnorm_ema demand, "
                 f"{nlab(ns)} seeds)", x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "fig1_emergent_B.png")); plt.close(fig)

    for k, b, bc, l, c, n, am in zip(kaps, B_m, B_c, L_m, C_m, ns, A_m):
        rows.append(("1 emergent B", f"kappa_end={k:g} (n={n}): B={b:.2f}+/-{bc:.2f} "
                     f"(k*={kstar}), converged loss={l:.5f}, "
                     f"frac of held phase at exactly k* = {am:.2f}, "
                     f"role_coverage={c:.2f}"))
    best = min(zip(kaps, B_m), key=lambda t: abs(t[1] - kstar))
    rows.append(("1 emergent B", f"closest to k*={kstar}: kappa_end={best[0]:g} "
                 f"gives B={best[1]:.2f}"))


def fig_territory(runs, out, rows):
    rs = sel(runs, supply="territory", demand="outnorm_ema", pool_beta=0.0)
    g = by(rs, "n_territories")
    if len(g) < 2:
        return
    R = rs[0]["cfg"]["n_rel"]
    Ts = list(g)
    cov = [agg(v, lambda r: r["recovery"]["role_coverage"]) for v in g.values()]
    spn = [agg(v, lambda r: r["recovery"]["territory_span"]) for v in g.values()]
    per = [agg(v, lambda r: r["recovery"]["n_perfused"]) for v in g.values()]
    ns = [len(v) for v in g.values()]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    ax.errorbar(Ts, [c[0] for c in cov], yerr=[c[1] for c in cov], color=CAT[0], lw=2,
                marker="o", ms=6, capsize=3, label="role coverage")
    ax.axvline(R, color=GOOD, lw=1.5, ls="--")
    ax.annotate(f"T = R = {R}", (R, 1.0), color=GOOD, fontsize=8, rotation=90,
                va="top", ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks(Ts); ax.set_xticklabels(Ts)
    ax.set_ylim(-0.05, 1.05)
    tidy(ax, "Prediction: coverage collapses below T = R", "n_territories",
         "fraction of true offsets found")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.errorbar(Ts, [s[0] for s in spn], yerr=[s[1] for s in spn], color=CAT[1], lw=2,
                marker="s", ms=6, capsize=3, label="territory span of role heads")
    ax.plot(Ts, Ts, color=INK3, lw=1.2, ls=":", label="span = T (no compaction)")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(Ts); ax.set_xticklabels(Ts)
    tidy(ax, "Compaction", "n_territories", "territories occupied")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.errorbar(Ts, [p[0] for p in per], yerr=[p[1] for p in per], color=CAT[2], lw=2,
                marker="^", ms=6, capsize=3, label="perfused heads")
    ax.axhline(R, color=GOOD, lw=1.5, ls="--", label=f"true k* = {R}")
    ax.set_xscale("log", base=2); ax.set_xticks(Ts); ax.set_xticklabels(Ts)
    tidy(ax, "Perfused count", "n_territories", "heads")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"2. Territory compaction  (territory supply, R = {R}, "
                 f"{nlab(ns)} seeds)", x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "fig2_territory.png")); plt.close(fig)

    for T, c, s, p, n in zip(Ts, cov, spn, per, ns):
        rows.append(("2 territory", f"T={T} (n={n}): role_coverage={c[0]:.2f}+/-{c[1]:.2f}, "
                     f"territory_span={s[0]:.2f}, perfused={p[0]:.2f}"))


def fig_delay(runs, out, rows):
    rs = sel(runs, supply="threshold", demand="outnorm_ema", pool_beta=0.0, leak=0.0,
             kappa_end=1.5)
    g = by(rs, "delay")
    if len(g) < 2:
        return
    taus = list(g)
    x = np.arange(len(taus))
    loss = [agg(v, lambda r: r["final_loss"]) for v in g.values()]
    flip = [agg(v, gate_flips) for v in g.values()]
    cov = [agg(v, lambda r: r["recovery"]["role_coverage"]) for v in g.values()]
    ns = [len(v) for v in g.values()]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    ax.errorbar(x, [l[0] for l in loss], yerr=[l[1] for l in loss], color=CAT[0], lw=2,
                marker="o", ms=6, capsize=3, label="val MSE at the converged gate")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(taus)
    bi = int(np.argmin([l[0] for l in loss]))
    ax.annotate(f"best tau = {taus[bi]}", (x[bi], loss[bi][0]), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=8, color=GOOD)
    tidy(ax, "Is there an optimal lag?", "delay tau (steps)", "val MSE (log)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.errorbar(x, [f[0] for f in flip], yerr=[f[1] for f in flip], color=CAT[1], lw=2,
                marker="s", ms=6, capsize=3, label="perfused-set changes per step")
    ax.set_xticks(x); ax.set_xticklabels(taus)
    tidy(ax, "Gate oscillation in the held phase", "delay tau (steps)",
         "head flips per step")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    ax.errorbar(x, [c[0] for c in cov], yerr=[c[1] for c in cov], color=CAT[2], lw=2,
                marker="^", ms=6, capsize=3, label="role coverage")
    ax.set_xticks(x); ax.set_xticklabels(taus); ax.set_ylim(-0.05, 1.05)
    tidy(ax, "Role recovery", "delay tau (steps)", "fraction of true offsets found")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"3. Delay  (threshold supply, kappa_end = 1.5, "
                 f"{nlab(ns)} seeds)", x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "fig3_delay.png")); plt.close(fig)

    for t, l, f, c, n in zip(taus, loss, flip, cov, ns):
        rows.append(("3 delay", f"tau={t} (n={n}): converged loss={l[0]:.5f}+/-{l[1]:.5f}, "
                     f"flips/step={f[0]:.2f}, role_coverage={c[0]:.2f}"))


def fig_pool(runs, out, rows):
    rs = sel(runs, supply="territory", demand="outnorm_ema", n_territories=8)
    g = by(rs, "pool_beta")
    if len(g) < 2:
        return
    bs = list(g)
    x = np.arange(len(bs))
    cov = [agg(v, lambda r: r["recovery"]["role_coverage"]) for v in g.values()]
    spn = [agg(v, lambda r: r["recovery"]["territory_span"]) for v in g.values()]
    loss = [agg(v, lambda r: r["final_loss"]) for v in g.values()]
    ns = [len(v) for v in g.values()]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, vals, col, mk, lab, yl in [
            (axes[0], cov, CAT[0], "o", "role coverage", "fraction of true offsets found"),
            (axes[1], spn, CAT[1], "s", "territory span of role heads", "territories"),
            (axes[2], loss, CAT[2], "^", "val MSE at the converged gate",
             "val MSE (log)")]:
        ax.errorbar(x, [v[0] for v in vals], yerr=[v[1] for v in vals], color=col, lw=2,
                    marker=mk, ms=6, capsize=3, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f"{b:g}" for b in bs])
        tidy(ax, lab.capitalize(), "pool_beta", yl)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)
    axes[2].set_yscale("log")

    fig.suptitle(f"4. Pooled demand  (territory supply, T = 8, "
                 f"{nlab(ns)} seeds)", x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "fig4_pool.png")); plt.close(fig)

    for b, c, s, l, n in zip(bs, cov, spn, loss, ns):
        rows.append(("4 pool", f"pool_beta={b:g} (n={n}): role_coverage={c[0]:.2f}, "
                     f"territory_span={s[0]:.2f}, converged loss={l[0]:.5f}+/-{l[1]:.5f}"))


def fig_demand_arms(runs, out, rows):
    rs = sel(runs, supply="topk")
    if not rs:
        return
    arms = [a for a in ["outnorm_ema", "outnorm_inst", "qnorm", "random"]
            if sel(rs, demand=a)]
    leaks = sorted({r["cfg"]["leak"] for r in rs})
    kstar = rs[0]["kstar"]
    x = np.arange(len(arms))
    w = 0.8 / max(1, len(leaks))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    for i, lk in enumerate(leaks):
        vals = [agg(sel(rs, demand=a, leak=lk), lambda r: r["loss_at_kstar"])
                for a in arms]
        pos = x + (i - (len(leaks) - 1) / 2) * w
        ax.bar(pos, [v[0] for v in vals], width=w * 0.92, color=CAT[i],
               edgecolor=SURFACE, linewidth=2, label=f"leak = {lk:g}")
        ax.errorbar(pos, [v[0] for v in vals], yerr=[v[1] for v in vals], fmt="none",
                    ecolor=INK2, capsize=3, lw=1)
        for p, v in zip(pos, vals):
            ax.annotate(f"{v[0]:.4f}", (p, v[0]), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7, color=INK2)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(arms, rotation=12)
    tidy(ax, f"5. Demand arms at matched perfusion (B = k* = {kstar})", None,
         "val MSE (log)")
    ax.legend(loc="best", fontsize=8)

    # rho(score, ablation delta) at FULL perfusion, pooled over every run in the repo
    ax = axes[1]
    names = ["demand", "outnorm", "neg_entropy", "qnorm"]
    vals = [agg(runs, lambda r: r["ablation"][f"rho_{n}"]) for n in names]
    ax.barh(np.arange(len(names)), [v[0] for v in vals], height=0.6,
            color=[GOOD if v[0] > 0 else CRIT for v in vals], edgecolor=SURFACE,
            linewidth=2)
    ax.errorbar([v[0] for v in vals], np.arange(len(names)),
                xerr=[v[1] for v in vals], fmt="none", ecolor=INK2, capsize=3, lw=1)
    for i, v in enumerate(vals):
        ax.annotate(f"{v[0]:+.2f}", (v[0], i), textcoords="offset points",
                    xytext=(6 if v[0] > 0 else -6, 0), ha="left" if v[0] > 0 else "right",
                    va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK3, lw=1)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names)
    ax.set_xlim(-1.05, 1.05)
    tidy(ax, f"rho(score, ablation delta), full perfusion, all {len(runs)} runs", "rho",
         None, grid="x")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_demand_arms.png"))
    plt.close(fig)

    for a in arms:
        for lk in leaks:
            v = agg(sel(rs, demand=a, leak=lk), lambda r: r["loss_at_kstar"])
            n = len(sel(rs, demand=a, leak=lk))
            rows.append(("5 demand arms", f"{a}, leak={lk:g} (n={n}): "
                         f"loss@B=k*={v[0]:.5f}+/-{v[1]:.5f}"))
    for n_, v in zip(names, vals):
        rows.append(("5 ablation", f"rho({n_}, ablation delta) = {v[0]:+.3f}+/-{v[1]:.3f} "
                     f"over {len(runs)} runs, at full perfusion"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.results)
    if not runs:
        print(f"no pickles in {a.results}/")
        return
    print(f"{len(runs)} runs loaded")
    rows = []
    for f in (fig_ground_truth, fig_emergent_B, fig_territory, fig_delay, fig_pool,
              fig_demand_arms):
        try:
            f(runs, a.out, rows)
        except Exception as e:                       # a missing sweep must not kill the rest
            print(f"  {f.__name__}: skipped ({type(e).__name__}: {e})")

    # the table view. Same numbers as the figures, read from the same pickles.
    md = os.path.join(a.out, "summary.md")
    with open(md, "w") as fh:
        fh.write(f"# Results\n\nComputed from {len(runs)} runs in `{a.results}/`. "
                 f"Every number here and in the figures is read from those pickles.\n\n")
        cur = None
        for k, v in rows:
            if k != cur:
                fh.write(f"\n## {k}\n\n"); cur = k
            fh.write(f"- {v}\n")
    print("\n".join(f"{k:<16} {v}" for k, v in rows))
    print(f"\nwrote {a.out}/*.png and {md}")


if __name__ == "__main__":
    main()
