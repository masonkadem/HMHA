"""Negative correlations between heads, before any global-signal removal. Are there pairs
that are reliably anticorrelated, and do the heads split into two opposed groups?
Null: circular shifts of one trace (keeps each trace's autocorrelation). One seed per process."""
import sys, os, json
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fc_check.py")).read()
exec(src[src.index("src = open"):src.index("out = dict(seed=a.seed)")])
rng = np.random.default_rng(a.seed); lab = lambda h: f"L{h // H + 1}.{h % H + 1}"
def smooth(x, w):
    if w == 1: return x
    k = np.ones(w) / w; return np.stack([np.convolve(x[:, j], k, mode="valid") for j in range(x.shape[1])], 1)
out = dict(seed=a.seed)
for kind in ("ln", "noln"):
    x, rn, loss = series(get(kind)); x = zs(x); out[kind] = {}
    for w in (1, 8, 32):
        xs = smooth(x, w); fc = np.corrcoef(xs.T); e = fc[iu]
        seg = np.stack([np.corrcoef(s.T)[iu] for s in np.array_split(xs, 4)], 1)
        null = []                                             # most negative r among all pairs under circular shifts
        for _ in range(200):
            sh = np.column_stack([np.roll(xs[:, h], rng.integers(200, len(xs) - 200)) for h in range(NH)]); null.append(np.corrcoef(sh.T)[iu].min())
        thr = float(np.quantile(null, 0.05))                  # family-wise 5% threshold for the most negative pair
        sig = e < thr; allneg = (seg < 0).all(1)
        order = np.argsort(e)[:5]
        ev, evec = np.linalg.eigh(fc - np.eye(NH)); v = evec[:, -1]; grp = v > 0
        between = fc[np.ix_(grp, ~grp)].mean() if grp.any() and (~grp).any() else float("nan")
        wi = np.mean([fc[np.ix_(g_, g_)][np.triu_indices(g_.sum(), 1)].mean() for g_ in (grp, ~grp) if g_.sum() > 1])
        out[kind][w] = dict(min_r=float(e.min()), frac_neg=float((e < 0).mean()), frac_below_m01=float((e < -0.1).mean()), null_thr=thr,
                            n_sig=int(sig.sum()), n_sig_same_layer=int((sig & same).sum()), n_neg_all_segments=int((allneg & (e < -0.1)).sum()),
                            top=[(lab(iu[0][i]), lab(iu[1][i]), round(float(e[i]), 3), [round(float(s), 2) for s in seg[i]]) for i in order],
                            split=(int(grp.sum()), int((~grp).sum())), between=float(between), within=float(wi),
                            icc_negative_edges=icc31(np.arctanh(np.clip(seg[e < 0], -0.999, 0.999))) if (e < 0).sum() > 2 else None)
        o = out[kind][w]
        print(kind, a.seed, f"scale {w}: min r {o['min_r']:.2f} (null 5% {thr:.2f}) significant negative pairs {o['n_sig']}/120 (same layer {o['n_sig_same_layer']}), <-0.1 in all 4 segments {o['n_neg_all_segments']}, "
              f"two-group split {o['split']} between {o['between']:+.2f} within {o['within']:+.2f}; top {o['top'][:3]}", flush=True)
json.dump(out, open(os.path.join(a.out, f"fcneg_s{a.seed}.json"), "w"))
