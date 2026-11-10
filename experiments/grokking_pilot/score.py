"""Score the predictions in predictions.md for each seed."""
import json, glob, sys, numpy as np
for f in sorted(glob.glob(sys.argv[1] + "/grok_*.json")):
    d = json.load(open(f)); L = d["log"]; s = d["args"]["seed"]
    st = np.array([l["step"] for l in L]); S = lambda side, k: np.array([np.sum(l[side][k]) for l in L])
    acc = np.array([l["test"]["acc"] for l in L]); tel = np.array([l["test"]["loss"] for l in L]); wn = np.array([l["wnorm"] for l in L])
    first = lambda cond: int(st[np.argmax(cond)]) if cond.any() else None
    g = first(acc >= 0.5); plat = (st >= 1000) & (st <= 3000); last = st >= st[-1] - 2000
    rz, zz, tr = S("train", "refit"), S("train", "zero"), S("test", "refit")
    def half(x):                       # first step after the plateau where x stays below half its plateau mean for 3 probes
        b = x < 0.5 * x[plat].mean(); ok = b & np.roll(b, -1) & np.roll(b, -2) & (st > 3000); return first(ok)
    def pos(x):
        b = x > 0; ok = b & np.roll(b, -1) & np.roll(b, -2) & (st > 3000); return first(ok)
    h_r, h_z, p_t = half(rz), half(zz), pos(tr)
    lead = lambda x: None if (x is None or g is None) else g - x
    ratio = lambda m: rz[m].sum() / zz[m].sum()
    print(f"seed {s}: grok step {g} | P1 refit half at {h_r} lead {lead(h_r)} | P2 zero half at {h_z} lead {lead(h_z)} | "
          f"P3 ratio plateau {ratio(plat):.3f} final {ratio(last):.4f} | P4 test refit plateau mean {tr[plat].mean():+.2f}, positive at {p_t} lead {lead(p_t)} | "
          f"baselines: test loss half at {half(tel)} lead {lead(half(tel))}, wnorm half at {half(wn)}")
