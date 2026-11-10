"""Importance against adaptation budget. For every head of the undamaged Shakespeare model:
the held-out loss rise without it after 0, 1, 5, 10, 20, 40, 150, 300, 600 gradient steps of
whole-network adaptation, against the dense model adapted the same. One seed per process."""
import sys, os, json, time, copy
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakes3.py")).read().split("zero = lambda")[0]
exec(src[src.index("src = open"):])
zero = lambda: [torch.zeros(D) for _ in range(L)]
MARKS = (0, 1, 5, 10, 20, 40, 150, 300, 600)
def curve(hd):
    mm = copy.deepcopy(m); g = torch.ones(NH); bias = zero()
    if hd is not None: g[hd] = 0
    opt = torch.optim.AdamW(mm.parameters(), lr=3e-4, weight_decay=0.0); gen = torch.Generator().manual_seed(777); log = []
    for st in range(MARKS[-1] + 1):
        if st in MARKS: log.append(vfull(mm, g, bias))
        x, y = batch(tr, BS, gen)
        loss = F.cross_entropy(fwd(mm, x, g, bias).reshape(-1, V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return log
ref = curve(None); out = dict(seed=a.seed, marks=MARKS, ref=ref, heads=[])
for h in range(NH):
    out["heads"].append(curve(h)); print(h, time.time() - t0, flush=True)
    json.dump(out, open(os.path.join(a.out, f"r5_{tag}.json"), "w"))
