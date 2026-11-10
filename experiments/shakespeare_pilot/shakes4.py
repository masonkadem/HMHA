"""Round 3 control: MEAN ablation (replace the head's write by its average, other heads
untouched) as score and as compensation, against zero ablation and the own-layer re-fit.
Uses the checkpoints and the adapted truth saved by shakes3.py."""
import sys, os, json, time, copy
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakes3.py")).read().split("zero = lambda")[0]
exec(src[src.index("src = open"):])
zero = lambda: [torch.zeros(D) for _ in range(L)]
r2 = json.load(open(os.path.join(a.out, f"r2_{tag}.json")))
m.eval(); E0 = hloss(m, torch.ones(NH), zero()); dense = vfull(m, torch.ones(NH), zero()); m.eval(); gap = TRIV - dense

@torch.no_grad()
def mean_write(mm, g, bias, hd):
    l, hh = divmod(hd, H); z, _ = fwd(mm, XA, g, bias, stop=l)
    return l, (z[:, hh].reshape(-1, DH) @ mm.blocks[l].o.data[hh]).mean(0) * g[hd]
def cand_mean(mm, g, bias, hd):
    l, mu = mean_write(mm, g, bias, hd); old = bias[l].clone(); gh = g[hd].item()
    bias[l] = old + mu; g[hd] = 0.0; e = hloss(mm, g, bias); bias[l] = old; g[hd] = gh; return e
mean = [cand_mean(m, torch.ones(NH), zero(), h) - E0 for h in range(NH)]
mm = copy.deepcopy(m); mm.eval(); g, bias, curve, tuned = torch.ones(NH), zero(), [], {}
for k in range(1, 13):
    opens = [h for h in range(NH) if g[h] > 0]; sc = [cand_mean(mm, g, bias, x) for x in opens]; h = opens[int(np.argmin(sc))]
    l, mu = mean_write(mm, g, bias, h); bias[l] = bias[l] + mu; g[h] = 0.0
    curve.append(vfull(mm, g, bias) - dense); mm.eval()
    if k in (4, 8, 12): tuned[k] = tune(copy.deepcopy(mm), g.clone(), [b.clone() for b in bias], a.ftp) - r2["ref_ftp"]
json.dump(dict(seed=a.seed, gap=gap, mean=mean, curve=curve, tuned=tuned), open(os.path.join(a.out, f"r3_{tag}.json"), "w"))
print("done", a.seed, flush=True)
