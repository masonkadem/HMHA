"""Round 4: does the advantage survive longer fine-tuning, and a gradient baseline?
Remove 8 of 16 heads by: ablate (greedy zero ablation), mean (greedy mean ablation, mean kept),
grad (smallest |dL/dgate|, Michel et al. 2019, one shot), local (greedy own-layer re-fit, kept).
Then fine-tune 600 steps, held-out loss at 0/150/300/600 against the dense model tuned the same."""
import sys, os, json, time, copy
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakes4.py")).read()
exec(src[src.index("src = open"):src.index("mean = [cand_mean")])
K = 8; MARKS = (0, 150, 300, 600)
def tune_log(mm, g, bias):
    opt = torch.optim.AdamW(mm.parameters(), lr=3e-4, weight_decay=0.0); gen = torch.Generator().manual_seed(777); log = {}
    for st in range(MARKS[-1] + 1):
        if st in MARKS: log[st] = vfull(mm, g, bias)
        x, y = batch(tr, BS, gen)
        loss = F.cross_entropy(fwd(mm, x, g, bias).reshape(-1, V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return log
out = dict(seed=a.seed, gap=gap, ref=tune_log(copy.deepcopy(m), torch.ones(NH), zero()))
for arm in ("ablate", "mean", "grad", "local"):
    mm = copy.deepcopy(m); mm.eval(); g, bias = torch.ones(NH), zero()
    if arm == "grad":
        gg = torch.ones(NH, requires_grad=True); imp = torch.zeros(NH); gen = torch.Generator().manual_seed(5)
        for _ in range(20):
            x, y = batch(tr, BS, gen); l = F.cross_entropy(fwd(mm, x, gg, bias).reshape(-1, V), y.reshape(-1))
            imp += torch.autograd.grad(l, gg)[0].abs()
        g[imp.argsort()[:K]] = 0.0
    else:
        for _ in range(K):
            opens = [h for h in range(NH) if g[h] > 0]
            sc = [cand_mean(mm, g, bias, x) if arm == "mean" else cand(mm, g, bias, x, arm == "local") for x in opens]
            h = opens[int(np.argmin(sc))]
            if arm == "local": apply_fit(mm, g, bias, h, local_fit(mm, g, bias, h))
            elif arm == "mean":
                l, mu = mean_write(mm, g, bias, h); bias[l] = bias[l] + mu; g[h] = 0.0
            else: g[h] = 0.0
    out[arm] = dict(removed=[h for h in range(NH) if g[h] == 0], log=tune_log(mm, g, bias))
    print(arm, {k: round((v - out["ref"][k]) / gap, 3) for k, v in out[arm]["log"].items()}, flush=True)
json.dump(out, open(os.path.join(a.out, f"r4_{tag}.json"), "w"))
