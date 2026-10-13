"""Do the two kept layer-1 heads split the previous-token job token by token? One seed per process.

Trains hemodynamic attenuation at the paper's induction settings, then, over the first copy (where
each token must be tagged with its predecessor), measures each kept layer-1 head's attention to t - 1:
  corr_token   correlation across tokens of the two heads' mean attention to t - 1 (negative: each
               head is reliable on tokens where the other is not)
  only_one     share of positions where exactly one head is weak (< 0.4)
  both         share where both are weak
Writes results/induction/team_s<seed>.pkl.

  python experiments/induction_team_check.py --seed 1
"""
import argparse, os, pickle, sys
from dataclasses import replace
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.induction import ICfg, train, make_batch

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
seed = ap.parse_args().seed
cfg = replace(ICfg(), rule="collateral", probe="logit", price_mode="bar", price_frac=0.05, steps=6000,
              seed=seed)                                  # trains on the GPU when there is one
model, out = train(cfg)
model = model.cpu()                                       # the analysis below runs on the CPU
kept = torch.tensor(out["kept"])
h1 = model.sizes[0]
L1 = [i for i in range(h1) if kept[i]]
res = dict(seed=seed, kept=(len(L1), int(kept[h1:].sum())), L1=L1)
if len(L1) == 2:
    A, B = L1
    tok, st = make_batch(2048, cfg, torch.Generator().manual_seed(5))
    with torch.no_grad():
        _, att = model.residual(tok, kept.float(), keep_attn=True)
    b = torch.arange(len(tok)).repeat_interleave(cfg.L - 1)
    t = (st[:, :1] + torch.arange(1, cfg.L)).reshape(-1)
    pa, pb = att[0][b, A, t, t - 1].numpy(), att[0][b, B, t, t - 1].numpy()
    cur = tok[b, t].numpy()
    vs = [v for v in range(cfg.vocab) if (cur == v).sum() > 20]
    ma = np.array([pa[cur == v].mean() for v in vs])
    mb = np.array([pb[cur == v].mean() for v in vs])
    res.update(corr_token=float(np.corrcoef(ma, mb)[0, 1]), corr_example=float(np.corrcoef(pa, pb)[0, 1]),
               only_one=float(((pa < 0.4) ^ (pb < 0.4)).mean()), both=float(((pa < 0.4) & (pb < 0.4)).mean()),
               mean_a=float(pa.mean()), mean_b=float(pb.mean()), per_token=(vs, ma, mb))
with open(os.path.join(ROOT, "results", "induction", f"team_s{seed}.pkl"), "wb") as f:
    pickle.dump(res, f)
print({k: v for k, v in res.items() if k != "per_token"})
