"""Modular addition, (a + b) mod P, one-layer transformer trained past memorisation until it
generalises (grokking). Every --every steps, score each attention head three ways on the
TRAIN half not used for fitting and on TEST:
  zero   switch the head off
  mean   replace what it writes by its average
  refit  the other heads get new output maps (least squares) to cover for it
One seed per process.  python grok.py --seed 0"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--P", type=int, default=97)
ap.add_argument("--frac", type=float, default=0.4); ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--wd", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--H", type=int, default=8); ap.add_argument("--every", type=int, default=100)
ap.add_argument("--out", default="."); a = ap.parse_args()
torch.set_num_threads(1); torch.manual_seed(a.seed)
P, H, D = a.P, a.H, 128; DH = D // H
ab = torch.cartesian_prod(torch.arange(P), torch.arange(P)); X = torch.cat([ab, torch.full((P * P, 1), P)], 1); Y = (ab[:, 0] + ab[:, 1]) % P
perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(a.seed + 100)); n = int(a.frac * P * P)
tr, te = perm[:n], perm[n:]; fitA, trB = tr[: n // 2], tr[n // 2:]          # fit the cover on A, score on B and on test

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(P + 1, D); s.pos = nn.Parameter(torch.randn(3, D) / math.sqrt(D))
        s.q, s.k, s.v = (nn.Linear(D, D, bias=False) for _ in range(3))
        s.o = nn.Parameter(torch.randn(H, DH, D) / math.sqrt(D))
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D)); s.un = nn.Linear(D, P, bias=False)
    def heads(s, x):
        """Residual before attention at the last position, and each head's output there."""
        h = s.emb(x) + s.pos; B = len(x)
        q = s.q(h[:, -1]).view(B, H, 1, DH); k = s.k(h).view(B, 3, H, DH).transpose(1, 2); v = s.v(h).view(B, 3, H, DH).transpose(1, 2)
        att = (q @ k.transpose(-2, -1) / math.sqrt(DH)).softmax(-1)
        return h[:, -1], (att @ v).squeeze(2)                                # (B, D), (B, H, DH)
    def after(s, r, write):
        r = r + write; return s.un(r + s.mlp(r))
    def forward(s, x):
        r, z = s.heads(x); return s.after(r, torch.einsum("bhd,hdm->bm", z, s.o))

def ridge(Z, T, lam=1e-4):
    G = Z.T @ Z / len(Z); G += lam * torch.diagonal(G).mean() * torch.eye(G.size(0), dtype=G.dtype)
    return torch.linalg.solve(G, Z.T @ T / len(Z))
one = lambda z: torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], 1)

@torch.no_grad()
def probe(m):
    rA, zA = m.heads(X[fitA]); per = torch.einsum("bhd,hdm->bhm", zA, m.o); fullA = per.sum(1)
    out = {}
    sets = dict(train=trB, test=te[torch.randperm(len(te), generator=torch.Generator().manual_seed(1))[:2000]])
    for name, idx in sets.items():
        r, z = m.heads(X[idx]); y = Y[idx]; pe = torch.einsum("bhd,hdm->bhm", z, m.o); full = pe.sum(1)
        def score(write):
            lg = m.after(r, write); return F.cross_entropy(lg, y).item(), (lg.argmax(-1) == y).float().mean().item()
        l0, a0 = score(full); res = dict(loss=l0, acc=a0, zero=[], mean=[], refit=[], zero_acc=[], mean_acc=[], refit_acc=[])
        for h in range(H):
            oth = [j for j in range(H) if j != h]
            l, ac = score(full - pe[:, h]); res["zero"].append(l - l0); res["zero_acc"].append(a0 - ac)
            l, ac = score(full - pe[:, h] + per[:, h].mean(0)); res["mean"].append(l - l0); res["mean_acc"].append(a0 - ac)
            W = ridge(one(zA[:, oth].reshape(len(zA), -1).double()), fullA.double())
            l, ac = score((one(z[:, oth].reshape(len(z), -1).double()) @ W).float()); res["refit"].append(l - l0); res["refit_acc"].append(a0 - ac)
        out[name] = res
    return out

m = Net(); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
log, t0 = [], time.time()
for st in range(a.steps + 1):
    if st % a.every == 0:
        pr = probe(m); pr["step"] = st; pr["wnorm"] = float(sum((p ** 2).sum() for p in m.parameters()).sqrt())
        with torch.no_grad(): pr["train_full_loss"] = F.cross_entropy(m(X[tr]), Y[tr]).item()
        log.append(pr)
        if st % (a.every * 20) == 0:
            print(st, f"train {pr['train']['loss']:.4f}/{pr['train']['acc']:.3f} test {pr['test']['loss']:.3f}/{pr['test']['acc']:.3f} "
                  f"zero {sum(pr['test']['zero']):.2f} refit {sum(pr['test']['refit']):.2f} {time.time()-t0:.0f}s", flush=True)
    loss = F.cross_entropy(m(X[tr]), Y[tr]); opt.zero_grad(); loss.backward(); opt.step()
json.dump(dict(args=vars(a), log=log), open(os.path.join(a.out, f"grok_P{P}_f{a.frac}_H{H}_s{a.seed}.json"), "w"))
