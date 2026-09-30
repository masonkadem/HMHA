"""Is each head one equation? (thesis proposal, experiment E1)

Train a dense 4-head model on multi_relation (R = 4), then refit the best linear read-out
from subsets of heads, or with one head replaced by a copy of another. Prediction:
loss = (1 - d/R) x trivial, where d counts the DIFFERENT heads. A copy adds nothing.

  python experiments/equations_test.py   ->  results/proposal/equations.pkl  (about 20 s)
"""
import os, sys, pickle
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch
from hemo.config import Cfg
from hemo.tasks import make_val, trivial_loss
from hemo.train import train, pick_device

cfg = Cfg(seed=0)
dev = pick_device(cfg)
val = make_val(cfg, dev)
triv = trivial_loss(val)
model, hist = train(cfg, val, dev, hemo=False, num_heads=4, steps=int(cfg.steps * cfg.scratch_mult))
print(f"4-head model, loss {hist['val_loss'][-1]:.6f}, trivial {triv:.4f}")

fit = make_val(cfg, dev, seed=123)          # fit the read-out on fresh data, score on val


def head_outputs(data):
    with torch.no_grad():
        _, _, out = model.heads(data[0], data[1])        # (batch, head, token, d_k)
    Z = out.permute(0, 2, 1, 3).reshape(-1, 4, cfg.d_k).double().cpu()
    return Z, data[2].reshape(-1, data[2].size(-1)).double().cpu()


Zf, Tf = head_outputs(fit)
Zv, Tv = head_outputs(val)


def refit_loss(cols):
    """Best linear read-out from these heads; a repeated index is a copied head."""
    def design(Z):
        X = torch.cat([Z[:, h] for h in cols], 1)
        return torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    W = torch.linalg.lstsq(design(Zf), Tf, driver="gelsd").solution   # safe with copies
    return float(((design(Zv) @ W - Tv) ** 2).mean())


cases = [([0, 1, 2, 3], 4), ([0, 1, 2], 3), ([0, 2, 3], 3), ([0, 1], 2), ([2, 3], 2),
         ([1], 1), ([3], 1), ([0, 0, 2, 3], 3), ([0, 1, 1, 1], 2)]
rows = []
print(f"\n{'heads used':<28}{'different':>10}{'measured':>10}{'predicted':>11}")
for cols, d in cases:
    name = " + ".join(f"h{h + 1}" for h in cols)
    loss = refit_loss(cols)
    rows.append(dict(heads=cols, name=name, different=d, loss=loss, predicted=(1 - d / 4) * triv))
    print(f"{name:<28}{d:>10}{loss:>10.4f}{(1 - d / 4) * triv:>11.4f}")

os.makedirs(os.path.join(ROOT, "results", "proposal"), exist_ok=True)
with open(os.path.join(ROOT, "results", "proposal", "equations.pkl"), "wb") as f:
    pickle.dump(dict(rows=rows, trivial=triv, dense_loss=hist["val_loss"][-1]), f)
