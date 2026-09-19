import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .model import CrossAttn, HemoAttn
from .tasks import make_batch


def pick_device(cfg):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")


def lr_lambda(cfg, total):
    def f(step):
        if step < cfg.lr_warmup:
            return (step + 1) / cfg.lr_warmup
        p = (step - cfg.lr_warmup) / max(1, total - cfg.lr_warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    return f


def schedule(step, cfg, H, total):
    """Returns (kappa, B_active). Phase 1 is fully perfused, phase 2 progressive
    ischemia, phase 3 held at the floor."""
    hold = int(cfg.budget_hold_frac * total)
    ann = max(1, int(cfg.budget_anneal_frac * total))
    if step < hold:
        p = 0.0
    elif step < hold + ann:
        p = (step - hold) / ann
    else:
        p = 1.0
    kappa = cfg.kappa_start + p * (cfg.kappa_end - cfg.kappa_start)
    B = H if p == 0 else max(cfg.budget_min, int(round(H * (cfg.budget_min / H) ** p)))
    return kappa, B


@torch.no_grad()
def evaluate(model, val, bs=256, **kw):
    X, Y, T = val[0], val[1], val[2]
    was = model.training
    model.eval()
    tot = 0.0
    for i in range(0, X.size(0), bs):
        p, _ = model(X[i:i + bs], Y[i:i + bs], **kw)
        tot += F.mse_loss(p, T[i:i + bs], reduction="sum").item()
    model.train(was)
    return tot / T.numel()


def train(cfg, val, device, hemo=True, num_heads=None, steps=None, desc="", verbose=False):
    """hemo=False trains a plain dense model (used for the ground-truth k* curve)."""
    torch.manual_seed(cfg.seed)
    total = steps or cfg.steps
    model = (HemoAttn(cfg, num_heads) if hemo else CrossAttn(cfg, num_heads)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = LambdaLR(opt, lr_lambda(cfg, total))
    H = model.H
    h = dict(val_step=[], val_loss=[], budget=[], kappa=[], ledger=[], n_perfused=[])
    prev = None

    for step in range(total):
        X, Y, T, _ = make_batch(cfg.batch_size, cfg, device)
        if hemo:
            kappa, B = schedule(step, cfg, H, total)
            pred, g = model(X, Y, kappa=kappa, B_active=B)
        else:
            kappa, B = None, H
            pred, g = model(X, Y)
        loss = F.mse_loss(pred, T)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        nper = int((g[0] > model.leak + 1e-9).sum()) if hemo else H
        if hemo:
            h["ledger"].append(g[0].detach().cpu().numpy())
        # always measure just before the perfused set changes, so every level is recorded
        nxt = schedule(step + 1, cfg, H, total)[1] if hemo else H
        if (step % cfg.val_every == 0 or step == total - 1
                or (hemo and (nxt != B or nper != prev))):
            prev = nper
            kw = dict(kappa=kappa, B_active=B, update=False) if hemo else {}
            h["val_step"].append(step)
            h["val_loss"].append(evaluate(model, val, **kw))
            h["budget"].append(B)
            h["kappa"].append(kappa)
            h["n_perfused"].append(nper)
            if verbose:
                print(f"    {desc} step {step:>5} loss {h['val_loss'][-1]:.5f} "
                      f"perfused {nper}/{H}")
    h["ledger"] = np.array(h["ledger"]) if hemo else None
    h["final_perfused"] = prev if hemo else H
    h["final_gate"] = g[0].detach().cpu().numpy()
    h["final_kappa"], h["final_budget"] = kappa, B
    return model, h
