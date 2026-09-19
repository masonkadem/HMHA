"""Ground-truth verification. The point of the project is not accuracy retention but
whether the surviving circuit is the CORRECT one, which is only checkable because the
benchmark has known head roles."""
import numpy as np
import torch

from .tasks import offsets_for, predicted_kstar, trivial_loss
from .train import evaluate, train


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float("nan") if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])


def mean_ci(arr):
    a = np.asarray(arr, dtype=float)
    m, n = np.nanmean(a, axis=0), a.shape[0]
    if n < 2:
        return m, np.zeros_like(m)
    return m, 1.96 * np.nanstd(a, axis=0, ddof=1) / np.sqrt(n)


def k_list(H, extra=()):
    ks, k = set(extra), 1
    while k < H:
        ks.add(k)
        k *= 2
    ks.add(H)
    return sorted(k for k in ks if 1 <= k <= H)


def threshold(full, triv, cfg):
    """Scale-free. A threshold relative to the dense loss alone becomes unreachable once
    the task is solved exactly, which is what produced the spurious 'no redundancy'
    result in earlier runs."""
    return max(full + cfg.gap_tol * (triv - full), full + cfg.abs_floor)


def redundancy_check(cfg, val, device, ks, verbose=True):
    triv, curve = trivial_loss(val), []
    for k in ks:
        steps = int(cfg.steps * (cfg.scratch_mult if k < cfg.num_heads else 1.0))
        m, h = train(cfg, val, device, hemo=False, num_heads=k, steps=steps)
        curve.append(h["val_loss"][-1])
        del m
    full = curve[ks.index(cfg.num_heads)]
    th = threshold(full, triv, cfg)
    kstar = next((k for k, l in zip(ks, curve) if l <= th), cfg.num_heads)
    if verbose:
        print("   k     " + "".join(f"{k:>9d}" for k in ks))
        print("   loss  " + "".join(f"{l:>9.5f}" for l in curve))
        print(f"   trivial {triv:.5f}  full {full:.5f}  thresh {th:.5f}  "
              f"k*={kstar}  predicted={predicted_kstar(cfg)}")
    return dict(ks=ks, curve=curve, full=full, trivial=triv, thresh=th, kstar=kstar,
                predicted=predicted_kstar(cfg), redundant=kstar < cfg.num_heads)


@torch.no_grad()
def head_offsets(model, val, cfg, device, bs=256):
    """offset_profile[h, delta] = mean attention from token i to position (p_i + delta)."""
    X, Y, _, aux = val[0][:bs], val[1][:bs], val[2][:bs], {k: v[:bs] for k, v in val[3].items()}
    N, H = cfg.seq_len, model.H
    _, attn, _ = model.heads(X, Y)
    shift = (torch.arange(N, device=device).view(1, 1, N) + aux["p"].unsqueeze(-1)) % N
    prof = torch.gather(attn, 3, shift.unsqueeze(1).expand(-1, H, -1, -1)).mean(dim=(0, 2))
    return prof.cpu().numpy()


@torch.no_grad()
def circuit_recovery(model, val, cfg, device, gate=None, bs=256):
    """role_coverage  fraction of true offsets implemented by some perfused head
       role_purity    how cleanly perfused heads implement their offset (1 = ideal)
       territory_span number of territories the role-bearing heads occupy (lower = more
                      readable; the territory mechanism's own prediction)"""
    if gate is None:
        gate = model.gate()
    act = (gate > model.leak + 1e-9).cpu().numpy()
    prof = head_offsets(model, val, cfg, device, bs)
    role, purity = prof.argmax(1), prof.max(1)
    true_offs = set(offsets_for(cfg))
    surv = np.where(act)[0]
    real = [h for h in surv if purity[h] > 2.0 / cfg.seq_len]
    found = {int(role[h]) for h in real}
    terr = model.territory.cpu().numpy()
    # the per-head attention-offset profile is what makes a head's ROLE readable, so
    # keep it rather than only the argmax the metrics reduce it to
    return dict(active=act, head_role=role, head_purity=purity, offset_profile=prof,
                n_perfused=int(act.sum()),
                role_coverage=len(found & true_offs) / len(true_offs),
                role_purity=float(purity[surv].mean()) if len(surv) else 0.0,
                n_distinct_roles=len(found),
                territory_span=int(len({int(terr[h]) for h in real})) if real else 0)


@torch.no_grad()
def head_ablation(model, val, gate=None):
    """Run at FULL perfusion. Ablating at a starved gate leaves most heads already at
    zero, which makes every rho a tie-dominated artefact."""
    H = model.H
    if gate is None:
        gate = torch.ones(H, device=val[0].device)
    base = evaluate(model, val, gate=gate)
    delta = np.array([evaluate(model, val, gate=gate.clone().index_fill_(
        0, torch.tensor([i], device=gate.device), 0.0)) - base for i in range(H)])
    X, Y = val[0][:256], val[1][:256]
    Q, attn, out = model.heads(X, Y)
    scores = {"qnorm": Q.norm(dim=-1).mean(dim=(0, 2)),
              "neg_entropy": (attn * attn.clamp_min(1e-9).log()).sum(-1).mean(dim=(0, 2)),
              "outnorm": out.norm(dim=-1).mean(dim=(0, 2)) * model.wo_head_norm(),
              "demand": model.demand}
    res = {"delta": delta}
    for k, v in scores.items():
        s = v.detach().cpu().numpy()
        res[f"score_{k}"], res[f"rho_{k}"] = s, spearman(s, delta)
    return res
