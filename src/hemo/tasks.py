"""Synthetic tasks with a KNOWN ground-truth minimal circuit.

multi_relation is the primary benchmark. Each query token carries a key position p.
The target is R blocks, block r being the content stored at position (p + offset_r) mod N.
A single head cannot implement two offsets, because W_q is linear and the required
query-key geometry differs per offset. So the minimal head count is exactly R, set by
FUNCTION rather than capacity, and every surviving head has a NAMED role (its offset).
This is what lets interpretability be scored as role-recovery accuracy.

qsa_cross is the legacy capacity-limited task. Minimal head count is
ceil(q*log2(N) / d_k). Heads are interchangeable, so it validates head COUNT but not
head ROLES. Kept for comparison only.
"""
import math
import torch
import torch.nn.functional as F


def offsets_for(cfg):
    return [(1 + r * max(1, cfg.seq_len // cfg.n_rel)) % cfg.seq_len for r in range(cfg.n_rel)]


def predicted_kstar(cfg):
    if cfg.task == "multi_relation":
        return cfg.n_rel
    return max(1, math.ceil(cfg.q * math.log2(cfg.seq_len) / cfg.d_k))


def d_out(cfg):
    return cfg.n_rel * cfg.m_content if cfg.task == "multi_relation" else cfg.seq_len


def make_batch(B, cfg, device, gen=None):
    """Returns X, Y, T, aux. aux holds ground truth used by the interpretability metrics."""
    N, d = cfg.seq_len, cfg.d_model
    dev = "cpu" if gen is not None else device
    kw = {"generator": gen} if gen is not None else {}

    if cfg.task == "qsa_cross":
        X = torch.randn(B, N, d, device=dev, **kw)
        Y = torch.randn(B, N, d, device=dev, **kw)
        Y[:, :, :N] = torch.eye(N, device=dev)
        idx = torch.rand(B, N, N, device=dev, **kw).topk(cfg.q, dim=-1).indices
        idx, _ = idx.sort(dim=-1)
        mh = torch.zeros(B, N, N, device=dev).scatter_(2, idx, 1.0)
        X[:, :, :N] = mh
        T = (torch.bmm(mh, Y) / cfg.q)[:, :, :N]
        aux = {"multi_hot": mh}

    elif cfg.task == "multi_relation":
        m = cfg.m_content
        assert d >= N + m, "d_model must hold the position block plus the content block"
        X = torch.randn(B, N, d, device=dev, **kw) * 0.1
        Y = torch.randn(B, N, d, device=dev, **kw) * 0.1
        Y[:, :, :N] = torch.eye(N, device=dev)
        content = torch.randn(B, N, m, device=dev, **kw)
        Y[:, :, N:N + m] = content
        p = torch.randint(0, N, (B, N), device=dev, **kw)
        X[:, :, :N] = F.one_hot(p, N).float()
        blocks = []
        for off in offsets_for(cfg):
            j = (p + off) % N
            blocks.append(torch.gather(content, 1, j.unsqueeze(-1).expand(-1, -1, m)))
        T = torch.cat(blocks, dim=-1)
        aux = {"p": p, "offsets": torch.tensor(offsets_for(cfg), device=dev)}
    else:
        raise ValueError(cfg.task)

    mv = lambda t: t.to(device)
    return mv(X), mv(Y), mv(T), {k: mv(v) for k, v in aux.items()}


def make_val(cfg, device, seed=10_000):
    return make_batch(cfg.val_size, cfg, device, gen=torch.Generator().manual_seed(seed))


def trivial_loss(val):
    T = val[2]
    return ((T - T.mean(dim=(0, 1), keepdim=True)) ** 2).mean().item()
