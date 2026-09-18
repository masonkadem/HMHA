"""Cross-attention with hemodynamic supply allocation.

Two axes, deliberately separated so the ablation grid is clean.

  DEMAND  what the astrocyte senses. A per-head scalar.
  SUPPLY  how a fixed total blood flow is distributed given demand.

Supply always conserves total flow: sum(gate) == H. A gate of 1 is the perfused
baseline, so a dense model is the special case gate == 1 everywhere.

Why this is not pruning. Any mechanism that reduces to "rank heads, keep the top B"
is a ranking, and rankings are what magnitude pruning already does, so it will at best
tie. The two mechanisms below break that.

  threshold  theta = mean(d) + kappa*std(d). The number of perfused heads EMERGES from
             how concentrated demand is, rather than being set as a hyperparameter.
             Prediction: emergent B converges on the task's true k*.
  territory  heads share a penetrating arteriole. Supply is fixed per territory and
             heads compete only with their territory-mates. Starving a head frees flow
             for its neighbours, not globally. No head-importance score has this,
             because all of them score heads independently.
             Prediction: circuits compact into territories, and role recovery collapses
             when the number of territories falls below the number of roles.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import d_out


def standardize(s, dim=-1):
    if s.size(dim) < 2:
        return torch.zeros_like(s)
    return (s - s.mean(dim, keepdim=True)) / (s.std(dim, keepdim=True) + 1e-6)


class CrossAttn(nn.Module):
    def __init__(self, cfg, num_heads=None):
        super().__init__()
        self.cfg = cfg
        self.H = num_heads or cfg.num_heads
        self.d_k, self.d_model = cfg.d_k, cfg.d_model
        self.d_out = d_out(cfg)
        inner = self.H * self.d_k
        self.W_q = nn.Linear(cfg.d_model, inner)
        self.W_k = nn.Linear(cfg.d_model, inner)
        self.W_v = nn.Linear(cfg.d_model, inner)
        self.W_o = nn.Linear(inner, self.d_out)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.H, self.d_k).transpose(1, 2)

    def heads(self, X, Y, attn_override=None):
        Q, K, V = self._split(self.W_q(X)), self._split(self.W_k(Y)), self._split(self.W_v(Y))
        if attn_override is None:
            attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        else:
            attn = attn_override.unsqueeze(1).expand(-1, self.H, -1, -1)
        return Q, attn, attn @ V

    def combine(self, out, gate):
        B = out.size(0)
        out = (out * gate[:, :, None, None]).transpose(1, 2).reshape(B, -1, self.H * self.d_k)
        return self.W_o(out)

    def wo_head_norm(self):
        return self.W_o.weight.detach().view(self.d_out, self.H, self.d_k).norm(dim=(0, 2))

    def forward(self, X, Y, gate=None, attn_override=None, **_):
        _, _, out = self.heads(X, Y, attn_override)
        if gate is None:
            gate = torch.ones(X.size(0), self.H, device=X.device)
        if gate.dim() == 1:
            gate = gate.unsqueeze(0).expand(X.size(0), -1)
        return self.combine(out, gate), gate


class HemoAttn(CrossAttn):
    def __init__(self, cfg, num_heads=None):
        super().__init__(cfg, num_heads)
        H = self.H
        self.demand_kind, self.supply_kind = cfg.demand, cfg.supply
        self.ema, self.leak, self.pool_beta = cfg.ema, cfg.leak, cfg.pool_beta
        self.register_buffer("demand", torch.zeros(H))
        self.register_buffer("hist", torch.zeros(max(1, cfg.delay + 1), H))   # delay line
        self.delay = cfg.delay
        g = torch.Generator().manual_seed(cfg.seed + 7)
        self.register_buffer("rand_score", torch.randn(H, generator=g))
        # contiguous territories, so head index is spatial
        T = max(1, min(cfg.n_territories, H))
        self.register_buffer("territory", torch.arange(H) * T // H)
        self.n_territories = T

    # ---------------- demand ----------------
    @torch.no_grad()
    def instantaneous(self, Q, attn, out):
        if self.demand_kind == "qnorm":
            s = Q.norm(dim=-1).mean(dim=(0, 2))
        elif self.demand_kind == "random":
            s = self.rand_score
        else:                                    # outnorm_ema | outnorm_inst
            s = out.norm(dim=-1).mean(dim=(0, 2)) * self.wo_head_norm()
        return standardize(s)

    @torch.no_grad()
    def update_demand(self, Q, attn, out):
        s = self.instantaneous(Q, attn, out)
        if self.demand_kind in ("random", "outnorm_inst"):
            d = s
        else:
            d = self.demand * self.ema + (1 - self.ema) * s
        if self.pool_beta > 0:                   # astrocytes sense neighbourhood spillover
            pooled = torch.zeros_like(d).index_add_(0, self.territory, d)
            cnt = torch.zeros_like(d).index_add_(0, self.territory,
                                                 torch.ones_like(d)).clamp_min(1)
            d = (1 - self.pool_beta) * d + self.pool_beta * (pooled / cnt)[self.territory]
        self.demand.copy_(d)
        self.hist.copy_(torch.roll(self.hist, 1, dims=0))
        self.hist[0] = d

    def sensed(self):
        """Functional hyperemia lags demand. Supply at t responds to demand at t - tau."""
        return self.hist[min(self.delay, self.hist.size(0) - 1)] if self.delay else self.demand

    # ---------------- supply ----------------
    def gate_topk(self, d, B_active, total=None):
        H = self.H
        total = total if total is not None else H
        if B_active >= d.numel():
            return torch.full_like(d, total / d.numel())
        idx = d.topk(B_active).indices
        g = torch.full_like(d, self.leak)
        g[idx] = (total - self.leak * (d.numel() - B_active)) / B_active
        return g

    def gate_threshold(self, d, kappa, total=None):
        """theta = mean + kappa*std. Perfused set, and therefore B, emerges."""
        total = total if total is not None else self.H
        theta = d.mean() + kappa * d.std().clamp_min(1e-6)
        active = d > theta
        if active.sum() == 0:                                  # never fully infarct
            active = torch.zeros_like(d, dtype=torch.bool)
            active[d.argmax()] = True
        g = torch.full_like(d, self.leak)
        g[active] = (total - self.leak * (~active).sum()) / active.sum()
        return g

    def gate(self, kappa=None, B_active=None):
        d = self.sensed()
        if self.supply_kind == "topk":
            return self.gate_topk(d, B_active if B_active is not None else self.H)
        if self.supply_kind == "threshold":
            return self.gate_threshold(d, 0.0 if kappa is None else kappa)
        if self.supply_kind == "territory":
            # each arteriole carries an equal, fixed share. Competition is LOCAL.
            g = torch.zeros_like(d)
            share = self.H / self.n_territories
            for t in range(self.n_territories):
                m = self.territory == t
                if m.sum() == 0:
                    continue
                sub = d[m]
                if B_active is not None:                       # local top-k
                    kt = max(1, round(B_active / self.n_territories))
                    g[m] = self.gate_topk(sub, min(kt, sub.numel()), total=share)
                else:                                          # local threshold
                    theta = sub.mean() + (0.0 if kappa is None else kappa) * \
                        sub.std().clamp_min(1e-6)
                    act = sub > theta
                    if act.sum() == 0:
                        act = torch.zeros_like(sub, dtype=torch.bool)
                        act[sub.argmax()] = True
                    sg = torch.full_like(sub, self.leak)
                    sg[act] = (share - self.leak * (~act).sum()) / act.sum()
                    g[m] = sg
            return g
        raise ValueError(self.supply_kind)

    def forward(self, X, Y, kappa=None, B_active=None, gate=None,
                attn_override=None, update=True):
        Q, attn, out = self.heads(X, Y, attn_override)
        if gate is None:
            if update and self.training:
                self.update_demand(Q, attn, out)
            gate = self.gate(kappa=kappa, B_active=B_active)
        if gate.dim() == 1:
            gate = gate.unsqueeze(0).expand(X.size(0), -1)
        return self.combine(out, gate), gate
