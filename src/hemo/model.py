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

             kappa is normalised per territory by local_kappa, because a threshold in
             raw standard deviations is not comparable between a 32-head pool and a
             4-head territory. See local_kappa.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import d_out


def zmax(n):
    """Largest z-score attainable within a group of n values. No member of a group can
    exceed mean + kappa*std once kappa >= this, so a threshold set above it always
    empties the perfused set and falls back to argmax."""
    return (n - 1) / math.sqrt(n) if n > 1 else 0.0


def local_kappa(kappa, n, H):
    """kappa expressed as a FRACTION OF WHAT IS ATTAINABLE at this group size, so the
    same kappa means the same thing to a 32-head pool and a 4-head territory.

    Without this the territory mechanism is untestable: at H=32 a territory of H/T
    heads caps out at zmax(H/T), which is 1.50 at T=8 and 0.71 at T=16, so the default
    kappa_end of 1.5 guarantees an empty perfused set and the fallback pins B to
    exactly T for every T >= 8. The measured 'emergent' count was then just T.
    """
    ref = zmax(H)
    return kappa if ref == 0 else kappa * zmax(n) / ref


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
        self.autoreg_gain = cfg.autoreg_gain
        self.autoreg_every = max(1, cfg.autoreg_every)
        self.loss_ema_c = cfg.loss_ema
        self.slow_c = 0.995
        self.stall_gate = cfg.stall_gate
        self.flow_exponent = cfg.flow_exponent
        self.watershed_penalty = cfg.watershed_penalty
        self.register_buffer("loss_state", torch.tensor(0.0))
        self.register_buffer("slow_loss", torch.tensor(0.0))
        self.register_buffer("deficit", torch.zeros(H))
        self.register_buffer("deficit_raw", torch.zeros(H))
        self.register_buffer("reserve", torch.zeros(H))
        self.flow_price = cfg.flow_price
        self.cvr_boost, self.cvr_every = cfg.cvr_boost, max(1, cfg.cvr_every)
        self.terr_kappa = cfg.terr_kappa
        self.kappa_floor, self.kappa_ceil = cfg.kappa_start, cfg.kappa_ceil
        self.register_buffer("kappa_state", torch.tensor(float(cfg.kappa_start)))
        T = max(1, min(cfg.n_territories, H))
        self.register_buffer("territory", torch.arange(H) * T // H)
        self.n_territories = T
        # ischemic preconditioning (autoreg): the kappa the loop may not constrict past
        self.precondition, self.precondition_relax = cfg.precondition, cfg.precondition_relax
        self.register_buffer("kappa_cap", torch.tensor(float(cfg.kappa_ceil)))
        self._last_B, self._drop, self._strikes = None, None, {}
        # local supply and the pruning baseline: which heads are open, and why
        self.register_buffer("open_mask", torch.ones(H, dtype=torch.bool))
        self.register_buffer("head_value", torch.zeros(H))
        self.register_buffer("michel", torch.zeros(H))
        # local only: each head's vessel tone in [0, 1], which follows open_mask at
        # 1/taper per step, so a closing head fades while the survivors take up its share
        self.register_buffer("tone", torch.ones(H))
        self.taper = cfg.taper
        self._removed, self._since_removal, self._prune_done = None, 0, False
        # controls on the local rule; the defaults (1, "refit", 0) are the rule itself
        self.conserve, self.local_value = cfg.conserve, cfg.local_value
        self.register_buffer("head_ablate", torch.zeros(H))
        self._pick = torch.Generator().manual_seed(cfg.seed + 11)
        self.head_dropout, self.probe_masks = cfg.head_dropout, cfg.probe_masks
        self._damage = torch.Generator().manual_seed(cfg.seed + 13)
        self.trial_head, self.trial_taper = None, cfg.trial_taper
        # BASELINE l0: learned hard-concrete gates (Louizos et al. 2018) as used for head
        # pruning by Voita et al. (2019). Only created for supply="l0".
        if cfg.supply == "l0":
            self.log_alpha = nn.Parameter(torch.full((H,), float(cfg.l0_init)))
        self._l0 = (2.0 / 3.0, -0.1, 1.1)                      # beta, gamma, zeta
        if cfg.plant_copies:
            self.plant_copies()

    @torch.no_grad()
    def plant_copies(self):
        """Make head h + H/2 an exact copy of head h, read-out included. Copies get equal
        gradients, so they stay copies until one is closed."""
        half, dk = self.H // 2, self.d_k
        noise = float(getattr(self.cfg, "copy_noise", 0.0))
        g = torch.Generator().manual_seed(self.cfg.seed + 11)
        jitter = lambda w: w + noise * w.std() * torch.randn(w.shape, generator=g).to(w) if noise else w
        for h in range(half):
            src, dst = slice(h * dk, (h + 1) * dk), slice((h + half) * dk, (h + half + 1) * dk)
            for lin in (self.W_q, self.W_k, self.W_v):
                lin.weight[dst] = jitter(lin.weight[src])
                lin.bias[dst] = lin.bias[src]
            self.W_o.weight[:, dst] = jitter(self.W_o.weight[:, src])

    # ---------------- demand ----------------
    def needs_gate_grad(self):
        return self.demand_kind == "deficit" or self.supply_kind in ("marginal", "prune")

    def needs_reserve_probe(self):
        return self.demand_kind == "reserve"

    @torch.no_grad()
    def probe_reserve(self, X, Y, T, loss_fn):
        """CEREBROVASCULAR RESERVE, the clinical CO2-challenge measurement.

        For each head, give it cvr_boost times its current share of the FIXED total
        supply, pay for that out of every other head, and record how much the loss
        falls. A head already at its ceiling shows no reserve; a starved head holding a
        latent role shows a lot. The two can have identical gradients, which is why a
        first-order signal cannot tell them apart and why this is not the gradient
        importance of Michel et al.

        Every head-importance score in that literature is first-order or leave-one-out
        REMOVAL. This is addition under conservation, and removing a head changes every
        other head's reserve, so it is not a per-head score at all.
        """
        base_g = self.gate()
        _, _, out = self.heads(X, Y)
        base = float(loss_fn(self.combine(out, base_g.unsqueeze(0).expand(X.size(0), -1)), T))
        H = self.H
        r = torch.zeros_like(base_g)
        for h in range(H):
            g = base_g.clone()
            extra = (self.cvr_boost - 1.0) * base_g[h]
            if extra <= 0:                       # a fully starved head is given a share
                extra = self.cvr_boost * float(base_g.sum()) / H
            others = torch.ones_like(g, dtype=torch.bool)
            others[h] = False
            pool = float(g[others].sum())
            if pool <= 0:
                continue
            g[h] = base_g[h] + extra
            g[others] = g[others] * (1.0 - extra / pool)   # conserved: sum is unchanged
            g.clamp_min_(0.0)
            L = float(loss_fn(self.combine(out, g.unsqueeze(0).expand(X.size(0), -1)), T))
            r[h] = base - L
        self.reserve.mul_(self.ema).add_((1 - self.ema) * r)

    @torch.no_grad()
    def absorb_gradient(self):
        """Per-head DEFICIT, -dL/dg_h: how much loss a head would shed if given more
        flow. Unlike a demand signal such as the query norm, this depends on the gate,
        so it is near zero for a head that is already well perfused and large for a
        starved head that would contribute if fed. That is the difference between
        measuring drive and measuring shortfall, and it is what lets a supply loop stop
        without being told a target.

        As a RANKING this quantity is gradient head importance (Michel et al. 2019), a
        named baseline. Its use here is as the sensor of a conserved supply loop, not as
        a score to prune by.
        """
        g = getattr(self, "_gate_leaf", None)
        if g is None or g.grad is None:
            return
        v = -g.grad.detach()
        if v.dim() > 1:
            v = v.sum(0)
        # Michel et al. (2019) head importance, |dL/dmask|, for the pruning baseline
        self.michel.mul_(self.ema).add_((1 - self.ema) * v.abs())
        # Project onto the conservation surface. Supply is fixed, so the useful question
        # is never "would this head benefit from more flow" (at a fitted solution no head
        # would, since the output scale is already learned) but "would this head benefit
        # MORE THAN THE OTHERS", which is the component orthogonal to the all-ones
        # direction along which sum(g) = H is preserved.
        v = v - v.mean()
        self.deficit_raw.copy_(v)
        self.deficit.copy_(standardize(v))

    @torch.no_grad()
    def instantaneous(self, Q, attn, out):
        if self.demand_kind == "reserve":
            s = self.reserve
        elif self.demand_kind == "deficit":
            s = self.deficit
        elif self.demand_kind == "qnorm":
            s = Q.norm(dim=-1).mean(dim=(0, 2))
        elif self.demand_kind == "q2norm":
            # squared query norm. Sharpens the demand distribution relative to qnorm,
            # which is the only way a SENSOR can change the count a z-cut admits.
            s = Q.pow(2).sum(-1).mean(dim=(0, 2))
        elif self.demand_kind == "entropy":
            # a head attending sharply is doing a job; a diffuse head is not. Scored
            # rho=+0.45 against ablation, above the outnorm demand actually in use.
            s = (attn * attn.clamp_min(1e-9).log()).sum(-1).mean(dim=(0, 2))
        elif self.demand_kind == "random":
            s = self.rand_score
        else:                                    # outnorm_ema | outnorm_inst
            s = out.norm(dim=-1).mean(dim=(0, 2)) * self.wo_head_norm()
        return standardize(s)

    @torch.no_grad()
    def update_demand(self, Q, attn, out):
        s = self.instantaneous(Q, attn, out)
        if self.demand_kind in ("random", "outnorm_inst", "q2norm"):
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

    def _from_active(self, d, active, total):
        """Equal share among the perfused set, leak to the rest. Conserves sum(gate)."""
        if active.sum() == 0:                                  # never fully infarct
            active = torch.zeros_like(d, dtype=torch.bool)
            active[d.argmax()] = True
        g = torch.full_like(d, self.leak)
        g[active] = (total - self.leak * (~active).sum()) / active.sum()
        return g

    def gate_gap(self, d, total=None):
        """Cut at the LARGEST GAP in the sorted demand. B is read off the shape of the
        distribution rather than from a fixed number of standard deviations, so a task
        that concentrates demand in more heads yields a larger perfused set."""
        total = total if total is not None else self.H
        s, _ = torch.sort(d, descending=True)
        k = int(torch.argmax(s[:-1] - s[1:]).item()) + 1
        thr = s[k - 1]
        return self._from_active(d, d >= thr, total)

    def gate_otsu(self, d, total=None):
        """Two-cluster split maximising between-class variance (Otsu's method in 1-D).
        Same motivation as gate_gap but robust to a single large gap in the tail."""
        total = total if total is not None else self.H
        s, _ = torch.sort(d, descending=True)
        n = s.numel()
        k = torch.arange(1, n, device=d.device, dtype=d.dtype)
        cs = torch.cumsum(s, 0)
        mu_a = cs[:-1] / k                                     # mean of the top k
        mu_b = (cs[-1] - cs[:-1]) / (n - k)
        between = k * (n - k) * (mu_a - mu_b) ** 2
        kk = int(torch.argmax(between).item()) + 1
        return self._from_active(d, d >= s[kk - 1], total)

    def gate_marginal(self, total=None):
        """Perfuse a head while the loss reduction an extra unit of flow would buy
        exceeds the metabolic price of that flow. The perfused count is then set by where
        marginal benefit meets marginal cost, with no target loss to calibrate. The price
        has units of loss per unit flow rather than loss, so it does not have to be
        rescaled per task the way target_frac does.
        """
        total = total if total is not None else self.H
        v = self.deficit_raw
        active = v > self.flow_price
        return self._from_active(v, active, total)

    def gate_poiseuille(self, d, kappa, total=None):
        """Flow through a vessel scales as the FOURTH POWER of its radius.

        Every other rule here hands the perfused heads an EQUAL share. This one splits
        the same conserved supply by r^n, with r the vessel tone above threshold, so a
        small difference in demand produces a large difference in flow. The perfused SET
        is identical to the threshold rule's, which isolates the effect of grading from
        the effect of selection; n = 1 is the linear-share control.
        """
        total = total if total is not None else self.H
        theta = d.mean() + kappa * d.std().clamp_min(1e-6)
        r = (d - theta).clamp_min(0.0)
        if not bool((r > 0).any()):                        # never fully infarct
            r = torch.zeros_like(d)
            r[d.argmax()] = 1.0
        w = r.pow(self.flow_exponent)
        active = w > 0
        g = torch.full_like(d, self.leak)
        g[active] = (total - self.leak * (~active).sum()) * w[active] / w[active].sum()
        return g

    def gate_watershed(self, d, kappa, total=None):
        """Territories, with the never-fully-infarct floor applied GLOBALLY.

        The territory rule applies that floor per territory, so every territory keeps a
        live head and the perfused count is at least T by construction. Compaction below
        T is then impossible and the mechanism cannot exhibit the effect it predicts.
        Here a territory may go completely dark and its share is redistributed to the
        territories still perfusing, which is collateral flow. Heads on a territory
        boundary have their demand discounted, since watershed zones between two arterial
        beds are the first tissue to infarct.
        """
        total = total if total is not None else self.H
        terr = self.territory
        boundary = torch.zeros_like(d, dtype=torch.bool)
        # aggregate demand per territory, the quantity arterioles would compete on
        agg = torch.stack([d[terr == t].mean() if bool((terr == t).any())
                           else d.min() for t in range(self.n_territories)])
        boundary[:-1] |= terr[:-1] != terr[1:]
        boundary[1:] |= terr[1:] != terr[:-1]
        dd = d - (1.0 - self.watershed_penalty) * d.std().clamp_min(1e-6) * boundary
        active = torch.zeros_like(d, dtype=torch.bool)
        for t in range(self.n_territories):
            m = terr == t
            if not bool(m.any()):
                continue
            sub = dd[m]
            kl = local_kappa(kappa, int(sub.numel()), self.H)
            active[m] = sub > sub.mean() + kl * sub.std().clamp_min(1e-6)
        if int(active.sum()) == 0:                  # the floor is GLOBAL, not per region
            active[int(dd.argmax())] = True
        live = [t for t in range(self.n_territories) if bool(active[terr == t].any())]
        if self.terr_kappa > -8.0:
            # INTER-TERRITORY COMPETITION. Without this every live territory takes an
            # equal share regardless of its demand, so flow never moves between
            # territories and vascular steal, the whole point of the mechanism, does not
            # happen. Here a territory whose aggregate demand falls below the threshold
            # goes dark and its flow is taken by the survivors in proportion to demand.
            thr = agg.mean() + self.terr_kappa * agg.std().clamp_min(1e-6)
            live = [t for t in live if float(agg[t]) > float(thr)]
            if not live:
                live = [int(agg.argmax())]
            w = torch.stack([agg[t] for t in live])
            w = (w - w.min() + 1e-6)
            w = w / w.sum()
            shares = {t: float(total) * float(wi) for t, wi in zip(live, w)}
        else:
            shares = {t: total / max(1, len(live)) for t in live}
        g = torch.full_like(d, self.leak)
        for t in live:
            m = (terr == t) & active
            if not bool(m.any()):
                continue
            dark = int(((terr == t) & ~active).sum())
            g[m] = (shares[t] - self.leak * dark) / int(m.sum())
        s = float(g.sum())
        if s > 0:                                   # conservation, exactly
            g.mul_(total / s)
        return g

    def autoregulate(self, loss, target, step=0, n_perfused=None):
        """Functional hyperemia as a closed loop. Flow responds to a metabolic DEFICIT,
        not to a fixed quantile: above target the vessel dilates (kappa falls, more heads
        perfuse), below target it constricts. The control error is a log ratio so the
        loop is scale-free in the loss, and the target is a fraction of the trivial
        baseline so it is scale-free across tasks.

        With loss_ema = 0 and autoreg_every = 1 this is per-step proportional control on
        the raw minibatch loss, which is what the reported autoreg runs used.

        stall_gate > 0 adds the distinction between an ACUTE and a CHRONIC deficit.
        Tissue dilates when a shortfall persists, not when it is merely transient.
        Without it the loop dilates through the whole early training transient, which is
        why a tight loss target over-perfused: at k* = 4 with target 0.005 the perfused
        count settled at 13.3 rather than 4.

        precondition > 0 adds ISCHEMIC PRECONDITIONING. Proportional control with a
        target the model can beat always constricts until it loses a head it needs, then
        dilates back, so it chatters between k* and k* - 1. Tissue that survives an
        ischemic episode tolerates the next one. Here, when a constriction has to be
        reversed, kappa is capped just below where the head was lost, and the cap
        drifts back up slowly so the memory fades.
        """
        import math
        if self.precondition > 0.0 and n_perfused is not None:
            if self._last_B is not None and n_perfused < self._last_B:
                self._drop = (self._last_B, float(self.kappa_state))
            self._last_B = n_perfused
            self.kappa_cap.add_(self.precondition_relax).clamp_(max=self.kappa_ceil)
        l = float(loss)
        if self.loss_ema_c > 0.0:
            self.loss_state.mul_(self.loss_ema_c).add_((1 - self.loss_ema_c) * l)
            sensed = float(self.loss_state) / (1 - self.loss_ema_c ** max(1, step + 1))
        else:
            sensed = l
        self.slow_loss.mul_(self.slow_c).add_((1 - self.slow_c) * l)
        if step % self.autoreg_every:
            return float(self.kappa_state)
        err = math.log(max(target, 1e-12)) - math.log(max(sensed, 1e-12))
        if self.stall_gate > 0.0 and err < 0.0:
            slow = float(self.slow_loss) / (1 - self.slow_c ** max(1, step + 1))
            rate = (slow - sensed) / max(slow, 1e-12)      # > 0 while still improving
            if rate > self.stall_gate:
                return float(self.kappa_state)             # acute: hold, do not dilate
        if err < 0.0 and self._drop is not None:           # the last constriction hurt
            # two strikes at the same count: a transient early in training strikes once,
            # the chatter at k* strikes again within a few hundred steps
            B_lost, kappa_lost = self._drop
            self._strikes[B_lost] = self._strikes.get(B_lost, 0) + 1
            if self._strikes[B_lost] >= 2:
                self.kappa_cap.fill_(min(float(self.kappa_cap),
                                         kappa_lost - self.precondition))
            self._drop = None
        self.kappa_state += self.autoreg_gain * max(-4.0, min(4.0, err))
        self.kappa_state.clamp_(self.kappa_floor, float(self.kappa_cap))
        return float(self.kappa_state)

    @torch.no_grad()
    def probe_ischemia(self, X, Y, T, ridge=1e-4):
        """Each head's own DEFICIT, measured with collateral compensation.

        For an open head: how much worse the model gets without it, once the remaining
        open heads have re-fitted their output weights to cover for it. For a starved
        head: how much better the model gets if it is fed and everyone re-fits. Tissue
        survives losing a vessel when collaterals can take over its territory, so the
        question is never 'what does this head carry' but 'what can nobody else carry'.
        A head that duplicates another is therefore worth nothing, however much it writes.

        Least squares on the heads' outputs, so both quantities are closed-form: one
        inverse of the open heads' Gram matrix, then a block downdate per open head and a
        Schur complement per starved head. The attention patterns, which ARE the heads'
        roles, are held fixed; only the linear read-out is re-fitted.
        """
        H, dk = self.H, self.d_k
        _, _, out = self.heads(X, Y)                                  # (b, H, n, dk)
        Z = out.permute(0, 2, 1, 3).reshape(-1, H * dk).double()
        t = T.reshape(-1, T.size(-1)).double()
        Z, t = Z - Z.mean(0), t - t.mean(0)
        n, d_out = Z.size(0), t.size(1)
        A, Bm = Z.T @ Z / n, Z.T @ t / n
        A += ridge * torch.diagonal(A).mean() * torch.eye(H * dk, dtype=A.dtype, device=A.device)

        open_ = self.open_mask
        blk = torch.arange(H * dk, device=Z.device).view(H, dk)
        if self.head_dropout > 0:
            self._probe_under_damage(A, Bm, blk, d_out)
            return
        S = blk[open_].reshape(-1)
        M = torch.linalg.inv(A[S][:, S])
        W = M @ Bm[S]                                                 # re-fitted read-out
        value = torch.zeros(H, dtype=A.dtype, device=A.device)

        ablate = torch.zeros(H, dtype=A.dtype, device=A.device)
        heads_open = torch.nonzero(open_).flatten()
        for i, h in enumerate(heads_open.tolist()):                  # fit lost without h
            J = slice(i * dk, (i + 1) * dk)
            value[h] = torch.trace(W[J].T @ torch.linalg.solve(M[J, J], W[J])) / d_out
            # CONTROL: remove h and keep everyone else's read-out as it is. At the fitted
            # optimum the loss rises by exactly W_J' A_JJ W_J, the head's own contribution
            Jg = blk[h]
            ablate[h] = torch.trace(W[J].T @ A[Jg][:, Jg] @ W[J]) / d_out
        self.head_ablate.copy_(ablate.float())

        shut = torch.nonzero(~open_).flatten()
        if len(shut):                                                 # fit gained with h
            Jall = blk[shut].reshape(-1)
            C = A[Jall][:, S] @ M
            R = Bm[Jall] - C @ Bm[S]
            Sc = A[Jall][:, Jall] - C @ A[S][:, Jall]
            for i, h in enumerate(shut.tolist()):
                J = slice(i * dk, (i + 1) * dk)
                value[h] = torch.trace(R[J].T @ torch.linalg.solve(Sc[J, J], R[J])) / d_out
        self.head_value.copy_(value.float())

    @torch.no_grad()
    def _refit_values(self, A, Bm, blk, working, d_out):
        """Collateral values when only `working` heads carry flow: for a working head, the
        fit lost without it (others re-fit); for any other head, the fit gained by adding
        it. Same algebra as probe_ischemia, allowing an empty working set."""
        H, dk = self.H, self.d_k
        value = torch.zeros(H, dtype=A.dtype, device=A.device)
        on = torch.nonzero(working).flatten().tolist()
        off = torch.nonzero(~working).flatten().tolist()
        if not on:                                   # nothing working: gain of each head alone
            for h in off:
                J = blk[h]
                value[h] = torch.trace(Bm[J].T @ torch.linalg.solve(A[J][:, J], Bm[J])) / d_out
            return value
        S = blk[working].reshape(-1)
        M = torch.linalg.inv(A[S][:, S])
        W = M @ Bm[S]
        for i, h in enumerate(on):
            J = slice(i * dk, (i + 1) * dk)
            value[h] = torch.trace(W[J].T @ torch.linalg.solve(M[J, J], W[J])) / d_out
        if off:
            Jall = blk[off].reshape(-1)
            C = A[Jall][:, S] @ M
            R = Bm[Jall] - C @ Bm[S]
            Sc = A[Jall][:, Jall] - C @ A[S][:, Jall]
            for i, h in enumerate(off):
                J = slice(i * dk, (i + 1) * dk)
                value[h] = torch.trace(R[J].T @ torch.linalg.solve(Sc[J, J], R[J])) / d_out
        return value

    @torch.no_grad()
    def _probe_under_damage(self, A, Bm, blk, d_out):
        """COLLATERAL VALUE UNDER DAMAGE. Tissue keeps collateral vessels where occlusions
        actually happen. Average each head's collateral value over random failure patterns
        (each head fails with probability head_dropout): a head that has failed is worth
        nothing in that pattern; a backup is worth what it saves when its partner fails."""
        total = torch.zeros(self.H, dtype=A.dtype, device=A.device)
        for _ in range(self.probe_masks):
            alive = (torch.rand(self.H, generator=self._damage) >= self.head_dropout).to(A.device)
            v = self._refit_values(A, Bm, blk, self.open_mask & alive, d_out)
            total += torch.where(alive, v, torch.zeros_like(v))
        self.head_value.copy_((total / self.probe_masks).float())
        self.head_ablate.zero_()

    def l0_gate(self, sample=True):
        """Hard-concrete gate per head: stochastic in training, deterministic otherwise.
        A head whose deterministic gate is exactly 0 is pruned."""
        beta, gamma, zeta = self._l0
        if sample:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (zeta - gamma) + gamma).clamp(0.0, 1.0)

    def l0_penalty(self):
        """Expected number of heads with a nonzero gate."""
        beta, gamma, zeta = self._l0
        return torch.sigmoid(self.log_alpha - beta * math.log(-gamma / zeta)).sum()

    @torch.no_grad()
    def local_step(self, price):
        """Each head must earn its keep. Close the cheapest open head if it is worth less
        than one head's price; otherwise reopen the most valuable starved head if it is
        worth more than twice that. The factor of two is hysteresis, so a head at the
        margin does not flicker. One change per probe lets the circuit re-form between."""
        v, open_ = self.head_value, self.open_mask
        if int(open_.sum()) > 1:
            # the controls change only WHICH value decides the closure (reopening below is
            # always the re-fit gain): ablate swaps in the standard score, random keeps the
            # rule's timing but picks the head blind
            score = self.head_ablate if self.local_value == "ablate" else v
            cheapest = torch.where(open_, score, torch.full_like(score, float("inf")))
            h = int(cheapest.argmin())
            if float(cheapest[h]) < price:
                if self.local_value == "random":
                    idx = torch.nonzero(open_).flatten().cpu()
                    h = int(idx[torch.randint(len(idx), (1,), generator=self._pick)])
                open_[h] = False
                return
        best = torch.where(~open_, v, torch.full_like(v, -float("inf")))
        h = int(best.argmax())
        if float(best[h]) > 2 * price:
            open_[h] = True

    @torch.no_grad()
    def relax_tone(self):
        """GRADED ISCHEMIA. Vessels constrict over time rather than shutting. Each step a
        head's tone moves 1/taper toward open (1) or shut (0); taper = 0 is instantaneous.
        Supply stays conserved throughout, so the survivors gain exactly what the closing
        head loses, and the network can adapt while it happens instead of after."""
        target = self.open_mask.to(self.tone.dtype)
        if self.taper <= 0:
            self.tone.copy_(target)
        else:
            speed = torch.full_like(self.tone, 1.0 / self.taper)
            if self.trial_head is not None:                # a trial closure fades slowly
                speed[self.trial_head] = 1.0 / max(1, self.trial_taper)
            self.tone.add_(torch.maximum(torch.minimum(target - self.tone, speed), -speed))

    @torch.no_grad()
    def prune_step(self, loss, bar, kstar, stop=True, patience=4):
        """BASELINE. Iterative head pruning as in Michel et al. (2019): remove the open head
        with the smallest |dL/dmask|, one per call. With stop, prune only while the loss
        meets the bar, and if a removal leaves it above the bar for `patience` calls, put
        that head back and stop for good. Without stop, prune to the true k* (an oracle)."""
        if self._prune_done:
            return
        open_ = self.open_mask
        if stop and loss > bar:
            if self._removed is not None:
                self._since_removal += 1
                if self._since_removal >= patience:
                    open_[self._removed] = True
                    self._prune_done = True
            return
        if not stop and int(open_.sum()) <= kstar:
            self._prune_done = True
            return
        if int(open_.sum()) <= 1:
            return
        imp = torch.where(open_, self.michel, torch.full_like(self.michel, float("inf")))
        h = int(imp.argmin())
        open_[h] = False
        self._removed, self._since_removal = h, 0

    def gate_threshold(self, d, kappa, total=None):
        """theta = mean + kappa*std. Perfused set, and therefore B, emerges."""
        total = total if total is not None else self.H
        theta = d.mean() + kappa * d.std().clamp_min(1e-6)
        return self._from_active(d, d > theta, total)

    def gate(self, kappa=None, B_active=None):
        d = self.sensed()
        if self.supply_kind == "topk":
            return self.gate_topk(d, B_active if B_active is not None else self.H)
        if self.supply_kind == "threshold":
            return self.gate_threshold(d, 0.0 if kappa is None else kappa)
        if self.supply_kind == "marginal":
            return self.gate_marginal()
        if self.supply_kind == "poiseuille":
            return self.gate_poiseuille(d, 0.0 if kappa is None else kappa)
        if self.supply_kind == "watershed":
            return self.gate_watershed(d, 0.0 if kappa is None else kappa)
        if self.supply_kind == "watershed_auto":
            return self.gate_watershed(d, float(self.kappa_state))
        if self.supply_kind == "gap":
            return self.gate_gap(d)
        if self.supply_kind == "otsu":
            return self.gate_otsu(d)
        if self.supply_kind == "autoreg":
            return self.gate_threshold(d, float(self.kappa_state))
        if self.supply_kind == "local":
            if not self.conserve:
                return self.tone.clone()            # CONTROL: no fixed total
            return self.H * self.tone / self.tone.sum()
        if self.supply_kind == "prune":
            return self.open_mask.to(d.dtype)       # a 0/1 mask: pruning does NOT conserve
        if self.supply_kind == "l0":
            return self.l0_gate(sample=self.training)
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
                    kl = local_kappa(0.0 if kappa is None else kappa,
                                     sub.numel(), self.H)
                    theta = sub.mean() + kl * sub.std().clamp_min(1e-6)
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
            if self.needs_gate_grad() and torch.is_grad_enabled():
                # a differentiable leaf, so dL/dg_h can be read after backward. The
                # gradient is well defined even where g_h == 0, which is what lets a
                # STARVED head still report the flow it would have used.
                gate = gate.detach().requires_grad_(True)
                self._gate_leaf = gate
            gate = gate.unsqueeze(0).expand(X.size(0), -1)
        used = gate
        if self.head_dropout > 0 and self.training:
            # DAMAGE: every head fails independently this step. The returned gate is the
            # allocation (what the ledger records); only the computation sees the failure
            keep = (torch.rand(self.H, device=gate.device) >= self.head_dropout).to(gate.dtype)
            used = gate * keep / (1 - self.head_dropout)
        return self.combine(out, used), gate
