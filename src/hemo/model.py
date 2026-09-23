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

    # ---------------- demand ----------------
    def needs_gate_grad(self):
        return self.demand_kind == "deficit" or self.supply_kind == "marginal"

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

    def autoregulate(self, loss, target, step=0):
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
        """
        import math
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
        self.kappa_state += self.autoreg_gain * max(-4.0, min(4.0, err))
        self.kappa_state.clamp_(self.kappa_floor, self.kappa_ceil)
        return float(self.kappa_state)

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
        return self.combine(out, gate), gate
