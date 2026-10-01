"""Two-layer induction: a benchmark whose circuit is known and needs heads in different
layers to cooperate (Elhage et al. 2021; Olsson et al. 2022).

Task. A sequence is a random prefix of P tokens, a segment A of L distinct random tokens,
a gap of G random tokens, then A again, then random padding (P and G drawn per sequence,
filler never from A), so neither copy sits at a fixed position. Inside the second copy the next token is predictable only by
induction: find the earlier copy of the current token and copy the token that followed it.
Because G changes per sequence, looking back a fixed number of positions does not work
(without the gap one layer solves it by position alone; checked). Minimal circuit, two heads:

  layer 1  previous-token head   position t attends to t - 1 (tags each token with its
                                 predecessor)
  layer 2  induction head        position t attends to the token after the earlier copy of
                                 the current token
A first layer can also serve induction as a duplicate-token head (attend to the earlier
copy of the current token and pass on its position); role_scores measures both.

Model. Attention only, as in Elhage et al.: token + learned position embedding, two causal
self-attention layers added into the residual stream, a linear unembedding. Every head has
a supply gain g_h multiplying its contribution, so a starved head is removed exactly.

Rules (which head to close; all start with every head on):
  dense        no gating
  collateral   value = rise in readout error when the head is removed, everything after it
               is recomputed, and the final readout is re-fitted (least squares on the
               residual stream to the one-hot next token). Others cover what they can.
  ablate       CONTROL, ordinary importance: the same removal with the readout NOT re-fitted
  random       CONTROL: close when the collateral rule would, but pick the head at random
  prune        BASELINE, Michel et al. 2019: remove the head with the smallest |dL/dg_h|
               while the loss meets the solved bar; restore the last one and stop when not
"""
import math
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ICfg:
    vocab: int = 64
    L: int = 8                      # length of the repeated segment
    gap_max: int = 16               # random gap between the copies, 0..gap_max
    pre_max: int = 8                # random prefix before the first copy, 0..pre_max
    d_model: int = 64
    d_head: int = 16
    heads: int = 8                  # per layer
    heads1: int = -1                # dense ground truth: heads in layer 1 (-1 = heads)
    heads2: int = -1                # dense ground truth: heads in layer 2
    rule: str = "collateral"        # dense | collateral | ablate | random | prune
    steps: int = 3000
    batch: int = 128
    lr: float = 1e-3
    price_frac: float = 0.03        # price of a head, fraction of the trivial readout error
    probe: str = "mse"              # mse: linear read-out of the one-hot next token (first batch)
                                    # logit: re-fit to reproduce the logits, value in nats
    trial: int = 0                  # 1 = TRIAL CLOSURE: once the rule has been quiet for trial_wait
                                    # probes, close the cheapest open head even if it looks
                                    # essential, fading it over trial_taper steps; reopen it at
                                    # once if the real loss leaves the solved bar, keep it shut if
                                    # the task stays solved for trial_settle more steps
    trial_taper: int = 300
    trial_settle: int = 200
    trial_wait: int = 4
    trial_cooldown: int = 20        # probes before a head whose trial failed is tried again
    squeeze_at: float = 0.0         # TEST: at this fraction of training force one more layer-1
                                    # head shut (0 = never); see train()
    price_mode: str = "absolute"    # absolute: close if value < price_frac * trivial error
                                    # relative: close if value < price_frac * most valuable head
                                    # bar: close if the error without the head (others re-fit)
                                    #      stays <= price_frac * trivial error
    taper: int = 100                # steps over which a closing head fades out
    start_frac: float = 0.25        # supply starts moving after this fraction of training
    probe_every: int = 25
    probe_batch: int = 128
    bar_frac: float = 0.02          # solved: loss <= bar_frac * ln(vocab)
    prune_patience: int = 4
    val_size: int = 512
    val_every: int = 50
    seed: int = 0
    device: str = "auto"


# ------------------------------------------------------------------ task
def seq_len(cfg):
    return cfg.pre_max + 2 * cfg.L + cfg.gap_max


def make_batch(n, cfg, gen=None, device="cpu"):
    """Returns tokens (n, T) and starts (n, 2): where the first and second copy begin.
    Layout: P prefix tokens | A (L distinct tokens) | G gap tokens | A | padding, with P and
    G drawn per sequence, so neither copy sits at a fixed position. Filler never uses A."""
    kw = {"generator": gen} if gen is not None else {}
    T, L = seq_len(cfg), cfg.L
    order = torch.argsort(torch.rand(n, cfg.vocab, **kw), dim=1)
    A, rest = order[:, :L], order[:, L:]                    # rest never contains a token of A
    tok = torch.gather(rest, 1, torch.randint(0, cfg.vocab - L, (n, T), **kw))
    P = torch.randint(0, cfg.pre_max + 1, (n,), **kw)
    G = torch.randint(0, cfg.gap_max + 1, (n,), **kw)
    first, second = P, P + L + G
    for s in (first, second):
        tok.scatter_(1, s[:, None] + torch.arange(L)[None], A)
    return tok.to(device), torch.stack([first, second], 1).to(device)


def predict_mask(starts, cfg):
    """(n, T) True where the next token is fixed by induction: the first L - 1 positions of
    the second copy (their next token is also inside the copy)."""
    second = starts[:, 1]
    t = torch.arange(seq_len(cfg), device=starts.device)[None]
    return (t >= second[:, None]) & (t < second[:, None] + cfg.L - 1)




def trivial_loss(cfg):
    return math.log(cfg.vocab)


# ------------------------------------------------------------------ model
class AttnLayer(nn.Module):
    def __init__(self, d_model, d_head, H):
        super().__init__()
        self.H, self.d_head = H, d_head
        self.W_q = nn.Linear(d_model, H * d_head, bias=False)
        self.W_k = nn.Linear(d_model, H * d_head, bias=False)
        self.W_v = nn.Linear(d_model, H * d_head, bias=False)
        self.W_o = nn.Parameter(torch.randn(H, d_head, d_model) / math.sqrt(H * d_head))

    def forward(self, x):
        """Per-head contributions to the residual stream (B, H, T, d_model), and attention."""
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.H, self.d_head).transpose(1, 2)
        q, k, v = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        attn = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        return torch.einsum("bhtd,hdm->bhtm", attn @ v, self.W_o), attn


class InductionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h1 = cfg.heads if cfg.heads1 < 0 else cfg.heads1
        h2 = cfg.heads if cfg.heads2 < 0 else cfg.heads2
        self.sizes = [h1, h2]
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(seq_len(cfg), cfg.d_model) * 0.1)
        self.layers = nn.ModuleList([AttnLayer(cfg.d_model, cfg.d_head, h) for h in self.sizes if h > 0])
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab)
        self.H = h1 + h2

    def residual(self, tokens, gate=None, keep_attn=False):
        """Final residual stream. gate: (H,) gains for all heads, layer 1 first."""
        x = self.embed(tokens) + self.pos[: tokens.size(1)]
        gate = torch.ones(self.H, device=tokens.device) if gate is None else gate
        attns, i = [], 0
        for layer in self.layers:
            out, attn = layer(x)
            x = x + (out * gate[i:i + layer.H, None, None]).sum(1)
            attns.append(attn if keep_attn else None)
            i += layer.H
        return x, attns

    def forward(self, tokens, gate=None):
        return self.unembed(self.residual(tokens, gate)[0])


def lm_loss(logits, tokens, second, cfg):
    """Cross-entropy of the next token, only where induction fixes it."""
    m = predict_mask(second, cfg)
    nxt = torch.roll(tokens, -1, dims=1)
    return F.cross_entropy(logits[m], nxt[m])


# ------------------------------------------------------------------ roles
@torch.no_grad()
def role_scores(model, cfg, device, n=256):
    """Per layer, per head, three attention scores, each with its chance level (causal
    attention spread evenly), so a role counts only well above chance.
    prev: from t to t - 1 over the FIRST copy (after its first token), where each token must
          be tagged with its predecessor for the induction head to find it later.
    dup:  from t in the second copy to the earlier copy of the SAME token (the other way a
          first layer can serve induction: find the earlier copy, pass on its position).
    ind:  from t in the second copy to the token AFTER the earlier copy of the current token.
    Returns (scores, chance) with scores[name] a list over layers of per-head arrays."""
    tok, starts = make_batch(n, cfg, torch.Generator().manual_seed(4242), device)
    _, attns = model.residual(tok, keep_attn=True)
    k = torch.arange(1, cfg.L, device=device)
    bf, tf = torch.arange(n, device=device).repeat_interleave(cfg.L - 1), (starts[:, :1] + k).reshape(-1)
    b, t = torch.nonzero(predict_mask(starts, cfg), as_tuple=True)
    earlier = t - (starts[b, 1] - starts[b, 0])              # earlier copy of the same token
    scores = dict(prev=[], dup=[], ind=[])
    for a in attns:                                          # a: (n, H, T, T)
        scores["prev"].append(a[bf, :, tf, tf - 1].mean(0).cpu().numpy())
        scores["dup"].append(a[b, :, t, earlier].mean(0).cpu().numpy())
        scores["ind"].append(a[b, :, t, earlier + 1].mean(0).cpu().numpy())
    chance = dict(prev=float((1.0 / (tf + 1).float()).mean()),
                  dup=float((1.0 / (t + 1).float()).mean()), ind=float((1.0 / (t + 1).float()).mean()))
    return scores, chance


# ------------------------------------------------------------------ collateral values
@torch.no_grad()
def probe_values(model, tokens, second, cfg, open_mask, ridge=1e-3):
    """For every head: the collateral value (readout re-fitted) and the ordinary importance
    (readout kept). Open heads: rise in readout error when removed. Starved heads: fall in
    readout error when added (re-fitted). Units: mean squared error per output."""
    m = predict_mask(second, cfg)
    nxt = torch.roll(tokens, -1, dims=1)
    Y = F.one_hot(nxt[m], cfg.vocab).double()
    Y = Y - Y.mean(0)

    def features(gate):
        z = model.residual(tokens, gate)[0][m].double()
        return z - z.mean(0)

    def fit(Z):
        A = Z.T @ Z / len(Z)
        A += ridge * torch.diagonal(A).mean() * torch.eye(A.size(0), dtype=A.dtype, device=A.device)
        return torch.linalg.solve(A, Z.T @ Y / len(Z))

    err = lambda Z, W: float(((Y - Z @ W) ** 2).mean())
    base = open_mask.float()
    Z0 = features(base)
    W0 = fit(Z0)
    E0 = err(Z0, W0)
    value, ablate = np.zeros(model.H), np.zeros(model.H)
    for h in range(model.H):
        g = base.clone()
        g[h] = 1.0 - g[h]
        Z = features(g)
        if open_mask[h]:
            value[h] = err(Z, fit(Z)) - E0
            ablate[h] = err(Z, W0) - E0
        else:
            value[h] = E0 - err(Z, fit(Z))
    return value, ablate, float((Y ** 2).mean()), E0


@torch.no_grad()
def probe_values_logit(model, tokens, second, cfg, open_mask, ridge=1e-4, tone=None):
    """Collateral values in real loss units (nats). For an open head h: remove it, recompute
    everything after it, re-fit a linear read-out (least squares) so the remaining residual
    stream reproduces the model's current logits as closely as it can, and measure the
    cross-entropy of those re-fitted logits against the true next tokens. Value = that loss
    minus the current loss. Ordinary importance (ablate) = the loss with the head removed and
    the unembedding left as it is. Starved heads get value 0 here; reopening restores the
    most recently closed head. Returns (value, ablate, trivial loss, current loss).
    With tone given (probe="logit_tone"), the model is measured as it actually runs, with a
    fading head at its current partial gain, instead of as if every closed head were gone."""
    m = predict_mask(second, cfg)
    nxt = torch.roll(tokens, -1, dims=1)[m]
    base = open_mask.float() if tone is None else tone.clone()
    z0 = model.residual(tokens, base)[0][m]
    target = model.unembed(z0).double()                          # the logits to reproduce
    E0 = float(F.cross_entropy(target, nxt))

    def refit_loss(z):
        Z = torch.cat([z.double(), torch.ones(len(z), 1, dtype=torch.float64, device=z.device)], 1)
        A = Z.T @ Z / len(Z)
        A += ridge * torch.diagonal(A).mean() * torch.eye(A.size(0), dtype=A.dtype, device=A.device)
        W = torch.linalg.solve(A, Z.T @ target / len(Z))
        return float(F.cross_entropy(Z @ W, nxt))

    value, ablate = np.zeros(model.H), np.zeros(model.H)
    for h in torch.nonzero(open_mask).flatten().tolist():
        g = base.clone()
        g[h] = 0.0
        z = model.residual(tokens, g)[0][m]
        value[h] = refit_loss(z) - E0
        ablate[h] = float(F.cross_entropy(model.unembed(z), nxt)) - E0
    return value, ablate, trivial_loss(cfg), E0


# ------------------------------------------------------------------ training
def pick_device(cfg):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, val, cfg, gate=None):
    tok, second = val
    logits = model(tok, gate)
    m = predict_mask(second, cfg)
    acc = (logits.argmax(-1) == torch.roll(tok, -1, dims=1))[m].float().mean().item()
    return lm_loss(logits, tok, second, cfg).item(), acc


def train(cfg, verbose=False):
    torch.manual_seed(cfg.seed)
    dev = pick_device(cfg)
    model = InductionNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    val = make_batch(cfg.val_size, cfg, torch.Generator().manual_seed(10_000), dev)
    H, bar = model.H, cfg.bar_frac * trivial_loss(cfg)
    open_mask = torch.ones(H, dtype=torch.bool, device=dev)
    tone = torch.ones(H, device=dev)
    michel = torch.zeros(H, device=dev)
    pick = torch.Generator().manual_seed(cfg.seed + 11)
    start = int(cfg.start_frac * cfg.steps)
    hist = dict(val_step=[], val_loss=[], val_acc=[], ledger=[], n_probes=0)
    smooth, removed, since, done = None, None, 0, False
    closed_order, value = [], None
    trial, quiet, cooldown = None, 0, {}
    hist["trials"] = []
    assert not cfg.trial or cfg.probe.startswith("logit"), "trial closure needs the loss-based probe"
    gated = cfg.rule in ("collateral", "ablate", "random")

    for step in range(cfg.steps):
        tok, second = make_batch(cfg.batch, cfg, device=dev)
        gate = tone.clone() if cfg.rule != "dense" else torch.ones(H, device=dev)
        if cfg.rule == "prune":
            gate = open_mask.float().clone().requires_grad_(True)
        loss = lm_loss(model(tok, gate), tok, second, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.rule == "prune":
            michel.mul_(0.9).add_(0.1 * gate.grad.abs())
        opt.step()
        hist["ledger"].append(gate.detach().cpu().numpy().copy())
        smooth = loss.item() if smooth is None else 0.92 * smooth + 0.08 * loss.item()

        if gated and step >= start and (step - start) % cfg.probe_every == 0:
            ptok, psec = make_batch(cfg.probe_batch, cfg, device=dev)
            if cfg.probe == "logit_tone":
                value, abl, e_triv, e_now = probe_values_logit(model, ptok, psec, cfg, open_mask, tone=tone)
            else:
                probe = probe_values_logit if cfg.probe == "logit" else probe_values
                value, abl, e_triv, e_now = probe(model, ptok, psec, cfg, open_mask)
            hist["n_probes"] += 1
            n_probe = hist["n_probes"]
            if trial is not None:
                # TRIAL CLOSURE under way (collaterals need time to take over a territory):
                # undo at once if the real loss leaves the solved bar; keep it shut if the task
                # stays solved through the slow fade and a settling period
                h_t, t0 = trial
                if e_now > bar:
                    open_mask[h_t] = True
                    tone[h_t] = 1.0
                    cooldown[h_t] = n_probe + cfg.trial_cooldown
                    hist["trials"].append((h_t, t0, step, False))
                    trial, quiet = None, 0
                elif step - t0 >= cfg.trial_taper + cfg.trial_settle:
                    hist["trials"].append((h_t, t0, step, True))
                    trial, quiet = None, 0
            else:
                score = abl if cfg.rule == "ablate" else value
                opened = open_mask.cpu().numpy()
                cheapest = np.where(opened, score, np.inf)
                h = int(cheapest.argmin())
                best = np.where(~opened, value, -np.inf)
                if cfg.price_mode == "absolute":            # a fixed price per head
                    price = cfg.price_frac * e_triv
                    close, reopen = cheapest[h] < price, best.max() > 2 * price
                elif cfg.price_mode == "relative":          # cheap relative to the most valuable head
                    price = cfg.price_frac * float(np.max(np.where(opened, value, -np.inf)))
                    close, reopen = cheapest[h] < price, best.max() > 2 * price
                else:                                       # "bar": the others must keep it solved
                    bar_e = cfg.price_frac * e_triv
                    close, reopen = e_now + cheapest[h] <= bar_e, e_now > 2 * bar_e
                if cfg.probe.startswith("logit"):           # starved heads are not valued: reopen the
                    reopen = e_now > 2 * bar                # last one closed if the loss is too high
                quiet += 1
                if opened.sum() > 1 and close:
                    if cfg.rule == "random":
                        idx = np.nonzero(opened)[0]
                        h = int(idx[torch.randint(len(idx), (1,), generator=pick)])
                    open_mask[h] = False
                    closed_order.append(h)
                    quiet = 0
                elif reopen and (~opened).any():
                    back = closed_order.pop() if (cfg.probe.startswith("logit") and closed_order) else int(best.argmax())
                    open_mask[back] = True
                    quiet = 0
                if cfg.trial and quiet >= cfg.trial_wait and int(open_mask.sum()) > 1:
                    cand = [i for i in range(H) if open_mask[i] and cooldown.get(i, 0) <= n_probe]
                    if cand:
                        h_t = min(cand, key=lambda i: value[i])
                        open_mask[h_t] = False
                        trial, quiet = (h_t, step), 0
        if cfg.squeeze_at > 0 and step == int(cfg.squeeze_at * cfg.steps):
            # TEST, not part of the rule: force one more layer-1 head shut (the one the last
            # probe valued least) and let training continue, to see whether the network can
            # re-route through the remaining head. Bar-mode reopening can still undo it.
            h1 = model.sizes[0]
            on1 = [i for i in range(h1) if open_mask[i]]
            if len(on1) > 1 and value is not None:
                h = min(on1, key=lambda i: value[i])
                open_mask[h] = False
                closed_order.append(h)
                hist["squeezed"] = int(h)
        if cfg.rule == "prune" and step >= start and not done and (step - start) % cfg.probe_every == 0:
            if smooth > bar:
                if removed is not None:
                    since += 1
                    if since >= cfg.prune_patience:
                        open_mask[removed] = True
                        done = True
            elif int(open_mask.sum()) > 1:
                imp = torch.where(open_mask, michel, torch.full_like(michel, float("inf")))
                removed, since = int(imp.argmin()), 0
                open_mask[removed] = False
        if gated:
            speed = torch.full((H,), 1.0 / cfg.taper if cfg.taper > 0 else 1.0, device=dev)
            if trial is not None:
                speed[trial[0]] = 1.0 / cfg.trial_taper           # a trial head fades slowly
            tone.add_(torch.maximum(torch.minimum(open_mask.float() - tone, speed), -speed))
        if cfg.rule == "prune":
            tone = open_mask.float()

        if step % cfg.val_every == 0 or step == cfg.steps - 1:
            vl, va = evaluate(model, val, cfg, None if cfg.rule == "dense" else tone)
            hist["val_step"].append(step)
            hist["val_loss"].append(vl)
            hist["val_acc"].append(va)
            if verbose:
                print(f"  step {step:>5} loss {vl:.4f} acc {va:.3f} heads on {int((tone > 0).sum())}")

    hist["ledger"] = np.array(hist["ledger"])
    final_loss, final_acc = evaluate(model, val, cfg, None if cfg.rule == "dense" else tone)
    roles, chance = role_scores(model, cfg, dev)
    kept = (hist["ledger"][-1] > 0) if cfg.rule != "dense" else np.ones(H, dtype=bool)
    heads_on = (hist["ledger"] > 0).sum(1)
    probe_cost = hist["n_probes"] * (H + 1) * cfg.probe_batch / (3 * cfg.batch)   # forward-only
    out = dict(cfg=asdict(cfg), hist=hist, final_loss=final_loss, final_acc=final_acc,
               trivial=trivial_loss(cfg), bar=bar, kept=kept, sizes=model.sizes,
               roles=roles, chance=chance,
               compute=float(heads_on.mean() / H + probe_cost / cfg.steps))
    return model, out
