import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .model import CrossAttn, HemoAttn
from .tasks import make_batch, trivial_loss, predicted_kstar


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
    # autoreg only: perfuse until the task is solved to within target_frac of the
    # trivial baseline. Expressed against trivial, so the target means the same thing
    # at every R rather than being a raw loss number tuned per task.
    autoreg = hemo and cfg.supply in ("autoreg", "watershed_auto")
    target = cfg.target_frac * trivial_loss(val)
    hold = int(cfg.budget_hold_frac * total)
    # local supply and the pruning baseline decide every probe_every steps after phase 1
    local = hemo and cfg.supply == "local"
    prune = hemo and cfg.supply == "prune"
    price = cfg.price_frac * trivial_loss(val)
    smooth, h["n_probes"] = None, 0
    quiet, cooldown, trial_start, h["trials"] = 0, {}, 0, []

    for step in range(total):
        X, Y, T, _ = make_batch(cfg.batch_size, cfg, device)
        if hemo:
            kappa, B = schedule(step, cfg, H, total)
            # The budget ladder belongs to topk supply (see Cfg: "budget schedule, topk
            # only"). Passing it to territory supply sent it down the local top-k branch
            # with a global budget that anneals to budget_min, which forces exactly
            # max(1, round(budget_min/T)) = 1 head per territory: B is then pinned to T
            # by construction and nothing about it is emergent.
            pred, g = model(X, Y, kappa=kappa,
                            B_active=B if cfg.supply == "topk" else None)
        else:
            kappa, B = None, H
            pred, g = model(X, Y)
        loss = F.mse_loss(pred, T)
        opt.zero_grad(set_to_none=True)
        if hemo and cfg.supply == "l0" and step >= hold:
            (loss + cfg.l0_lambda * model.l0_penalty()).backward()   # BASELINE: L0 penalty
        else:
            loss.backward()
        if hemo and model.needs_reserve_probe() and step % model.cvr_every == 0:
            model.probe_reserve(X, Y, T, lambda a, b: F.mse_loss(a, b))
        if hemo and model.needs_gate_grad():
            model.absorb_gradient()          # the deficit lags by one step, as it should
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if autoreg and step >= hold:      # phase 1 stays fully perfused, then the loop
            model.autoregulate(loss.item(), target, step=step - hold,
                               n_perfused=int((g[0] > model.leak + 1e-9).sum()))
        if local:
            model.relax_tone()
        if (local or prune) and step >= hold:
            smooth = loss.item() if smooth is None else 0.92 * smooth + 0.08 * loss.item()
            if local and model.trial_head is not None and smooth > cfg.trial_undo * target:
                h_t = model.trial_head                         # undo early, every step
                model.open_mask[h_t] = True
                model.tone[h_t] = 1.0
                cooldown[h_t] = h["n_probes"] + cfg.trial_cooldown
                h["trials"].append((h_t, trial_start, step, False))
                model.trial_head, quiet = None, 0
            if (step - hold) % cfg.probe_every == 0:
                if local and model.trial_head is not None:
                    # TRIAL CLOSURE under way: undo at once if the loss leaves the solved bar,
                    # keep the head shut if the task stays solved through fade and settling
                    h_t, t0 = model.trial_head, trial_start
                    if smooth > cfg.trial_undo * target:
                        model.open_mask[h_t] = True
                        model.tone[h_t] = 1.0
                        cooldown[h_t] = h["n_probes"] + cfg.trial_cooldown
                        h["trials"].append((h_t, t0, step, False))
                        model.trial_head, quiet = None, 0
                    elif step - t0 >= cfg.trial_taper + cfg.trial_settle:
                        h["trials"].append((h_t, t0, step, True))
                        model.trial_head, quiet = None, 0
                elif local:
                    before = model.open_mask.clone()
                    model.probe_ischemia(*make_batch(cfg.probe_batch, cfg, device)[:3])
                    h["n_probes"] += 1
                    model.local_step(price)
                    quiet = quiet + 1 if torch.equal(before, model.open_mask) else 0
                    n_fail = sum(1 for tr in h["trials"] if not tr[3])
                    if (cfg.trial and quiet >= cfg.trial_wait and int(model.open_mask.sum()) > 1
                            and n_fail < cfg.trial_max_fail and step < cfg.trial_stop * total):
                        cand = [i for i in range(H) if model.open_mask[i]
                                and cooldown.get(i, 0) <= h["n_probes"]]
                        if cand:
                            h_t = min(cand, key=lambda i: float(model.head_value[i]))
                            model.open_mask[h_t] = False
                            model.trial_head, trial_start, quiet = h_t, step, 0
                else:
                    model.prune_step(smooth, target, predicted_kstar(cfg),
                                     stop=bool(cfg.prune_stop))

        nper = int((g[0] > model.leak + 1e-9).sum()) if hemo else H
        if hemo:
            if cfg.supply == "l0":                       # record the deterministic gates
                h["ledger"].append(model.l0_gate(sample=False).detach().cpu().numpy())
            else:
                h["ledger"].append(g[0].detach().cpu().numpy())
        # always measure just before the perfused set changes, so every level is recorded
        nxt = schedule(step + 1, cfg, H, total)[1] if hemo else H
        if (step % cfg.val_every == 0 or step == total - 1
                or (hemo and (nxt != B or nper != prev))):
            prev = nper
            kw = (dict(kappa=kappa, update=False,
                       B_active=B if cfg.supply == "topk" else None)
                  if hemo else {})
            h["val_step"].append(step)
            h["val_loss"].append(evaluate(model, val, **kw))
            h["budget"].append(B)
            h["kappa"].append(float(model.kappa_state) if autoreg else kappa)
            h["n_perfused"].append(nper)
            if verbose:
                print(f"    {desc} step {step:>5} loss {h['val_loss'][-1]:.5f} "
                      f"perfused {nper}/{H}")
    h["ledger"] = np.array(h["ledger"]) if hemo else None
    h["final_perfused"] = prev if hemo else H
    h["final_gate"] = (model.l0_gate(sample=False) if hemo and cfg.supply == "l0" else g[0]).detach().cpu().numpy()
    h["final_kappa"], h["final_budget"] = kappa, B
    return model, h
