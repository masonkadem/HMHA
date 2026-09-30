"""The benchmark's ground truth must hold or nothing downstream means anything.
Run: pytest -q tests/  (about 3 min on CPU)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataclasses import replace
import numpy as np
import torch

from hemo.config import Cfg
from hemo.tasks import make_val, trivial_loss, offsets_for, predicted_kstar
from hemo.train import train, pick_device
from hemo.analysis import head_offsets

SMALL = Cfg(task="multi_relation", seq_len=8, n_rel=2, m_content=8, d_model=32,
            num_heads=8, d_k=32, steps=2500, batch_size=128, val_size=512,
            val_every=2500, lr=2e-3, device="cpu")


def test_one_head_fails_two_heads_succeed():
    """k* == R by construction. One head cannot implement two offsets."""
    dev = pick_device(SMALL)
    val = make_val(SMALL, dev)
    triv = trivial_loss(val)
    loss = {}
    for k in [1, 2]:
        _, h = train(SMALL, val, dev, hemo=False, num_heads=k)
        loss[k] = h["val_loss"][-1]
    assert loss[1] > 0.2 * triv, f"1 head should fail, got {loss[1]:.4f} vs trivial {triv:.4f}"
    assert loss[2] < 0.01 * triv, f"2 heads should solve it, got {loss[2]:.4f}"


def test_heads_implement_distinct_offsets():
    dev = pick_device(SMALL)
    val = make_val(SMALL, dev)
    m, _ = train(SMALL, val, dev, hemo=False, num_heads=2)
    prof = head_offsets(m, val, SMALL, dev)
    assert set(prof.argmax(1).tolist()) == set(offsets_for(SMALL))


def test_supply_is_conserved():
    cfg = replace(SMALL, steps=20, num_heads=8, n_territories=4)
    dev = pick_device(cfg)
    val = make_val(cfg, dev)
    for supply in ["topk", "threshold", "territory", "gap", "otsu", "autoreg",
                   "poiseuille", "watershed", "watershed_auto", "local"]:
        c = replace(cfg, supply=supply)
        m, _ = train(c, val, dev, hemo=True, steps=20)
        g = m.gate(kappa=1.0, B_active=3)
        assert abs(float(g.sum()) - m.H) < 1e-3, f"{supply} broke conservation: {g.sum()}"


def test_local_rule_decisions():
    """Close the cheapest open head below the price; reopen a starved head only above
    twice the price; one change per call; never close the last head."""
    from hemo.model import HemoAttn
    m = HemoAttn(replace(SMALL, supply="local", num_heads=4))
    m.head_value.copy_(torch.tensor([0.5, 0.05, 0.2, 0.01]))
    m.local_step(price=0.1)
    assert m.open_mask.tolist() == [True, True, True, False]      # cheapest goes first
    m.head_value.copy_(torch.tensor([0.5, 0.5, 0.5, 0.15]))
    m.local_step(price=0.1)
    assert m.open_mask.tolist() == [True, True, True, False]      # 0.15 < 2 x price: stays shut
    m.head_value[3] = 0.25
    m.local_step(price=0.1)
    assert m.open_mask.tolist() == [True, True, True, True]
    m.open_mask.copy_(torch.tensor([True, False, False, False]))
    m.head_value.zero_()
    m.local_step(price=0.1)
    assert int(m.open_mask.sum()) == 1
    assert abs(float(m.gate().sum()) - m.H) < 1e-6


def test_head_value_matches_brute_force_refit():
    """The local rule's per-head deficit is closed-form least squares. It must equal
    re-fitting the read-out with that head removed (open) or added (starved)."""
    from hemo.model import HemoAttn
    from hemo.tasks import make_batch
    torch.manual_seed(0)
    cfg = replace(SMALL, supply="local", num_heads=6, d_k=4)
    m = HemoAttn(cfg)
    X, Y, T, _ = make_batch(256, cfg, "cpu")
    open_ = [0, 1, 3, 5]
    m.open_mask.copy_(torch.tensor([h in open_ for h in range(6)]))
    m.probe_ischemia(X, Y, T, ridge=0.0)
    _, _, out = m.heads(X, Y)
    Z = out.permute(0, 2, 1, 3).reshape(-1, 24).double()
    t = T.reshape(-1, T.size(-1)).double()
    Z, t = Z - Z.mean(0), t - t.mean(0)

    def fit(heads):
        cols = [c for h in heads for c in range(4 * h, 4 * h + 4)]
        W = torch.linalg.lstsq(Z[:, cols], t).solution
        return float(((Z[:, cols] @ W - t) ** 2).mean())

    base = fit(open_)
    for h in range(6):
        brute = (fit([x for x in open_ if x != h]) - base if h in open_
                 else base - fit(open_ + [h]))
        assert abs(float(m.head_value[h]) - brute) < 1e-6, (h, float(m.head_value[h]), brute)


def test_predicted_kstar():
    assert predicted_kstar(SMALL) == SMALL.n_rel


def test_territory_threshold_is_not_pinned_to_T():
    """The territory threshold must be comparable across territory sizes.

    A group of n values has a largest attainable z-score of (n-1)/sqrt(n). With H=32
    and T=8 that cap is exactly 1.50, so a raw kappa of 1.5 makes 'd > mean + kappa*std'
    unsatisfiable, the never-fully-infarct fallback fires in every territory, and the
    perfused count is pinned to exactly T. That reports T back as if it had emerged.
    """
    from hemo.model import HemoAttn, zmax
    cfg = replace(SMALL, num_heads=32, n_territories=8, supply="territory", seed=0)
    m = HemoAttn(cfg)
    n = cfg.num_heads // cfg.n_territories
    assert zmax(n) <= 1.5, "this test is only meaningful while kappa_end exceeds the cap"
    torch.manual_seed(0)
    # demand concentrated in the first head of each territory plus a runner-up
    d = torch.zeros(cfg.num_heads)
    for t in range(cfg.n_territories):
        d[t * n] = 5.0
        d[t * n + 1] = 4.0
    m.demand.copy_(d)
    g = m.gate(kappa=1.5)
    assert abs(float(g.sum()) - m.H) < 1e-3, f"conservation broken: {g.sum()}"
    n_perf = int((g > m.leak + 1e-9).sum())
    assert n_perf > cfg.n_territories, (
        f"perfused count pinned to n_territories ({n_perf}); the local threshold is "
        f"unreachable at kappa=1.5 for {n}-head territories")


def test_a_copy_is_worth_nothing_to_the_collateral_rule():
    """The collateral principle. With planted copies, the re-fit value of every head is
    ~0 (its twin covers for it), while the standard ablation score is not, and the
    no-conservation control keeps open gains at 1 rather than summing to H."""
    from hemo.model import HemoAttn
    from hemo.tasks import make_batch
    c = replace(SMALL, supply="local", num_heads=4, d_k=4, plant_copies=1)
    torch.manual_seed(0)
    m = HemoAttn(c)
    X, Y, T, _ = make_batch(256, c, "cpu")
    _, _, out = m.heads(X, Y)
    assert torch.allclose(out[:, 0], out[:, 2]) and torch.allclose(out[:, 1], out[:, 3])
    m.probe_ischemia(X, Y, T)
    assert float(m.head_value.abs().max()) < 1e-3 * float(m.head_ablate.max())
    m.open_mask[1] = False
    m.conserve = 0
    m.relax_tone()
    assert m.gate().tolist() == [1.0, 0.0, 1.0, 1.0]


def test_local_controls_pick_differently():
    """ablate closes by the ablation score; random keeps the timing but picks blind."""
    from hemo.model import HemoAttn
    m = HemoAttn(replace(SMALL, supply="local", num_heads=4, local_value="ablate"))
    m.head_value.copy_(torch.tensor([0.5, 0.01, 0.5, 0.5]))
    m.head_ablate.copy_(torch.tensor([0.5, 0.9, 0.02, 0.5]))
    m.local_step(price=0.1)
    assert m.open_mask.tolist() == [True, True, False, True]
    r = HemoAttn(replace(SMALL, supply="local", num_heads=4, local_value="random"))
    r.head_value.copy_(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    r.local_step(price=0.1)
    assert r.open_mask.all()                                     # rule says no closure
    r.head_value[1] = 0.01
    r.local_step(price=0.1)
    assert int(r.open_mask.sum()) == 3                           # closes exactly one


def test_a_backup_is_worth_something_only_under_damage():
    """With no damage a copy is worth ~0 (its twin covers it). When heads fail at random,
    a copy is worth what it saves when its twin fails. Damage never touches the allocation
    the ledger records."""
    from hemo.model import HemoAttn
    from hemo.tasks import make_batch
    c = replace(SMALL, supply="local", num_heads=4, d_k=4, plant_copies=1)
    torch.manual_seed(0)
    X, Y, T, _ = make_batch(256, c, "cpu")
    safe = HemoAttn(c)
    safe.probe_ischemia(X, Y, T)
    torch.manual_seed(0)
    hurt = HemoAttn(replace(c, head_dropout=0.5, probe_masks=64))
    hurt.probe_ischemia(X, Y, T)
    assert float(hurt.head_value.min()) > 100 * float(safe.head_value.abs().max())
    hurt.train()
    y1, g = hurt(X, Y)
    y2, _ = hurt(X, Y)
    assert abs(float(g[0].sum()) - hurt.H) < 1e-4                # allocation untouched
    assert not torch.allclose(y1, y2)                            # damage in training
    hurt.eval()
    assert torch.allclose(hurt(X, Y)[0], hurt(X, Y)[0])          # none in evaluation
