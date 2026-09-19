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
    for supply in ["topk", "threshold", "territory", "gap", "otsu", "autoreg"]:
        c = replace(cfg, supply=supply)
        m, _ = train(c, val, dev, hemo=True, steps=20)
        g = m.gate(kappa=1.0, B_active=3)
        assert abs(float(g.sum()) - m.H) < 1e-3, f"{supply} broke conservation: {g.sum()}"


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
