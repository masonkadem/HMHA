"""Ground truth for the two-layer induction benchmark. Run: pytest -q tests/"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataclasses import replace
import torch

from hemo.induction import ICfg, InductionNet, make_batch, predict_mask, train


def test_sequences_repeat_after_a_random_gap():
    cfg = ICfg()
    tok, second = make_batch(64, cfg, torch.Generator().manual_seed(0))
    for b in range(64):
        A = tok[b, :cfg.L]
        assert len(set(A.tolist())) == cfg.L                               # A has no repeats
        assert torch.equal(tok[b, second[b]:second[b] + cfg.L], A)          # second copy
        gap = tok[b, cfg.L:second[b]].tolist()
        assert not set(gap) & set(A.tolist())                              # gap avoids A
    assert len(set(second.tolist())) > 5                                   # the gap varies
    m = predict_mask(second, cfg)
    assert int(m[0].sum()) == cfg.L - 1


def test_a_starved_head_is_removed_exactly():
    cfg = replace(ICfg(), heads=3, device="cpu")
    torch.manual_seed(0)
    full = InductionNet(cfg)
    tok, _ = make_batch(8, cfg, torch.Generator().manual_seed(1))
    gate = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    small = InductionNet(replace(cfg, heads1=2, heads2=2))
    keep = {0: [0, 2], 1: [0, 1]}
    with torch.no_grad():
        small.embed.weight.copy_(full.embed.weight)
        small.pos.copy_(full.pos)
        small.unembed.load_state_dict(full.unembed.state_dict())
        for l in (0, 1):
            src, dst, d = full.layers[l], small.layers[l], cfg.d_head
            rows = torch.cat([torch.arange(h * d, (h + 1) * d) for h in keep[l]])
            for name in ("W_q", "W_k", "W_v"):
                getattr(dst, name).weight.copy_(getattr(src, name).weight[rows])
            dst.W_o.copy_(src.W_o[keep[l]])
        assert torch.allclose(full(tok, gate), small(tok), atol=1e-5)


def test_one_layer_fails_one_head_per_layer_succeeds():
    """k* = one head in each layer: induction needs two layers."""
    base = replace(ICfg(), rule="dense", steps=1200, seed=0)
    _, both = train(replace(base, heads1=1, heads2=1))
    _, one = train(replace(base, heads1=2, heads2=0))
    assert both["final_loss"] < both["bar"], both["final_loss"]
    assert one["final_loss"] > 10 * both["bar"], one["final_loss"]
