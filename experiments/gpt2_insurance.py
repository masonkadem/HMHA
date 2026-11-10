"""Direction 1: are GPT-2's backup name movers insurance against name movers that fail on some inputs?
Predictions written first: results/gpt2_insurance/predictions.md

On 600 held-out IOI prompts (frozen GPT-2 small, no ablation), for every head of the published circuit's
mover roles (Wang et al. 2023):
  works (attention)  at the final position the head attends more to the indirect object (IO) than to
                     either copy of the subject (S1, S2)
  DLA                direct logit attribution: the head's output at the final position, through the final
                     layer norm (frozen scale) and the unembedding, onto logit(IO) - logit(S)
Then:
  a  failure rate of each head
  b  are the three name movers' failures independent? (pairwise correlation)
  c  insurance: backup movers' DLA on prompts where 0, 1, 2 or 3 name movers fail
  d  model accuracy against the number of name movers that work
  e  empirical tree value of each backup: P(backup works and fewer than R name movers work), R = 1, 2,
     compared with the upkeep tier at which hemodynamic attenuation kept that backup (results/gpt2_ioi)

  python experiments/gpt2_insurance.py  ->  results/gpt2_insurance/insurance.pkl and printed summary
"""
import os, pickle, sys
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from gpt2_ioi import GPT2, make_prompts, WANG, L_, H_, D, DH

OUT = os.path.join(ROOT, "results", "gpt2_insurance")
NM, BACK, NEG = WANG["name mover"], WANG["backup name mover"], WANG["negative name mover"]
HEADS = NM + BACK + NEG


@torch.no_grad()
def run(model, ids):
    """Attention from the last position (L, B, H, T), head outputs z at the last position (L, B, H, DH),
    the final residual at the last position (B, D)."""
    w, (B, T) = model.w, ids.shape
    x = w["wte.weight"][ids] + w["wpe.weight"][:T]
    mask = torch.full((T, T), float("-inf"), device=model.dev).triu(1)
    atts, zs = [], []
    for l in range(L_):
        p = f"h.{l}."
        a = model.ln(x, p + "ln_1") @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = (t.view(B, T, H_, DH).transpose(1, 2) for t in a.split(D, -1))
        att = (q @ k.transpose(-1, -2) / DH ** 0.5 + mask).softmax(-1)
        z = (att @ v).transpose(1, 2)
        atts.append(att[:, :, -1]); zs.append(z[:, -1])
        x = x + z.reshape(B, T, D) @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        m = F.gelu(model.ln(x, p + "ln_2") @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"], approximate="tanh")
        x = x + m @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
    return torch.stack(atts), torch.stack(zs), x[:, -1]


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
    model = GPT2(dev)
    w = model.w
    works, dla, correct = {h: [] for h in HEADS}, {h: [] for h in HEADS}, []
    for ids, io, s in make_prompts(tok, 100, 5000):
        ids, io, s = ids.to(dev), io.to(dev), s.to(dev)
        att, z, xf = run(model, ids)
        logits = model.ln(xf, "ln_f") @ w["wte.weight"].T
        correct.append((logits.gather(1, io[:, None]) > logits.gather(1, s[:, None]))[:, 0].cpu().numpy())
        pos_io = (ids == io[:, None]).float().argmax(1)
        s_mask = (ids == s[:, None])
        sd = torch.sqrt(xf.var(-1, unbiased=False) + 1e-5)                  # frozen final layer-norm scale
        direction = w["wte.weight"][io] - w["wte.weight"][s]                 # (B, D)
        for (l, h) in HEADS:
            a = att[l][:, h]                                                 # (B, T)
            a_io = a.gather(1, pos_io[:, None])[:, 0]
            a_s = torch.where(s_mask, a, torch.zeros_like(a)).max(1).values
            works[(l, h)].append((a_io > a_s).cpu().numpy())
            out = z[l][:, h] @ w[f"h.{l}.attn.c_proj.weight"][h * DH:(h + 1) * DH]
            out = (out - out.mean(-1, keepdim=True)) / sd[:, None] * w["ln_f.weight"]
            dla[(l, h)].append((out * direction).sum(-1).cpu().numpy())
    works = {h: np.concatenate(v) for h, v in works.items()}
    dla = {h: np.concatenate(v) for h, v in dla.items()}
    correct = np.concatenate(correct)
    n = len(correct)
    name = lambda h: f"{h[0]}.{h[1]}"
    rep = []
    P = lambda *a: (print(*a), rep.append(" ".join(str(x) for x in a)))

    P(f"{n} prompts; model picks IO over S on {correct.mean():.1%}")
    P("\na  failure rate (attends to S at least as much as to IO) and mean DLA")
    for grp, hs in (("name mover", NM), ("backup", BACK), ("negative", NEG)):
        for h in hs:
            P(f"   {grp:<11} {name(h):>5}  fails {1 - works[h].mean():6.1%}   mean DLA {dla[h].mean():6.2f}")

    P("\nb  name-mover failures: pairwise correlation (0 = independent)")
    F_nm = np.stack([1 - works[h] for h in NM]).astype(float)
    C = np.corrcoef(F_nm)
    for i in range(3):
        for j in range(i + 1, 3):
            P(f"   {name(NM[i])} x {name(NM[j])}: {C[i, j]:+.2f}")
    k_fail = F_nm.sum(0).astype(int)
    p_ind = [np.prod([1 - works[h].mean() for h in NM])]
    P(f"   all three fail together: observed {np.mean(k_fail == 3):.1%}, if independent {p_ind[0]:.1%}")

    P("\nc  insurance: backup movers' total DLA by number of name movers that fail")
    back_dla = np.sum([dla[h] for h in BACK], 0)
    nm_dla = np.sum([dla[h] for h in NM], 0)
    for k in range(4):
        sel = k_fail == k
        if sel.any():
            P(f"   {k} fail ({sel.sum():>3} prompts): backups DLA {back_dla[sel].mean():6.2f}   name movers DLA {nm_dla[sel].mean():6.2f}")
    r = np.corrcoef(nm_dla, back_dla)[0, 1]
    P(f"   correlation across prompts, name movers' DLA vs backups' DLA: {r:+.2f}")

    P("\nd  accuracy by number of name movers that work")
    for k in range(4):
        sel = (3 - k_fail) == k
        if sel.any():
            P(f"   {k} work ({sel.sum():>3} prompts): correct {correct[sel].mean():6.1%}")

    tiers = {}
    for frac in ("0.05", "0.02", "0.01"):
        path = os.path.join(ROOT, "results", "gpt2_ioi", f"ioi_pf{frac}_s0.pkl")
        if os.path.exists(path):
            for h in pickle.load(open(path, "rb"))["kept"]:
                tiers.setdefault(tuple(h), frac)
    P("\ne  empirical tree value of each backup: P(it works and fewer than R name movers work)")
    vals = {}
    for h in BACK:
        v1 = float(np.mean(works[h] & ((3 - k_fail) < 1)))
        v2 = float(np.mean(works[h] & ((3 - k_fail) < 2)))
        vals[h] = (v1, v2)
    for h in sorted(BACK, key=lambda h: -vals[h][1]):
        P(f"   {name(h):>5}  R=1 {vals[h][0]:6.1%}   R=2 {vals[h][1]:6.1%}   kept by the rule at upkeep: {tiers.get(h, 'never')}")
    pickle.dump(dict(works=works, dla=dla, correct=correct, k_fail=k_fail, values=vals, tiers=tiers, report=rep),
                open(os.path.join(OUT, "insurance.pkl"), "wb"))
    open(os.path.join(OUT, "analysis.txt"), "w").write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
