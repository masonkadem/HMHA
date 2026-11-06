"""Hemodynamic attenuation on GPT-2 small, indirect object identification (IOI).

Every one of the 144 attention heads gets a valve. A closed head is replaced by its mean output
over prompts with three different names (the "ABC" set), as in Wang et al. (2023), so closing a
head removes what it says about these names and keeps everything else.

The model is frozen, so this runs the rule without training: there is no fade and no re-fit, and
a head's value is the loss rise without it (as on induction). It is exactly the greedy
minimisation of Proposition 1:
    J(S) = L(S) + price * |S|
    close the open head with the smallest value  L(S \\ h) - L(S)   while it is below price,
    reopen the closed head with the largest gain L(S) - L(S + h)    when it is above 2 x price.
L is cross-entropy of the indirect object's name at the last position. The price is a fraction
of the gap between the all-closed and the all-open loss.

The heads kept are compared with the 26-head circuit of Wang et al. (2023).

  python experiments/gpt2_ioi.py --price 0.01 0.02 0.05   ->  results/gpt2_ioi/*.pkl
Needs torch (GPU if available), safetensors, tokenizers, huggingface_hub.
"""
import argparse, os, pickle, random, time
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
L_, H_, D, DH = 12, 12, 768, 64

# The circuit reported by Wang et al. (2023), layer.head
WANG = {
    "name mover": [(9, 9), (9, 6), (10, 0)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative name mover": [(10, 7), (11, 10)],
    "S-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}
ROLE = {h: r for r, hs in WANG.items() for h in hs}

NAMES = ["Mary", "John", "Tom", "James", "Dan", "Sid", "Kate", "Alice", "Bob", "Paul", "Sarah", "Mike",
         "Laura", "David", "Anna", "Mark", "Lisa", "Jack", "Emma", "Peter", "Rose", "Steve", "Amy", "Chris"]
PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "park", "station"]
OBJECTS = ["drink", "ring", "kiss", "bone", "basketball", "computer", "necklace", "snack"]
# {A} is the indirect object, {B} the subject; {S2} is the repeated subject (a third name in ABC)
TEMPLATES = [
    "Then, {A} and {B} went to the {P}. {S2} gave a {O} to",
    "Then, {B} and {A} went to the {P}. {S2} gave a {O} to",
    "When {A} and {B} got a {O} at the {P}, {S2} decided to give it to",
    "When {B} and {A} got a {O} at the {P}, {S2} decided to give it to",
    "After {A} and {B} went to the {P}, {S2} gave a {O} to",
    "After {B} and {A} went to the {P}, {S2} gave a {O} to",
]


class GPT2:
    """GPT-2 small forward pass with a valve on every head. gate[l, h] = 1 keeps the head, 0 replaces
    its output by the mean output zmean[t][l, :, h] for the prompt's template t."""

    def __init__(self, dev):
        w = load_file(hf_hub_download("gpt2", "model.safetensors"))
        self.w = {k.replace("transformer.", ""): v.to(dev).float() for k, v in w.items()}
        self.dev = dev

    def ln(self, x, name):
        return F.layer_norm(x, (D,), self.w[name + ".weight"], self.w[name + ".bias"], 1e-5)

    @torch.no_grad()
    def __call__(self, ids, gate=None, zmean=None, record=False):
        w, (B, T) = self.w, ids.shape
        x = w["wte.weight"][ids] + w["wpe.weight"][:T]
        mask = torch.full((T, T), float("-inf"), device=self.dev).triu(1)
        zs = []
        for l in range(L_):
            p = f"h.{l}."
            a = self.ln(x, p + "ln_1") @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
            q, k, v = (t.view(B, T, H_, DH).transpose(1, 2) for t in a.split(D, -1))
            att = (q @ k.transpose(-1, -2) / DH ** 0.5 + mask).softmax(-1)
            z = (att @ v).transpose(1, 2)                                  # (B, T, H, DH)
            if record:
                zs.append(z)
            if gate is not None:
                g = gate[l].view(1, 1, H_, 1)
                z = g * z + (1 - g) * zmean[l]
            x = x + z.reshape(B, T, D) @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
            m = self.ln(x, p + "ln_2") @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]
            m = F.gelu(m, approximate="tanh")
            x = x + m @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
        logits = self.ln(x[:, -1], "ln_f") @ w["wte.weight"].T
        return (logits, zs) if record else logits


def make_prompts(tok, n_per, seed, abc=False):
    """Per template: token ids (n, T), and the token ids of the indirect object and of the subject."""
    rng = random.Random(seed)
    out = []
    for ti, tpl in enumerate(TEMPLATES):
        ids, io, s = [], [], []
        for _ in range(n_per):
            A, B_, C = rng.sample(NAMES, 3)
            text = tpl.format(A=A, B=B_, S2=C if abc else B_, P=rng.choice(PLACES), O=rng.choice(OBJECTS))
            ids.append(tok.encode(text).ids)
            io.append(tok.encode(" " + A).ids[0])
            s.append(tok.encode(" " + B_).ids[0])
        assert len({len(i) for i in ids}) == 1, f"template {ti} has prompts of different lengths"
        out.append((torch.tensor(ids), torch.tensor(io), torch.tensor(s)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    ap.add_argument("--n", type=int, default=50, help="prompts per template")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gpt2_ioi"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
    for nm in NAMES + PLACES + OBJECTS:
        assert len(tok.encode(" " + nm).ids) == 1, nm
    model = GPT2(dev)

    def to(data):
        return [(i.to(dev), io.to(dev), s.to(dev)) for i, io, s in data]

    fit = to(make_prompts(tok, a.n, 1000 + a.seed))
    test = to(make_prompts(tok, a.n, 2000 + a.seed))
    abc = to(make_prompts(tok, a.n, 3000 + a.seed, abc=True))
    zmean = []                                     # per template: (L, 1, T, H, DH) mean head outputs
    for ids, _, _ in abc:
        zs = model(ids, record=True)[1]
        zmean.append(torch.stack([z.mean(0, keepdim=True) for z in zs]))

    def evaluate(gate, data):
        """Loss (cross-entropy of the indirect object), logit difference IO - S, accuracy IO > S."""
        ce, ld, acc, n = 0.0, 0.0, 0.0, 0
        for t, (ids, io, s) in enumerate(data):
            lg = model(ids, gate, zmean[t])
            ce += F.cross_entropy(lg, io, reduction="sum").item()
            d = lg.gather(1, io[:, None]) - lg.gather(1, s[:, None])
            ld += d.sum().item(); acc += (d > 0).float().sum().item(); n += len(ids)
        return ce / n, ld / n, acc / n

    ones, zeros = torch.ones(L_, H_, device=dev), torch.zeros(L_, H_, device=dev)
    full, empty = evaluate(ones, fit), evaluate(zeros, fit)
    gap = empty[0] - full[0]
    print(f"all open: loss {full[0]:.3f}, logit diff {full[1]:.2f}, IO>S {full[2]:.1%}")
    print(f"all closed: loss {empty[0]:.3f}, logit diff {empty[1]:.2f}, IO>S {empty[2]:.1%}")
    wang_gate = zeros.clone()
    for (l, h) in ROLE:
        wang_gate[l, h] = 1
    wt = evaluate(wang_gate, test)
    print(f"Wang et al. circuit (26 heads), held out: loss {wt[0]:.3f}, logit diff {wt[1]:.2f}, IO>S {wt[2]:.1%}")

    for frac in a.price:
        price = frac * gap
        gate = ones.clone()
        L = evaluate(gate, fit)[0]
        t0, log, n_eval = time.time(), [], 0
        for it in range(1000):
            vals = torch.full((L_, H_), float("nan"))
            for l in range(L_):
                for h in range(H_):
                    g = gate.clone(); g[l, h] = 1 - g[l, h]
                    vals[l, h] = evaluate(g, fit)[0] - L       # closing: value; opening: minus gain
                    n_eval += 1
            open_ = gate.cpu() > 0
            v_open = torch.where(open_, vals, torch.tensor(float("inf")))
            g_closed = torch.where(~open_, -vals, torch.tensor(float("-inf")))
            if g_closed.max() > 2 * price:
                l, h = divmod(int(g_closed.argmax()), H_)
                gate[l, h] = 1; act = "reopen"
            elif v_open.min() < price:
                l, h = divmod(int(v_open.argmin()), H_)
                gate[l, h] = 0; act = "close"
            else:
                break
            L = evaluate(gate, fit)[0]
            log.append((act, l, h, L, int(gate.sum())))
        kept = [(l, h) for l in range(L_) for h in range(H_) if gate[l, h] > 0]
        te = evaluate(gate, test)
        hit = [k for k in kept if k in ROLE]
        print(f"\nprice {frac:g} x gap: kept {len(kept)} heads after {len(log)} decisions "
              f"({sum(1 for x in log if x[0] == 'reopen')} reopenings), {time.time() - t0:.0f} s")
        print(f"  held out: loss {te[0]:.3f}, logit diff {te[1]:.2f} "
              f"({(te[1] - empty[1]) / (full[1] - empty[1]):.0%} of full), IO>S {te[2]:.1%}")
        print(f"  in Wang's circuit: {len(hit)}/{len(kept)} kept; by role:")
        for r, hs in WANG.items():
            print(f"    {r:<20} {sum(h in kept for h in hs)}/{len(hs)}  {[f'{l}.{h}' for l, h in hs if (l, h) in kept]}")
        print(f"  kept outside Wang's circuit: {[f'{l}.{h}' for l, h in kept if (l, h) not in ROLE]}")
        pickle.dump(dict(price_frac=frac, price=price, kept=kept, log=log, test=te, full=full, empty=empty,
                         wang_test=wt, fit_n=a.n, seed=a.seed, n_eval=n_eval, seconds=time.time() - t0),
                    open(os.path.join(a.out, f"ioi_pf{frac:g}_s{a.seed}.pkl"), "wb"))


if __name__ == "__main__":
    main()
