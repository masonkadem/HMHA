# Working agreement

Read README.md first. It contains the evidence table, the negative results, and the
ranked experiment roadmap. Do not restate it back to the user.

## Before changing anything

Run `pytest -q tests/`. If `test_one_head_fails_two_heads_succeed` or
`test_heads_implement_distinct_offsets` fails, the benchmark's ground truth is broken and
no downstream result means anything. Fix that before touching the model.

## Non-negotiables

1. **Conservation.** Every supply mechanism must satisfy `sum(gate) == H`. Add a case to
   `test_supply_is_conserved` for any new mechanism.
2. **Ablation at full perfusion.** Never call `head_ablation` with a starved gate. Most
   heads sit at zero and every Spearman becomes a tie artefact.
3. **Scale-free thresholds.** Use `analysis.threshold`, which is a fraction of the
   dense-to-trivial gap. A threshold relative to the dense loss alone becomes unreachable
   once the task is solved exactly, which produced a spurious "no redundancy" result.
4. **Nothing hardcoded into figures.** Every claim in a caption or summary panel is
   computed from the run. This repo exists partly because an earlier version shipped a
   text box contradicting the plots beside it.
5. **Measure loss during annealing**, not by post-hoc re-gating a converged model. A model
   annealed to small B is already adapted to it, which flatters small B.
6. **One seed per process.** Long multi-seed runs get reaped in some environments.

## Do not

- Do not make magnitude scoring the contribution. It is a standard named baseline.
- Do not delete the qnorm arm. Its negative result (rho = -0.71 below the capacity bound)
  is a finding.
- Do not quietly drop a negative result when a variant underperforms.
- Do not add PPG or any physiological dataset here. This repo is synthetic only.
- Do not claim biological support from the astrocyte-attention literature for
  supply-based gating. That correspondence is about substrate, not about metabolic
  gating, and conflating them is the mistake a neuroscience reviewer will catch.

## Style

The user writes in a concise scientific register. No em dashes. Rank options explicitly.
Give direct assessments rather than hedged ones. When a result is null, say so plainly and
say what it rules out.
