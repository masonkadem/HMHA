# Status: NeuroAI workshop paper (updated 2026-11-12)

TODO.md and HANDOFF.md are from September and describe the older thesis framing. This file is the
current picture for the workshop paper.

## Goal

Submit a 4-page, non-archival paper to the AAAI-27 NeuroAI workshop by **Nov 20, 2026**.
Paper: `paper/neuroai/main_simple.tex`. Long version kept for later: `paper/neuroai/main_long.tex`.
Meeting figure: `paper/neuroai/fig_story.png` (simpler: `fig_story_simple.png`).

## The claim

Charge every attention head a fixed upkeep, as the body does for a vessel. The rule then minimises
loss + upkeep x open heads, so what it keeps can be predicted before training.
The supervisor's concern to answer: "it's just a heuristic".

## Results that hold (safe to present)

| Result | Number | Where |
|---|---|---|
| Right head count, model intact | 29/30 and 30/30; ZipLM, Michel, L0 gates break it 30/30 | retrieval task |
| Collapse where theory says | 18/18 keep the count below the threshold, 18/18 collapse above | retrieval task |
| Backups predicted from a failure tree | 104/130 exact, 130/130 within one (100 predicted in writing first: 79 exact) | retrieval task |
| Extra backup seeds (browser session) | 26/32 exact, 32/32 within one; check they are separate from the 130 | `results/confirm10` |
| Induction circuit | 10/10 | induction task |
| GPT-2 name task (IOI), Wang's prompts, 5 seeds | 10-11 / 17-18 / 20-21 heads, nested 5/5; 20-21 heads 98-99% vs published 26 at 99.0-99.4%; 19 heads kept in all seeds | `results/gpt2_ioi_wang/analysis.md` |
| Re-checking beats ranking once | IOI 89% vs 9%; repeated tokens 73% vs 27%; greater-than a tie | `results/gpt2_core/analysis.md` |
| Valuing step on real text (pilot) | rank agreement with true importance 0.86 vs 0.07 for switching off | `experiments/shakespeare_pilot` |

## Tried and did not work (do not present as support)

- GPT-2 backups as insurance on natural prompts (direction 1): name movers rarely fail; no effect.
- Ordinary dropout in a language model: more dropout gave fewer heads, not more.
- Whole-head failure in a language model, fixed-upkeep rerun (5 seeds): count FALLS with failure rate
  (26 -> 17 -> 15 -> 14 of 32); valuing under failure keeps 1-3 more than without. Backup law not shown on text.
- Removing the name ability from GPT-2 by head removal (5 seeds): neither method reached chance within 20 heads;
  self-repair confirmed (98-99% after removing the 3 name movers). Rerun targeting the logit difference is running.
- Induction under head failure: count rises, but only 12/20 exact.
- Grokking pilot: our score gives no earlier warning than simpler ones.

## Running now (restarted 2026-11-12 after the machine restarted)

1. Removal test with the logit-difference target, 5 seeds. `results/gpt2_remove/run_margin.log`.
2. Seeds 1-4 for greater-than and repeated tokens. `results/gpt2_core/seeds.log`.

## Supervisor's questions from the meeting

- How stable is the published circuit, and ours? One prompt or many? -> run 1 above.
- Does pruning hurt or help on other things? -> not run yet: score the 22-head model on the other
  tasks and on plain text.

## To do before Nov 20

- [x] GPT-2 paragraph, abstract and figure 4 updated to Wang's prompts, 5 seeds (2026-11-12).
- [ ] Cross-task test of the 22-head circuit (about 1 hour of GPU).
- [x] Backups scoped to the benchmark; Limitations reports the failed language-model test.
- [x] Comparison table: CHG and ACDC rows (not run), 'predicts size' column; gaps and RigL sentences; benchmark as a contribution.
- [ ] Add citations: ACDC, Causal Head Gating (2025), SFBS (Pudil 1994), k-out-of-n reliability,
      and read arXiv 2607.01940 for overlap.
- [ ] Decide whether the Shakespeare pilot gets one sentence.
- [ ] Use the tree figure `figures/fig_backup_tree.png` for explaining backups.
- [x] Abstract collapse wording fixed.
- [ ] Check new references: Causal Head Gating, Tracr, InterpBench, Sanford et al. (written from memory).
- [ ] Page limit, anonymisation, references, OpenReview.

Dates: draft to supervisor Nov 14, comments back Nov 17, fixes Nov 18-19, submit Nov 20.

## Decisions still needed from the supervisor

Approval to submit; 4-page non-archival or 8-page; whether a formal proof is wanted; authors and order;
whether a comparison with ACDC or Causal Head Gating is required first.

## After the workshop (not for Nov 20)

1. The rule during training of a real language model, against other pruning methods.
2. A circuit map of GPT-2 across 8-10 tasks.
3. Head-to-head with ACDC, attribution patching and Causal Head Gating.
4. GPT-2 medium.
5. One redundancy law for synapses, vessels and heads (`results/direction3_notes.md`).

## Branches

- `claude/latest-experiment-ideas-fualsp`: this work, pushed.
- `claude/wonderful-lovelace-r7owsf`: browser session. Its new files were copied here; do not merge
  it, because it starts from the trimmed `main` and a merge would delete files this branch uses.
