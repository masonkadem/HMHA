# Thesis plan and to do

Proposal notebook: `notebooks/thesis_proposal.ipynb` (being built). Plain chapter:
`paper/thesis_chapter.tex`. Detailed walkthrough: `notebooks/walkthrough.ipynb`.

## The proposal in one paragraph

Can a model find how many attention heads a task needs, and which ones, during a single
training run? Blood supply in the brain is a working example of a system that allocates a
roughly fixed total by feedback (cerebral autoregulation), senses need locally (astrocytes
in the neurovascular unit), tolerates losing a vessel when neighbours can cover it
(collateral circulation), and prunes low-flow vessels during development. Each of these is
turned into a mechanism and tested by removing it. The candidate novel claim (Aim 3):
vascular network theory predicts when a circuit is minimal (a tree, easy to verify) and when
it keeps backup heads (loops, robust), which would explain the backup heads of the Hydra
effect. The biology is a source of design, never evidence: say "inspired by".

## Status of current claims (read before presenting)

- Holds: benchmark with known k* = R; rank argument predicts every smaller model's loss;
  starving = exact removal; fixed-threshold rules cannot count; loss-referenced rules count
  exactly; local rule never falls back above the solved bar.
- Correction to earlier wording: the local rule does NOT reach dense loss at a third of the
  compute. It beats dense at EQUAL compute in 5/6 runs, matches a full dense run in 2/6.
- Not unique to the idea: pruning with the same bar also finds the count.
- Search cost: training sizes 1, 2, 4, ... until solved costs 0.09 / 0.31 / 0.88 dense runs
  for k* = 2 / 4 / 8, against 0.36 / 0.37 / 0.45 for the local rule. One-run allocation only
  wins for larger circuits.
- qsa_cross: the first "failed 20/20" was an unfair test (target unreachable even by the
  dense model; formula for k* wrong). Fair re-test at d_k = 8, 16 (true k* = 2): loop and
  pruning keep 22 to 29 heads; local rule keeps 3 at d_k = 16 (solved), 8 at d_k = 8 (failed).

## Aim 1: a benchmark with a known answer, and why it has that answer

- [x] multi_relation, k* = R measured at R = 2, 3, 4, 6, 8.
- [x] Rank argument: each head returns one blend of the R contents; R unknowns need R
      different blends; loss = (1 - k/R) x trivial. Matches every dense size (max miss 0.03).
- [x] E1 copy test (`experiments/equations_test.py`): removing heads costs 1/4 each; a copied
      head counts as no new equation (0.2537 vs 0.2540).
- [x] E2a qsa_cross dense sweep (`experiments/qsa_dense.py`): cliff only at d_k = 8, 16
      (k* = 2); big models train worse; repo formula ceil(q log2 N / d_k) is wrong.
- [ ] Replace the qsa k* formula in `tasks.predicted_kstar` with the measured values, or drop
      qsa from claims.
- [ ] Update `thesis_chapter.tex` and `walkthrough.ipynb`: qsa is "inconclusive as first set
      up", not "failed"; add the copy test.

## Aim 2: does each piece of biology earn its place? (local rule controls)

Running now into `results/proposal/` (jobs in `results/proposal/jobs.txt`), k* = 2, 4, 8,
seeds 0 to 2, best local rule (`price_frac=0.03, taper=100, budget_hold_frac=0.1`).

- [~] E3a Autoregulation: `conserve=0` (no fixed total). Same results = the total is framing.
- [~] E3b Collateral value vs chance: `local_value=random` (same timing, blind choice).
- [~] E3c Collateral value vs standard score: `local_value=ablate` (loss rise, no refit).
- [~] E3d Third seed of the rule itself.
- [~] E4 Planted copies (`plant_copies=1`, k* = 4): refit should close one of each twin
      pair, ablate should not.
- [ ] 10 seeds for whichever arms separate; price over 5 values so price is shown not to
      encode k*.
- [ ] The extra head at small k*: close once the others reconstruct most of a head's output
      (refit R^2), not only when its value is below the price.
- [ ] Wall-clock, with starved heads sliced out rather than multiplied by zero.
- [ ] Global loop chatter: PI / nuPI controller (Sohrabi et al. 2024).

## Aim 3 (headline if it holds): minimal or robust, set by vascular physics

Vessel networks minimise dissipation under a cost sum(C^gamma). Below gamma = 1 the optimum
is a tree, above it loops (Bohn and Magnasco 2007); fluctuating load and damage create loops
(Katifori et al. 2010, Corson 2010; adaptation dynamics Hu and Cai 2013). Developing brain
vessels with low flow regress and loops are pruned (Chen et al. 2012, zebrafish).

- [ ] Redundancy meter: heads kept minus rank of their offset-mixing matrix (from E1).
- [ ] New supply `murray`: g_h ~ flow_h^(2/(1+gamma)), rescaled so sum(g^gamma) is fixed;
      gamma = 1 is today's budget. Add to `test_supply_is_conserved` (generalised).
- [ ] Sweep gamma in {0.5, 0.75, 1, 1.5, 2} x head dropout {0, 0.2} x fluctuating demand
      (target blocks masked at random per batch) {off, on}, k* = 4, 3 seeds.
- [ ] Measures: heads kept, redundancy, loss after knocking out one kept head, final loss,
      compute. Holds if redundancy switches near gamma = 1 and dropout or fluctuation add
      backups at low gamma. If not, Aims 1 and 2 stand alone.
- [ ] Expect "gamma < 1 is just an Lq sparsity penalty": the claim must rest on the dynamics
      and the damage-to-redundancy prediction.
- [ ] Verify the gamma values cited for blood vessels before citing them.

## Aim 4: beyond the toy

- [ ] 2-layer attention-only model on induction (previous-token + induction head).
- [ ] Tracr / InterpBench (Gupta et al. 2024): score precision and recall of heads kept.
- [ ] GPT-2 small on IOI, where backup heads are known (tests Aim 3 in a real model).

## Baselines and literature

- [ ] Constrained L0 gates with a loss constraint (Gallego-Posada 2022, GECO + L0 2020).
- [ ] Differentiable subset pruning (Li et al. 2021), EarlyBERT (Chen et al. 2021).
- [ ] Proper literature review before writing "novel": head gating with budgets (2025, 2026
      arXiv), reconstruction pruning (ThiNet, OBS), neurovascular-inspired learning (Philips,
      Chhabria and Chakravarthy 2016), neuron-astrocyte attention (Kozachkov et al. 2023),
      Hydra effect (McGrath et al. 2023), vascular adaptation (Hu and Cai 2013).

## Framing rules

- Lead with the benchmark and the rank argument, then the controls. Efficiency is
  secondary; EarlyBERT already reports 35 to 45% training-time savings.
- Treat fade as gradual pruning (Zhu and Gupta 2017) and refit as OBS-style.
- Biology: blood flow is not strictly zero-sum and vascular steal is weak; collateral
  circulation is well supported. Never claim biological support from the astrocyte-attention
  literature for supply-based gating.

## Housekeeping

- [x] Push to origin (branch `claude/latest-experiment-ideas-fualsp`).
- [ ] Set a git name and email for this repo (commits so far pass them per command).
- [ ] The venv runs torch 2.14 + CUDA 13 (the CUDA 12.8 build lost its division kernels on
      this GPU). Runs before that used 2.11.
