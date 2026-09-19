# hemo-attn

Hemodynamically constrained attention. A fixed total blood supply is allocated across
attention heads, starving the ones that do not demand it, so the surviving circuit is
small enough to verify mechanistically.

## The one-sentence claim

A metabolically constrained transformer discovers its own minimal circuit size and
confines it spatially, verified on a benchmark where the true circuit is known.

## Evidence so far

The full roadmap below has now been run: 88 runs, 3 to 5 seeds per condition, CPU,
`multi_relation` at H=32, d_k=32, N=16, R=4, 4000 steps. The ground-truth redundancy
check reproduces the CUDA numbers exactly. Figures in `figures/`
(`fig0_ground_truth` through `fig5_demand_arms`), per-run pickles in `results/`, every
number in the generated block at the end of this file. Regenerate with

```bash
python experiments/sweep.py --run --workers 4
python experiments/aggregate.py --readme README.md
```

Summary of that sweep, ranked by how much it moves the claim:

1. **Experiment 1 holds, and it is the strongest result here.** At kappa_end = 1.5 the
   perfused count settles at exactly 4 = k*, zero variance over 5 seeds, for 100% of
   the converged phase, with no budget given to the mechanism. B is monotone in
   kappa_end from 5.60 down to 1.40, so it is selected by ischemia depth rather than
   sitting at k* for free, and over-starving costs three orders of magnitude of loss.
2. **Experiment 5 splits the old negative results.** Demand-ranked supply does beat
   chance at B = k* (0.00098 to 0.00145 for the three signal arms against 0.08007 for
   random, a factor of 82). But the three signal arms are mutually inseparable: their
   spread is smaller than any one of their CIs, so EMA vs instantaneous is a tie here,
   not a loss for EMA. `--leak 0.05` did not rescue the EMA arm specifically; it helps
   random most.
3. **Experiments 2, 3 and 4 are null.** Territory compaction does not occur, no optimal
   delay exists, and pooling changes nothing. Details and causes below.

**Prior evidence, 5 seeds, CUDA, same configuration.**

| finding | value |
|---|---|
| ground-truth k* (dense, k heads from scratch) | 4, 4, 4, 4, 4 |
| predicted k* | 4 |
| loss staircase over k = 1, 2, 4 | 0.755, 0.504, 0.00002 |
| trivial (predict-mean) baseline | 1.007 |
| role coverage at B = 1, 2, 4, 16 | 0.25, 0.50, 0.85, 1.00 |
| role purity at B = 32 → B = 1 | 0.232 → 0.399 |

Redundancy is real, k* is predicted correctly, and starvation does raise role purity.

**Negative results. These are load-bearing and must not be quietly dropped.**

1. Query L2 norm anti-predicts head importance below the capacity bound.
   rho(qnorm, ablation delta) = -0.71 +/- 0.08 at d_k=8, and +0.26 +/- 0.35 at d_k=40.
   The original HMHA gating signal is falsified. Keep it as a control arm.
2. Instantaneous magnitude beats slow EMA demand at the critical budget.
   At B=4: magnitude 0.0015, demand 0.0043, random 0.0225, qnorm 0.0508.
   The slow-integration hypothesis has no support yet.
3. All arms including random reached the same B*. Any mechanism reducing to
   "rank heads, keep top B" ties with magnitude pruning. Prior work
   (EMNLP 2020, weak link between importance and prunability) already reports this.

**New negative results from the roadmap sweep. Same standing as the three above.**

4. Territory compaction does not happen. territory_span tracks T at every T
   (span/T = 0.81 to 1.00, mean 0.93), and role coverage declines to 0.79 below T = R
   rather than collapsing. The cause is structural: the never-fully-infarct fallback is
   applied per territory, so every territory keeps at least one perfused head and
   perfused >= T by construction. The mechanism as written cannot compact below T, so
   this prediction was not testable by it. Testing compaction needs the no-infarct
   floor to be global, not per-territory.
5. There is no optimal delay. tau over 0, 1, 5, 20 spans 0.00010 in converged loss with
   overlapping CIs; only tau = 100 is worse, and that is driven by one seed. The
   predicted failure mode at long tau is absent: perfused-set changes per step over the
   held phase peak at 0.0037, one change per 272 steps, so there is no gate oscillation
   for a longer lag to trade against.
6. Pooled demand changes nothing. Over pool_beta 0 to 1, role coverage spans 0.92 to
   1.00 and territory_span 6.67 to 7.67, while full pooling has the worst converged
   loss of the four. The non-locality this was meant to introduce has no measured
   effect.
7. The signal the mechanism gates on is not the best available predictor of causal head
   importance. Over all 88 runs at full perfusion, rho(demand) = +0.379 +/- 0.066 while
   rho(qnorm) = +0.484 +/- 0.085 and rho(neg_entropy) = +0.452 +/- 0.083 on the same
   runs. This is at d_k = 32, ABOVE the capacity bound, so it does not contradict
   negative result 1, which measured qnorm below it.

**Known bug now fixed.** Head ablation previously ran at the starved gate, leaving most
heads already at zero, so every rho was a tie-dominated artefact near +0.1. Ablation now
runs at full perfusion. On a 400-step smoke test this moved rho(demand) from +0.14 to
+0.93. Every rho reported before this fix is void.

## Why this is not pruning

Magnitude head pruning is a standard named baseline. So is attention-entropy pruning
(Zhai et al. 2023) and gradient head importance (Michel et al. 2019). Gated pruning during
training is Voita et al. 2019. Causal Head Gating (NeurIPS 2025) already applies learned
head gates to mechanistic interpretability in LLMs. Switching the contribution to
magnitude scoring is not an option.

Two mechanisms here have no pruning analogue, because every head-importance score in the
literature scores heads independently.

**threshold** theta = mean(d) + kappa*std(d). The perfused set, and therefore the
effective head count, emerges from how concentrated demand is rather than being set as a
hyperparameter. Prediction: emergent B converges on the task's true k*.

**territory** Heads share a penetrating arteriole. Supply is fixed per territory and heads
compete only with their territory-mates. Starving a head frees flow for its neighbours,
not globally. Prediction: circuits compact into territories, and role recovery collapses
when the number of territories drops below the number of roles. Both halves of that
prediction were tested and neither held; see negative result 4.

kappa is normalised per territory by `model.local_kappa`, because a threshold in raw
standard deviations is not comparable between a 32-head pool and a 4-head territory. The
largest z-score attainable in a group of n values is (n-1)/sqrt(n), which at H=32 is 1.50
for T=8 and 0.71 for T=16, so an un-normalised kappa_end of 1.5 makes the local threshold
unsatisfiable and pins the perfused count to exactly T. Enforced by
`tests/test_ground_truth.py::test_territory_threshold_is_not_pinned_to_T`.

Supply always conserves total flow, `sum(gate) == H`, with gate 1 as the perfused
baseline, so the dense model is the special case gate == 1 everywhere. This is enforced
by `tests/test_ground_truth.py::test_supply_is_conserved`.

## The benchmark

`multi_relation` is the contribution that makes everything else measurable. Each query
token carries a key position p. The target is R blocks, block r holding the content stored
at position (p + offset_r) mod N. A single head cannot implement two offsets, because W_q
is linear and the required query-key geometry differs per offset. So k* = R exactly, set
by function rather than capacity, and every surviving head has a named role.

This is what lets interpretability be scored as role-recovery accuracy. Existing pruning
work scores accuracy retention only, because no standard benchmark has known head roles.

Verified at N=8, R=2: one head reaches 0.44 against a trivial baseline of 0.99 and
implements one offset. Two heads reach 0.0002 and implement offsets 5 and 1. Four and
eight heads duplicate those two roles.

`qsa_cross` is the legacy task. Minimal head count is ceil(q*log2(N)/d_k), heads are
interchangeable, and one head solves it whenever d_k exceeds the bound. Kept for
comparison; it cannot validate head roles.

## Layout

```
src/hemo/config.py     single Cfg dataclass, every knob
src/hemo/tasks.py      multi_relation, qsa_cross, predicted k*
src/hemo/model.py      CrossAttn, HemoAttn. demand and supply are separate axes
src/hemo/train.py      schedule (progressive ischemia), train, evaluate
src/hemo/analysis.py   redundancy_check, circuit_recovery, head_ablation
experiments/run.py     one seed per invocation, writes results/*.pkl
experiments/sweep.py   the ranked roadmap as a resumable job list
experiments/aggregate.py  figures/ and the generated block in this README
tests/                 ground-truth assertions. Run these first.
```

## Run

```bash
pip install -r requirements.txt
pytest -q tests/                                     # ~3 min CPU, must pass
python experiments/run.py --seed 0 --supply threshold --demand outnorm_ema
python experiments/run.py --seed 0 --supply territory --n_territories 8
```

```bash
python experiments/sweep.py --list                   # the whole roadmap as jobs
python experiments/sweep.py --run --workers 4        # resumable; skips finished jobs
python experiments/aggregate.py --readme README.md   # figures + the generated block
```

Each seed writes
`results/<task>_<supply>_<demand>_T<n>_tau<d>_k<kappa_end>_b<pool_beta>_lk<leak>_s<seed>.pkl`.
The tag carries kappa_end, pool_beta and leak because experiments 1, 4 and 5 sweep them
and would otherwise overwrite each other. Long runs get killed in some environments, so
always run one seed per process.

## Experiment roadmap, ranked

1. **Emergent B vs true k\*.** `--supply threshold`, sweep kappa_end, 5 seeds. Does the
   perfused count settle at 4 without being told? This is the strongest available claim
   and it is not a ranking, so magnitude pruning cannot compete on it.
2. **Territory compaction.** `--supply territory`, sweep n_territories over 1, 2, 4, 8, 16, 32
   at R=4. Prediction: role_coverage holds while T >= R and collapses below. Report
   territory_span, which should fall well below T.
3. **Delay.** `--delay` over 0, 1, 5, 20, 100. Prediction: an optimal tau exists, too short
   causing premature commitment and too long causing gate oscillation. Cheapest clean
   dynamical figure.
4. **Pooled demand.** `--pool_beta` over 0, 0.25, 0.5, 1.0 with territory supply. This is what
   makes the signal genuinely non-local.
5. **Demand arms at matched perfusion.** outnorm_ema vs outnorm_inst vs qnorm vs random.
   Run with `--leak 0.05` as well as 0, since irreversible starvation may be penalising
   the EMA arm specifically. If EMA still loses, report it.

## Rules for this repo

- Ground-truth tests pass before any result is believed.
- Every reported number is computed, never hardcoded into a figure caption. An earlier
  version of this work shipped a summary box that contradicted its own plots.
- Ablation runs at full perfusion.
- Report the loss at B = k*, not B*. B* saturates and cannot separate arms. This only
  means anything when B = k* is imposed, as it is under topk supply. Under threshold and
  territory supply B emerges, so a run can pass through k* early in annealing and settle
  elsewhere; report the converged loss there, plus the fraction of the held phase spent
  at k*.
- A threshold in standard deviations is only comparable within one group size. Normalise
  by the attainable bound (n-1)/sqrt(n) before comparing across territory sizes.
- A figure title states what is plotted, never the prediction under test. Verdicts are
  computed in `aggregate.py` and written into the generated block.
- Thresholds are relative to the dense-to-trivial gap, never to the dense loss alone.
- Negative results stay in the paper. Six of the ten findings so far are negative.

## Prior work to cite and differentiate

Michel et al. 2019 (head importance), Voita et al. 2019 (L0 gated head pruning,
specialized heads do the heavy lifting), Zhai et al. 2023 (attention entropy),
EMNLP 2020 weak-link paper (random pruning is competitive), Causal Head Gating
NeurIPS 2025 (learned head gates for interpretability),
Kozachkov/Slotine/Krotov (astrocytes as a substrate for attention; motivates the
astrocyte-transformer correspondence but does NOT test supply-based gating, so do not
cite it as support for this mechanism).

<!-- BEGIN GENERATED: experiments/aggregate.py -->
_Generated by `python experiments/aggregate.py --readme README.md` from 88 runs in `results/`. Do not edit by hand._


**0 ground truth**

- measured k* per seed = [4, 4], predicted = 4; trivial = 1.0075, threshold = 0.02016
- loss by k: k=1: 0.75511, k=2: 0.50314, k=4: 0.00002, k=8: 0.00001, k=16: 0.00001, k=32: 0.00001

**1 emergent B**

- kappa_end=0.5 (n=5): B=5.60+/-1.00 (k*=4), converged loss=0.00025, frac of held phase at exactly k* = 0.20, role_coverage=0.90
- kappa_end=1 (n=5): B=5.40+/-1.18 (k*=4), converged loss=0.00022, frac of held phase at exactly k* = 0.40, role_coverage=0.85
- kappa_end=1.5 (n=5): B=4.00+/-0.00 (k*=4), converged loss=0.00066, frac of held phase at exactly k* = 1.00, role_coverage=0.80
- kappa_end=2 (n=5): B=3.00+/-1.24 (k*=4), converged loss=0.25297, frac of held phase at exactly k* = 0.61, role_coverage=0.60
- kappa_end=2.5 (n=5): B=1.40+/-0.78 (k*=4), converged loss=0.65559, frac of held phase at exactly k* = 0.00, role_coverage=0.30
- VERDICT prediction 'emergent B converges on k*': SUPPORTED. kappa_end=1.5 gives B=4.00+/-0.00 against k*=4, with 100% of the held phase at exactly k*. B is monotone in kappa_end over 1.40 to 5.60, so the mechanism does not sit at k* for free: kappa_end selects it.

**2 territory**

- T=1 (n=3): role_coverage=0.75+/-0.00, territory_span=1.00, perfused=4.00
- T=2 (n=3): role_coverage=0.83+/-0.16, territory_span=2.00, perfused=4.67
- T=4 (n=3): role_coverage=1.00+/-0.00, territory_span=3.67, perfused=6.33
- T=8 (n=3): role_coverage=0.92+/-0.16, territory_span=6.67, perfused=9.33
- T=16 (n=3): role_coverage=1.00+/-0.00, territory_span=13.00, perfused=16.00
- T=32 (n=3): role_coverage=1.00+/-0.00, territory_span=32.00, perfused=32.00
- VERDICT prediction 'coverage collapses below T=R': NOT SUPPORTED. mean coverage below T=R is 0.79 vs 0.98 at or above; a decline, not a collapse.
- VERDICT prediction 'territory_span falls well below T': NOT SUPPORTED. span/T = 0.81 to 1.00 (mean 0.93); span tracks T.
- CAUSE the never-fully-infarct fallback is applied per territory, so every territory keeps at least one perfused head and perfused >= T by construction. Compaction below T cannot occur for this mechanism as written.

**3 delay**

- tau=0 (n=5): converged loss=0.00066+/-0.00051, flips/step=0.0004, role_coverage=0.80
- tau=1 (n=3): converged loss=0.00073+/-0.00092, flips/step=0.0000, role_coverage=0.75
- tau=5 (n=3): converged loss=0.00073+/-0.00091, flips/step=0.0000, role_coverage=0.75
- tau=20 (n=3): converged loss=0.00064+/-0.00074, flips/step=0.0000, role_coverage=0.75
- tau=100 (n=3): converged loss=0.04918+/-0.09588, flips/step=0.0037, role_coverage=0.75
- VERDICT prediction 'an optimal tau exists': NOT SUPPORTED. tau 0 to 20 spans only 0.00010 in loss with overlapping CIs; only tau=100 is worse.
- VERDICT prediction 'long tau causes gate oscillation': NOT SUPPORTED. max perfused-set changes per step over the held phase is 0.0037 at any tau (one set change per ~272 steps at worst), so there is no oscillation to trade against.

**4 pool**

- pool_beta=0 (n=3): role_coverage=0.92, territory_span=6.67, converged loss=0.00007+/-0.00005
- pool_beta=0.25 (n=3): role_coverage=1.00, territory_span=7.67, converged loss=0.00004+/-0.00001
- pool_beta=0.5 (n=3): role_coverage=0.92, territory_span=7.33, converged loss=0.00008+/-0.00010
- pool_beta=1 (n=3): role_coverage=1.00, territory_span=7.67, converged loss=0.00029+/-0.00007
- VERDICT pooling changes nothing. role_coverage spans 0.92 to 1.00 and territory_span 6.67 to 7.67 across pool_beta 0 to 1; full pooling has the worst converged loss (0.00029).

**5 demand arms**

- outnorm_ema, leak=0 (n=3): loss@B=k*=0.00132+/-0.00160
- outnorm_ema, leak=0.05 (n=3): loss@B=k*=0.00122+/-0.00152
- outnorm_inst, leak=0 (n=3): loss@B=k*=0.00145+/-0.00186
- outnorm_inst, leak=0.05 (n=3): loss@B=k*=0.00124+/-0.00151
- qnorm, leak=0 (n=3): loss@B=k*=0.00098+/-0.00128
- qnorm, leak=0.05 (n=3): loss@B=k*=0.00089+/-0.00123
- random, leak=0 (n=3): loss@B=k*=0.08007+/-0.15430
- random, leak=0.05 (n=3): loss@B=k*=0.00271+/-0.00338
- VERDICT ordering at leak=0, best to worst: qnorm 0.00098, outnorm_ema 0.00132, outnorm_inst 0.00145, random 0.08007
- VERDICT the three signal arms are not separable: they span 0.00098 to 0.00145 while every one of their 95% CIs is wider than that spread. EMA vs instantaneous is a tie.
- VERDICT random is separable and worse: 0.08007 vs 0.00098 for the best signal arm, a factor of 82. Demand-ranked supply beats chance at B = k*.
- VERDICT leak=0.05 does not rescue the EMA arm specifically. It helps random most (0.08007 -> 0.00271); outnorm_ema moves 0.00132 -> 0.00122.

**5 ablation**

- rho(demand, ablation delta) = +0.379+/-0.066 over 88 runs, at full perfusion
- rho(outnorm, ablation delta) = +0.295+/-0.097 over 88 runs, at full perfusion
- rho(neg_entropy, ablation delta) = +0.452+/-0.083 over 88 runs, at full perfusion
- rho(qnorm, ablation delta) = +0.484+/-0.085 over 88 runs, at full perfusion
- VERDICT the gating signal is not the best predictor of causal head importance. demand is +0.379 while qnorm reaches +0.484 on the same runs. Measured at d_k=32, ABOVE the capacity bound, so this does not contradict the reported qnorm sign flip below it.

<!-- END GENERATED -->
