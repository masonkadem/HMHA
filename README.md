# hemo-attn

Hemodynamically constrained attention. A fixed total blood supply is allocated across
attention heads, starving the ones that do not demand it, so the surviving circuit is
small enough to verify mechanistically.

## The one-sentence claim

A metabolically constrained transformer discovers its own minimal circuit size and
confines it spatially, verified on a benchmark where the true circuit is known.

## Evidence so far

**Established, 5 seeds, CUDA, `multi_relation` at H=32, d_k=32, N=16, R=4.**

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
when the number of territories drops below the number of roles.

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
tests/                 ground-truth assertions. Run these first.
```

## Run

```bash
pip install -r requirements.txt
pytest -q tests/                                     # ~3 min CPU, must pass
python experiments/run.py --seed 0 --supply threshold --demand outnorm_ema
python experiments/run.py --seed 0 --supply territory --n_territories 8
```

Each seed writes `results/<task>_<supply>_<demand>_T<n>_tau<d>_s<seed>.pkl`.
Long runs get killed in some environments, so always run one seed per process.

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
- Report the loss at B = k*, not B*. B* saturates and cannot separate arms.
- Thresholds are relative to the dense-to-trivial gap, never to the dense loss alone.
- Negative results stay in the paper. Two of the three findings so far are negative.

## Prior work to cite and differentiate

Michel et al. 2019 (head importance), Voita et al. 2019 (L0 gated head pruning,
specialized heads do the heavy lifting), Zhai et al. 2023 (attention entropy),
EMNLP 2020 weak-link paper (random pruning is competitive), Causal Head Gating
NeurIPS 2025 (learned head gates for interpretability),
Kozachkov/Slotine/Krotov (astrocytes as a substrate for attention; motivates the
astrocyte-transformer correspondence but does NOT test supply-based gating, so do not
cite it as support for this mechanism).
