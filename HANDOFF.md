# Shortest path to a paper and a thesis chapter

Read `notebooks/walkthrough.ipynb` first. It executes in seconds off the committed
pickles and walks the entire result in order.

---

## The claim, stated as it can currently be defended

> A conserved supply field over attention heads, steered by a metabolic deficit rather
> than by a fixed threshold, selects the minimal sufficient circuit in a single training
> run. A fixed-quantile rule provably cannot, and the starved model is exactly, not
> approximately, a model of the selected size.

Two things make this publishable and both are already done. The **impossibility result**
is a proof, not a measurement, and it explains a number we then measured. The **exact
reduction** means the reduced model needs no error bound, which no pruning paper can say.

The claim that is **not** defensible, and must not be written: that metabolic constraint
*discovers* circuit size from first principles. The loop finds the smallest circuit
meeting a performance bar, which is close to the definition of the minimal circuit. It is
an efficiency result. Say so.

---

## Status

### Done

- [x] Benchmark with a known circuit. `k* = R` verified by dense-from-scratch training at
      R = 2, 3, 4, 6, 8. Head roles are named and recoverable.
- [x] **Proposition 1**, exact reduction. The starved layer *is* a B-head layer; starved
      heads get zero gradient.
- [x] **Theorem 1**, impossibility. `B = H(1 - F_d(kappa))` for standardised demand.
      Measured slope 0.29 against an ideal 1.00.
- [x] **Theorem 2**, the loop's fixed point is the minimal sufficient B.
- [x] **Proposition 2**, the stall gate makes the loop control the converged loss.
- [x] Autoregulated supply tracks `k*` exactly at all five sizes, slope 1.00, MAE 0.00.
- [x] Target control: stall gate holds MAE to 0.28 heads over a 10x target range against
      2.07 for the plain loop, and converts an order-of-magnitude failure into off-by-one.
- [x] Four null results, each with what it rules out.
- [x] `paper/chapter.tex`, `paper/main.tex`, both figures, both tables, prose macros. Every
      number generated from the pickles.

### Blocking publication

- [ ] **`stall_eps` (12 runs).** Is the stall threshold itself tuned? Proposition 2 says B
      is insensitive to it. *If this fails the paper's central claim weakens to "we moved
      the hyperparameter".* Highest priority by a distance.
- [ ] **`slowgain_ctl` (6 runs).** Is the gate just a slower loop? Plain autoreg at
      `eta/10`. A reviewer will ask this in the first round.
- [ ] **Seeds.** Most arms are n = 3. Take the two closed-loop arms to n = 10 before
      submission; nothing else needs it.
- [ ] **`arm_stall` at k\* = 3 and 6 (6 runs).** So the stall arm is compared against the
      plain loop on the same five sizes rather than a subset.

`make controls` runs the first two. 18 runs, about 10 minutes on a GPU.

### Would strengthen it, in order of value per run

- [ ] **`qsa_kstar` (24 runs).** A second benchmark where `k*` comes from *capacity*
      rather than from function. Turns "works on our task" into "works on circuit size".
      Best single addition.
- [ ] **`watershed_auto`, `watershed_compete`, `compaction` (36).** The only route back to
      the spatial half of the original claim, which is currently dead. Optional for the
      paper, valuable for the thesis.
- [ ] **`arm_reserve` + `baseline_gradient` (27).** Cerebrovascular reserve, addition under
      conservation, against the gradient-importance baseline it must be distinguished from.
- [ ] A local per-head deficit driving the loop instead of the global loss. This is the
      one change that would remove the tuned target entirely and answer the sharpest
      objection. Not yet implemented.

### Do not bother

`sensors`, `arm_marginal`, `arm_poiseuille` (73 runs). Poiseuille already failed with the
same perfused set as the fixed threshold, which is the informative result; more of it adds
nothing.

---

## The order to do things

1. `make test` then `make controls`. Read the two numbers. **If `stall_eps` shows B moving
   with epsilon, stop and rewrite the claim before writing anything else.**
2. `make figures`, then read `paper/chapter.tex` Section 7 and update it to whatever the
   controls said.
3. Seeds to 10 on the two closed-loop arms.
4. `qsa_kstar`. If it tracks there too, promote generality into the abstract.
5. Submit the conference paper from `paper/main.tex`. Expand `paper/chapter.tex` with the
   watershed and reserve arms for the thesis.

## Venue

The impossibility result plus a working repair is a **methods** contribution, not a
neuroscience one. Target an interpretability or ML venue. The astrocyte connection is an
identity of algebraic form and is scoped that way in the text; do not lead with it, and do
not let it drift into a biological-plausibility claim, which is the one thing a
neuroscience reviewer will reject outright.

---

## Instructions for Claude, to paste at the start of a session

> Read `CLAUDE.md`, `README.md` and `HANDOFF.md`. Run `pytest -q tests/` before anything.
>
> Goal: get this to a submittable paper. Work the "Blocking publication" list in
> `HANDOFF.md` in order. After each batch of runs, regenerate with
> `python experiments/aggregate.py --readme README.md && python experiments/paper_assets.py`
> and update `paper/chapter.tex` Section 7 to match what the controls actually showed.
>
> Rules that matter here: every number in a figure, table or the prose is generated from
> the run pickles, never typed. Negative results stay. Any field a sweep varies must be in
> `TAG_FIELDS` or `TAG_OPTIONAL` in `experiments/run.py` or two arms silently overwrite
> each other. Any new supply mechanism goes into `test_supply_is_conserved`. Do not claim
> biological support from the astrocyte literature for supply-based gating.
>
> If a control fails, say so plainly and weaken the claim in the paper rather than
> reframing around it.
