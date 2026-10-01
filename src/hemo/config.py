from dataclasses import dataclass, field


@dataclass
class Cfg:
    # --- task ---
    task: str = "multi_relation"        # multi_relation | qsa_cross
    d_model: int = 128
    seq_len: int = 16                   # N
    n_rel: int = 4                      # R. Ground-truth k* for multi_relation.
    m_content: int = 16
    q: int = 2                          # qsa_cross only

    # --- architecture ---
    num_heads: int = 32
    d_k: int = 32

    # --- demand signal: what the astrocyte senses ---
    #   outnorm_ema   slow integration of what the head writes  (the HMHA hypothesis)
    #   outnorm_inst  instantaneous magnitude       (standard L2 pruning baseline)
    #   qnorm         query L2 norm                 (falsified control, rho < 0)
    #   random        fixed random ranking          (lower bound)
    #   reserve       CEREBROVASCULAR RESERVE. Loss drop when this head is given a
    #                 large extra share of the FIXED supply, taken from the others.
    #                 Nonlinear and supraphysiological by design, so unlike a gradient
    #                 it separates a head at its ceiling from a starved head with a
    #                 latent role. Every head-importance score in the literature is
    #                 first-order or leave-one-out REMOVAL; this is addition under
    #                 conservation, which has no analogue among them.
    #   deficit       -dL/dg, the per-head shortfall. This is gradient head importance
    #                 (Michel et al. 2019) once projected onto the conservation surface,
    #                 so it is carried as a BASELINE, never as the contribution.
    #   q2norm        squared query norm            (sharper than qnorm)
    #   entropy       attention sharpness           (rho=+0.45 vs ablation)
    demand: str = "outnorm_ema"
    ema: float = 0.99
    delay: int = 0                      # tau, steps of lag between demand and supply
    pool_beta: float = 0.0              # 0 = per-head demand, 1 = fully territory-pooled

    # --- supply mechanism: how blood is allocated ---
    #   topk        conserved budget, exactly B heads     (ranking; loses to magnitude)
    #   threshold   theta = mean + kappa*std, B EMERGES   (not a ranking)
    #   territory   shared arteriole per group of heads   (non-local; no pruning analogue)
    #   poiseuille  flow ~ radius^4 within the perfused set   (graded, not equal share)
    #   watershed   territories with a GLOBAL infarct floor   (compaction is possible)
    #   watershed_auto  the same, with the local threshold driven by the autoregulated
    #               loop instead of a fixed kappa. Territory supply is otherwise an
    #               OPEN-loop rule and so inherits the impossibility of Theorem 1
    #               applied per territory.
    #   gap         cut at the largest gap in sorted demand   (shape, not quantile)
    #   otsu        two-cluster split of demand               (shape, not quantile)
    #   autoreg     closed loop: kappa tracks a loss DEFICIT  (task-referenced)
    supply: str = "threshold"
    n_territories: int = 8
    kappa_start: float = -3.0           # theta far below the demand mean, all heads perfused
    kappa_end: float = 1.5              # progressive ischemia
    leak: float = 0.0                   # residual gate on starved heads
    kappa_ceil: float = 3.0             # autoreg only, upper clamp on the control state
    autoreg_gain: float = 0.003         # autoreg only, proportional gain. The loop must
                                        # be SLOWER than the learner it steers; at 0.05
                                        # it drove a limit cycle between 4 and 32 heads.
    autoreg_every: int = 1              # autoreg only, controller update interval
    loss_ema: float = 0.0               # autoreg only, smoothing on the sensed loss.
                                        # 0 senses the raw minibatch loss, which is what
                                        # the reported autoreg runs used.
    stall_gate: float = 0.0             # autoreg only. >0 dilates only when the loss is
                                        # above target AND its relative improvement rate
                                        # has fallen below this, i.e. chronic rather
                                        # than acute deficit. 0 disables.
    flow_exponent: float = 4.0          # poiseuille only. Flow ~ r^n; n=4 is Poiseuille,
                                        # n=1 recovers a linear share.
    cvr_boost: float = 3.0              # reserve only. Flow multiple applied to the
                                        # probed head, paid for by the rest.
    cvr_every: int = 100                # reserve only. Steps between reserve probes.
    flow_price: float = 1e-4            # marginal only. Metabolic price per unit flow.
                                        # A head is perfused while the loss reduction it
                                        # would buy exceeds this. Units are loss per unit
                                        # flow, not loss, so unlike target_frac it does
                                        # not have to be calibrated to the task's scale.
    terr_kappa: float = -9.0            # watershed only. Threshold on AGGREGATE territory
                                        # demand. Above -9 the territories compete for
                                        # total flow and a whole territory can go dark;
                                        # at the default they split flow equally, which
                                        # is no competition at all.
    watershed_penalty: float = 0.5      # watershed only. Demand multiplier for heads at
                                        # a territory boundary, which in tissue sit
                                        # between two arterial beds and perfuse worst.
    target_frac: float = 0.02           # autoreg only, target loss as a fraction of
                                        # the trivial baseline. Scale-free across tasks.
    precondition: float = 0.0           # autoreg only. Ischemic preconditioning. When a
                                        # constriction has to be reversed, cap kappa this
                                        # far below where the lost head went dark, so the
                                        # loop stops re-testing a head it needs. 0 disables.
    precondition_relax: float = 2e-5    # autoreg only. Per-step drift of that cap back up,
                                        # so the memory fades if the task changes.

    # --- local supply: each head regulates its own flow, no loss target ---
    #   local   every probe_every steps, measure each head's own deficit: the loss the
    #           model would lose if that head's flow were given to the others, or gain
    #           if a starved head were fed. Close the cheapest open head if it is worth
    #           less than a metabolic price per head; reopen the most valuable starved
    #           head if it is worth more than twice that price.
    #   prune   BASELINE, standard iterative head pruning (Michel et al. 2019): 0/1 masks,
    #           not conserved, remove the head with the smallest |dL/dmask|, stop when the
    #           loss bar (target_frac) breaks or, with prune_stop = 0, at the true k*.
    price_frac: float = 0.01            # local only. Price of one head, as a fraction of
                                        # the trivial loss.
    probe_every: int = 25               # local and prune. Steps between decisions.
    taper: int = 0                      # local only. Steps over which a closing head's
                                        # share fades to zero. 0 shuts it at once.
    probe_batch: int = 512              # local only. Fresh sequences per probe, so the
                                        # re-fit has far more tokens than features.
    prune_stop: int = 1                 # prune only. 1 finds k by the loss bar; 0 is
                                        # told k* (an oracle upper bound).
    # --- controls on the local rule (thesis proposal). Defaults reproduce the rule. ---
    conserve: int = 1                   # local only. 0 = CONTROL: no fixed total, an open
                                        # head keeps gain = its tone, so survivors never
                                        # take up a closing head's share.
    local_value: str = "refit"          # local only. Which head to close:
                                        #   refit   collateral value, others re-fit (the rule)
                                        #   ablate  CONTROL: standard ablation importance,
                                        #           the loss rise with NO re-fit
                                        #   random  CONTROL: close when the rule would, but
                                        #           pick the open head at random
                                        # Reopening always uses the re-fit gain.
    plant_copies: int = 0               # 1 = start with head h and head h + H/2 identical,
                                        # so every head has an exact duplicate.
    # --- trial closure (local only): once the rule has been quiet for trial_wait probes,
    # close the cheapest open head even if it looks essential, fade it over trial_taper
    # steps, reopen it at once if the smoothed loss leaves the solved bar (target_frac), keep
    # it shut if the task stays solved for trial_settle more steps. 0 = off. ---
    trial: int = 0
    trial_taper: int = 300
    trial_settle: int = 200
    trial_wait: int = 4
    trial_cooldown: int = 20
    trial_undo: float = 1.0             # undo when the smoothed loss passes trial_undo x bar
    trial_max_fail: int = 1000          # stop trying after this many failed trials
    trial_stop: float = 1.0             # no new trials after this fraction of training
    # --- BASELINE l0 (supply="l0"): learned hard-concrete head gates with an L0 penalty
    # (Louizos et al. 2018; Voita et al. 2019), penalty switched on at budget_hold_frac ---
    l0_lambda: float = 0.01
    l0_init: float = 3.0
    # --- damage: when is a backup head worth keeping? (proposal Aim 3) ---
    head_dropout: float = 0.0           # hemo only. Each training step every head fails
                                        # independently with this probability (inverted
                                        # scaling, as dropout). Evaluation is undamaged.
    probe_masks: int = 32               # local only, used when head_dropout > 0. The
                                        # collateral value is averaged over this many random
                                        # failure patterns, so a backup is worth what it
                                        # saves when others fail.

    # --- budget schedule, topk only ---
    budget_min: int = 1
    budget_hold_frac: float = 0.25
    budget_anneal_frac: float = 0.5

    # --- optimisation ---
    batch_size: int = 128
    steps: int = 4000
    scratch_mult: float = 1.5
    lr: float = 2e-3
    lr_warmup: int = 200
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # --- eval ---
    val_every: int = 100
    val_size: int = 1024
    gap_tol: float = 0.02               # threshold = full + gap_tol*(trivial - full)
    abs_floor: float = 1e-4
    seed: int = 0
    device: str = "auto"
