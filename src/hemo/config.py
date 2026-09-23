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
    watershed_penalty: float = 0.5      # watershed only. Demand multiplier for heads at
                                        # a territory boundary, which in tissue sit
                                        # between two arterial beds and perfuse worst.
    target_frac: float = 0.02           # autoreg only, target loss as a fraction of
                                        # the trivial baseline. Scale-free across tasks.

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
