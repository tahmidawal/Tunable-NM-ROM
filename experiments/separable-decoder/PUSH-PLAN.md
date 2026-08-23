# N=256 focused push — accuracy AND speed (2026-08-23)

User-directed pivot: the four-resolution scaling round is closed (its findings live on the
sepdec-n* branches and in LAB-LOG.md). This round pushes BOTH fronts as hard as possible at
one resolution, N=256, both PDEs. No hard job cap — iterate with judgment, report per round.
Cluster namespace: /cluster/tufts/paralab/tawal01/n256_push/ (one subdir per job).

## Starting point (verified, in runs/ on this branch)

- Poisson best: K16/R64/200k cached 2.54 ms @ 2.89e-2 held (3.53e-2 fresh) — a Pareto
  point vs the whole CG ladder (loosest useful rung 7.9 ms @ 1.1e-2), 100% censored at
  tau=1e-3, oracle-saturated on held data but NOT on fresh (~2x objective gap at N=128).
- Burgers best: K16 cached e2e 106.9 ms @ 2.45e-2 vs tolerance-Newton 60.5 ms @ 3.0e-4 —
  classical wins. IC fit ~18 ms of the e2e. All ROM steps stop on 'stalled'.
- f(K) flat K16->K32 => capacity/generalization-limited, not K-limited.

## ACCURACY TARGET (user directive, 2026-08-23, supersedes "as far as it goes")

The accuracy goal is **rel-L2 ~1e-3 solve error**, not merely improving on 3e-2.
**Poisson is the primary 1e-3 target.** For Burgers, the POD-floor diagnostic
(below) decides feasibility BEFORE jobs are sunk into it: if the trajectory-family
floor says 1e-3 is infeasible at practical R (<=512), say so and propose the
multi-bank span extension as the follow-up rather than brute-forcing R.

Target ladder — every rung must reach ~1e-3 for the goal to be met:

- recon <= ~5e-4 (evidence it is approachable: the N=512 j4 cell hit rel-MSE
  1.75e-5 ~ 4e-3 rel-L2 at 108k steps and was still falling; plan 200-300k
  steps, R per the POD diagnostic, multi-scale Fourier features, denser
  mu-sampling for fresh-cohort generalization),
- oracle ~1e-3 (held AND fresh),
- objective gap closed (weak-EQ optimum tracks the L2 oracle),
- solver terminating on tol, not stalls (censoring -> 0 at the reported tau).

### POD-floor diagnostic (IMMEDIATE, done while round-1 jobs run)

The separable architecture's oracle is hard-lower-bounded by the rank-R POD
floor of the training family (all x-dependence lives in span{g}, dim R). So the
f64-Gram POD spectrum of the N=256 training snapshots (both PDEs) and the
rank-R projection floors for R in {64,128,256,512} tell us the R required for
1e-3 to be possible at all, per PDE, before any training job runs.
`pod_floor_n256.py` + committed numbers in `runs/pod_floor/`. GRAM64-style f64
Gram per project rules. POD is a bound here, never a model ingredient.

### Architecture and optimization are IN SCOPE (user authorization, 2026-08-23)

1e-3 is to be treated as an ARCHITECTURE AND OPTIMIZATION problem: the decoder
architecture and the training scheme may change, not merely be tuned. Picked by
evidence (POD floors, ladder measurements), not all at once:

- Capacity/structure of both tracks: multi-scale Fourier features (built,
  `FF_SCALES`), wider/deeper g and h, R per the POD diagnostic, and the
  multi-bank span extension u = sum_j bc*g_j(x)^T h_j(z) (or low-rank
  z-modulated bank mixing) if a single span floors above target. The ONLY
  architectural invariant: the model must still collapse to cached banks at
  any fixed point set — that separability IS the method.
- Training/optimization: 200-300k+ steps, two-stage schemes (fit codes+span,
  then joint refine), L-BFGS polish on the small nets, EMA weights,
  restarts/annealed ff_scale, stronger orthogonality/conditioning constraints
  on G, weight decay tuned for fresh-cohort generalization, denser
  mu-sampling / more snapshots (append extra seeds; NEVER change the
  canonical seed-0 draw or the cohort definitions).
- PURE NEURAL stands: no POD basis inside the model. SVD is a diagnostic
  (floors) only. If a POD-initialized span ever looks decisively better than
  anything neural we can train, DO NOT implement it — report the finding
  upward for a user decision.
- Every new architecture's banks re-pass gate 0 (<=1e-12); incumbent
  discretization/residual definitions, no-test-truth, audit measurement
  rules, ladder reporting, per-round analysis all unchanged.

Strategy order: drive recon to <=5e-4 first (it floors everything), then
verify oracle ~1e-3 on FRESH data, then re-certify the objective (M/m/EQ
tails) at that scale. Poisson primary; Burgers effort gated on its POD-floor
verdict.

### Objective certification at 1e-3

M=4K (64 modes) leaves residual components outside the test span uncontrolled,
and EQ row tails (1e2-1e4) are fatal at 1e-3 scale. Rounds >=2 must include
arms scaling M well beyond 4K (128/256 modes) with m scaled accordingly and
tail-capped NNLS, and must measure weak-optimum vs L2-oracle tracking
explicitly — **that gap IS the objective-truncation error**. Report the online
cost growth honestly: still N-free, but bigger constants.

## Measured error ladder (keep reporting all rungs every round)

train recon <= representation oracle <= weak-EQ optimum <= solver output.
Every accuracy claim names which rung moved. Every round reports the ladder for its cells.

## Round 1 — solver + objective + IC (the measured slack, no architecture change)

Accuracy levers, in one Poisson job and one Burgers job:
1. LM termination repair: adaptive trust region (Burgers TR currently clamped 0.01x),
   damping-schedule tuning, restart-on-stall from perturbed z, more iterations. Target:
   stop-reason distributions no longer 100% 'stalled'; quantify how much error was
   "solver gave up".
2. Weak-EQ objective on fresh data: M 4K->8K arm; EQ tail control — cap per-row NNLS
   errors (current p95 0.1-0.4, max 1e2-1e4), larger m arm (32K); one adaptive-quadrature
   arm (cheap solve -> re-select EQ points at the solution -> re-solve).
3. Burgers IC: high-budget t=0 fit (it's one-time cost) + train a small offline encoder
   u0 -> z0 (train-data only) as an init; report ic_rel before/after.
Speed lever in the same jobs (free): latent warm-start extrapolation (2-step) for the
Burgers rollout; count jacobians before/after.

## Round 2 — representation ceiling -> the 1e-3 ladder (informed by round 1 + POD floors)

- R set by the POD-floor diagnostic (the floor at the chosen R must be <= ~5e-4
  for the 1e-3 goal to be reachable); 200-300k training steps.
- Multi-scale Fourier features: bands ff_scale {1,4,16} concatenated (n_ff split across
  bands), vs single-scale control.
- Generalization: weight decay on h, wider/denser training sampling, more snapshots
  (data is cheap to regenerate at N=256) — sized using the dense-family arm of the
  POD diagnostic (how much of the fresh floor is sampling density).
- Objective certification arms per the 1e-3 section above: M in {128, 256} modes with
  m scaled and tail-capped NNLS; weak-opt vs oracle tracking reported per arm.
- Optional (flag before running): test-time refinement of h on the PDE residual
  (g frozen => caches stay valid; NO truth data — this is legitimate and must be timed
  as part of the solve when reported).

## Round 3 — speed measurement done right

- Batched multi-query: solve ALL test sources / trajectories in one batched (vmapped)
  solve; report per-query amortized time AND single-query latency separately.
- Fuse the LM loop: move stall/restart control into lax.while_loop so one solve is one
  kernel chain; measure launch-overhead reduction.
- Encoder-init (from round 1) to cut iteration counts.
- Report the full honest grid: {single-query, batched} x {ROM, CG ladder / tol-Newton},
  same-job, same GPU, end-to-end including IC fit and full-field decode, raw reps kept,
  balanced order. The scaling-round measurement rules in HANDOFF.md remain binding.

## Non-negotiables (unchanged)

Gate 0 (<=1e-12) in every job on every arm incl. modified solvers; incumbent
discretization untouched (solver INTERNALS — damping, TR schedule, restarts, batching —
may change; the residual/Jacobian definitions and tolerances being reported against may
not); no test-truth in any solve path (encoder trained on training data only); fresh-seed
cohort in every round; summaries generated by script; lab-log append per session.

## Definition of victory (both fronts, honest)

- Accuracy (updated by the 2026-08-23 user directive): Poisson fresh-cohort solve error
  ~1e-3 with the whole ladder at ~1e-3 (recon <= ~5e-4, oracle ~1e-3, objective gap
  closed, solver terminating on tol). Burgers: trajectory error at the level the POD
  floor certifies as feasible at practical R (<=512); if 1e-3 is infeasible there, an
  explicit feasibility statement + the multi-bank span extension proposal is the
  deliverable, with zero IC failures at whatever level is reached.
- Speed: single-query e2e at least at parity with the best classical rung of equal
  accuracy, and batched amortized cost decisively below it. All claims Pareto-framed
  against the full classical ladder.
