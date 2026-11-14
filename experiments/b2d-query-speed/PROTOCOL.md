# b2d-query-speed: faster online query of the Burgers 2D linear rung at fixed accuracy

Lane `worktrees/2026-11-14-burgers2d-query-speed`, branch `exp/2026-11-14-burgers2d-query-speed`, forked from
`exp/2026-11-13-iso-error` @ 2fd48d20f. Cluster namespace `/cluster/tufts/paralab/tawal01/b2d-query-speed-20261114/`.
Written 2026-11-14 before any job of this lane was submitted and before any outcome of this lane existed.

## 1. Question

Can the online query of our own method (frozen bank, linear rung, 256^2, 50 backward-Euler steps) be made faster
at unchanged accuracy by changing only the online reduced solve? Second: why is width 512 less accurate than 384?

Known before this lane (iso-error job 7175375, test64, A100 80GB; width / median % / worst % / ms):
96: 0.529/3.85/15.27; 128: 0.268/2.55/18.45; 192: 0.0973/1.57/25.58; 256: 0.0485/1.16/33.81;
384: 0.0264/0.875/45.55; 512: 0.0407/1.38/59.99. Operators (burgers-ops-matched): FNO F0 0.195 % median in
4.06 ms, U-Net U0 about 0.21 % in 7.4 ms. Reconstruction is below 1.6 % of the query. These numbers are the reason
for the lane; none of them was produced by a setting of this lane.

## 2. What is fixed and what may change

Fixed: checkpoint `sep_hfit_dense_mid_N256_dense.pkl` (SHA 18f0266a...), rotation `rotation_R512.npz`, the nested
bank prefix R', test count M = 4R' and rule lat64 for the frozen settings, the initial fit, the decoder, the mesh,
dt = 0.005 with 50 steps, the six output times, the reference solver `fft_tight`, the error measure
(max over the five evolved output times of ||u - u_ref|| / ||u_ref(0)||, per case; median and worst over cases).
The time step is not a tuning knob here: the reference is the discrete backward-Euler solution at dt = 0.005, and
the stored iso-error sweep shows that dt = 0.01 alone costs about 1 % against it.

May change (online solve only): how the reduced nonlinear least-squares problem of each step is solved (predictor,
number of iterations, stopping, damping, reuse of a lagged Jacobian / factorisation / pseudo-inverse, exact or
lagged gradient), arithmetic precision of solver-internal operators, kernel fusion, loop emission and compile
mode, batching of queries. Width, M and rule are changed only in the diagnosis arms of section 6 and in the
equal-error width choice of section 5; such arms are named as different reduced models, never as speedups.

The unchanged baseline is the iso-error primary arm text per width (`B2.make_linear_query`, eng, chol, clip,
damping carry, quadratic predictor, gtol 1e-2, primary compile mode of iso-error PROTOCOL section 2), built by the
same driver in the same job.

## 3. Cohorts

- Training: `params_draw(0, 128)` (never evaluated; overlap asserted by the driver).
- Development (all tuning, profiling, diagnosis and selection): **dev38** = `params_draw(7090702, 4)` ++
  `params_draw(911702, 2)` (the existing dev6) ++ `params_draw(2026111401, 32)` (new validation draw; seed chosen
  now, checked unused in LAB-LOG.md and in every worktree's code and configs).
- test64 = `params_draw(20260916, 64)`, SHA cd058fd2...6044527c. Evaluated ONCE by this lane, after the freeze.
- fresh64 = `params_draw(2026111402, 64)`: sealed. The seed is recorded here before the draw; no job touches it
  before the freeze commit; evaluated ONCE.
- The driver asserts that dev38, test64, fresh64, training and the certificate population are pairwise disjoint.

No job on test64 or fresh64 is submitted before the commit "freeze" that adds `frozen.json`. Development jobs may
be repeated freely on dev38.

## 4. Candidates (decided before outcomes; details may be amended, with the reason, in section 9)

Step problem: minimise ||r(w)||, r(w) = A w - S.(A w_prev) + dt S.Pq^T adv(G5 w), S = 1/(1 + dt nu lam).
Baseline: about one Levenberg-Marquardt iteration per step; per step three residual evaluations for the
predictor guard, two residual+Jacobian evaluations, J^T J, a Cholesky factorisation and solve.

Candidate family "qs" (new module `qsfast.py`):
- predictor: quadratic extrapolation of the last three states, no residual guard;
- k fixed corrections per step, w <- w - P r(w), with a lagged pseudo-inverse P = (J_ref^T J_ref + ridge)^-1 J_ref^T
  refreshed every s steps (s in {1, 2, 5, 10, 25, 50}; s = 50 is one factorisation per query), k in {1, 2, 3, 4};
- variant "grad": w <- w - H_ref^-1 J(w)^T r(w) (exact least-squares gradient, lagged Gauss-Newton matrix);
- variant "picard": P = A^+ computed offline (no online factorisation at all);
- emission variants: explicit pseudo-inverse / kept factor / fused step matrices; scan unrolling; compile mode;
  float32 stencil contraction (mixed precision) with the update in float64.

The lagged-P fixed point satisfies J_ref^T r = 0, not J(w)^T r = 0: it is a Petrov-Galerkin solution with a lagged
test space, not the exact least-squares stationary point. This is stated wherever such a setting is reported, and
its accuracy is measured, not assumed. No safeguard is added silently: a candidate that produces a non-finite
field or a divergent step on any dev case is rejected for that width.

## 5. Selection rule (dev38 only)

Per width R' in {96, 128, 192, 256, 384, 512}, with e_med, e_worst the dev38 median and worst error and
b_med, b_worst those of the unchanged baseline in the same job:
- **accurate**: the fastest candidate (dev median GPU time) with e_med <= 1.02 b_med and e_worst <= 1.02 b_worst
  and every per-case error <= 1.05 x the baseline's error of that case + 1e-5 (absolute, in units of relative
  error), finite everywhere;
- **fast**: the fastest candidate with e_med <= 1.10 b_med and e_worst <= 1.10 b_worst, finite everywhere, if it
  is at least 10 % faster than "accurate"; otherwise no "fast" setting is frozen for that width.
"fast" is reported as its own accuracy point with its own measured error, never as a speedup of "accurate".
At most these two settings per width are frozen. If no candidate satisfies "accurate" at a width, the baseline
stays and that is reported.

Equal-error width for the operator comparison: among widths {128, 144, 160, 176, 192} the smallest width whose
frozen "accurate" setting has dev38 median error <= 0.195 % (the FNO test64 median); frozen with the rest.

## 6. Width-512 diagnosis (dev38)

Arms at R' = 512 (and the same at 384 as control): baseline; gtol 1e-4 and 1e-6 (tolerance); M = 3R' and 2R'
(fewer high test modes on the same lattice); rule lat128 (127^2 nodes; quadrature resolution); plus two numbers
needing no solve: cond_2 of A[:, :R'] and of J at t = 0, and the projection floor (least-squares fit of the
reference fields in the span of the R' bank columns). Reading: if the floor at 512 is below 384 but the query
error is above, the loss is in the reduced problem (tests/quadrature/conditioning/tolerance), and the arm that
removes it names the cause. A fix is adopted for 512 only if it is principled (one rule stated for all widths, or
a statement that the full bank is outside the certified range) and it then counts as a different reduced model.

## 7. Final evaluation (once)

One job per cohort (test64, fresh64), A100 80GB: per width the unchanged baseline, "accurate" and "fast"; errors
against `fft_tight` computed in the job; timing by the iso-error rule (A-B-A, randomised order seed 20260924,
0.1 s burn before each invocation, cooldown between phases, block_until_ready on both sides, warm-up excluded,
2 x 2 repetitions per case, median over invocations; drift and neighbour gates at 1.1; spread reported as the
5-95 % range of per-case medians and min/max). Single-query latency only in these tables.
Batched throughput (vmap over B queries) is a separate table, labelled as such, never mixed with latency.

Operators: if the frozen F0/U0 4,608-trajectory checkpoints under `rebuttal-burgers-ops` are readable, they are
re-timed in the same job on the same GPU by burgers-ops-matched `evaluate.py` conventions (batch 1, 20 burn-in
calls, 5 repetitions per case, synchronised) on test64 and evaluated on fresh64 (reference fields generated by the
same `fft_tight` solver, `burgers_fom.make_fom`). Checkpoints are read only; nothing is retrained or altered.

Extension to 512^2 and 1024^2 (frozen settings only, same rule) if time allows; otherwise stated as not run.

## 8. Reporting

All numbers in RESULTS.md are generated by `summarise.py` from the job JSON. Reported whatever the outcome,
including "no improvement". A setting that is faster with higher error is listed as a different accuracy point.
Every job log must show `jax_backend=gpu`. GPU jobs: at most 3 queued or running; short walls.

## 9. Amendments

(none yet)
