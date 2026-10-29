# Shared Burgers head: affine16 initialization of the fine64 solve

This lane implements Test A of the coordinator's binding `experiments/burgers-shared/DESIGN.md` in the `2026-10-28-burgers-shared` worktree. It is a fixed development cost screen; no numerical result or speed claim exists yet. Root alone owns GPU submissions, accounting, shared training and the canonical log.

At each time step keep the last48 coordinates of the original fine temporal predictor fixed, optimize its first16 coordinates with the same fine residual, guard the proposal by a relative squared-residual decrease of1e-8, and then release all64 coordinates to the original fine LM. Auxiliary damping resets to1e-6; the fine damping carry is unchanged. Invalid or unconverged auxiliary exits restore the original predictor. This affine slice is not the standalone zero-tail D16 decoder and is not multigrid.

$$z_w=(a_w,b_w),\qquad a_*\approx\arg\min_{a\in\mathbb R^{16}}\|r_n(h([a;b_w]))\|^2.$$

$$z_{\rm start}=\begin{cases}[a_*;b_w],&\text{valid and }\|r_n(h([a_*;b_w]))\|^2\le(1-10^{-8})\|r_n(h(z_w))\|^2,\\z_w,&\text{otherwise.}\end{cases}$$

`pilot.py --model-dir <bundle> --out <new-output>` requires the coordinator's completed h16/h32/h64 bundle and MODEL.json. It fixes L1024 intervals, M256 frozen test modes, Gauss128 quadrature, BE dt0.005 for50steps, and the original production initial fit and fine solver. Only the existing dev6 cohort is generated. Seven randomized, interleaved complete-query timing repetitions include initialization, auxiliary solves, guards, fine evolution and six full output fields. No old-model cold baseline is substituted.

The runner checks exact zero-padding prefix transfer for fields and spatial derivatives, nonzero expansion directions, true16-column Jacobians against AD, full-cohort disabled/original parity, forced-failure fallback, all-step fresh fine stationarity, accepted-candidate monotonicity, and timing witnesses. `stepcore.py` copies the retained fine solver and adds only the seed callback and diagnostics. `checks/check_source.py` verifies selected retained-source invariants without importing JAX. Full numerical controls run only inside an approved cluster allocation.

`analyze.py <result.json> --out <summary.json>` independently reconstructs the cost gate and checks retained witnesses, counts and certificates. The screen passes only when all controls pass, the median of the six paired median ratios is at most0.85, and no case exceeds1.10. A pass is provisional until independently reviewed conditional tight/tighter and quadrature checks run. Those checks are deliberately not implemented or claimed by this phase1 lane.

`cluster/stage.py --model-dir <bundle> --dest <new-directory>` stages committed source and the immutable model bundle locally with hashes. It does not create a job, contact the cluster or change the shared7200-second ledger. Do not synchronize any generated data/ directory. The coordinator supplies mandatory GPU/f64/highest-precision preflight and an allocation bounded by the shared remaining budget.

## Plain-language glossary

Fine64: the final solver adjusts64 latent numbers. Affine16: it first adjusts16 numbers while retaining the other48 from its temporal prediction. Decoder/head: the neural map from those numbers to the fixed spatial bank. L: intervals per axis; L1024 has1025 boundary-inclusive nodes. M: number of residual test modes. Quadrature: points and weights approximating spatial integrals. BE: backward Euler time stepping. LM: damped nonlinear least-squares solver. AD: automatic differentiation used for independent checks. Dev6: the existing six exposed development cases. Witness: the exact retained arrays corresponding to a timed query. Paired ratio: candidate runtime divided by direct runtime in the same repetition and case. Stationarity: the specified normalized gradient criterion or residual criterion has been met. Conditional accuracy: stronger agreement checks allowed only after the cost screen passes.
