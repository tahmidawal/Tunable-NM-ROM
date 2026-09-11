# Final bounded Poisson correction family

This protocol is predeclared before query evaluation. It tests one frozen widened nonlinear head with one training-only linear residual basis; it combines added correction capacity with exact analytic elimination, and is not a pure width-only solver comparison. All development cases are already opened, final cases remain sealed, and no further search follows this family.

The retained `r128_joint` checkpoint and the 32-direction basis are content-hash-pinned in `config-correction-accuracy.json`. Directions come from the right singular vectors of normalized training residuals in the exact physical QR metric. They use the saved training codes, which are optimized training variables rather than independently certified stationary reconstructions. No evaluation field enters basis construction. Use the nested prefixes with 8, 16 and 32 directions, with 32 declared primary; retain both the original rank-64 checkpoint and the uncorrected rank-128 checkpoint as paired controls. No new neural training, bank modification or latent-head width change occurs.

For frozen spatial bank $G$, nonlinear head $h$, correction directions $C_q$, nonlinear coordinates $z$ and linear coordinates $y$, the decoder is

$$D_q(z,y)=G\bigl(h(z)+C_qy\bigr).$$

The old manifold is included exactly at $y=0$, including values and derivatives with respect to $z$. The nominal coordinate counts are 24, 32 and 48. The nonlinear optimizer keeps 16 coordinates because the linear block is solved exactly. This distinction remains explicit in all tables.

Let $B$ be the exact contracted weak bank and $b$ the supplied-source weak target. With $A_q=BC_q=Q_qR_q$, form the projected operator and target:

$$B_q=(I-Q_qQ_q^T)B,\qquad b_q=(I-Q_qQ_q^T)b.$$

Solve the unchanged guarded nonlinear least-squares problem for $B_qh(z)-b_q$, then recover

$$y=R_q^{-1}Q_q^T\bigl(b-Bh(z)\bigr).$$

This minimizes the full weak objective over the linear coordinates for every nonlinear iterate. Offline matrices are rebuilt per mesh and prefix; the retained weak test count remains 257, comfortably above every nominal coordinate count. At zero correction count the objective, source projection, initialization and gradient must reproduce the uncorrected model. The source-only cold start compares projected training predictions with the projected supplied target. No descriptor, evaluation truth, or oracle latent is used online.

Both full and projected normalized gradient criteria must pass their unchanged threshold. Their denominators differ, so both are recorded explicitly. Retain projected-Jacobian singular values/rank, correction-matrix rank/conditioning, linear recovery backward error, full/projected residual reconstruction, actual solver exits, all latent and correction coordinates, and every repetition. Query timing includes source input/contraction, projected cache search, nonlinear solving, linear recovery, complete decoding, full/projected stationarity and projected-rank diagnostics, and output copying. Numerical failures remain failures, even if their physical error is small.

Tests must prove old-manifold value/Jacobian inclusion; zero-prefix legacy solve agreement; exact physical-metric basis orthogonality; full/eliminated residual and gradient consistency; and generic-source initialization. GPU preflight, double precision, highest matrix-multiply precision, per-invocation burn-in and complete integration smoke precede the real job. No cost comparison across jobs is used.

Evaluate all five neural methods plus named CG tolerances and direct DST in the same private GPU job, on 64, 256 and 1024 intervals, all 42 already-opened development cases and three paired repetitions. Retain full same-invocation fields, reference refinement, bank projections, source/checkpoint/basis/operator hashes and every numerical diagnostic. Exact bank projection remains a separate diagnostic: corrections do not change the bank or its already missed proposed 3% development target. The 5% physical eligibility and reference allowance also remain unchanged. A physical pass will not be relabeled as a bank-target pass. No success or speedup is predicted by training reconstruction alone.

All results require complete checksum collection, restored archive verification, independent NumPy/SciPy field/source/gradient/initialization audit, retained checkpoints/basis and exact remote removal. The existing worktree and namespace remain separate; root owns main reports and any future merge decision.

Glossary: **bank** is the learned spatial feature matrix; **head** maps nonlinear coordinates to coefficients; **correction directions** are fixed linear coefficient vectors learned from training residuals; **prefix** chooses the first directions of a shared ordered basis; **QR** gives an orthogonal field or weak metric; **variable projection** solves the linear coordinates analytically while optimizing the nonlinear ones; **nominal dimension** counts all decoder coordinates; **stationarity** is the normalized gradient check; **weak target** is the exact supplied-source contraction; **CG/DST** are iterative/direct Poisson controls; **development** cases may guide experiments but are distinct from sealed final confirmation.
