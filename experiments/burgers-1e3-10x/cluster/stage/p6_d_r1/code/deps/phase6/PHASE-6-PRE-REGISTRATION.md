# Burgers 1e-3 / 10x Phase-6 numerical-identity repair preregistration

Status: prospectively locked after the audited Phase-5 P5-D hard stop and
before any Phase-6 scientific output. Phase 6 contains exactly one diagnostic
cell, P6-D. It repairs one numerical implementation mismatch in the selected
G1 path; it does not change the nonlinear manifold, decoder space, latent
dimension, generator, predictor, weak objective, data, loss, or gate.

## Immutable evidence and exposed diagnosis

P6-D binds the completed P5-D JSON, NPZ, independent AUDIT, and staged root
manifest from job 2669249. Their SHA-256 values are, respectively,
`97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96`,
`5235b81b19c4ed459e7fda4291fe67eb3f4b87ba07413eb36e861a0b147dfe54`,
`c84ee29e1b9fe84f5e90949e18be26c07a6c54c00320a2f7a82bd1cb8dee0eff`,
and `6135791d3a5cca08b0ff1c424d93451579f3dd1048bef2a5cf314e2b8bf317d6`.
The scientific source commit recorded by P5-D is
`e18edda9b124be8f7fa21c07ef21804c8dbecc48`.

P5-D exposed that G1's K3 and Cox fields agree to at most
`2.858010930216804e-16` relative L2 and its weak stencils to about `2e-16`,
but the weak residual differs by `3.533981881245193e-14` to
`4.8255283342982255e-14`. A read-only replay of the worst case through one
shared downstream accumulator retained a `4.8329267510683603e-14` residual
difference. The temporal and upwind-advection projection differences were
`3.778377375085906e-14` and `4.351891213127314e-14`; the diffusion projection
was `1.0156454739677864e-15`. Thus separate XLA fusion/reduction order is not
the active cause. Cox-versus-Horner decoder roundoff is amplified by temporal
cancellation and the N=1024 `dx^-1=1023` upwind difference.

These diagnostics are exposed development evidence. They fix the repair below
and may not be used to change its method or tolerance after P6-D starts.

## Fixed method and finite bracket

Only Phase-5 arm G1 is eligible: H1 global R48 plus central P32, k=24/q=19,
M=96, m=384, the 30,594-parameter DualUpConv16 generator with the immutable
P5-D train-only coefficient mean/scales, and the unchanged 2,104-parameter
direct state predictor. Parameters are the same deterministic nonconstant
cost-screen values as P5-D. No learned training is performed.

P6-D reports two G1 routes:

- `R0_polynomial_weak` is the immutable P5-D control: polynomial/Horner weak
  stencils and K3/Pallas full-grid decode.
- `R1_cox_weak_k3_full` is the only candidate. It evaluates the mathematically
  identical clamped Cox--de Boor basis for the fixed weak current/previous
  stencil queries, then applies the unchanged exact FOM upwind weak residual.
  Its final 51 by N^2 fields still use the unchanged K3/Pallas decoder.

Both generate the 51 coefficient grids exactly once before the 50 weak
evaluations. R1 is the actual timed route, not an audit-only substitution. Its
charged invocation also repeats the locked at-most-64-by-64 initial-condition
sample, nine-point local log-quadratic fit, and raw seven-feature construction,
as in P5-D. Its
identity probe computes states and coefficients once, applies one common Cox
weak operator to the candidate and Cox control, and independently compares the
K3 full decode with the Cox full decode. It returns the actual candidate
fields, weak current stencils, previous centers, residuals, and rho values.
The Cox control is never eligible or used as a deployment denominator.

The repair is mathematically equivalent: it changes only the arithmetic used
to evaluate the same fixed cubic weak-query basis. It introduces no POD/linear
skip, full-grid correction, FOM fallback, extra state, grid-growing learned
parameters, altered weak form, or altered accuracy tolerance.

## Locked data, truth, and timing

The 35,904 Phase-5 coefficient targets are not regenerated or refit. P6-D reads
only the immutable coefficient mean and two head scales from the P5-D NPZ and
checks them against the audited P5-D JSON. No training or selection fields are
loaded.

The paired cost panel regenerates only the locked seed-20260822 N=1024 live
cases 0:4. The tight exact-Helmholtz reference is outer/inner `1e-12/1e-7` and
its independent tighter chain is `3e-13/3e-8`; both must be finite, have zero
flags/breakdowns, meet their returned-residual tolerances, and differ by at
most `1e-4`. The like-for-like FOM is cubic history, exact Helmholtz,
outer/inner `3e-3/1e-1`, and is eligible only with mean trajectory error
`<=1e-3`, worst `<=3e-3`, and zero health failures.

Timing uses one H200 job and one invocation for each method's timing, accuracy,
and work. Methods are `fom`, `R0_polynomial_weak`, and
`R1_cox_weak_k3_full`. There are exactly 24 repetitions per trajectory and a
cyclic/reversed three-method order with exactly eight observations of every
method in every clock position. Each timed block follows an immediate GPU burn;
one warmup per case/method precedes timing. Lowering, compilation, first
execution after compilation, steady-state arrays, per-trajectory medians,
within-trajectory outliers, and compiled memory are all persisted. No wall
clock is compared across jobs.

## P6-D promotion gates and hard stop

R0 is descriptive and cannot license training. R1 passes only if all conditions
hold simultaneously on all four live cases:

1. every K3-full/Cox-control output component--full fields, weak current
   stencils, previous centers, weak residual, and rho--has global relative L2
   `<=2e-14`, with finite arrays and exact binary boundary;
2. the independent K3 and Cox basis/support probes pass, with maximum combined
   local support 32;
3. every untimed canonical record has output shape `[51,1048576]`, exactly 50
   weak objective evaluations, zero Jacobian/trial evaluations, exactly 51
   coefficient-grid evaluations, finite residual/rho arrays, and zero failures;
4. compiled eligibility device bytes are `<=20,000,000,000`;
5. the paired live FOM is healthy and accuracy eligible;
6. paired median end-to-end speedup is `>=10x` and the trajectory-clustered 95%
   speedup lower bound is `>=8x`.

The clustered interval uses the same four-trajectory resampling estimator as
P5-D with fixed seed 20266100. There is no absolute millisecond gate.

If R1 passes, the immutable decision is `repair_licensed=true`, but this only
licenses a separate proposal/audit for G1 seed-11 training. Phase 6 itself does
not authorize training. If any gate fails, `phase6_hard_stop=true`; there is no
second arithmetic route, tolerance change, retry, or training cell.

## Cell cap and untouched locks

Phase 6 authorizes exactly one scientific cell, P6-D. Infrastructure-only
attempts with zero scientific output do not change its method or cap, but any
completed scientifically valid P6-D consumes the cell and cannot be repeated.
Generator training, loss changes, seed retraining, weak/EQ rollout, N256/N512
scaling, model-validation indices 576:640, and seed-20261031 confirmation are
not authorized and remain untouched.
