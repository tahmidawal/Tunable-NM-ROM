# Algebraic multigrid in the accuracy-matched cylinder comparator (fixed 2026-11-13, committed 00:45 EST in e60f943b0, before any AMG run)

Question (reviewer iMp8, multigrid): the accuracy-matched finite-element comparator of the cylinder-flow rebuttal
(held-out Re 105) solves its linear systems with a sparse direct factorisation. Is it faster or slower when those solves
use algebraic multigrid (AMG)? The answer is reported whichever way it goes.

## What the comparator is (read from the code, not from prose)

Source lane `exp/2026-11-03-cylinder-unsteady-rom`, `experiments/cylinder-unsteady-rom`:
`artifacts/rb-fb/results/timed/SELECT-re105.json` selects `c4_p3_dt300_frozen_oseen_K30`: mesh c4, Taylor-Hood P3/P2,
16,912 unknowns, step 1/300 (510 steps over the 1.7 s window), scheme `frozen_oseen` of `vendor/ref/transient.py`,
refresh K = 30, median stage wall time 42.25 s over 5 repetitions (rb-fb, job 7140851, node pax024, Xeon Gold 6448Y,
16 threads, shared node).

Corrections to the brief, from the code:
- The scheme is `frozen_oseen`, not `oseen_bdf2`: BDF2 with the convection frozen at w_f = u^n, the matrix
  coef*M + A + C(w_f) assembled and LU-factorised (UMFPACK, `inverse_backend`) every K = 30 steps, an explicit
  second-order correction for the convection change in between, and ONE direct solve with the stored factors per step.
  There is no GMRES in this scheme (GMRES is used only by `oseen_bdf2`). The linear system is the full monolithic
  velocity-pressure saddle-point system; there are no separate velocity and pressure solves inside the timed loop.
- The `spl.factorized` calls at `fe_ladder2.py` lines 93-94 factor the velocity and pressure L2-projection mass matrices
  used ONCE to project the reference start state onto the comparator's space. They are outside the timed window
  (stage wall time) for every entry, so replacing them cannot change the timed cost. They are left as they are.
- In the 5 timed repetitions the stepper's own timers give, per run: factor (assembly + LU, 18 refreshes) ~3.9-4.1 s,
  solve (bc, residual, LU back-substitution, residual checks) ~10.5-11.9 s, convection apply 11.6-25.4 s,
  observe 7.9-10.3 s. The direct linear algebra is therefore ~15-16 s of the 42 s; the rest is unchanged by any solver.

## Variants (only the linear solve of the frozen step changes; everything else is byte-identical)

Implemented as a subclass of the vendored stepper (`amg_stepper.py`, vendored file unchanged) that overrides the
frozen-scheme refresh (`prepare_frozen`) and solve (`solve`). The assembly of coef*M + A + C(w_f) and of C(w_f), the
right-hand side, the explicit correction, outputs and all bookkeeping are those of the original.

Saddle-point system on the free unknowns: [F B^T; B 0] with F = coef*M_u + nu*K_u + C(w_f) (the two velocity components
decouple; one AMG hierarchy is built per component block, or one shared hierarchy when the two blocks are identical) and
B = -div. Schur complement approximation (Cahouet-Chabard, the standard one for time-dependent (Navier-)Stokes with a
dominant mass term): S^{-1} ~ -(coef * L_p^{-1} + nu * M_p^{-1}), L_p the pressure Laplacian on the pressure space with
p = 0 imposed at the outlet (the do-nothing boundary), M_p the pressure mass matrix (applied through its diagonal).
L_p and M_p do not change in time and are set up once per stepper (construction, untimed, like the original's matrix
setup), and are recorded.

- `lu`    : the original (UMFPACK LU every K steps, one direct solve per step), run through the same script and
            timed in the same allocation (paired control).
- `amg1`  (variant 1, block preconditioner): right-preconditioned flexible GMRES (pyamg fgmres, restart 60) on the full
            saddle-point system with the upper block-triangular preconditioner [F B^T; 0 S]; F^{-1} = one V-cycle of
            pyamg smoothed aggregation per velocity component (symmetry='nonsymmetric'), L_p^{-1} = one V-cycle of
            smoothed aggregation (SPD), M_p^{-1} = inverse diagonal. Hierarchies rebuilt on the original refresh schedule
            (every K = 30 steps and at every coefficient change, i.e. exactly where the LU was refactored).
- `amg2`  (variant 2, AMG where the direct solver was): the same outer Krylov solver and block structure, but the
            velocity and pressure systems are SOLVED (not just preconditioned) by AMG: F by AMG-preconditioned GMRES and
            L_p by AMG-preconditioned CG, each to inner relative tolerance 1e-8 (i.e. AMG replaces the direct solves of
            the separate velocity and pressure systems; the outer FGMRES then only couples them). Same refresh schedule.
- `amg1rs` (screen only): variant 1 with classical Ruge-Stueben AMG instead of smoothed aggregation for F and L_p.
- `amg1m` (screen only): variant 1 with the pressure-mass-only Schur approximation S^{-1} ~ -(1/nu) M_p^{-1}
            (the steady-Stokes choice), recorded to show whether it is adequate at this step.
Initial guess for every Krylov solve: the previous step's solution, extrapolated linearly (2x^n - x^{n-1}) when the
two previous solves used the same coefficient (standard for iterative time stepping). Iteration counts, Krylov
residuals, setup (hierarchy + matrix extraction) and solve seconds are recorded per run.

## Linear tolerance and stopping

The original checks ||K x - rhs|| / ||rhs - K x_bc|| <= 1e-10 on refresh steps and every 200 steps (the LU gives ~3e-16).
Iterative variants stop when ||r - K d|| <= rtol ||r|| with r = rhs - K x_bc on the free unknowns (the same relative
residual as the original check), at every step; non-convergence within 300 iterations fails the entry.
Tolerance ladder on development Re 95: rtol in {1e-10, 1e-8, 1e-6, 1e-4}.

## Selection (development Re 95 only), then held-out

Development job (Re 95, one run each, also timed once): `lu`; `amg1` and `amg2` at all four rtol; `amg1rs` and
`amg1m` at rtol 1e-6. Accuracy is scored exactly as `fe_ladder2.py` (same reference, quadrature, start projection,
six targets of rom_r4.TARGETS: e_u RMS <= 1e-2, e_u max <= 3e-2, e_p RMS <= 2e-2, St rel <= 1e-2, CD_max rel <= 2e-2,
CL_max abs <= 2e-2).

Rule per variant (amg1, amg2): the loosest rtol of the ladder at which (a) all six targets are met on Re 95 and
(b) every one of the six metric values is within 1 % (relative) of the `lu` run's value on Re 95, i.e. the AMG run
reproduces the LU discretisation rather than a different, perturbed one. If the screening shows `amg1rs` faster than
`amg1` at equal accuracy, `amg1rs` replaces `amg1` as variant 1 (recorded as an amendment before the held-out job).

Held-out job (Re 105, opened once for this lane): the selected setting of each variant and `lu`, one scored run
each plus 5 timed repetitions, all in ONE allocation, order lu, amg1, amg2, lu (the second `lu` block checks drift).
Timed quantity: identical to the original - stage wall seconds of the stepper (setup/construction excluded), same
mesh, order, step, window (1.7 s, 510 steps), outputs (`timing_mode`), 16 threads. Pass rule: all six targets on
Re 105, same thresholds. Reported: median/min/max wall seconds per variant vs the paired `lu` and vs the 42.25 s of
record, targets met, Krylov iterations (mean/max per step), setup vs solve seconds.

## Hardware

CPU partition `batch`, constraint `sapphirerapids` (the original node pax024 is a Sapphire Rapids Xeon Gold 6448Y in
the gpu partition; the GPU was never used by the FE). 16 CPUs, 96 GB. The CPU model is printed by the job. Because the
node model may differ from pax024, the claim rests on the PAIRED `lu` timing in the same allocation; the 42.25 s
record is quoted next to it.

## Not covered (to be stated)

AMG implementations other than pyamg (e.g. NGSolve's own h1amg/BDDC, hypre BoomerAMG, PETSc fieldsplit),
monolithic AMG, geometric multigrid, other Schur approximations (PCD, LSC), and larger meshes where AMG's
better asymptotic scaling may win. pyamg's kernels are single-threaded (so is UMFPACK's factorisation/solve).

## Amendments

- S1 (2026-11-13, committed 00:51 EST, smoke amg-smoke2 job 7161713, Re 95, first 0.2 s = 63 solves, plumbing only; job 7161708
  failed on a missing staged import): every kind runs. Node pax028, Xeon Gold 6438M (Sapphire Rapids). Wall seconds:
  lu 8.2 (factor 2.5, solve 2.3), amg1 SA 15.3 (Krylov 9.5 s, 27.9 FGMRES iterations per solve, max 50), amg1 RS 15.4
  (28.7 / 64), amg2 130.0 (17.9 outer, 91,582 inner iterations). e_u/e_p of the three AMG runs equal the LU values to
  <= 2e-5 relative. amg1m (mass-only Schur) fails: relative residual 8.5e-3 after 120 iterations at the first step, as
  expected when coef*M dominates (recorded; it stays in the development job with the 300-iteration cap).
- D1 (same time, before any development result): the development job is split into three jobs run concurrently (same
  partition/constraint, different directories) only for wall time, because amg2 costs ~2 s per step: dev-a = lu, amg1
  ladder, amg1rs, amg1m; dev-b = lu, amg2 1e-10, 1e-8; dev-c = lu, amg2 1e-6, 1e-4. Each repeats lu so that every job has
  its own paired LU timing; the accuracy rule uses the lu of dev-a (LU is deterministic, the three must agree, checked).
- H1 (2026-11-13, committed 01:04 EST, after dev-a 7161797 was read, before Re 105 is opened; Re 95 values in
  artifacts/amg-deva). Rule applied to amg1: rtol 1e-8 is selected (all six metrics within 0.06 % of lu; 1e-6 meets
  6/6 but CL_max differs by 2.0 % and e_p RMS by 1.0 %, outside the 1 % reproduction band; 1e-4 fails 4 targets). amg1rs
  at 1e-6 fails three targets (pressure error 0.18), so it does not replace amg1; amg1m fails to converge (residual 2e-3
  after 300 iterations). Changes, made because amg2 (dev-b/dev-c) needs much longer: the held-out timing is split into
  two allocations, each with its own paired lu: ho-a = lu, amg1 r1e-8 (selected), amg1 r1e-6, lu_b; ho-b = lu, amg2
  (selected when dev-b/dev-c finish), lu_b. amg1 r1e-6 is added as a DECLARED SECONDARY row: the loosest tolerance that
  meets all six targets on Re 95 (the setting most favourable to AMG); it is reported whatever it shows on Re 105.
- H2 (2026-11-13, committed 01:33 EST, after dev-b 7161798 and dev-c 7161799, before ho-b; artifacts/SUMMARY-dev.md). Rule
  applied to amg2: rtol 1e-8 is selected (all six metrics within 0.007 % of lu; 1e-6 meets 6/6 but St rel differs by
  2.0 % from lu's; 1e-4 fails 4 targets). LU agrees across the three dev jobs to rounding. ho-b = lu, amg2 r1e-8, lu_b.
  No secondary amg2 row is timed on Re 105: on Re 95 its loosest passing setting (1e-6) costs 824 s, 16x the paired LU,
  so it cannot change the conclusion and would cost another ~70 min of a 16-core allocation.
- Note (02:20 EST): the amendment times above were first written as estimates that ran ahead of the clock; they are
  replaced by the commit times (git log), which are authoritative. Order and content are unchanged.
