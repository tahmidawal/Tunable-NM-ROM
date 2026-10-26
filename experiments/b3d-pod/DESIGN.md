# Burgers 3D: qualify the frozen POD hyper-reduction comparator

Status: design approved by the independent reviewer through coordinator messages;
code, controls and exact job configuration remain gated.
The governing campaign design is the coordinator's `experiments/b3d-refine/DESIGN.md`.

Reuse the archived rank-512 central POD basis and the existing ECSW supports k8 and
k16. No basis training, weight fitting or tuning grid. All three arms use the full
central finite-difference residual, or its archived weighted restriction, BDF2 at
$\Delta t=1/60$, one backward-Euler start, the common LM production solver,
$c_0=\Phi^T u_0$, and six output times from zero to one quarter.

The initial pilot uses validation cases 0, 5, 30 and 43 (seed 923801). It is a
diagnostic subset and cannot qualify the complete validation population. A full
validation job, if supported by measured pilot cost, uses ordered cases 0 through
63. Total cap: four GPU-hours, including at most one GPU-hour for the pilot.

All arms retain historical Q1: valid finite trajectories, at most one percent pooled
nonstationary production steps under a fresh Jacobian, distance to native tight
anchors at most $2.5\times10^{-4}$ on every case, and tight-to-tighter at most
$2.5\times10^{-5}$ on the two prescribed initial cases. Native anchor roundoff
handling remains the original implementation; no tolerance is relaxed here.

New conservative hyper-reduction admission contract: save every accepted step's
history, weighted residual gradient $g_E=J_E^T r_E$, full gradient $g_F=J_F^T r_F$,
and denominator $D=\|J_F\|_F\|r_F\|$. Use the archived square-root weights in the
same unnormalised objective units. Require $\|g_E-g_F\|/D\le10^{-3}$ on every step
and full stationarity $\|g_F\|/D\le10^{-3}$ on at least 99 percent of pooled steps.
When $D=0$, the ratio is zero for a zero numerator and infinity otherwise; reject
nonfinite inputs. Save the actual per-step metrics. Additionally require every-case
rollout distances k8 to k16, k16 to no-HR and k8 directly to no-HR at most $2.5\times10^{-4}$.
The projected-advection relative discrepancy is diagnostic only.

Tighter-anchor case ids are [0, 5] for the diagnostic pilot and [0, 1] for full validation.

Production costs are paired on one physical GPU, with burn-in and coefficient
fingerprints binding each repetition to its scored trajectory. Initialization,
evolution and decoding split measurements are diagnostic; full fused queries are
the comparison cost. Persist all repetitions, including drift-rejected triples.
Offline basis/setup and memory are separate. Existing archived training cost stays
historical, not remeasured.

Stop on provenance, precision, backend, cohort or schedule mismatch; missing outputs;
nonfinite values; memory risk; or a measured pilot projection beyond the cap. A
scientific failure of admission is retained as a measured failure and does not
release a broader tuning search. Keep sealed cohorts closed.

## Glossary

- POD: a linear basis derived from earlier training snapshots.
- ECSW: weighted sampling of the full least-squares residual.
- k8 / k16: archived supports of eight / sixteen points per reduced coefficient.
- Q1: nonlinear solver validity, stationarity and tight-anchor checks.
- HR admission: the new whole-residual gradient and trajectory consistency checks.
- LM: the reused damped least-squares nonlinear solver.
- Tight / tighter: independently stricter native nonlinear solves.
- E0: the approved refined reference, with conditional uncertainty.
