# Frozen Poisson mesh-transfer development pilot

This cell tests the current continuous-coordinate separable NM-ROM against an
FFT DST-I full solver. Numerical results are provisional until collected with
verified hashes and reviewed. Existing sealed final cohorts are untouched.

The tracked configuration fixes the training checkpoint, new development seed and
cohort count, mesh interval convention, weak mode budget and solver tolerance
ladder before any pilot output is inspected. Both trained networks stay frozen.
Each new mesh gets a newly evaluated spatial bank and exact weak operator. The
source-field projection uses only the required thin sine matrices. No quadrature
fit or field fitting is needed: initialization is the mean training latent and all
source dependence enters the weak PDE equations. Gaussian descriptors are retained
only to regenerate the physical source across meshes.

Every query receives the same full host source array and returns a complete host
solution array. Input, projection/initialization, latent solve and output have
synchronized timestamps inside the total timer. Deliberate tolerance stops and
measured stationary solves are distinguished; a small projected residual is never
used as a full-state error certificate. Component synchronization is part of this
measured implementation and can affect latency. Setup/compilation is separate.

Direct DST solves the same five-point discrete Poisson equation to roundoff. A
separately labeled coarse DST arm restricts the supplied source, solves and returns
bilinearly prolonged output; restriction/interpolation and full input/output are
charged. This is an initial coarse-grid cost envelope, not an exhaustive search.

The independent SciPy DST reference uses successively refined finite-difference
meshes. All physical errors use the same nested observation nodes. The last
refinement difference supplies conservative empirical uncertainty without a
Richardson reduction; this is refinement evidence, not a rigorous PDE error bound.
Accuracy qualification requires every declared source to satisfy the error target
including uncertainty and the reference uncertainty budget. Timed repetitions are
retained with the output hash, latent, error, status and solver counters from that
same invocation. Development accuracy qualification is not final-cohort evidence.

Run `test_core.py` locally with the repository's guarded GPU command. Commit source
before `cluster.py stage pilot01`; then `cluster.py submit pilot01`. Collection
verifies input/result/pull hashes, creates an archive and deletes the exact remote
attempt only after validation. All experiment writes belong to this worktree.

## Plain-language glossary

- **Intervals / nodes / unknowns:** cells per axis / sampled coordinates including
  walls / interior solution values the discrete equation determines.
- **NM-ROM / FOM:** nonlinear reduced model / full discrete model.
- **Bank / head / latent:** learned spatial functions / map producing their
  coefficients / compressed coordinates solved from the weak PDE.
- **DST-I:** a discrete sine transform appropriate to zero Dirichlet walls.
- **Weak operator / modes:** PDE equations averaged against smooth sine functions /
  the chosen averaging functions. Exactness refers to the discrete operator.
- **Tau / stationarity:** requested reduction of the initial weak residual /
  normalized gradient showing whether further local minimization is possible.
- **Physical error / same-grid error:** discrepancy from refined reference /
  discrepancy from the full model on the reduced model's own mesh.
- **Uncertainty / refinement:** estimated remaining reference error / recomputing
  with smaller grid spacing.
- **Paired repetitions / median:** alternating comparable runs on one GPU /
  middle timing value, not a best-case selected value.
- **Prolongation / cost envelope:** interpolating a coarse result to requested
  output coordinates / cheapest measured valid configuration for a target.
- **Manifest / checksum / namespace:** file inventory / content hash / isolated
  cluster directory prefix owned by this experiment.
