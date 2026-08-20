# Phase-4 P4-D resource estimate and implementation checkpoint

Status: prospectively recorded after the excluded local execution smoke and before
any scientific P4-D staging or submission.  It is resource planning, not scientific
evidence and not an additional search cell.

The one authorized diagnostic evaluates H1 and H2 on the complete exposed
selection cohorts.  There are 64+32+16=112 trajectories, all 51 times, hence 5,712
joint sparse projection fits per arm and 11,424 fits total.  Each fit uses the locked
single-threaded SuperLU normal-equation solve; `B10_ORACLE_WORKERS=8` runs at most
eight independent fits concurrently.  H1 has 3,328 coefficients and H2 has 4,608.

The excluded N24/H1 one-fit smoke took 0.0525 s for its oracle fit.  That number does
not predict the larger sparse factorization cost.  The complete execution smoke,
including the actual H1 Pallas mandatory/max-one compile and execution paths, took
33.9864 s on the local GB10.  It used 33.65--33.68 MB of compiled eligibility memory
for the reduced one-step route.  These are smoke diagnostics only and cannot be used
for a cross-job scientific timing claim.

The scientific request is one H200, eight CPUs, 64 GB RAM, and eight hours.  The
walltime deliberately allows for sparse-factor growth at N128/N256, two full
N1024 hierarchical compile/identity routes, four live tight+tighter reference
chains, and the balanced 20-repetition five-method cost panel.  The scientific
compiled-device-memory hard gate remains 20 GB per selected route; a larger
allocation does not relax it.  BLAS remains pinned to one thread per fit.

The stage binds the exact commit, the complete immutable S0 and P3 artifact chains,
the P3 staged manifest, and the external Burgers FOM/data source hash carried by
that manifest.  `bh_common.py` must be clean and present at the exact P4 commit.
The pull path independently verifies all hashes, scheduler state, GPU/precision,
health-warning absence, negative or positive scientific gates, and removes only
the exact completed remote cell after a passing audit.
