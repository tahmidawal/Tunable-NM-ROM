# NS3D diagnosis: snapshot floor, manifold fit, and CNAB2 rollout

Development-only diagnosis for why the current NM-ROM misses the pre-registered
5% per-case bar on periodic 3D Navier–Stokes at $N=32$. No number in this file
is a new measurement. The final cohort (seed 202609203) stays closed.

## Question

On the development cohort, at each saved time, how large are

1. the linear snapshot projection floor,
2. the best affine $K$-dimensional section of that subspace (a stand-in for the
   best linear head), and
3. the solved CNAB2 Galerkin trajectory,

and does an oracle periodic shift of each snapshot lower the floor enough to
matter?

## Why this is the next measurement

The frozen final panel (`experiments/ns3d/runs/final07/paper_summary.json`,
job 4027788) already separates three contributions on 32 held-out cases.
Same-grid relative $L^2$ against $\|u_0\|_2$:

| model | initial median / worst | evolved median / worst | cases with evolved error $>5\%$ |
|---|---:|---:|---:|
| NM-ROM $q=0$ | 7.39% / 12.21% | 9.77% / 20.34% | 31/32 |
| NM-ROM $q=256$ | 6.59% / 11.16% | 8.91% / 18.92% | 31/32 |
| free-bank Galerkin, $R=1536$ | 1.57% / 2.03% | 5.04% / 10.17% | 16/32 |
| POD Galerkin $k=320$ | 17.54% / 21.50% | 18.42% / 30.91% | 32/32 |
| POD weak $k=320$ | 17.54% / 21.50% | 18.34% / 30.75% | 32/32 |
| CNAB2 $\Delta t=0.01$ | 0 / 0 | 0.39% / 2.47% | 0/32 |

POD weak and POD Galerkin agree, so the fixed Fourier test space is not the
accuracy gap. $q=256$ removes about one percentage point. The whitened learned
bank already represents $u_0$ to 2.03% worst, and every free-bank trajectory
then grows (median $+3.35$ points, worst $+8.28$). Capacity05, on development
snapshots, found a POD-3072 floor of 3.84% worst and a learned $R=3072$ floor
of 6.32% worst, but never integrated those ranks. The binding unknown is
whether a subspace whose snapshot floor is under 5% stays under 5% when the
coefficients are integrated with the same CNAB2 scheme as the FOM.

## Arms

Fit POD only on training seed 202609201 (512 trajectories, the same 4-copy
integer translation table as coverage04). Evaluate only on development seed
202609202 (16 trajectories). Truth is CNAB2 at $\Delta t=0.001$, $T=0.2$, six
output times. Ranks: 64, 128, 256, 512, 1024, 1536, 2048, 3072, clipped to the
available positive spectrum.

- Snapshot floor of the training POD, per output time.
- CNAB2 Galerkin rollout in that POD at $\Delta t=0.001$, plus the largest
  rank also at the deployed ROM step $\Delta t=0.004$.
- One-interval error: restart from the exact projection of the truth at each
  output time and integrate to the next output. This separates local closure
  error from accumulation.
- Affine PCA of the training POD coefficients at $K\in\{16,64,256\}$. This is
  the best training-fit linear head inside the POD bank, not a solved ROM.
- Oracle-shift floor: remove each snapshot's energy centroid with a Fourier
  shift, fit POD on the centered training snapshots, and shift the
  reconstruction back. The centroid is taken from the truth, so this is a
  representation floor for a shift-equivariant linear bank, not an online model.
- Same-job timings after a discarded warmup: FOM at $\Delta t=0.001$ and
  $0.01$, Galerkin at several ranks, and one production dense weak POD-64
  query. Medians of retained repetitions. No ratio against any other job.

## Pass bar and stop rules

The accuracy target is unchanged: worst evolved relative $L^2\le 5\%$ on every
case. This job does not declare that bar met or missed for a new NM-ROM. It
stops after the development diagnosis. It does not train a coordinate network,
does not retune $q$ or $M$, and does not open seed 202609203.

A later fix is justified only if this job shows a concrete gap:

- rollout near the floor and the floor $\le 5\%$ at some rank: the block is
  reaching that subspace with the current head, and cost is whatever this job
  measures;
- rollout far above a floor that is already $\le 5\%$: the block is the
  projected dynamics;
- oracle-shift floor much lower than the plain floor at the same rank: test a
  shift-equivariant bank next.

## Controls

- Final seed is not read.
- POD uses training snapshots only. Development fields are projected, never
  included in the basis.
- Galerkin uses the production `make_dense_galerkin_run` CNAB2 stepper and
  requires an orthonormal basis, because that stepper takes the mass matrix to
  be the identity.
- Frame 0 of every Galerkin run is checked against the orthogonal projection.
- Errors are recomputed in NumPy from the saved development fields before the
  remote directory is deleted.
- F64 and `JAX_DEFAULT_MATMUL_PRECISION=highest`. GPU preflight is mandatory.

## Glossary

- **Snapshot floor:** error of the orthogonal projection of a true field onto
  the training POD, before any time stepping.
- **Galerkin rollout:** CNAB2 evolution of the POD coefficients, with the
  nonlinearity evaluated on the full grid and then projected back.
- **One-interval error:** the same stepper started from the true field's
  projection, advanced only to the next saved time.
- **Affine PCA head:** a $K$-dimensional linear reconstruction of POD
  coefficients, fit on training snapshots.
- **Oracle shift:** a translation estimated from the true field, used only to
  measure a representation floor.
- **Development cohort:** the 16 trajectories from seed 202609202. The final
  cohort is a different seed and is not used here.

## Amendment — diag02, after diag01

diag01 (job 4139559) is recorded in `results/diag01.md`, generated from that
job's `summary.json`. The plain POD reaches an evolved floor under 5% only at
rank 3072, and CNAB2 in that subspace stays under 5% on every development
case. The centered family, with each training snapshot shifted by its own
energy centroid, has only 237 modes above the spectral cutoff, and the
per-time oracle floor at rank 64 is far below 5%. That center uses the truth
at the future time, so it is not a model.

diag02 asks whether the centroid of the initial field alone is enough. The
basis is still the centered training POD. The solved arm shifts $u_0$ by that
centroid, integrates with the same CNAB2 Galerkin stepper, and shifts the
result back. It also records the illegal per-time floor as a ceiling. Ranks
32, 64, 128, clipped to the available spectrum. Time steps 0.001, 0.004 and
0.01, timed in the same job as the errors, including the shifts. Development
seed only. A rank-64 per-time floor above 1% aborts the job, because diag01
already measured that floor well below 1% and a larger value means the shift
was not reproduced.
