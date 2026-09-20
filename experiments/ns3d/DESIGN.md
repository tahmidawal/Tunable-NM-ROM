# Genuine three-dimensional Navier–Stokes: current NM-ROM overnight pilot

Prospective design for the authorized September 20 overnight campaign. No result
is asserted here. The first attempt verifies the physical solver, then trains a
small learned spatial bank and nonlinear head; later comparisons use this same
data and output contract.

Base: `exp/2026-09-17-ns2d` at
`2d70f36a02ef5109f3e406e513dcf89a513fb271`. This lane exclusively owns worktree
`2026-09-20-paper-ns3d` and namespace
`/cluster/tufts/paralab/tawal01/paper_ns3d_20260920/`. No old 2D checkpoint is a
3D model. No historical number transfers into the results.

## Physics and observations

The unit periodic cube has $N$ endpoint-excluded nodes on each axis. The unknown
is the three-component velocity, with zero spatial mean:

$$\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u
=-\nabla p+\nu\Delta\mathbf u,\qquad \nabla\cdot\mathbf u=0.$$

Fourier differentiation, the orthogonal Leray projector
$P_k=I-kk^\top/|k|^2$, and a strict two-thirds cutoff define the semidiscrete
problem. The quadratic term is evaluated as $P(\mathbf u\times\nabla\times
\mathbf u)$, with cutoff applied to input and output. Zero and Nyquist modes
are handled explicitly. This is a Fourier Galerkin approximation, not a
finite-difference stencil or an extruded 2D equation.

The primary FOM is second-order Crank–Nicolson/Adams–Bashforth (CNAB2), with a
second-order startup. Diffusion is solved by the exact diagonal Fourier inverse.
A separately implemented NumPy/SciPy RK4 solver uses advective-form derivatives
and serves as a small-grid independent check. Finer meshes and smaller steps
bound the reference discrepancy. A tight reference is not the only timed FOM:
the later panel searches a time-step ladder against the declared physical target.

Family: the initial velocity is the curl of two nonparallel periodic vector
potentials. Each scalar potential is
$\exp[2\sum_{i=1}^3(\cos(2\pi(x_i-c_i))-1)]$. The common center varies over
the torus, the separation over $[0.12,0.24]$, relative strength over $[0.6,1.4]$,
and viscosity log-uniformly over $[0.002,0.01]$. Directions are fixed nonparallel
vectors, centers separated along a fixed oblique direction. The resulting
velocity has all three components and derivatives on all three axes. No
unobserved family coordinates are supplied to a prediction method: inputs are
the full initial velocity and viscosity. The horizon is $T=0.2$, with six output
times including zero. The pilot uses $N=24$, $\Delta t=0.001$, refinement to
$N=48$, $\Delta t=0.0005$, and additional temporal refinement where necessary.

Training and validation parameters use independent seeds 202609201 and
202609202; a separate seed 202609203 is reserved for the final cohort and is not
drawn by the pilot. The initial pilot has 96 training and 16 validation
trajectories. Parameters are generated row-by-row, so extending a cohort
preserves its existing rows. Every attempt regenerates data from those seeds
on the cluster and persists hashes, parameters and reference fields.

Primary error is $\|\mathbf u_{pred}(t)-\mathbf u_{ref}(t)\|_2/
\|\mathbf u_0\|_2$, summed over all velocity components and nodes. Report
per-case errors, median and worst over evolved times, all-times worst, and
initial compression separately. The physical target is 5%; reference
discrepancy must be at most 0.5% on every verification case before training is
accepted. This is an empirical refinement gate, not a rigorous continuum bound.
Pressure is eliminated by solenoidal testing and is not an output or an accuracy
claim. Divergence, kinetic energy, vorticity and evolution size are diagnostics.

## Learned model and controls

The current model is retained:
$$\mathbf u=G_\theta(\mathbf x)[h_\theta(z)+C_qy].$$
The periodic coordinate MLP produces vector columns, and the same discrete
Leray/cutoff projection makes every bank column divergence-free and mean-zero.
The pilot uses bank rank $R=64$ and latent size $K=8$, with a wider-head
refinement arm on the frozen bank only if underfitting is measured. The bank is
learned; POD is a separate control, never a replacement labelled neural.
Train the bank/head/codes, then use field-metric coefficient training for
bounded head refinement. Persist learning curves, durations and checkpoints.

Whitening uses a thin QR with a finite, full-rank check. Held-out multistart
latent fits are best-found estimates, not mathematical oracle guarantees.
Report bank projection, neural reconstruction, initialization, and solved
trajectory errors separately. Record stationarity as well as objective values.
The representation target is 5% worst; if it fails, corrections remain new
candidates, and the head-only result stays failed rather than blocking all
diagnostic comparisons or changing the gate.

Nested correction directions come from the field-metric POD of training
reconstruction residuals. Primary ladder $q\in\{0,8,16,32\}$ keeps a common
$M=160$ smooth divergence-free vector Fourier tests. The $R=64$ free-bank
endpoint is an unrestricted coefficient model, without redundant latent
coordinates. POD at dimensions 8, 16, 24, 40 and 64 uses the identical data.
Its implicit weak stepping is compared with the neural ladder; efficient
Galerkin stepping is also an admissible classical control.

An implicit-midpoint ROM uses the weak vector momentum residual. Its quadratic
advection tensor is built from the exact dealiased operator and independently
checked against full-grid evaluation. Pressure vanishes against divergence-free
tests. Choosing midpoint for the ROM does not force midpoint on the primary FOM.
Use derivative and chunk-order checks before accepting a cached residual.
An exact tensor replaces quadrature in this pilot; it is only labelled exact
after parity with the selected discrete operator is established.

## Prospective verification and acceptance

- Backend GPU, all floating arrays f64 (complex arrays complex128), and highest
  matmul precision are mandatory for every accepted run.
- Leray idempotence, normalized divergence and spectral derivative identities:
  relative discrepancies below $10^{-11}$ on small synthetic fields.
- Rotational-form versus independently coded advective-form projected RHS:
  relative discrepancy below $10^{-10}$ on dealiased fields.
- A manufactured three-dimensional divergence-free Fourier field has nonzero
  projected advection; analytic forcing verifies temporal order (approximately
  two for CNAB2). Flipping advection sign must produce a clear failure.
- Independent NumPy RK4 trajectories agree with the temporally refined JAX
  solver to the measured time-error budget. No shared RHS function is imported.
- Projected advection, all velocity components, and derivatives along all three
  axes must be nonzero on the interacting family. Deleting the third-axis
  dynamics or projection must fail its corresponding check.
- Unforced semidiscrete advection contributes zero kinetic-energy derivative to
  floating precision; full trajectories have finite fields and the expected
  viscous energy decay, within the observed integration error.
- Spatial and temporal refinement on a separate four-case verification seed
  (202609200) must satisfy the reference budget. If not, increase resolution or
  shorten the attempt to verification only; do not train on unverified truth.
- Small-grid tensor/full-grid contraction and Jacobian-direction comparisons
  must agree below $10^{-9}$. Projection cannot amplify reconstruction error.

The first job requests at most two hours on one GPU. It emits partial JSON and
checkpoints throughout. No successful runtime, nonlinear advantage, monotone
ladder or speedup is promised. The coordinator schedules subsequent training,
operator comparisons and held-out evaluation after reviewing these artifacts.

## Timing, evidence and follow-up

Later matched panels use GPU burn-in immediately before timed blocks, balanced
order, synchronized complete velocity queries and persisted repetition arrays.
Initialization and all requested dense outputs are included. Offline training,
bank/tensor preparation and compilation are recorded separately. Every accuracy
and cost pair comes from the same invocation. No ratio crosses GPU jobs.

FNO3D and periodic U-Net3D are the first operator comparisons; DeepONet and
Transolver remain separately tracked work. All internals are f64/complex128.
The inherited 2D wrappers cannot be used unchanged: output channels, endpoint
convention, coordinates, masks, padding and all convolution axes must be ported.
Final configurations freeze on validation before the reserved cohort is opened.
Independent finalist seeds and held-out field audits precede paper claims.

## Glossary

- **FOM:** the numerical solver on the full spatial grid.
- **NM-ROM:** a reduced numerical model constrained by a learned neural decoder.
- **Bank/head:** learned spatial vector fields/a network mapping latent states
  to their coefficients.
- **POD:** a linear basis fitted to the training snapshots.
- **Correction rank $q$:** extra linear coefficient coordinates solved beside
  the latent state; **$K$** is latent size, **$R$** bank rank, **$M$** test count.
- **Leray projection/solenoidal:** removing the velocity's gradient component/
  divergence-free.
- **Dealiasing:** removing Fourier modes that would corrupt a quadratic product
  through finite-grid wraparound.
- **CNAB2/RK4:** second-order implicit-diffusion/explicit-advection stepping and
  fourth-order explicit Runge–Kutta stepping.
- **Weak residual:** momentum mismatch integrated against smooth vector tests.
- **Tensor:** the precomputed coefficients of quadratic reduced advection.
- **Whitening:** changing coefficient coordinates so Euclidean distance measures
  field error; **stationarity** checks the objective's gradient at a fitted code.
- **Validation/final:** data for selection/data reserved for frozen evaluation.
- **Reference budget:** the allowed discrepancy measured under numerical
  refinement, distinct from the reduced model's target error.
