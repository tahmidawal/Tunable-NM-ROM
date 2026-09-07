# Verified heat port and first frozen-mesh pilot

This is the pinned protocol for a restricted development pilot, not a statement
of achieved accuracy or speed. It establishes a current continuous-bank heat port
before a broader multi-bump heat cohort and per-resolution optimization.

The unit-square problem is $u_t=\nu\Delta u$, with zero Dirichlet boundaries,
no source and fixed $\nu=0.02$. The initial field is

$$u_0(x,y)=16x(1-x)y(1-y)a\exp\left[-\frac{(x-c_x)^2+(y-c_y)^2}{2s^2}\right].$$

The exact family bounds, random seeds, output times and dimensions are frozen in
`config-pilot.json`. This single-bump family is narrower than the archived heat
family. Gaussian parameters generate data only; they never enter the decoder head
or initialization path. The final evaluation cohort is unopened.

The architecture imports only the current `sep_common.features`, initialization
and nonlinear `head`: random-Fourier coordinate lift, learned spatial MLP, hard
polynomial boundary factor, and a SiLU coefficient MLP plus linear skip. Both
networks train jointly with per-snapshot latent codes. Equal relative snapshot
weights prevent decayed fields from disappearing in the loss. No CP head, old heat
decoder, POD initialization or Gaussian-descriptor conditioning is present.

The five-point negative Laplacian has orthonormal sine eigenvectors and eigenvalues
$\lambda_{ij}$. Independently checked FFT-based DST-I propagation evaluates the
semi-discrete FOM directly at requested times, with no artificial time-step count.
A refined spectral-in-space solution using continuum sine eigenvalues supplies
physical-reference values. Comparing two independently regenerated spectral grids
bounds interpolation/truncation uncertainty; direct nested-node restriction avoids
interpolation. Discrete eigenmode checks, an independent SciPy DST, stencil
diagonalization, second-order spatial refinement and Crank–Nicolson time refinement
are mandatory before training.

For every mesh the frozen bank $G$ is reevaluated and the exact weak matrix
$A=\Phi^T G\,\Delta x^2$ is rebuilt. The test set has comfortably more modes than
latent coordinates. The normalized Crank–Nicolson weak residual is

$$r(z_{j+1})=Ah(z_{j+1})-q\odot Ah(z_j),\qquad
q_i=\frac{1-\nu\lambda_i\Delta t/2}{1+\nu\lambda_i\Delta t/2}.$$

This is the Crank–Nicolson projected residual weighted by its inverse diagonal
left-side factor. It is minimized by damped monotone LM. The scan carries the
new decoded state, never the previously satisfied implicit operator. An exactly
representable linear mode exercises the actual scan and solver; a deliberately
frozen negative control must fail. The trained bank's weak residual and Jacobian
are checked against the explicit stencil separately at each mesh.

For initialization, $G=QR$ compresses the full-field least-squares objective:
$\|Gh(z)-u_0\|^2=\|Rh(z)-Q^Tu_0\|^2+\|(I-QQ^T)u_0\|^2$.
The full input projection is charged per query; grid-sized work stays outside LM.
The nearest training snapshot code and training-code mean are both fitted, and
the lower objective is selected. The physical initial error and both starts'
stationarity/counters are retained. Truth reconstruction uses extra multistart
fitting only as a separately labeled diagnostic, never to initialize evolution.

The timed contract is a host f64 interior initial field to host f64 interior fields
at all requested times; homogeneous boundary nodes are implied identically for
both methods. It includes input transfer, full-input preparation and initial
latent fitting, evolution, field readout and output transfer. Raw per-phase and
total repetitions, errors, stop reasons and corresponding output fields are
persisted from the same invocation. Compilation, model training, reference-data
generation, coordinate-bank evaluation and new-mesh operator/QR assembly are
separate costs. Paired order alternates forward/reversed after GPU clock burn-in.

Every trajectory reports absolute spatial L2, current-reference normalization and
initial-reference normalization at every time, plus a vanishing-amplitude flag,
energy and actual state change. Unrestricted bank projection, multistart
reconstruction, same-grid rollout discrepancy, CN-only FOM time error, discrete
spatial error and refined-reference error remain distinct. LM stationarity is
not a global-minimum certificate. Failed/nonstationary fits remain in the cohort.
An accuracy target that is not attained has no qualifying speedup.

## Glossary

- **FOM / NM-ROM:** full-grid solver / nonlinear-manifold reduced solver.
- **Bank / head / code:** coordinate-network spatial functions / nonlinear
  coefficient map / compressed unknown coordinates fitted to a supplied field.
- **DST-I / FFT:** sine transform for zero-boundary fields / fast Fourier transform
  used to compute it. **Semi-discrete:** spatially discretized, exact in time.
- **CN / LM / QR:** Crank–Nicolson timestep / damped nonlinear least-squares solver /
  orthonormal–triangular matrix factorization.
- **Weak residual / test mode:** spatially averaged PDE mismatch / sine function
  used to form one average. **Stationarity:** small objective gradient.
- **Intervals / interior unknowns:** cells along one axis / all nonboundary field
  values. **Frozen transfer:** identical learned weights, freshly rebuilt operators.
- **Current / initial normalization:** divide field error by reference magnitude
  at that time / at the initial time. **Time maximum:** largest error on the fixed
  requested output times. **Energy:** half the squared spatial L2 field norm.
- **Reconstruction / rollout:** fitting supplied snapshots / autonomous evolution
  from only the initial field. **Query:** the entire stated input-to-output task.
- **Validation / final cohort:** development cases / independent confirmation cases
  excluded from this pilot. **Provisional:** awaiting independent review.
