The design should not be implemented as written. The continuum equations and reflective Crank–Nicolson algebra are sound, but the absorbing-boundary semi-discretization is missing; consequently the proposed absorbing ROM residual, W1, and F3 are not valid.

No files were changed and no jobs or JAX processes were run.

## 1. Signs and formulation

The continuum first-order system is correct:

\[
u_t=v,\qquad v_t=c^2u_{xx}.
\]

The outgoing Sommerfeld signs in [WAVES-DESIGN.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-waves-vector/experiments/separable-decoder/WAVES-DESIGN.md:31) are also correct:

- Right-going \(u=f(x-ct)\): \(u_t+c u_x=0\) at \(x=1\).
- Left-going \(u=g(x+ct)\): \(u_t-c u_x=0\) at \(x=0\).

Equivalently,

\[
v_0-c(D_+u)_0=0,\qquad v_J+c(D_-u)_J=0.
\]

Thus the initial velocity must be \(v_0=-cu_x\) for a right-going pulse and \(v_0=+cu_x\) for a left-going pulse.

For a genuine semi-discrete ODE

\[
\dot u=v,\qquad \dot v=c^2Lu,
\]

CN gives, with \(\bar u=(u^{n+1}+u^n)/2\),

\[
\frac{u^{n+1}-u^n}{\Delta t}-\bar v=0,\qquad
\frac{v^{n+1}-v^n}{\Delta t}-c^2L\bar u=0.
\]

Projecting and substituting \(u=Gh_u,\ v=Gh_v\) produces exactly the two residuals in lines 80–81. Therefore those residuals are correct for the reflective interior problem.

They are not correct at Sommerfeld boundaries. There, \(v_t=c^2u_{xx}\) is replaced by a boundary equation involving both \(u\) and \(v\). The design currently applies both interior evolution equations at all \(N\) absorbing nodes without saying how the boundary rows are obtained.

There is also a sign inconsistency with the repository convention. If \(L=D_{xx}\), as required by the stated \(r_v\), then the positive eigenvalues in [b1d_common.py](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-waves-vector/experiments/separable-decoder/b1d_common.py:319) satisfy

\[
L\Phi=-\Phi\Lambda,
\]

so

\[
A=\Phi^\top LG=-\Lambda B.
\]

If instead \(L=-D_{xx}\) is used to obtain \(A=\Lambda B\), then the CN residual must contain \(+c^2A\bar h_u\), not the minus sign currently written. The design cannot retain both its \(r_v\) sign and its W1 identity.

## 2. The central claim

The abstract algebraic claim is true:

\[
u=Gh \quad\Longrightarrow\quad P^\top L u=(P^\top LG)h
\]

for any fixed linear \(L\), regardless of symmetry.

The hidden conditions are that \(L\), \(P\), and \(G\) are fixed and dimensionally compatible. For parameter-dependent operators, exact collapse still works if

\[
L(\mu)=\sum_q\theta_q(\mu)L_q,
\]

but one matrix \(P^\top L_qG\) is needed per affine component. Non-symmetry itself is irrelevant. Even a non-symmetric matrix has a diagonal shortcut if \(P^\top\) consists of its left eigenvectors; self-adjointness is sufficient, not necessary.

Sommerfeld is not an \(L u\)-only boundary modification. With the full \(2N\)-component state, the natural formulation is a DAE:

\[
u_t-v=0 \quad\text{at all nodes},
\]

\[
v_t-c^2D_{xx}u=0 \quad\text{at interior nodes},
\]

\[
v_0-cD_+u=0,\qquad v_J+cD_-u=0.
\]

Let \(S\) select interior rows, \(Q=I-S\), let \(L_i\) contain only the interior Laplacian rows, and define \(C\) by

\[
(Cu)_0=-D_+u,\qquad (Cu)_J=D_-u.
\]

CN applied to the DAE gives the second residual block

\[
r_2
=
S\frac{\Delta v}{\Delta t}
-c^2L_i\bar u
+Q\bar v+cC\bar u.
\]

After projection, exact collapse requires

\[
B_i=\Phi^\top SG,\quad
A_i=\Phi^\top L_iG,\quad
B_b=\Phi^\top QG,\quad
C_b=\Phi^\top CG,
\]

and

\[
\widehat r_2
=
B_i\frac{\Delta h_v}{\Delta t}
-c^2A_i\bar h_u
+B_b\bar h_v+cC_b\bar h_u.
\]

Better still, project only the interior evolution equations and stack the two boundary equations explicitly. Then the additional exact tables are simply \(G_b\) and \(D_bG\). This prevents two physical boundary equations from being diluted or annihilated by the weak projection.

A full \(2N\)-state ODE is also possible by differentiating the constraints:

\[
(v_t)_0=cD_+v,\qquad (v_t)_J=-cD_-v.
\]

That formulation preserves the original boundary constraint only if the initial state satisfies it discretely. Its ROM needs a further table

\[
A_v=\Phi^\top C_vG
\]

and has

\[
\widehat r_v
=
B\frac{\Delta h_v}{\Delta t}
-c^2A_u\bar h_u-cA_v\bar h_v.
\]

So the central conclusion survives—Sommerfeld remains exactly quadrature-free—but the “one \(A\) plus one \(B\)” formula does not.

The clean general statement is:

\[
R=P^\top\left[
E(\mu)\mathcal G\frac{\Delta\eta}{\Delta t}
-K(\mu)\mathcal G\bar\eta
\right],
\qquad
\mathcal G=\operatorname{diag}(G,G).
\]

Precompute \(P^\top E_q\mathcal G\) and \(P^\top K_q\mathcal G\) for every affine operator component. This handles ODEs, DAEs, cross-component coupling, and non-symmetric operators uniformly.

## 3. Gate W1

W1 is wrong in two independent ways.

First, under the repository’s positive-\(\Lambda\) convention,

\[
A=-\Lambda B,
\]

so the reflective gate must be

\[
\frac{\|A+\Lambda B\|}{\|A\|}\le 10^{-12}.
\]

As written, \(\|A-\Lambda B\|/\|A\|\) will be approximately \(2\), not \(10^{-12}\).

Second, the absorbing comparison is not presently defined meaningfully. Reflective \(\Phi\) has \(N-2\) rows; absorbing \(G\) has \(N\). One could embed the sine functions on all \(N\) nodes, but their endpoint values are zero. Then:

- They are not eigenvectors of the absorbing generator.
- They do not span the two boundary degrees of freedom.
- Multiplying a residual by \(\Phi^\top\) deletes every residual row supported only at the endpoints.
- W0 can pass while the ROM enforces no Sommerfeld boundary equation at all.

The absorbing configuration therefore needs its own test operator and explicit boundary residual rows. It has no natural Dirichlet \(\Lambda B\) reference.

A defensible W1 would be:

1. Reflective: verify \(\|A+\Lambda B\|/\|A\|\).
2. Absorbing: verify the full block-matrix QF residual against an independently assembled stencil-plus-boundary residual.
3. If a diagonal-failure diagnostic is desired, define the comparator explicitly—for example the best row-diagonal \(D_\star B\)—rather than borrowing the Dirichlet \(\Lambda\).

## 4. Energy gate F1

For the reflective semi-discretization, the correct invariant is

\[
E_h
=
\frac{\Delta x}{2}v^\top v
-\frac{c^2\Delta x}{2}u^\top Lu
=
\frac{\Delta x}{2}\|v\|^2
+\frac{c^2\Delta x}{2}\|D_eu\|^2,
\]

where \(D_e\) is the forward edge-difference operator including both boundary edges and

\[
D_e^\top D_e=-L.
\]

The common \(\Delta x\) factor may be omitted when only relative drift is reported.

CN preserves this quadratic energy exactly in exact arithmetic, not merely to \(O(\Delta t^2)\). Indeed,

\[
E^{n+1}-E^n
=
\Delta x\left(
\bar v^\top\Delta v-c^2\bar u^\top L\Delta u
\right)=0
\]

after substituting \(\Delta u=\Delta t\,\bar v\), \(\Delta v=\Delta t\,c^2L\bar u\), and using symmetry of \(L\). CN still has \(O(\Delta t^2)\) phase/solution error; that is separate from preservation of the discrete invariant. Exact energy preservation for linear Hamiltonian systems is a standard CN property; see [Simo, Tarnow, and Wong](https://doi.org/10.1016/0045-7825(92)90115-Z).

Thus \(10^{-10}\) is reasonable in f64 if the linear systems are solved sufficiently accurately. But the design must define \(D\). A centered nodewise first derivative generally does not satisfy \(D^\top D=-L\) and would make F1 fail for the wrong reason.

For the absorbing configuration, unconditional stability does not follow merely from choosing CN. The spatial/DAE discretization must first satisfy a discrete energy estimate; energy-compatible boundary constructions such as SBP treatments are designed precisely for this reason ([Wang, Appelö, and Kreiss](https://arxiv.org/abs/2103.02006)).

## 5. Gate F3

At the continuum level the Sommerfeld condition is exactly reflectionless in 1D.

The proposed first-order one-sided spatial discretization is not.

For an interior discrete mode with \(\theta=k_h\Delta x\), centered \(D_{xx}\), CN in time, and the right-boundary condition discretized consistently at the midpoint as

\[
\delta_tu_J+cD_-\,\bar u_J=0,
\]

the interior CN dispersion relation gives

\[
\widetilde\omega
=
\frac{2}{\Delta t}\tan\frac{\Omega\Delta t}{2}
=
\frac{2c}{\Delta x}\sin\frac{\theta}{2}.
\]

Solving the boundary equation for incident plus reflected modes gives

\[
|R(\theta)|=\tan\frac{\theta}{4},
\qquad
\frac{E_{\rm reflected}}{E_{\rm incident}}
=
|R|^2
=
\tan^2\frac{\theta}{4}
\sim \frac{(k_h\Delta x)^2}{16}.
\]

The CN time-step dependence cancels only because the boundary and interior use the same midpoint time symbol. A differently timed boundary row introduces additional \(\Delta t\)-dependent reflection.

For a Gaussian displacement with matched traveling velocity, the small-\(\Delta x\) spectrum-weighted estimate is

\[
\frac{E_{\rm reflected}}{E_{\rm incident}}
\approx
\frac{3}{32}\left(\frac{\Delta x}{\sigma}\right)^2.
\]

For the narrowest proposed pulse, \(\sigma=0.02\):

| \(N\) | expected reflected-energy fraction |
|---:|---:|
| 128 | \(1.45\times10^{-2}\) |
| 256 | \(3.60\times10^{-3}\) |
| 512 | \(8.98\times10^{-4}\) |
| 1024 | \(2.24\times10^{-4}\) |

Therefore a resolution-independent \(10^{-3}\) gate must fail at \(N=128\) and \(256\) for valid numerical reasons. Replace it with a convergence gate: reflection energy should scale as \(O((\Delta x/\sigma)^2)\) and agree with the discrete spectral prediction. If \(10^{-3}\) is required at all resolutions, use a higher-order or discrete-matched absorbing boundary.

“Reflected energy” also needs a precise measurement: characteristic decomposition or a spatial/time window after the incident packet reaches the wall. Remaining total domain energy mixes reflection, dispersive tails, and portions of the incident pulse that have not yet exited.

## 6. Other week-wasters

Several additional controls are essential:

- The Gaussian initial data are not boundary-compatible. At \(x_0=0.25,\sigma=0.08\), the opposite-wall value is about \(7.6\times10^{-3}\), not negligible, and its derivative is larger. A right-going Gaussian therefore violates the left Sommerfeld condition at \(t=0\), creating a startup wave that can be misidentified as reflection. Use a compactly supported/tapered pulse or initialize the boundary values using the exact discrete constraint.

- The vector loss and ROM residual have unspecified block scaling. \(u\), \(v\), \(r_u\), and \(r_v\) have different units and magnitudes; for \(\sigma=0.02\), \(v\) can be tens of times larger than \(u\). Raw MSE and raw concatenated least squares will be dominated by one component. Define an energy-based nondimensionalization for training, reconstruction, POD, and ROM residuals.

- W0 must use an independent full-grid implementation: decode \(u,v\), apply the stencil and boundary equations, then project. Comparing two paths that both use the same cached matrix proves only self-consistency. Retain the Poisson cached-bank-vs-decoder gate as well. At converged solutions, add a backward-error normalization for \(r\), because relative residual-to-residual error becomes meaningless when \(r_{\rm full}\approx0\).

- There is no ROM rollout gate. The repository already records a wave case where residual operators were correct but nonlinear-manifold stepping destroyed energy. Add FOM rollout error, oracle reconstruction error, reflective ROM energy consistency, absorbing energy flux, and a fixed POD-Galerkin/CN control. Otherwise every proposed gate can pass while the vector ROM fails structurally.

- F2 does not specify separate spatial and temporal studies. To claim order two “in both,” hold \(\Delta t\) negligible while refining \(\Delta x\), then hold \(N\) fine while refining \(\Delta t\). Add an absorbing-boundary convergence study; reflective standing waves never exercise the new boundary rows.

- F4 is not implied by CN for an arbitrary non-symmetric boundary discretization. CN is stable for an already stable/dissipative semi-discrete generator; it does not repair a bad boundary operator. Define the norm, prove or measure the absorbing energy inequality, and audit generator eigenvalues/non-normal growth.

- W3 is not a Kolmogorov \(n\)-width measurement as written. POD singular tails are training-set Frobenius quantities, while the decoder result is held-out reconstruction. Fit POD on training trajectories, evaluate both methods on the same held-out trajectories with the same energy-weighted concatenated \((u,v)\) metric, and call it an empirical linear-width/nonlinear-manifold probe.

- Absorbing trajectories that have already exited are nearly zero and can make absolute reconstruction artificially easy or relative errors singular. Run W3 on a fixed pre-exit/constant-energy window, or use a periodic translation control, and report reflective and absorbing datasets separately.

- Fixed \(R\) must be at least comfortably above \(K_{\max}=64\), and the numerical rank of \(G\) must be checked. Otherwise the high-\(K\) plateau is imposed trivially by \(R\) or the spatial-track hidden-width cap. The independent-bank control must match total feature/parameter budget; two \(R\)-wide banks are not a fair comparison with one \(R\)-wide bank.

## Structured verdict

| Item | Verdict | Required fix |
|---|---|---|
| 1. Signs and formulation | **WRONG** | Keep the continuum Sommerfeld signs and reflective CN equations, but specify an absorbing DAE, eliminated-boundary ODE, or differentiated-constraint ODE. Add the resulting \(u\)-\(v\) boundary coupling. Resolve the \(L\)/\(\Lambda\) sign inconsistency. |
| 2. Central claim | **NEEDS-RESTATEMENT** | State the claim for a general block linear residual \(E\dot w-K(\mu)w\). Precompute one matrix per affine block/operator component. Sommerfeld remains zero-sample, but needs more than \(A=\Phi^\top LG\) and \(B=\Phi^\top G\). |
| 3. W1 | **WRONG** | Reflective gate is \(\|A+\Lambda B\|/\|A\|\) under current conventions. Give absorbing its own test space, enforce boundary rows explicitly, and replace the undefined Dirichlet-\(\Lambda\) comparison with a specified diagonal comparator. |
| 4. F1 energy | **NEEDS-RESTATEMENT** | Define the compatible edge difference \(D_e\) with \(D_e^\top D_e=-L\). State that CN preserves this discrete energy exactly in exact arithmetic; \(10^{-10}\) measures solver/roundoff error, not \(O(\Delta t^2)\) conservation. |
| 5. F3 | **WRONG** | Replace the universal \(10^{-3}\) target with the discrete prediction \(|R|^2=\tan^2(k_h\Delta x/4)\) and an \(O(\Delta x^2)\) reflected-energy convergence gate, or adopt a higher-order/matched boundary. |
| 6. Other controls | **NEEDS-RESTATEMENT** | Add compatible initial data, component scaling, independent full-stencil gates, ROM rollout/energy controls, separate convergence studies, and a matched-metric empirical-width protocol. |

## Required design changes, ranked by likely wasted time

1. Specify the absorbing semi-discrete equations and replace the two-matrix residual with the correct block formulation.
2. Replace endpoint-zero sine testing at absorbing boundaries with explicit boundary residual enforcement.
3. Fix the \(A=-\Lambda B\) sign and rewrite W1.
4. Define a discrete reflection prediction and boundary-compatible pulse family before setting F3 thresholds.
5. Add ROM rollout, energy/flux, oracle-floor, and POD-Galerkin controls.
6. Define energy-consistent scaling for \((u,v)\), residual blocks, POD, and decoder training.
7. Rewrite W3 as a held-out, energy-weighted empirical-width experiment with \(R\ge64\), matched bank budgets, and pre-exit snapshots.