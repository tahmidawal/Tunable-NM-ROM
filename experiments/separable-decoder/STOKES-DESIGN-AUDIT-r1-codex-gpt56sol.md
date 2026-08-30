The design should not be implemented as written. The central quadrature-free algebra is valid for a properly paired MAC discretization, but several boundary, gate, forcing, and POD details are presently strong enough to turn implementation bugs into “findings.”

Repository files were not changed. The only scratch work is under the permitted directory.

## Numerical check used below

I assembled a standard \(N\times N\)-cell MAC grid with:

- \(p\): \(N^2\) cell centers.
- Active \(u_x\): \(N(N-1)\) interior vertical faces.
- Active \(u_y\): \(N(N-1)\) interior horizontal faces.
- Boundary-normal velocities eliminated.
- Odd tangential ghosts for no-slip.
- Vertex streamfunction with zero boundary values.

At \(N=32\):

| Check | Result |
|---|---:|
| \(\|D+\mathrm{Grad}^\top\|_\infty\) | exactly \(0\) |
| \(\|DC\|_\infty\) | exactly \(0\) |
| \(\operatorname{rank}D\) | \(1023=N^2-1\) |
| \(\dim\ker D\) | \(961=(N-1)^2\) |
| \(\operatorname{rank}C\) | \(961\) |
| normalized \(\|\Phi^\top\mathrm{Grad}\,p\|\) | \(5.91\times10^{-18}\) |

Thus the full vertex-curl space exactly spans the discrete solenoidal space on this simply connected grid.

However, evaluating \(D\Phi\) through the actual floating-point field path gave maximum absolute errors:

- \(N=64:\ 2.27\times10^{-13}\)
- \(N=128:\ 4.55\times10^{-13}\)
- \(N=256:\ 9.09\times10^{-13}\)

So S2’s absolute \(10^{-14}\) threshold will reject a correct implementation.

Scratch checks: [mac_check.py](/home/tahmid/.claude/jobs/9755d4a3/tmp/mac_check.py), [mac_s5_scaling.py](/home/tahmid/.claude/jobs/9755d4a3/tmp/mac_s5_scaling.py).

## 1. The pressure argument — NEEDS-RESTATEMENT

The pressure is eliminated exactly on the standard MAC unknown space, but “MAC” alone does not establish the identity.

With velocity and pressure mass matrices \(M_u,M_p\), the required relation is

\[
M_u\,\mathrm{Grad}_h=-D_h^\top M_p.
\]

Then

\[
\Phi^\top M_u\mathrm{Grad}_h p
=-(D_h\Phi)^\top M_p p=0.
\]

On the uniform square layout above, both mass matrices differ only by a common \(h^2\), so the unweighted statement \(\mathrm{Grad}_h=-D_h^\top\) is valid. Compatible adjoints and discrete integration by parts are properties of the chosen MAC operator pair, not of staggering in isolation; this is also how MAC analyses define the operators ([MAC discrete integration-by-parts formulation](https://link.springer.com/article/10.1007/s00211-023-01346-y)).

Exact cancellation requires all of the following:

1. `D` and `Grad` use the same face ordering, signs, spacings, and active unknown set.
2. Projection uses the compatible velocity inner product.
3. \(D_h\Phi=0\) for the operator actually used by the solver.
4. Boundary-normal test velocities are zero, or those faces have been eliminated.
5. No independent boundary closure, penalty, or stabilization contributes an unmatched pressure term.

If boundary-normal faces are retained, a summation-by-parts boundary term survives algebraically. It vanishes when \(\Phi\cdot n=0\). Therefore full adjointness need not hold on rows where \(\Phi\) is identically zero, but it must hold on every active row on which \(\Phi\) can be nonzero. The blanket claim “including all boundary rows” is stronger than necessary; restricted weighted adjointness is the precise requirement.

Also, S1 is not what makes pressure disappear. Bank divergence controls whether decoded velocities satisfy continuity. Pressure cancellation depends on the test space and the `D`/`Grad` pair, independently of \(G\).

Required fix: specify the exact MAC index sets, mass matrices, boundary elimination, gradient and divergence stencils, and gate

\[
\frac{\|M_u\mathrm{Grad}+D^\top M_p\|}
     {\|M_u\mathrm{Grad}\|+\|D^\top M_p\|}
\]

plus its restriction to the actual test space.

## 2. The divergence-free test space — NEEDS-RESTATEMENT

For a vertex streamfunction and compatible incidence differences,

\[
u_{i,j+1/2}=\frac{\psi_{i,j+1}-\psi_{i,j}}h,\qquad
v_{i+1/2,j}=-\frac{\psi_{i+1,j}-\psi_{i,j}}h,
\]

the cell divergence telescopes exactly, including boundary cells. This requires \(\psi\) to be constant on each connected boundary component; the sine functions give zero.

It is not guaranteed if “curl” is implemented with unrelated centered differences, interpolation, or a cell-centered streamfunction.

Richness has two different answers:

- The complete set \(1\le k,\ell\le N-1\) spans all interior vertex values. Its curls span all of \(\ker D\) on the square.
- A truncated \(M\)-mode set spans only an \(M\)-dimensional low-frequency portion. It necessarily annihilates a large residual subspace. That is legitimate weak testing, but it must be shown adequate for this force family and decoder.

Unlike the waves failure, curl-sine modes do not systematically delete the no-slip boundary-adjacent momentum rows. In my \(N=32\), \(k,\ell=1,\dots,4\) check, none of the 1,984 active velocity rows—and none of the boundary-adjacent tangential rows—had zero test leverage. But this depends on the selected mode set; a single sine curl can have zero rows on symmetry lines.

S3’s anti-vacuity control is inadequate. For arbitrary \(q\),

\[
\|\Phi^\top(\alpha q)\|
\]

can be made “\(O(1)\)” merely by choosing \(\alpha\). It proves only that \(\Phi\neq0\), not that pressure cancellation or boundary-row coverage is correct.

Replace it with:

- A matched non-solenoidal test basis \(\widetilde\Phi\), with equal column norms/frequencies, verifying \(\widetilde\Phi^\top\mathrm{Grad}\,p\) is non-negligible for the same pressure.
- Per-row leverage \(\|\Phi_{j,:}\|_2\), reported separately for boundary-adjacent rows.
- Deterministic injected residuals on each boundary stencil family.
- An \(M\)-ladder and the numerical rank/conditioning of \(A\,J_h(z)\), with \(M>K\) comfortably.
- A normalized structural check for \(DC\), followed by a scale-aware field-path check such as
  \[
  \frac{\|D\Phi\|}{\|D\|\,\|\Phi\|}.
  \]

The raw \(10^{-14}\) S2 threshold must be removed.

## 3. The divergence-free POD bank — NEEDS-RESTATEMENT

The linear-span argument is airtight only in exact arithmetic.

Let \(X\) contain snapshots. If \(DX=0\), then any \(G=XT\) satisfies \(DG=0\). Consequently:

- Mean subtraction preserves the constraint because the mean is a linear combination of snapshots.
- Truncation preserves it.
- Normalization and QR reorthogonalization preserve it in exact arithmetic.
- Direct SVD and Gram POD preserve it mathematically if retained modes are constructed inside the snapshot span.

But there are important qualifications.

For Gram POD,

\[
g_i=\frac{Xv_i}{\sigma_i},
\qquad
Dg_i=\frac{(DX)v_i}{\sigma_i}.
\]

Thus a snapshot divergence residual of \(10^{-8}\) does not imply mode residuals of \(10^{-8}\). Tail modes can amplify it by \(1/\sigma_i\). The design’s statement at [STOKES-DESIGN.md:64](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/STOKES-DESIGN.md:64) is false numerically.

Additional distinctions:

- A snapshot-Gram implementation \(XV\Sigma^{-1}\) stays in the computed snapshot span but squares conditioning when finding eigenvalues.
- Using eigenvectors of \(XX^\top\) directly only preserves the range approximately in floating point.
- QR/reorthogonalization adds roundoff but no new exact directions; random completion, coefficient thresholding, or sparsification can break the constraint.
- If component scaling is applied for POD, it must be undone before applying \(D\).

Mean subtraction creates another missing term. If centered POD is used, the decoder must be

\[
u=\bar u+Gh(z),
\]

and the residual becomes

\[
r=-\nu\left(\Phi^\top L\bar u+A h(z)\right)-\Phi^\top f.
\]

The current formula omits \(\bar u\).

Required fix: either use uncentered POD, or include the affine mean explicitly. Project snapshots or modes onto the discrete nullspace before POD, and gate normalized \(DG\) after every normalization/reorthogonalization step. Prefer constructing the bank from an explicit solenoidal basis if “exact by construction” is intended.

## 4. Gate S5 — WRONG

Under homogeneous no-slip velocity conditions, curl-sine modes are not eigenvectors of the standard MAC vector Laplacian. The boundary decides this analytically; it is not an open empirical question.

For

\[
\psi_{k\ell}=\sin(k\pi x)\sin(\ell\pi y),
\]

the curl components are sine in the normal direction and cosine in the tangential direction. Cosines have an even/free-slip-like extension. No-slip MAC tangential velocities use odd ghosts.

Let

\[
\lambda_{k\ell}=
\frac4{h^2}\left[
\sin^2\frac{k\pi}{2N}+
\sin^2\frac{\ell\pi}{2N}
\right].
\]

Away from tangential walls, \(L\Phi=-\lambda\Phi\). At boundary-adjacent tangential rows,

\[
(L\phi+\lambda\phi)_{u,i,0}
=(L\phi+\lambda\phi)_{u,i,N-1}
=-\frac{2}{h^2}\phi_u,
\]

and analogously for \(v\) at the left and right walls. The defect is zero elsewhere. This is the odd-ghost versus even-ghost difference.

For 64 curl modes \(k,\ell=1,\dots,8\), the direct eigen-residual

\[
\frac{\|L\phi+\lambda\phi\|}{\|L\phi\|}
\]

was:

| \(N\) | min | median | max |
|---:|---:|---:|---:|
| 64 | 0.769 | 0.921 | 0.998 |
| 128 | 0.943 | 0.982 | 0.999 |
| 256 | 0.988 | 0.996 | 0.99985 |

Using six smooth clamped, divergence-free surrogate bank vectors, the actual S5 quantity was:

| \(N\) | \(\|A+\Lambda B\|/\|A\|\) |
|---:|---:|
| 64 | 0.341 |
| 128 | 0.343 |
| 256 | 0.343 |

The actual POD-bank value is bank-dependent and cannot be predicted to three digits, but it should be order \(10^{-1}\)–\(1\), not roundoff.

A near-zero S5 result under the stated no-slip BC should be treated as evidence of:

- even/free-slip tangential ghosts,
- omitted boundary stencil terms,
- an incorrect \(L\),
- or accidental orthogonality of \(G\) to the boundary defect.

It is not an informative alternative finding. Dense \(A\) is required. Make the direct operator eigen-residual the primary gate; keep the bank-dependent ratio only as a secondary diagnostic.

## 5. Are the velocity conditions mutually satisfiable? — NEEDS-RESTATEMENT

Divergence-free and homogeneous no-slip are mutually consistent. For example,

\[
\psi=\sin^2(\pi x)\sin^2(\pi y)
\]

has both \(\psi=0\) and \(\partial_n\psi=0\) on the boundary, so its curl is divergence-free and zero on all walls.

The simple sine streamfunctions used for \(\Phi\) do not have this property. They satisfy:

- \(\psi=0\) on the boundary, hence zero normal velocity.
- Generally \(\partial_n\psi\ne0\), hence nonzero tangential velocity.

For no-slip streamfunction formulations, the two conditions are precisely “\(\psi\) constant” plus “\(\partial_n\psi=0\)” ([streamfunction no-slip boundary formulation](https://www.sciencedirect.com/science/article/abs/pii/S0045793003000379)).

This does not overconstrain the proposed bank because the bank is POD-based, not spanned by the sine tests. But the design must distinguish two formulations:

- For the strong discrete residual \(\Phi^\top L u\), the sine curls are permissible Petrov tests: pressure elimination needs only zero normal trace, and the no-slip boundary remains hard-enforced on the trial field/operator.
- For the standard integrated-by-parts weak Stokes form, they are not admissible \(H_0^1\) tests; viscous boundary terms survive.

There is also a MAC bookkeeping correction. In the standard eliminated layout, tangential wall values are not stored velocity DOFs. No-slip is imposed through odd ghosts in \(L\); it is not literally “inherited from POD snapshots.” Boundary-normal values are eliminated or stored as zero and are inherited linearly. Rewrite that claim accordingly.

If genuinely no-slip test vectors are desired, use curls of clamped streamfunctions or discrete Stokes eigenmodes—not individual sine curls.

## 6. Other week-wasters — WRONG as a complete design

### Non-affine right-hand side breaks the end-to-end cost statement

For moving Gaussian centers,

\[
b(\mu)=\Phi^\top f(\mu)
\]

cannot be globally precomputed once. It is computed once per query, generally at \(O(Mn_u)\) cost. Only repeated LM residual evaluations are \(O(MR)\).

This contradicts the stated Poisson timing convention, which explicitly times source projection inside the source-to-solution pipe at [sep_poisson_qf.py:44](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/sep_poisson_qf.py:44).

Either:

- scope the claim to “latent residual cost after per-query source projection,” or
- give \(f\) a finite affine decomposition \(f=\sum_q\theta_q(\mu)f_q\), or
- derive an exact analytic discrete projection.

### The force family is insufficiently specified

Scalar \(\nu\) changes steady Stokes velocity only by an overall \(1/\nu\) factor; it adds no shape richness. The Gaussian centers must therefore carry the manifold complexity.

More importantly, only the solenoidal part of \(f\) drives velocity. A gradient-like Gaussian family can generate large pressure and nearly zero velocity; a purely solenoidal family can make the pressure gate vacuous.

Use a controlled mixture

\[
f=Cq+\mathrm{Grad}\,\chi
\]

with independently varied amplitudes. Report:

- solenoidal-force energy fraction,
- gradient-force energy fraction,
- \(\|\mathrm{Grad}\,p\|/\|f\|\),
- velocity snapshot singular spectrum,
- and numerical ranks versus \(R\).

### A manufactured-solution FOM gate is missing

All ROM gates can pass against a consistently wrong FOM. Add a smooth no-slip divergence-free manufactured velocity, nonconstant pressure, and mesh-convergence study. It must exercise:

- gradient/divergence signs,
- pressure gauge,
- tangential ghost coefficients,
- force evaluation on both face grids,
- second-order velocity and pressure convergence.

This is the cleanest defense against accidentally solving free-slip Stokes.

### `n_u` and \(N\) are ambiguous

For \(N\) cells, the active velocity dimension is

\[
n_u=2N(N-1),
\]

not the total geometric-face count \(2N(N+1)\). Pressure has \(N^2\) entries with one gauge null mode.

Existing collocated code uses scalar \((N-2)^2\) interiors and \(h=1/(N-1)\). MAC commonly uses \(N\) cells and \(h=1/N\). These conventions cannot share reshapes, masks, or spacing silently.

### The later NS tensor is not the existing Burgers tensor

The current machinery is explicitly scalar, collocated, and fixed-positive-upwind at [b2d_tensor_common.py:4](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/b2d_tensor_common.py:4). MAC Navier–Stokes requires:

- two staggered component grids,
- cross-component interpolation,
- a defined skew-symmetric bilinear convection operator,
- and
  \[
  T_{mjk}=\langle\phi_m,\mathcal B(g_j,g_k)\rangle.
  \]

Sign-upwind convection is piecewise/non-polynomial for sign-changing velocities and does not admit one global tensor. “Stokes plus the tensor on the same code” is therefore too strong. A new staggered bilinear tensor gate against full convection and its Jacobian is required.

### The \(R\)-frontier is confounded

Sweeping \(R\) changes simultaneously:

- POD truncation error,
- head output dimension and parameter count,
- possible constraint amplification in small-singular-value modes,
- and how much of \(G\) the fixed low-mode \(\Phi\) can see.

Use nested POD banks from one factorization, hold \(K\) fixed, report parameter count, run an \(M\)-ladder, and gate the smallest singular value of \(A J_h(z)\).

S7 also needs three controls, not one vague “matched-rank POD”:

1. POD-\(K\): same online dimension.
2. POD-\(R\): same trial span ceiling.
3. Direct linear reduced solve in the \(G\) span: removes the nonlinear head entirely.

Because Stokes is linear and \(G\) is itself POD, that third control may be both faster and more accurate than the nonlinear head.

### Modal normalization is missing

Curl-sine mode norms grow with \(\sqrt{\lambda_{k\ell}}\). Without kinetic-inner-product normalization, high-frequency equations receive larger implicit weights. Orthonormalize \(\Phi\), or divide each curl by \(\sqrt{\lambda_{k\ell}}\), and use the same mass-weighted metric everywhere.

### Further gate corrections

- S1 should be normalized; an absolute divergence threshold scales with \(h^{-1}\), mode normalization, and the POD tail.
- S3’s denominator can be nearly zero. Use a backward-error normalization and separately require nontrivial pressure.
- S4 needs a cancellation-aware residual check as well as the gradient check; relative residual-to-residual error is meaningless when both residuals are near zero.
- S5 presently cannot fail—it accepts either answer. Under no-slip, it must have the noncommutation expectation above.
- The NNLS arm must define separate sampling on the two face lattices and must apply sampling only after analytical pressure elimination; sampled quadrature does not itself preserve exact gradient annihilation.
- If the eventual NS experiment is unsteady, steady Stokes does not validate the time/mass block or nonlinear-manifold time stepping. State explicitly whether the target is steady NS.

## Required design changes, ranked by likely wasted time

1. **Fully specify and manufacture-test the MAC discretization**, including active DOFs, mass matrices, no-slip ghosts, pressure gauge, and weighted `D`/`Grad` adjointness.
2. **Rewrite S5 as an expected boundary noncommutation result.** A roundoff result should fail the no-slip implementation.
3. **Resolve online force projection and force-family Hodge richness.** Moving Gaussians are non-affine, and uncontrolled gradient/solenoidal content can make the entire problem vacuous.
4. **Fix POD constraint handling and the missing mean term.** Project onto \(\ker D\) and do not assume FOM residuals transfer unchanged to normalized tail modes.
5. **Replace S2/S3 with scale-aware structural, pressure-sensitivity, and row-leverage gates.**
6. **Specify \(M\), normalize \(\Phi\), and gate \(\operatorname{rank}(AJ_h)\)** before any training or \(R\)-sweep.
7. **Redesign the \(R\)-frontier with POD-\(K\), POD-\(R\), and direct linear-span controls.**
8. **Treat MAC skew-convection as new tensor machinery.** The current scalar positive-upwind implementation is not a usable NS tensor backend.
9. **Clarify cost scope and MAC bookkeeping**, especially \(n_u=2N(N-1)\), \(h=1/N\), per-query source projection, and whether the eventual NS problem is steady or unsteady.