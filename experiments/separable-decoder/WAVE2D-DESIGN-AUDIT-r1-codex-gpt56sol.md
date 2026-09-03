No files were changed, and no Python, GPU, installation, or cluster commands were run.

### 1. Absorbing semi-discretisation — CORRECT

The ghost elimination is correct. At \(x=1\),
\[
u_t+c u_x=0,\qquad
u_{J+1}=u_{J-1}-\frac{2\Delta x}{c}v_J,
\]
which gives
\[
(u_{xx})_J=\frac{2(u_{J-1}-u_J)}{\Delta x^2}
-\frac{2}{c\Delta x}v_J.
\]
The mirrored face has the same damping sign. At a non-corner face node, \(L_N\) must retain the ordinary centered tangential second derivative; at a corner, both reflected normal contributions are included, producing \(D_B=4/\Delta x\). With \(M=M_x\otimes M_y\), \(M_x=\Delta x\,\mathrm{diag}(1/2,1,\ldots,1,1/2)\), \(ML_N\) is symmetric negative semidefinite. Consequently,
\[
\dot E=-c\,v^\top MD_Bv
\]
and midpoint/CN gives the stated stepwise identity exactly in exact arithmetic. At a corner, \(M_{ii}D_{B,ii}=\Delta x\), equal to the two half-corner contributions from the adjoining faces. Required fix: spell out the face and corner rows so tangential terms cannot be accidentally omitted. Also correct the later solver statement: the absorbing \(u\)-solve is
\[
I+\frac{c\Delta t}{2}D_B-aL_N,
\]
not \(I-aL_N\); it is SPD only in the \(M\)-inner product, so ordinary Euclidean CG requires an \(M^{1/2}\) similarity scaling or a weighted-CG equivalent. Finally, absorbing “energy norm” is a seminorm in \(u\) because constants lie in \(\ker L_N\).

### 2. Cosine eigenmodes — CORRECT

The DCT-I modes
\[
\phi_{k\ell}(i,j)=\cos(k\pi x_i)\cos(\ell\pi y_j)
\]
are exact right eigenvectors of the tensor-product ghost-reflected Laplacian, including its edge and corner rows, with
\[
L_N\phi_{k\ell}=-\lambda_{k\ell}\phi_{k\ell},\qquad
\lambda_{k\ell}=\frac{2}{\Delta x^2}\left[1-\cos(k\pi\Delta x)+1-\cos(\ell\pi\Delta x)\right].
\]
However, \(L_N\) is not symmetric in the unweighted Euclidean inner product. Thus the design’s absorbing identity \(A=-\Lambda B\) is false for \(A=\Phi^\top L_NG\), \(B=\Phi^\top G\). Required fix: use the left modes \(\Phi^\top M\):
\[
B=\Phi^\top MG,\qquad A=\Phi^\top ML_NG=-\Lambda B,\qquad
C=\Phi^\top MD_BG.
\]

### 3. Damped Newmark residual — CORRECT

Subtracting consecutive CN kinematic equations and using both adjacent dynamic equations gives
\[
u^{n+1}-2u^n+u^{n-1}
-aL_N(u^{n+1}+2u^n+u^{n-1})
+\frac{c\Delta t}{2}D_B(u^{n+1}-u^{n-1})=0,
\]
where \(a=(c\Delta t/2)^2\). Hence the design’s \(B/A/C\) residual has the correct signs, factors, and time centering. For general \(v_0\), the first step is
\[
(u^1-u^0)-aL_N(u^1+u^0)
+\frac{c\Delta t}{2}D_B(u^1-u^0)-\Delta t\,v_0=0;
\]
for \(v_0=0\), the design’s first-step residual is correct. Required fix: use the \(M\)-weighted \(A,B,C\) above for the cosine shortcut, and solve the actual damped matrix \(I+(c\Delta t/2)D_B-aL_N\).

### 4. Arm C — NEEDS-RESTATEMENT

Part (a) is false as written but correct after qualification. The pulled-back Euler–Lagrange equation is
\[
J_g^\top M\left(J_g\ddot z+H_g[\dot z,\dot z]-c^2L g(z)\right)=0,
\]
not unweighted \(J_g^\top(u_{tt}-c^2Lu)=0\) for the absorbing mass. Continuous energy conservation holds for an autonomous conservative problem when \(g\) is at least \(C^2\) and is an immersion along the trajectory; a fixed-step variational integrator generally preserves a nearby modified energy, not the exact \(E_r\). Part (b) is sound: the displayed centered Rayleigh term follows from splitting the discrete virtual work between adjacent intervals and is a second-order-consistent forced variational scheme. It does not imply the FOM CN flux identity exactly. The stated Newton Jacobian is wrong for C-Verlet: because \(J_h(z_n)\) and the potential at \(n\) are fixed during the \(z_{n+1}\) solve,
\[
F'(z_{n+1})
=J_h(z_n)^\top\left(\mathsf M+\frac{c\Delta t}{2}\mathsf D\right)J_h(z_{n+1});
\]
there is no stiffness or \(\partial J_h(z_n)\) term in that Jacobian. C-mid needs its own full derivation. Part (c): \(K\le R\) is necessary but not sufficient; \(J_h\) must have full column rank. At rank loss, \(J_h^\top\mathsf M J_h\) becomes singular, the pulled-back dynamics cease to be regular, and Newton can have non-unique steps or fail. Required fix: add singular-value/condition-number gates along training points, oracle projections, and every rollout; rank loss must make the rollout incomplete rather than be hidden by damping or a pseudoinverse.

### 5. Gate audit — WRONG

The assertion that every gate is independent, normalized, and equipped with a firing negative control is not satisfied.

| Gate | Audit finding and required fix |
|---|---|
| F0a | Valid only if \(L_D\) is stencil-built independently. “Wrong \(k\)” is circular if both the vector and eigenvalue use the same wrong \(k\); hold one fixed and perturb the other. |
| F0b | Same issue as F0a. Ensure the test includes full corner rows and handles the \((0,0)\) zero mode without a zero denominator. |
| F0c | Valid and \(M=I\) will fire, but its relative error is not \(O(\Delta x^{-1})\); that scaling claim is wrong and norm-dependent. |
| F1a | Valid. Make the backward-Euler control’s required measured separation explicit rather than promising \(O(10^{-2})\) for every IC. |
| F1b | Valid as a time-integrator identity, but it self-certifies whatever \(D_B\) the solver also uses. Add an independent manufactured face/corner-row test for the coefficients \(2/\Delta x\) and \(4/\Delta x\). The endpoint-velocity control must be tested during active boundary flux. |
| F2 | No negative control, and no 08-14 absorbing \(N=512\) reference exists. Use separate spatial and temporal self-convergence against newly refined references for each BC. |
| F3 | Wrong order and unsuitable test; see item 7. The reflective control should fire once the correct metric is defined. |
| F4 | No negative control and largely implied by F1. Define the norm and use an anti-damped or sign-mutated operator control. |
| F5 | \(10^{-12}\) is an absolute eigenvalue threshold on a spectrum scaling like \(c/\Delta x\). Gate \(\max\Re\lambda/(c/\Delta x)\), and do not use eigenvalues alone as a non-normal-growth test. |
| V1a/b | Must compare an independently assembled block CN solve with the recurrence. “Wrong damping sign” cannot fire V1a because reflective dynamics have no damping. The \(10^{-12}\) target is also inconsistent with a reference CG tolerance of \(10^{-10}\) unless the gate uses tighter solves. |
| D0 | For a POD/SVD bank, `rank(G)=R` is nearly tautological because returned singular vectors remain orthonormal even beyond meaningful snapshot rank. Gate \(\sigma_R/\sigma_1\), \(G^\top MG\), and coefficient round-trip accuracy. Add \(J_h\)-rank separately. |
| D1 | No negative control, and the 08-14 value uses a different field/error path from the newly declared energy metric. Recompute the FiLM comparator on the identical dataset and metric; exceeding it is not automatically a code bug. |
| G0a | No negative control; the ratio is unstable when training error is tiny and its aggregation is unspecified. Add an absolute normalized gap and predeclare mean/median/worst-case aggregation. |
| G0b | No negative control; \(v_0=0\) makes its denominator singular, and \(P_T\) is undefined at rank loss. Exclude low-kinetic-energy states and use a rank-revealing \(M\)-orthogonal projector. |
| G0c | `hold` is a baseline, not a guaranteed negative mutation; it can legitimately beat a poor ROM at short horizons. Keep it as a comparator and add a deterministic wrong-sign/time-step mutation. |
| W0 | Conceptually good, but no negative control is specified. Gradients must use an independent directional derivative/full-grid path, not autodiff of the same cached residual twice. |
| W1 | Absorbing \(A=-\Lambda B\) is wrong without \(M\). Moreover \(\|C\|/\|A\|\) scales with \(\Delta x\), and dropping \(C\) may not exceed \(10^{-3}\) for a near-zero boundary velocity. Gate the actual term \((c\Delta t/2)C\Delta h\) on a manufactured boundary-active state. |
| W2 | A POD-Galerkin trajectory is not mathematically guaranteed to lie within \(1.05\times\) its instantaneous projection floor unless the POD space is invariant under \(L\). Keep energy preservation as a gate; report floor proximity as a result. No negative control is supplied. |
| W3 | “Completed rollouts only” permits selection bias. Require 16/16 completion. Add timestep/solver convergence and decay-safe absorbing metrics; no negative control is supplied. |
| W4 | A fixed-step variational integrator need not keep exact physical energy within 1% at every chosen RS. Restrict this to reflective runs and require bounded error plus convergence under timestep refinement before calling failure an implementation bug. |
| W5 | The uncorrected FOM flux identity is not obeyed exactly by a projected ROM. It must include residual work, or be replaced by the appropriate forced discrete variational balance. Its denominator and negative control are also unspecified. |

The minimum missing gates are: an independent face/corner ghost-row gate; \(G^\top MG\) and coefficient round-trip gates; a \(J_h\) rank/conditioning gate; a C-Verlet first-step and timestep-convergence gate; mandatory 16/16 completion; and an absorbing error gate normalized by initial energy over a predeclared pre-exit window.

### 6. Verdict table and predictions — WRONG

Only “G0 pass + reflective W3 pass under a converged Arm C” strongly refutes a universal structural-failure claim. “G0 fail + reflective fail + absorbing pass” does not confirm manifold quality: the absorbing dataset can be intrinsically easier, damping can erase accumulated error, and the two BCs use separately trained banks and different initial fields. Conversely, “G0 pass + reflective fail + W4 pass” does not restore the structural diagnosis because G0 is only a proxy, W4 does not certify trajectory accuracy, and timestep, rank, initialization, extrapolation to \(4T\), or curvature errors remain alternatives. The table also omits mixed outcomes such as G0-fail/reflective-pass. Required fix: make the decisive prediction an across-head interaction—after rank and timestep convergence, reflective Arm C should pass if and only if G0 passes—and label mixed results inconclusive. For absorbing runs, report pre-exit error normalized by \(E^0\), retained-energy/flux histories, and post-exit error separately. Because \(L_N\) has a constant nullspace, also report constant/mean-field error, which the stated energy seminorm cannot see.

### 7. F3 reflection convergence — WRONG

For this centered ghost-elimination closure, the normal-incidence discrete reflection coefficient is not the one-sided-boundary result used in the previous audit. With the same CN midpoint symbol in the interior and boundary,
\[
R(\theta)
=\frac{\cos(\theta/2)-1}{\cos(\theta/2)+1}
=-\tan^2(\theta/4).
\]
Thus amplitude is \(O(\Delta x^2)\), while reflected energy is
\[
|R|^2=\tan^4(\theta/4)=O(\Delta x^4).
\]
For a Gaussian traveling pulse, the small-\(\Delta x\) energy-weighted prediction is approximately
\[
\frac{E_{\rm refl}}{E_{\rm inc}}
\approx \frac{15}{1024}\left(\frac{\Delta x}{w}\right)^4.
\]
A slope near 2 in reflected energy would therefore indicate contamination or a different boundary discretization, not success. The broad \(w=0.20\) cases reach \(10^{-7}\)-to-\(10^{-9}\) reflection on this ladder, where leakage and measurement contamination can dominate; the radial blob also contains oblique incidence whose continuum reflection does not vanish with refinement. Required fix: gate a quasi-1D, boundary-compatible traveling pulse with \(v_0=-cu_x\), isolate it from transverse absorbing faces, and require slope \(4\pm0.5\) or agreement with the exact \(\tan^4(\theta/4)\) prediction. If amplitude or square-root energy is gated instead, slope 2 is appropriate.

### 8. Cost ladder fairness — NEEDS-RESTATEMENT

Matched accuracy is the right principle, but “largest \(\Delta t\) and loosest CG tolerance” is not a well-defined optimum because the two parameters form a partial order and the cheapest pair need not be the coordinatewise loosest. Select the fastest predeclared FOM configuration whose identical trajectory metric and completion criteria meet the ROM error, with accuracy and cost taken from the same invocation. Excluding decoding is defensible only for a separately labeled latent-state kernel benchmark. Since W3 judges full-field trajectories, the headline end-to-end comparison must include initialization and decoding at the same 51 requested output times. The current ladder can flatter the ROM by omitting full-grid cold-start fitting and rendering, timing only successful ROM rollouts, using generic CG instead of a competitive preconditioned/direct separable solver, amortizing ROM tables but not equivalent FOM setup, tuning on the evaluation cases, and changing \(R\) from 64 to 96 despite claiming fixed \(K,R,M\). Required fix: publish both latent-only and end-to-end timings, require all cases to complete, use the fastest credible FOM, apply identical setup-amortization rules, and keep \(R\) fixed or run separate fixed-\(R\) ladders.

| item | verdict | one-line fix |
|---:|---|---|
| 1 | CORRECT | Keep the ghost ODE and energy law, but specify all boundary rows and use the mass-scaled damped CN solve. |
| 2 | CORRECT | Keep the cosine spectrum; use \(\Phi^\top M\) as the absorbing left test operator. |
| 3 | CORRECT | Keep the three-level and first-step formulas; replace unweighted tables and the stated absorbing solve. |
| 4 | NEEDS-RESTATEMENT | State the \(M\)-weighted continuous EL result, correct C-Verlet’s Jacobian, and gate \(J_h\) rank. |
| 5 | WRONG | Replace circular/missing controls, mesh-dependent thresholds, invalid W2/W5 rules, and completion selection. |
| 6 | WRONG | Base the causal verdict on an across-head reflective interaction and treat mixed/decayed cases as inconclusive. |
| 7 | WRONG | Gate reflected energy against \(O(\Delta x^4)\) or amplitude against \(O(\Delta x^2)\) using an isolated traveling pulse. |
| 8 | NEEDS-RESTATEMENT | Choose the fastest matched-accuracy FOM and headline an end-to-end, fixed-dimension comparison. |

Most compute-saving changes, ranked:

1. Fix the absorbing implementation contract before coding: \(M\)-weighted cosine projection and the mass-scaled \(I+(c\Delta t/2)D_B-aL_N\) solve.
2. Replace F3’s slope-2 energy gate with the correct discrete reflection prediction and a genuinely one-dimensional traveling-pulse test.
3. Add \(J_h\)-rank, timestep-convergence, 16/16-completion, and decay-safe W3 requirements before interpreting or timing any ROM.