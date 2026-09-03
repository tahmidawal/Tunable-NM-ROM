Overall: the FOM mathematics is correct, and the reported numbers are internally consistent, but the phase-1 “ALL PASS” is not fully earned. The decisive defect is V1: its reference is not an independently assembled block-CN solve.

### 1. `make_cn_fom` — CORRECT

[make_cn_fom](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_common.py:248) implements exactly

\[
A=I+sD_B-aL,\qquad
Au_1=(I+sD_B)u+\Delta t\,v+aLu
\]

followed by the stated velocity update.

For absorbing boundaries, it solves

\[
M^{1/2} A M^{-1/2}y=M^{1/2}b,\qquad u=M^{-1/2}y,
\]

with the correctly scaled initial guess. Since \(MA\) is symmetric positive definite, the transformed operator is Euclidean SPD.

The reflective branch has the specified RHS, `x0=u+dt*v`, and caller-supplied CG tolerance. V0 agrees with the frozen rollout at \(2.27\times10^{-15}\) and \(3.36\times10^{-15}\), supporting the op-for-op claim.

### 2. `make_newmark_fom` — CORRECT

[make_newmark_fom](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_common.py:363) has the correct first step and recurrence:

\[
Au_1=(I+sD_B)u_0+\Delta t\,v_0+aLu_0,
\]

\[
Au_{n+1}=2u_n-u_{n-1}+aL(2u_n+u_{n-1})+sD_Bu_{n-1}.
\]

The scan counts are correct: the precomputed first step plus `rs-1` steps produces the first stored interval, then `rs` thereafter.

Its stored velocity is propagated through the CN dynamic equation. It is not the kinematic recursion \(2(u_n-u_{n-1})/\Delta t-v_{n-1}\).

### 3. Stencils and independent paths — CORRECT

`jnp.pad(..., mode="reflect")` gives \(u_{-1}=u_1\) and \(u_N=u_{N-2}\). Consequently:

- A face has doubled normal-neighbour weight, retains both tangential neighbours, and has centre coefficient \(-4/\Delta x^2\).
- A corner has two doubled inward neighbours and centre coefficient \(-4/\Delta x^2\).
- The separately applied damping is \(2/\Delta x\) per incident face, hence \(4/\Delta x\) at corners.

[assemble_L_independent](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_common.py:157) is a separate sparse row assembly and does not call the padded stencil. [ghost_row_closed_form](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_fom_gates.py:61) is a genuine third implementation: it inserts each ghost value, including the velocity term, directly. It is generic directional code rather than separate face/corner branches, but it is computationally independent.

### 4. Gate audit — WRONG as a complete certification suite

| Gate | Finding |
|---|---|
| F0-stencil | Correct; independent assembly is mutated and compared to the unchanged solver stencil. |
| F0a/F0b | Main backward-error quantity is mesh-normalized, but the 1%-lambda control now scales like \(\Delta x^2\) under that denominator. Its fixed \(10^{-9}\) threshold is still mesh-dependent. |
| F0b-zero-mode | No negative control, although it is included in `all_passed`. |
| F0c | Correct. `min(err_w,err_w2)` really requires both mutations to fire. |
| F0d | Absorbing symmetry control is genuine. Reflective symmetry and the SPD subgate have no relevant controls. The SPD test is run at outer \(N=64,128\), not the specified \(N=32\). |
| F1a | Correct BE mutation. `F1a-form`, however, is separately pass-counted without a control. |
| F1b | Correct. The active mask is dimensionally sound: `flux` already includes \(\Delta t\), so comparison with \(10^{-3}E_0\Delta t\) selects steps by dissipation rate. The endpoint-velocity mutation genuinely changes the flux quadrature. |
| F2 | Spatial convergence has no spatial negative control. The BE control mutates only the temporal scheme. The final smooth-bump BE orders are 0.92–0.96, so removing the original \(1\pm0.3\) condition was unnecessary for the final setup. |
| F3 | `_XFaceGrid` correctly removes only y-face damping. The reflective mutation is genuine, but the `slope OR coefficient agreement` rule can pass an absorber with the correct \(h^4\) rate and an arbitrarily wrong coefficient. |
| F4 | `_AntiGrid` really flips damping throughout `energy_trace`. However, nonfinite anti-damped traces are converted to `inf` and then clipped to \(10^{300}\), which makes a NaN/overflowing control pass. Reflective F4 has no control and is already implied by F1a. |
| F5 | Correct normalized spectrum check and genuine sign mutation. |
| V0 | Genuine state mutation, but not the declared one-coefficient mutation: `_PerturbedGrid.dx` changes every Laplacian coefficient and also the mass. Mass cannot cause the state-difference control to fire because only `S2` is compared. |
| V1 | Defective independence: it does not build the \(2n\times2n\) CN block system. It repeats the same eliminated \(u\)-equation and velocity formula as the implementation, merely using assembled \(L\) and LU. Thus a shared algebraic mistake passes. Its absolute \(10^{-8}\) threshold is explicitly placed on a mesh/conditioning-dependent accumulated quantity. |

V1 is therefore the effective self-comparison: not literal array-to-itself subtraction, but the same derived algebra on both sides.

### 5. Retractions and amendments — NEEDS-RESTATEMENT

[WAVE2D-NOTES.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/WAVE2D-NOTES.md:9) contains only three amendments—1, 2, and 4. Amendment 3 is absent and cannot be verified.

- Retraction 1: plausible mechanism, insufficient proof. Accumulation of CG error over 4,000 solves is real, but one loose/tight ratio cannot exclude a fixed algebraic discrepancy beneath the loose-solve error.

- Amendment 1: valid reasons. The original grids are not nested, and the inherited initial conditions are boundary-incompatible. The replacement bump is only approximately compatible, however, and nearly suppresses the absorbing boundary. Use an exactly compatible manufactured solution or mode.

- Amendment 2: weak/post-hoc. BE can be pre-asymptotic, but the replacement “BE is 10× worse” test does not verify first-order convergence. On the final smooth bump the measured BE orders already satisfy the original condition; require both order and separation.

- Retraction 2: the roundoff-scaling diagnosis is valid. The revised denominator removes the \(O(\Delta x^{-2})\) growth. But \(8/\Delta x^2\) is not generally the Euclidean 2-norm upper bound claimed for nonsymmetric \(L_N\), and the mutation control now shrinks as \(\Delta x^2\). Use an actual/safe norm bound and separately normalized control.

- Retraction 3: not adequately justified. The claimed \(10^3\!-\!10^4\) ratios conflict with the JSONs, whose range is approximately 40–4,138; the N=128 absorbing case is only 40.2.

- Amendment 4: valid data-property explanation. High-frequency content from incompatible initial data can make a temporal study pre-asymptotic. Again, an exactly compatible smooth mode would make the argument cleaner.

No: `value(CG 1e-11)/value(CG 1e-13) >= 10` is not sufficient evidence of solver limitation. If

\[
d(\tau)=d_{\rm algebra}+d_{\rm CG}(\tau),
\]

the ratio may exceed ten while \(d_{\rm algebra}\neq0\). Required fix: compare against a genuinely independent block-CN solve and show convergence over several tolerances toward zero—or solve the Newmark recurrence directly by LU and obtain roundoff agreement.

### 6. F3 reflection result — CORRECT

The implementation is a valid measurement of the discrete pulse reflection:

- `v0` has the correct sign for a right-going pulse.
- `_XFaceGrid` leaves Neumann closure on y-faces but removes their damping.
- Axis ordering is correct: the damping outer product selects the x coordinate.
- A y-uniform state is an exact invariant subspace, so the y-faces cannot contribute.
- The stored sample is at \(t=0.86\), just after the declared six-width exit time \(0.85\). The reflected pulse remains inside the domain.
- A constant displacement remainder has zero wave energy, so omitting it from the energy seminorm does not manufacture the \(h^4\) reflected-wave energy.

The continuous-derivative initialization is not an exact one-way mode of the discrete operator and creates a small opposite-going component, but that component reaches and exits the left absorber before the sampled time. A nearby-snapshot plateau and y-variation diagnostic would be worthwhile corroboration, not a prerequisite for accepting the result.

### 7. Other phase-PASS issues — WRONG

Additional defects:

- F4 explicitly converts a nonfinite negative control into a passing huge value, contradicting the no-silent-NaN rule.
- CG return information is ignored everywhere, and no achieved residual is recorded. The strong energy and V1 discrepancies give indirect evidence, but not explicit convergence certification.
- V1 logs `PASS` before applying its loose/tight-ratio condition; the JSON is corrected afterward, but console output can be false.
- The usage example says `F2_REF=256`, which is incompatible with the default nested grids; the actual successful default is 257.
- Both module headers still identify design r2 although the audited design is r3.
- There is no problematic JIT closure over trajectory data: state and parameter data remain explicit arguments.
- I found no current wrong sign in \(L\), \(D_B\), CN, or Newmark. The problem is gate independence, not the implemented equations.

| item | verdict | one-line fix |
|---|---|---|
| 1 | CORRECT | No solver change required. |
| 2 | CORRECT | No recurrence or velocity change required. |
| 3 | CORRECT | Optionally spell the third path as explicit face/corner cases for stronger independence. |
| 4 | WRONG | Replace V1 with a true block-CN reference, add missing controls, normalize controls, and reject nonfinite mutations. |
| 5 | NEEDS-RESTATEMENT | Restore rigorous convergence evidence, document missing Amendment 3, and replace two-tolerance inference with a tolerance ladder/direct solve. |
| 6 | CORRECT | Add a time-plateau, y-invariance, and mean-field report as corroborating diagnostics. |
| 7 | WRONG | Check achieved CG residuals, never convert NaN/overflow to a passing control, and repair stale metadata. |

**Single most important defect:** V1’s “independent block CN” reference is not a block solve at all; because it repeats the same eliminated algebra, the principal equivalence gate cannot certify that algebra.