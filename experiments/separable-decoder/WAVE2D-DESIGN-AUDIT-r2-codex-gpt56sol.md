1. **WRONG.** Item 6’s verdict logic is fixed. Item 7 is not fully fixed: a \(y\)-uniform pulse has \(v\neq0\) on absorbing \(y\)-faces, so those faces dissipate energy rather than carry zero flux. Use periodic/reflective transverse boundaries or disable transverse damping for F3. Item 8’s fairness rules are fixed, but the absorbing operator is not DCT-diagonal because of \(D_B\); use a verified absorbing solver. Item 5 still has defective gates listed under item 6.

2. **CORRECT.** The displayed first step is the discrete-Legendre/half-kick initialization for forced variational Verlet. The \(1/2\) stiffness impulse and \((c\Delta t/2)D_r(h^1-h^0)\) damping impulse are correct.

3. **WRONG.** A linear head gives reduced damped Störmer–Verlet:
\[
M_K\Delta^2z_n+c^2\Delta t^2K_Kz_n+
\frac{c\Delta t}{2}D_K(z_{n+1}-z_{n-1})=0,
\]
not Galerkin CN. W7 must compare against an independently implemented POD-\(K\) Verlet/central-difference recurrence and matching half-kick first step. It is stiffness-CFL-limited; without damping,
\[
c\Delta t\sqrt{\lambda_{\max}(M_K^{-1}K_K)}\le2.
\]

4. **NEEDS-RESTATEMENT.** The balance is correct only if \(r_{\rm full}\) means the time-integrated, \(M\)-weighted CN momentum residual and the CN kinematic equation is exact. Specifically,
\[
R_m=M(v^{n+1}-v^n)+c\Delta t\,MD_B\bar v+c^2\Delta t\,K\bar u,
\]
and the work is \(\bar v^\top R_m\). For an \(u\)-only rollout,
\[
\bar v=\frac{u^{n+1}-u^n}{\Delta t}
      =\frac{v^{n+1}+v^n}{2},
\]
not a three-time-level central difference. If “dynamic” endpoint velocities do not satisfy this equality, W5 also needs the kinematic-residual work term. The displayed three-level Newmark residual cannot simply be substituted for \(R_m\).

5. **CORRECT.** Since \(ML_N\) is symmetric negative semidefinite and \(M,D_B\) are positive diagonal,
\[
M^{1/2}\!\left(I+\frac{c\Delta t}{2}D_B-aL_N\right)M^{-1/2}
\]
is symmetric positive definite.

6. **WRONG.**

- Self-certifying: **W5** is an algebraic identity if its residual is formed from the same fields/operators; **D0** round-trip is likewise tautological unless \(P_Ru\) comes from an independent path.
- Absolute mesh-scaling threshold: **F0b** uses an unnormalized absolute zero-mode residual; normalize by \(\Delta x^2\|\phi\|\). D0’s round-trip error should also be explicitly relative in the \(M\)-norm.
- Broken/missing controls: **F0c** mutates \(D_B\) while its stated quantity is \(L_Nu\); **D1** lowering POD rank makes its pass condition easier; **W6** has no control; **W7** drops \(\dot z_0\) although the declared family has \(\dot z_0=0\). **W4** and **W5** use controls that are not guaranteed to separate.

| item | verdict | one-line fix |
|---:|---|---|
| 1 | WRONG | Isolate F3 transversely, repair the remaining gates, and remove the false absorbing-DCT solver claim. |
| 2 | CORRECT | Keep the stated half-stiffness and centered half-interval damping terms. |
| 3 | WRONG | Compare W7 with POD-\(K\) variational Verlet and enforce its reduced CFL condition. |
| 4 | NEEDS-RESTATEMENT | Define a time-integrated momentum residual, interval midpoint velocity, and any kinematic-residual work. |
| 5 | CORRECT | Keep the stated \(M^{1/2}\) similarity scaling. |
| 6 | WRONG | Repair D0, F0b/F0c, D1, W4–W7 controls and make W5 independently diagnostic. |