Revision 2 is substantially better, but it is still not implementation-ready. Only two of the nine required changes are fully closed; the force geometry, executable gate thresholds, and strong-form baseline remain material blockers.

Line references below are to [STOKES-DESIGN.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/STOKES-DESIGN.md).

## Part A — closure check

| # | Required change | Verdict | Evidence |
|---:|---|---|---|
| 1 | Fully specify and manufacture-test MAC | **PARTIALLY-CLOSED** | Lines 58–85 now fix \(h\), active DOFs, masses, boundary elimination, odd ghosts, and weighted adjointness. S-FOM at 189–197 adds the correct kind of manufactured test. Still missing: explicit indexed \(D\), Grad, and \(L\) stencils; an actual gauge prescription; the chosen pressure and force formulas; error norms; and numerical convergence/adjoint pass bands. “At roundoff” and “require second order” are not executable thresholds. |
| 2 | Rewrite S5 as expected no-slip boundary noncommutation | **PARTIALLY-CLOSED** | Lines 232–245 correctly identify the odd/even ghost defect, make roundoff a failure, require dense \(A\), and quote the old direct residual ranges correctly. But “must be \(O(1)\)” is not a pass threshold. Also, the quoted \(\approx0.34\) secondary diagnostic belongs to the old unnormalized \(\Phi\); r2’s normalization changes it. |
| 3 | Resolve force projection and Hodge richness | **NOT-CLOSED** | Lines 151–162 make \(b\) affine-precomputable; lines 164–175 request Hodge-energy and snapshot-rank reports. But no \(Q\), force shapes, amplitude ranges, parameter dimension, minimum solenoidal response rank, or pass threshold is specified. With independently varying affine amplitudes, the nonlinear-head experiment is either linear-trivial or dimensionally under-resolved; see Part B. |
| 4 | Fix POD constraints and the missing affine mean | **PARTIALLY-CLOSED** | Lines 120–133 contain the correct mean term, acknowledge \(1/\sigma_i\) amplification, project snapshots into \(\ker D\), and re-gate after normalization. Missing: the projector, normalized pass thresholds, and a gate on \(D\bar u\). The text gates only \(g_i\), even though a non-solenoidal mean breaks the entire constrained decoder. |
| 5 | Replace S2/S3 with scale-aware, pressure-sensitive, row-coverage gates | **NOT-CLOSED** | Lines 208–223 name all requested diagnostics. They do not state acceptance criteria for field-path divergence, “non-negligible” matched pressure response, boundary leverage, injected residual detection, numerical rank, or conditioning. Under the requested closure standard, these remain reports, not gates. |
| 6 | Specify \(M\), normalize \(\Phi\), and gate \(\operatorname{rank}(AJ_h)\) | **NOT-CLOSED** | Lines 99–102 establish mass normalization; lines 222–223 and 263–267 mention an \(M\)-ladder and conditioning. No actual \(M\) values, mode ordering, \(M/K\) ratio, rank requirement, \(\sigma_{\min}\) normalization, or threshold is fixed. “\(M>K\) comfortably” is not a design parameter. |
| 7 | Deconfound the \(R\)-frontier and add three linear controls | **PARTIALLY-CLOSED** | S7 and S9, lines 253–267, include POD-\(K\), POD-\(R\), the direct \(G\)-span solve, nested banks, fixed \(K\), and parameter counts. But the \(R/M\) schedule and thresholds are absent, and S7c may be underdetermined when \(M<R\). A unique direct \(G\)-span solve requires \(M\ge R\) and \(\operatorname{rank}A=R\), or a different Galerkin test space. |
| 8 | Treat MAC skew convection as new NS machinery | **CLOSED** | Lines 269–285 explicitly reject reuse of the scalar collocated positive-upwind tensor, require staggered interpolation and a new bilinear tensor with build/Jacobian gates, and state that target NS is unsteady. |
| 9 | Clarify cost scope and MAC bookkeeping | **CLOSED** | Lines 58–75 fix \(h=1/N\) and \(n_u=2N(N-1)\); lines 151–162 separate affine and timed non-affine forcing; lines 280–285 identify target NS as unsteady and state what steady Stokes does not test. The shorthand “cost per residual is \(MR\)” still needs a minor correction, discussed below. |

The quoted numerical values were transcribed accurately:

- Raw \(\|D\Phi\|\): \(2.27\times10^{-13},4.55\times10^{-13},9.09\times10^{-13}\) at \(N=64,128,256\). Removing the absolute \(10^{-14}\) gate is correct.
- Normalized pressure projection: \(5.91\times10^{-18}\). The conclusion—roundoff-level pressure annihilation for the tested compatible MAC pair—is correct.
- S5 eigen-residual ranges are correct, including \(0.988\)–\(0.99985\) at \(N=256\).
- The old surrogate-bank secondary ratio was indeed approximately \(0.34\), and the conclusion that dense \(A\) is required is correct.

## Part B — new errors and unresolved issues

### 1. Affine forcing

**Precomputability verdict: valid, with a cost correction.**

For fixed \(f_q\),

\[
b_q=\Phi^\top M_u f_q
\]

is precomputable, and each query forms

\[
b(\mu)=\sum_q\theta_q(\mu)b_q
\]

in \(O(MQ)\), independent of the grid. Thus the entire vector \(b(\mu)\) is not globally precomputed; its basis projections are. Repeated LM evaluations can reuse the assembled query-specific \(b\).

**Nonlinear-manifold verdict: blocking and currently vacuous.**

Steady Stokes is linear. After pressure elimination,

\[
u(\theta)=\sum_{q\in\mathrm{sol}}\theta_q\,u_q,
\]

where only independent solenoidal force components count. Gradient components change pressure but contribute exactly zero velocity.

Let \(Q_s\) be the number of linearly independent solenoidal velocity responses.

- If amplitudes vary independently and \(Q_s\le K\), the velocity family lies in a linear subspace of dimension at most \(K\). A linear POD-\(K\) decoder can represent it; the nonlinear head is unnecessary.
- If amplitudes vary independently and \(Q_s>K\), the family has intrinsic dimension \(Q_s>K\). A continuous \(K\)-latent decoder cannot represent it exactly; the experiment measures lossy compression, not useful nonlinear geometry.
- A nontrivial \(K\)-latent nonlinear test needs at least \(K+1\) independent solenoidal response directions, but their amplitudes must be generated by at most \(K\) underlying parameters through a nonlinear, curved map \(\theta(\mu)\). The centered snapshot rank must then be demonstrably \(>K\).

Therefore “independently varied amplitudes” is incompatible with the intended nonlinear-head test. The quadrature-free and pressure-elimination rehearsal remains meaningful, but the nonlinear-decoder comparison is vacuous as written.

### 2. Affine mean

**Formula verdict: correct. Constraint handling: incomplete.**

With \(L=\Delta_h\), the extra term

\[
-\nu\,\Phi^\top M_uL\bar u
\]

is the correct and complete momentum-residual contribution. No additional pressure term is created.

But \(\bar u\) must satisfy

\[
D\bar u=0.
\]

Otherwise \(u=\bar u+Gh(z)\) is never divergence-free, even if every column of \(G\) is. The mean should be formed after projecting every snapshot into \(\ker D\), or projected separately, and then gated with the same normalized criterion as the bank modes.

No-slip requires less stored-state machinery than the text suggests:

- Boundary-normal values are eliminated and therefore automatically zero.
- Tangential wall values are not DOFs; no-slip is supplied by the odd-ghost extension used by \(L\).

So the mean inherits the representation’s no-slip treatment, but its divergence still needs an explicit gate.

### 3. Mass normalization

**Pressure-elimination verdict: unaffected.**

For any nonsingular test-space transform \(\widehat\Phi=\Phi S\),

\[
D\widehat\Phi=0,\qquad
\widehat\Phi^\top M_u\mathrm{Grad}\,p
=S^\top\Phi^\top M_u\mathrm{Grad}\,p=0.
\]

The pressure identity survives mass normalization or orthonormalization.

Per-mode S5 eigen-residuals are invariant under scalar column normalization. A generic QR that mixes modes with different \(\lambda_{k\ell}\), however, destroys the meaning of a diagonal \(\Lambda\); use diagonal normalization of the already orthogonal sine-curl modes, or transform the modal operator consistently.

**The predicted secondary diagnostic changes.** Re-running the archived surrogate calculation with r2’s mass normalization gave:

| \(N\) | Archived unnormalized | Mass-normalized |
|---:|---:|---:|
| 64 | 0.340584 | 0.370931 |
| 128 | 0.342758 | 0.372325 |
| 256 | 0.343303 | 0.372674 |

Thus “the audit measured \(\approx0.34\)” is historically accurate, but it is not r2’s expected value under its new normalization. The qualitative \(10^{-1}\)–\(1\) expectation and dense-\(A\) conclusion remain correct.

Likewise, \(5.91\times10^{-18}\) is a correct archived measurement, not a value the normalized implementation must reproduce digit-for-digit.

### 4. Strong-form/Petrov framing and the incumbent

**Mathematical distinction: correct. Baseline inheritance: incorrect.**

At the algebraic discrete level, divergence-free sine curls are legitimate left tests of the strong MAC residual. They need not lie in \(H_0^1\), because no integration by parts is performed. For the integrated weak form, their nonzero tangential trace leaves viscous boundary terms, so they are not admissible \(H_0^1\) tests.

One sentence should still be corrected: pressure elimination does not follow merely from zero normal trace; it requires \(D\Phi=0\) together with weighted \(D/\mathrm{Grad}\) adjointness.

The incumbent [sep_poisson_qf.py](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/sep_poisson_qf.py:1) explicitly uses a weak/eigen-reduced, values-only residual. It declares `kind="weak"` at line 227, fits NNLS using decoded values at lines 263–271, and evaluates

\[
W_l\bigl(\Phi^\top W_q u\bigr)-f_m
\]

without applying \(L\) at lines 317–325.

For scalar Dirichlet Poisson, the sine eigenrelation makes that equivalent to a row-scaled strong projection. S5 proves that equivalence fails for the no-slip Stokes curl modes.

Consequences:

- Timing protocol, burn-in, balanced ordering, cohorts, and cancellation-aware checks can be inherited.
- The incumbent NNLS nodes, weights, fit target, and “weak residual” identity cannot be inherited.
- Stokes needs a newly defined strong-form EQ arm on the two face lattices, including stencil gathers or cached \(LG\), its own NNLS target, and a gate against the full strong MAC projection.

Without that, calling S6 the “incumbent NNLS-sampled arm” is misleading and the comparison is invalid. The QF-versus-full strong-form comparison itself remains valid.

### 5. Manufactured Stokes solution

**Validity verdict: valid, but under-specified in r2.**

The proposed streamfunction gives

\[
u=\left(
\pi\sin^2(\pi x)\sin(2\pi y),
-\pi\sin(2\pi x)\sin^2(\pi y)
\right),
\]

which is divergence-free and zero on all walls. Choose, for example,

\[
p=\sin(2\pi x)+\cos(2\pi y),
\qquad
f=-\nu\Delta u+\nabla p.
\]

Everything is analytic and directly evaluable on the two face lattices.

A read-only CPU sparse-MAC check using the archived stencil conventions produced:

| \(N\) | Velocity relative error | Pressure relative error |
|---:|---:|---:|
| 8 | \(5.303\times10^{-2}\) | \(2.617\times10^{-2}\) |
| 16 | \(1.295\times10^{-2}\) | \(6.455\times10^{-3}\) |
| 32 | \(3.219\times10^{-3}\) | \(1.608\times10^{-3}\) |
| 64 | \(8.036\times10^{-4}\) | \(4.017\times10^{-4}\) |

The \(32\to64\) observed orders were 2.002 for velocity and 2.001 for pressure. So second-order convergence in both variables is the correct expectation for this manufactured solution.

R2 must nevertheless freeze the actual \(p\), analytic \(f\), mean-zero gauge, mass-weighted error norms, and an accepted order band over the stated \(32,64,128,256\) ladder.

### 6. New contradictions, duplication, and overclaims

- Line 108 writes a strong residual containing \(+\nu Lu\), while the governing equation and lines 141–142 use \(-\nu Lu\). With S5’s convention \(L\Phi=-\Phi\Lambda\), the latter is correct.
- Line 147 calls this a “single-component case,” contradicting the vector-valued design.
- S7c is not necessarily a valid direct control. If \(M<R\), \(Ac=d\) is underdetermined. Either require \(M\ge R\) and full column rank, or use a proper \(G\)-Galerkin reduced Stokes solve.
- “Cost per residual is \(MR\)” omits head evaluation and query-specific \(b\) assembly. The honest expression is head cost \(+\ O(MR)\) per LM evaluation, plus \(O(MQ)\) once per affine query.
- S3(d) and S9 duplicate the \(M\)-ladder and \(AJ_h\) rank/conditioning gate. Consolidate them so one threshold cannot drift from the other.
- S8 duplicates the mass-metric rule already stated at lines 99–102.
- “Every threshold below is normalized” is false: many entries contain no threshold at all.
- S-ADJ’s “restriction to rows the test space can reach” is not mathematically precise. Gate the test-projected adjoint defect, e.g. \(\Phi^\top(M_u\mathrm{Grad}+D^\top M_p)\), rather than an undefined row restriction.
- S4 mentions a cancellation-aware residual test but never defines its formula or whether it is an alternative to the relative \(10^{-12}\) test.

## Required fixes, ranked by likely wasted time

1. **Freeze the complete MAC/FOM contract first:** indexed stencils, gauge, manufactured \(p\) and \(f\), norms, convergence bands, and adjoint thresholds. A wrong FOM contaminates every snapshot and control.
2. **Redesign the affine parameter family before generating data:** distinguish \(Q_s\) from gradient-only shapes, require at least \(K+1\) independent solenoidal responses, and use a curved \(K\)-parameter amplitude map rather than independently varying all amplitudes.
3. **Define the new strong-form full/EQ baseline:** fit target, two-lattice sampling, stencil closure, setup accounting, and exactness gate. Do not reuse the incumbent weak Poisson EQ rule.
4. **Make the \(M/R\) geometry executable:** fix mode ordering and the \(M\)-ladder, require \(M\ge R\) for S7c or replace it with \(G\)-Galerkin, and set rank/\(\sigma_{\min}\) thresholds.
5. **Turn S1–S5 into actual gates:** numerical pass/fail rules for mean and bank divergence, pressure sensitivity, leverage/injections, S5’s analytic boundary defect, and cancellation-aware residual/gradient comparisons.
6. **Apply and gate the affine mean explicitly:** project before centering or project \(\bar u\) separately, then gate \(D\bar u\).
7. **Update normalization-dependent expectations and remove contradictions/duplicates**, including the S5 secondary value, viscous sign, cost formula, and duplicated \(M\)-rank gate.

No repository files were modified.

**Overall verdict: IMPLEMENT WITH LISTED FIXES.**