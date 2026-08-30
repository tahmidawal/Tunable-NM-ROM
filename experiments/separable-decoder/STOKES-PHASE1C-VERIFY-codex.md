# Final verdict: NOT SIGNED OFF

The phase-1 numerics remain accepted, and the required `pred_dev` correction is correct. However, the new harness still has three certification defects: S-BACKERR underweights the bordered constraints, PRECOND does not guarantee that `complete=true` means a full certified run, and the new scaled S3 gate is tautological.

## 1. `pred_dev` demotion — PASS

Let \(y=az\), \(\epsilon=a-1>0\), and \(\rho=\|x-y\|/\|y\|\). Reverse triangle inequality gives

\[
\frac{\left|\|x-z\|/\|z\|-\epsilon\right|}{\epsilon}
\le \frac{\|x-y\|}{\epsilon\|z\|}
= \frac{a\rho}{\epsilon}
\le \frac{a\,\tau_{\rm field}}{\epsilon}.
\]

That is exactly what `implied_by_field_gate_*` records in [stk2d_fom_gates.py](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_fom_gates.py:708).

Therefore any `pred_dev` gate using this bound is redundant: passing the field gate already forces it. Demotion is the right correction, and no `pred_dev` assertion remains.

## 2. S-BACKERR — USEFUL, BUT NOT ACCEPTED AS CURRENTLY CLAIMED

### (a) Normalization and threshold

\[
\eta=\frac{\|Kx-b\|_2}{\|K\|_F\|x\|_2+\|b\|_2}
\]

is a legitimate global normwise backward error if relative Frobenius perturbations of \(K\) and 2-norm perturbations of \(b\) are intended. It is independent of the manufactured truth and condition number affects forward error, not backward stability.

The \(10^{-13}\) threshold is a reasonable engineering threshold—about 450 machine epsilons—and repository history places it before the certified artifact. It is not, however, rigorously derived from the stated \(O(\sqrt{\mathrm{nnz}}u)\) argument: at \(N=256\), that expression is \(1.27\times10^{-13}\) using unit roundoff, or \(2.54\times10^{-13}\) using NumPy epsilon. Sparse LU pivot growth also prevents that asymptotic statement from being a hard bound.

Verdict: defensible operationally, but the prose overstates its theoretical derivation.

### (b) Direction independence

It is independent of alignment with the manufactured solution, unlike `pred_dev`. It is not literally perturbation-direction independent because it measures \(\|K\delta x\|\).

At \(N=32\), equal relative velocity perturbations of \(10^{-11}\) gave:

- random, seed 20260830: \(1.99976\times10^{-13}\);
- alternating high-frequency: \(2.78032\times10^{-13}\);
- parallel to velocity: \(2.38238\times10^{-15}\).

Thus there is no parallel/orthogonal truth-field pathology, but sensitivity still varies by singular direction of \(K\). The documentation should say “reference-direction independent,” not simply “direction-independent.”

### (c) Falsifiability and real coverage

The reported perturbation is reproduced exactly: baseline \(5.96612\times10^{-18}\), then \(1.99976\times10^{-13}\), which fails.

It is not decoration. It can catch:

- an inaccurate factorization or silent sparse-solve failure;
- accidental use of a loose iterative solve;
- corruption of the returned solution producing a large algebraic residual;
- such errors well below the \(10^{-8}\) closed-field tolerance, including on generic/free-slip arms.

It cannot detect a wrong \(K\) or wrong \(b\) that is solved accurately.

The “every solve” claim is false. The 14 rows cover six frozen, three generic, and five free-slip solves, but the harness actually calls `solve_stokes` 17 times: the S0 probe and both S-NU solves are excluded ([S0](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_fom_gates.py:473), [S-NU](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_fom_gates.py:806)).

### (d) Bordered/gauge structure — FAIL

The global \(\|K\|_F\) is dominated by the \(O(h^{-2})\) momentum block. The divergence and unit-valued gauge borders contribute negligibly, making this metric dependent on arbitrary block/row scaling.

Concrete negative control at \(N=128\): adding a constant \(10^{-8}\) pressure offset gives

- gauge-row residual: \(1.6384\times10^{-4}\);
- S-BACKERR: \(4.46\times10^{-14}\), which passes.

The field gates recenter pressure before comparison, so they do not catch this gauge violation either. `p_mean_raw` is only diagnostic.

Required correction: retain global S-BACKERR, but add asserted blockwise residuals—especially normalized continuity and raw mean-zero gauge—or a suitably equilibrated/componentwise backward error.

## 3. PRECOND — FAIL

The `if not __debug__: raise RuntimeError(...)` reasoning is correct: an assertion cannot detect its own removal under `-O` ([PRECOND](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_fom_gates.py:495)).

The overall guard is incomplete:

- `SMOKE=1` bypasses `ALLOW_CPU`, rank, and free-slip requirements, yet execution still sets `complete=true`.
- Nonempty ladders are insufficient. Environment overrides can shorten or alter `NS`, `LADDER`, `GEN_NS`, `ADJ_NS`, `RANK_NS`, `FREESLIP_NS`, `M_MODES`, or `NU` and still produce `complete=true`.
- There is no expected-gate manifest or exact row-count assertion.
- A later NaN can be silently ignored by Python `max`/`min`; e.g. `max([finite, nan])` returns the finite value. A failed last solve can therefore pass multiple aggregate assertions, including S-BACKERR.
- There is no exception handler that actively writes `complete=true`; intermediate saves correctly retain `false`. A failure before the first save can nevertheless leave an older same-path `complete=true` artifact untouched.

The committed JSON itself has the correct frozen configuration, `SMOKE=0`, `ALLOW_CPU=0`, all expected gates, and 14 finite backward-error rows. The defect is in what the harness generally allows `complete=true` to mean.

## 4. \(0.99/\sqrt M\) derivation — DERIVATION SOUND; GATE TAUTOLOGICAL

The implication is mathematically correct: one matched column with cosine \(c\) contributes at least \(c/\sqrt M\) to the normalized Frobenius metric.

But here [the control is defined directly as](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_fom_gates.py:233)

\[
\Psi_j=\frac{X_j}{h\|X_j\|},\qquad X_j=\mathrm{Grad}\,\chi_j.
\]

Consequently:

- the matched cosine is identically 1;
- the self component alone guarantees  
  \(\sqrt M\,\mathrm{ctl\_fro}_j\ge1\);
- orthogonality of these cosine-gradient modes makes it exactly 1.

So the measured `1.000000000000` is indeed evidence of comparison with a normalized copy of itself. The `>=0.99` assertion cannot fail for any nonzero \(X_j\), even if `Grad` itself is wrong.

My prior suggestion to gate the derived constant missed this self-normalized construction. The scaled quantity should be demoted to a diagnostic or rebuilt using an independently constructed control.

## Scope confirmation

The operators, generic MMS, and underlying repaired S3 construction did not change. From the previously verified revision:

- [stk2d_common.py](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_common.py:383) changes only by adding solve-residual diagnostics.
- The generic MMS has no diff.
- S3 changes only by recording/asserting the new scaled value; its `Chi/X/Psi` construction is unchanged.
- The certified JSON was generated from commit `e4c3231`, and the two source files are unchanged between that commit and current HEAD.

Before sign-off: repair PRECOND completion semantics and finite checks, cover all solve invocations plus bordered constraints in S-BACKERR, and demote/rework the tautological scaled S3 assertion. Also correct stale text that still describes the retracted prediction gate in [the driver header](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_fom_gates.py:41) and [the notes glossary](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/STOKES-NOTES.md:800).

No repository files were modified; verification was CPU-only.