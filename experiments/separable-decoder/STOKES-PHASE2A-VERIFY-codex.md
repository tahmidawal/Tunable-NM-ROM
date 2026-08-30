# Verdict: CONDITIONAL PASS

Phase 2b may proceed. The phase-2a go/no-go decision is numerically sound: the family is generically eight-dimensional and genuinely non-affine, and POD-8 leaves a meaningful accuracy gap.

Two claims need tightening:

- The single-blob parameter count is misstated.
- The POD-32 floor is not the neural decoder’s demonstrated reconstruction ceiling.

## The decisive manifold-richness question

### (a) Curvature and intrinsic dimension — PASS, with a singular-subset caveat

The centred rank 32 is not an artifact of the \(10^{-9}\) cutoff.

- Stored \(N=256\): \(\sigma_{32}/\sigma_1=4.02\times10^{-5}\), \(\sigma_{33}/\sigma_1=1.44\times10^{-12}\), a \(2.8\times10^7\) gap.
- Independent \(N=32\) recomputation: \(7.75\times10^{-5}\) versus \(2.70\times10^{-16}\); rank stayed 32 for cutoffs from \(10^{-5}\) through \(10^{-14}\).
- An analytic Jacobian recomputation at 256 fresh interior points returned rank 8 every time, with minimum \(s_8/s_1=2.0\times10^{-4}\). That is six orders above the gate’s rank cutoff.

Thus:

- rank 32 establishes that the sampled manifold is not contained in an affine 8-plane;
- Jacobian rank 8 establishes generic local dimension 8.

At coincident blobs, the Jacobian correctly falls to rank 4. The family is therefore generically an eight-dimensional immersed manifold, not a regular 8-manifold at every parameter value. Phase 2b should either include this collision/near-collision case in conditioning diagnostics or explicitly exclude it from the parameter domain.

### (b) Is POD-8 error \(3.84\times10^{-2}\) enough? — YES TO RUN, NOT YET A POSITIVE RESULT

There is meaningful room:

- POD-8: \(3.84\times10^{-2}\)
- POD-16: \(3.71\times10^{-3}\)
- POD-32: numerical exhaustion of the trial span

That is about a 10.4× gap between POD-8 and POD-16. Moreover, the exact response is a smooth nonlinear function \(U_{\rm dict}\theta(\mu)\) of eight parameters, so there is no structural nonlinear floor at 3.84%.

But “ten orders of headroom” is an overclaim. POD-32’s \(10^{-12}\) value is the full bank-span ceiling, not the finite MLP head’s demonstrated ceiling. Phase 2a contains no trained decoder and therefore cannot establish whether optimization/capacity stops at \(4\times10^{-3}\), \(4\times10^{-2}\), or elsewhere.

Required phase-2b hard gate: run reconstruction-only first and require the held-out nonlinear reconstruction oracle to beat POD-8 by a predeclared material margin, using the identical cohort and norm. Report aggregate error plus per-case median/max. If it sits at the POD-8 floor, stop before residual and timing work.

### (c) Single blob versus two blobs — CORE CLAIM PASS; PARAMETER COUNT NEEDS CORRECTION

The r3 formula [factors through \((m,s)\)](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/STOKES-DESIGN.md:201), so it is rank-limited independently of \(K=8\).

There are two distinct interpretations:

- Literal physical 2-D centre \(m=(x,y)\) plus scalar width \(s\): rank at most 3. My recomputation gave rank 3.
- The implemented expanded descriptor \(m=(x,y,\tau)\) plus kernel bandwidth \(s\): rank at most 4. My recomputation gave rank 4 at every tested point, matching `[4,4,4]`.

Therefore the reported rank-4 control is correct for the implemented single-blob analogue, but the notes cannot simultaneously call that analogue “three-parameter.” The header also says “six” degenerate directions where \(8-3=5\).

The committed [two-blob map](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_bank.py:163) genuinely achieves rank 8 generically. One provenance caveat: the single-blob control is described as having been run out of band; it is not stored in the JSON and is not the in-band S-RICH control. The in-band control is the affine rank-8 family. My independent analytic check confirms the missing single-blob result.

## 1. S1 and \(1/\sigma\) amplification — PASS

The mechanism is real:

\[
g_i=Xv_i/\sigma_i,\qquad Dg_i=(DX)v_i/\sigma_i.
\]

The stored naive bank grows to \(2.59\times10^{-12}\), only \(3.86\times\) below the \(10^{-11}\) gate, with observed amplification up to \(8.99\times10^3\) against spectral ratios \(1.3\)–\(2.5\times10^4\). Exact equality is not expected because it depends on alignment of \(DX\) with \(v_i\).

The psi bank is genuinely divergence-free by construction: [it is explicitly lifted as \(G=C\Psi\)](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_bank.py:329), and phase 1 established \(DC=0\). Independent recomputation reproduced \(DC=0\), the naive failure under contamination, and psi divergence near \(10^{-18}\).

Two caveats:

- `build_bank` obtains \(V,\sigma\) from the unprojected velocity Gram matrix and only then lifts the projected streamfunctions. Thus gradient contamination is removed from the resulting field’s divergence, but not necessarily completely from the POD orientation/scaling. Harmless for these clean snapshots; if the bank is regenerated, form the Gram matrix from the projected snapshots/induced metric.
- The injected perturbation is relative \(10^{-6}\), so calling it “invisible to any field tolerance” is false relative to phase 1’s \(10^{-8}\) field tolerance.

Phase 2b must use the metric-reorthogonalized psi bank, not the raw Gram-POD columns.

## 2. Gram-POD \(\sqrt{\epsilon}\) retraction — PASS

The numerical-analysis claim is correct. Forming \(X^\top X\) perturbs nominally zero eigenvalues at relative \(O(\epsilon)\); taking their square roots creates spurious singular values at \(O(\sqrt{\epsilon})\approx1.5\times10^{-8}\).

Independent actual-data recomputation produced a Gram rank of 137 at \(10^{-9}\), with the first spurious value \(1.42\times10^{-8}\), while direct SVD gave the physical rank 32. The stored 128–130 values are entirely plausible BLAS/eigensolver-dependent counts.

Using direct SVD plus an asserted gap is the correct fix. Strictly, “true rank 32” means the physical/effective algebraic rank; floating-point FOM noise makes the stored matrix technically higher rank.

## 3. S5 — PASS ON THE FROZEN LADDER

Independent closed-form array calculations reproduced all six minima exactly:

\[
0.035653,\ 0.164399,\ 0.430821,\ 0.769233,\ 0.942987,\ 0.987628.
\]

So the design’s flat 0.5 floor really fails at \(N=32\).

The \(N=8,\ k=8\) diagnosis is also correct: 15 modes alias to essentially zero; the minimum mass norm fell to \(5.6\times10^{-30}\), and the even-ghost control read 0.134 rather than roundoff.

The replacement is sound for the frozen ladder:

- retain the 0.5 floor only at the anchored \(N\ge64\);
- assert nondegenerate modes;
- use the enormous odd/even separation at every frozen mesh;
- separately establish nonzero \(A+\Lambda B\).

Do not call the odd/even ratio universally mesh-independent: its denominator has an \(\epsilon N^2\) floor, and the measured ratio already decreases from \(1.5\times10^{13}\) to \(8.0\times10^9\). It is robust over this ladder, not asymptotically constant.

The actual-bank A-ratio of 0.62–0.82 is expected. The ratio is basis-dependent and measures how strongly \(G\) overlaps the boundary-supported eigen-defect; the auditor’s clamped surrogate is not an anchor for a response POD bank. Both values decisively require dense \(A\).

## 4. Phase-1 defects — BOTH CONFIRMED; PHASE 1 REMAINS NUMERICALLY VALID

For a pure gradient atom, independent \(N=32\) results were:

- old continuity normalization: \(2.4835927\times10^{-2}\);
- corrected blockwise form: \(1.29\times10^{-17}\);
- global backward error: \(3.06\times10^{-19}\).

The old denominator collapses because the exact velocity and gauge multiplier are zero. This exposes a domain-of-applicability defect in the old diagnostic, not a bad solve.

At \(N=64\), 32 independent relative-\(10^{-11}\) random perturbations gave backward errors \(8.95\)–\(9.24\times10^{-14}\); none crossed \(10^{-13}\). A parallel perturbation gave \(6.1\times10^{-17}\). The reported \(9.2\times10^{-14}\) is confirmed.

Neither invalidates the phase-1 numerical certification: phase 1 did not use pure-gradient forcing, its accepted solutions remain backward-clean, and the phase-1 sources are unchanged from certified commit `7010872`. What is retracted is the generality/robustness of those two controls. Phase 2’s corrected continuity metric and \(10^{-9}\) perturbation should remain mandatory.

## 5. R=64 — PASS

Only 32 solenoidal dictionary directions can drive velocity; the 16 gradient atoms are pressure-balanced. Independently, the maximum gradient response was \(7.1\times10^{-17}\|f\|\), while the 32 solenoidal responses had robust rank 32.

Therefore \(R=64\) is algebraically unavailable. The stored \(2.8\times10^7\)–\(9.0\times10^9\) gaps confirm where the physical spectrum ends.

“\(Q\ge80\)” is correct only while retaining the current 16 gradient atoms: \(Q_s\ge64\) plus \(Q_g=16\) implies \(Q\ge80\). It is necessary, not sufficient; response rank would need to be regated.

## 6. S-SOLVE and self-comparisons — AGREED, WITH ADDITIONAL CASES

Exactly 0.0 is an assembly/factor-reuse regression check, not independent solver validation. Both routes use `stokes_saddle` and SuperLU.

Other near-self-comparisons:

- `nested_bank_max_diff == 0` repeatedly invokes the same deterministic [bank construction](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_bank_gates.py:629). It is a structural prefix regression check, not evidence of numerical accuracy or one persisted factorization.
- The affine-control POD error uses QR of the same control snapshots it reconstructs. Its rank-8 algebra is meaningful, but the \(10^{-14}\) error is not an independent held-out test.
- The blockwise residuals use the same assembled \(K\) that was solved; contrary to the notes, they are not based on independently reassembled \(D,\mathrm{Grad},L\). They certify solve accuracy, not assembly.
- The psi divergence and Hodge-purity checks are also by construction, but appropriately so: these are structural invariants, supported by the independent phase-1 operator gates.

The affine-superposition test is not literally self-comparison—it solves separate right-hand sides—but shares the same matrix and solver, so its evidence is limited to linearity, indexing, and force assembly.

## Phase-2b carried-forward conditions

Phase 2b may proceed provided that:

1. The two-blob family is frozen, with the 3-versus-4 single-blob wording corrected and the coincident-blob rank drop disclosed.
2. \(R\) is restricted to \(\{8,16,32\}\); any \(R=64\) attempt requires a larger solenoidal dictionary and a complete phase-2a rerun.
3. The reorthogonalized psi-route bank is used. Any changed family, snapshots, or \(R\) ladder must rerun S1.
4. Decoder reconstruction is gated before residual/timing work against POD-8/16/32 on the same held-out cohort, with per-case statistics. POD-32 must not be presented as the neural head’s ceiling.
5. S5 uses the frozen per-mesh rules; dense \(A\) remains mandatory.
6. S4’s full-grid path must genuinely reassemble/apply/project the MAC residual independently. S-SOLVE cannot substitute for that.
7. The corrected continuity normalization and \(10^{-9}\) perturbation control carry forward unchanged.
8. S8 includes near-collision conditioning or the family domain explicitly excludes coincident blobs.

The repository remained clean; no repository files were modified.