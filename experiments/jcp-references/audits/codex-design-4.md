**Overall verdict: READY to implement.** A3 repairs the blocking findings from audit 3 for this release’s scope. This does **not** certify numerical gate passage or authorize sealed opening.

I applied the precedence **A3 > A2 > A1 > §§1–14**. The table covers all **15** rows in [audit 3](experiments/jcp-references/audits/codex-design-3.md) marked WRONG or NEEDS-RESTATEMENT with severity B or B-open, including separately listed findings that overlap.

| Audit-3 row | Previous severity | Re-audit | Does A3 repair it? |
|---|---|---|---|
| **15. Evidence for an asymptotic regime** | B | **CORRECT** | A3.1 makes direction-inconsistent cases ineligible. The last-difference fallback no longer supports ranking. |
| **17. Numerical tolerance floor** | B | **CORRECT** | A3.2 states the stability, normalization and spatial-distribution assumptions; identifies the calibration’s limited scope; and retains case-specific residual evidence. This remains an empirical proxy, not a certified error bound. |
| **38. Restricted scoring lattice** | B | **CORRECT** | A3.3 prohibits ranking on every restricted metric. Canonical dimension validation therefore cannot confer restricted-metric eligibility. |
| **42. Spectral MMS** | B | **CORRECT** | A3.4 replaces the defective temporal ladder, uses fixed-space self-differences, and requires timestep-halving evidence before interpreting spatial convergence in both dimensions. Actual passage remains a numerical gate. |
| **52. Global freeze/opening** | B-open | **CORRECT — deferred** | A3.8 removes opening from this release. It records the authoritative-roster and selection-verification requirements for the separately audited coordinator tool. |
| **54. Store/provenance schema** | B | **CORRECT** | A3.6 distinguishes PS runs by resolution, timestep and attempt; retains stage maxima, finiteness and job-gate evidence; and identifies the selected comparison fields. |
| **A1-specific: opening transaction** | B-open | **CORRECT — deferred** | No opening transaction ships. The future specification explicitly requires locking, resumable/aborted intents and manifest-bound completion. |
| **A1-specific: A1 versus §10’s E0 verdict** | B | **CORRECT** | A3.5 defines disjoint branches: invalid dimension → FAIL; validated and every required case-kind eligible → PASS; otherwise PASS-WITH-K0-CASES. |
| **A1-specific: post-opening loader behavior** | B-open | **CORRECT — deferred** | This release has no sealed/OOD loading path. Future loading requires verification of the opening bindings, rather than mere completion-record existence. |
| **Floor validation establishes the tight proxy** | B | **CORRECT** | A3.2 adds tight-versus-tighter comparisons and a nonzero arithmetic floor. It explicitly limits the calibration’s evidential scope. |
| **Floor gate protects the measured order** | B | **CORRECT** | Both adjacent differences must exceed their respective summed proxies. The unresolved-numerator counterexample is rejected. |
| **Direction-inconsistent last difference controls true error** | B | **CORRECT** | A3.1 excludes the counterexample from ranking; its last difference is descriptive only. |
| **G4 temporal ladder: exact-error order in 2D** | B | **CORRECT** | Temporal self-differences remove the common fixed-space limit from the order calculation. An independent check of the replacement ladder also passes; details below. |
| **G4 temporal ladder: 3D stability** | B | **CORRECT** | The two previously demonstrated unstable steps are removed. The new coarsest step is the one audit 3 completed successfully. Stability and fourth-order behavior of the complete replacement ladder must still pass G4. |
| **Restriction’s metric identity and floor evidence** | B | **CORRECT** | A3.3 specifies valid factors and metric identity, while withdrawing restricted-metric ranking. Restricted proxy calculations cannot become validated accuracy claims. |

**New blockers and remaining contradictions:** I found none that require another design amendment before implementation.

In particular, amendment precedence resolves the obsolete last-difference ranking fallback, restricted eligibility, overlapping E0 statuses, original G4 timesteps and opening-enabled loader. A3.7 also resolves the previous uncertainty-source nit by fetching uncertainty from the authoritative store. A3.6 correctly makes the comparison field a *further* half-step after a stability rerun.

As an additional check, I adapted audit 3’s independent NumPy/SciPy f64 prototype **in memory**. For A3’s 2D temporal ladder, the five positive-output-time self-difference orders were:

\[
3.97230,\quad 3.99519,\quad 4.00002,\quad 4.00530,\quad 4.00856.
\]

All meet the registered interval. This used 512-point independent forcing quadrature and 64 contour points; it is supporting prototype evidence, **not** the prescribed production gate. I did not rerun the complete 3D ladder.

**Non-blocking implementation clarifications:**

- Apply PS validity checks to the selected attempt’s required runs; retain superseded failed attempts as history.
- Report achieved residuals and target attainment separately in tighter calibration runs. A requested tolerance is not evidence that it was achieved.
- A3’s metric identifier specifies \(k\ge1\); implement that evolved-time maximum consistently, superseding §5’s older unqualified maximum.
- Keep restricted outcomes explicitly diagnostic and opening-related G7 expectations aligned with unconditional sealed/OOD refusal.

No files were modified, including the lab log.