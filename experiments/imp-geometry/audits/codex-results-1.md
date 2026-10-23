Read-only audit completed; no files changed, jobs submitted, or arrays copied.

**(a) CORRECT — A4.2 labels and side status.**

| Cell | τ=1% | τ=3% | τ=10% |
|---|---|---|---|
| L | No evidence: reference | No evidence: reference | No evidence: reference |
| H | Loss | Loss | Loss |

All validation method sides are matched. Zero-field exclusion does not apply.
L has 4/32 ineligible validation references and 6/32 on test; its ratios are descriptive, not qualified losses.
H has 32/32 E0 references on both cohorts.
Allowances recompute correctly: L validation/test 0.000920459/0.001073271; H 0.000272796/0.000272817. All satisfy U≤τ/5; L fails eligibility independently.

**(b) CORRECT — S arithmetic.** Independently recomputed median paired ratios and max–min over eligible finalists reproduce every stored summary, including nulls.

| Cell/statistic | Validation S at 1%, 3%, 10% | Test S at 1%, 3%, 10% |
|---|---|---|
| L worst | 0.033576, 0.039898, 0.091561 | null, 0.024175, 0.090301 |
| H worst | 0.033220, 0.067675, 0.119209 | 0.035667, 0.066717, 0.120212 |
| L median | 0.014655, 0.060096, 0.214449 | 0.012782, 0.062492, null |
| H median | 0.029227, 0.094950, 0.216100 | 0.030590, 0.095479, null |

Worst-error validation winners among competitors: L ECSW/PODG/PODG; H ECSW/ECSW/PODG.

**(b) NEEDS-RESTATEMENT — timing audit completeness.**
All 480 timing records have zero compilations and T-id≤4.62×10⁻¹⁶.
Sentinel drift: L validation/test 0.397%/1.047%; H 0.308%/5.380%, all below 10%.
Validation T-dbl ratios pass: L 0.9955–1.0041; H 0.9889–1.0216.
Test T-dbl stores only family-level `passed=true`, discarding individual ratios. Its pass status is recorded, but cannot be independently recomputed from these JSONs.

**(c) CORRECT — qualification filtering; blanket “all test finalists passed” would be WRONG.**
All validation finalists meet τ unbatched; all selected validation banks pass recomputed Q1/Q2 predicates.
On L test, both 1%-worst bank finalists are unresolved and fail Q2; hence null S.
L’s 3%-worst nt20 finalist misses τ and fails Q2; the qualified nt40 finalist supplies S.
L’s r128 1%-median finalist also fails Q2 and is excluded.
Both cells’ 10%-median banks fail Q1 tight-solve distance: L ≈2.840×10⁻⁴, H ≈2.590×10⁻⁴ against 2.5×10⁻⁴; hence null S.
Other test competitors that miss or remain unresolved are correctly excluded.

**(d) CORRECT — A10 stop outcome:** “not supported at S1; S2 untested.” No confirmed candidate win funds continuation.
Under A4.2, H is withdrawn for measured loss; L has insufficient evidence. No geometry WIN or attribution is established.

**(e) CORRECT — specified controls and diagnostics.**
MF_MG_finest fires on both cells; MF_REF_real is true.
G-FLOOR passes independent error-versus-floor checks.
SEM cross-check maxima: L 0.00015813, H 0.00014933, below 0.001.
D_bnd medians, validation/test: L 0.09254/0.08133; H 0.09952/0.06761, all above 0.05 with converged solves.
None of these fails or rescues L’s reference eligibility; D_bnd alone does not establish attribution.

**(f) WRONG — `analysis/decide_s1.py` is only partly faithful to A10.**
Its current stop decision is correct, but line 27 unconditionally emits “WIN not decidable,” ignoring the reached withdrawal.
Lines 25–26 also equate every absence of confirmation—including missing cells or pending tests—with the stop condition.
It omits descriptive S for reference-ineligible L and test S for losses, despite its stated readout purpose.

**(g) CORRECT — GPU ledger.**
Live sacct agrees for all 11 job states and rounded charges: 30.6788889 GPU-h → ledger 30.6789.
Step 2 used 17.9863889 GPU-h, below 30; total below 62.
Smoke used 1.0741667 GPU-h: the disclosed 0.07417-hour allowance overrun is real.

**OVERALL: NEEDS-FIX (A10 readout logic/withdrawal wording; qualify test T-dbl auditability; distinguish L descriptive ratios and failed test finalists from qualified results).**