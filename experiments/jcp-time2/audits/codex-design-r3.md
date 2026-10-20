**Overall verdict: NOT CLEAN.**

- **10 — RESOLVED.** A3.1 fixes output divisibility, freezes RNG/draw order, adds oracle refinement and explicit mutation thresholds. These gates still require implementation validation.
- **12 — NEEDS-RESTATEMENT.** Primary/adjacent triples and denominators are now explicit, but A3.3 leaves the zero-denominator outcome undefined: “≥80%” of zero jointly valid cases cannot be evaluated. Specify whether insufficient adjacent coverage permits a primary-only qualified claim or withholds the claim, and whether a failed nonempty adjacent check always blocks it.
- **14 — RESOLVED.** A3.4 restricts the label to “alternating” and withdraws numerical-ringing attribution.
- **21 — RESOLVED.** A3.5 defines the discrepancy metric and timed tight-tolerance fallback; inherited verification requirements still apply.
- **22 — RESOLVED.** A3.6 requires independent identity/reference checks, unmutated acceptance, and actual rejection of corrupted artifacts.

**A3 itself:** no additional substantive error found beyond the A3.3 decision-rule ambiguity above. No files modified.