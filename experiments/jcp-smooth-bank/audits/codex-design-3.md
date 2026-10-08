**No remaining scientific-design issue blocks round-1 training.** However, A2 requires a passing R2b test **before submission**; the corrected test exists, but I found no saved passing record. Training/job implementation clearance remains separate from this design review.

1. **Provenance — CORRECT.** A1’s audited corrections remain intact. **Fix:** none.

2. **Lane evaluation procedure — CORRECT.** The common rotation procedure and initialization distinction remain valid. **Fix:** none beyond the safeguards in item 9.

3. **Sobolev justification — CORRECT.** A2.2 correctly limits conclusions to mesh-derivative agreement, reports common-support comparisons, and requires H1 to survive both stencils. A2.1 explicitly registers stochastic gradient sampling during the final phase. **Fix:** none to the training design.

4. **Metrics — NEEDS-RESTATEMENT.** Ladder-specific censoring and unresolved bandwidth outcomes are repaired; A3’s block classifier works on the manufactured controls. But projected common-reference states still differ physically between banks. **Fix:** retain projection errors beside those ladders and describe agreement as supporting—not isolating—the mechanism. Explicitly state that A3 replaces A1’s old fit endpoint; “other rules unchanged” is unnecessarily ambiguous.

5. **Decision equations — NEEDS-RESTATEMENT.** Conservative censoring bounds, ties, ranking and most eligibility branches are repaired. Remaining ambiguities:
   - A1 still compares a **relative change in gradient error** with a stencil discrepancy normalized by the target norm. Those are different scales. Use an absolute error change with consistent normalization, or explicitly replace that threshold with A2.2’s two-stencil criterion.
   - A treatment censored at 0.06 can pass H2 at 0.01; its 0.06 ranking factor is described only as “no reduction,” leaving the logarithmic score undefined.
   - Explicitly reject nonfinite metric **numerators**, missing measurements and ineligible comparators too.  
   **Fix before evaluation/selection; not training.**

6. **Controls — WRONG as a complete repair.** C2/C3/C5b now pass; R0’s scope, R2b’s actual-path test and C4’s populations are repaired in the contract. **C6 remains insufficient:** default and tighter-tolerance runs can stall at the same state and pass the distance check; validation cases also lack that sensitivity check. **Fix:** require a registered stationarity/accepted-exit criterion on every required rollout. Before training submission, record a passing R2b execution; updated test source alone does not establish passage.

7. **Cohorts/references — CORRECT.** A2.4 explicitly propagates provisional reference status to every verdict and preserves development/validation scope. **Fix:** none.

8. **GPU accounting — NEEDS-RESTATEMENT.** `tr1.json` now matches **4 GPUs × 7 h = 28 GPU-hours**, and evaluation scheduling is explicit. But the amended allocation totals **28 + 18 + 20 + 16 + 10 = 92**, not ≤90. **Fix:** correct the total or reduce another allocation. This does not invalidate the specified training allocation.

9. **Safeguards/coarse control — NEEDS-RESTATEMENT.** Common-129-node scoring and the limited coarse-data interpretation are correct. Artifact identifiers/hashes and bank-conditioning thresholds improve matters. **Fix:** specify handling of singular/nonfinite rotations and tested-operator conditioning. Preserve the entire fitted spectrum: `spectra()` currently saves only the first 256 entries of the 512-degree envelope, although classification can use degrees through 511. **Evaluation/postprocessing concern, not a reason to postpone fitting banks.**

I independently executed the NumPy/SciPy control functions read-only: the saved pre-check matches **exactly**. Both resolutions give C2 bandwidths **20 < 64** and geometric classifications; C3 is algebraic at both resolutions, with the recorded 512-point exponent **2.15048**; C5b checks both axes and reproduces **1.68754×10⁻¹³**. These validate the manufactured controls, not asymptotic decay claims for arbitrary banks.

No files were modified, including the lab log.