1. **CORRECT — A17 improves the NNLS construction without changing the ECSW objective.**  
   The gradient-contribution target remains \(b=\sum_i J_i^\top R_i\), and online ECSW remains weighted residual minimization. Replacing normal-equation refits with direct least squares and single-index entry is a defensible numerical repair. It changes the selected samples and weights, but does not inherently advantage or handicap the baseline. The support-targeted stopping rule remains a disclosed departure from tolerance-stopped ECSW.  
   **Evidence:** `experiments/jcp-hr-baselines/DESIGN.md:103–108,1176–1180,1215–1221`.  
   **Fix:** None to the objective; retain the qualified name “ECSW with support-targeted greedy NNLS.”

2. **NEEDS-RESTATEMENT — the evidence supports a promising repair, not all of A17’s causal certainty.**  
   A17 compares the old path’s **256-point** outcome with the repaired path’s **128-point** outcome. Those are explicitly different endpoints; the latter does not demonstrate completion of the former. “Two causes” and “same path as the reference” also need preserved diagnostics defining what was compared. The scratch fixture generator supports the training-only description: it constructs `Family(2, 'train')`, a training basis, and training contributions. However, A17 supplies no durable evidence path or hashes for its iteration/ban counts. I could not independently establish those exact counts from the inspected saved logs.  
   **Evidence:** `DESIGN.md:1213–1219`; scratch `nnlsdbg/mk.py:4–12` and `nnlsdbg/t7.py:4–8` under the session scratchpad.  
   **Fix:** Archive a small reproducibility record: fixture/source hashes, training-row identities, endpoint, iteration definition, bans, residuals, and reference comparison. Describe the mechanisms as diagnoses unless supported by isolated comparisons; distinguish the demonstrated 128-point result from the proposed 256-point continuation.

3. **WRONG — the iteration-bound explanation is arithmetically incorrect.**  
   The largest registered support target is \(16\times1024=16{,}384\). A bound of 40,000 is approximately **2.44 times** that target, not approximately 1.15 times it. The candidate pool has 32,768 entries; candidates and target support are different quantities. The bound itself is permissible, but the stated justification is wrong.  
   **Evidence:** `DESIGN.md:1185–1188,1220–1221`.  
   **Fix:** State “40,000 iterations for both attempts, approximately 2.44 times the largest support target,” and define whether an iteration counts entry attempts, passive-set refits, or outer iterations.

4. **CORRECT — A17 preserves the substantive A16 retry and checkpoint contract.**  
   A17 explicitly replaces 20,000 with 40,000 **for both attempts**, leaving A16.5’s unconditional “more than 1% banned” trigger and whole-path training-only selection intact. A15.6’s first-crossing checkpoints and A16.2’s construction through \(16r\) also remain applicable. Single-index entry removes the old batch-cap caveat rather than requiring retrospective checkpoint rebuilding.  
   **Evidence:** `DESIGN.md:1159–1161,1182–1188,1199–1207,1218–1221`.  
   **Fix:** Add one consolidated sentence retaining all four checkpoints, both saved attempts, the selection order, and coverage/banning labels. Explicitly retire the obsolete batch-dependent caveat.

5. **NEEDS-RESTATEMENT — rank-deficient passive sets and the proposed updating QR need a frozen numerical policy.**  
   “Every passive-set least-squares problem” solved by Householder QR is incomplete for dependent columns: ordinary triangular back-substitution need not define a usable solution. A7.2 previously registered an SVD-based fallback; A17 does not clearly preserve or supersede it. An updating QR can preserve the mathematical objective, but accumulated numerical differences can change entry/drop decisions and therefore checkpoints.  
   **Evidence:** `DESIGN.md:939–941,1167–1170,1217–1224`.  
   **Fix:** Specify the rank-detection and fallback policy, gradient refreshes, update/refactor criteria, and checkpoint-comparison tolerances. Freeze these before production. Do not describe the updating version as merely a performance substitution without checking its construction outcomes.

6. **NEEDS-RESTATEMENT — A17 acknowledges a substantial new feasibility problem; it does not resolve it.**  
   Repeated full QR has approximately cubic cumulative dependence on attained support. The stated production-scale flop estimate is plausible in order of magnitude, but is not a measured GPU-hour projection. Measuring only the smoke rank cannot certify the full rank-1024 path. QR storage, factorization workspace, gradient passes, both retry attempts, and compilation behavior must replace the previous normal-system inventory. Expensive construction could otherwise consume the cap and indirectly weaken baseline coverage.  
   **Evidence:** `DESIGN.md:627–631,719–730,865,1222–1224`; lane brief `jcp-hr-baselines.md:9–18`.  
   **Fix:** Regenerate the resource certificate for the final algorithm, including the mandatory \(16r\) continuation and retry. Preserve the 30 GPU-hour cap and incomplete-coverage labels; do not infer production readiness from the small fixture.

7. **CORRECT — the amended mathematical and fairness framework remains defensible, with its existing qualifications.**  
   The central-FD residual and LSPG gradient target are consistent. The original truncated-SVD compression of \(B\) was already withdrawn: the governing construction uses Householder QR to preserve the DEIM residual norm without discarding linear directions. Q1’s residual-tolerance exception was also withdrawn. Selection now occurs on the full Stage-B developmental cohort, with unequal selection histories disclosed. These amendments matter: auditing only the original sections would produce obsolete findings.  
   **Evidence:** `DESIGN.md:340–351,385–405,578–583,675–685,788–791`. Scheme coefficients agree with combined `t2core.py:36–37`; the 2D training configuration records `n_traj=576`, `data_seed=0`, and `dt=0.005`.  
   **Fix:** Retain the outstanding generator-deviation and budget-band ratification requirements (`DESIGN.md:544–554,788–791`). A training-only A17 fixture belongs in construction accounting; it does not erase A15/A16’s validation-informed development expenditure.

8. **CORRECT — A17 neither repairs nor invalidates the existing limits on statistical and control claims.**  
   The decision table distinguishes dominance, equivalence, crossing objectives, unresolved comparisons, missing eligibility, and absent picks. Multiplicity and in-sample selection remain limitations, disclosed through exploratory wording. Likewise, smk1 does **not** establish that every required control fires; its acceptance record contains failed phases, and A15 explicitly leaves Phase I incomplete. A17’s CPU fixture cannot replace that evidence.  
   **Evidence:** `DESIGN.md:445–464,617–619,644–650,782–785,875–877,1151–1158`; `runs/smk1/archive/output/ACCEPTANCE.json:24–43`.  
   **Fix:** Keep statistical claims developmental and conditional. Require the amended construction’s real-size control evidence before declaring Phase I complete.

**Overall verdict: NEEDS-RESTATEMENT.** A17 is a reasonable, openly motivated repair, not evidence of an unfair ECSW objective change. Top blockers are the incorrect iteration-bound rationale, insufficiently preserved dry-run evidence, unspecified rank-deficient/update-QR behavior, and an unverified revised resource envelope. **It is not production-ready evidence.**