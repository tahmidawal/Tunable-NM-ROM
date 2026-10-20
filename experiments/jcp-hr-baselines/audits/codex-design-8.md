A13 resolves all four design-audit-7 restatement items. I found no remaining **WRONG** governing requirement in A1–A13. This is design readiness, not acceptance of the implementation, control results or production feasibility.

Evidence shorthand: **D** = [experiments/jcp-hr-baselines/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md). Later amendments govern superseded text.

1. **CORRECT — A13.1 resolves A6’s underdetermination claim.**  
   **Evidence:** D:899–906, 947–949, 1075–1077.  
   Rejecting actual ECSW support below \(r\) is now explicitly a lane control-admissibility restriction. It no longer falsely follows from Q1 or nonuniqueness. The overdetermined control ladder remains frozen.  
   **Fix:** None.

2. **CORRECT — A13.2 resolves A8’s mathematical overstatements.**  
   **Evidence:** D:974–987, 1079–1085.  
   The unconditional quadratic-convergence justification is withdrawn. Additional iterations and trajectory improvements are identified as dry-run observations, with cost still to be measured. Square residuals are no longer asserted necessarily to have roots or necessarily to fail Q1. Production settings are explicitly frozen.  
   **Fix:** None. The cited dry-run measurements remain empirical claims, not a guarantee for production.

3. **CORRECT — A13.3 resolves A9’s unsupported round-off diagnosis.**  
   **Evidence:** D:998–1003, 1013–1015, 1087–1091; COMMON-BRIEF.md:49–56.  
   Exhausting eight halvings is correctly described as a bounded backtracking stall. Round-off is only a proposed explanation. Effective exceptional thresholds are explicit, nominal and exceptional acceptance are reported separately, and non-finite steps cannot receive the allowance. Production Q1 thresholds and anchor-agreement requirements remain unchanged.  
   **Fix:** None. Preserve the distinction between agreement of anchors and proof of the stall mechanism.

4. **CORRECT — A13.4 resolves A11’s immediate storage-scope conflict.**  
   **Evidence:** D:483–485, 1017–1023, 1093–1097.  
   Full field-and-trajectory retention is explicitly a **phase-I smoke requirement**, removing the interpretation that every production grid run must be pulled under the obsolete local storage bound. Production instead requires a release amendment, cluster verification, selected field transfers, checksums and a manifest-derived peak.  
   **Fix:** No further phase-I restatement. Before phase II, the promised amendment must concretely specify retained evidence, verification before deletion, and cluster/local peak storage; A13 is not itself that completed production contract.

5. **CORRECT — The governing baseline mathematics and Q1 contract remain consistent.**  
   **Evidence:** D:338–352, 529–534, 578–591, 883–897; references `refcore.py`:64–87,112–114; combined `t2core.py`:36–37.  
   The residual signs and product-rule Jacobian agree with the central FD operator. ECSW targets the appropriate sum of \(J_i^\top\mathcal R_i\) contributions. Q-DEIM’s governing compression is **untruncated Householder QR**, superseding the original truncated SVD; it preserves the assembled residual norm algebraically and leaves linear terms exact. The stationarity ratio and pooled Q1 predicate match the common brief.  
   **Fix:** None. Describe the final method as weighted strong-residual LSPG with QR compression.

6. **CORRECT — Fairness and H-base remain defensible as a developmental comparison.**  
   **Evidence:** D:385–394, 617–619, 644–655, 675–685, 782–791, 854–857, 915–921; lane brief:27–46.  
   Final selection is on the full development cohort after subset screening—not solely dev6/cases 0–5. Unequal historical selection exposure and actual budget ratios remain disclosed. Dominance requires resolved improvement in both objectives; matching requires the registered equivalence tests and worst-envelope guard. Ties, duplicate picks, missing eligibility, failed picks and insufficient timing coverage have explicit outcomes. Multiplicity remains unadjusted and conclusions explicitly descriptive.  
   **Fix:** None. Preserve those qualifications beside conclusions. A2.3/A4.3 ratification is not re-flagged.

7. **CORRECT — Controls, dependency facts and feasibility remain appropriately conditional.**  
   **Evidence:** D:354–370, 424–436, 600–610, 715–730, 799–802, 859–877, 1025–1027. Training JSON:5,19,33–36; `burgers2d_film.py`:160–185; `sep_burgers_r3.py`:321–332; vendor `common.py`:91–106; speed3d `DESIGN.md`:60–63.  
   The amended draws, snapshot provenance, mesh conventions and stepping options agree with the checked dependencies. Independent residual/Jacobian checks, full-rank recovery and per-family under-sampling controls are required. Their real-size firing is not established by this audit. Likewise, maximal-shape pilots and the complete workload certificate—not the withdrawn estimate or the 1,024-support NNLS pilot alone—must establish feasibility under 30 GPU-hours.  
   **Fix:** None at design level; enforce the registered gates and report incomplete scope when they fail.

**Overall verdict: READY.** No remaining design blockers identified. Outstanding execution gates are real-size control acceptance, the complete resource certificate, and the phase-II evidence/storage amendment. No files changed, jobs submitted or arrays copied.