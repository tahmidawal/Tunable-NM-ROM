**Overall verdict: NEEDS-RESTATEMENT.** No files changed, scratch created, or jobs submitted.

Citation shorthand: `D` = audited `experiments/jcp-qualify/DESIGN.md`; `P1`, `Combined`, `Speed3D`, `Time2` = the respective experiment directories named in your prompt; `Campaign` = the plan’s `campaign/` directory.

1. **CORRECT — configuration matrix and certificate reuse.**  
   The matrix has 108 configurations: 20 inherited frozen-policy configurations and 88 new ones. P1’s other two certified configurations are nominal-policy R384 arms, correctly excluded. Completing four 4096² configurations with 22 additional cases each, then recomputing Q1 over the pooled 38-case population, is sound. Q2 remains explicitly a 16-case check. Numerical manifest equality, with provenance recorded separately, matches amended AM-1.  
   Evidence: `D:62–84`; `P1/report.md:92–99`; `Campaign/P1-RESTATEMENT-AMENDMENTS.md:15–21,173–177`.

2. **CORRECT — larger-step selection; minor numerical correction needed.**  
   Recomputed choices are BE **2.5**, CN-R **2.5**, GAL-CN **5**. The baseline worst error is **3.00826749%**. This is prospectively frozen **conditional on historical exploration**, not selection independent of data. One displayed number is wrong at its stated precision: CN-R at factor 10 is **7.13484106% → 7.13%**, not 7.14%.  
   Evidence: `D:86–99`; `Combined/checks/analysis-c2grid-2d_e0.json:3978,4056,4472,4810`.

3. **CORRECT — inherited LSPG Q1.**  
   Fresh autodiff Jacobians, pooled ≤1% nonstationarity, every-case tight distance, dev6 tight/tighter agreement, budget-exit treatment, and G-cert agree with P1’s amended contract.  
   Evidence: `D:186–200`; `P1/DESIGN.md:475–483`; `P1/p1score.py:197–203`.

4. **NEEDS-RESTATEMENT — GAL certificate is reasonable, but its justification and authority are incomplete.**  
   Relative Galerkin residual is a sensible root certificate, supplemented here by trajectory-distance checks. However, “the literal ratio … cannot certify anything” is false as a universal claim: an exact zero residual gives γ=0; sufficiently ill-conditioned square Jacobians can also yield γ≤10⁻³ for nonzero residuals. The singular-value lower bound explains why γ need not decrease with residual magnitude; it does not establish impossibility for every GAL configuration. The stated bound also omits the denominator’s regularizer.  
   Explicitly adopt the replacement as a governing-contract amendment and describe 10⁻⁸ as a registered residual threshold, not an established roundoff requirement. Define zero-denominator handling.  
   Evidence: `D:202–215`; `Campaign/briefs/COMMON-BRIEF.md:49–56`; `P1/p1core.py:69–70,192–198`.

5. **CORRECT — GAL G-cert reconstruction.**  
   `step_tolratio` records `rn/(tol+1e-300)`, with `tol=tolf*||A c_n||`. Multiplication by `tolf` recovers the proposed η for nonzero ordinary-scale denominators. It is not the stationarity ratio. The exceptional zero-scale case needs the policy requested above.  
   Evidence: `D:212–213`; `P1/p1core.py:192,218`.

6. **CORRECT — principal Q2 predicates and populations.**  
   The 2D 16-case rollout/own-state-ρ checks and dev6 finer-rule check match AM-2. The 3D extension explicitly freezes all 64 validation cases and cases 0–7 for finer convergence; its lattice metric matches Speed3D. Rule names, including `lat131072`, exist in the two NPZ rule inventories, which I inspected without loading large tables.  
   Evidence: `D:103–104,143–155,217–224`; `Campaign/P1-RESTATEMENT-AMENDMENTS.md:24–38`; `Speed3D/s3common.py:34–55`; `Speed3D/s3ctx.py:122–128`.

7. **WRONG — fallback timing violates the binding paired-front contract.**  
   The one-level numerical fallback is adequately declared. Its late timing path is not: comparing fallback timings with original-job timings through A does not provide the required one-job-per-mesh paired comparison. Common-reference normalization does not establish identical relative hardware effects across configurations. Finish fallback selection before the final timing panel, or rerun the candidate panel together.  
   Evidence: `D:106–114`; `Campaign/briefs/jcp-qualify.md:35–36`.

8. **CORRECT — recomputed 3D screening-front list.**  
   Applying §6 literally to **436 eligible configurations with valid screening panels**, using `ratio*t_A`, gives **12 median-front members, 5 worst-front members, 16 unique members**. These exactly match rows 1–16; row 17 is the additional H2 winner.  
   Evidence: `D:118–145,281–286`; `Speed3D/checks/analysis-grid.json:62712,151104,162498`.

9. **CORRECT — 3D negative-control triangle inequality and populations.**  
   The lower bound is  
   \(0.004212277353005088-0.00020100113425956345=0.004011276218745525\).  
   Both distances use the same `gl56` comparator and metric. I checked the archived case-ID arrays: all three trajectories contain cases **0–63 of 923801**. Thus the bound does establish failure of the successor rollout bar.  
   Evidence: `D:319`; `Combined/checks/SELECTION-3D-POSTHOC.json:1266,1285`; `Combined/configs/c3descB.json:5–7,60–79`; `Combined/select3d.py:115–138,234–246`.

10. **WRONG — “qualified 3D front” admits configurations without Q2.**  
    Deployed/post-hoc rule status is not an own-configuration Q2 certificate for arbitrary solver/step combinations. The exception directly contradicts the qualification contract. Also, qualifying only the original screening front cannot recover the global qualified front: previously dominated configurations may become nondominated when their dominators fail Q2. Either qualify the necessary remaining candidates or label the result a **qualified subset of the screened candidates**.  
    Evidence: `D:301–308`; `Campaign/briefs/jcp-qualify.md:25–30`; `PLAN-v3-amendments.md:73`; `PLAN-v2.md:107`.

11. **NEEDS-RESTATEMENT — B1 population bookkeeping.**  
    H-acc-N correctly inherits the nondegenerate per-case procedure and margin floor; P1 handles empty populations and all-zero shifted samples explicitly. But the new R512 Δt₀ three-mesh H-acc-N tests should be named explicitly. For H-acc-r at 4096², P1’s inherited transitions used 16 cases, whereas the new front has 38. State whether those transitions remain labelled 16-case results or are recomputed on 38; do not splice different populations into an unlabelled whole-range verdict.  
    Evidence: `D:169–181`; `P1/report.md:51–56`; `P1/DESIGN.md:521–531`; `P1/p1stats.py:82–123,129–161`.

12. **CORRECT — B4 anchor choice does not automatically reject honest first-order convergence.**  
    For ideal first-order error, the anchor-check ratio is  
    \((1/64-1/128)/(1/4-1/64)=1/30<0.1\).  
    Thus the /64 anchor avoids the first-order anchor problem. The vector quadrature-contamination check also matches P1 A1.7. Nevertheless, an “order 2” result over these three steps is a **finite-range observed order**, not proof against Time2’s generic asymptotic-order-1 prediction for LSPG-CN-family methods. State that limitation beside the label.  
    Evidence: `D:244–257`; `P1/DESIGN.md:502–506`; `Time2/DESIGN.md:264–269`; `Time2/report.md:61`.

13. **NEEDS-RESTATEMENT — B4 controls permit a vacuous success.**  
    TH06 satisfies “must not receive order 2” even if every case is invalid or uninterpretable. That does not exercise the order classifier. Require an independently established interpretable fixture and an order-1 result, or explicitly mark an uninterpretable control as **control unavailable**, blocking B4 labels. The manufactured oracle tests the stepper, but not the new §5 fitting/interpretability/classification pipeline; add known first-/second-order sequences through that pipeline.  
    Also explicitly size `nstep`/`nrec` for **3,200/6,400 steps**: inherited default `nstep=800` silently overwrites its final record slot.  
    Evidence: `D:259–262`; `Time2/test_lmm.py:96–98,160–168`; `P1/p1core.py:46,217–221`.

14. **CORRECT — primary 2D timing and candidate qualification.**  
    Full-field fused cost, separate evolution cost, recorded-executable identity checks, and requiring both accuracy-job and timing-job numerical qualification preserve P1’s final timing contract. Failed configurations remain separately visible. This approval excludes the cross-job fallback exception in item 7.  
    Evidence: `D:266–279,287–289`; `P1/DESIGN.md:593–602,637–643,656`.

15. **NEEDS-RESTATEMENT — dominance implementation and synthetic control.**  
    Symmetric margins prevent mutual pairwise dominance; tied pairs can remain. The relation is non-transitive, so compute dominance against **every original candidate**, not an incrementally pruned frontier. For example, successive 4% cost increases with >2% error reductions can each dominate their predecessor while the final point exceeds the first point’s 5% cost allowance.  
    The “10% costlier and 10% less accurate must be dominated” control lacks an ε fixture: sufficiently large ε makes the accuracy difference tied, and a 10% cost advantage alone still detects dominance here, but the test would not test the error branch. Freeze separate cost-only, error-only, tie-boundary, and non-transitive fixtures.  
    Evidence: `D:281–286,331`.

16. **NEEDS-RESTATEMENT — FRONT-2D handoff is close, but not fully replayable as specified.**  
    Add explicit ordered-mode asset paths/hashes, rule-content hashes, reference/store identity and eligibility, executable/configuration entry points, and per-case timed-output identity evidence. A mode hash alone cannot reconstruct the mode list; scalar timing summaries cannot verify the paired replay. Require the frozen file’s commit to be announced as the brief requests.  
    Evidence: `D:291–299`; `Campaign/P1-RESTATEMENT-AMENDMENTS.md:15–21`; `Campaign/briefs/jcp-qualify.md:39–42`; `Campaign/briefs/jcp-fom2d.md:28–33`.

17. **NEEDS-RESTATEMENT — control evidence and claim-blocking map.**  
    The new GAL budget-zero control has an explicit independent NumPy precondition, which is good; GAL zero-update motion needs the same explicit establishment. The 3D under-resolution control currently blocks only “Q2-3D”; explicitly propagate that block to B5-3D. Add positive and boundary fixtures for the new GAL certificate so an implementation that rejects everything cannot pass solely through negative controls.  
    Evidence: `D:312–333`; `Campaign/P1-RESTATEMENT-AMENDMENTS.md:146–159,188–191`.

18. **NEEDS-RESTATEMENT — resources are plausible estimates, but memory failure conflicts with never-dropped scope.**  
    The listed budget sums to **13.5**, not 13 GPU-h; as a rough estimate it is plausible against P1’s 1.9133 GPU-h and Speed3D’s 10.8597 GPU-h. It is not yet a cost certificate, especially for /128 anchors and fallback timing. The pilots appropriately include `gauss384` and `lat131072`. However, dropping R512 at 4096² on memory failure contradicts the never-dropped BE/CN-R core. Specify memory-lifetime mitigation and, if still infeasible, an explicit **core incomplete** outcome.  
    Evidence: `D:341–358`; `P1/runs/GPU-LEDGER.json:49–88,126`; `Speed3D/runs/GPU-LEDGER.json:38–44,97`.

Ranked blockers before code:

1. Remove uncertified configurations from the qualified 3D front; define its candidate universe.
2. Make fallback timing satisfy the same-job paired contract.
3. Formally restate GAL Q1, correct the impossibility claim, and define exceptional cases and positive controls.
4. Make B4 controls non-vacuous and validate the actual classification pipeline.
5. Resolve B1’s inherited-versus-expanded populations and explicitly include R512 mesh tests.
6. Complete the replayable FRONT-2D contract and the memory-failure/core-incomplete policy.

**Overall: NEEDS-RESTATEMENT.**