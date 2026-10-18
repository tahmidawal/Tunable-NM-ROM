**Overall verdict: READY for the targeted design re-audit.** A16.5 resolves audit 10 items 3 and 4. I found no remaining **WRONG** statement in the governing A14–A16.5 specification, applying later corrections to earlier text. This is design acceptance, not production release or code acceptance.

Below, **D** means [experiments/jcp-hr-baselines/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md). Earlier amendments are cited where their rules remain governing.

1. **CORRECT — A16.5 closes the material-banning loophole.**

   **Evidence:** D:1199–1207; `audits/codex-design-10.md:19–27`.

   The retry now triggers whenever banned candidates exceed 1%, including when the initial path reaches every support target. If the selected path retains material banning, the explicit construction-quality limitation blocks “adequately tuned ECSW” and exhaustive non-domination claims for the cell. Support attainment can no longer conceal this limitation.

   **Fix:** None. Preserve the limitation beside conclusions, not merely in construction diagnostics.

2. **CORRECT — Retry selection and termination are now frozen.**

   **Evidence:** D:1200–1204; `audits/codex-design-10.md:29–37`.

   Both attempts have a 20,000-iteration bound. Selection chooses one whole path by reached-target count, aggregate training residual, then fixed preference for the initial path. Selection precedes validation, and both attempts’ weights and supports remain available for recomputation. This removes the prior discretion to select favourable checkpoints or resolve ties after seeing validation results.

   **Fix:** None to the design. Implementation must preserve that whole-path choice and count both attempts’ costs.

3. **CORRECT — Governing baseline mathematics survives the amendments.**

   **Evidence:** D:66, 103–108, 122–126, 338–352, 529–534, 1176–1180; references `refcore.py:64–87,112–132`.

   The central Burgers operator, residual signs, product-rule Jacobian and ECSW target \(g_i=J_i^\top\mathcal R_i\) are consistent. Positive weights correspond to a weighted least-squares objective. The Petrov–Galerkin ECSW construction is supported by [Grimberg et al., §4.1](https://arxiv.org/html/2008.02891).

   **The governing Q-DEIM compression is untruncated Householder QR, not the original cutoff SVD.** It preserves the assembled DEIM residual norm algebraically while leaving linear terms exact. The cancellation-aware checks appropriately acknowledge floating-point error.

   Support-targeted greedy NNLS is a disclosed variant; neither its support count nor its training residual guarantees a strong baseline.

   **Fix:** None. Present the QR formulation and the qualified ECSW name in the eventual report.

4. **CORRECT — Q1 remains consistent with the binding contract.**

   **Evidence:** D:158–166, 396–405, 578–582, 980–982, 1087–1091; `COMMON-BRIEF.md:§3`; `PLAN-v4-amendments.md:V4-2`.

   The governing rules retain fresh in-job Jacobians at every step, finite validity, pooled non-stationarity ≤1%, every-case tight-solve distance ≤\(2.5\times10^{-4}\), and tight/tighter agreement ≤\(2.5\times10^{-5}\). The residual-tolerance exemption was explicitly withdrawn.

   The small-gradient anchor-stall allowance is disclosed separately; A16.5 does not weaken production stationarity thresholds.

   **Fix:** None. Continue reporting nominal versus allowance-based anchor steps separately.

5. **CORRECT — Fairness is defensible as a qualified developmental comparison, not demonstrated equal-budget baseline supremacy.**

   **Evidence:** D:354–394, 544–558, 675–685, 788–791, 854–857, 1195–1197.

   Final baseline selection uses the full development cohort after subset screening. Thus the design no longer claims that `val32` or cases 6–63 provide an independent evaluation. Its “in-sample, unequal selection histories” qualification is necessary.

   Validation-informed smoke and dry-run choices now count toward baseline V expenditure. Offline construction is separately reported. The factor-1.25 expenditure band is not automatically “equal budget.” Operator-matched snapshots remain an explicit deviation from literal generator reuse, requiring the registered release decision. Deferred stepping options exclude corresponding comparators.

   **Fix:** No new amendment needed. At release, record the actual V ratio and the required generator/budget-band decisions; retain the limitations in every affected conclusion.

6. **CORRECT — H-base retains explicit outcomes for ties, missing coverage and overlapping intervals.**

   **Evidence:** D:228–243, 441–463, 617–619, 644–655, 705–713, 782–786, 915–921; lane brief `jcp-hr-baselines.md:37–46`.

   Dominance requires resolved improvement in both time and worst error. Matching requires the registered equivalence checks; overlap alone is insufficient. Crossing objectives, failed or absent picks, unavailable eligibility, reference-limited identical arms and incomplete timing coverage have explicit outcomes.

   Worst-error intervals are conditional reference-error envelopes, not population confidence intervals. Multiple comparisons and selection-data reuse remain limitations: the design explicitly restricts conclusions to developmental, in-sample, unadjusted observations.

   **Fix:** None. Preserve coverage counts and incompleteness labels alongside “no observed domination.”

7. **CORRECT — Required controls remain binding, but amended control success is not established.**

   **Evidence:** D:424–428, 600–606, 875–877, 1139–1158; `runs/smk1/archive/output/ACCEPTANCE.json:840–859`.

   Independent residual/Jacobian checks, full-rank all-points FOM recovery and per-family under-sampling controls remain required. The inspected smoke’s overall acceptance is false. A15.6 correctly retracts the implication that passing executed subchecks meant complete G-fd gates passed.

   The historical 3D ECSW degradation at \(4r\) does not establish success of the revised ladder, which excludes \(4r\). Requiring a Q1-qualified N-swap fixture is sensible but does not guarantee that it discriminates.

   **Fix:** Demonstrate the amended controls in an authorized smoke. Unfulfilled gates must continue blocking phase-I completion and affected family acceptance.

8. **CORRECT — Resource remedies and factual corrections are plausible; production feasibility remains conditional.**

   **Evidence:** D:715–730, 1093–1097, 1110–1120, 1145–1165, 1182–1188, 1200–1204.

   Separating shape pilots from POD certification is valid. Host staging and transpose-avoiding products address identified allocations. A15.6 correctly makes numerical rank diagnostic while retaining the actual POD certificate.

   Dependency checks support the training draws, snapshot-selection distinction, IC conventions and dimension-specific steps: `burgers2d_film.py:160–177`, `sep_burgers_r3.py:321–332`, `inputs/model_W1024/training.json:9–26`, `engines.py:33–38`, vendor `common.py:38–41,91–107`, combined `c2common.py:24`, `c3common.py:35`, and `t2core.py:35–38`.

   Neither the memory remedies nor the iteration bound proves the workload fits 30 GPU-hours. Advance construction through \(16r\), possible retries, snapshot storage, maximal `none` Jacobians and comparator compilation all remain material.

   **Fix:** Before production, certify measured peak memory, disk/evidence retention and **spent plus projected cost ≤30 GPU-hours**, including every allowed retry and extension.

**Overall: READY. Top remaining production blockers:** successful amended controls, a complete resource/storage certificate, and the registered phase-II fairness decisions. **No remaining design blocker from audit 10 items 3 or 4.**