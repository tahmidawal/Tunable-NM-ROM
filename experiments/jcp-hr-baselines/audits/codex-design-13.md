1. **WRONG — A18’s claimed evidence archive is incomplete. Audit 12 item 2 remains open.**  
   A18.3 says the fixture’s scripts and logs are archived. However, `runs/nnls_fixture/README.md:3–6` names `G.sha256`, `t4.log`, and `t7.log`; none exists in that directory. Only the old algorithm’s log records the claimed counts. Moreover, `t_old.py:3–8` imports the current `nnls` module without pinning the historical implementation, so the script alone cannot reproduce the old algorithm. The asserted repaired/reference trajectory agreement is therefore not independently established by this archive.

   **Evidence:** [DESIGN.md:1233](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md:1233); `runs/nnls_fixture/README.md:1–6`, `old_algorithm_t.log:1–3`, `t_old.py:3–8`, `t7.py:3–8`.  
   **Fix:** Preserve the missing logs, fixture hash, exact source revisions, and checkpoint comparison records—or explicitly mark the repaired counts and trajectory agreement as unverified historical observations. This finding disproves archival completeness, **not** the reported numerical counts.

2. **CORRECT — A18 fixes audit 12 item 3 and qualifies the causal claims appropriately.**  
   \(40000/(16\times1024)=2.44140625\). A18 defines an outer iteration, applies the bound to both attempts, distinguishes the 256-point and 128-point endpoints, and replaces causal certainty with “diagnoses.” It also explicitly retains all four checkpoints, whole-path selection, retry rules, and coverage/banning labels.

   **Evidence:** [DESIGN.md:1233](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md:1233), lines 1233–1240; A16.5, lines 1199–1207.  
   **Fix:** None to these corrections; supply the missing evidence identified above.

3. **CORRECT, with a remaining qualification — A18 defines the current passive-set fallback; the production update algorithm remains unfrozen.**  
   Householder QR on original columns, followed by minimum-norm SVD with the registered cutoff when the stated trigger fires, is a concrete numerical policy. This resolves the immediate ambiguity in audit 12 item 5. The diagonal-ratio trigger is a heuristic for numerical rank, however, not a condition-number guarantee. A18.5 correctly treats updating QR as future work requiring frozen tolerances and checkpoint comparisons.

   **Evidence:** [DESIGN.md:1230](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md:1230), lines 1230–1232, 1242–1246; A7.2, lines 939–941.  
   **Fix:** Before production, specify update/refactor triggers, rank handling, gradient refreshes, and acceptable checkpoint discrepancies. Describe the diagonal test as a diagnostic trigger, not a comprehensive stability certificate.

4. **NEEDS-RESTATEMENT — A18 defers audit 12 item 6; it does not resolve feasibility.**  
   The acknowledgment is honest, but no final algorithm/resource envelope is established. The maximal 3D contribution matrix is approximately 27.4 GB. At support 16,384, one \(104448\times16384\) f64 passive-column or thin-Q array is approximately 13.7 GB, before additional factors, workspaces, copies, and retries. The maximal 2D `none` Jacobian at rank 512 is approximately 4.29 GB. These allocations must be considered together with snapshots, POD workspaces, and compiled executables.

   The existing maximal-rank NNLS pilot stops at support 1,024; it cannot directly certify a 16,384-point production path. The earlier GPU-hour estimate is explicitly withdrawn, and production storage remains deferred.

   **Evidence:** `DESIGN.md:627–631,719–730,865,1025–1027,1093–1097`; [A18.5:1242](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md:1242); COMMON-BRIEF.md:87–92.  
   **Fix:** Supply the final manifest-derived certificate, covering full continuation, retry, QR/SVD workspace, compilation shapes/counts, host/device peaks, cluster disk peaks, and cumulative expenditure under 30 GPU-hours. A reduced scope must explicitly restrict the resulting claims.

5. **CORRECT — the governing baseline formulas remain mathematically consistent.**  
   The central-FD operator, multistep residual, and corrected residual Jacobian agree. ECSW’s target is the sum of nodewise \(J_i^\top R_i\), consistent with weighted residual minimization. The amended Q-DEIM construction preserves the full DEIM-approximated residual norm through **untruncated Householder QR** of \(B\); the original truncated-SVD construction is no longer governing. No new objective error follows from A17–A18.

   Support-targeted ECSW is a disclosed variant, not evidence that the baseline is automatically strong or exhaustively tuned.

   **Evidence:** `DESIGN.md:52–66,103–108,340–351,1176–1180`; references `refcore.py:64–87,112–114`; combined `t2core.py:36–37`, `fom2.py:44–50`.  
   **Fix:** Retain “ECSW with support-targeted greedy NNLS” and distinguish the Q-DEIM approximation from exact preservation of its residual norm.

6. **NEEDS-RESTATEMENT — fairness is conditional on two outstanding ratifications.**  
   Central, operator-matched training snapshots are defensible and avoid imposing the upwind generator’s smoothing on the POD baseline. Nevertheless, this departs from the brief’s explicit generator-reuse requirement. Likewise, expenditure within a factor-1.25 band is not literally equal expenditure. The design acknowledges both deviations and requires coordinator ratification; A18 supplies neither.

   The revised T/V accounting appropriately separates training-only construction from validation-informed work, including the development evidence that changed the ECSW knob.

   **Evidence:** `DESIGN.md:544–554,675–685,788–791,1195–1197`; lane brief `jcp-hr-baselines.md:27–31`.  
   **Fix:** Record both ratifications in the release amendment. Otherwise use the registered generator fallback and unequal-budget descriptive wording, without claiming the equal-budget H-base contract was met.

7. **CORRECT — the amended cohort selection is defensible as an explicitly developmental comparison.**  
   The premise “select only on dev6/cases 0–5, judge on the larger cohort” is superseded: those subsets screen Stage A; Stage B and final picks use the full developmental cohort. This is more symmetric with ours, although screening and historical selection remain unequal. It does not provide independent generalization evidence.

   **Evidence:** `DESIGN.md:385–394,572–576,854–857`; references `cohorts.py:35–38`; combined `report.md:443–445,570`.  
   **Fix:** None to the amended selection rule. Retain the unequal-history, in-sample qualification beside every comparison.

8. **CORRECT — the governing Q1 predicate matches COMMON-BRIEF §3/V4-2.**  
   A2.7 removes the residual-tolerance exception. Certification uses a fresh Jacobian at every step in-job, pooled non-stationarity, every-case lattice distance, and tight/tighter agreement. The normalization uses the canonical physical IC, not its POD projection. The later small-gradient anchor allowance is disclosed separately and does not relax production Q1.

   **Evidence:** `DESIGN.md:158–170,396–405,578–582,1087–1091`; COMMON-BRIEF.md:49–60; PLAN-v4-amendments.md:14–25.  
   **Fix:** None; continue reporting nominal versus allowance anchor steps separately.

9. **CORRECT — H-base’s amended labels are defined, but remain exploratory and conditional.**  
   Dominance requires resolved improvements in time and worst error; crossing objectives produce “neither dominates.” “Matches” requires the registered paired equivalence procedure plus worst-envelope inclusion. Missing eligibility, failed picks, empty fronts, and timing failures cannot support a favorable conclusion. Worst-error envelopes are explicitly conditional reference-error envelopes, not confidence intervals for population maxima.

   Multiplicity and selection on the judged cases remain real limitations. The design addresses them by withdrawing formal falsification/confirmation wording, not by providing multiplicity correction. The reference-limited identical-arm “unresolved” outcome is intentional; the relative-margin regime can return “matches.”

   **Evidence:** `DESIGN.md:441–464,617–619,644–655,782–785,915–921`; lane brief:37–46; PLAN-v4-amendments.md:74.  
   **Fix:** Preserve “developmental, in-sample, unadjusted.” Do not promote these labels into a campaign-wide confirmatory H-base result without a separately frozen analysis.

10. **NEEDS-RESTATEMENT — required controls are specified, but successful execution is still incomplete.**  
    Independent residual/Jacobian checks, full-rank/all-points FOM recovery, and under-sampling controls are present. Their blocking consequences are appropriately explicit. However, smk1 contains failed phases, and A15.6 acknowledges incomplete G-fd gates. A18’s training-only fixture cannot establish that every required control fires after the NNLS changes.

    **Evidence:** `DESIGN.md:424–438,600–602,875–877,1151–1158`; `runs/smk1/archive/output/ACCEPTANCE.json:23–37`.  
    **Fix:** Obtain the amended construction’s complete real-size smoke evidence before declaring Phase I complete. Preserve unavailable/failed controls as unfulfilled gates.

11. **CORRECT — the amended provenance claims checked against the supplied sources agree.**  
    The 2D training configuration records 576 trajectories, seed 0, and \(\Delta t=0.005\); its state selection differs from the lane’s 12-step choice, as A1.3 admits. Both supplied 3D bank configurations record seed 923701, 1,536 draws, and the stated 12 steps. The deployed time steps and startup conventions agree with combined. The IC functions and E0 scoring lattices also agree; 2D node-versus-interval terminology is corrected in A3.12.

    **Evidence:** `sep_burgers_r3_N256_K16_R512.json:17–36`; `sep_burgers_r3.py:321–332`; `burgers2d_film.py:160–177`; both bank `training.json:3–26`; `engines.py:33–38`; vendor `common.py:91–106`; combined `c2common.py:24`, `c3common.py:35`, `t2core.py:36–37`; references `refstore.py:10–15,132–154`.  
    **Fix:** None to the amended claims; retain their provenance distinctions.

**Overall verdict: WRONG as written**, because A18 claims a completed evidence archive that is demonstrably incomplete. The numerical framework is not thereby disproved.

**Top blockers:** missing reproducibility evidence; unfrozen production QR policy and resource certificate; incomplete real-size controls; outstanding generator/budget ratifications. Audit 12 item 3 is resolved, item 2 is not, and items 5–6 are only partially resolved or explicitly deferred.