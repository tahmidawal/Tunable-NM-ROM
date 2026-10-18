**Overall: WRONG.** A21 resolves round 14 items 4 and 6, and correctly separates the 3D query paths. However, its claimed quadrature-evidence mapping is false: speed3d’s finalist evidence belongs to different configurations. A20 also understates the cost of its registered periodic refactorisations.

References below use `D` for `experiments/jcp-hr-baselines/DESIGN.md`, `P1` for the specified p1-scaling export, and `S3` for the specified speed3d export.

1. **CORRECT — round 14 item 4: 2D comparator admission is now explicit and consistent with the export.**

   A21 admits only BE/CN-R at Δt = 0.005 and marks other options “not assessed.” Its frozen list exactly matches the qualified configurations at the two baseline meshes: eight at 256² and eight at 1024². I independently verified the export’s SHA-256 prefix `8e3b120855c702f2`.

   **Evidence:** `D:1306–1312`; `P1:1659–1687`; lane brief `jcp-hr-baselines.md:27–30`.

   **Fix:** None. Preserve the restricted scope when describing the comparison.

2. **WRONG — round 14 item 5: the query-path correction is sound, but the quadrature-certificate mapping is not.**

   The five registered deployed-rule FS-BE configurations do have eligible Q1 entries. The export hash also matches. Mapping them to `contract.grid`, excluding mesh-free `m1`, and retaining full-field projection is appropriate.

   But A21 assigns them “the finalists’ converged-rule δ_quad.” The export’s `p1.conv` entries belong to **different configurations**:

   - `B_C_old` is **M2_512/gl24**, whereas Batch 1 imports **M2_512/gl32**.
   - `B_C_W768` is **W768/lat16384/QN/k4**, whereas Batch 1 imports **W768/lat32768/FS/K1**.
   - `B_C_W1024` is **W1024/lat32768**, whereas Batch 1 imports **W1024/lat65536**.
   - That collection supplies no corresponding deployed W256 or W512 entry.

   Moreover, a rollout δ_quad alone does not establish COMMON-BRIEF’s own-state Q2 contract, which requires both the ρ and rollout-distance bars. Changing the label to “Q1 + deployed-rule δ_quad” does not satisfy admission.

   **Evidence:** `D:1314–1324`; `S3:4–8,7535–7565,8804–8839`; `D:744–745`; `COMMON-BRIEF.md:57–60`.

   **Fix:** Keep these five configurations diagnostic and outside H-base comparisons until exact-configuration Q2 certificates are available. Freeze a mapping from each X to its rule, solver, initialization path, certificate, population and source hash. Do not transfer finalist evidence between configurations.

3. **CORRECT — round 14 item 6: versioned batches now address selection and accounting.**

   New X members require fresh same-job timings against every screened baseline, recomputation of X-specific fronts, qualification of additional Stage-B members, frozen picks before verdict timing, and additional V-budget charges on both sides. This closes the late-addition loophole.

   **Evidence:** `D:1326–1331`, together with `D:820–822,854–857`.

   **Fix:** None to the procedure. A batch still must pass the admission requirements identified in item 2.

4. **WRONG — A20’s complete-path complexity omits its periodic full QR work.**

   Incremental appends cost \(O(mK)\), but A20 also requires full Householder refactorisation every 512 appends. Even on a monotone path, those refactorisations cost

   \[
   O\!\left(m\sum_{j=1}^{K/512}(512j)^2\right)
   =O(mK^3/512).
   \]

   Thus “no longer cubic in the support” is false for the complete registered algorithm. Using the design’s maximal 3D dimensions, refactorisations alone are approximately \(6\times10^{14}\) flops under the usual leading Householder count—already above the stated roughly \(10^{14}\)-flop whole-path estimate. Deletions and retries add work.

   **Evidence:** `D:719–720,1283–1299`.

   **Fix:** Replace the complexity and cost claim with a complete model covering appends, deletions, triangular solves, gradient refreshes, periodic QR, orthogonality checks and retries. Include simultaneous factorisation buffers/workspaces in the memory certificate. Retain the pre-job conservative certificate and the 30 GPU-hour cap; first-job measurements cannot retroactively establish affordability.

5. **NEEDS-RESTATEMENT — A20’s equivalence gate does not establish the production update schedule.**

   The binding random-problem comparison refactorises every 16 appends; production uses 512. The cited 147-iteration training fixture also does not exercise a full production refactorisation interval. Recording drift before refactorisation provides evidence, but A20 does not state a production failure/recovery action when drift exceeds its accepted tolerance.

   **Evidence:** `D:1242–1246,1283–1296`.

   **Fix:** Register an equivalence/stability fixture using the production interval, spanning multiple refactorisations and deletions. Specify acceptance tolerances and a deterministic fail-closed or refactor-and-recheck action. Include rank-skip diagnostics so skipped candidates cannot silently masquerade as adequate ECSW coverage.

6. **CORRECT — the amendments introduce no new defect in the governing baseline formulas or statistical decision rules.**

   The amended Jacobian differentiates both advection factors; ECSW targets \(J_i^\top R_i\); untruncated QR preserves the norm of the DEIM-approximated residual while keeping linear terms exact. The original truncated-SVD wording is superseded.

   Q1’s final predicate removes the residual-tolerance exception. The decision table distinguishes dominance, crossing objectives, TOST-based matching, unresolved outcomes and invalid inputs. Its labels remain explicitly developmental, in-sample and unadjusted; they cannot support a confirmatory campaign-wide H-base conclusion.

   **Evidence:** `D:338–352,578–582,617–619,644–655`; references `refcore.py:64–87`; `COMMON-BRIEF.md:49–60`; lane brief `jcp-hr-baselines.md:37–46`.

   **Fix:** None to these governing rules. Retain the selection-history and budget qualifications.

7. **CORRECT — controls and feasibility remain production gates, not established successes.**

   Required independent residual/Jacobian checks, full-rank/all-points recovery and under-sampling controls remain specified. The design itself records incomplete smoke acceptance, including missing arms in both complete G-fd gates. A20–A21 do not establish that those controls now pass or that the final workload fits memory, disk and GPU-hour limits.

   **Evidence:** `D:1124–1158,1242–1246,1264–1269`; speed3d `resources.py:1–12`.

   **Fix:** Require complete smk2 acceptance and the corrected final resource certificate before production. Successful individual subchecks are insufficient.

**Overall verdict: WRONG.** Top blockers are the incorrect 3D quadrature-evidence assignment, incomplete NNLS cost accounting, and the production-schedule stability gate. Round 14 items **4 and 6 are resolved; item 5 is not**. No files were changed and no jobs were submitted.