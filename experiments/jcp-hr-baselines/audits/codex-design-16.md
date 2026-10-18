**Overall: WRONG, narrowly because A22’s replacement cost calculation is incorrect.** A22 resolves round 15’s quadrature-attribution issue and supplies the missing stability policy. A23 is a legitimate, disclosed post-smoke construction-rule amendment; it does not invalidate the numerical controls.

Below, `D` means `experiments/jcp-hr-baselines/DESIGN.md`; `smk2/…` means files under that experiment’s `runs/smk2/`.

1. **CORRECT — A22 resolves round 15 item 2.**

   A22 explicitly retracts the transferred quadrature evidence, makes the five deployed 3D configurations diagnostic only, and requires exact-configuration certificates covering both own-state Q2 bars before admission. The full-field query mapping is consistent with `fs3.make_fs_query`: projection, evolution, then decoding.

   **Evidence:** `D:1337–1342`; combined `fs3.py:171–183`; COMMON-BRIEF.md:57–60.

   **Fix:** None. Until those certificates arrive, these five configurations cannot yield substantive H-base labels.

2. **WRONG — A22’s replacement cost arithmetic is still wrong, although the missing operations are now acknowledged.**

   Substituting its own dimensions, \(m=102\times1024\), \(K=s=16384\), gives:

   \[
   8mKs=2.2430\times10^{14}\text{ flops},
   \]

   **not \(2.2\times10^{17}\)**. Its memory-traffic estimate is consistent:

   \[
   32mKs=8.9720\times10^{14}\text{ bytes}.
   \]

   The displayed refactorisation sum also evaluates to

   \[
   \sum_{j=1}^{16}2m(1024j)^2
   =3.2769\times10^{14}\text{ flops},
   \]

   not \(4.6\times10^{14}\). A different count including explicit-Q formation or other operations needs its own formula. Periodic factorisations still contribute cubic growth in attained support; A20’s “no longer cubic” statement needs explicit withdrawal.

   **Evidence:** `D:1297–1300,1344–1354`.

   **Fix:** Correct the arithmetic, distinguish factorisation, explicit-Q formation and drift checks, and express the iteration allowance unambiguously as a count—not “1.2 s per path.” Regenerate the resource certificate from that model, including deletions, retries and compilations. Round 15 item 4 is **not fully resolved**.

3. **CORRECT — A22 supplies the missing production-interval stability policy.**

   The registered fixture now crosses two production refactorisations with deletions, compares checkpoint supports and residuals, and specifies a drift threshold plus consequences. Orthogonality events prevent “adequately tuned ECSW” and exhaustive non-domination claims.

   **Evidence:** `D:1362–1367`; inherited consequences at `D:1199–1207`.

   **Fix:** None to the design-level response to round 15 item 5. Passing this new gate remains a production prerequisite; the full-QR smk2 cannot itself establish updating-QR stability.

4. **CORRECT — A23 is a legitimate post-smoke rule change, with the original failure honestly retained.**

   The archived path reached supports 256, 512 and 1024, then terminated at support **1340**, before the 2048 target. Its recorded residual is \(4.6128\times10^{-7}\); both recorded KKT gradient diagnostics are well below the registered tolerance.

   Comparing the two acceptance JSONs, only `2d_L1024|ecsw-smoke-rank.ok` and the aggregate `ALL` result change. The absent `k16` checkpoint remains explicitly `null`. There is no evidence here of changing an E0 accuracy threshold, selecting a favourable verdict outcome, or weakening N-under/Q1.

   **Evidence:** `D:1374–1386`; `smk2/archive/output/smoke.json:1095–1147`; `smk2/ACCEPTANCE-staged.json:176–197,967–971`; `smk2/ACCEPTANCE-A23.json:176–197,967–970`.

   **Fix:** Preserve both acceptance records and the post-smoke label. Treat future targets mapping to the same terminal solution as aliases of one actual construction, rather than independent coverage evidence.

5. **NEEDS-RESTATEMENT — A23 overstates numerical optimality and algorithmic equivalence.**

   “Solved exactly” is stronger than satisfying finite-tolerance KKT inequalities. Likewise, a finite regression gate establishes agreement on its fixtures, not universal equality of full-QR and updating-QR paths. Nearly tied entry gradients can change support sequences without materially changing the objective.

   Also, NNLS optimality concerns the **training gradient-matching objective**. It does not guarantee adequate online sampling, a strong baseline, or Q1 qualification. A23 is defensible because those separate gates remain operative.

   **Evidence:** `D:850–852,1176–1180,1379–1386`; raw numerical diagnostics at `smk2/archive/output/smoke.json:1100–1108`.

   **Fix:** Say “satisfies the registered numerical KKT certificate” and “agrees on the binding regression fixtures.” Explicitly retain Q1, actual-support disclosure and construction-quality restrictions for every optimum alias.

6. **CORRECT — the governing baseline mathematics is sound.**

   The amended Jacobian differentiates both factors of central advection. ECSW matches the correct LSPG gradient contributions \(J_i^\top R_i\). Q-DEIM approximates only the nonlinear term; the untruncated QR of \([\Phi,\Delta_h\Phi,U]\) preserves the approximated residual norm without discarding linear components. The original truncated-SVD prescription is superseded.

   Central-FOM snapshots are a reasonable operator-matched baseline choice, with the generator deviation explicitly ratified. The design correctly withdraws both the claim that spectral sampling is impossible and the claim that the FOM error is a strict ROM floor.

   **Evidence:** `D:338–352,511–521,529–534,1258–1261`; references `refcore.py:64–87`; combined `fom2.py:34–50`.

   **Fix:** None. Describe this specifically as support-targeted ECSW, not an exhaustive demonstration of the strongest possible classical ROM.

7. **CORRECT — the final cohort and fairness contract is defensible as developmental evidence.**

   The governing design no longer selects only on dev6/first-six and then presents the rest as independent validation. Stage B and picks use the full verdict cohort, and the withdrawn “out-of-selection” view is explicitly identified. Unequal selection histories remain disclosed.

   Validation-informed construction decisions, certification and smoke-driven changes count toward the baseline budget. The coordinator ratifies the factor-1.25 wording, without calling it equal expenditure. Time-stepping restrictions also restrict comparator admission.

   **Evidence:** `D:385–394,572–576,675–685,1195–1197,1258–1263,1306–1312`; combined `report.md:3`.

   **Fix:** None. Retain “developmental, in-sample, unequal selection histories” and actual budget ratios beside conclusions.

8. **CORRECT — Q1 and the final decision table respect the binding contract.**

   The residual-tolerance exception is withdrawn. Certification uses fresh per-step Jacobians, pooled nonstationarity, every-case anchor distance and the tighter-anchor subset check. Anchor stall allowances are separately disclosed.

   Dominance requires resolved time and worst-error superiority; crossing objectives produce “neither dominates”; matching requires the registered equivalence tests and worst-envelope condition. Missing inputs, failed picks and incomplete timing coverage cannot become support for ours. Worst-error intervals are correctly described as conditional reference-error envelopes, not population confidence intervals.

   Multiplicity is **disclosed, not corrected**. That is acceptable for the explicitly exploratory labels, not a confirmatory campaign-wide H-base claim.

   **Evidence:** `D:396–405,441–463,578–582,617–619,644–655,709–713,915–921,1087–1091`; COMMON-BRIEF §3; V4-5.

   **Fix:** None to the governing rules. Keep the developmental and unadjusted qualifications.

9. **CORRECT — the supplied smoke evidence shows genuine accuracy-degradation controls, not merely solver failures.**

   Accepted Q1-qualified examples include:

   | Cell | `none` median error | ECSW under-sampled error | Q-DEIM under-sampled error |
   |---|---:|---:|---:|
   | 2D | 0.03115 | 0.39043 | 0.44494 |
   | 3D | 0.05040 | 1.23214 | 0.27655 |

   Each exceeds twice its corresponding `none` error. Independent residual/Jacobian checks are present; the CPU gate record reports full-rank recovery across 84 runs, worst distance \(2.8\times10^{-13}\). A23 does not change these gates.

   **Evidence:** `smk2/ACCEPTANCE-A23.json:355–410,831–880`; `checks/test_core.json`, `G-full` and `G-fd` entries; control-blocking rule `D:600–602`.

   **Fix:** None. These establish the registered fixtures, not universal sampling sensitivity.

10. **CORRECT — provenance is consistent after amendments; feasibility remains conditional.**

    The checked sources support the 2D training draw/count, 3D seed 923701/count 1536 and twelve saved steps, node-versus-interval distinction, IC definitions, and differing CNR conventions. The governing design explicitly distinguishes reused parameter draws from regenerated trajectories.

    The original 13 GPU-hour estimate is withdrawn. Memory, disk retention and compile inventories require manifest-derived certificates; smk2’s maximal 2D Q-DEIM phase reports approximately **107.18 GB** device peak. That is useful evidence, but does not certify every production branch or the updating-QR workload.

    **Evidence:** `D:354–365,627–631,715–730,751–754,1093–1097,1356–1360`; 2D training JSON:19,33,36; `burgers2d_film.py:160–177`; `common.py:91–106`; both 3D banks’ `training.json` configuration blocks; `smk2/ACCEPTANCE-A23.json:165–173`.

    **Fix:** Complete the corrected final resource certificate before production; retain incomplete-scope labels if the cap cannot accommodate the registered workload.

**Overall verdict: WRONG.** The concrete blocker is A22’s incorrect replacement cost calculation and its resulting unresolved resource-accounting claim. A23 additionally needs narrower numerical wording, but its post-smoke acceptance change is legitimate. Exact-configuration 3D Q2 certificates and the production updating-QR/resource gates remain prerequisites, not demonstrated successes.