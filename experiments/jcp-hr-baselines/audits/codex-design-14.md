**Overall: NEEDS-RESTATEMENT.** Round 13’s missing-archive objection is resolved. I found no new error in the governing baseline formulas, but A19’s comparator expansion is not fully reconciled with the registered options, timing contracts, selection procedure, and budget accounting.

Below, `D` means [experiments/jcp-hr-baselines/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md). Other lane paths refer to the worktrees named in the request.

1. **CORRECT — round-13 item 1’s archival-completeness defect is resolved.**

   Commit `6bd9b807e` contains `t4.log`, `t7.log`, `G.sha256`, and `mk.log`. The historical driver now retrieves `nnls.py` from `c57098e00`; that pin was committed in `9686f21be` and is present in `6bd9b807e`. The repaired log records 147 iterations, no bans, and support checkpoints 32, 64, and 128.

   **Evidence:** `D:1248–1252`; `runs/nnls_fixture/t_old.py:3–9`; `t7.log:1–5`; commit tree of `6bd9b807e`.

   **Fix:** None for the missing-files finding. There is a narrower evidentiary limitation: `t4.log:1–8` records periodic support counts and residuals, not the reference’s weights/support indices at the three repaired checkpoints. Consequently, the archive corroborates progress but does not independently demonstrate the stronger “same trajectory” assertion in `D:1236–1238`. Archive explicit checkpoint comparisons or narrow that wording. I did not regenerate the fixture.

2. **CORRECT — the governing residual, Jacobian, ECSW target, and Q-DEIM reduction are mathematically consistent.**

   The amended Jacobian differentiates both factors of \(uD_hu\), includes the diffusion term, and includes \(a_0I\). ECSW matches nodewise contributions \(J_i^\top R_i\); these sum to the gradient of the unweighted least-squares objective and become the weighted objective’s gradient under positive weights.

   The original truncated-SVD reduction is superseded: untruncated Householder QR gives \(\|Bz\|=\|R_Bz\|\) algebraically, preserving the **DEIM-approximated** residual objective without discarding linear terms. The cancellation-scaled checks appropriately avoid dividing by a nearly zero residual.

   **Evidence:** `D:338–352,529–534`; references `refcore.py:64–87,112–114`; combined `fom2.py:44–50`.

   **Fix:** None. Continue distinguishing exact norm reduction from the nonlinear DEIM approximation.

3. **CORRECT — the baseline is a defensible registered construction, with appropriately limited strength claims.**

   Uncentred POD, operator-matched snapshots, native damped Gauss–Newton, and the complete common-solver ablation are reasonable. Support-targeted greedy NNLS is a disclosed ECSW variant; finite candidates, limited training trajectories, support caps, and construction failures prevent a claim of exhaustive or automatically optimal tuning.

   A19 ratifies the generator deviation. Its “best data” rationale should be understood as motivation, not a proven ordering of resulting ROM accuracy: operator-matched snapshots need not always outperform upwind snapshots.

   **Evidence:** `D:354–365,536–554,633–635,883–897,1176–1207,1259–1261`.

   **Fix:** Retain the generator probe and all adequacy, coverage, and banning qualifications. Prefer “intended to avoid handicapping the baseline” to an unconditional “best data” claim.

4. **NEEDS-RESTATEMENT — A19’s new 2D comparator source exceeds the baseline’s registered time-stepping options.**

   The baseline registers BE and CN-R only at \(0.005\). The qualify lane’s prospective front includes BE/CN-R at \(0.01\) and \(0.0125\), and GAL-CN at additional steps including \(0.025\). A19 admits front members without specifying how these additional options receive a comparable baseline opportunity.

   This is a concrete fairness gap if those members enter comparisons. Adding GAL-CN as an external comparator does not require changing POD-LSPG into Galerkin, but the applicable CN time-step choices must be addressed.

   **Evidence:** `D:367–370,556–558,1271–1276`; qualify `DESIGN.md:72–97,291–299`; lane brief `jcp-hr-baselines.md:27–30`.

   **Fix:** Either restrict admitted 2D comparators to the registered baseline option set, marking others “not assessed,” or register the additional baseline scheme/step options and regenerate construction, controls, and resource inventories before their comparison.

5. **NEEDS-RESTATEMENT — the 3D export contains different query contracts; A19 needs an explicit member-to-contract mapping.**

   Speed3d’s export distinguishes the full-field **grid** from mesh-free **m1**. Its large `q1` collection is not a blanket mesh-free certificate. The qualify lane likewise states that the grid uses the mesh path and that mesh-free evidence covers only four configurations.

   A19 authorizes mesh-free comparisons, but simply labelling all 3D pairs “ours under the mesh-free query contract” does not identify which exported configuration, initialization, trajectory, and certificate are being replayed.

   **Evidence:** `D:207–216,744–745,1273–1277`; speed3d `runs/EXPORT.json:4–9`; speed3d `DESIGN.md:345–362`; qualify `DESIGN.md:301–307`.

   **Fix:** Freeze each imported X’s exact path and certificate identity. Define both methods’ timed inputs, preparation exclusions, initialization, and outputs explicitly. Preserve speed3d’s separately reported input-preparation costs. Require Q1/Q2 for the actual imported path; do not transfer mesh-path qualification merely by changing its label.

6. **NEEDS-RESTATEMENT — “front members added later” needs a deterministic selection and accounting procedure.**

   Stage B includes an X-specific front derived from same-job timing of X and every screened baseline. A newly arriving X can therefore require additional Stage-B members, anchors, timing, and revised picks. Appending X to an already frozen comparison table is insufficient.

   A3.1’s general rule correctly charges all validation-informed work, but its concrete source mapping predates qualification-front work and mesh-free selection/qualification.

   **Evidence:** `D:675–685,820–822,854–857,1271–1276`; qualify `DESIGN.md:291–299`.

   **Fix:** Specify versioned comparator batches. For each new X, recompute its screening front using same-job timings, run newly required Stage-B candidates, freeze its picks, and certify the added cost. Update ours’ V ledger with the qualification and mesh-free work that selected or certified imported configurations, avoiding double counting inherited evidence.

7. **CORRECT — budget ratification and developmental cohort selection are now explicit.**

   A19 resolves the two outstanding ratifications: central snapshots are authorized, and the expenditure band must be called “within the registered factor-1.25 band,” never “equal budget.”

   The premise of selecting solely on dev6/cases 0–5 is superseded. Those cases screen Stage A; Stage B and final picks use the full developmental cohort. This is defensible as an in-sample comparison with unequal selection histories, not independent generalization evidence.

   **Evidence:** `D:385–394,572–576,788–791,1195–1197,1259–1263`; combined `report.md:443–451,570`; references `cohorts.py:35–38`.

   **Fix:** None beyond incorporating the new comparator work identified in item 6. Preserve the in-sample and unequal-history labels.

8. **CORRECT — the governing Q1 predicate matches COMMON-BRIEF §3/V4-2.**

   The final rules require validity, fresh per-step Jacobians in-job, pooled non-stationarity at \(10^{-3}\), every-case distance at \(2.5\times10^{-4}\), and prescribed-subset anchor agreement at \(2.5\times10^{-5}\). A2.7 removes the residual-tolerance exception. Normalization uses the physical IC, not its POD projection.

   The anchor backtracking allowance is explicitly limited and disclosed; it does not relax the production Q1 thresholds.

   **Evidence:** `D:158–170,396–405,578–582,980–1003,1087–1091`; COMMON-BRIEF `49–60`; PLAN-v4 V4-2.

   **Fix:** None. Keep nominal and allowance anchor steps separately reported.

9. **CORRECT — H-base’s amended decision table is defined and appropriately exploratory.**

   Dominance requires resolved improvements in time and worst error. Crossing objectives give “neither dominates”; matching requires the registered paired equivalence procedure and worst-envelope inclusion. Empty fronts, failed picks, missing eligibility, zero margins, and incomplete timing coverage cannot become favorable evidence.

   The worst-error intervals are conditional reference-error envelopes, not population confidence intervals. Multiplicity and selection bias remain; A2.17 handles this by withdrawing formal falsification/confirmation language, not by correcting multiplicity. The identical-arm reference-limited “unresolved” outcome is intentional.

   **Evidence:** `D:441–464,617–619,644–655,705–713,782–785,915–921`; lane brief `37–46`; PLAN-v4 V4-5.

   **Fix:** None for the developmental labels. Do not promote them into a confirmatory campaign-wide H-base verdict.

10. **CORRECT — required controls are specified; successful execution remains a production prerequisite.**

    Independent residual/Jacobian checks, full-rank/all-points FOM recovery, and under-sampling degradation controls are present. Under-sampling must degrade a Q1-qualified run; mere solver failure does not count. Unavailable required fixtures leave acceptance unfulfilled.

    The archived first smoke is not a complete demonstration: both 2D under-sampling controls are unfulfilled, both complete G-fd gates fail, and the 3D N-grad fixture is unavailable. A19 correctly requires successful smk2 acceptance.

    **Evidence:** `D:424–438,600–610,875–877,899–906,1151–1158,1267–1269`; smk1 `ACCEPTANCE.json:266,329,353,667,798,840`.

    **Fix:** Obtain and audit complete smk2 evidence before production. Do not treat the fixture archive or passing individual subchecks as substitutes.

11. **CORRECT — A19 makes feasibility an explicit gate; feasibility itself is still unestablished.**

    The old approximately 13 GPU-hour estimate is withdrawn. Relevant allocations include approximately 57.9 GB of 2D snapshots, 36.9 GB of 3D snapshots, a 27.4 GB maximal ECSW contribution matrix, and a 4.29 GB maximal 2D `none` Jacobian. At 3D support 16,384, a single passive-column or thin-Q array is approximately 13.7 GB before other arrays and workspaces.

    A support-1,024 pilot does not certify the full continuation. Updating QR, fallback workspaces, retry, compilation counts, new comparator batches, and retained artifacts must enter the final certificate.

    **Evidence:** `D:627–631,719–730,865,1025–1027,1093–1097,1242–1246,1264–1269`; speed3d `resources.py:1–12`; COMMON-BRIEF `87–92`.

    **Fix:** Before production, freeze the updating-QR policy and checkpoint comparison tolerances, then supply measured/estimated host, device, disk, compilation, and GPU-hour totals for the final manifest. Include smk2 expenditure. A19 correctly blocks production until this is done.

12. **CORRECT — the checked provenance and discretization claims agree with their sources.**

    The 2D bank records 576 trajectories, seed 0, and step \(0.005\); its early-weighted state selection differs from the lane’s 12-step choice, as amended. Both 3D bank configurations record seed 923701, 1,536 draws, and the stated snapshot steps. The dimensional time steps, startup coefficients, IC functions, and scoring lattices agree. The amended node-versus-interval distinction is necessary and correct.

    **Evidence:** 2D training JSON `17–36`; `sep_burgers_r3.py:321–332`; `burgers2d_film.py:160–177`; both 3D `training.json:3–26`; `engines.py:33–38`; vendor `common.py:91–106`; combined `t2core.py:36–37`, `c2common.py:24`, `c3common.py:35`; references `refstore.py:10–15,132–141`.

    **Fix:** None. Retain these provenance distinctions in the report.

**Overall verdict: NEEDS-RESTATEMENT.** The top design blockers are A19’s expanded 2D option coverage, exact 3D query/certificate mapping, and the procedure for late comparator additions. Successful smk2 controls and the final updating-QR/resource certificate remain explicit production gates. No files were changed and no jobs were submitted.