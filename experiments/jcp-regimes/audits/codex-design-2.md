**Overall verdict: NEEDS-RESTATEMENT.** A1 resolves most of audit 1’s conceptual errors, including the A−/C1 contrast, bank/head provenance, reference acceptance predicates and unsupported speed-up claim. Remaining blockers concern duplicate checks, calibration claims, missing statistical rules, pilot selection and resource reservation.

Read-only audit completed. No files changed, jobs submitted, sealed/OOD cohorts accessed, or scratch files created. Small NumPy checks confirmed the registered A−/C1 pairing and N0s bounds.

Evidence abbreviations:

- **D**: [regimes DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/DESIGN.md)
- **RG / T**: [rcohorts.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/rcohorts.py) / [test_regimes.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/test_regimes.py)
- **E0**: `worktrees/2026-10-06-jcp-references/experiments/jcp-references/`
- **Q**: `worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/`
- **Briefs**: `reports/2026-10-17-full-experiment-plan/campaign/briefs/`

1. **Regime distributions and attribution — CORRECT under A1’s revised interpretation.**

   The amplitude intervals remain feasible:

   | Regime | At minimum ν | At maximum ν |
   |---|---:|---:|
   | A+ | [0.5, 2] | [1.99054, 2] |
   | A− | [0.5, 0.63246] | [0.5, 1.41589] |
   | C1 | [0.79057, 1.58114] | [1.76986, 2] |
   | C2 | [0.79245, 1.58489] | [1.77407, 2] |

   All are nonempty and inside `[0.5,2]`, including intermediate viscosities. **D:44–55.**

   A1 correctly retracts pure one-factor attribution for A± versus N0 and identifies both C regimes as low-ν/high-s. The revised C1 construction copies A−’s geometry and viscosity exactly, changing only amplitude. I checked all 16 registered pairs: those four coordinates are identical, and every C1 amplitude is larger. **D:308–320; RG:78–83.**

   Thus this is a valid **paired amplitude intervention**, with steepness, transport and front-formation consequences. It does not identify a pure stiffness mechanism. The revised interpretation of \(a/\nu\) is defensible. Passing σ separation checks cannot restore the original “clean viscosity effect” language.

2. **Training distribution and provenance — CORRECT.**

   A1 distinguishes the canonical seed-0 bank draw from the additional seed-1000 head-refit trajectories. The ranges agree with the actual samplers. **D:322–329; `experiments/mr-burgers2d/engines.py:20–25`; `burgers2d_film.py:160–168`; `sep_burgers_r3.py:203,293–304`; `cluster/run_dn256b.sbatch:41–47`.**

   The pinned checkpoint, rotation and mode-list identities are appropriate. Checking both descriptor draws remains conservative even though these linear-span configurations do not use the head.

3. **Seeds, duplicate checks and sealed/OOD protection — WRONG in the draft implementation; corrected design mostly sound.**

   A1’s revised SeedSequence claim is correct: domain separation with negligible collision risk, not mathematical stream disjointness. High public seed integers avoid equality with E0’s 31-bit secret seeds without reading custody data. Descriptor-only historical test64 checking is now explicit. **D:331–338; E0/cohorts.py:35–48,67–88.**

   But the promised extended duplicate protection is absent:

   - Tests check **N0 only** against training/dev/val/test descriptors. **T:57–67.**
   - No within-regime or cross-regime duplicate checks are performed.
   - The helper explicitly ignores non-five-component rows, so it cannot enforce the promised B1/B2 checks. **RG:243–248.**
   - N0s is not included in those tests.

   Also, RG’s header still says “disjoint by construction,” contradicting A1. **RG:4–7.**

   N0s stays inside the training width support and does not open or regenerate OOD. However, “no OOD case is … approximated” is too strong: the width supports meet at 0.05, so there is no positive separation guaranteeing dissimilar fields. Its ν endpoint, 0.0126, is also a rounded approximation to E0’s actual \(0.01\,10^{0.1}\). **D:365–367; E0/cohorts.py:48,69–77.** Say “does not sample the reserved OOD distribution.”

4. **References, calibration and σ — NEEDS-RESTATEMENT.**

   **Resolved:** A1 now matches the actual E0 acceptance predicates, including the residual-derived floor and direction consistency. The reference levels and escalation remain appropriate as a conditional procedure. Low ν alone does not imply a Péclet violation: nominal C2 \(Pe_{2048}\le1000/2048\approx0.488\). The actual running numerical maximum must still pass. **D:340–346; E0/verdict.py:84–133; E0/refjob.py:137–152.**

   The family-aware nodal and spectral adapters address the earlier omissions. **D:353–357; E0/refcore.py:239–245; E0/refjob.py:422–423.**

   **Remaining calibration issue:** production–tight agreement does not establish that “the proxy bounds the actual solver error.” Both solves can share an error or fail to converge sufficiently. A1 does not explicitly require probe validity or a tight/tighter anchor; E0’s calibration checks both. **D:347–352; E0/refjob.py:346–356.** A limited coarsest-level agreement probe is acceptable, but its conclusion must remain limited even when it passes.

   **σ:** its grid, normalization, output times and final-triple selection are now well defined. **D:359–369.** Two edge cases remain:

   - Failed/non-finite solves cannot necessarily yield a finite σ “for every case”; specify an unavailable value and reason.
   - The pooled N0∪N0s coverage rule can pass with eight N0 cases and **zero eligible N0s cases**. That would defeat the stated stiff-corner coverage purpose. Specify separate N0s coverage or explicitly label the stiff-corner comparison unavailable.

   N0s itself is implemented correctly: eight rows with the stated ranges. The registered sample has \(s\approx130.03\)–168.69; it does not reach the support endpoint 200. This is not a generator error, but the control should not be described as covering the entire stiff corner.

5. **Frozen configurations and qacc adapters — CORRECT as a design specification.**

   I independently verified the FRONT-2D file hash:

   `a51ac1a250f00d4b0006a84f2434ab3c96407b1f575f4ffff7a3eb25b7a8b77d`

   The eight factor-1 LSPG configurations and R384 GAL diagnostic are in the qualified 1024 panel. Main/successor/check rules agree with the qualification configuration. **D:153–165; Q/runs/FRONT-2D.json:23599,27630,28286,28963,33655,34309,36309,37636,39636; Q/configs/acc1024.json:14–18,72–76,130–134,188–192.**

   A1 now explicitly covers selection, tighter/check subsets, new-family ICs, normalization, cohort hashes and reference routing. Keeping the existing bilinear cold-quadrature sampling contract is sound. **D:372–383.**

   These are specified adapters, not yet verified completed implementation. The eventual code audit must include the scorer’s expected-population construction, not just the driver.

6. **Q1/Q2 per regime — CORRECT as amended.**

   A1 resolves the GAL exception, auxiliary-run validity, independent recomputation and all-sampled-case scope. **D:385–390.** The thresholds match **Briefs/COMMON-BRIEF.md:49–60** and **Q/qscore.py:29,35–38,175–195**.

   Ensure aggregation actually occurs separately for each regime. A single pooled certificate across all regimes would allow one regime’s failures to be diluted by other regimes’ steps.

7. **Statistical decisions — NEEDS-RESTATEMENT.**

   A1 resolves the exposed-mixture baseline, overlapping N0 comparison, exploratory multiplicity scope, eligible-case qualification and descriptive maxima. The factor-2 margin is a legitimate preregistered practical threshold. **D:392–407.**

   New or remaining gaps:

   - **Paired coverage:** eight eligible cases in each regime do not imply eight eligible matched pairs; the intersection can be empty. Require a minimum number of complete eligible-valid pairs.
   - **Paired zeros:** the bootstrap-denominator rule does not define treatment of \(e_{A-}=0\), \(e_{C1}=0\), \(0/0\), infinite ratios or undefined \(\log q\).
   - **Wilcoxon:** freeze zero-difference handling, alternative and exact/permutation/asymptotic method, including the all-zero case.
   - **Baseline coverage:** the coverage threshold does not explicitly apply to the eligible-valid denominator population.
   - **N0s status conflicts:** A1.5 calls N0s descriptive, while A1.8 counts it among 56 primary labels. Choose one interpretation.

   **Evidence:** D:369–370,398–409. These omissions can change labels without changing any measured errors.

8. **Controls — NEEDS-RESTATEMENT.**

   A1 meaningfully improves the controls. Upwind references, coarse quadrature and zero-budget solves are real numerical interventions capable of failing; synthetic order/Péclet fixtures are correctly distinguished from them. The new IC consistency checks are appropriate. **D:411–431.**

   However, the pilot **case identities remain unfrozen**. A1 names “the C2 pilot,” C2/B2 cases “two each,” and an A+ calibration pilot without specifying indices or a deterministic selection rule. **D:347–351,413–427.** This leaves post-selection freedom affecting controls, calibration and budget estimates.

   Clarify whether R-upwind must fail specifically on order. Rejection through an unrelated solver failure would not demonstrate discrimination between first- and second-order convergence.

   The existing draft tests also lack an explicit A−/C1 pairing gate and N0s-specific negative bounds/IC gates. **T:39–86.**

9. **Budget and reservation — NEEDS-RESTATEMENT.**

   A1 acknowledges the costly tight path, but the revised total is not yet a conservative certificate. **D:433–438.**

   E0’s A100 ordinary ST cases took 448, 392, 341 and 325 seconds. **E0/runs/fd2ddev6p0/logs/5217591.out:30,38,46,54.** At 325 seconds, the amended 120-case panel alone costs about **10.8 GPU-h**, before refinement, tight attempts, controls or compilation. A roughly four-hour ordinary-reference estimate therefore cannot apply indiscriminately to both permitted GPU types.

   Further gaps:

   - The 3.5-hour observation concerns **one 8192 tight level**, not an entire tight triple. **E0/DESIGN.md:896–901.**
   - Tight refinement at 12288 is possible under the unchanged procedure. **E0/refjob.py:145–152.**
   - Two reserved tight cases do not bound the number of floor failures.
   - N0s adds eight references and nine-arm qualification work without an explicit revised count.
   - Solver retries, gradient-memory overhead and all calibration/control runs are not separately accounted for.
   - “×1.5 for the time limit” does not clearly say whether that amount is reserved against the cap.

   Use device-specific projections and an explicit reservation/reprojection rule before expensive escalation. A forecast based on historical medians is not a cap guarantee.

10. **Brief completeness, FOM comparison and 3D — NEEDS-RESTATEMENT, with the major false claim resolved.**

   The unsupported cross-job speed-up upper bound is explicitly withdrawn. Accuracy-only FOM comparison is defensible. **D:440–443.**

   But “coarsest member” is ambiguous on a two-dimensional \(N\times\Delta t\) ladder: smaller N and smaller Δt can trade off. Freeze a total ordering or report all nondominated matching pairs. Also state what happens when no member matches.

   A1 correctly limits readiness to **2D** and requires a separate 3D amendment with the qualified configuration list and 65³ contract. **D:445–448.** The 3D sign/absolute-peak correction is sound; the existing draft `draw_table_3d` still lacks that rejection and must remain unusable for production. **RG:199–227.**

   Error-versus-distance reporting, regime qualification and defer order are retained correctly. **D:243–245,273–278; Briefs/jcp-regimes.md:18–29.**

**Required fixes**

1. Implement A1.3’s complete duplicate checks, including N0s and two-bump rows; remove the contradictory RNG-disjointness claim.
2. Limit calibration conclusions to observed solver agreement; require valid probes and define handling of unverified tight accuracy.
3. Define unavailable σ values and minimum stiff-corner coverage; replace the claim that N0s cannot approximate OOD fields.
4. Freeze complete-pair and baseline coverage, zero-error handling and Wilcoxon behavior; resolve N0s’s descriptive-versus-primary status.
5. Freeze pilot case IDs and control success predicates; add pairing and N0s-specific generator gates.
6. Replace the budget summary with device-specific, count-complete reservations covering tight triples, refinement, retries and controls, with a stop/reprojection rule.
7. Define FOM ladder ordering or a nondominated matching set, including no-match behavior. Keep 3D explicitly deferred until its separate amendment.