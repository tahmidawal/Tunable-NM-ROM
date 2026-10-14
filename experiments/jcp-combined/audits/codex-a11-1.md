**Overall: WRONG — A11 is a defensible post-hoc experiment, but its implementation is not ready for acceptance.** The main blockers are an incomplete reproduction gate, insufficient archive binding, and a resource-certification path that rejects the new smoke.

I performed read-only inspection, regenerated quadrature arrays in memory, and exercised extracted gate/status code without running solvers or writing files.

1. **A11.1 numerical claim — CORRECT, with wording needing restatement.**

   The archived distances confirm that **case 5 is the only case above \(2.5\times10^{-5}\)** at every W1024 escalation pair, for all four arms.

   | Setting / pair | Fixed BE worst | Adaptive CN worst | A10 status |
   |---|---:|---:|---|
   | W1024, 48–56 | 1.716486937e-4 | 3.589263756e-4 | fail |
   | W1024, 56–64 | 1.114331046e-4 | 2.330305298e-4 | fail |
   | W1024, 64–80 | 2.694246576e-5 | 5.641906299e-5 | fail |
   | W768, 64–80 | 4.568580674e-6 | 9.770076970e-6 | pass |

   Case 5 is the worst throughout. However:
   - “Every other case is orders of magnitude below” overstates the first pair: the next-largest fixed-BE distance is \(1.63694\times10^{-6}\), only **15.3× below the bar**.
   - W768 passes **A10 and the adaptive axis** at 64–80; its pre-registered selection axis remains **unresolved** because fixed CN is ineligible.
   - The data identify the case driving quadrature disagreement and show decreasing successive-rule differences. They do not establish a complete causal explanation of solver behaviour or eliminate A10’s separate eligibility failure.

2. **Rule generation, hashes, loading and staging — CORRECT.**

   Both new files match the vendored generator **array-for-array**; file and array hashes match `rules_ext2.json`. The regenerated gl64 also matches. Coordinates and weights are float64, weights are positive, and sums agree with one. Point counts are 884,736 and 1,404,928.

   `RULE_FILES` pins both files, and `FILES3` stages both. No rule-generation discrepancy found.

3. **Conditional final rule and inventory — CORRECT; A11 wording NEEDS-RESTATEMENT.**

   [c3run.py:310](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:310) skips gl112 exactly when 80–96 **passes** A10. It runs gl112 after either **fail or unresolved**. The two-rule smoke always runs both rules. Inventory includes all four arms and correctly marks the production last rule conditional, with no timing panels.

   Replace “only if … fails” with **“unless … passes”** to remove ambiguity between the explicit `fail` and `unresolved` statuses.

4. **Driver eligibility and axis semantics — CORRECT.**

   `elig_x` matches `elig_of`: fixed arms require valid runs, valid adaptive counterparts, and finite fixed–adaptive distance ≤ \(2.5\times10^{-4}\); adaptive arms require all runs valid. Pair statuses preserve unresolved precedence and the convergence bar.

   I compared the extracted original and extension implementations across **96 combinations**, including threshold boundaries, invalid runs and non-finite distances; statuses agreed. The A10 axis correctly uses fixed BE plus adaptive CN.

5. **Continued A2.4 order and trajectory source lookup — CORRECT for matching, complete archives.**

   The selector tries gl80 from 80–96, then gl96 from 96–112, only when the original A10 selection has no converged rule. `W_of` correctly reads gl80 and ladder trajectories from the original archive, and gl96 from the extension. Ladder distances are recomputed; original timing panels and candidate eligibility remain the selection inputs.

   A11’s “\(m^\star\) … taken unchanged” should mean **the selection procedure**, not its previously computed value.

6. **gl80 reproduction gate — WRONG.**

   At [select3d.py:147](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/select3d.py:147), absent arms are silently skipped. Acceptance requires only a **nonempty** reproduction dictionary.

   I exercised that exact code with only the fixed-BE gl80 array present in both archives: **`rep_ok=True`**, despite three missing arms.

   The numerical test itself is correct: finite field distance, normalized by the physical initial norm, ≤ \(10^{-10}\), across all supplied cases and times. The defect is coverage. Require **exactly all four expected arms**, expected array shapes and complete case coverage before accepting.

7. **Resistance to mismatched archives and invalid evidence — WRONG.**

   [select3d.py:143](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/select3d.py:143) checks matching cohort seed and configured case list; accessed arrays also undergo ordered-ID checks. Flagged smokes are refused by default. Those protections work.

   Missing protections remain:
   - No binding of extension bank, \(R'\), startup `a`, mesh, `gtol`, trust fraction or convergence bar to the original setting.
   - No enforcement that this production extension is **W1024, cohort 923801, all 64 cases**. Matching reduced subsets can pass if unflagged.
   - Extension `posthoc_status` is trusted without recomputing its consistency with distances, eligibility and the prescribed bar. A run using a looser `conv_bar` can supply a passing status.
   - Imported converged-rule eligibility is not independently checked against complete extension rows before selection.
   - `--allow-smoke` permits smoke-derived output without an explicit smoke designation in the selection output.

   Require a matching numerical contract, exact production population, complete extension evidence, and independently reconstructed acceptance statuses.

8. **Memory and residency — CORRECT structurally; capacity unverified.**

   Prepared rule arrays are dropped after each rule’s rollouts. Retained trajectories are NumPy coefficient arrays; solver caches take large rule arrays as arguments. I found no reference retaining every prepared rule.

   However, at gl112 and \(R'=1024\), the actual \(M=4097\) gives **69.07 GB for B, D and P alone**. A single additional \(m\times R'\) float64 temporary is **11.51 GB**. Setup and contraction temporaries, mesh tables and compilation allocations add to this. Dropping previous rules does not establish that the peak fits.

   The proposed full-size cluster smoke is necessary.

9. **Cluster smoke/resource integration — WRONG.**

   `smk3ext` intentionally produces no timing panels. But [inventory.py:248](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:248) requires `bool(sm['timing'])` in `smoke_gates`.

   Therefore **a successfully completed smk3ext is still rejected as resource evidence**. I confirmed the same rejection on the supplied local extension output. Production staging requires the certificate, so A11’s proposed certification workflow cannot work as written. It needs an extension-specific machinery gate and evidence covering the largest rule.

10. **Local machinery evidence — NEEDS-RESTATEMENT.**

    The output is complete: 12 finite trajectories, all rows marked valid, first pair failing and second passing. It exercises the branch that **runs** the conditional rule, not the branch that skips it.

    It does not validate production memory: it uses W256 and gl24–40, with an 8.40 GB reported peak. Moreover, the log contains **“Red zone modified”**, the warning already implicated in A8’s local-GPU corruption, and takes about **852 seconds**, exceeding the repository’s local smoke limit. Treat it as control-flow evidence, not accepted numerical validation.

11. **Post-hoc disclosure, replication and case-5 treatment — CORRECT, with a scope qualification.**

    A11 explicitly declares itself post hoc, and the selector labels extension use accordingly. The configured production job uses validation cohort **923801**, not replication cohort **923901**. I found no replication-based selection in this path.

    Case 5 is neither excluded nor given different solver settings, weights or thresholds. The extension runs all 64 cases uniformly. The decision to extend W1024 was nevertheless motivated by observed validation outcomes; it remains exploratory selection on that cohort, not independent confirmation. The claim about what humans had previously read cannot be verified from code.

12. **Downstream qualification integration — WRONG, adjacent blocker.**

    [make_budget_configs.py:14](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/make_budget_configs.py:14) still defines only `gl56 → gl64` and `gl64 → gl80`. A successful A11 selection of **gl80 or gl96** causes budget-config generation to fail at `FINER[r['converged']]`. The required successor mappings are missing.

**Acceptance verdict: withhold.** The extension experiment and driver semantics are sound, but the selector can accept incomplete or incompatible evidence, and the smoke-certification and downstream-budget paths need correction. No files were created, modified or appended by my tool actions.