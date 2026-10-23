**CLEAN.** All four round 4 findings are resolved. No files were changed and no jobs were submitted.

Locations refer to [final/report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/final/report.md). Source aliases below mean `worktrees/<dated-lane>/experiments/<lane>/`: E1 = `2026-10-22-imp-box`; E2/E3/E5/E6/E7 = the corresponding `2026-10-21-imp-*` lanes; E8–E10 share `2026-10-21-imp-fronts`.

**Opening and §1 — Status, provenance and summary**

1. **CORRECT — Final status.** Report: “final for every lane”; E4 not run. Verified: `E1/EXPORT.json → stage = final`; `campaign/LANES.json → lanes[*].status = done` for seven lanes; `planned[0].status` records E4 unrun. Completed reporting is correctly distinguished from conditional numerical evidence.

2. **CORRECT — Numerical provenance.** Report: measured numbers come from lane outputs. Verified **226 file-backed entries**, including their values, key paths and SHA-256 hashes, plus **13 source/copied figure hashes**. Source: `final/PROVENANCE.json → [*].{file,key_path,value,sha256}, _figures`. No invented measured scalar found.

3. **CORRECT — Round 4 cohort correction.** Report now limits “in-sample on dev6 + val32” to E5–E7 and distinguishes E1 val32 selection from fresh test32 evaluation. Verified E1 raw `final.json → n_dev = 8, n_val = 32` and separate test outputs. Round 4 item 3 is resolved.

4. **CORRECT — Problem verdicts.** Report: Problem 1 partly fixed; Problems 2 and 3 not fixed. Recomputed E1 bank-ETDRK4 crossing **2.068727994**, E2 maximum **0.216099933**, E3 ratios **0.858652903/0.923172269**. Sources: `E1/EXPORT.json → cells.W08_A2.entries["bank_e4|0.01|worst"].S`; `E2/EXPORT.json → cells[*].S_val`; `E3/EXPORT.json → tau_*.G`. These satisfy the supplied coordinator rule.

**§2 — E1 box**

5. **CORRECT — Raw-output reconciliation.** Report’s E1 numbers reconcile with **186 exported entries**, **164 non-null validation ratios**, **164 test records**, and **142 available raw test timing ratios**. Sources: `E1/EXPORT.json → cells,test,sources`; corresponding `runs/cells/*/out/cell/final.json → S,pairs` and `runs/cells/test_*/out/cell/test.json → pairs`. Ratios reproduce as medians of `(a1+a2)/(2b)`.

6. **CORRECT — Complete-search translation crossings.** Report W40_A3 validation/test values:
   - 3% median **2.14/2.2** → **2.143782370/2.196643004**.
   - 10% worst **2.14/2.16** → **2.143782370/2.163983882**.

   Sources: `E1/EXPORT.json → cells.W40_A3.entries["trans|0.03|median","trans|0.1|worst"].S`, corresponding `test.W40_A3.pairs[*].S_test`. Both searches return finalists; FOM key `spec|N24|dense|nt10` agrees.

7. **CORRECT — Complete-search W08_A2 crossings.** At 1% worst, report validation/test values reproduce:
   - POD-Galerkin **2.14/2.05** → **2.140504760/2.048901592**.
   - POD+ECSW **2.01/2** → **2.009344770/1.996464231**.
   - Bank-ETDRK4 **2.07/2.04** → **2.068727994/2.042736408**.

   Sources: `E1/EXPORT.json → cells.W08_A2.entries["{podg,ecsw,bank_e4}|0.01|worst"].S`, corresponding test pairs. All use `spec|N48|dense|nt10`; ECSW’s test value rounds to 2 but is slightly below it, consistent with the table’s explicit validation-based selection.

8. **CORRECT — Incomplete-search separation.** Report W40_A3 translation values **3.32/3.3**, **3.27/3.24**, **2.26/2.27** reproduce as **3.318997407/3.304465168**, **3.273637883/3.237562129**, **2.261765949/2.269167599**. Sources: `E1/EXPORT.json → cells.W40_A3.entries["trans|0.01|median","trans|0.01|worst","trans|0.03|worst"]`, corresponding test pairs. Each has `fom_status.specad = search-incomplete`; their separate presentation is correct.

9. **CORRECT — Round 4 ladder-minimum correction.** Report now says **5 of 5 complete-search crossings** carry the reduced-model minimum label. Recomputed: **5/5 complete**, versus **7/8 across all crossings**. Source: `E1/EXPORT.json → cells[*].entries[*].{S,fom_status,ladder_minimum}`. The incomplete `W40_A3/trans|0.01|worst` exception is no longer swept into a blanket claim; round 4 item 9 is resolved.

10. **CORRECT — Bank scope and deployed-stepper range.** Report: bank crosses 2 only in W08_A2 with ETDRK4 at 1% worst; wide-family maximum **1.46**; deployed LSPG **0.0244–0.396 over 28 entries**. Recomputed: **1.455409026**, **0.024442664–0.396322645**, **28**. Source: `E1/EXPORT.json → cells[*].entries`, filtered by arm and width. The narrow-family/low-steepness condition and comparable POD performance are explicit.

11. **CORRECT — Stop-rule outcomes.** Report: SPEED-STOP, scoped reading and BANK-STOP **NOT FIRED**; **95** matched wide comparisons, maximum **2.14**. Verified **95**, maximum **2.143782370**, with **80** comparisons at least 0.5. Sources: `E1/runs/stoprule.json → speed_stop,bank_stop,entries`; `E1/EXPORT.json → stop_rule`. Counterexamples prevent the universal speed-stop condition from firing.

12. **CORRECT — Scope and coordinator rulings.** Report lists **9 collected/7 deferred cells**, including all A4 cells, W16_A2/A3 and W08_A3. Source: `E1/EXPORT.json → scope`; `E1/DESIGN.md → A14–A17`. Conditional spectral references and no downward ladder extension are explicitly accepted, labelled deviations.

13. **CORRECT — Round 4 timing-caveat correction.** Report: dense-FOM hoisted/embedded ratio **0.898–0.906**, qualifying comparisons across differently timed cells. Recomputed from raw timing medians: **0.898388945–0.905892681**. Source: `E1/EXPORT.json → a16_timing[*].fom_dense.{raw_hoisted,raw_embedded,ratio}`. Disclosure appears beside the E1 plots and in §7; round 4 item 13 is resolved.

14. **CORRECT — Round 4 search-limitation correction.** Report defines “complete” as both FOM searches returning finalists and discloses incomplete ordering records within **six-candidate batches**. Source: `E1/report.md → Search limitations`. This preserves the lane’s auditability limitation; round 4 item 14 is resolved.

**§2 — E2–E4**

15. **CORRECT — E2 numerical table.** Report ranges reproduce:
   - L validation/test **0.0147–0.214 / 0.0128–0.0903**, with **2** empty test entries.
   - H **0.0292–0.216 / 0.0306–0.120**, with **1** empty.
   - SEM worst-target ranges **0.0856–0.108 / 0.0865–0.234**.
   - FD+MG 3%/10% worst ratios **0.471/1.36 / 0.653/1.77**.

   Sources: `E2/EXPORT.json → cells[*].{S_val,S_test,per_family_val}`. Extrema, null counts, target assignments and rounding agree.

16. **CORRECT — E2 verdict and reference qualification.** Report: H withdrawn; L descriptive/not decidable; S2 untested; **4/32** L references fail. Sources: `E2/EXPORT.json → geometry_labels,A10,cells["L|wide|S1"].allowance`; lane report §4. The measured loss and reference-limited outcome remain distinguished.

17. **CORRECT — E3 ratios, errors and test population.** Report **0.859/0.923**, ROM errors **2.71%/6.73%**, CIs **[0.799,0.900]/[0.877,0.943]**, **14/16** eligible test cases, test errors **11.7%/2.04%** agree. Ratios independently reproduce as **0.280741898/0.326956209 = 0.858652903** and **0.136499024/0.147858670 = 0.923172269**. Sources: `E3/checks/gate.json → targets[*].{fom.cost,rom.cost,rom.worst,G_ci95}`; `E3/EXPORT.json → test`; test reference `result.json → cases[*].judge.outcome`.

18. **CORRECT — E3/E4 decisions.** Report: **FAIL under post-hoc A10 / UNRESOLVED under the registered rule**, no variable-coefficient ROM; E4 unrun. Sources: `E3/EXPORT.json → gate,gate_registered_rule.outcome`; `AGENT-PROMPT.md §2`; `INTERVENTIONS.md`. Neither E3 reading authorizes training, and E4’s E2-win prerequisite was unmet.

**§3 — Problem 2**

19. **CORRECT — E5 table and stopping.** Report’s eight error rows reproduce as **1.850974, 1.815248, 1.908120, 1.926565; 1.850320, 1.814506, 1.907564, 1.925850%**. Evolution ratios reproduce as **0.726166, 14.277634, 22.849436, 1.583933; 0.732675, 14.974450, 24.149574, 1.592867**. Sources: `E5/EXPORT.json → verdicts["f{mesh}|{arm}"].{worst,worst_arm1,speedup_evolve_r384,fired,qualified}`; `checks/f{mesh}.json → stop`. Better/worse counts, qualification and every worst-error stop agree.

20. **CORRECT — E5 POD comparison and frozen selection.** Report **0.623/0.796/0.0363** reproduces as **0.623042485/0.796270139/0.036340605**; selections are `None` and `G-E4a|384`. Sources: `E5/EXPORT.json → verdicts["f1024|matches"][*].direct.value`, `verdicts["f{mesh}|best_stepper"]`. The failed selection and HR-unqualified comparator labels are retained.

21. **CORRECT — E6 table and rules.** Reported node counts, errors and gains match `E6/EXPORT.json → arms[*].{m,worst,e2e,q2}`. F2 gives **9216 nodes, 2.131173991%, 1.022153017×**; F3 gives **0/6** Q2 passes and best gain **1.241094885×**; F2F3_k16 gives **6144, 2.130550249%, 1.069997960×**. Microbenchmark ratios **1.373800808/2.138603559** round to **1.37/2.14**. Sources: `stop,jac_bench`; both registered stopping outcomes follow.

22. **CORRECT — E7 workload counts and cost ratios.** Report **21×6=126** pairs; MATCH counts **126,126,126,84,0**, NO-MATCH **0,0,0,42,126** agree. Cost range/median **0.0991–0.131 / 0.105** reproduce as **0.099119332–0.130716684 / 0.104847227**. Sources: `E7/EXPORT.json → verdict_counts,primary_ratio,nomatch_without_admissible`. Primary FIRES, sensitivity explanations and compilation-projection qualifications are consistent.

**§4 — Problem 3**

23. **CORRECT — E8 diagnostics and retraining.** Phase-A outcomes match `E8/EXPORT.json → A1–A4`. Coverage improvement **1.40×** recomputes as **0.081215148/0.058100539 = 1.397838111**; arm-c tail improvement is **1.482458600×**, with **16384 > 12800** nodes and diagnostic median **7.55%**. Guards **0.94/1.27×** reproduce as **0.941709707/1.265312241**. Sources: `phase_b.{routes,guards,score_b,score_c}`. Both retrains stop, with zero FC1 qualification.

24. **CORRECT — E9 descriptive indicator.** Report threshold **113**, flagged **79.2%**, FI **16.7%**, AUC **1**, false alarms **0/0.286**, **19 bad/5 good** agree. Exact values include **112.774210457**, **19/24**, **1/6**, **2/7**. Source: `E8/EXPORT.json → phase_b.e9`. The **30-good/20-bad** minimum is unmet and the result is correctly labelled inconclusive.

25. **CORRECT — E10.** Report median **8.29%**, bar **4.06%**, **0/8** qualified and **0** step-converged reproduce from **8.286403341%**, **4.060757379%**. Quadrature maximum **0.0386**, **17600 accepted-step states**, **0 exceedances** agree. Source: `E8/EXPORT.json → e10_results.{verdicts,score_e10s}`. Both STOP outcomes and test-case reuse are disclosed.

**§§5–8 — Rules, claims, deviations and deferrals**

26. **CORRECT — Stop-rule table and claims.** Report’s E1 partly/E2–E3 unsuccessful/E4 unrun/E5–E8 and E10 stopped/E9 descriptive outcomes match the verified lane fields and `AGENT-PROMPT.md §2`. The narrowed box claim, withdrawal of comparative multi-mesh advantage and retention of mesh-independent operation as a property follow from those outcomes.

27. **CORRECT — Procedural incidents.** Report’s E6 DO-NOT-SUBMIT smoke, E8 SM-c breach, E3 post-hoc A10, E2 smoke-informed redesign, E5/E7 scratchpad collision, pkill incident, usage-limit stops and GitHub outage agree with `INTERVENTIONS.md` and lane reports. E2 overrun recomputes as **1.0742−1 = 0.0742 GPU-h → 0.07**; SM-c arithmetic is **6.327+0.45+5.5 = 12.277 → 12.28 ≤ 15**. Sources: E2 ledger; E8 ledger `jobs[name=SM-c].note`.

28. **CORRECT — Retractions and deferred work.** Reported E6 gate/fixture/timer corrections, E8 control and wall-residual qualifications, E7 cost corrections and E5 audit correction match lane reports. E8 wall residual **1.03e−13 versus 1e−14** matches `E8/EXPORT.json → phase_b.armc_wall`. Deferred cells, controls, competitors, fallback timing and retrains match lane scope records and `E8/runs/GPU-LEDGER.json → deferred`.

**§9 — GPU-hour ledger**

29. **CORRECT — Spending, splits and totals.** Independent job sums:
   - E1 **48.967777778 → 48.97**
   - E2 **30.6789 → 30.68**
   - E3 **2.0372 → 2.04**
   - E5 **1.4076 → 1.41**
   - E6 **1.4608 → 1.46**
   - E7 **3.1602 → 3.16**
   - Fronts **10.7443 → 10.74**, split **10.5451/0.1992 → 10.55/0.20**

   Sources: each lane’s `runs/GPU-LEDGER.json → jobs[*].{gpu_hours,elapsed_hours,charged_gpu_h}`. Total **98.456777778 → 98.46**; caps sum to **204** from `campaign/LANES.json → lanes[*].gpu_hours_cap`. E4’s unspent **80** is correctly excluded.

**§10 — Branches and mirrors**

30. **CORRECT — All seven rows.** Reported local/mirror hashes match branch heads and local remote-tracking refs: **9c35fdd83/17e4876d6**, **0dc801d2b/b89dbd615**, **e8d39fa1a/6a98c018a**, **a3e417b1e/ed52f5477**, **cf351b62a/c891ee4cc**, **dd4cd7fc4/4ade95d16**, **506dbd3a5/a2f64ebda**. Commit timestamps also match `git log -1`. Live `ls-remote` failed DNS resolution; permitted corroboration against `refs/remotes/origin/codeonly/exp/*` passed.

**§11 — Report hygiene**

31. **CORRECT — Structure and qualifications.** Report has a level-1 title, final status with conditional-evidence qualifications, LaTeX mathematics, colour-coded Mermaid, generated numerical tables and an ending glossary explaining the table quantities and arms. Sources: `CLAUDE.md → Reports`; report opening, numerical sections and §11. Conditional references, in-sample comparisons, finite-ladder bounds and E1 search/timing limitations are carried beside the relevant results.

**WRONG: 0, NEEDS-RESTATEMENT: 0 — overall verdict: CLEAN.**