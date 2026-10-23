**NOT CLEAN.** The E1 crossings, revised problem verdicts, totals, and branch table check out. One blanket statement is wrong, and three qualifications need restatement. No files were changed or jobs submitted.

Locations below refer to [final/report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/final/report.md). `E1/` means `worktrees/2026-10-22-imp-box/experiments/imp-box/`; other `E<n>/` aliases mean their specified lane directories. E8–E10 share `imp-fronts`.

**Opening and §1 — Status and summary**

1. **CORRECT — Final status.** Report: “final for every lane”; E4 not run. Sources: `E1/EXPORT.json → stage = final`; `campaign/LANES.json → lanes[*].status = done` for all seven lanes. Final refers to completed reporting, while numerical limitations remain disclosed.

2. **CORRECT — Numerical provenance.** Verified **226 file-backed provenance entries**, their key values and SHA-256 hashes, and **13 figures** against source and copied-file hashes. Source: `final/PROVENANCE.json → [*].{file,key_path,value,sha256}`, `_figures`. Derived extrema, counts, ratios and ledger sums reproduce the displayed rounding; no invented measured scalar found.

3. **NEEDS-RESTATEMENT — Opening cohort qualification.** Report: all POD comparisons are “in-sample (dev6 + val32).” E1 instead uses **dev8, val32 selection and fresh test32 evaluation**. Sources: `E1/runs/cells/*/out/cell/final.json → n_dev,n_val`; `E1/report.md → Test-once`; `E1/EXPORT.json → test`. Restrict the dev6+val32 statement to the applicable lanes and distinguish E1 validation selection from frozen-pair test evaluation.

4. **CORRECT — Problem verdicts.** Report: Problem 1 partly fixed; Problems 2 and 3 not fixed. Recomputed: E1 bank-ETDRK4 reaches **2.068727994** in W08_A2; E2 maximum validation ratio is **0.216099933**; E3 gate ratios are **0.858652903/0.923172269**. Sources: `E1/EXPORT.json → cells.W08_A2.entries["bank_e4|0.01|worst"].S`; E2/E3 fields below. This satisfies the supplied coordinator rule.

**§2 — E1 box**

5. **CORRECT — Raw E1 reconciliation.** All **186 exported cell entries** reconcile with cell `final.json` records; **164 non-null validation ratios** reproduce as medians of paired \((a_1+a_2)/(2b)\) timings. All **164 test records**, including **142 raw test timing ratios**, reconcile. Sources: `E1/EXPORT.json → sources,cells,test`; corresponding `runs/cells/*/out/cell/{final,test}.json → S,pairs[*].raw`.

6. **CORRECT — Complete-search W40_A3 translation crossings.** Report:
   - 3% median: **2.14 / 2.2** validation/test.
   - 10% worst: **2.14 / 2.16**.

   Recomputed: **2.143782370 / 2.196643004** and **2.143782370 / 2.163983882**. Sources: `E1/EXPORT.json → cells.W40_A3.entries["trans|0.03|median","trans|0.1|worst"].S` and corresponding `test.W40_A3.pairs[*].S_test`. Both FOM searches are finalists; `spec|N24|dense|nt10` and the printed minimum labels agree.

7. **CORRECT — Complete-search W08_A2 crossings, 1% worst.** Report validation/test values reproduce:
   - POD-Galerkin **2.14/2.05** ← **2.140504760/2.048901592**.
   - POD+ECSW **2.01/2** ← **2.009344770/1.996464231**.
   - Bank-ETDRK4 **2.07/2.04** ← **2.068727994/2.042736408**.

   Sources: `E1/EXPORT.json → cells.W08_A2.entries["{podg,ecsw,bank_e4}|0.01|worst"].S`, corresponding `test` pairs. All use `spec|N48|dense|nt10`. ECSW’s test value rounds to 2 but is slightly below 2; the table correctly selects rows by validation crossings.

8. **CORRECT — Incomplete-search separation.** Report W40_A3 translation values **3.32/3.3**, **3.27/3.24**, **2.26/2.27** reproduce as:
   **3.318997407/3.304465168**, **3.273637883/3.237562129**, **2.261765949/2.269167599**. Sources: `E1/EXPORT.json → cells.W40_A3.entries["trans|0.01|median","trans|0.01|worst","trans|0.03|worst"]`, corresponding test pairs. Each has `fom_status.specad = search-incomplete`; exclusion from the complete-search table is correct.

9. **WRONG — Blanket ladder-minimum claim, E1 “Reading” paragraph.** Report: “All \(S\ge2\) entries carry the reduced-model ladder-minimum label.” Recomputed: **7 of 8**, not all eight. Source: `E1/EXPORT.json → cells.W40_A3.entries["trans|0.01|worst"].ladder_minimum = []`, with `S = 3.273637883`; its test labels contain only the incomplete-search qualification. Change this to “all five complete-search entries,” or explicitly identify the exception.

10. **CORRECT — Bank scope and deployed-stepper range.** Report: bank crossing only W08_A2, ETDRK4, 1% worst; wide-family ETDRK4 maximum **1.46**; deployed LSPG **0.0244–0.396 over 28 entries**. Recomputed: **1.455409026**, **0.024442664–0.396322645**, **28**. Source: `E1/EXPORT.json → cells[*].entries`, filtered by arm, finalist status and wide-family cells. The narrow-family/low-steepness condition and comparable POD performance are stated correctly.

11. **CORRECT — Stop rules.** Report: SPEED-STOP, scoped SPEED-STOP and BANK-STOP **NOT FIRED**; **95** matched wide comparisons, maximum **2.14**. Recomputed: **95**, maximum **2.143782370**, including **80** comparisons at least 0.5. Sources: `E1/runs/stoprule.json → entries,speed_stop,bank_stop`; `E1/EXPORT.json → stop_rule`. Existing counterexamples prevent the universal speed-stop condition from firing.

12. **CORRECT — Scope and coordinator rulings.** Report lists **9 collected and 7 deferred cells**, including all A4 cells, W16_A2/A3 and W08_A3; accepts conditional spectral references and no downward ladder extension as labelled deviations. Sources: `E1/EXPORT.json → scope`; `E1/DESIGN.md → A2.2,A14–A17`; `INTERVENTIONS.md → final E1 ruling`. Scope matches.

13. **NEEDS-RESTATEMENT — Missing A16 timing caveat beside E1 cross-cell plots and in §7.** Report: no disclosure. Recomputed dense-FOM hoisted/embedded timing ratios span approximately **0.898–0.906**; other query ratios differ too. Sources: `E1/EXPORT.json → a16_timing[cell][query].{raw_embedded,raw_hoisted,ratio}`; `E1/report.md → A16 comparability`; `E1/DESIGN.md → A16`. The lane explicitly requires this qualification beside comparisons spanning embedded and hoisted cells; it does not invalidate the within-cell crossings.

14. **NEEDS-RESTATEMENT — Missing finalist-selection audit limitation.** Report calls searches “complete” and finalists fastest without carrying the lane’s narrower auditability qualification. Source: `E1/report.md → Search limitations`: ordering within each **six-candidate screening batch** is retained only for attempted candidates, so superiority over every unattempted batch member cannot be independently reconstructed. State that “complete” means both searches returned finalists, while retaining this documented selection limitation.

**§2 — E2–E4**

15. **CORRECT — E2 table and verdict.** Reported validation/test ranges reproduce:
   - L: **0.0147–0.214 / 0.0128–0.0903**, **2** empty test cells.
   - H: **0.0292–0.216 / 0.0306–0.120**, **1** empty.

   SEM worst-target ranges **0.0856–0.108 / 0.0865–0.234** and FD+MG ratios **0.471/1.36 / 0.653/1.77** also agree. Sources: `E2/EXPORT.json → cells[*].{S_val,S_test,per_family_val,allowance,labels}`, `geometry_labels,A10`. L’s **4/32** reference failures and H’s measured withdrawal support the qualified verdict.

16. **CORRECT — E3 gate, errors and test population.** Report: **0.859/0.923**, ROM errors **2.71%/6.73%**, intervals **[0.799,0.900]/[0.877,0.943]**, test **14/16 eligible**, errors **11.7%/2.04%**. Recomputed gate ratios:
   **0.280741898/0.326956209 = 0.858652903**;
   **0.136499024/0.147858670 = 0.923172269**.
   Sources: `E3/checks/gate.json → targets[*].{fom.cost,rom.cost,rom.worst,G_ci95}`; `E3/EXPORT.json → test`; test-reference `cases[*].judge.outcome`. Values and target meanings agree.

17. **CORRECT — E3/E4 decisions.** Report: E3 **FAIL under A10 / UNRESOLVED under the registered rule**, no ROM; E4 unrun. Sources: `E3/EXPORT.json → gate,gate_registered_rule.outcome`; `AGENT-PROMPT.md §2`; `INTERVENTIONS.md`. Neither E3 reading authorizes training, and E4’s E2-win prerequisite was unmet.

**§3 — Problem 2**

18. **CORRECT — E5 table and stopping.** Reported worst errors reproduce, in row order, as **1.850974, 1.815248, 1.908120, 1.926565; 1.850320, 1.814506, 1.907564, 1.925850%**. Evolution ratios reproduce as **0.726166, 14.277634, 22.849436, 1.583933; 0.732675, 14.974450, 24.149574, 1.592867**. Sources: `E5/EXPORT.json → verdicts["f{mesh}|{arm}"]`; `E5/checks/f{1024,4096}.json → stop`. Counts, qualification and every worst-error stop agree.

19. **CORRECT — E5 POD comparison and selection.** Report: POD ratios **0.623/0.796/0.0363**, selections `None` and `G-E4a|384`. Sources: `E5/EXPORT.json → verdicts["f1024|matches"][*].direct.value`, `verdicts["f{mesh}|best_stepper"]`. Exact ratios **0.623042485/0.796270139/0.036340605** agree; HR-unqualified labels remain disclosed.

20. **CORRECT — E6 table and rules.** Reported errors, nodes and gains match `E6/EXPORT.json → arms[*].{worst,m,e2e,q2}`. F3 gives **0/6** Q2 passes, best gain **1.241094885**; microbenchmark ratios **1.373800808/2.138603559** round to **1.37/2.14**. Sources: `stop`, `jac_bench`. Both stop verdicts follow; F2’s roughly 1.02 point estimate does not establish the required faster result.

21. **CORRECT — E7 workloads and costs.** Report: **21×6=126** pairs; MATCH counts **126,126,126,84,0**; corresponding NO-MATCH counts **0,0,0,42,126**. Cost ratios **0.0991–0.131**, median **0.105**, reproduce as **0.099119332–0.130716684**, median **0.104847227**. Sources: `E7/EXPORT.json → verdict_counts,primary_ratio,nomatch_without_admissible`. Primary FIRES and the projected-cost qualifications are correct.

**§4 — Problem 3**

22. **CORRECT — E8 diagnostics and retraining.** Report’s Phase-A outcomes match `E8/EXPORT.json → A1–A4`. Coverage improvement recomputes as **0.081215148/0.058100539 = 1.397838111**; arm-c tail improvement **1.482458600**; **16384 > 12800** nodes; guards **0.941709707/1.265312241** pass. Sources: `phase_b.routes,guards,score_b,score_c`. Neither retrain meets its improvement rule; zero FC1 qualification is preserved.

23. **CORRECT — E9 descriptive indicator.** Report: threshold **113**, flagged **79.2%**, FI **16.7%**, AUC **1**, false alarms **0/0.286**, **19 bad/5 good**. Source: `E8/EXPORT.json → phase_b.e9`. Exact values include **112.774210457**, **19/24**, **1/6**, **2/7**; the **30-good/20-bad** minimum remains unmet.

24. **CORRECT — E10.** Report: diagnostic median **8.29%**, bar **4.06%**, **0/8** qualified; quadrature maximum **0.0386**, **17600** states, **0** exceedances. Sources: `E8/EXPORT.json → e10_results`; `E8/checks/phaseb/e10q_every_step.json → runs[*]`. Values reproduce as **8.286403341%**, **4.060757379%**, maximum **0.038633612**. STOP outcomes and test-reuse disclosure agree.

**§§5–8 — Verdicts, claims, deviations and deferrals**

25. **CORRECT — Stop-rule table and claims.** Report: E1 partly; E2/E3 unsuccessful; E4 unrun; E5–E8/E10 stopped; E9 descriptive. Sources: the lane verdict fields above and `AGENT-PROMPT.md §2`. Retaining mesh-independent operation as a property, dropping comparative multi-mesh advantage, and limiting the box claim to named arms/cells follow from the evidence.

26. **CORRECT — Previously disclosed incidents and corrections.** E6 DO-NOT-SUBMIT smoke, E8 SM-c admission breach, E3 post-hoc A10, E2 smoke-informed redesign, E5/E7 scratchpad and pkill incidents, usage-limit stops and GitHub outage agree with `INTERVENTIONS.md` and lane reports. E2 overrun recomputes to **1.0742−1 = 0.0742 GPU-h**; SM-c arithmetic **12.277≈12.28≤15**. The additional E1 omissions are identified in items 13–14.

27. **CORRECT — Deferred scope.** Reported E1 cells and second training size, E2 S2/narrow/box controls, E3 fallback and competitors, E5 controls, E7 uncosted items and fronts deferrals match `E1/EXPORT.json → scope`, lane reports and `E8/runs/GPU-LEDGER.json → deferred`. E10-q is correctly described as resolved without another job.

**§9 — GPU-hour ledger**

28. **CORRECT — Spending and totals.** Independent job sums reproduce:
   **E1 48.967777778**, E2 **30.6789**, E3 **2.0372**, E5 **1.4076**, E6 **1.4608**, E7 **3.1602**, fronts **10.7443**.
   Sources: each lane’s `runs/GPU-LEDGER.json → jobs[*].{gpu_hours,elapsed_hours,charged_gpu_h}`. Fronts splits **10.5451+0.1992**; recorded campaign total **98.4568 → 98.46**, caps **204**; unspent E4 cap **80** is correctly excluded.

**§10 — Branches and mirrors**

29. **CORRECT — All seven rows.** Reported local/mirror hashes match local branch and remote-tracking refs:
   **9c35fdd83/17e4876d6**, **0dc801d2b/b89dbd615**, **e8d39fa1a/6a98c018a**, **a3e417b1e/ed52f5477**, **cf351b62a/c891ee4cc**, **dd4cd7fc4/4ade95d16**, **506dbd3a5/a2f64ebda**. Commit timestamps and `LANES.json` statuses also match. Live `ls-remote` failed DNS resolution; the explicitly permitted local corroboration passed.

**§11 — Hygiene**

30. **CORRECT — Structure and definitions.** Report has a level-1 title, explicit status, LaTeX mathematics, a colour-coded Mermaid diagram and an ending glossary defining the principal table columns, arms and quantities. Source: `CLAUDE.md → Reports`; report opening and §11. Conditional-reference, finite-ladder and diagnostic qualifications are present; the cohort qualification needs the correction in item 3.

**WRONG: 1, NEEDS-RESTATEMENT: 3 — overall verdict: NOT CLEAN.**