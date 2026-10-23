**CLEAN.** All round-2 WRONG and NEEDS-RESTATEMENT findings are resolved. No files were changed, no jobs submitted, and no write-capable git commands run.

Locations refer to [final/report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/final/report.md). Source aliases such as `E5/` mean the corresponding `worktrees/<dated-lane>/experiments/<lane>/` directory; E8–E10 share `imp-fronts`.

**Opening and §1 — Summary**

1. **CORRECT — Numerical provenance throughout.** Report claims source-generated numbers. Verified **209 file-backed entries**, including exact key values and SHA-256 hashes; independently recomputed **30 derived entries**; verified **11 figures** against both source and copied-file hashes. Source: `final/PROVENANCE.json → [*].{file,key_path,value,sha256,derived_from}`, `_figures`. No invented measured scalar found.

2. **CORRECT — Diagram legend and summary verdicts.** Report: negative/unsupported outcomes, with E3’s registered UNRESOLVED and E2’s descriptive L result distinguished; none of the completed experiments fixes its problem. Sources: lane `EXPORT.json` verdicts detailed below. These conclusions preserve the pending E1 scope and the qualification limitations.

**§2 — Problem 1**

3. **CORRECT — E1 pending and spending.** Report: **pending, 41.35/87 GPU-h**. Recomputed job sum **41.3547222**; recorded subtotal **41.3547**. Sources: `E1/runs/GPU-LEDGER.json → jobs[*].gpu_hours,total_gpu_hours`; `E1/EXPORT.json → stage = interim`; `campaign/LANES.json → lanes[lane=imp-box].{status,gpu_hours_cap} = active,87`. Pending treatment is appropriate.

4. **CORRECT — E2 comparison table.** Reported values reproduce from `E2/EXPORT.json → cells["{L,H}|wide|S1"].{S_val,S_test,per_family_val}`:
   - L: validation **0.0147–0.214**, test **0.0128–0.0903**, **2** empty; SEM worst-target range **0.0856–0.108**; FD+MG **0.471/1.36**.
   - H: validation **0.0292–0.216**, test **0.0306–0.120**, **1** empty; SEM **0.0865–0.234**; FD+MG **0.653/1.77**.
   
   Extrema, null counts, target selection, competitor names and rounding agree.

5. **CORRECT — E2 verdict and exceptions.** Report: H withdrawn; L descriptive with **4/32** ineligible references; maximum validation ratio **0.216**. Recomputed maximum **0.216099933161**; L eligibility **28+4=32**. Source: `E2/EXPORT.json → cells[*].{allowance,S_val}`, `geometry_labels`, `A10`. All five FD+MG exceptions are correctly listed, including H’s **3% median** ratio **1.120718094963**. Competitor admission asymmetry is disclosed consistently with `E2/DESIGN.md A4.2`.

6. **CORRECT — E3 gate table.** Report: ROM errors **2.71%/6.73%**, ratios **0.859/0.923**, intervals **[0.799,0.900]/[0.877,0.943]**. Recomputed:
   - **0.280741898228017 / 0.326956209414297 = 0.858652902574**;
   - **0.1364990241029282 / 0.14785867018314236 = 0.923172269397**.
   
   Source: `E3/checks/gate.json → targets["0.03|C100","0.1|C100"].{fom.cost,rom.cost,rom.worst,G_ci95}`. Winner identifiers and the unqualified 1% target match.

7. **CORRECT — E3 test population and GPU-hours; round-2 items 9–10 resolved.** Report now says **14 eligible of 16**, **2** excluded references, test errors **11.7%/2.04%**, and **2.04 of 15 GPU-h**. Raw eligible-row maxima are **11.6600047428%/2.0447805721%**. Sources: `E3/runs/test/output/vtest_test-run/result.json → keys[ETD winner].rows[*].{eligible,e},n_eligible`; reference result `cases[*].judge.outcome`; ledger `total_gpu_hours = 2.0372`. Population and rendering are repaired.

8. **CORRECT — E3/E4 decisions.** Report: E3 **FAIL under post-hoc A10 / UNRESOLVED under the registered rule**, no ROM trained; E4 unrun. Sources: `E3/EXPORT.json → gate,gate_registered_rule.outcome`; `AGENT-PROMPT.md §2`; `INTERVENTIONS.md` final E2 entry. Neither E3 reading authorizes training, and E4’s prerequisite was unmet.

**§3 — Problem 2**

9. **CORRECT — E5 table.** In table order, recomputed worst errors (%) are **1.850974, 1.815248, 1.908120, 1.926565; 1.850320, 1.814506, 1.907564, 1.925850**. Incumbents are **1.775269%/1.774683%**. Median evolution ratios are **0.726166, 14.277634, 22.849436, 1.583933; 0.732675, 14.974450, 24.149574, 1.592867**. Sources: `E5/checks/f{1024,4096}.json → stop[arm].worst_all38["384"]`, `.speed.statistic.per_case`. All displayed rounding, **36/2 or 34/4** counts, qualification labels and fired conditions agree.

10. **CORRECT — E5 summary and POD comparisons.** Report: ETDRK4 range **14.3–24.1×**, POD query ratios **0.623/0.796/0.0363**, frozen selections `None` and `G-E4a|384`. Source values are **0.623042485/0.796270139/0.036340605** in `E5/EXPORT.json → verdicts["f1024|matches"][configuration].direct.value`. The matched ETDRK4 competitors’ **HR-unqualified** labels are explicitly retained from `E5/checks/f1024.json → pod[configuration].Q2.label`. Every bank arm exceeds incumbent worst error, so the stop verdict follows.

11. **CORRECT — E6 table and stop rules.** Raw-score maxima reproduce **1.851, 2.131, 1.859, 1.889, 1.851, 5.951, 1.851, 1.853, 2.131%**. Source: `E6/checks/analysis-prd1.json → scores[arm][case].e`. Nodes, gains and qualification match `E6/EXPORT.json → arms[arm].{m,e2e,q2,qualified}`. F3 has **0/6** Q2 passes and best gain **1.241094885**. F2’s **1.022153017×** still fails “faster” because its interval lower bound is **0.992451990**.

12. **CORRECT — E6 microbenchmark.** Report: **1.37×/2.14×** for doubled/quadrupled points. Source: `E6/EXPORT.json → jac_bench.{dup_ratio,quad_ratio} = 1.373800808,2.138603559`. The stated sublinear scaling follows.

13. **CORRECT — E7 workloads, counts and ratios.** Recounting `E7/checks/analysis.json → verdicts[criterion|accounting].table[workload][configuration]` gives **21×6=126** pairs; MATCH counts **126,126,126,84,0** and NO-MATCH counts **0,0,0,42,126**. Primary `.best.cost/.C_ours` gives **0.099119332–0.130716684**, median **0.104847227**. A100 and no-head each retain **126** matches; all sensitivity NO-MATCH entries lack admissible competitors. Reported values and FIRES verdict agree.

14. **CORRECT — E7 provisional accounting.** Report identifies projected bank-fit time and conditional compilation costs beside the **0.0991–0.131** comparison. Source: `E7/report.md → Our offline cost`, compilation discussion. The ratio is not presented as a wholly measured training-cost comparison.

**§4 — Problem 3**

15. **CORRECT — E8 Phase A.** Report: **0 GPU-h**, A1 INCONCLUSIVE, A2 RUN, A3 CPU screen passed with timing deferred, A4 STOP F15. Sources: `E8/EXPORT.json → gpu_hours_phase_a,A1.verdict,A2.decision_arm_c,A3.verdict,A4.verdict`. All match.

16. **CORRECT — E8 numerical endpoints.** Raw FC1 medians reproduce **5.810053892%** for b, **8.121514759%** for K1, and **7.545365848%** for c; each has **0/8 qualified**. Source: `E8/checks/phaseb/score_{b,K1,c}.json → cases[cohort=FC1].{e,qualified}`. Improvement ratios are **1.397838111×** and **1.482458600×**; **16384>12800** nodes. Source for tail/node quantities: `E8/EXPORT.json → phase_b.routes.c`. Both endpoints correctly STOP.

17. **CORRECT — E8 registered logic and guard.** Report correctly stops arm c when **both** improvement routes fail and calls the guard a representation-floor comparison. Recomputed guard ratios **0.941709707/1.265312241** both pass the **2×** bar. Sources: `AGENT-PROMPT.md §2`; `E8/EXPORT.json → phase_b.guards.{b,c}.{arm_worst,ctrl_worst,ratio}`.

18. **CORRECT — E9.** Reported threshold **113**, flagged **79.2%**, FI **16.7%**, AUC **1**, false alarms **0/0.286**, and **19 bad/5 good** match `E8/EXPORT.json → phase_b.e9`. Exact threshold **112.774210457**; fractions **19/24**, **1/6**, **2/7**; target thresholds **0.003834528307579255/0.01**. The **30-good/20-bad** minimum is unmet, supporting “inconclusive.”

19. **CORRECT — E10 arithmetic and inline qualification; round-2 item 22 resolved.** Report: diagnostic, conditional **8.29%**, bar **4.06%**, K1 **8.12%**, **0/8 qualified**. Raw median **8.286403341%**; half-K1 bar **4.060757379%**. Sources: `E8/checks/phaseb/score_e10s.json → cases[cohort=FC1]`; `E8/EXPORT.json → e10_results.verdicts["E10-s"]`. The repaired text explicitly says qualification is ignored for these medians.

20. **CORRECT — E10-q.** Report: maximum **0.0386**, **17600** accepted-step states, **0** exceedances, STOP without another job. Independently summing `states−1` gives **17600**; maximum **0.0386336118107999**, attained at step index **82**; summed exceedances **0**. Source: `E8/checks/phaseb/e10q_every_step.json → runs[*].{states,rho_max,rho_argmax,n_over}`. The registered trigger does not fire.

**§§5–6 — Stop-rule table and claims**

21. **CORRECT — All experiment and problem verdicts.** Report: E1 pending; E2 unsupported; E3 dual verdict; E4 unrun; E5/E6 fired; E7 FIRES; E8/E10 STOP; E9 descriptive. Sources: the verdict fields and registered rules cited above. The withdrawal of comparative claims follows, while mesh-independent operation remains a property.

22. **CORRECT — Reference-arm identity.** Report now distinguishes incumbent LSPG CN-R from E6’s precontracted Galerkin CN reference. Source: `E6/EXPORT.json → arms.R.worst = 0.01850973809729437`, matching E5 Galerkin CN rather than incumbent **0.01775269104350181**. The earlier identity error remains resolved.

**§7 — Retractions and deviations**

23. **CORRECT — Incident coverage.** Report faithfully discloses the E6 DO-NOT-SUBMIT smoke, E8 admission breach, E3 post-hoc rule, E2 diagnostic-guided redesign, E5/E7 scratchpad and pkill incidents, early E8 release, E10 test reuse, usage-limit stops and GitHub outage. Sources: `INTERVENTIONS.md`, cited lane audit notes and reports. E2 overrun recomputes to **1.0742−1=0.0742 GPU-h**; SM-c quoted arithmetic is **12.277≈12.28≤15**.

24. **CORRECT — Corrections and chronology; round-2 item 26 resolved.** Report now explicitly places E8 wall-residual and E7 resampling findings in results audits. Wall residual **1.03e−13** versus **1e−14** matches `E8/EXPORT.json → phase_b.armc_wall.{max_rel_wall_rms,registered_bar}`. F2’s diffusion change and E7’s phase-wise/resampling corrections match the respective lane report correction sections.

**§8 — Deferred items**

25. **CORRECT — Deferrals; round-2 item 27 resolved.** Report now includes E3’s **A8.3 fallback**, reused factorisations, **C=10** timing/validation and **CCv L512**. Sources: `E3/report.md §§2,5`. Other deferrals match lane reports and `E8/runs/GPU-LEDGER.json → deferred`; E10-q is correctly resolved without a job.

**§9 — GPU-hour ledger**

26. **CORRECT — Spending and caps.** Independent job sums: E1 **41.3547222**, E2 **30.6789**, E3 **2.0372**, E5 **1.4076**, E6 **1.4608**, E7 **3.1602**, fronts **10.7443 GPU-h**. Fronts splits into **10.5451 E8 + 0.1992 E10**. Sources: lane `runs/GPU-LEDGER.json → jobs[*].{gpu_hours,elapsed_hours,charged_gpu_h}`. Finished subtotal **49.4890**, recorded overall **90.8437**, and cap sum **204** reproduce the report’s rounding; E4’s unspent **80** is excluded.

**§10 — Branches and mirrors**

27. **CORRECT — All seven rows.** `git log -1` reproduces every reported local hash and timestamp: **576ad5677, 0dc801d2b, e8d39fa1a, a3e417b1e, cf351b62a, dd4cd7fc4, 506dbd3a5**. Local remote-tracking refs reproduce mirrors **c77739769, b89dbd615, 6a98c018a, ed52f5477, c891ee4cc, 4ade95d16, a2f64ebda**. Statuses match `campaign/LANES.json → lanes[*].status`. Live `ls-remote` failed DNS resolution; the permitted local corroboration passed.

**§11 — Glossary and hygiene**

28. **CORRECT — Definitions; round-2 item 32 resolved.** The report now defines evaluated cohorts, overloaded κ, C, CCv and F3a/F3b fitting targets; earlier FN0s, accounting, criterion and representation-floor corrections remain present. Sources: E3/E6/E7/E8 lane glossaries and report definitions. The previously identified terminology errors are repaired.

29. **CORRECT — Structure and mathematical formatting; round-2 item 33 resolved.** Report has a level-1 title, explicit provisional status, colour-coded Mermaid diagram and final glossary. Previously flagged glossary inequalities and τ use LaTeX; the F2 status and verbatim SM-c note are code-quoted source strings. Source: `CLAUDE.md → Reports`. Inline conditional, diagnostic, in-sample and finite-ladder qualifications are present at the audited comparisons.

**WRONG: 0, NEEDS-RESTATEMENT: 0 — overall verdict: CLEAN.**