**NOT CLEAN.** All recorded numerical values and derived arithmetic check out, but two report errors remain. No files were changed.

Locations refer to [final/report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/final/report.md). Below, `E5/…` means `worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/…`; other experiment aliases follow your lane mapping.

**Opening, provenance and §1**

1. **CORRECT — L3, numerical provenance throughout.** All **208 file-backed entries** match their source values and SHA-256 hashes; all **28 derived entries** independently recompute; all **11 copied figures** match their source hashes. Source: `final/PROVENANCE.json → [*].{file,key_path,value,sha256,derived_from}` and `_figures`. No invented measured scalar was found.

2. **CORRECT — L27, diagram legend.** Report now says red means “negative or unsupported outcome,” explicitly distinguishing E3’s post-hoc FAIL from registered UNRESOLVED and E2’s descriptive L result. Sources: `E3/EXPORT.json → gate_registered_rule.outcome`; `E2/EXPORT.json → geometry_labels`. **Round-1 item 26 resolved.**

3. **CORRECT — L31, per-problem conclusions.** Report: none of the completed experiments fixes the three weaknesses; E1 pending. Recomputed outcomes: E2 unsupported, E3 FAIL/UNRESOLVED, E5–E8 negative, E9 inconclusive, E10 STOP. Sources: lane `EXPORT.json` verdict fields detailed below. The scoped conclusions follow.

**§2 — Problem 1**

4. **CORRECT — L37, E1 pending.** Report: pending, **41.35/87 GPU-h**. Source: `E1/EXPORT.json → stage = interim`; `E1/runs/GPU-LEDGER.json → total_gpu_hours = 41.3547`; `campaign/LANES.json → lanes[lane=imp-box].{status,gpu_hours_cap} = active,87`. Pending treatment is appropriate.

5. **CORRECT — L43–48, E2 ranges and comparisons.** Reported values recompute as:
   - L: validation **0.0147–0.214**, test **0.0128–0.0903**, **2** empty; SEM **0.0856–0.108**; FD+MG **0.471/1.36**.
   - H: validation **0.0292–0.216**, test **0.0306–0.120**, **1** empty; SEM **0.0865–0.234**; FD+MG **0.653/1.77**.

   Source: `E2/EXPORT.json → cells["{L,H}|wide|S1"].{S_val,S_test,per_family_val}`. Extrema, null counts, target selection and rounding agree.

6. **CORRECT — L48, FD+MG exceptions and admission asymmetry.** Report now includes H’s **3% median** exception, recomputed as **1.120718095**, and correctly enumerates all five cells with FD+MG ratios above one. Source: `E2/EXPORT.json → cells[*].per_family_val[*].fdmg`. The validity/accuracy-only competitor admission disclosure matches `E2/report.md §5` and `DESIGN.md A4.2`. **Round-1 items 7–8 resolved.**

7. **CORRECT — L48, E2 verdict.** Report: H withdrawn, L descriptive, **4/32** references ineligible, S2 untested. Source: `E2/EXPORT.json → cells["L|wide|S1"].allowance` gives **4 ineligible + 28 eligible**; `.geometry_labels` and `.A10` support the verdict. Largest validation ratio is **0.2160999332**, below the registered **2** bar.

8. **CORRECT — L58–64, E3 gate arithmetic.** Report: ROM errors **2.71%/6.73%**, ratios **0.859/0.923**, intervals **[0.799,0.900]/[0.877,0.943]**. Recomputed ratios:
   - **0.280741898228017 / 0.326956209414297 = 0.858652902574**;
   - **0.1364990241029282 / 0.14785867018314236 = 0.923172269397**.

   Source: `E3/checks/gate.json → targets["0.03|C100","0.1|C100"].{fom.cost,rom.cost,rom.worst,G_ci95}`. Winner identifiers and the unqualified 1% target also match.

9. **WRONG — L64, E3 GPU-hour sentence.** Report literally prints **`GPU-hours {gh(e3_gpu)} of {lane_cap["imp-varcoef"]}`**. Correct rendering: **2.04 of 15**. Sources: `E3/runs/GPU-LEDGER.json → total_gpu_hours = 2.0372`; `campaign/LANES.json → lanes[lane=imp-varcoef].gpu_hours_cap = 15`. The generator left interpolation placeholders in the report.

10. **WRONG — L64, E3 test population.** Report: “on the **16 test-once cases** … **11.7%** … **2.04%**.” Those errors cover **14 eligible cases**, with **2/16 order-fail references excluded**. Values recompute to **11.6600047428%** and **2.0447805721%** from `E3/EXPORT.json → test["etd|N48|0.0125|mid@C100","etd|N48|0.003125|mid@C100"].worst`; population and exclusion evidence is in `E3/report.md §5`. State “14 eligible cases out of 16”; **round-1 item 31 is only partly resolved.**

11. **CORRECT — L64–76, E3/E4 decisions.** Report: E3 **FAIL under A10 / UNRESOLVED under the registered rule**, no ROM trained; E4 not run. Sources: `E3/EXPORT.json → gate, gate_registered_rule.outcome`; `AGENT-PROMPT.md §2`; `INTERVENTIONS.md` E3 ruling and final E2 entry. Neither E3 reading authorizes training, and E4’s prerequisite was unmet.

**§3 — Problem 2**

12. **CORRECT — L84–95, E5 errors, speed-ups and verdicts.** In table order, independently recomputed errors (%) are **1.850974, 1.815248, 1.908120, 1.926565; 1.850320, 1.814506, 1.907564, 1.925850**. Evolution ratios are **0.726166, 14.277634, 22.849436, 1.583933; 0.732675, 14.974450, 24.149574, 1.592867**. Sources: `E5/checks/f{1024,4096}.json → stop[arm].worst_all38["384"]` and median of `.speed.statistic.per_case`. All printed rounding, **36/2 or 34/4** counts, qualification labels and fired conditions agree.

13. **CORRECT — L95, E5 POD comparisons.** Report: **0.623, 0.796, 0.0363**; recomputed **0.623042485, 0.796270139, 0.036340605**. Source: `E5/EXPORT.json → verdicts["f1024|matches"][arm+"|384"].direct.value`. Both matched ETDRK4 competitors are now explicitly **HR-unqualified**, matching `E5/checks/f1024.json → pod[configuration].Q2.label`. **Round-1 item 13 resolved in the comparative result.**

14. **CORRECT — L105–117, E6 table and stop rules.** Raw-score maxima reproduce **1.851, 2.131, 1.859, 1.889, 1.851, 5.951, 1.851, 1.853, 2.131%**. Source: `E6/checks/analysis-prd1.json → scores[arm][case].e`; nodes, gains and Q2 labels match `E6/EXPORT.json → arms[arm]`. Both stops fire; **0/6** F3 arms pass Q2; best gain is **1.241094885**. F2’s “faster=False” is consistent with its confidence interval **[0.992451990,1.068039189]**, despite point estimate **1.022153017**.

15. **CORRECT — L117, microbenchmark.** Report: **1.37×/2.14×** for doubled/quadrupled points. Source: `E6/EXPORT.json → jac_bench.{dup_ratio,quad_ratio}` gives **1.373800808/2.138603559**. The sublinear-cost observation follows.

16. **CORRECT — L129–139, E7 counts and cost ratios.** Recounting `E7/checks/analysis.json → verdicts[criterion|accounting].table[workload][configuration]` gives **21×6=126** pairs; MATCH counts **126,126,126,84,0** and NO-MATCH counts **0,0,0,42,126**. Primary `.best.cost/.C_ours` yields **0.099119332–0.130716684**, median **0.104847227**. A100 and no-head accountings also give **126** matches. All reported values and the primary FIRES verdict agree.

17. **CORRECT — L139, E7 projected costs.** Report now explicitly identifies both the **phase-wise projected bank-fit time** and compilation projections beside the ratios. Source: `E7/report.md`, opening and “Our offline cost.” **Round-1 item 17 resolved.**

**§4 — Problem 3**

18. **CORRECT — L149–159, E8 diagnostics and endpoints.** Phase A outcomes and **0 GPU-h** match `E8/EXPORT.json → gpu_hours_phase_a,A1,A2,A3,A4`. Raw FC1 medians are **5.810053892%** for b, **8.121514759%** for K1, and **7.545365848%** for c, with **0/8 qualified** for each. Source: `E8/checks/phaseb/score_{b,K1,c}.json → cases[cohort=FC1].{e,qualified}`. Coverage improvement is **1.397838111×**; tail improvement **1.482458600×**; **16384 > 12800** nodes. Both endpoints correctly STOP.

19. **CORRECT — L153–157, L199, E8 rule and guard.** Report now correctly requires stopping when **both improvements fail**, and identifies the guard as a **representation-floor** ratio. Source: `AGENT-PROMPT.md §2`; `E8/EXPORT.json → phase_b.guards.{b,c}.ratio = 0.941709707,1.265312241`. **Round-1 items 20–21 resolved.**

20. **CORRECT — L179, E9.** Reported threshold **113**, flagged **79.2%**, FI **16.7%**, AUC **1**, false alarms **0/0.286**, and **19 bad/5 good** match `E8/EXPORT.json → phase_b.e9`. Exact threshold is **112.774210457**; fractions are **19/24**, **1/6**, and **2/7**. Target definitions now match conditional-error thresholds **0.0038345283** and **0.01**. Both populations miss the **30-good/20-bad** minimum. **Round-1 item 23 resolved.**

21. **CORRECT — L183–186, E10 arithmetic and decisions.** E10-s median recomputes to **8.286403341%**, versus **4.060757379%=K1/2**, with **0/8 qualified** and **0 step-converged**. Source: `E8/EXPORT.json → e10_results.verdicts["E10-s"]`, corroborated by `checks/phaseb/score_e10s.json`. Summing E10-q run states minus initial states gives **17600**; maximum is **0.038633611811**, **0** exceedances. Source: `E8/checks/phaseb/e10q_every_step.json → runs[*].{states,rho_max,n_over}`. The maximum occurs at accepted-step index **82**, supporting the reported scope.

22. **NEEDS-RESTATEMENT — L183, remaining inline qualification.** Report gives **8.29%, 4.06%, 8.12%** without calling them conditional, diagnostic errors locally. Source: `E8/EXPORT.json → e10_results.verdicts["E10-s"]`, including `k1_fc1_diagnostic_median`; lane report’s conditional-reference status. Add “diagnostic, conditional E0 errors; qualification ignored for these medians.” **Round-1 item 3 is substantially, but not completely, resolved.**

**§§5–6 — Stop-rule table and claims**

23. **CORRECT — L190–215, experiment outcomes and retained claims.** E1 pending, E2 unsupported, E3 dual verdict, E4 unrun, E5/E6 fired, E7 FIRES, E8/E10 stopped and E9 inconclusive match the registered rules and lane exports. L213 now correctly says E6 reproduces **E5’s precontracted Galerkin CN reference**, whose error is **0.0185097381**, rather than incumbent LSPG CN-R’s **0.0177526910**. Sources: `E6/EXPORT.json → arms.R.worst`; `E5/checks/f1024.json → stop["G-CN"].worst_all38["384"]`. **Round-1 item 27 resolved.**

**§7 — Retractions and deviations**

24. **CORRECT — L221–241, incident coverage.** The E6 DO-NOT-SUBMIT smoke, E8 admission breach, E3 post-hoc rule, E2 diagnostic-guided geometry, scratchpad/pkill incidents, early release, test reuse, usage-limit stops and credential outage agree with `INTERVENTIONS.md` and lane reports. E2 overrun recomputes to **1.0742−1=0.0742 GPU-h** from `E2/runs/GPU-LEDGER.json → smoke_gpu_hours`; quoted SM-c arithmetic is **12.277≈12.28≤15** from `E8/runs/GPU-LEDGER.json → jobs[name=SM-c].note`.

25. **CORRECT — L237–239, previously omitted corrections.** Report now discloses wall residual **1.03e−13** against **1e−14**, F2’s diffusion-discretisation change, and E7’s corrected bank-fit/resampling accounting. Source: `E8/EXPORT.json → phase_b.armc_wall.{max_rel_wall_rms,registered_bar}`; E6/E7 report correction sections. **Round-1 item 29 resolved.**

26. **NEEDS-RESTATEMENT — L233, chronology.** Report labels the entire correction list **“before any verdict.”** E8’s trained-wall discrepancy was found by the **results audit**, and E7’s resampling correction occurred in **results audit 1**. Sources: `E8/report.md`, arm-c identity disclosure; `E7/report.md → What was wrong or corrected`. Remove the blanket chronology claim; the individual disclosures are supported.

**§8 — Deferred items**

27. **NEEDS-RESTATEMENT — L64, L249, E3 omissions.** Report now includes deferred **C=10 timing/validation** and **CCv L512**, but still omits the **A8.3 fallback for two reference-ineligible test cases** and **reused factorisations**. Sources: `E3/report.md §§2,5`. These remain outstanding parts of **round-1 item 31**.

28. **CORRECT — L247–258, other deferrals.** E2 reduced scope, E4 prerequisite, E5 missing 4096² rank/control, E7 uncosted items and fronts deferrals match lane reports and `E8/runs/GPU-LEDGER.json → deferred`. E10-q is correctly described as resolved without another job.

**§9 — GPU-hour ledger**

29. **CORRECT — L264–273, spending and caps.** Summing individual job entries gives E1 **41.3547222**, E2 **30.6789**, E3 **2.0372**, E5 **1.4076**, E6 **1.4608**, E7 **3.1602**, fronts **10.7443 GPU-h**. Sources: lane `runs/GPU-LEDGER.json → jobs[*].{gpu_hours,elapsed_hours,charged_gpu_h}`. Fronts splits into **10.5451 E8 + 0.1992 E10**; finished lanes total **49.4890**, overall **90.8437**. Caps sum to **204**, excluding E4’s unspent **80**. All displayed rounding agrees.

**§10 — Branches and mirrors**

30. **CORRECT — L279–287, all seven branch/mirror rows.** Read-only `git log -1` reproduces every local hash and timestamp. Local `refs/remotes/origin/codeonly/exp/*` reproduce all seven mirror hashes: **c77739769, b89dbd615, 6a98c018a, ed52f5477, c891ee4cc, 4ade95d16, a2f64ebda**, in table order. `git ls-remote` failed because github.com could not resolve; this is not counted as an error. **Round-1 item 35 resolved under your specified fallback.**

**§11 — Glossary and report hygiene**

31. **CORRECT — L1–27, L289–325, structure and repaired definitions.** The report has a level-1 title, provisional status, Mermaid diagram and final glossary. FN0s now correctly means the **stiff in-range corner**; the previously requested accounting, criteria, CPR/CPU, M, κ, ρ, floor and CI entries were added. Sources: `CLAUDE.md → Reports`; E6/E7/E8 lane glossaries. **Round-1 item 37 resolved; item 38’s enumerated additions are present.**

32. **NEEDS-RESTATEMENT — L301–321, remaining glossary scope.** “Worst/median error” says **over test cases**, although E5/E6 use **dev6+val32**. κ is defined only as quadrature nodes per unknown, although E3 uses it for **diffusion coefficient**. **C**, **CCv**, and the distinction between **F3a/F3b** remain undefined. Sources: `E3/report.md §§2,8`; `E6/report.md → Glossary`; report L82/L103. Define the evaluated cohort generically and explain the overloaded notation and fitting targets.

33. **NEEDS-RESTATEMENT — L117, L299–302, mathematical formatting.** Expressions such as **`m* = gauss96`**, **`> 1`**, **`≥ 2`**, and the τ glossary label remain outside LaTeX. Source: report text against `CLAUDE.md → Reports`. The E10 expression is repaired, and the verbatim SM-c note is now appropriately enclosed in code; **round-1 item 39 is only partly resolved.**

**WRONG: 2, NEEDS-RESTATEMENT: 5 — overall verdict: NOT CLEAN.**