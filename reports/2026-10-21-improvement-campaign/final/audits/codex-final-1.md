**NOT CLEAN.** Numerical transcription and arithmetic pass; several statements misidentify quantities or registered rules. No files were changed.

Locations below refer to [final/report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/final/report.md). Source shorthand `E5/EXPORT.json`, for example, means `worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/EXPORT.json`; the other experiment aliases follow the lane mapping in your request.

**Opening and §1 — Summary**

1. **CORRECT — L3, provenance throughout.** All **202 file-backed entries across 17 source files** match both their JSON key values and SHA-256 hashes. All **28 derived entries** recompute correctly; all **11 figures** match their source and copied-file hashes. Source: `final/PROVENANCE.json`, each entry’s `file`, `key_path`, `sha256`, or `derived_from`. No invented measured scalar was found.

2. **CORRECT — L31, per-problem conclusions.** Report: none of the completed experiments fixes the three weaknesses, with E1 pending. Recomputed: E2 unsupported, E3 FAIL/UNRESOLVED, E5–E8 negative, E9 inconclusive, E10 stopped. Sources: respective `EXPORT.json` verdict keys detailed below. These support the scoped conclusions, despite the wording errors identified below.

3. **NEEDS-RESTATEMENT — L31, L43–64, L103–117, L153–184.** Report: conditional-reference qualifications appear principally in the preamble and retrospective caveat list; several numerical tables omit them locally. Required: label the E2/E3 reference-dependent comparisons, E6 errors, and E8/E10 **diagnostic, unqualified** errors beside those results. Sources: lane reports’ status/error captions; E8 `EXPORT.json → phase_b.score_c.FC1.median_e_diagnostic`; `CLAUDE.md:151`. The global preamble does not satisfy the explicit inline-label requirement.

**§2 — Problem 1**

4. **CORRECT — E1, L37.** Report: pending; **41.35/87 GPU-h**. Source values: E1 `EXPORT.json → stage = "interim"`; `campaign/LANES.json → lanes[lane=imp-box].status = "active"` and `.gpu_hours_cap = 87`; E1 `runs/GPU-LEDGER.json → total_gpu_hours = 41.3547`. Pending treatment is appropriate.

5. **CORRECT — E2 table, L45–46.** Reported ranges recompute as follows:
   - L validation **0.0147–0.214**, test **0.0128–0.0903**, **2** empty; SEM **0.0856–0.108**; FD+MG **0.471/1.36**.
   - H validation **0.0292–0.216**, test **0.0306–0.120**, **1** empty; SEM **0.0865–0.234**; FD+MG **0.653/1.77**.
   
   Sources: E2 `EXPORT.json → cells["{L,H}|wide|S1"].{S_val,S_test,per_family_val}`, selecting `sem` at worst-error targets and `fdmg` at `0.03|worst`/`0.1|worst`. All rounding, extrema, null counts, competitor names and decision labels match.

6. **CORRECT — E2 verdict, L48.** Report: H withdrawn; L descriptive, **4/32** references ineligible; S2 untested. Recomputed: L `.allowance.n_ineligible = 4`, `.n_eligible = 28`; H has zero ineligible; `.geometry_labels` and `.A10` match the prose. Source: E2 `EXPORT.json`. The measured maximum validation speed ratio is **0.2160999332 < 2**.

7. **WRONG — E2, L48: “The bank beats FD+MG only at the loosest target.”** Counterexample: **1.120718095 > 1 at the 3% median-error target** on H. Source: E2 `EXPORT.json → cells["H|wide|S1"].per_family_val["0.03|median"].fdmg`. The sentence is true only when restricted to **worst-error matching**.

8. **NEEDS-RESTATEMENT — E2, L41–48.** Report omits the asymmetric competitor admission rule. Source: E2 `report.md §5` and `DESIGN.md A4.2`: competitors are admitted by validity and accuracy; their Q1 is reported, not gated. State this alongside the comparative result because the bank and competitors do not face identical qualification requirements.

9. **CORRECT — E3, L60–64.** Reported **2.71%/6.73%**, **0.859/0.923**, and intervals **[0.799,0.900]/[0.877,0.943]** match sources. Independently dividing anchor-normalised costs gives:
   - `0.280741898228017 / 0.326956209414297 = 0.8586529025735055`;
   - `0.1364990241029282 / 0.14785867018314236 = 0.9231722693965544`.
   
   Source: E3 `checks/gate.json → targets["0.03|C100" / "0.1|C100"].{fom.cost,rom.cost,rom.worst,G_ci95}`. Winner identifiers match; the 1% target has no qualified ROM.

10. **CORRECT — E3/E4, L64–76.** Report: E3 **FAIL under A10 / UNRESOLVED under the registered rule**, no ROM trained; E4 not run. Sources: E3 `EXPORT.json → gate`, `gate_registered_rule.outcome`; `AGENT-PROMPT.md §2`; `INTERVENTIONS.md` E3 ruling and final E2 entry. Both E3 readings block training, and E4’s prerequisite was unmet.

**§3 — Problem 2**

11. **CORRECT — E5 table, L86–93.** In table order, recomputed worst errors (%) are **1.850974, 1.815248, 1.908120, 1.926565; 1.850320, 1.814506, 1.907564, 1.925850**. Incumbents are **1.775269%/1.774683%**. Evolution ratios are **0.726166, 14.277634, 22.849436, 1.583933; 0.732675, 14.974450, 24.149574, 1.592867**. Sources: E5 `checks/f{1024,4096}.json → stop[arm].worst_all38["384"]` and `.speed.statistic.per_case` (median). All displayed rounding, **36/2 or 34/4** counts, qualification labels and fired conditions match.

12. **CORRECT — E5, L95.** Reported POD ratios **0.623, 0.796, 0.0363** recompute to **0.623042485, 0.796270139, 0.036340605**. Source: E5 `EXPORT.json → verdicts["f1024|matches"]["G-E4a|384" / "G-E4b|384" / "L-CNR|384"].direct.value`. Frozen selections are respectively `None` and `G-E4a|384`; the latter fails on all 38 cases. The evolution range **14.3–24.1×** is also correct.

13. **NEEDS-RESTATEMENT — E5, L95 and claims L213.** Report presents the matched POD comparisons without their **HR-unqualified** label. Source: E5 `checks/f1024.json → pod["pod-ecsw|384|k4|ETDRK4|0.0125" / "...|0.025"].Q2.label = "HR-unqualified"`; Q1 passes. The ratios are valid reported comparisons, but the omitted successor-rule failure materially qualifies their interpretation.

14. **CORRECT — E6 table, L107–115.** Recomputing maxima from E6 `checks/analysis-prd1.json → scores[arm][case].e` reproduces all displayed errors: **1.851, 2.131, 1.859, 1.889, 1.851, 5.951, 1.851, 1.853, 2.131%**. E6 `EXPORT.json → arms[arm].{m,e2e,q2}` reproduces all node counts, gains and Q2 labels, including **6144 nodes, 1.069997960×** for qualified F2F3_k16.

15. **CORRECT — E6 stop rules and microbenchmark, L117.** Report: both rules fire, **0/6** F3 arms pass Q2, best gain **1.24×**, Jacobian ratios **1.37×/2.14×**. Sources: E6 `EXPORT.json → stop`, `arms`, `jac_bench.{dup_ratio,quad_ratio}` give **1.241094885**, **1.373800808**, **2.138603559**. F2’s **1.022153×** does not meet the registered “faster” predicate: its bootstrap lower bound is **0.992452 < 1** (`checks/analysis-prd1.json → arms.F2_gauss96.timing.e2e_ci`).

16. **CORRECT — E7, L129–139.** Independently counting `checks/analysis.json → verdicts[criterion|accounting].table[workload][configuration]` gives **21×6 = 126** pairs: MATCH counts **126,126,126,84,0**; NO-MATCH counts **0,0,0,42,126**. Every NO-MATCH lacks an admissible competitor. Dividing `.best.cost / .C_ours` gives min/median/max **0.099119332/0.104847227/0.130716684**. A100 and no-head variants each retain **126** matches. The primary rule fires.

17. **NEEDS-RESTATEMENT — E7, L139.** Report qualifies compilation projections but omits that **bank-fit cost itself is projected**. Source: E7 `report.md`, opening and “Our offline cost”: H200 bank-fit cost is a phase-wise projection from measured training rates, not a complete measured fit. Add this qualification beside **0.0991–0.131**, rather than describing the costs simply as “measured items.”

**§4 — Problem 3**

18. **CORRECT — E8 Phase A, L149.** Report: **0 GPU-h**, A1 inconclusive, A2 RUN, A3 CPU screen passed/timing deferred, A4 STOP F15. Sources: E8 `EXPORT.json → gpu_hours_phase_a`, `A1.verdict`, `A2.decision_arm_c`, `A3.verdict`, `A4.verdict`. All match.

19. **CORRECT — E8 numerical endpoints, L156–157.** Raw case medians reproduce **5.8100539%**, K1 **8.1215148%**, and arm-c **7.5453658%**, with **0/8** qualified FC1 cases for each bank. Sources: E8 `checks/phaseb/score_{b,K1,c}.json → cases[cohort=FC1].{e,qualified}`. Coverage improvement is **1.397838111×**; tail improvement is **0.0003739746680 / 0.0002522665173 = 1.482458600×**; nodes are **16384 > 12800**. Source: E8 `EXPORT.json → phase_b.routes`. Both arms correctly stop.

20. **WRONG — E8 registered rule, L157 and L199.** Report requires **“m* ≥2× smaller AND tail ≥3× smaller.”** Registered rule stops only when **both improvements fail**, so either qualifying improvement can permit continuation. Sources: `AGENT-PROMPT.md §2`; E8 `DESIGN.md §B0`, explicitly “continues if either.” The actual STOP remains correct because both routes fail.

21. **WRONG — E8 guard, L153 and glossary L309.** Report calls this an in-range worst-**error** guard; printed ratios **0.94/1.27** are worst-**representation-floor** ratios. Source: E8 `EXPORT.json → phase_b.guards.{b,c}`: **0.941709707/1.265312241**. Actual diagnostic ROM worst-error ratios from `score_{b,c,K1}.json → cases[cohort=FI].e` are **1.232094547/1.410821728**. Replace “error” with “floor”; both registered guards still pass.

22. **CORRECT — E9, L179.** Reported threshold **113**, flagged **79.2%**, FI **16.7%**, AUC **1**, detection **1**, false alarms **0/0.286**, and **19 bad/5 good** match E8 `EXPORT.json → phase_b.e9`. Exact threshold is **112.774210457**; flagged fraction **19/24**; FI **1/6**; target-B false alarms **2/7**. Both populations fall below the **30-good/20-bad** minimum, so “inconclusive” is correct.

23. **NEEDS-RESTATEMENT — E9, L179.** “Target A” and “target B” are undefined. Source: E8 `EXPORT.json → phase_b.e9.targets.{A,B}.threshold`: **0.003834528307579255** and **0.01**, respectively. Define them as conditional-error thresholds **0.38345% and 1%** so the detection/false-alarm results are interpretable.

24. **CORRECT — E10, L183–186.** Recomputed E10-s median **8.2864033%**, bar **4.0607574% = K1/2**, qualified **0/8**, step-converged **0**. Sources: E8 `EXPORT.json → e10_results.verdicts["E10-s"]` and raw `score_e10s.json`. E10-q raw run summaries give **17600 accepted steps + 171 initial states**, maximum **0.0386336118**, zero threshold exceedances; the maximum occurs at step index **82**, so quoting it over accepted steps is valid. Source: `checks/phaseb/e10q_every_step.json → runs[*]`. STOP without another job is supported.

**§§5–6 — Stop-rule table and claims**

25. **CORRECT — L192–201 and L207–215, overall outcomes.** The experiment outcomes and three problem verdicts match the exports and registered prerequisites. E1 remains pending; E4 unrun; E9 descriptive. Withdrawing comparative multi-mesh advantage while retaining mesh-independent operation as a property follows from E7.

26. **NEEDS-RESTATEMENT — Diagram L21–27.** Red means “registered stop rule fired,” but E3 is **UNRESOLVED under the registered rule** and FAIL under post-hoc A10; E2 also includes descriptive L results. Sources: E3 `EXPORT.json → gate_registered_rule.outcome`; E2 `.geometry_labels`. Change the legend to describe negative/unsupported outcomes under the reported rules, or distinguish these statuses visually.

27. **WRONG — L213: “E5/E6 reference arms reproduce” the incumbent configuration and accuracy.** E6’s reference R is **precontracted sine-tested Galerkin CN**, not incumbent **LSPG CN-R**. Sources: E6 `report.md` status/glossary R; E6 `EXPORT.json → arms.R.worst = 0.0185097381`, versus E5 incumbent **0.0177526910** at 1024². State that E6 reproduces E5’s Galerkin reference arm.

**§7 — Retractions and deviations**

28. **CORRECT — L221–238, listed incidents.** The E6 audit breach, E8 SM-c breach, E3 post-hoc rule, E2 diagnostic-guided redesign and smoke overrun, E5/E7 scratchpad collision, possible pkill collateral damage, early E8 release, test reuse, usage stops and credential outage are supported by `INTERVENTIONS.md` and the cited lane records. E2 overrun recomputes to **1.0742−1 = 0.0742 GPU-h**; SM-c’s quoted arithmetic is **6.327+0.45+5.5 = 12.277 ≈ 12.28 ≤ 15**.

29. **NEEDS-RESTATEMENT — L233–240, omitted material qualifications/corrections.** Add:
   - E8 trained arm-c wall residual exceeded the registered numerical identity tolerance: **1.0312413×10⁻¹³** maximum versus **10⁻¹⁴**; passing identity evidence concerned the initial bank, and scoring zeroes walls. Source: E8 `checks/phaseb/armc_wall.json → max_rel_wall_rms`; lane report’s arm-c identity disclosure.
   - E6 F2 changes **both test space and diffusion discretisation**. Source: E6 `report.md`, “What is provisional…”.
   - E7 corrected its single-rate bank-fit model to a phase-wise projection and corrected resampling-cost accounting. Source: E7 `report.md`, “What was wrong or corrected”.
   
   These affect interpretation and are absent from the consolidated deviations section.

**§8 — Deferred items**

30. **CORRECT — L244–255, listed deferrals.** E2 reduced scope, E4 prerequisite, E5 missing 4096² rank/control, E7 uncosted items, and E8/E9/E10 deferrals agree with lane reports and E8 `runs/GPU-LEDGER.json → deferred`. E10-q is explicitly resolved rather than awaiting a necessary experiment.

31. **NEEDS-RESTATEMENT — E3, L54–64 and L246.** Omitted: the winning 10% ETD configuration reaches **11.7% worst error on test**, and the fallback for **2 reference-ineligible cases out of 16** was deferred. Also omitted are C10 timing/validation, CCv L512 and reused-factorisation deferrals. Sources: E3 `EXPORT.json → test["etd|N48|0.0125|mid@C100"].worst`; E3 `report.md §§2,5`. Label the gate as a development/validation result and preserve these limitations.

**§9 — GPU-hour ledger**

32. **CORRECT — L261–268 and repeated lane totals.** Summing individual ledger jobs gives E1 **41.3547222**, E2 **30.6789**, E3 **2.0372**, E5 **1.4076**, E6 **1.4608**, E7 **3.1602**, fronts **10.7443 GPU-h**. Sources: each lane’s `runs/GPU-LEDGER.json → jobs[*].{gpu_hours,elapsed_hours,charged_gpu_h}`. Fronts splits into **10.5451 E8 + 0.1992 E10**. Finished lanes total **49.4890**; adding the recorded E1 subtotal gives **90.8437**. All displayed two-decimal totals match.

33. **CORRECT — L268–270, caps.** **87+62+15+4+3+8+25 = 204 GPU-h**; E4’s separate **80 GPU-h** reserve was unspent. Source: `campaign/LANES.json → lanes[*].gpu_hours_cap`, `planned[0].gpu_hours_cap/status`. No E4 reserve is incorrectly included in spending.

**§10 — Branches and mirrors**

34. **CORRECT — L276–282, local branches.** Read-only `git log -1` reproduces all seven reported hashes and timestamps: **576ad5677, 0dc801d2b, e8d39fa1a, a3e417b1e, cf351b62a, dd4cd7fc4, 506dbd3a5**, in table order. Lane statuses match `campaign/LANES.json`.

35. **NEEDS-RESTATEMENT — L276–282, live GitHub verification incomplete.** All reported mirror hashes match local `refs/remotes/origin/codeonly/exp/...`. However, the requested `git ls-remote origin 'refs/heads/codeonly/exp/2026-10-2*-imp-*'` failed with **“Could not resolve host: github.com.”** These are locally corroborated mirror values, not independently verified current remote heads in this audit.

**§11 — Glossary and report hygiene**

36. **CORRECT — L1–3, L5–25, L286–314.** The report has a level-1 title, explicit provisional status, a colour-coded Mermaid diagram, and a glossary at the end. Source: the report itself; structural requirements in `CLAUDE.md → Reports`.

37. **WRONG — L305, FN0s definition.** Report: **“noise-free steep.”** Actual: **stiff corner inside the training range**, with low viscosity and high amplitude. Source: E8 `DESIGN.md` cohort table, row `FN0s (stiff corner)`, and lane glossary’s N0s definition. The printed expansion invents a different cohort meaning.

38. **NEEDS-RESTATEMENT — L286–314, glossary completeness.** Missing definitions include **A_dep, worst_eps, worst_median, casewise, rollout, M, κ, ρ, F3a/F3b, representation floor, CI**, and the distinct meanings of CPR versus CPU. Sources: E6/E7 lane glossaries and E7 `report.md → Criteria (A2.1)`. These are necessary to interpret the report’s tables; CLAUDE.md requires every table column and term of art to be defined.

39. **NEEDS-RESTATEMENT — L117, L184, L222 and glossary inequalities.** Mathematical prose includes unformatted expressions such as **“max ρ = 0.0386”** and **“6.327 + … = 12.28 <= 15.”** Source: report text; `CLAUDE.md:152–156`. Render mathematical expressions in LaTeX; retain code identifiers as code.

WRONG: 5, NEEDS-RESTATEMENT: 12 — overall verdict: NOT CLEAN.