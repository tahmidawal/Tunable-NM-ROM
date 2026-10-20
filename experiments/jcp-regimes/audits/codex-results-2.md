**A — RESULTS-CONFIRMED.**  
**B — NOT-READY.**

Audited commit `1ed8a8779`. No repository files changed, no jobs submitted, no lab-log writes. Python checks were CPU-only; rendering was intercepted in memory.

**A. Results audit, round 2**

1. **CORRECT — A11 implements §7.1.** [analyze.py:81](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/analyze.py:81) invalidates production runs containing reason `0` budget exits. Q1 retains its separately disclosed predicate.

2. **CORRECT — GAL/C2 is FAILS, with two invalid runs and 14/16 used.** Scanning all nine raw production files found budget exits only in GAL/C2: c8 has seven; c13 has three. Independently decoding the remaining cases reproduced:
   
   | Quantity | Recomputed |
   |---|---:|
   | Median error | 21.4361901% |
   | Worst error | 66.4917028% |
   | Median ratio against I | 433.246544 |
   | Bootstrap interval | [237.114966, 676.581218] |

   These agree with [report.md:52](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:52) and line 66. Raw-error agreement used relative tolerance `1e−11`, absolute `1e−14`.

3. **CORRECT — no unexpected movement from round 1.** Comparing the complete analysis against `f1af0ba81`, excluding newly added `budget_exits` metadata, found exactly 26 changed fields. All belong to the two GAL/C2 exclusions, their two baseline comparisons, or the pooled A−/C1/C2 trend. That trend now uses 46 cases instead of 48. Every other existing field—including all primary LSPG results and the round-1 recomputation targets—is unchanged. Both corrected baseline comparisons and the trend reproduce.

**B. Report audit, round 1**

1. **CORRECT — title and status.** [report.md:1](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:1) supplies the level-1 title; line 3 identifies conditional E0 estimates and exploratory development evidence without family-wise control.

2. **NEEDS-FIX — “every number generated” is only partly satisfied.** All measured results reproduce, and `report.md` and `EXPORT.json` regenerate byte-for-byte. However, [make_report.py:174](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/make_report.py:174) hard-codes the mesh and configuration count; lines 299–318 hard-code protocol quantities, including resampling counts, training bounds and thresholds. Read these from configuration/constants rather than embedding numerical literals in prose. No numerical mismatch was found.

3. **NEEDS-FIX — qualifications are not consistently adjacent.** Detailed label tables correctly carry coverage, unresolved counts and A+ relabelling. But the [summary at report.md:9](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:9) omits case coverage/unresolved counts; later numerical sections rely on the opening conditional/exploratory disclaimer. Add local qualification captions. Explicitly print **two invalid runs** beside GAL/C2’s 14/16 coverage.

4. **NEEDS-FIX — mathematical formatting.** [report.md:96](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:96) uses `x_ps` and `2 eps_ref`; glossary lines 205–217 use `s = a/ν`, `log10(error)` and `Δt0 = 0.005` outside LaTeX. Convert mathematical expressions to `$…$`.

5. **CORRECT — Mermaid diagram.** [report.md:20](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:20) contains the requested flow diagram with role colours.

6. **NEEDS-FIX — glossary is incomplete.** [report.md:194](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:194) does not define the actual error metric, ROM/FOM, LSPG/GAL, E0, IC, Péclet, Richardson estimate, certificate, Q2’s distinct `ρ max`, or separation fields such as `frac_above_q90`. Explain these in plain language and define table populations and units.

7. **CORRECT — deferrals and corrections disclosed.** [report.md:18](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:18) explicitly defers 3D regimes and the FOM comparison. Lines 36–39 disclose A7, A9, A10 and A11.

8. **NEEDS-FIX — make the failure conclusion plain.** The evidence is accurately presented, including the stiff **in-range N0s** degradation at line 15. Representation floors are labelled diagnostic at line 134; no causal proof is asserted. Nevertheless, [summary lines 11–12](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:11) should plainly state that the frozen method fails numerical qualification on both tested steep, low-viscosity regimes outside training. Preserve the distinction between this conclusion and the formal invalid-run label **FAILS**.

9. **NEEDS-FIX — plots reproduce and are readable, but lack essential context.** All four PNGs regenerated byte-for-byte. [make_report.py:55](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/make_report.py:55) plots the correct numerical data. However:
   - C1/C2’s unqualified points are visually indistinguishable from qualified results.
   - Scatter plots omit conditional/exploratory and selection qualifications.
   - A+’s steepness confounding is absent.
   - The steepness plot’s dashed training boundary is unexplained.
   - The sigma plot includes sampled reference-ineligible cases, whereas separation statistics use eligible cases. Identify that population difference explicitly.

10. **CORRECT — EXPORT numerical consistency.** [EXPORT.json:5078](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/runs/EXPORT.json:5078) records two invalid runs; line 5115 records FAILS. Exported summaries and qualifications agree with the analysis and report.

**Required before REPORT-READY:** resolve items **2, 3, 4, 6, 8 and 9** in the generator and regenerate the artifacts. No new experiment is required.