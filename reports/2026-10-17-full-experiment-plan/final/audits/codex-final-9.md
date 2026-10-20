OVERALL: CLEAN

All seven WRONG / NEEDS-RESTATEMENT findings from round 8 are fixed. I found no remaining material numerical or scope defect.

Read-only throughout: no files changed, generator executed, jobs submitted, commits made, or scratch storage used.

Below, `F` denotes the final compilation directory; `R` denotes `worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/`.

| Item | Verdict | Evidence |
|---|---|---|
| Missing D1 verdict rows | **CORRECT — fixed** | Seven rows now appear at `F/report.md:69–75`. Report and export both contain 43 verdicts. Rendering now occurs after verdict collection: `F/make_final.py:639,1195`. |
| C1/C2 verdict terminology | **CORRECT — fixed** | **NOT QUALIFIED ×8**, preserved in both report and export: `F/report.md:71–72`; `F/FINAL-EXPORT.json:354,363`. |
| Finished-lane total | **CORRECT — fixed** | Independently summed nine ledgers: **122.3621 GPU-h**. Report prints 122; export and provenance retain the exact value: `F/report.md:479`; `F/FINAL-EXPORT.json:439`; `F/PROVENANCE.json:3983`. |
| Reference/cohort scope | **CORRECT — fixed** | Fresh regimes cohorts and lane-owned conditional references are distinguished from baseline dev6/val32 E0 references: `F/report.md:3,356`; consistent with `R/DESIGN.md:380`. |
| Floor interpretation | **CORRECT — fixed** | Explicitly diagnostic, not proof of the degradation mechanism; Q1/Q2 failures remain acknowledged: `F/report.md:370`. Matches `R/report.md:239` and `LAB-LOG.md:19319`. |
| Glossary’s one-factor premise | **CORRECT — fixed** | Distributional contrasts, including joint viscosity/steepness changes, are explicit: `F/report.md:548`; consistent with `R/DESIGN.md:308–319`. |
| Glossary’s label thresholds | **CORRECT — fixed** | Correct precedence and interval thresholds: `F/report.md:549`; matches `R/rscore.py:386–397` and `R/report.md:45`. |
| Section 1 regimes sentence | **CORRECT** | C1/C2 each qualify 0/8 primary configurations; uncertified errors and large representation floors are retained: `F/report.md:25`; `R/report.md:17`. |
| Exploratory and selection qualifications | **CORRECT** | Exploratory development evidence, no family-wise control, conditional estimates, cohort-dependent allowances and unknown bias direction remain explicit: `F/report.md:356`; `R/report.md:3–5`. |
| Regime table | **CORRECT** | All displayed coverage counts, qualification counts, labels, median ranges and ratios reproduce: `F/report.md:360–366`. Independent bootstrap reconstruction also matches the lane intervals and labels. |
| GAL diagnostic | **CORRECT** | Exactly two invalid production runs, **C2 c8 and c13**; diagnostic label **FAILS**, with no primary FAILS cells: `F/report.md:368`; `R/DESIGN.md:705`. |
| Spearman statistics | **CORRECT** | Recomputed from stored descriptors and per-case errors: **ρ = 0.882653061–0.927160226**; all permutation p-values **1/10001**. Matches `F/report.md:370`. |
| Representation-floor numbers | **CORRECT** | All nine percentage conversions match lane JSONs: `F/report.md:370`; provenance starts at `F/PROVENANCE.json:9278`. Both steep-regime floors exceed every primary baseline median. |
| A+ confounding | **CORRECT** | Independently recovered **2/14** inside the reference steepness range for A+, **15/16** for A−. Confounded relabel and rejection of isolated viscosity attribution are preserved: `F/report.md:360,372`. |
| Paired A−/C1 contrast | **CORRECT** | Ratios **2.580159333–5.185570567**, signed-rank p **0.000030517578125**, labels **7 DEGRADES / 1 INCONCLUSIVE** reproduce. Interpretation remains amplitude/steepness at matched geometry and viscosity: `F/report.md:374`; `R/report.md:113`. |
| B1/B2/N0s and PS counts | **CORRECT** | B1/B2 each qualify 8/8, neither has DEGRADES; N0s has eight DEGRADES; PS has seven passes among seven scheduled cases: `F/report.md:376–378`. |
| Retractions and pointer | **CORRECT** | A7, A9, A10, A11 and all three design withdrawals are retained: `F/report.md:425,443`; `R/DESIGN.md:644,674,693,705`; `LAB-LOG.md:19318`. |
| Deferred items | **CORRECT** | 3D regimes require their own amendment; FOM comparison remains deferred; wider-steepness retraining remains open: `F/report.md:452`; `R/report.md:19`; `LAB-LOG.md:19319`. |
| Copied figures | **CORRECT** | Both PNGs are byte-identical to lane originals and match provenance hashes. Visual inspection confirms conditional/exploratory labels, hollow unqualified markers and the A+ relabel marker: `F/report.md:380–384`; `F/PROVENANCE.json:9954,9960`. |
| Robust pending handling | **CORRECT** | Explicit pending block at `F/report.md:352`. The handback guard excludes its partial export: `F/make_final.py:48–52,960–975`. No robust `runs/EXPORT.json` source occurs in provenance; operational heartbeat and ledger information remain separate. |
| Non-regimes regression | **CORRECT** | Independently checked **24 numerical entries**—eight kernel exponents and sixteen interval endpoints—against lane JSONs at `F/report.md:87–94`. All match printed precision and CLEAN round-7 commit `ee48d8009`. The report diff contains no changes to the other established scientific tables. |

Independent regime reconstruction:

| Regime | Eligible | Qualified | Median error range (%) | Ratio to baseline |
|---|---:|---:|---:|---:|
| A+ | 14/16 | 8/8 | 0.245772–1.372354 | 0.550091–1.526747 |
| A− | 16/16 | 8/8 | 0.882836–4.165784 | 1.628602–9.247703 |
| C1 | 16/16 | 0/8 | 5.197932–11.176600 | 5.784725–36.674504 |
| C2 | 16/16 | 0/8 | 12.716144–25.110646 | 14.151665–130.714304 |
| B1 | 16/16 | 8/8 | 0.200173–1.153005 | 0.988758–1.549539 |
| B2 | 15/16 | 8/8 | 0.244904–1.404208 | 1.112042–1.641101 |
| N0s | 8/8 | 8/8 | 1.220121–4.222132 | 2.970465–9.793131 |

**Provenance: CORRECT.** All **93 direct regimes records**, **562 direct JSON records overall**, and **63 referenced disk-file hashes** checked successfully. Derived regimes statistics reproduce independently. No hand-typed measured regimes result found; literal configuration counts and thresholds describe the design.

The floor check verifies the stored JSON projection summaries and percentage conversions; it does not rerun projections from large field arrays. This verdict covers the current compilation, not future robust results.