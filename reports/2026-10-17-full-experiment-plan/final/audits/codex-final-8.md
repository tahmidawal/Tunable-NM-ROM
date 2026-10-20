OVERALL: NOT-CLEAN

The regimes numbers reproduce, but the compilation has missing verdict rows, an incorrect finished-lane total, and several scope/definition regressions.

Read-only audit completed: no files changed, no jobs submitted, no generator executed, and no scratch storage used.

Below, `F` denotes [the final compilation directory](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final), and `R` denotes [the regimes lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

| Item | Verdict | Evidence and finding |
|---|---|---|
| D1 rows in the hypothesis table | **WRONG** | **All seven rows are missing from `report.md`.** Its table contains 36 verdicts; `FINAL-EXPORT.json` contains 43. The table is rendered at `F/make_final.py:634`, before `render_regimes` appends D1 verdicts at `:1327`. Exported D1 rows begin at `F/FINAL-EXPORT.json:333`. |
| C1/C2 exported verdict names | **NEEDS-RESTATEMENT** | `F/FINAL-EXPORT.json:354,363` says **“fail (not numerically qualified)”**. Preserve **NOT QUALIFIED**, as required by `R/report.md:17,45`; FAILS denotes invalid production runs. The conversion originates at `F/make_final.py:1325`. Section 3.12 itself preserves the distinction correctly. |
| Finished-lane GPU total | **WRONG** | `F/report.md:472` still says **eight finished lanes, 107 GPU-h**, although regimes is marked finished at `:468`. The nine finished lanes total **122.3621 GPU-h**, including regimes’ **14.865**. `F/make_final.py:549` excludes both priority-4 lanes using the obsolete `PENDING_LANES` list. The same error appears in `F/PROVENANCE.json:3983` and `F/FINAL-EXPORT.json:439`. The all-lane total, **149.8933 GPU-h**, is correct. |
| Introductory reference/cohort scope | **NEEDS-RESTATEMENT** | `F/report.md:3` lists only the original cohorts and “two” reference exceptions, omitting D1’s fresh development cohorts and lane-owned references. Section 3.12 acknowledges those at `:349`, but its “every error … against new references” also needs to distinguish the baseline’s existing dev6/val32 E0 references. Routing is explicit at `R/DESIGN.md:380`. |
| Section 1 regimes sentence | **CORRECT** | `F/report.md:25`: C1 and C2 each qualify **0/8** primary configurations. The inability to certify their errors agrees with `R/report.md:17` and `LAB-LOG.md:19312`. |
| Regime table and diagnostic row | **CORRECT** | Every displayed count, median range and ratio range at `F/report.md:353–361` matches independent calculations. The diagnostic GAL row has exactly two invalid C2 runs, **c8 and c13**, and is FAILS. |
| Exploratory/selection qualifications | **CORRECT** | `F/report.md:349` retains exploratory development status, no family-wise control, conditional estimates, cohort-dependent tight-path allowances and unknown selection-bias direction; consistent with `R/report.md:3–5`. |
| A+ confounding | **CORRECT** | `F/report.md:353,365` preserves the confounded relabel and explicitly rejects isolated viscosity attribution. Separation fractions reproduce as **2/14** for A+ and **15/16** for A−. |
| Spearman statistics | **CORRECT** | `F/report.md:363`: independently recomputed correlations span **0.882653061–0.927160226**; all permutation p-values are **1/10001**. Printed values match. |
| Representation-floor numbers | **CORRECT** | All nine percentages at `F/report.md:363` match `checks/analysis.json` and the export. C1/C2 floors exceed every primary baseline median error. |
| Representation-floor interpretation | **NEEDS-RESTATEMENT** | `F/report.md:363` adds “the bank’s span, **not the solver or the quadrature**, is the binding limit.” The lane explicitly calls its floor a diagnostic, **“not a proof of why the ROM degrades”** (`R/report.md:239`). The coordinator says floors *suggest* a limit (`LAB-LOG.md:19319`). Retain that inference without excluding solver/quadrature contributions, particularly where Q1/Q2 fail. |
| Paired contrast | **CORRECT** | `F/report.md:367`: median paired ratios, signed-rank p-values and label counts reproduce. The amplitude/steepness interpretation at matched geometry and viscosity, excluding a pure-stiffness interpretation, matches `R/DESIGN.md:313` and `R/report.md:113`. |
| B1/B2/N0s and PS counts | **CORRECT** | `F/report.md:369–371`: B1/B2 each qualify **8/8**, neither has a DEGRADES label; N0s has **8 DEGRADES**; PS has **7 PASS out of 7 scheduled**. |
| Retractions and source pointer | **CORRECT** | `F/report.md:418,436` accurately carries A7, A9, A10, A11 and the three design withdrawals. The pointer correctly identifies `R/report.md:35–42`; amendments agree at `R/DESIGN.md:644,674,693,705` and `LAB-LOG.md:19318`. |
| Deferred items | **CORRECT** | `F/report.md:445` retains 3D regimes requiring an amendment, the FOM accuracy comparison, and wider-steepness retraining as open; matches `R/report.md:19` and `LAB-LOG.md:19319`. |
| New glossary: attribution | **WRONG** | `F/report.md:541` defines regimes as differing in **“one declared way.”** This revives the withdrawn one-factor premise. `R/DESIGN.md:308–319` explicitly establishes distributional contrasts and says C1/C2 change both viscosity and steepness. |
| New glossary: label rules | **NEEDS-RESTATEMENT** | `F/report.md:542` describes “inside / below / above” a factor-2 margin, which does not state the implemented rule correctly. `R/rscore.py:386–397`: IMPROVES if interval upper endpoint **<1**; otherwise HOLDS if **≤2**; DEGRADES if lower endpoint **>2**; otherwise INCONCLUSIVE. Invalid-run FAILS takes precedence over NOT QUALIFIED (`R/report.md:45`). |
| Copied regimes figures | **CORRECT** | Both PNGs are byte-identical to the lane originals and match registered hashes at `F/PROVENANCE.json:9947,9953`. Visual inspection confirms exploratory/conditional labels, hollow unqualified markers and the A+ relabel marker. |
| Robust pending handling | **CORRECT** | `F/make_final.py:48–52,958–969` prevents reading its partial export and renders the pending block at `F/report.md:345`. Its provenance references only heartbeat, ledger and git metadata—not `runs/EXPORT.json`. |
| Non-regimes regression check | **CORRECT**, except finished-total accounting above | Compared against CLEAN commit `ee48d8009`. Independently checked **24 non-regimes numerical entries**: eight kernel exponents and sixteen interval endpoints at `F/report.md:80–87`. All match lane JSONs and round 7. Other pre-existing scientific prose/tables are unchanged in the report diff. |

Independent numerical reconstruction, without importing `make_final.py`:

| Regime | Eligible references | Qualified primary configurations | Median error range (%) | Ratio to baseline |
|---|---:|---:|---:|---:|
| A+ | 14/16 | 8/8 | 0.245772–1.372354 | 0.550091–1.526747 |
| A− | 16/16 | 8/8 | 0.882836–4.165784 | 1.628602–9.247703 |
| C1 | 16/16 | 0/8 | 5.197932–11.176600 | 5.784725–36.674504 |
| C2 | 16/16 | 0/8 | 12.716144–25.110646 | 14.151665–130.714304 |
| B1 | 16/16 | 8/8 | 0.200173–1.153005 | 0.988758–1.549539 |
| B2 | 15/16 | 8/8 | 0.244904–1.404208 | 1.112042–1.641101 |
| N0s | 8/8 | 8/8 | 1.220121–4.222132 | 2.970465–9.793131 |

These calculations used per-case errors from `R/checks/analysis.json`, independently resampled bootstrap intervals, and stored reference descriptors for the Spearman calculation. Paired ratios reproduce as **2.580159333–5.185570567**, with **p = 0.000030517578125**, **7 DEGRADES / 1 INCONCLUSIVE**.

The nine floor percentages independently read and converted from JSON are **0.0265922, 0.0286480, 0.8810801, 0.00520255, 0.5853618, 3.4705439, 9.8818125, 0.0519053, 0.0593031**, in the report’s order. This verifies the compilation’s floor values; it does not recompute projections from large field arrays.

**Provenance verdict: CORRECT for the regimes results.** All 93 direct regimes records match their JSON values; the regimes source paths and hashes validate, and the derived statistics reproduce. Across the compilation, 562 direct JSON records and 63 referenced disk-file hashes checked successfully. I found no hand-typed measured regimes result. The introductory configuration counts are literal design descriptions; the numerical findings are sourced. The incorrect finished-total provenance faithfully records an incorrect population selection, so its valid hash does not rescue that result.