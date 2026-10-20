**NEEDS-FIX at `3d2be8a82`.** Several round-1 fixes are correct, but production evidence binding and reporting still have gaps.

Read-only audit completed. No files changed, jobs submitted, GPU work, or LAB-LOG writes. Synthetic checks used CPU-only Python with writes intercepted in memory. Missing `smoke-accept.json` is expected and is **not** a finding.

Paths below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

**Round-1 verification**

| Finding | Verdict | Evidence |
|---|---|---|
| 1. Planner crash | **CORRECT** | `resources.py:53–59`: `calib_agree` is required and returned. |
| 2. Resource accounting | **CORRECT** | `resources.py:28–42,52–59,67–79,103–114`: rollout populations enumerated; verification measurement required; rounded reservations and ledger commitments participate in reductions. |
| 3. Plan membership/retries | **CORRECT** | `cluster/stage.py:47–58`: plan/certificate/requested runs agree; reused attempt names refused; remaining reservations and time-limit equality checked. |
| 4. Empty baseline | **CORRECT** | `analyze.py:226–232`, `rscore.py:350–379`: no whole-analysis abort; empty denominators yield null ratios and coverage-inconclusive accuracy labels. |
| 5. Production evidence binding | **NEEDS-FIX** | `analyze.py:171–178`: verification-config and field hashes now checked, but “certified” ROM configuration is only the current local config; no certificate or staged configuration is consulted. |
| 6. Markdown columns | **CORRECT** | `make_report.py:78–84`: central escaping handles configuration pipes. |
| 7. Secondary qualifications | **NEEDS-FIX** | `analyze.py:241–245`, `make_report.py:143,164–168`: sensitivity loses annotations; paired labels still omit numerical qualification and allowance exclusions. |
| 8. Publication omissions | **NEEDS-FIX** | Many outputs now appear (`make_report.py:129–191,223–230`), but denominator coverage, complete Q2 explanations, and scheduled-diagnostic completeness remain deficient. |
| 9. Reduced populations/glossary | **NEEDS-FIX** | Deferred cases are now recorded, printed and exported (`analyze.py:180–186`; `make_report.py:115–117,224`). Cohort sizes are generated. The glossary remains incomplete (`make_report.py:202–220`). |

**Remaining findings**

1. **WRONG — N0-only sensitivity prints false allowance counts and loses separation relabelling.**  
   `analyze.py:242–244` attaches `unresolved_under_allowance` and `regime_label` only to primary cells. `make_report.py:95` converts the missing sensitivity annotation into **zero**, and line 143 prints it.

   An in-memory fixture reproduced primary output with **3 unresolved** and a confounding label, while sensitivity printed **0 unresolved** without that label. Carry both annotations into sensitivity cells; missing counts must not become zero. The same incomplete cells enter `EXPORT.json` at line 228.

2. **WRONG — paired results can say HOLDS despite failed numerical qualification.**  
   `analyze.py:117–134,245` computes paired accuracy without qualification context. `make_report.py:166–168` adds separation labels but omits both cohorts’ qualification status, invalid-run counts and allowance exclusions.

   Eight identical usable pairs reproduced ratio **1**, interval **[1,1]**, Wilcoxon **p=1**, and **HOLDS**, even alongside failing Q1/Q2 cells. Preserve the paired accuracy statistic, but print/export its qualification and exclusion context beside it.

3. **NEEDS-FIX — ratio denominator coverage is not adequately recorded or reported.**  
   `analyze.py:98–105` stores numerator coverage only. Primary baseline usable count exists at line 230 but is not printed beside ratios. N0-only denominator coverage is not persisted. The consistency check at line 232 stores only ratio/interval/dropped replicates, without either population count or a coverage verdict.

   Consequently, `make_report.py:135` can display an apparently ordinary consistency interval based on fewer than eight usable cases. Record both sides’ usable counts and exclusions, apply the coverage qualification to consistency, and print denominator coverage beside each ratio.

4. **NEEDS-FIX — analysis does not establish certified production configuration identity.**  
   `analyze.py:171–173` compares `result.json` with mutable `configs/rom1024.json`, without checking `resources-rom1024.json`, plan membership, or the staged committed configuration.

   A truncated result and identically truncated local configuration can still redefine the expected inventory. The new `fverify` bindings correctly establish which arrays were verified; they do not establish that those arrays cover the certified workload. Bind analysis to the certificate and staged configuration.

5. **NEEDS-FIX — scheduled diagnostics can disappear silently.**  
   `analyze.py:205–209,263–277` ingests whatever production probes and PS files happen to exist. It never derives their expected inventory from the plan/configurations. Missing scheduled PS outputs therefore produce `{}`, indistinguishable in the report from a deliberately deferred PS job. Partial collections produce partial results without a completeness warning.

   Require every scheduled diagnostic or explicitly mark it missing/incomplete. Record deliberate PS deferral separately. Apply equivalent inventory checks to conditional calibration probes.

6. **NEEDS-FIX — publication detail remains incomplete.**  
   `make_report.py:175–184` prints Q1 failing predicates, but Q2 output omits check-rule distance and explicit successor/check validity predicates. `rscore.py:179–180` also labels an invalid **check** replay as “successor invalid.”  
   The absolute-error table (`make_report.py:133–137`) omits the affected cells’ qualification and separation annotations. The glossary (`202–220`) still leaves displayed terms such as OLS slope, successor/check rule, non-stationary steps, and arm identifiers unexplained.

**Other scope checks**

| Item | Verdict | Evidence |
|---|---|---|
| Full/reduced populations | **CORRECT** | `make_configs.py:94–108`, `rscore.py:460–479`: reproduced **158/150 cases per arm**, with **38 tighter cases** in both. |
| PS selection and allowances | **CORRECT** | `make_configs.py:97–111`: largest configured \(s\); A+ c7; reduced C2 c2; only full A+ receives allowance 1. |
| Conditional probes | **CORRECT configuration/costing** | `make_configs.py:99–100`, `resources.py:70,98–104`: case-zero probes added when calibration disagrees. |
| Reduction order/cap | **CORRECT** | Synthetic plans exercised PS removal, then reduced C2, then mandatory-remainder refusal—including refusal below the projection limit when reservations exceed the cap. |
| Reference routing | **CORRECT** | `analyze.py:44–54`, `rscore.py:95–123`: configured union, separate E0/lane routing, identity checks; missing references fail closed. |
| Primary labels/exclusions | **CORRECT** | `analyze.py:88–114`: validity and qualification precedence; eligible, valid, finite positive-error filtering. |
| Paired statistical calculation | **CORRECT** | `analyze.py:117–134`: complete pairs, pair bootstrap, Wilcoxon exact/approximate selection and all-zero handling. Qualification presentation remains finding 2. |
| Trends/separation/floors | **CORRECT core calculations** | `rscore.py:389–457`: tie-aware ranks, degenerate handling, amended separation thresholds and lattice least-squares projection. |
| Numerical traceability | **CORRECT for displayed result values inspected** | Report tables consume analysis fields; expenditure consumes the ledger. Remaining problems concern missing or misleading qualifications, rather than invented result numbers. |

**Production readiness remains blocked by findings 1–6.**