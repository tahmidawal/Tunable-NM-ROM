**NEEDS-FIX at `f08c6fb31`.** Most round-2 fixes are resolved, but diagnostic completeness and some qualification/reporting paths remain incorrect.

Read-only audit completed. No files changed, jobs submitted, GPU work, or LAB-LOG writes. Synthetic checks ran CPU-only with writes intercepted in memory. The missing `smoke-accept.json` is expected and is **not a finding**.

All locations below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

**Round-2 verification**

| Finding | Verdict | Evidence |
|---|---|---|
| 1. Sensitivity allowance counts and relabelling | **CORRECT** | `analyze.py:257–259` copies both annotations; `make_report.py:95–100` no longer turns missing counts into zero. |
| 2. Paired qualification context | **CORRECT** | `analyze.py:260–265` carries qualification, invalid/excluded counts, allowance exclusions and separation labels; `make_report.py:174–178` prints them beside the paired result. Export preserves this context. |
| 3. Denominator coverage | **NEEDS-FIX** | Usable denominator counts and consistency coverage checks are fixed at `analyze.py:102,243–245`, `make_report.py:94,137–138`. Denominator exclusions and allowance counts are still absent beside these coverage entries; see finding 3 below. |
| 4. Certified production configuration identity | **CORRECT** | `analyze.py:173–180` checks plan membership, certificate runs, local/staged config hashes and result-config equality. |
| 5. Scheduled diagnostic completeness | **NEEDS-FIX** | Missing PS outputs now receive `MISSING`, but calibration uses count-only completeness, and missing PS configs silently shrink the expected inventory. |
| 6. Publication detail | **CORRECT for the cited fixes** | `make_report.py:139–140,193–195,233–237` adds absolute-error qualifications, check distances/validity and the requested glossary entries. `rscore.py:180–183` distinguishes invalid successor and check replays. Further reporting gaps remain below. |

**Remaining findings**

1. **WRONG — calibration completeness can pass with a missing cohort.**

   `analyze.py:211–220` counts probe records without retaining or validating their `(cohort, case, kind)` identity. It compares that count with eight rather than comparing an expected inventory with the observed inventory.

   Executing this block with an in-memory fixture containing **two N0 records and no B2 record** produced:
   ```text
   production_probes = 8
   production_probes_expected = 8
   complete = True
   ```
   The calibration warning remains conservative because smoke disagreement persists, but the reported completeness is false.

   Also, `analyze.py:288–291` skips a scheduled PS configuration when its local file is missing. Its case then disappears from the expected set instead of being reported missing.

   **Fix:** derive identity sets from certified configurations; reject duplicates/unexpected identities and explicitly report missing configurations and outputs.

2. **WRONG — val32 receives a false Q2 failure because its check subset is intentionally empty.**

   `rscore.py:472,479` correctly excludes val32 from the tighter/check subset. However, `rscore.py:329–330` requires nonempty `valid_qpp` and `conv` whenever the setting has a check rule, including when no check replay is scheduled for the selected cohort.

   A small fixture with valid production/tight/successor runs, zero distances, passing rho and no expected check cases returned:
   ```text
   Q1.pass = True
   Q2.check_valid = False
   Q2.convergence = False
   Q2.label = "unverified (check replay invalid)"
   numerically_qualified = False
   ```
   `analyze.py:232–250` evaluates and stores this false qualification for val32 in every configuration; `make_report.py:245` exports it. This does **not** change the primary regime labels, but it makes the exported qualification evidence wrong.

   **Fix:** distinguish “no check replay scheduled for this subset” from a missing or invalid required replay. Base the requirement on the expected check population.

3. **NEEDS-FIX — allowance-selection qualifications remain incomplete in publication.**

   `analyze.py:102,240–245` records usable denominator counts but does not attach denominator exclusion/allowance summaries to the ratios or consistency result. `make_report.py:94–96` prints the numerator’s allowance count beside a separate baseline count; `make_report.py:137–138` prints consistency coverage without allowance exclusions.

   Further, the absolute-error table at `make_report.py:139–140` omits the `not_usable` flag and coverage/allowance context. A cell whose median exceeds the registered absolute-error limit can therefore appear there with only its relative-accuracy label.

   The report also never states A3.2’s required selection qualification: **eligibility depends on different cohort resource policies, and the direction of the resulting accuracy bias is unknown**. Explaining the allowance in the glossary does not supply that disclosure.

   **Fix:** carry denominator exclusion summaries into analysis/export, print allowance exclusions beside both populations’ coverage, and include the absolute-unusability and selection qualifications with the affected results.

4. **NEEDS-FIX — `EXPORT.json` loses deliberate PS deferral.**

   `analyze.py:284–292` creates `ps_status`, and `make_report.py:157–159` prints deliberate deferral. But the export constructor at `make_report.py:240–247` copies only `ps`, omitting `ps_status`.

   Executing the actual export expression on a deferred-PS fixture produced `ps: {}` with **no scheduling/deferral status**. Export consumers cannot distinguish intentional deferral from absent diagnostic evidence.

   **Fix:** export the expected inventory, deferral status and completeness alongside PS verdicts.

**Other scope checks**

| Item | Verdict | Evidence |
|---|---|---|
| Production populations and reduced C2 | **CORRECT** | `make_configs.py:94–108`, `rscore.py:462–481`: reproduced **158/150 cases per arm**, **38 tighter cases**, reduced C2 exactly c0–c7. |
| PS selection and allowances | **CORRECT** | `make_configs.py:97–111`: largest configured \(s\); A+ c7; reduced C2 c2; only full A+ has allowance one. |
| Resource workload accounting | **CORRECT** | `resources.py:28–42,63–84`: reproduced **4,950/4,734 rollouts** and **1,422/1,350 verification fields**; no duplicate workload entries found. |
| Measured verification and conditional-probe costing | **CORRECT** | `resources.py:45–59,70,75–77,98–104`: measurements required; conditional probes configured and costed. |
| Reduction/reservation logic | **CORRECT after declared FOM deferral** | `resources.py:99–125`: PS precedes reduced C2; rounded reservations and existing commitments participate in acceptance. |
| Staging gate | **CORRECT** | `cluster/stage.py:47–58`: membership, run equality, reused-attempt refusal, remaining reservations and time-limit equality. |
| Reference routing and missing references | **CORRECT** | `analyze.py:44–54`, `rscore.py:95–123`: configured union, separate E0/lane routing, identity checks; missing references fail closed. |
| Arm inventory and bound fverify | **CORRECT** | `analyze.py:166–186`: complete inventory, certified config, verification-config hash, R-file hashes and field hashes required. |
| Empty baseline and label precedence | **CORRECT** | `analyze.py:95–115`, `rscore.py:352–388`: empty-baseline fixture returned null ratios and coverage-inconclusive accuracy without aborting. |
| Paired statistics | **CORRECT** | `analyze.py:118–135`: complete pairs, pair bootstrap, Wilcoxon selection/all-zero handling. Eight identical pairs returned ratio one, interval `[1,1]`, p-value one. |
| Trends, separation and projection calculations | **CORRECT** | `analyze.py:138–151,197–208`; `rscore.py:391–459`: tie handling, degenerate cases, amended separation thresholds and lattice projection. |
| A7 split-smoke consumption | **CORRECT by inspection** | `smoke_accept.py:74–96,298–315`: separate reference/ROM roots, permitted smk1 traceback and per-job timing intervals; `resources.py:50–59` consumes the resulting measurements. |
| Numerical traceability | **CORRECT for inspected result values** | Report values come from analysis fields or the ledger. Remaining issues concern incorrect or missing qualification/status metadata. |

Production readiness remains blocked by findings **1–4**.