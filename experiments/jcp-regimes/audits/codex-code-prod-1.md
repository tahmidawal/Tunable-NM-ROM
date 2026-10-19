**NEEDS-FIX — not ready for production.**

Audited commit `76f038c98caf2d2ef0a1d6b419cbd43bfa27505f`. No files changed, jobs submitted, GPU work, or LAB-LOG writes. Synthetic checks ran CPU-only with writes intercepted in memory. The absent smoke acceptance is expected; it is not itself a finding.

Locations below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

1. **WRONG — resource planning always crashes.**  
   `resources.py:36`, `resources.py:80`: `calib_agree=sm.get('calib_agree')` is swallowed by the trailing comment. Consequently, `basis()` never returns that key and `plan()` raises **`KeyError: 'calib_agree'`**. Reproduced with an otherwise accepted synthetic smoke.

2. **NEEDS-FIX — resource certificates do not fully implement the accounting contract.**  
   `resources.py:35–36`: missing probe/verification measurements silently receive fallback rates. A6 explicitly requires measured verification cost; absent measurement must prevent certification.  
   `resources.py:51–59`: ROM costing scales total smoke time by production case-arm count, without enumerating prod/tight/tighter/qp/qpp workloads as A3.3 requires. This is conservative for the current subset sizes, but it is not the required auditable accounting.  
   `resources.py:85–101`: reductions stop based on projection, before rounded reservations and existing ledger commitments are considered. A projection below 50 hours can produce reservations above 60 hours; staging then refuses without the planner trying further reductions. Make the completed plan satisfy the reservation cap, or explicitly refuse it.

3. **NEEDS-FIX — staging does not bind the workload to the plan entry.**  
   `cluster/stage.py:45–57`: membership and time-limit equality are checked, and submitted runs are bound to the individual certificate. However, neither certificate nor supplied runs are compared with `plan['attempts'][attempt]['runs']`. A stale plan and updated certificate can therefore describe different workloads while passing the gate.  
   `cluster/stage.py:48–50`: excluding every previously seen attempt is correct for ordinary first submissions, but also excludes terminal failed attempts being retried. Such a retry’s new reservation is missing from the remaining-plan calculation. Reject reuse or account for it explicitly.

4. **WRONG — insufficient baseline coverage aborts the entire analysis.**  
   `analyze.py:214–216`: fewer than eight usable baseline cases raises `SystemExit`, discarding every configuration’s analysis. A1.8/A2.4 require inconclusive coverage labels while preserving validity, qualification and available errors.  
   The existing `label_cell()` already handles an empty baseline correctly: an in-memory test returned null ratios and `inconclusive (coverage)`. Remove the contradictory whole-analysis exit and preserve denominator coverage.

5. **NEEDS-FIX — production inventory and verification are incompletely bound.**  
   `analyze.py:162–174`: expected arms and populations come exclusively from `result.json`’s own configuration. There is no comparison with the certified production configuration. A self-consistent truncated configuration can redefine the supposedly complete inventory.  
   The verification acceptance checks R-file hashes, but omits the verification-config hash and explicit field-hash binding used by smoke acceptance at `smoke_accept.py:228–235`. Carry those checks into production, together with the certified configuration identity.

6. **WRONG — generated Markdown tables misassign columns.**  
   `make_report.py:104`, `119`, `128`, `136`: configuration identifiers such as `R128|LSPG_BE_1` are inserted without escaping the pipe. Every configuration row acquires an extra column, shifting values away from their headers. Reproduced in memory:
   ```text
   | R128|LSPG_BE_1 | IMPROVES; ×0.5 ...
   ```
   Escape table-cell content centrally.

7. **WRONG — secondary tables can display misleading accuracy labels.**  
   `make_report.py:119–120`: sensitivity prints `accuracy_label` rather than the precedence-aware `label`. Thus an invalid or unqualified regime can appear simply **HOLDS/IMPROVES**, without qualification, usable coverage, denominator coverage, allowance-unresolved count, or separation relabelling.  
   `make_report.py:128–129`: paired labels likewise omit the relevant qualification/separation context and allowance exclusions.  
   `analyze.py:187`, `make_report.py:156–168`: the “stiff-part comparison unavailable” qualification is computed but not printed or exported.  
   Calibration and separation labels exist, but are not consistently carried beside the affected secondary numbers.

8. **NEEDS-FIX — required results disappear between analysis and publication.**  
   `analyze.py:222`, `239–250`; `rscore.py:403–405`; `make_report.py:115–169`:
   - In-range consistency is computed but neither reported nor exported.
   - Projection floors are exported but never reported.
   - OLS trend slopes are exported but never reported.
   - N0s’s descriptive labels against the primary baseline are not printed; only its N0-only sensitivity appears.
   - Per-cell absolute medians, worst errors and worst ratios are not tabulated; median plots do not replace these required results.
   - Q1/Q2 reporting shows aggregate status and non-stationary counts, but omits the failing distance, validity and convergence predicates.
   - Scheduled PS cross-check results have no analysis/report ingestion path.

   Synthetic sentinel values confirmed that floors, slopes and consistency never reach the report. The export also omits qualification detail and the exploratory-development qualification supplied in the report preamble.

9. **WRONG — reduced populations are silently omitted and the glossary contradicts them.**  
   `analyze.py:62–69`: coverage records only configured cases; it never records the eight deferred C2 cases. Neither report nor export identifies them as deferred, contrary to A3.3.  
   `make_report.py:147`: the hard-coded definition “a set of 16 … cases” is wrong for N0s and reduced C2, and is not generated from analysis. The glossary also leaves several displayed terms undefined, including Wilcoxon, Spearman, bootstrap intervals, tight allowance, refined, and diagnostic.

The following items checked out, subject to the integration failures above:

| Item | Verdict | Evidence |
|---|---|---|
| Full/reduced production populations | **CORRECT** | `make_configs.py:94–108`, `rscore.py:460–479`: reproduced 158/150 cases per arm; tighter subset remains 38; reduced C2 is exactly c0–c7. |
| PS case selection and allowances | **CORRECT** | `make_configs.py:98–111`: largest \(s\) within the configured cohort; A+ selects c7. A+ allowance is one; others zero. |
| Conditional probe configuration | **CORRECT** | `make_configs.py:99–100`: enabled branch adds case-zero coarse probes to every regime. Planner invocation is blocked by finding 1. |
| Reduction sequence | **CORRECT in order** | `resources.py:81–90`: after declared FOM deferral, drops PS before reducing C2. Reservation handling needs finding 2 fixed. |
| Ordinary remaining-reservation accounting | **CORRECT** | `cluster/stage.py:48–51`: unseen planned attempts plus existing open reservations avoids double-counting first submissions; time-limit equality is enforced. |
| Reference routing/configured union | **CORRECT** | `analyze.py:44–54`, `rscore.py:97–123`: reduced populations are respected; E0 and lane references route separately. Missing reference files fail closed rather than silently shrinking coverage. |
| Primary labels and exclusions | **CORRECT** | `analyze.py:88–114`: invalid-run precedence, qualification precedence, positive finite eligible-error exclusions, and absolute unusability flag. |
| Paired contrast/Wilcoxon | **CORRECT** | `analyze.py:117–134`: complete-pair filtering, pair bootstrap, exact/approximate rule and all-zero handling. Eight identical pairs reproduced ratio 1, interval [1,1], p=1. |
| Trends, separation and projection helpers | **CORRECT** | `rscore.py:389–457`: tie-aware ranks, degenerate handling, amended separation coverage and lattice least-squares projection. Publication omissions remain above. |

The decisive blockers are the deterministic planner crash, incomplete production evidence binding, analysis abort on a legitimate coverage outcome, and reports that can misplace or misqualify results.