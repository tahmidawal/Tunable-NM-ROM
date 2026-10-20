**Overall verdict: NEEDS-FIX. Do not submit the real-size smoke yet.**

Audited commit `eb84942f8`. No files changed, jobs submitted, GPU computations run, or lab-log writes made. No scratch files were needed.

All provenance entries matched their recorded source and destination hashes, including the eight unchanged copies. I diffed `refjob`, `refcore`, `p1common`, and `qacc` against the requested originals and compared qualification with `qscore.qualify`.

CPU unit gates passed. The seven smoke configurations reproduce exactly from the generator. Two-bump PS coefficients agreed with direct quadrature to **3.20×10⁻¹⁵**; tested nested nodal ICs agreed exactly.

Paths below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

**Verification of audit 1’s ten required fixes**

| Prior fix | Verdict | Committed-code evidence |
|---|---|---|
| 1. Output boundaries | **NEEDS-FIX** | Direct-root guards exist at `refjob.py:609` and `refslim.py:29`. They do not protect descendant symlinks. Other output drivers remain unrestricted: `qacc.py:95`, `fverify.py:26`. |
| 2. Qualification populations/shapes | **NEEDS-FIX** | Checks are restored at `rscore.py:214–233`; missing-population fixtures now fail. However, callers derive expected form/steps from the record being checked: `smoke_accept.py:177`, `analyze.py:177`. |
| 3. Reference/ROM and frozen identities | **CORRECT** | Descriptor/t0 checks: `rscore.py:103–120`; n0 comparison: `141–145`; frozen checkpoint/rotation: `qacc.py:107–110`; complete main-rule manifest: `144–146`. |
| 4. Raw probe evidence and resume | **NEEDS-FIX** | Evidence is saved at `refjob.py:202–229`, and probes rerun on resume at `380–383`. Offline validation still omits independently expected stage counts and can accept truncated evidence: `smoke_accept.py:67–85`. |
| 5. Sigma and separation coverage | **NEEDS-FIX** | Failed/non-finite finest attempts now return unavailable; amended separation coverage is implemented. But a missing finest/middle attempt raises `KeyError` before unavailable handling: `rscore.py:128–133`. |
| 6. Smoke acceptance completeness | **NEEDS-FIX** | Arm inventory, stderr scanning, local unit gates and LSPG discrepancy gating were added. Control populations remain unchecked; GAL budget-zero still trusts saved certificates; wrong-mesh still trusts a flag: `smoke_accept.py:135–158`. |
| 7. Full-field independent decode | **WRONG** | `fverify.py:38–46` can report success with a non-finite decoded-field comparison. Confirmed by executing its actual `main()` with mocked in-memory inputs. Production analysis never requires its result. |
| 8. E6 configuration/resume enforcement | **NEEDS-FIX** | Ascending unique indices, ST-only, frozen allowance and resume pre-deduction are present at `refjob.py:337–360`. Configuration refusal happens after `main()` has overwritten the previous result/manifest: `674`, then `681`, then `350`. |
| 9. Submission/resources and A+ PS selection | **NEEDS-FIX** | H200, finite positive reservations, fail-closed queue query and A+ PS c7 are fixed. Full workload/reduction certification remains missing; `resources.py` cannot cost the mandatory `fverify` step. |
| 10. Committed populations/statistical orchestration | **NEEDS-FIX** | Smoke uses tighter=`all`; statistical helpers and paired Wilcoxon are committed. Analysis still crashes on an empty baseline and cannot handle the registered reduced-C2 population; reporting omits required qualifications and outputs. |

**Whole-pipeline re-audit**

1. **Sealed/OOD and E0 boundaries — NEEDS-FIX.**

   The reference and ROM cohort providers correctly restrict their normal routes. I found no normal solver route that opens secret sealed cohorts.

   The stronger “no path writes E0” guarantee is not enforced. `qacc --out` and `fverify --out` accept arbitrary destinations. For the guarded reference writers, an allowed output root containing a descendant symlink can redirect `write_case()` or slimming into E0: `refjob.py:263–275`, `388–391`; `refslim.py:42–45`.

   Also, `refslim.py:37–40` accepts absolute or escaping source paths and scans every `2d/*/ST` cohort without an allowlist. It can read plaintext OOD references if directed there.

2. **IC identity and adapters — CORRECT.**

   Reference solver ICs, canonical lattice t0, ROM nodal ICs and two-bump spectral initialization follow the same registered field and normalization convention: `rcohorts.py:156–170`, `295–314`; `refcore.py:239–246`; `p1common.py:116–122`; `qacc.py:119–127`.

   The diffs preserve qualify’s numerical query, tolerances and rules. Explicit control lists resolve correctly. I found no explicit-list controls crash. Q-repro remains a required real-size measurement, not an established result.

3. **E6 numerical escalation — CORRECT; resume transaction — NEEDS-FIX.**

   Mocked executions confirmed:

   - No allowance: `unresolved-floor`, `tight_skipped=true`.
   - Tight base triple followed by order failure: no refinement, `refine_skipped=true`.
   - Refined triple followed by floor failure: no tight refined triple.
   - Ordinary accepted triple: unchanged path.

   Evidence: `refjob.py:158–188`.

   However, a refused configuration change already replaces `result.json` with the new configuration and no previous manifest. A subsequent restart can therefore lose verified-resume accounting and recompute/delete previously recorded outputs. The compatibility check must precede **every write**, not merely precede the case loop.

4. **E5 probes and broken-control detection — WRONG offline validation.**

   The in-job production-member selection and exact upwind `order-fail` requirement are correct.

   Offline acceptance does not reproduce `attempt()`’s complete validity predicate:

   - Upwind compares residual length with the recorded actual `stages`, not the independently expected schedule length.
   - Calibration uses `steps_ok(res, len(res))`, with no expected-count check.
   - Calibration does not independently establish the required complete, finite lattice evidence and selected production-attempt identity.

   Evidence: `refjob.py:93–101` versus `smoke_accept.py:67–85`.

   CPU counterexample: `steps_ok([], 0)` returns **True** and its floor proxy is **0**. Equal saved fields with empty histories can consequently be labelled “agree”. This cannot establish A3.1 validity.

   Whether the actual upwind control rejects specifically on order remains unmeasured.

5. **Stiffness statistic — CORRECT arithmetic; NEEDS-FIX missing-attempt handling.**

   The inclusive box, zero padding, central differences, gradient magnitude, output indices 1–5 and amplitude denominator match A1.5/A4: `refjob.py:105–120`, `rscore.py:126–138`.

   Failed and non-finite recorded attempts are now handled correctly. Missing attempt records still crash instead of returning unavailable with a reason.

   At L=12288, one f64 plane is approximately **1.21 GB**, and six output planes approximately **7.25 GB**. `refcore.py:230–233` additionally pads all outputs, while stiffness computation creates plane-sized temporaries. I found no demonstrated inevitable OOM, but this read-only CPU audit does **not** certify the combined solver/FFT/cache peak on an H200. The corrected smoke must measure it.

6. **Full-field verification — WRONG.**

   A concrete in-memory execution of `fverify.main()` used two finite coefficient trajectories, with the second saved field containing NaNs. It printed:

   ```text
   max_rel: 0.0
   passed: True
   FVERIFY ALL True
   ```

   The cause is `max([0.0, nan]) == 0.0`; filtering only `None` does not filter or reject NaNs: `fverify.py:43–46`. Non-finite coefficient cases are also silently skipped.

   Additionally, failure does not produce a nonzero exit, and `analyze.py:152–169` neither requires nor binds a successful production verification result.

7. **Qualification and statistics — mostly CORRECT helpers; NEEDS-FIX orchestration.**

   Per-regime pooling is correct: `rscore.py:244–251`. Expected-population checks now reject missing cases, and tighter completeness is restored.

   But expected form and step count are supplied from `R` itself. The restored assertions therefore cannot catch a consistently malformed record in the actual smoke/analysis callers. Derive them from the frozen arm configuration and time schedule.

   Bootstrap coverage, label bands, amended separation coverage, average-rank handling and paired Wilcoxon selection are implemented correctly. The slope is also **correct**: callers supply ln(error), and dividing by ln(10) produces the requested log10 slope. A known-slope fixture returned **2.0**. Constant ranks now return undefined correlation/p-value.

   Remaining orchestration defects:

   - Empty usable baseline crashes at `analyze.py:189`.
   - `references()` loads all C2 cases regardless of an approved c0–c7 reduction: `analyze.py:44–50`.
   - Production arm discovery is glob-based, without an expected inventory or completion gate: `analyze.py:152–169`.

8. **Smoke acceptance — NEEDS-FIX.**

   Improvements are real, but acceptance still does not recompute every predicate:

   - Under-resolved and budget-zero files are not checked against the six registered control identities or expected shapes.
   - GAL budget-zero uses saved certificates without independent recomputation.
   - Wrong-mesh rejection is explicitly accepted from the saved flag.
   - Full-field acceptance inherits the NaN defect above.

   Evidence: `smoke_accept.py:135–158`.

   Thus an incomplete or broken control can still escape rejection.

9. **Staging, submission and resource accounting — NEEDS-FIX.**

   Committed-tree staging, GPU preflight, H200 restriction, unique attempt directory, smoke time cap, reservation-before-submit, campaign flock and fail-closed account query are correctly implemented.

   The resource certificate remains incomplete:

   - `resources.py:42–59` has no `fverify` rate; a production ROM attempt including that required step raises `no rate`.
   - No code generates or validates the complete A3.3 plan and ordered reductions.
   - `stage.py:45–48` trusts membership and a supplied aggregate `remaining_reservation_hours`; it does not derive that total from the certified workloads.
   - Conditional calibration probes are omitted from generated production configurations and costing.
   - ROM costing uses a single elapsed-time scaling, rather than the specified workload accounting.

10. **Reporting and other paths — NEEDS-FIX.**

    `refslim` and `rscore.STORE` layouts agree. Smoke filenames agree. A+ PS selection is correctly **c7**.

    The report does not implement the required scientific qualifications:

    - Failed calibration does not trigger production probes or the “floor proxy not supported” label.
    - Failed separation does not trigger the required confounding/retraction labels.
    - Allowance-unresolved counts are in a separate table, not beside each accuracy coverage entry.
    - N0s descriptive labels, sensitivity ratios, paired results, trends and detailed Q1/Q2 outcomes are not presented in the report.

    Evidence: `make_configs.py:85–99`; `analyze.py:145–218`; `make_report.py:78–129`. `SM` is loaded but its calibration result is never used.

**Required fixes**

1. Enforce resolved read/write boundaries across drivers and actual descendant destinations; restrict slimming sources/cohorts.
2. Validate resume configuration before overwriting any existing result or manifest.
3. Recompute probe validity using independently expected schedules, complete finite fields and verified attempt identity.
4. Make `fverify` reject every non-finite comparison, validate populations/shapes, return failure status, and bind its result to production analysis.
5. Derive expected qualification form/steps from frozen configuration; add regression fixtures for the demonstrated false-pass paths.
6. Validate every control’s exact population and evidence; independently recompute GAL budget-zero and verify wrong-mesh rejection.
7. Handle missing sigma attempts, empty baselines, missing arms and approved reduced populations explicitly.
8. Implement the complete resource plan, reduction order, conditional probes and `fverify` costing.
9. Propagate calibration and separation failures into production configuration, analysis, report and export; complete the registered statistical reporting.

The numerical core has improved substantially, but the acceptance layer still admits false passes and the production/reporting path is incomplete.