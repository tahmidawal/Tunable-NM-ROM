**Overall verdict: WRONG.** A24 fixes the optimum wording and most flop arithmetic, but gradient memory traffic remains wrong, and the cumulative-peak admission rule can accept an uncertified memory footprint.

References below use `D = experiments/jcp-hr-baselines/DESIGN.md`.

1. **CORRECT — A24.1 narrows the optimum claim appropriately.**

   It replaces “solved exactly” with satisfaction of the registered numerical KKT certificate, requires the solver flag and both gradient inequalities, adds finite nonnegative terminal weights and residual evidence, and limits full-QR/updating-QR agreement to regression fixtures. Q1 and construction-quality restrictions remain separate. The legacy smk2 exception is explicitly disclosed.

   **Evidence:** `D:1385–1391`; KKT sign convention at `D:850–852`; all-candidate check, including banned indices, at `D:815–818`.

   **Fix:** No substantive design correction required. In acceptance, count **positive weights**, bind substituted targets to the saved terminal solution, and retain the legacy exception visibly. Multiple targets mapping to that solution are aliases, not independent constructions.

2. **WRONG — A24.2 still contains incorrect traffic arithmetic and an unsupported runtime estimate.**

   For the stated dimensions, taking the stated iteration count literally as \(I=19{,}000\):

   | Quantity | Recomputed value | Assessment |
   |---|---:|---|
   | Append flops, \(8mKs\) | \(2.24300372066304\times10^{14}\) | Correct |
   | Append traffic, \(32mKs\) | \(8.97201488265216\times10^{14}\) bytes | Correct |
   | Gradient flops per iteration, \(2mN\) | \(6.845104128\times10^9\) | Correct formula |
   | Gradient flops, \(2mNI\) | \(1.30056978432\times10^{14}\) | Approximately \(1.3\times10^{14}\), rather than \(1.2\times10^{14}\) |
   | Reading \(G\) each iteration, \(8mNI\) | \(5.20227913728\times10^{14}\) bytes | **Not \(4.2\times10^{15}\)** |
   | Refactorisations, \(\sum_{j=1}^{16}6m(1024j)^2\) | \(9.83066474446848\times10^{14}\) flops | Correct |
   | Listed flop total | \(1.337423824945152\times10^{15}\) | Correct rounded total |
   | Append traffic + gradient reads | \(1.417429401993216\times10^{15}\) bytes | **Not approximately \(5\times10^{15}\)** |

   The last traffic figure excludes refactorisation, deletion, residual formation and other traffic; it is not a complete workload total. Refactorisations contribute about **73.5% of the listed flops**. “Not dominant” therefore needs to distinguish runtime from operation count.

   At the H200’s advertised **4.8 TB/s**, the corrected listed traffic corresponds to **295 seconds, or 4.92 minutes**, at ideal bandwidth—not a measured runtime. Twenty minutes could be a conservative projection, but requires an explicit efficiency assumption and accounting for omitted work. [NVIDIA H200 specifications](https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23?xs=443995).

   **Evidence:** `D:1393–1400`; the actual allowed bound is 40,000 outer iterations per attempt at `D:1233–1235`.

   **Fix:** Correct gradient bytes and totals; identify 19,000 iterations as an assumption rather than a bound; separately account for accepted/rejected appends, deletions, residual formation, refactorisations, compilation and retries. Generate the certificate from that inventory. Replace “memory-bound: ≈20 min” with a conditional projection supported by measured throughput.

3. **CORRECT — A24.3 supplies explicit quality labels and consequences.**

   Nonfinite drift now counts as a limitation, alongside drift above \(10^{-8}\). Banning uses the selected path’s count and the registered 1% threshold. Offline reconstruction from saved evidence and propagation to all configurations of the rank prevent a successful checkpoint from silently hiding a compromised construction.

   These labels restrict “adequately tuned ECSW” and exhaustive non-domination claims; they do not automatically establish or revoke online Q1 qualification.

   **Evidence:** `D:1402–1405`, `D:1199–1207`, `D:1354–1359`.

   **Fix:** No substantive design change required. Enforce the inherited wording restrictions in every downstream verdict and report.

4. **WRONG — A24.4’s selection structure is permissible, but its memory admission rule is unsound.**

   The selection sequence itself respects the governing amendments:

   - Stage-B membership comes from Stage-A screening, including uncapped absolute and X-specific fronts.
   - Selecting final picks on the full Stage-B cohort is expressly allowed by A1.6. This is disclosed in-sample selection, not prohibited test leakage.
   - Screening ratios pair X and B within the tuning job; verdict A–B–A measurements occur within the Stage-B job. No cross-job timing comparison is required.

   **Evidence:** `D:385–394`, `D:572–576`, `D:854–857`, `D:1407–1414`.

   However, the memory rule fails when the process already has a high peak. For example, **before = after = 130 GB** cannot establish whether the ECSW path used 110 GB or 125 GB, yet A24.4 rejects only a *new* peak above 120 GB. This is also implemented literally in `experiments/jcp-hr-baselines/tune.py:177–187`.

   Measuring after the full path also replaces A22.2’s intended pre-path admission with retrospective rejection.

   **Evidence:** `D:1351–1352`, `D:1414–1416`.

   **Fix:** Require a measured cumulative peak below 120 GB, conservatively treating a higher inherited peak as inconclusive; alternatively isolate the maximal-shape pilot in a fresh process and gate the production path on its result. Missing/nonfinite telemetry must fail admission.

   Also state explicitly that A24 supersedes the pre-verdict **pick** freeze in §6/A21.3: Stage-B **membership** is frozen beforehand; picks afterward. Ensure every potential role candidate has the required timing coverage, including B_cost candidates outside A1.12’s exploratory error-envelope filter.

5. **CORRECT — A24.5 retains the verdict-job G-ours requirement.**

   The specified six cases are exactly §7’s timing population. A24.5 requires replay both in tuning **and again in Stage B**, which is now the verdict/timing job. It therefore does not substitute a tuning-only check for §8’s verdict replay.

   Referencing `r1.json` for bank/rank/start settings does not import its replication cohort: the scoring inputs explicitly use seed **923801**, cases 0–5.

   **Evidence:** `D:206–211`, `D:222–224`, `D:1418–1423`. The local `inputs/speed3d_batch1_e0.json` explicitly records that cohort and source-score hashes.

   **Fix:** No design correction required. Preserve the timed-output/scored-output equality assertion and source identity checks. These replay checks do not supply the missing 3D Q2 certificates; A22.0’s diagnostic-only restriction still applies (`D:1337–1342`).

6. **NEEDS-RESTATEMENT — A19’s production preconditions are not yet established.**

   A24 provides design-level responses to optimum evidence and quality propagation, but the cost and memory defects above prevent a clean design/resource conclusion. The ongoing gate-suite result cannot be presumed passed, and smk2 used full QR rather than the production updating-QR algorithm.

   **Evidence:** production conditions at `D:1264–1269`; smk2 algorithm distinction at `D:1376–1378`; binding stability gate at `D:1354–1358`.

   **Fix:** Record an explicit superseding clean design audit, completed source-pinned updating-QR code/gate evidence, and a regenerated resource certificate using corrected arithmetic and sound memory admission. A19’s literal reference to “round 14” should become a reference to the accepted successor audit.

**Overall: WRONG.** Production readiness remains unestablished. No files were changed, jobs submitted, GPU work performed, scratch created, or tests run.