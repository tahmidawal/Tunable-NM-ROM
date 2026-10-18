**Verdict: SUBMIT all five staged attempts for data collection. FIX-FIRST before accepting production verdicts:** the audit trail misidentifies the first acceptance file, and `analyze.py` does not preserve the registered R384 population or fully enforce control failures.

Audited HEAD `49d6dcb1db88f4228883fab279d4ab75a7659056`. No files changed or jobs submitted.

| Item | Assessment | Finding |
|---|---|---|
| A6.1 dispatch correction | **CORRECT** | Restores the already registered warm-up protocol without changing thresholds. |
| A6.1 causal certainty | **NEEDS-RESTATEMENT** | Raw timings support a first-dispatch penalty; they do not establish that warming fixes it. smkC must demonstrate that. |
| A6.2 reproduction correction | **CORRECT for these audited jobs** | Coefficients reproduce; actual generator hashes match the combined archive’s provenance. |
| A6.2 automated identity enforcement | **NEEDS-RESTATEMENT** | The predicate trusts a manifest boolean rather than verifying the evaluated job’s generator provenance. |
| Preserved first acceptance | **WRONG** | Both acceptance files at HEAD are byte-identical and contain only three failures. |
| Manifest revision description | **WRONG** | Revision 2 changes more than the added rule-identity fields. |
| Smoke numerical health | **CORRECT** | No additional numerical blocker found in the tested population. |
| Staged job matrix and integrity | **CORRECT** | Configurations, allocations and staged bytes match the requested attempts. |
| Resource certificates | **CORRECT** | Bound to configurations/runs; projections reproduce; allocations satisfy 1.5×. |
| Full-bank 4096² choice | **CORRECT** | Follows the registered pilot threshold. |
| Production analysis population | **WRONG** | R384’s primary 4096² comparison would use 38 cases instead of the registered 16. |

**A6 and the raw evidence**

The old sync rows are consistent with an additive first-dispatch penalty. Using \(s=(t_1+t_2)/2\), the median inferred penalty \(2s-t_b\) is approximately:

| Kernel | Single interval | Double interval | Inferred penalty |
|---|---:|---:|---:|
| 2D N | 609 µs | 800 µs | 411 µs |
| 2D J | 1,458 µs | 2,530 µs | 383 µs |
| 3D N | 749 µs | 1,090 µs | 394 µs |

The smaller JTJ/chol penalties support this explanation. The new `dispatch_sync` performs burn → three synchronized warm calls before **each** single/double/single interval. Both double calls remain separately dispatched and synchronized; the actual ratio remains subject to **[1.8, 2.2]**. It can still fail—there is no normalization to two, forced pass, or threshold relaxation. **Kernel production remains blocked until smkC passes.**

I independently compared smoke W against the combined arrays:

- R128 BE: worst relative discrepancy **5.033×10⁻¹⁶**.
- R128 CN-R: **7.466×10⁻¹⁶**.
- R384 BE and CN-R: **exactly zero**.

The R384 local/job point hashes differ; R128’s agree. Both generator files match hashes recorded in the combined archive’s `PROVENANCE.json`. This supports accepting the correction for these actual jobs, rather than interpreting the hash mismatch as changed quadrature.

However, [the new predicate](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/p1score.py:268) only consults `rule_generator_identical_to_archive`. Bind that assertion to the evaluated job’s provenance before treating it as a generally enforced identity check.

Two historical statements need correction:

- `smoke-accept-1.json` and `smoke-accept.json` have the same SHA256, **b69ff757…151**, including in HEAD. Neither preserves the claimed five-failure acceptance. The old predicate’s R384 failures are reconstructible, but the purported original record is not present.
- Revision 2 also changes the manifest commit, removes `written_before_any_lane_job`, and updates `p1core_sha256`/`p1common_sha256`. Those updated hashes match the smoke’s source revision, so this is a disclosure problem, not evidence of a hidden post-smoke solver change.

**Smoke findings beyond acceptance**

- All six accuracy configuration/scheme combinations pass Q1 on dev2; all five with Q2 pass it. H640 correctly has no Q2 qualification.
- Ordinary accuracy runs have **0 budget exits and 0 nonstationary steps across 600 steps**. Timing certification also reports zero budget exits.
- Order check: GAL slope **2.0617**, LSPG **2.0441**; both have all interpretability checks true. LSPG remains descriptive under the registered rule.
- Component/evolution ratios **1.0153** and **1.0420** are reassuring diagnostics, not gates.
- Recording probe: nonrecording/recording ratios **0.9704–1.0032**, with identical W. No alarming overhead appears.
- A100 normalized evolution medians are approximately **0.151** for R128, **1.002** for R384 and **3.419** for R512. These change rank and quadrature together; they do not establish a fixed-\((M,m)\) exponent.
- Accuracy smoke peak: **21.74 GB**; timing smoke: **17.58 GB**. Pilot peak: **113,469,241,344 bytes**, below **120×10⁹** by **6.53 GB**. The full-bank choice follows A6.3; the pilot is not a guarantee against every production peak.
- All **28 archived output checksums** verified. All three stderr logs are empty; stdout records GPU backend, f64/highest precision and `ALL-DONE`.

**Staged attempts**

All five have **128 verified manifest entries**, staged provenance matches checkout source, and batch scripts pass shell syntax checks. GPU requests, absolute venv, GPU preflight, precision, isolated paralab paths and output checksum generation are correct.

| Attempt | Projection | 1.5× projection¹ | Allocation | Verdict |
|---|---:|---:|---:|---|
| smkC | — | — | 0.50 h, A100-80G | **SUBMIT** |
| acc256 | 3.303 h | 4.954 h | 5.00 h, A100-80G | **SUBMIT** |
| acc1024 | 2.670 h | 4.006 h | 4.25 h, A100-80G | **SUBMIT** |
| time | 1.781 h | 2.672 h | 2.75 h, H200 | **SUBMIT** |
| acc4096 | 3.823 h | 5.734 h | 5.75 h, H200 | **SUBMIT** |

¹Using recomputed, unrounded projections.

The matrix covers the registered accuracy/timing subjects and A1.8 historical rules: gauss64, fib6765, gauss640 and gauss96; frozen/nominal at 256² and 4096²; shared frozen construction at 1024². R384’s extra 4096² cases correctly support P1g. Timing includes all **24 subjects**, dev6 and three repetitions. acc1024 includes the full reproduction population, vendor bridge, order check and 2D quadrature slice.

These attempts do **not** complete P1a/P1b production or the 3D quadrature slice. Their allocations plus recorded spending total **18.5405 GPU-h**, leaving **1.4595 h** before unused reservations are released. Respect the four-job concurrency cap.

**Fix before interpreting production outputs**

1. [ `h_acc_n` ](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/analyze.py:104) selects every available 4096² case. For R384 this means **38**, including its three-mesh slopes, whereas the registered primary population is **sub16**. Explicitly restrict primary statistics; report the additional cases separately.

2. `timing()` computes doubled-work failures after constructing H-cost-N outcomes, without propagating ordinary ratio failures into those outcomes. Kernel blocking likewise omits that governing doubled-work control. Wire required failures into verdict suppression before exporting claims.

These are analysis and audit-record repairs. They do not require changing the staged numerical experiments or discarding the smoke data.