# Independent audit of b-qxm

**The reported numerical values reproduce, but the report has provenance errors, post-result gate changes, and incomplete retraction disclosure.** The original \(q=0\ldots256\), fixed-\(M=1088\) result remains supported.

This audit was read-only. I independently recomputed all 231 error aggregates from the archived fields and all 77 timing medians from `result.json`. Errors agreed within \(2.0\times10^{-14}\) percentage points; timing medians matched exactly.

All paths below are relative to `experiments/b-qxm/`. Summary row indices are zero-based.

## Verified numbers

Notation: `A201[arm].key` means `artifacts/bqx201/audit.json`, selecting the entry in `arms` by `arm`. Define:

- \(E_j(q,M)\): `worst_evolved_percent`.
- \(C_j(q,M)\): `worst_all_times_percent`.
- \(T_j(q,M)\): `median_gpu_ms`, independently checked against `1000 × median(result.json::invocations[name=arm].gpu_seconds)`.
- Arm names for these functions are `q{q}_M{M}_dense`.

Values below are shortened for readability; comparisons used full precision.

| # | Report location and printed number | `reports/summary.json` row, metric and value | Artifact verification | Finding |
|---|---|---|---|---|
| 1 | T4, \(q=0,M=64\): **1.8890%** evolved | `[13]`, `worst_evolved_percent` = `1.8889895724293762` | \(E_{101}(0,64)\), job **3780175** | Match |
| 2 | T4, \(q=32,M=768\): **2.3534%** compression | `[111]`, `worst_t0_compression_percent` = `2.353373356110006` | `A101[q32_M768_dense].worst_t0_compression_percent` | Match |
| 3 | T4, \(q=256,M=1088\): **4377.9 ms** | `[214]`, `median_gpu_ms` = `4377.883864566684` | \(T_{201}(256,1088)\), job **3780177** | Match |
| 4 | Saturation, \(q=256,M=6528\): **12130.8 ms** | `[367]`, `saturation.median_gpu_ms` = `12130.830192589201`; cites **3780178** | \(T_{501}(256,6528)\), actual job **3783899**; cited S1 has no such arm | **Wrong source** |
| 5 | E1, \(q=512,M=3168\): **0.1844%** | `[319]`, `e1_disambiguation.value` = `0.18436044342439203` | \(E_{401}(512,3168)\), job **3783898** | Value matches; summary omits job and uses an arm alias |
| 6 | E1 prose: worst gradient **1.77e-01** | `[313]`, `e1_disambiguation.worst_joint_gradient` = `0.17744151533960245` | `A401[q512_M1088_dense].max_joint_stationarity`; reproduced from raw stationarity arrays | Value matches; same provenance defect |
| 7 | Verdict: pure-rank span **2.437×** | `[2]`, `verdict.span_q_at_M1088` = `2.4368429602045354` | \(E_{101}(0,1088)/E_{201}(256,1088)\) | Match |
| 8 | Within-job ladder: cost span **5.163×** | `[264]`, `fixed_M_within_job.cost_span.evolved` = `5.1626587819982905` | \(T_{201}(256,1088)/T_{201}(0,1088)\) | Match; same job |
| 9 | Corner-path prose: rank **69.0%** | `[280]`, `corner.share_q.evolved` = `0.6898510248025506` | \(\log[E_{101}(0,1088)/E_{201}(256,1088)]/\log[E_{101}(0,64)/E_{201}(256,1088)]\) | Match |
| 10 | Rung-path prose: tests **34.4%** | `[281]`, `rung.share_M.evolved` = `0.34394576984102004` | Independently summed the five DESIGN §4.2 bridge increments using G1/G2 errors | Match |
| 11 | Variance prose: rank **85.1%** | `[283]`, `anova.frac_q.evolved` = `0.8508031836710135` | Independently decomposed log-errors for \(q=\{0,16,32,64,128\}\), \(M=\{256,1088\}\) | Match |
| 12 | All-times ladder: error span **2.831×** | `[288]`, `fixed_M_within_job.error_span.all-times` = `2.831042036886115` | \(C_{201}(0,1088)/C_{201}(256,1088)\) | Match; table heading is wrong |
| 13 | Saturation prose, \(q=0\): **1.210×** cost | `[325]`, `saturation.cost_ratio_at_M_star` = `1.2101110662903052` | \(T_{301}(0,256)/T_{301}(0,64)\), job **3780178** | Match |
| 14 | Saturation prose, \(q=256\): **1.493×** cost | `[357]`, `saturation.cost_ratio_at_M_star` = `1.4926131635219955`; cites **3780178** | \(T_{501}(256,2176)/T_{501}(256,1088)\), actual job **3783899** | **Wrong source; arithmetic correct** |
| 15 | G2 full-order control, `fft_loose`: **37.382 ms** | `[399]`, `fom.median_gpu_ms` = `37.38191397860646` | `A201[fft_loose].median_gpu_ms` and raw invocation times | Match |

The summary’s five audit hashes match the artifact audits; each audit’s `result_sha256` matches its `result.json`. The copies under `checks/` are byte-identical to the artifact audits.

## Mismatches

**1. Twelve saturation rows cite the wrong job.**

`summary.json::rows[356]` through `[367]`, all with `q=256` and `metric=saturation.*`, say:

```text
attempt = bqx301
job_id  = 3780178
```

They should identify **`bqx501`, job `3783899`**. S1 contains no \(q=256\) arms. The ten arm-level values match E2 exactly; the other two rows are E2’s derived saturation point and cost ratio.

The cause is visible in [generate_xm.py:822](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-qxm/experiments/b-qxm/reports/generate_xm.py:822): summary rows use the top-level S1 source instead of `saturation.sources["256"]`. The report’s overarching “Saturation … job S1” heading is consequently misleading, although its \(q=256\) subsection correctly names E2.

**2. E1 summary identifiers do not resolve directly.**

Rows `[311]`–`[323]` have `job_id=null`. The twelve arm-level rows use names such as `q512_M3168`; the artifact names are `q512_M3168_dense`. Their values match after resolving this alias, but the requested direct provenance chain is incomplete.

**3. Two table presentations misidentify what their values describe.**

- At report lines **122 and 188**, the fixed-\(M=1088\) table lists only \(q=0\ldots256\), yet says “every rung converged: no” and “span unavailable.” **Every displayed rung converged.** Those flags include the undisplayed \(q=512\) failure.
- At lines **171 and 178**, tables inside the all-times section label their values “worst evolved %.” For example, `2.5629` is all-times error; the corresponding evolved error is approximately `1.2657`.

There were **no rounding or arithmetic mismatches among the 15 sampled values**. These findings concern provenance and meaning.

## Gate audit

| Verdict or claim | Applicable DESIGN rule | Independent assessment |
|---|---|---|
| Fixed-\(M=1088\) headline | §6: monotone, all declared rungs converged, error span ≥2 | **Supported for the original \(q=0\ldots256\) range.** Span `2.4368429602045354`; every original rung converged. The extended range through 512 is unavailable. |
| Fixed-\(M=1088\) passes the operating-point tunability bar | Report’s stated inherited bar: ≥3 nondominated points, ≥2× error and cost, monotone, no unconverged point | **Pass within G2:** four points, error `2.4368429602045056×`, cost `5.1626587819982905×`. The additional cost/frontier criteria are not explicitly specified in the original DESIGN §6. |
| Fixed-\(M=256\) fails that bar | Same rule | **Correct:** error span `1.2200715106073892×`, despite cost span `2.2693196281456323×`. |
| “Rank claim false: no” | §6: false only if every specified fixed-\(M\) span <1.5 | **Correct:** the original \(M=1088\) span exceeds 1.5. |
| H(rank) false | §6: eligible \(M\)-span ≥1.5 at some \(q\ge64\), or no qualifying \(M^\star\) at \(q=64\) | **Correct already in round 1:** \(q=256\), \(M=544\rightarrow2176\), span `2.0251950370155725×`. No round-2 amendment is needed. |
| H(tests) false | §6: corner rank share ≥0.5 | **Correct:** `0.6898510248025506`. |
| “Both factors matter”; A2 explicitly says “H(both) holds” | §1.1 H(both): rank span ≥2 **and** \(q=0\) test-count span ≥1.5 | **Formal H(both) does not pass.** The original corner span is `1.4924959166518024×`; even the full S1 sweep reaches only `1.4933999602809243×`. Qualitative evidence that both matter is supportable; the registered conjunction is not satisfied. |
| Paths do not disagree strongly | §4.2: share difference >0.15 | **Correct:** difference approximately `0.033797`. |
| Saturation at \(M^\star=256\), \(q=0,64\) | §4.3: first next-step improvement <5% | **Correct:** next improvements `0.377103%` and `3.655863%`. Cost-neutral verdicts also match: `1.210111>1.1` and `0.922468≤1.1`. |
| E2 saturation at \(M^\star=2176\) | A3 plus §4.3; final improvement must be <5% | **Correct:** \(2176\rightarrow3264\) improves `4.603967%`; the final step improves `0.374555%`. Cost `1.492613×` is not neutral. |
| E1 “uninformative” | A3: all three \(q=512\) cells fail convergence | **Correct under the unchanged convergence gate.** All three fail because their initial-condition joint stationarity exceeds \(10^{-6}\). |
| “No failed gates” for E1/E2 | §5: declared fidelity comparisons are blocking | **True only after a post-result exemption**, detailed below. |

### Gate changes after data arrived

**A5 changed headline selection after round-2 results.** Commit `b4e38103` introduced `certified_span` and changed verdict selection to read it instead of the full attempted span. This affects `headline_fixed_M`, `rank_claim_false`, and the chosen within-job ladder.

That change must be disclosed as post-result. **It does not invalidate the original result:** the retained \(q=0\ldots256\) range was explicitly registered before round 1 and passes independently. The present wording incorrectly combines its span with the extended range’s convergence flag.

**Fidelity exemptions also changed after results, without the claimed DESIGN amendment.** The same commit changed missing comparator arms from blocking failures to informational “not applicable”:

- `artifacts/bqx401/audit.json::checks.cross_job_fidelity.not_applicable = 10`
- `artifacts/bqx501/audit.json::checks.cross_job_fidelity.not_applicable = 5`

Example: `q16_M1088_dense__bqx201` has `passed=null` because G2 never ran that arm. The pre-submission auditor at `76072bf4` would fail it.

These are plausible corrections to misdeclared comparisons, rather than observed numerical disagreements. Nevertheless, they change the blocking gate after results. [audit_xm.py:233](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-qxm/experiments/b-qxm/audit_xm.py:233) says “See DESIGN.md A5,” but **A5 does not document this exemption**.

A0’s relaxation of the initial-field invariance gate occurred **before the cluster jobs**, following smoke results. It did not affect acceptance here: independently recomputed initial fields agree exactly across \(M\).

## Cross-job check

**No cost ratio crosses jobs or GPUs.** I recomputed all 24 cost-ratio entries from raw invocation times, including repeated entries for the two error metrics.

| Ratio family | Actual source | GPU | Result |
|---|---|---|---|
| Fixed-\(M=256,1088\) cost spans | G2, **3780177** | A100-PCIE-40GB | Same-job operands |
| \(q=0,64\) saturation ratios | S1, **3780178** | A100-PCIE-40GB | Same-job operands |
| \(q=256\) saturation ratios | E2, **3783899** | A100 80GB PCIe | Same-job operands; summary source is wrong |
| LU/Gauss–Jordan ratio | G1, **3780175** | A100 80GB PCIe | `0.9999151004947566×`, same-job operands |

The headline cost ratio is specifically `4377.883864566684 / 847.9901634855196`, both from G2. It does not use G1’s cheaper \(q=0\) timing.

Cross-job **error** comparisons are separately justified: all 20 reported anchors pass their declared tolerances. The largest evolved-error relative difference is `9.889238757751388e-9`.

## Retraction completeness

**Incomplete: the report has no retractions section.** It discusses the failed extension and its reporting treatment, but does not carry most of the ledger’s corrections.

| Retraction or deviation | Source | Coverage in report |
|---|---|---|
| Smoke misdiagnosis and invariance-gate relaxation | `checks/retractions.md:1`; A0 | Gate outcomes appear; correction history absent |
| False memory bound and excessive local probe/smoke durations | `checks/retractions.md:15`; A0 | Absent |
| Substitute design auditor and absence of the originally required independent final audit | `checks/retractions.md:25,48`; A0/A5 | Absent |
| Four original design defects: shortened spans, vacuous fidelity, near-square falsification, solver confound | `checks/retractions.md:31`; A0 | Corrected methods partly visible; retractions absent |
| First generator missed G2’s anchor and incorrectly failed tunability | `checks/retractions.md:40` | Absent |
| Solver-control number corrected from `4.7e-14` to `0.0` | `checks/retractions.md:53`; A2 | Correct number appears; retraction absent |
| Failed extension erased the original certified span | `checks/retractions.md:58`; A5 | Substantively disclosed |
| Withdrawal of “roughly invariant tests-per-unknown ratio” | A3 | Absent from both report and `checks/retractions.md` |
| Fifteen post-result fidelity exemptions | `audit_xm.py`, commit `b4e38103` | No explicit disclosure in report, DESIGN amendments, or retraction ledger |

A5 also retains erroneous prose at [DESIGN.md:626](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-qxm/experiments/b-qxm/DESIGN.md:626): the first improvement beyond \(M^\star\) is **4.604%**, not below 1.2%; \(M=2176\) is not the best converged accuracy after E2’s **0.3509754472%** result at \(M=6528\); and \(q=256,M=1088\) is not the cheapest fixed-\(M\) rung. The generated tables correctly contradict those sentences.

## Three weakest claims

1. **Formal acceptance of H(both).** Its \(1.5×\) test-count threshold is missed, although narrowly. A reviewer would ask why A2 declares it satisfied and would require the registered verdict to be separated from the qualitative observation that both factors affect error.

2. **The implication that E1/E2 passed all pre-registered blocking gates unchanged.** Fifteen declared comparisons became exempt after results. A reviewer would request the original gate outcomes, a dated amendment listing the exemptions, and explicit separation of corrected configuration mistakes from numerical fidelity passes.

3. **The explanation that the \(q=512\) solve “stalls short of stationarity.”** The decisive `0.17744151533960245` comes from `ic_joint_stationarity`. For \(M=1088,2112,3168\), maximum **evolution-step** joint stationarities are respectively `5.694695644945215e-7`, `3.487426437601458e-7`, and `2.99730451248605e-7`; all 900 evolution exits per arm are gradient exits. Initial-fit relative residuals are at most `6.357066600617744e-17`. A reviewer would ask for an initial-fit diagnosis and an explanation of the normalized gradient near such tiny residuals. **E1 remains uninformative under the registered gate, but the report obscures which part failed.**