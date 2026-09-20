# Independent audit of the b-seeds report

**The numerical results and C1–C4 verdicts check out.** I found inaccurate gate-pass counts, incomplete coverage in `summary.json`, and missing retraction disclosures. No cost ratio crosses jobs or GPUs, and no verdict threshold was changed after the production results landed.

Audit scope: [report](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-seeds/experiments/b-seeds/reports/2026-09-17-b-seeds.md), SHA256 `cf791634c5bf1a1c7511c6db6f77a79bb15afd483eb2d9c9338425b7442371db`. All paths below are relative to `experiments/b-seeds/`.

I independently recomputed **420 error aggregates from saved fields**, **140 timing medians from invocation records**, and convergence for **1,980 ROM invocations**. These agree with the audits. No files were changed, no GPU work ran, and no jobs were submitted.

## Verified numbers

Notation:

- `S` = `reports/summary.json`; array indices below are zero-based.
- `Lᵢ` = `artifacts/s<i>/result-ladder-seed.json`.
- `Iᵢ` = `artifacts/s<i>/result-ladder-incumbent.json`.
- `Fₓ` = `artifacts/final/result-sealed_<x>.json`.
- `Eᵢ` = `artifacts/s<i>/result-eqcert-seed.json`.

For same-grid errors, I followed `invocations[].artifact` to the saved fields and recomputed against that block’s `fft_tight`, using the reference initial-field norm. The artifact JSON’s `error` field alone measures a different reference and is insufficient.

| # | Report location and printed number | Summary trace | Artifact verification | Outcome |
|---|---|---|---|---|
| 1 | §2, q=0 seed-mean evolved error: **2.0002%** | Mean of `S[0,196,392].value`, metric `evolved` | `L₁–L₃`, `q0_M64_dense` fields: mean **2.000185696157%** | Match |
| 2 | §2, q=256 evolved standard deviation: **0.1164** | Sample SD of `S[45,241,437].value` | `L₁–L₃`, `old_q256_M1088_dense`: SD **0.116386842051** | Match |
| 3 | §2, q=256 mean seed/incumbent cost: **1.117×** | Mean of `S[50]/S[638]`, `S[246]/S[809]`, `S[442]/S[980]`, using `.value` | `Lᵢ/Iᵢ`, median `invocations[].gpu_seconds`: ratios **1.084946283, 0.704087569, 1.561651709**; mean **1.116895187** | Match |
| 4 | §2, fixed-M seed2 q=128 error: **1.1288%** | `S[286]`, `evolved` | `L₂`, `old_q128_M256_dense` fields: **1.128802827272%** | Match |
| 5 | §3, seed3 bank floor: **0.3511%** | `S[511]`, `three_layer_bank_floor_percent` | `L₃`, `100 × max(reconstruction[q=0].cases[].bank_projection_max)` = **0.351056924308%** | Match |
| 6 | §4, seed1 head loss: **8.121e-05** | `S[142]`, `head_final_loss` | `artifacts/s1/train-hfit_full.json`, `arms.mid.train.final_loss` = **8.120960157337e-05** | Match |
| 7 | §4, seed2 cond(G): **21741** | `S[330]`, `bank_cond_G` | `artifacts/s2/train-sep_coeff_N256_K16_R512.json`, `span.cond_G` = **21740.93690556745** | Match |
| 8 | §5, sealed incumbent q=0: **10.1120%** | `S[1101]`, `evolved` | `F_incumbent`, `q0_M64_dense` fields: **10.112013091090%**, worst case 4 | Match |
| 9 | §1/§5, incumbent q=0 sealed/dev ratio: **5.353** | `S[1741]`, `C2_ratio_incumbent` | `F_incumbent/L₁` is **not** the calculation: the denominator correctly comes from **`I₁`**. Recomputed error ratio **5.353133356942** | Match |
| 10 | §5, q=32 normalised seed-mean ratio: **1.137** | `S[1760]`, `C2_normalised_ratio_seed_mean` | `L₁–L₃`, `F_seed1–F_seed3`: ratio of means of evolved error / worst best-found = **1.137432575413** | Match |
| 11 | §5, sealed seed3 q=256, case 2: **0.0990%** | **No per-case error row** | `F_seed3`, `old_q256_M1088_dense`, case 2 fields: **0.098963066611%**; agrees with `checks/final-sealed_seed3-audit.json`, `arms[].per_case_evolved_percent["2"]` | Raw value matches; summary gap |
| 12 | §6, sealed incumbent `fft_tight`: **115.159 ms** | `S[1216]`, `median_gpu_ms` | `F_incumbent`, median of 18 matching `gpu_seconds` values × 1000 = **115.159076522104 ms** | Match |
| 13 | §7, seed1 reachable q=32 ρ-max: **0.0948** | `S[1808]`, `rho_max` | `E₁`, `rules[q=32,population=reachable].certification.rho_max` = **0.094844672647** | Match |
| 14 | §7, seed2 reachable q=64 ρ-95: **0.1097** | **No `rho_p95` metric** | `E₂`, matching rule’s `certification.rho_p95` = **0.109676464297**; identical in `checks/s2-eqcert-audit.json` | Raw value matches; summary gap |
| 15 | §8 prose: **20 of 33** incumbent arms meet 1e-9 | Derived from incumbent `reference`, `all_times`, `evolved` rows and `comparators/qtd02-audit.json` | Recomputed comparison for `I₁–I₃`: **6 + 6 + 8 = 20**; largest relative discrepancy **1.336656293531e-08** | Match |

**None of these 15 printed numbers is numerically wrong. Two cannot complete the requested trace through `summary.json`.**

## Mismatches and reporting defects

**1. §9 incorrectly counts every recorded check as passed.**

`reports/generate_b_seeds.py:525` computes:

```text
len(checks) - len(failed)
```

But `failed` excludes non-blocking probes and unevaluated checks.

| Audit file under `checks/` | Report says | Actual `checks.*.passed`: true / false / null |
|---|---:|---:|
| `s1-ladder_incumbent-audit.json` | 45/45 | **41 / 3 / 1** |
| `s2-ladder_incumbent-audit.json` | 45/45 | **41 / 3 / 1** |
| `s3-ladder_incumbent-audit.json` | 45/45 | **42 / 2 / 1** |
| `s1-ladder_seed-audit.json` | 43/43 | **39 / 2 / 2** |
| `s2-ladder_seed-audit.json`, `s3-ladder_seed-audit.json` | 43/43 each | **40 / 1 / 2** each |
| `final-sealed_incumbent-audit.json` | 34/34 | **30 / 2 / 2** |
| `final-sealed_seed{1,2,3}-audit.json` | 33/33 each | **29 / 2 / 2** each |

For example, `s1-ladder_incumbent-audit.json` explicitly records `checks.incumbent_reproduces_qtd02_at_1e-9.passed=false`, `blocking=false`.

**No applicable blocking gate failed.** The defect is the claim that every recorded check passed.

**2. The summary does not cover every reported quantity.**

Missing metrics include `per_case_evolved_percent`, `rho_p95`, and NNLS `relative_fit`. Their values exist in the audits/artifacts, but the promised report → summary → artifact trace is incomplete.

**3. The report omits the operative F2 conclusion.**

The report correctly prints C2 and C2n as failures, but never states **“F2 applies”**, never expressly demotes development results, and does not explain the incumbent’s failed generalisation claim. Those disclosures exist in `DESIGN.md` A5 and `checks/retractions.md`.

## Gate audit

| Verdict or claim | Design requirement | Independent assessment |
|---|---|---|
| **C1: yes, 2/3** | §6; A1.20–21: evolved monotonicity **and every rung converged** on the same two seeds | Correct. All three ladders are monotone; seeds 1 and 2 fully converge. |
| **C2: no** | §6: every seed-mean and incumbent sealed/dev ratio ≤1.5 | Correct. Incumbent q=0 is **5.353133357**; largest seed-mean ratio is **1.401032271**. |
| **C2n: no** | A1.3: corresponding error/best-found ratios ≤1.5 | Correct. Incumbent q=0 is **5.337**; largest seed-mean ratio is **1.413682298**. |
| **C3: no** | §6: every primary rung converges on every checkpoint/cohort | Correct. Development seed3 q=256 fails on cases 2 and 3; sealed seed2 q=64 fails on case 4. |
| **C4: yes, 2/3** | §6: converged non-dominated primary set spans ≥2× error and cost | Correct. Seed1 spans **4.067566× / 16.258454×**; seed2 **4.007464× / 10.336649×**. Seed3’s converged subset spans only **1.767120×** in error. |
| **TR: 3/3** | §6: best-found ≤1.5× incumbent; floor ≤2× incumbent | Correct. Exact bars are **3.816994241% / 0.783689291%**; all seeds pass. |
| **F1 / F3** | C1 fails / at least two seeds fail TR | Neither fires. |
| **F2** | A1.3: both C2 and C2n fail | **Fires. Its reporting consequence is omitted**, although A5 explicitly preserves it. |
| Monotonicity counts | Both ladders, evolved and all-times metrics | Reported development counts **3/3** and sealed primary counts **4/4** agree. |
| EQ “certified” labels | §5: ρ-max ≤0.116 | All **24** labels agree with raw `rules[].certification.rho_max`. |
| Sealed opening / provenance | §3; A1.5–6; A4 | Records support compliance: no sealed configuration in seed manifests; six development flags true, four sealed flags false; seed checkpoint hashes match across cohorts. |

The two convergence failures are directly supported by raw invocation fields:

- `L₃`, `old_q256_M1088_dense`, cases 2/3: `budget_exits=1` per invocation; worst joint stationarity **1.521169e-06 / 5.831415e-06**.
- `F_seed2`, `old_q64_M320_dense`, case 4: `budget_exits=1`; worst joint stationarity **2.861839e-06**.

Each repeats identically across all three repetitions, against `gtol=1e-6`.

### Amendment timing

| Amendment | Evidence and timing | Post-result gate change? |
|---|---|---|
| **A1** | Commit `70ea41f2`, September 17, 05:26 EDT, before seed submission. Weakened fidelity 1e-9→1e-3; introduced C2n and changed F2 to require both failures. | **No production results yet.** Material changes, disclosed. |
| **A2** | Commit `46bd35c8`, 09:29 EDT. Changed cohort equality from bytes to ≤1 ulp after a smoke-test discrepancy. | **After smoke evidence, before production results.** All development blocks also pass original bitwise equality, so no verdict was rescued. |
| **A3** | Commit `cff8df73`, 09:59 EDT. Deferred Codex audit and required explicit fidelity disclosure. | Procedural change before results. |
| **A4** | Commit `32070874`, 18:16 EDT, after seed results. Recorded convergence failure and sealed opening. | No threshold change; budget stayed frozen. |
| **A5** | Commit `be9415ab`, 22:00 EDT, after sealed results. Recorded C2/C2n/C3 failures and F2. | No threshold change; added missing summary ratio rows. |

**No post-production-data threshold change was found that converts a failing verdict into a pass.**

## Cross-job cost check

**Pass: no cross-job or cross-GPU cost ratio found.**

| Attempt | Job | GPU | Cost comparisons |
|---|---|---|---|
| s1 | 3783776 | A100 80GB PCIe | Seed1/incumbent; within-ladder spans |
| s2 | 3783777 | A100 80GB PCIe | Seed2/incumbent; within-ladder spans |
| s3 | 3783778 | A100-PCIE-40GB | Seed3/incumbent; within-ladder spans |
| final | 3804465 | A100-PCIE-40GB | All sealed ladder spans and controls |

All six T12 cost means average **ratios formed within each job**. They do not divide timings from different allocations. Development/sealed ratios compare errors only.

Timing-definition clarification: reported GPU medians pool **18 invocations—six cases × three repetitions**. The report’s “medians of three timed repetitions” wording should specify that case aggregation.

## Retraction completeness

**Incomplete. The report has integrity notes, but no retractions section.**

| Required disclosure | Report coverage |
|---|---|
| A1/A3: substituted Claude audit; deferred Codex review | Present in §8 |
| A1.1: weakened fidelity gate and original-tier failures | Present, with per-arm evidence |
| A1.3: changed F2 reading through C2n | C2n is shown; the revised falsification consequence is unexplained |
| A2: byte-hash→value gate; TR/F3 display correction | Absent |
| `checks/retractions.md`: initial CUDA smoke failure, mesh-specific audit filename bug, smoke-duration deviations | Absent |
| A4: seed3 development convergence failure | Aggregate flags present; failed cases and stationarity values absent |
| A5: **F2 applies; sealed results become headline** | **Absent** |
| A5: sealed seed2 convergence failure | Aggregate flags present; failed case and stationarity absent |
| A5: incomplete lab entry and missing summary-ratio correction | Absent |

Additionally, `reports/self-audit-report.md:53` still says the sealed job is pending. Its final paragraph records post-result EQ-label and lab-helper bugs that are missing from both the report and `checks/retractions.md`.

## Three weakest claims

1. **C1/C4 as evidence for a general correction-rank knob.**  
   The passing ladder changes both q and M. At fixed M=256, the three seeds’ error spans are only **1.364×, 1.356×, and 1.220×**. The registered scheduled-ladder verdict passes, but attributing its full benefit to q alone exceeds this experiment.  
   **Reviewer request:** a sufficiently large fixed-M ladder across seeds, separating correction rank from test-space growth.

2. **Seed-mean transfer as evidence of checkpoint robustness.**  
   At q=0 the seed-mean sealed/dev ratio is **1.401**, but individual ratios are **0.837, 1.617, and 2.042**. Two retrained checkpoints exceed 1.5 even before considering the incumbent’s **5.353**. This satisfies the specified mean-based component, but supports a narrower claim than uniform transfer.  
   **Reviewer request:** explicit per-checkpoint ratios, failure counts, and uncertainty over independent cohorts.

3. **C2n as “difficulty-normalised.”**  
   Its numerator is worst **evolved same-grid** error; its denominator is worst best-found error against the reference, potentially at another case/time and including t=0. The denominator also depends on the trained checkpoint and optimisation. It is therefore an imperfect measure of cohort difficulty.  
   **Reviewer request:** matched case/time/reference normalisation, plus a difficulty measure independent of checkpoint fit.