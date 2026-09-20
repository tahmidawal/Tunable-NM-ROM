**The numerical audit passes: all 15 sampled numbers match, and no reported cost ratio crosses jobs or GPUs. The report needs corrections to its P3 verdict, construction-status labels, and retraction disclosures.** P1 and P2 remain supported for the specific rules measured.

This was read only; no files were changed and no GPU work or jobs were run. Error values were checked against the archived audit JSON and its per-time arrays; I did not independently recompute fields or certification residuals.

**Verified numbers**

Paths below are relative to `experiments/b-eqtop/`. Abbreviations:

- `S`: [reports/summary.json](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-eqtop/experiments/b-eqtop/reports/summary.json).
- `R101` / `A101`: `artifacts/bet101/result.json` / `audit.json`, job **3780164**.
- `R201`: `artifacts/bet201/result.json`, job **3780165**.
- `R301`: `artifacts/bet301/result.json`, job **3783811**.
- Report line numbers refer to [2026-09-17-b-eqtop.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-eqtop/experiments/b-eqtop/reports/2026-09-17-b-eqtop.md). Array indices are zero-based.

All entries below **match within displayed rounding**.

| # | Report location and number | Summary key | Artifact key and checked value |
|---|---|---|---|
| 1 | Title/L22: top construction passes **1/5** draws | `rows[10].value=0.2`, `draws=5` | `R101.rules[4].certification.rho_max=0.1073668739`; four `R301.rules[q=256, arm=reprow64*]` values are `0.1820128561, 0.1899115968, 0.1298824390, 0.2420930284`. Exactly one is ≤0.116. |
| 2 | L21: exported q128 rule, **ρmax=0.0277** | `rows[9].value` | `R101.rules[10].certification.rho_max=0.027690277590777298` |
| 3 | L45: q0 primary evolved error **1.8891%** | `rows[12].value` | `A101.arms[4].worst_evolved_percent=1.8891456982772188` |
| 4 | L49: q128 primary all-times error **1.8116%** | `rows[29].value` | `A101.arms[11].worst_all_times_percent=1.811592916083009` |
| 5 | L35 prose: q256 primary evolved error **0.5389%** | `rows[32].value` | `A101.arms[15].worst_evolved_percent=0.5389317143107647` |
| 6 | L50: q256 primary cost **722.2 ms** | `rows[35].value` | `A101.arms[15].median_gpu_ms=722.2137585049495`; independently reproduced as `median(1000 × R101.invocations[name=q256_eq_primary].gpu_seconds)` |
| 7 | L60: q128 tight evolved error **0.8944%** | `rows[52].value` | `A101.arms[13].worst_evolved_percent=0.8944431892364022` |
| 8 | L80: q256 dense cost **3915.1 ms** | `rows[95].value` | `A101.arms[14].median_gpu_ms=3915.141240460798`; independently reproduced from its 18 raw timings |
| 9 | L86: `fft_loose` evolved error **0.0338%** | `rows[96].value` | `A101.arms[0].worst_evolved_percent=0.03384942323363766` |
| 10 | L122: q128/fs64 replication sample SD **0.0751** | `rows[114].value` | Sample SD of `R301.rules[q=128, arm=reprow64*].certification.rho_max` = `0.07513552920392098` |
| 11 | L126: q64/fs64 replication spread **9.57×** | `rows[150].value` | Maximum/minimum of `R301.rules[q=64, arm=reprow64*].certification.rho_max` = `9.56890052130386` |
| 12 | L165 prose: q0/std/m512 **ρmax=0.0641** | `rows[520].value` | `R201.rules[5].certification.rho_max=0.0641477527697432` |
| 13 | L170 prose: q256 incumbent best **ρmax=0.1308** | `rows[470].value` | `R101.rules[18].certification.rho_max=0.13084679583283124`; minimum among its incumbent rules |
| 14 | L193: q128/fs64/m2048 NNLS fit **3.52e−4** | `rows[363].value` | `R101.rules[6].fit.relative_fit=0.0003522284941771245` |
| 15 | L351: q256/fs64 slope **α=2.268** | `rows[989].value=2.26830076211687` | Independently computed from `R101.rules[q=256, arm=fs64]` at m2048/2560: `2.2683007621168594` |

Additional checks: all **695** summary rule-metric values match the raw results; all **17** timing medians reproduce exactly; all **17** error maxima reproduce from the archived audit’s per-time arrays. The report SHA256 matches `S.report_sha256`.

**Mismatches**

1. **P3 is a different criterion in the report.** [Report L32](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-eqtop/experiments/b-eqtop/reports/2026-09-17-b-eqtop.md:32) labels “tight ladder monotone” as P3 and says **yes**. [DESIGN §7](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-eqtop/experiments/b-eqtop/DESIGN.md:170) instead defines a conditional test of the primary bar’s sufficiency at q256. That condition did not trigger. DESIGN A3 correctly says **“P3 is moot.”** The report should say **not triggered**, with five-rung tight monotonicity reported separately. No amendment authorizes this relabeling.

2. **A T9 construction status disagrees with the summary and uses the wrong population.** Report L157 labels the q16 **static**, m512 secondary rule `[certified in one draw]`. `S.rows[264].construction_status=null`; `R101.archived_rules[3]` has:
   - `certification.rho_max=0.12786323452392534`
   - `certified_primary=false`
   - `certified_secondary=true`.

   `draws.py:26` keys constructions without population, while its draw collection excludes static rules. The report’s status lookup therefore borrows the reachable construction’s status. The q32 static row similarly says `[no draw]` despite an archived rule being present.

3. **Tight-column status is ambiguous.** Report L157 labels q16/std/m1024’s tight-certified entry `[confirmed (3/3)]`, but L136 reports only **1/3 tight passes**. The 3/3 count is for the primary bar. It should explicitly say “primary construction status” or use the tight-bar status.

There are **no numerical mismatches among the 15 sampled values**.

**Gate audit**

| Verdict or gate | Applicable DESIGN rule | Audit finding |
|---|---|---|
| P1: primary rules exist | §7: q128 and q256, m≤6144, ρmax≤0.116, untruncated | **Pass for the measured draws.** Both qualify at m2048: `0.0669086267` and `0.1073668739`. |
| P2: primary ladder passes | §7: six monotone evolved errors; zero budget exits; worst joint gradient≤1e−6; cheaper than measured dense twins | **Pass.** Errors decrease `1.8891457 → 1.4269573 → 1.2492799 → 1.2274713 → 0.8936 → 0.5389317%`. Raw primary invocations have zero budget exits; maximum joint gradient is `9.996126384625569e−7`. All four available dense comparisons are cheaper. |
| P3 “yes” | §7: conditional primary-bar sufficiency test | **Incorrect label/verdict.** Condition not triggered; no q256 tight-certified rule exists. |
| Tight ladder monotone | §6 tight arms; §7 conditional follow-up | **True for the five displayed rungs only**, q0–128. Not evidence for tight certification at q256. |
| Hybrid ladder monotone | §7 falsification fallback | **True but duplicates the primary ladder.** All six entries use EQ; the dense-fallback condition never triggered. |
| Dense ladder monotone/converged | §6 dense controls | **True for the four measured rungs**, q0/64/128/256. The templated “cheaper than dense: no” is self-comparison at ratio 1, not a failed scientific gate. |
| Per-rule primary/tight/secondary flags | §5 thresholds plus nontruncation | **All 417 flag comparisons agree** across the three results, including archived-rule recertifications. |
| Marginal timed-rule constructions | A2: mixed pass/fail draws imply marginal status | **Counts agree:** q64 2/6, q128 4/5, q256 1/5. Interpret pooling qualifications below. |
| Exported choices/statuses | A4 export policy | **Consistent with A4, but the policy was chosen after results.** It replaces the q64/128/256 rules used for timing. |
| “No blocking gates failed” | §8 | **Matches all 72 displayed gate rows** and each artifact audit’s `failed=[]`. Hash mismatches were informational from the original design. |

The artifact checks also record `reference_residuals.detail=9.875956374345395e−12`, below `2e−11`, and archived-rule recertification differences below `1e−6`. Saved-field and bank checks are recorded passes; this audit did not rerun those computations.

Amendment timing is traceable:

| Amendment | Timing evidence | Effect on verdicts |
|---|---|---|
| A1 | Commit `ace8936e`, 05:24; submissions 05:31, September 17 | Before jobs: self-audit substitution, arm handling, cap raised 5400→7200 s. |
| A2 | Commit `b2844c60`, 09:29; bet301 submitted 09:31 | **After bet201 data**, but before replication. Adds the marginal-status reporting rule; explicitly leaves original pass/fail unchanged. |
| A3 | Commit `d6071e3c`, 10:55, after bet101 landed | Records provisional P1/P2 passes and P3 being moot. |
| A4 | Commit `fc7ca639`, 12:01, after bet301 landed | **Post-data definition of “confirmed” and export selection.** DESIGN explicitly discloses this; the report does not state its post-data timing. |

The original DESIGN §§1–9 are unchanged in git history. **No lowering of the original P1/P2 thresholds was found.** The post-data export policy must not be presented as a pre-registered success criterion.

**Cross-job check**

All reported online cost ratios use **bet101/job 3780164, NVIDIA A100 80GB PCIe**:

| Pair | Numerator / denominator, ms | Recomputed ratio |
|---|---:|---:|
| q0 primary / q0 dense | 59.069202 / 292.872016 | 0.201689 |
| q64 primary / q64 dense | 120.625367 / 624.928582 | 0.193023 |
| q128 primary / q128 dense | 246.869689 / 1190.488409 | 0.207368 |
| q256 primary / q256 dense | 722.213759 / 3915.141240 | 0.184467 |
| q128 tight / q128 dense | 250.758338 / 1190.488409 | 0.210635 |

Each median uses 18 records: six cases × three repetitions. `R201.invocations=[]` and `R301.invocations=[]`, so neither contributes online timing denominators.

The cross-job ratios elsewhere measure **ρ**, not cost. No prohibited cost ratio was found. Archived `qrg304` fit times are inherited historical measurements, not new fit timings in the recertifying job.

**Retraction completeness**

**Incomplete. The report has no retractions section.** The requested `checks/retractions.md` does not exist; the actual ledger is [checks/retractions.json](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-eqtop/experiments/b-eqtop/checks/retractions.json). Its disclosures appear in `checks/lab-entry.md:102`, rather than in the audited report.

| Ledger/amendment item | Coverage in report |
|---|---|
| `retractions[0]`: withdraw m≈2522, α=1.772 crossing prediction | **Partial:** generic slope warning at L344; no explicit withdrawal of that prediction. |
| `[1]`: withdraw incumbent fit-state convention at q≥128 | **Missing explicit withdrawal.** |
| `[2]`: walltime sizing wrong; gradient saturation prevented target supports | **Partial:** support counts and gradient explanation appear, without the correction to the sizing claim. |
| `[3]`: ρ need not decrease with m | Values appear; explicit correction absent. |
| `[4]`: 64 fit states are not uniformly better | Values appear; explicit qualification absent. |
| `[5]`: withdraw construction-wide certification at every rung | **Substantively present** in title, status tables and conditional wording. |
| `[6]`: superseded exported rules | Old/new choices shown; explicit supersession and post-data timing missing. |
| `[7]`: monotonicity applies to passing draws; failing draws not timed | Passing-draw restriction present; unmeasured failing-draw comparison not explicit. |
| `[8]`: four recovered 8.50-GiB allocator warnings on bet301 | **Missing.** “No blocking gates failed” does not disclose this operational exception. |
| `[9]`: no jobs retracted; three of eight submissions used | Jobs listed, but this explicit disposition is absent. |
| A1/A3: independent audits unavailable; self-audits substituted; local runtime deviations | **Missing from the report.** |

The ledger itself also needs clarification: `[3]` calls q128/std “marginal” based on different support targets, whereas the report’s formal status for its passing target is “certified in one draw.”

**Three weakest claims and reviewer questions**

1. **“Confirmed” means reliable independent replication.** The arithmetic is correct, but the pooled evidence is weaker than that wording suggests. Report L134/L136 counts q0/q16 as 3/3, although `R201.designs[std/fs64].candidates_sha256` is identical within each rung: those two fits share a candidate pool. Historical draws also use 8192 candidates versus 16384 here. A reviewer would ask: **What are the pass rates for independent draws under one fixed pool-size recipe, with uncertainty intervals?**

2. **The primary bar is sufficient or predictive of trajectory error.** P3 is untriggered, and q256 has only two timed EQ examples: `ρmax=0.1073669 → 0.5389317%` and `0.1677660 → 1.0361161%`. The replication’s failing draws were not timed, and the same certification cohort was reused. A reviewer would ask: **Does the threshold predict errors across passing and failing draws on a fresh certification cohort?**

3. **The exported top-rung rules inherit the measured ladder result.** The timed q128/q256 rules use m2048; exports use q128/rhow64/m2319 and q256/fs64/m2560. A4 selected these after the replication, and both remain single-draw constructions. A reviewer would ask: **Where are repeated certification results and same-job timed rollouts for the exact exported rule hashes?**