Reviewed HEAD `9afee6d8`, read-only. No files changed, SSH used, GPU execution, or git writes. I independently recomputed selections and gates from raw JSONs, then reproduced `summary.json`, `tables.generated.md`, and the report **byte-for-byte in memory**, with writes disabled.

Paths below are relative to `experiments/poisson-bank-knob-3d/`; “report” means `reports/2026-09-23-lshape-cube-bank-truncation-knob.md`.

1. **CORRECT — Every main “Table-1 rows” entry is audit-PASS and gate-clean in its stated scope.**

   All seven published entries have **32 distinct pairs per arm: 16 cases × 2 rounds**, with no duplicate pair keys. The insufficient-coverage original l256 run is excluded. Publication filtering is implemented at `make_report.py:47–56`; offline gate recomputation and acceptance are at `make_tables.py:135–145,187–190`.

   My independent recomputation gives:

   | Run | Scope | Pooled paired ratio | Median after/main | Maximum arm ratio | Verdict |
   |---|---|---:|---:|---:|---|
   | l2048b | total_seconds | 1.102338 | 0.970257 | 1.155730 | **FAIL** |
   | l2048b | fused_device_seconds | 1.009782 | 1.010333 | 1.192157 | **PASS** |
   | c128 | total_seconds | 1.011649 | 1.009286 | 1.027072 | **PASS** |
   | c128 | fused_device_seconds | 1.009539 | 1.003719 | 1.017475 | **PASS** |

   Evidence: `runs/l2048b/archive/output/result.json:116674`, `runs/c128/archive/output/result.json:45640`; audit verdicts at both `audit.json:2`. These match `report:222,225`.

   **Remaining implementation defect, not a defect in these runs:** `pbk3_core.py:207–218,232–235` counts zipped observations, rather than enforcing unique case/round coverage. Repeating one identical pair 32 times passes. Thus the missing-arm fix works, but the function still does not fully enforce A1’s distinct 16×2 design.

2. **CORRECT — l2048b’s exclusion and measured component attribution are supported.**

   Median paired after-CG minus control differences recompute to:

   | Component | Difference, ms |
   |---|---:|
   | Input transfer | 0.123165 |
   | Device work | 0.047115 |
   | Output transfer | 1.163768 |
   | Total | 1.412440 |

   These match `report:7`. The output interval specifically measures `jax.device_get(out)` (`pbk3_lshape.py:100–111`). Calling this an effect concentrated in the device-to-host output interval is supported; identifying a deeper hardware cause would exceed the evidence. Component medians need not sum to the median total difference.

3. **CORRECT — GPU-scope selection, FOM selection, and arithmetic.**

   Independently aggregating the invocation arrays and applying the rule gives:

   | Run | Accurate: error %, ms | Fast: error %, ms | FOM: error %, ms | Accurate / fast speedup |
   |---|---|---|---|---|
   | l2048b | R384_q64: 2.120097%, 6.429604 | R128_linear: 3.029607%, 3.692967 | cg_0.03: 0.760756%, 311.117806 | **48.388331× / 84.246027×** |
   | c128 | R128_linear: 0.144229%, 0.909755 | R64_linear: 0.241271%, 0.578756 | cg_0.01: 0.074758%, 12.418161 | **13.650013× / 21.456627×** |

   These match `report:51,29`. The generator genuinely rebuilds subjects and reselects development arms in the alternate scope (`make_tables.py:203–219`), rather than merely relabelling the original ratios. Raw invocation arrays begin at `runs/l2048b/archive/output/result.json:318` and `runs/c128/archive/output/result.json:580`.

4. **NEEDS-RESTATEMENT — Attribution arithmetic is correct; some descriptions are not.**

   The truncation ratios use the full-rank **same-family, same-q** twin for NM-ROM arms and the full-rank linear rung for linear arms. Both numerator and denominator use the displayed scope (`make_report.py:158–178`). All displayed times and ratios reproduce.

   However:

   - `report:61` defines “vs paper setting” as using the **same q**. That is false for several rows: for example, selected q=64 versus parent q=128 at L-shape 1024², and linear q=128 versus parent q=96 on the cube (`report:69,73`). Say **the paper’s designated accurate or fast setting**.
   - The claim that the historical paper used q=64 at L-shape 256²/512² is **hardcoded**, not established by the run JSONs (`make_report.py:170–174,180–182`). The arithmetic against `orig_q64` is correct; that historical label needs a pinned paper-table citation. I cannot certify it from the scoped evidence.
   - “The part that R′ alone buys” is stronger than warranted without the layout qualification already given much later at `report:206–213`. Bring that qualification beside the definition.

5. **CORRECT — Final Table-1 settings are frozen development choices; final cohort matches seed 920499.**

   Both final archives embed exactly the corresponding `frozen-N32.json` / `frozen-N64.json`; their development-result hashes match. Both parameter arrays match regeneration from **seed 920499, count 64, bit-for-bit**, with SHA256 `27ec2cf52d2eb0ad84d6b3e83c46f504a67c0dd56a7a1f968263d3179430be60`.

   Freeze commits `f6c10ffe` and `d2c775a9` precede final staging; both final jobs record source commit `d2c775a9`. Evidence: `frozen-N32.json:3–8`, `frozen-N64.json:3–8`, both final `result.json:296,751`, and `pbk3_cube.py:161–172`.

   The generator uses frozen settings (`make_tables.py:191–197`). Crucially, c64final retains **R64_linear**, although retrospective GPU-scope selection chooses **R48_linear** (`reports/tables.generated.md:357–359`).

   **Wording exception:** `report:44` says selection is “re-applied entirely” in the alternate scope, but final rows deliberately retain frozen choices (`make_tables.py:207–208`). State that exception explicitly.

6. **NEEDS-RESTATEMENT — No numerical transcription/arithmetic mismatch found, but several claims remain vulnerable.**

   - **Usable-only publication is still incomplete outside the main table.** `report:39` presents c32final complete-query speedups **2.27× / 2.32×** without qualification, although that scope fails with maximum arm ratio **1.483469**. The failure appears later at `report:52,223`. The unfiltered output comes from `make_report.py:125–127`. Remove or explicitly label these as failed-gate diagnostics. The opening “every gate” statement at `report:3` also needs scope qualification.
   - **“Dominated by host transfers” is too broad** (`report:57`). For l256b’s accurate arm, median device work is **1.908539 ms** versus **2.626583 ms** total; median input/output intervals are only **0.319602/0.407451 ms**. Qualify by mesh and arm.
   - **The L-shape NM-ROM truncation benefit is modest:** **1.02×, 1.03×, 1.05×, 1.12×** for the selected accurate arms (`report:65–71`). Those numbers support a modest deployed timing tradeoff, not a large NM-ROM acceleration attributable to truncation. The larger fast-arm changes also drop the head. Small timing differences have no reported uncertainty establishing their significance.
   - The L-shape linear ladder is not monotone: R′=256 is markedly less accurate than R′=128 (`report:94–95,105–106`). Avoid implying that retaining more columns guarantees a better solved field.
   - `PROGRESS.md:24–27` remains stale: it describes c256 as pending and c128 as next, while both are collected. Its earlier six-pair acceptance statement at lines 17–18 is corrected later, but should be explicitly marked superseded.
   - A skeptical reviewer can reasonably request a timed spectral cube baseline. The report already limits its claims to tested CG settings (`report:82`); preserve that limitation.

**Concrete defects remaining:** duplicate-pair coverage loophole; unqualified failed-scope speedups; incorrect “same q” and final-scope reselection wording; unpinned historical paper-rank attribution; overbroad transfer/truncation attribution; stale progress status. **No incorrect main Table-1 row or requested speedup arithmetic found.**