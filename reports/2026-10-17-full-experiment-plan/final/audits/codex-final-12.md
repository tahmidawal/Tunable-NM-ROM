OVERALL: CLEAN

Both round-11 NEEDS-RESTATEMENT items are fixed. I found no remaining material numerical, provenance, or scope defect.

Read-only throughout: no files changed, generator executed, jobs submitted, commits made, or scratch storage used.

Paths below:

- **F** = [final compilation](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final)
- **R** = [robust lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust)

| Item | Verdict | Independent check and file:line evidence |
|---|---|---|
| Round 11: E5 glossary | **CORRECT — fixed** | Now requires the same **definitive** label, pass or fail; inconclusive labels cannot establish robustness. `F/report.md:613`; matches `R/DESIGN.md:269–270`. |
| Round 11: SIREN/RBF definitions | **CORRECT — fixed** | SIREN is explained as a network with sinusoidal activations; RBF is expanded to radial basis functions. `F/report.md:608`. |
| E1 counts and verdict rows | **CORRECT** | Registered 2D: **6 fragile / 6 inconclusive**, 12 typical, 12 insensitive. Registered 3D: **2 robust**, 1 typical / 1 atypically unfavourable, 2 insensitive. Supplementary configurations are excluded. `F/report.md:69–70,363–364`; `R/checks/e1-verdicts.json:74`. |
| E1 “fragile” interpretation | **CORRECT** | Checked **192 primary 2D draws**. All **20 failures** exceed the successor-distance bar while passing the stored production-validity, stationarity, tight-distance, advection and independent-target checks. Maximum target distance: **0.000201975858653**, below 0.00025. `F/report.md:366`; `R/checks/e1-2d-{256,1024,4096}.json`, `groups.R128.arms.*.rules`. |
| E1 deployed-draw ranks | **CORRECT** | Registered 3D ranks are **9–9** and **16–16**, each among 16 draws. The unfavourably ranked deployed draw still qualifies. `F/report.md:366`; `R/report.md:68–69,97`; provenance at `F/PROVENANCE.json:7711`. |
| E2 totals | **CORRECT** | Recounting individual verdict rows gives **140/21/22 of 183** in 2D and **45/17/67 of 129** in 3D, ordered robust/dependent/inconclusive. `F/report.md:71–72,375–376`; `R/checks/e2-verdicts-{2d,3d}.json`, `rows`. |
| Section-1 headline | **CORRECT** | All **24** selected-rule 2D accuracy rows are seed-robust and agree with the incumbent; all **3** rule-selection rows are seed-dependent. `F/report.md:25,392`; individual verdict rows corroborate both counts. |
| Selected-rule table | **CORRECT** | All twelve selections reproduce from the four bank files. Incumbent R256 selects **fib10946**, distinct from deployed **gauss128**. `F/report.md:378–384`; `R/checks/e2-2d-{0,1,2,3}.json:3`. |
| Deployed-rule failures | **CORRECT** | At 1024², **step factor 1**, either LSPG scheme: R128 fails on seeds **1,2,3**; R256 on **none**; R384 on **1,2**. The report explicitly states this restriction. `F/report.md:386–392`; bank files’ `configurations` records, starting at `R/checks/e2-2d-1.json:52`. |
| E2 2D spread ratios | **CORRECT** | Recomputed from individual-bank median errors and paired changes: **4.56104135**, **2.53099230**, **2.58514790** for 1024², 256², 4096². Printed rounding matches. `F/report.md:392`; bank files’ `configurations` and `h_acc_r` records. |
| E2 3D improvement and spread | **CORRECT** | Fixed-gl64 FS-CN range is monotone on banks **0–3**. Largest seed max/min ratios are **1.008907189** for median error and **1.911376244** for worst error, both at W1024. `F/report.md:394`; `R/checks/e2-3d-{0,1,2,3}.json`, `h_acc_r` and `errors`. |
| Wide-width sparsity | **CORRECT** | Only bank **2** has a certified W768 selection; none has W1024. The report restricts the inconclusive statement to selection verdicts and makes no certified wide-width front claim. `F/report.md:394`; `R/report.md:148`; bank files’ `selection` arrays. |
| G-E2.2 anchor | **CORRECT** | Recounted **8 certified matches, 1 diagnostic match, 1 difference, 2 certified absences, 4 unverified absences**. `F/report.md:399`; `R/report.md:150`; `R/runs/EXPORT.json`, `G_E2_2`. |
| E5 quantities and agreement | **CORRECT** | Recomputed all ten numerator/denominator ratios and classified their intervals: **6 consistent / 4 dependent**. Dependent ratios: **1.6437174, 1.6344867, 0.7082791, 1.7658714**. Agreement independently gives **5 robust / 1 inconclusive / 0 contradictory**. `F/report.md:73,401`; `R/checks/e5.json:2080,2368,2506`. |
| E5 scope and anchor | **CORRECT** | Four distinct GPU UUIDs; two devices per type. Conditional scope, frozen H200-selected comparators and two-mesh timing scope are explicit. Anchor count is **7/12**, non-blocking. `F/report.md:401`; `R/checks/e5.json`, `jobs` and `gates`; `R/report.md:31`. |
| Scientific scope | **CORRECT** | Three seeds remain descriptive, not population evidence; E0 remains conditional. Rule transfer is correctly described as **not guaranteed**, preserving successful transfers. `F/report.md:357,392,401`; `R/report.md:29–31,105`; `LAB-LOG.md:19321–19332`. |
| Retractions | **CORRECT** | A17’s original roll-up and R256-front interpretation remain withdrawn. A19–A21 reporting corrections and all listed operational failures are retained. `F/report.md:477`; `R/report.md:640–648`; `R/DESIGN.md:1014–1057`; `LAB-LOG.md:19328`. |
| Deferred items | **CORRECT** | Five-seed extension, L40S and additional 3D draws are included, alongside coordinator **C3-full and C4**. `F/report.md:503–505`; `R/DESIGN.md:328–330`; `LAB-LOG.md:19329`. |
| Ten-lane ledger | **CORRECT** | Independent ledger sum: **168.7697 GPU-h**, printed as 169. Robust job entries sum to **46.4076 GPU-h**, with cap **95** and **54 ledger entries**. `F/report.md:520–533`; each lane’s `runs/GPU-LEDGER.json`. |
| Lane commits | **CORRECT** | All ten branch tips and ten local mirror refs match the displayed hashes. Robust is **cbf574373 / d49bd0085**. `F/report.md:542–551`; `LAB-LOG.md:19321`. |
| Provenance and figures | **CORRECT** | Verified **638 direct JSON value/path records** and **73 referenced file hashes** without discrepancies. Robust derived values reproduce independently. All three robust figures match their lane originals byte-for-byte. `F/PROVENANCE.json`; `F/make_final.py:1446–1652`. |
| Hand-typed numbers | **CORRECT for measured results** | No hand-typed measured robust result found. Design constants and sourced narrative text remain literal—including “two transient login-node failures”; this is corroborated by the lane record. `F/make_final.py:1107–1115,1143,1455–1457`; `R/report.md:646`. |
| Non-robust regression | **CORRECT** | Independently matched **24 numbers**: eight fitted exponents and sixteen interval endpoints. Sections **3.1–3.10 and 3.12 are byte-identical** to round-9 CLEAN commit `3634eb628`. `F/report.md:92–99`; P1 `runs/EXPORT.json:135,185`. |
| Regimes regression | **CORRECT** | Recomputed all seven regimes’ qualification and label counts, median-error ranges and baseline-ratio ranges. C1/C2 remain **0/8 qualified**; exploratory, conditional-reference and diagnostic-floor qualifications remain intact. `F/report.md:408–430`; regimes `runs/EXPORT.json:10,520`. |
| New glossary coverage | **CORRECT** | E1/E2/E5 labels, shifts, successor/target distances, rule selection, seed spread, hardware ratios and C3-full/C4 terminology are explained consistently with the lane definitions. `F/report.md:606–613`. |

This audit independently checks the compilation against stored lane outputs; it does not rerun solvers or reconstruct large field arrays.