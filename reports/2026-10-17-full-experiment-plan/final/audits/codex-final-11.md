OVERALL: NOT-CLEAN

The numerical results reproduce, and all round-10 findings have been addressed. Two glossary issues remain; neither changes the computed verdicts.

Read-only throughout: no files changed, generator run, jobs submitted, commits made, or scratch storage used.

Paths below:

- **F** = [final compilation](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final)
- **R** = [robust lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust)

| Item | Verdict | Evidence |
|---|---|---|
| E5 glossary decision rule | **NEEDS-RESTATEMENT** | `F/report.md:613` says verdict-robust means both GPU types give “the same label.” The registered rule requires the same **definitive** label; two inconclusive labels remain inconclusive. See `R/DESIGN.md:269–270` and `R/e5analyze.py:338–344`. Replace with “the same definitive label (pass or fail); otherwise inconclusive unless the definitive labels contradict.” |
| New glossary completeness | **NEEDS-RESTATEMENT** | C4 is now explained, fixing round 10, but its new explanation introduces **SIREN** and **RBF** without defining either: `F/report.md:608`. Expand these as a neural network using sinusoidal activations and radial basis functions, respectively. |

The remaining checks passed:

| Item | Verdict | Recomputed result and file:line evidence |
|---|---|---|
| E1 registered verdicts | **CORRECT** | 2D: **6 fragile / 6 inconclusive**, 12 typical, 12 insensitive. Registered 3D: **2 robust**, 1 typical / 1 atypically unfavourable, 2 insensitive. Supplementary configurations excluded. `F/report.md:69–70,363–364`; `R/report.md:21–22`; `R/checks/e1-verdicts.json`. |
| “Fragile” interpretation | **CORRECT** | Examined **192 primary draws**. All **20 failures** exceed only the successor-distance criterion; maximum independent-target distance is **0.000201975858653**. Stored stationarity, tight-solve and advection checks support that reading. `F/report.md:366`; `R/report.md:97`; `R/checks/e1-2d-{256,1024,4096}.json`. |
| E1 ranks and round-10 provenance fix | **CORRECT** | Deployed ranks reproduce as **9–9** and **16–16**. Both rank arrays and draw counts now have file/key/hash records. `F/report.md:366`; `F/PROVENANCE.json:7711–7758`; `R/checks/e1-verdicts.json`. |
| E2 totals and verdict rows | **CORRECT** | Recounted individual rows: **140/21/22 of 183** in 2D; **45/17/67 of 129** in 3D. Headline and selected-rule subsets also match. `F/report.md:71–72,375–376`; `R/checks/e2-verdicts-2d.json:4755`; `R/checks/e2-verdicts-3d.json:2399`. |
| Section-1 headline and scope | **CORRECT** | All **24** selected-rule accuracy rows agree across the three retrains and with the incumbent; all **3** rule-selection rows are seed-dependent. Description versus population and conditional E0 qualifications remain explicit. `F/report.md:25,357,392`; `R/report.md:23,29–31`. |
| Selected-rule table | **CORRECT** | All twelve selections match the four individual-bank check files, including incumbent R256 **fib10946**, distinct from deployed **gauss128**. `F/report.md:378–384`; `R/checks/e2-2d-{0,1,2,3}.json`, `selection`. |
| Deployed-rule failures: round-10 fix | **CORRECT** | At **step factor 1**, failures are R128: seeds **1,2,3**; R256: **none**; R384: **1,2**. The table now states the restriction and either-scheme aggregation. `F/report.md:386–392`; individual-bank `configurations` records. |
| Rule-transfer conclusion: round-10 fix | **CORRECT** | Now says transfer is **not guaranteed** and requires qualification on each bank, preserving successful transfers. This correctly narrows the lane/coordinator shorthand. `F/report.md:392`; compare `R/report.md:29,648`. |
| E2 2D spread ratios | **CORRECT** | Independently recomputed from bank median errors and paired changes: **4.56104135**, **2.53099230**, **2.58514790** for 1024², 256², 4096². Printed rounding matches. `F/report.md:392`; `R/checks/e2-2d-{1,2,3}.json`. |
| E2 3D spread and sparsity | **CORRECT** | Fixed-gl64 FS-CN improves through every width transition on banks 0–3. Maximum seed max/min ratios: median **1.008907189**, worst **1.911376244**. Only bank 2 has a certified W768 selection; none has W1024. `F/report.md:394`; `R/checks/e2-3d-{0,1,2,3}.json`. |
| G-E2.2 omission: round-10 fix | **CORRECT** | Restored **8 certified matches, 1 diagnostic match, 1 difference, 2 certified absences, 4 unverified absences**. `F/report.md:399`; `R/runs/EXPORT.json:10722`; `R/report.md:150`. |
| E5 numbers and labels | **CORRECT** | Recomputed all ten numerator/denominator ratios and interval classifications: **6 hardware-consistent / 4 hardware-dependent**. Independently classified agreement: **5 verdict-robust / 1 inconclusive / 0 contradicting**. `F/report.md:73,401`; `R/checks/e5.json`, `phi`, `per_type`, `agreement`. |
| E5 scope: round-10 fix | **CORRECT** | Report and export now specify **four physical GPUs**, conditional conclusions, frozen **H200-selected** comparators, and the two-mesh timing test. Four distinct device UUIDs verified. `F/report.md:73,401`; `F/FINAL-EXPORT.json:369`; `R/checks/e5.json`, `jobs`. |
| G-E5.2 omission: round-10 fix | **CORRECT** | Recomputed **7/12** H200 anchors outside P1 intervals; explicitly non-blocking. `F/report.md:401`; `R/checks/e5.json:2941,3068`. |
| Retractions | **CORRECT** | A17’s roll-up and R256-front interpretation remain withdrawn. A19–A21 corrections and all listed operational failures are retained. `F/report.md:477`; `R/report.md:640–648`; `R/DESIGN.md:1014–1057`; `LAB-LOG.md:19328`. |
| Deferred items | **CORRECT** | Five-seed extension, L40S and additional 3D draws are present, together with coordinator **C3-full and C4**. `F/report.md:503–505`; `R/DESIGN.md:328–330`; `LAB-LOG.md:19329`. |
| Ten-lane GPU ledger | **CORRECT** | Independent sum: **168.7697 GPU-h**, printed as **169**. Robust job entries sum to **46.4076 GPU-h**, cap **95**, **54 ledger entries**. `F/report.md:518–533`; `R/runs/GPU-LEDGER.json:645`. |
| Lane commits | **CORRECT** | All ten displayed branch tips and ten mirror refs match git. Robust: **cbf574373 / d49bd0085**, matching the coordinator. `F/report.md:542–551`; `LAB-LOG.md:19321`. |
| H-E1c glossary: round-10 fix | **CORRECT** | Margin now uses the converged-rule error, **\(e_{\rm conv}\)**. `F/report.md:607`; `R/report.md:660`. |
| Regression against round 9 | **CORRECT** | Sections **3.1–3.10 and 3.12 are byte-identical** to CLEAN commit `3634eb628`. Independently checked **24 non-robust numbers**—eight exponents and sixteen interval endpoints—and recomputed all seven regimes’ qualification/label counts and error/ratio ranges. `F/report.md:92–99,412–418`; P1 export `:135,185`; regimes export `configs` and `coverage`. |

**Provenance: CORRECT.** Verified **638 direct JSON value/path records** and **73 referenced disk-file hashes**, including derived-source references. All three robust figures match their lane originals byte-for-byte. No hand-typed measured robust result found.

These checks used stored lane outputs independently of `make_final.py`; they did not rerun solvers or reconstruct large field arrays.