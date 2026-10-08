**J3b’s reported numerical gates pass.** The main issues are interpretation and incomplete memory accounting. No files were modified.

I recomputed the gates from [training.json](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/runs/j3b/archive/output/training.json) and checked the archived code, logs, checksums, and [report section](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md:76).

| Item | Verdict | Evidence / qualification |
|---|---|---|
| Archive and installed bank | **CORRECT** | All six `OUTPUTS.sha256` entries verify. Installed `bank.pkl` and `training.json` match their archived copies. Bank SHA256: `86094f6884e55643240197e560f7dc1924bac3885aed79b20faf2e54446aa036`. All five staged provenance files match their recorded hashes and commit contents. |
| Execution evidence | **CORRECT** | Job 5012847, H200, commit `156599de0`; log confirms GPU, x64, highest precision, and completion. Archived stderr files are empty. |
| B1 checkpoint selection | **CORRECT** | Minimizing the maximum validation error over groups across all 13 recorded checkpoints selects **11000**, with **1.3062892767%**. Step 10000: 1.3099135990%; step 12000: 1.3352713022%. The final training objective is lower at 12000, but that is not the selection criterion. BANK records agree across JSON and both training logs. |
| B2′ training checks | **CORRECT** | Inverse residual **5.8349005786×10⁻¹² ≤ 10⁻⁸**; `condition_65` **10145.8073613 < 10¹⁰**, giving singular-value ratio **9.8562881×10⁻⁵ > 10⁻¹⁰**. |
| B2′ as complete deployment certification | **NEEDS-RESTATEMENT** | These pass the **J3 portion**. A2-B2 also requires J4’s ordered-bank mesh Gram-condition gates at both meshes; J3 does not establish those. |
| B3, 129 nodes | **CORRECT** | New 512 floor **3.2305140370%**, old **3.1216058933%**; threshold **3.4337664827%**. Ratio **1.034888499 ≤ 1.10**. It passes the comparison flag while being **3.49% relatively worse**, not better. Both use the same regenerated fields. |
| B4 strict decrease | **CORRECT** | **3.2305140370% > 1.9277203774% > 1.2889770849%** for 512 → 768 → 1024. |
| B4′ material gain | **CORRECT** | **1.2889770849% ≤ 2.5844112296%**. Ratio 1024/512 = **0.3990006142**, a **60.10% reduction** in worst projection floor. |
| Step-rate gate, A2-16 | **CORRECT** | BANK seconds: **1038.587479848** at 1000 and **1317.704554206** at 2000. Difference/1000 = **0.2791170744 s/step**; 12000-step projection **0.9303902479 h < 7 h**. Actual recorded bank time **1.1400 h**, total **1.8501 h**, agree with the report’s rounding. |
| Compare-bank ranks above 512 | **CORRECT** | JSON explicitly lists **640, 768, 896, 1024** as unavailable; comparison floor dictionaries stop at 512. The report correctly marks its 768/1024 rows unavailable. |
| Dropped POD tail | **CORRECT** | Mean discarded normalized training energy: **1.2756495×10⁻¹⁰**, **3.9105755×10⁻¹⁰**, **8.4034070×10⁻¹⁰** at 33/65/129 nodes. These are squared-energy means, not worst validation errors. At 129, they concern the sampled training grid. Whitening is also recorded: **0.0888610677** at the selected checkpoint. |
| Numerical report table | **CORRECT** | All displayed new/compare floor values agree with JSON after rounding. |

**Reporting and interpretation findings:**

- **WRONG — applying the report’s general reference/metric definitions to J3.** These floors use **native-grid bank-validation fields**, not the 513-node refined reference. The full-grid metric covers **96 cases × six times, including time zero**, normalized by each snapshot’s norm. Checkpoint selection instead uses twelve snapshots per case and the group’s sampled points at 129 nodes. The glossary’s “five evolved output times” definition does not describe J3.

- **NEEDS-RESTATEMENT — host-memory compliance.** The separate [RSS log](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/runs/j3b_rss/sstat_rss.log) contains 96 samples, approximately one minute apart. Its largest reported MaxRSS is **287467092 KiB = 274.15 GiB ≈ 294.37 GB**, comfortably below the requested `420G`. The step-rate note’s **“287.5 GB” is an incorrect conversion**. There is no archived `output/rss.log`, `/usr/bin/time -v` summary, or final `sacct` record in the supplied evidence. Thus sampled accounting supports feasibility, but the promised A4-J3 whole-process and phase-resolved measurement protocol is not fully evidenced.

- **WRONG — “lower span floor implies a more accurate ROM.”** This establishes better best-approximation capacity on these validation fields. Reduced dynamics, solver behavior, quadrature, and reference error can prevent that capacity from producing better rollout accuracy. J3 supplies no rollout result.

- **NEEDS-RESTATEMENT — “fair equal-width comparison.”** It is a valid **equal deployed-column-count** comparison on identical fields. It does not isolate training rank or capacity: network width, bank rank, POD truncation, seed, and effective whitening weighting differ. The report’s “single-seed, capacity-scaled baseline-recipe” label is appropriate.

**Statements safe to report:**

> On the native-grid bank-validation fields at 129 nodes, this single-seed capacity-scaled bank’s worst projection floor decreases from 3.2305% to 1.9277% to 1.2890% as its deployed prefix grows from 512 to 768 to 1024 columns. B3, B4, B4′, the J3 numerical-stability checks, and the training-rate gate pass. Its 512-column prefix is 3.49% relatively worse than the old bank on the same fields, within the registered 10% allowance. These results establish span improvement, not improved ROM rollout or physical accuracy. Available sampled memory accounting remains below the allocation; complete A4-J3 memory evidence is missing.