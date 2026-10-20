**E1: ACCEPT-WITH-FIXES.**  
**E5: ACCEPT-WITH-FIXES.**

The numerical verdicts reproduce. Required fixes concern reporting, including one incorrect plotting statistic. E2 was excluded.

1. **CORRECT — Job integrity and provenance.** All eight named jobs have GPU/x64/highest evidence, completion markers, `ALL-DONE`, matching commit/job identities, and separate job directories. All **451 retained output files** are git-tracked and pass their checksums. Staged Python sources match their recorded commits. The external `W_e1.npz` matches `83d5dbb2…1d98ea5`; its accompanying result JSON matches the committed archive. No GPU fallback, OOM, disk-full, or truncated execution was found. E13d’s CPU-affinity warning did not prevent GPU execution. Evidence: [job ledger](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/runs/GPU-LEDGER.json), archived logs and checksum listings.

2. **CORRECT — Analyses reproduce from archived outputs.** CPU reruns completed in scratch:
   - E1 2D: maximum numerical difference \(1.12\times10^{-15}\); labels and gates unchanged.
   - E1 3D scoring: all **14,240 case/trajectory records** reproduced within \(2.84\times10^{-15}\).
   - E1 3D analysis: differences within \(2.23\times10^{-15}\); integrity checks and labels unchanged.
   - **E1 verdict roll-up reproduced byte-for-byte.**
   - E5 reproduced exactly apart from four directory strings. A separate implementation also reproduced all ten \(\phi\) estimates and bootstrap intervals within \(10^{-14}\).

3. **CORRECT — Amended rules and registration history.** The analyses use 38 cases at every 2D mesh, 64 cases in 3D, and 16 draws. Eligibility, one-sided Clopper–Pearson bounds, rank intervals with \(\delta=0\) for distances and \(2\max\epsilon_{\rm ref}\) for errors, and H-E1c margins agree with the amendments. Controls and required gates pass. All regenerated draw hashes match. No post-production change to decision thresholds was found. The cancelled `e1m4096` output is excluded; A10.3’s resource correction and A11’s supplementary additions are disclosed.

4. **CORRECT — E1 2D qualification labels.** Independent checks of raw validity flags, certificates, distances, and indicators recover:

   | Configuration | Failures at 256² / 1024² / 4096² | H-E1a |
   |---|---:|---|
   | BE ×1 | 1 / 1 / 1 | Inconclusive |
   | BE ×2 | 2 / 2 / 2 | Fragile |
   | BE ×2.5 | 3 / 2 / 3 | Fragile |
   | CN-R ×1 | 1 / 1 / 1 | Inconclusive |

   **Every failure comes solely from draw-matched successor distance.** All primary draws pass the independent-target distance bar; their largest target distance is approximately \(2.020\times10^{-4}\), below \(2.5\times10^{-4}\). These labels therefore do not establish accuracy sensitivity or failure against the independent family. E1’s every-case successor test is stronger than qualify’s historical subset test. Evidence: [2D analysis](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/checks/e1-2d-256.json) and corresponding other-mesh files.

5. **CORRECT — E1 typicality, accuracy, and supplementary separation.** All twelve primary 2D cells are typical and accuracy-insensitive. Registered 3D configurations a/b both qualify on **16/16 draws**, with every eligible case within the accuracy margin. Configuration b is correctly **atypically unfavourable solely through \(D_0\)’s rank of 16/16**; its median/worst E0-error rank intervals are insensitive. Its \(D_0=2.04065\times10^{-4}\) still passes qualification. Supplementary c/d are separately flagged, also robust/typical/insensitive, and excluded from the registered population. Their unavailable historical replay gates are explicitly identified. Evidence: [E1 verdicts](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/checks/e1-verdicts.json).

6. **WRONG — The report generator’s plotted “median.”** [make_report.py:131](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/make_report.py:131) uses `sorted(e)[len(e)//2]`. Both cohorts have even sizes, so this selects the upper middle observation rather than the `np.median` estimator used by the analyses. It overstates the plotted 256² BE×1 median by approximately **6.79%**. Fix this before generating/publishing the plot. **The audited hypothesis labels are unaffected.**

7. **NEEDS-FIX — E1 reporting omits registered qualifications and summaries.** The generator prints only the all-draw confidence bound, omitting the fresh-draw bound required by A1.5. For the robust 3D rows these are **0.170750 versus 0.181036**; draw 0 is selection-conditioned. Also carry the registered median/IQR/min–max summaries, failed-draw counts, Sobol diagnostic spreads, and conditional-Richardson reference qualification into the report. The raw evidence exists. Audit-only primary/supplementary summaries are available in [scratch](/tmp/jcp-robust-audit-scratch/e1-descriptive-audit.json).

8. **CORRECT — E5 controls, populations, and estimators.** All four jobs pass required gates, with distinct physical GPUs within each type. All six cost cases and sixteen crossover cases survive the registered intersection. The 1,320 ordinary timing triplets are valid, without retries. Doubled-work medians span approximately **1.955–2.014**, inside the required band. Injected fixtures return \(\phi=1.5\); null fixtures return \([1,1]\); the drift fixture rejects its injected step. Pooling, paired case bootstrap, 10,000 resamples, seed 20261019, and classification boundaries match the registration. Evidence: [E5 analysis](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/checks/e5.json).

9. **CORRECT — E5 contains genuine hardware-dependent quantities.** Four intervals lie entirely outside \([0.8,1.25]\):

   | Quantity | \(\phi\) | 95% interval |
   |---|---:|---:|
   | R128 BE cost ratio, 256² | 1.644 | [1.631, 1.651] |
   | R128 BE cost ratio, 1024² | 1.634 | [1.592, 1.651] |
   | R128 / same-mesh CC speed-up ratio | 0.708 | [0.698, 0.713] |
   | R384 / spectral speed-up ratio | 1.766 | [1.747, 1.774] |

   The other six quantities are hardware-consistent. In particular, \(\phi_1\) compares costs normalized against the reference ROM; it is **not an absolute H200/A100 execution-time ratio**.

10. **CORRECT — E5 verdict agreement is narrower than quantity agreement.** Both H-cost-N(2) rows pass on both types. R128’s same-mesh S2 row passes on both; both spectral S2 rows fail on both. R384’s same-mesh row is **inconclusive on A100**, whose lower bound is **1.9993498**, and passes on H200. Its agreement is correctly inconclusive. Thus five agreement rows are verdict-robust and one is inconclusive; none has opposite definitive labels.

11. **NEEDS-FIX — Preserve E5’s scope and descriptive failures in publication.** [make_report.py:264](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/make_report.py:264) omits material information already present in the analysis: conclusions conditional on observed devices, frozen H200-selected comparators, per-job/unit-to-unit results, and **seven of twelve G-E5.2 timing-anchor cells outside P1’s intervals**. These anchor failures are explicitly non-blocking, so retaining the numerical verdicts is correct. Report them alongside the limited H-cost-N(2)/S2-subset scope.

No repository or lab-log files were changed, and no cluster/SSH commands were executed. Scratch usage was approximately **11 MB**. Remote cleanup state was not independently checked.