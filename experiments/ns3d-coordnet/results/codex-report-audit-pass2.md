Line references below are to the [regenerated report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-ns3d-coordnet-bank/reports/2026-10-01-ns3d-coordnet-bank.md).

| Required fix | Verdict | Evidence |
|---|---|---|
| Qualify causal/general claims | **FIXED** | Lines 161, 169–175 label causation as inference, distinguish optimization/capacity/sampling, restrict conclusions to tested meshes, and say cancelled fits “would have tested.” |
| Distinguish original registration, A3, A4 | **FIXED** | Line 160 correctly separates original five-hour fits, A3 ten-hour penalized fits, and A4 pilot freeze. Matches DESIGN. |
| Generate numerical prose from sources | **NOT FIXED** | Pilot durations now use summary seconds, but generator lines 83–95 still hard-code architecture/data counts; thresholds and `report_plan.json`’s “roughly 25 minutes” remain literals. |
| Correct cancelled jobs and pilot durations | **FIXED** | Lines 163, 176, 194 distinguish cancellation from no computation. Pilot summary seconds round to the printed **28/39/42 minutes**. |
| Qualify test reuse and LM convergence | **FIXED** | Line 3 discloses parent-lane reuse; lines 150–155 and glossary say “generic LM,” without convergence certification. |
| Complete/correct glossary and math | **NOT FIXED** | Requested entries added, but MLP/SiLU/QR/SVD remain undefined; prose still uses unformatted `δ`, `Δt = 0.001`. “Near 1 means the solve adds little error” still overinterprets a ratio of potentially different worst cases. |

**Remaining unsupported wording:** line 42 still says larger ranks “would have answered” the required-rank question; unsuccessful fits need not answer it. No distinct new unsupported result claim found.

No edits, jobs, or pushes.