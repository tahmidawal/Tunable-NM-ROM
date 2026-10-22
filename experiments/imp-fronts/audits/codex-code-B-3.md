**Overall: WRONG — admission, dry-run validation, and verdict defects remain.**
Checks: **148 pytest tests passed**, `resources.py --selftest` passed, training `--static-only`: **0 failures**. No numerical GPU probes.

Paths relative to `experiments/imp-fronts/`. Every B-2 WRONG item is covered below; B-2 had no NEEDS-RESTATEMENT items.

| B-2 item | Verdict | Fix status and adversarial evidence |
|---|---|---|
| Cost certificate | **WRONG** | Producer→consumer schema is fixed, including 1,152 + 4,032 scaling in `train_item`. But `resources.py:106–118` accepts a single arbitrary population with **1 measured trajectory / 5,184 production trajectories** as certified. Require the two registered populations and exact counts. |
| Dry-run evidence completeness | **WRONG** | Unique policy IDs are fixed. However, marking **every tight record skipped** and removing its timing rows still passes every `check_dry` check (`evidence.py:62–82`). Invented terminal labels also pass. Require executed mandatory solves and validate decisions against the state machine. |
| Admission | **WRONG** | Bare certificates and missing completed-job charges now fail. But `admit(ledger_with_20_hours_spent, None, 'SM-b', NaN)` returns **admit=True**: NaN bypasses ceiling and cap comparisons (`resources.py:438–459`). Require finite positive proposed ceilings before arithmetic. |
| Test-once | **NEEDS-RESTATEMENT** | Atomic claims fix repeated execution into different empty directories. However, `testonce.bound():101` accepts a copied `RUN.json` outside the claimed directory because it checks only key/job. Collector directory binding remains unimplemented; define how verified local relocation is allowed. |
| B0 route integrity | **WRONG** | Unscheduled IDs, unregistered rules and noninteger node counts are rejected. But `validate():262` accepts prefixed IDs without normalizing downstream lookups: identical synthetic evidence gives **CONTINUE** with numeric IDs and **INCONCLUSIVE** with `FI|4`-style IDs. Normalize consistently or reject aliases. |
| Training smoke recipe | **CORRECT** | Amendment 15 explicitly registers the N64/16-trajectory/20-step chaining slice used by `train_body.sh`; the B-2 specification mismatch is resolved. |
| Smoke failure propagation | **CORRECT** | `smoke_datagen.py` now exits nonzero on failure; chaining/timing failures block `ALL-DONE`. `phase_times.py` rejects failed gates, incomplete chaining and mismatched job/device provenance. Numerical passage remains untested here. |
| Training timing projection | **NEEDS-RESTATEMENT** | Paired generator timing and burn-in fix the original comparison. But `phase_times.py:141–158` still counts seed-0 once and uses rounded **4.17 h**. Probe: summary says **4.3922 h, fundable**; `train_item` correctly computes **4.6161 h**, above the **4.5667 h** cutoff. Derive both from one formula. |
| Training cutoff | **CORRECT** | Amendment 15 registers the 120-second reserve; shell cutoff and certificate limit agree. |
| Staging/static regression | **CORRECT** | Static checks retain baseline/provenance integrity and declared arm changes. `stage_job.sh` includes and hash-checks the incumbent generator needed by paired timing. Staging/push was not executed. |

No repository, main, or LAB-LOG changes; no SSH, jobs, or sealed/OOD access. Test paths were redirected in memory; caches/bytecode disabled. Scratch was approximately **29 MB** and deleted.