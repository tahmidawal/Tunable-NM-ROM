Read-only audit of `d8c5867ff`; no writes, submissions, or array copies. Paths relative to `experiments/imp-geometry/`.

Prior blockers:

- **WRONG — inherited failure propagation:** `cluster/job.sbatch.tmpl:25` still increments only the child’s counter. In-memory reproduction of `(false; echo continued)` and `x=$(false; echo continued)` leaves parent `FAILED=0`. The scan at `:29` cannot detect swallowed failures that produce no nonzero `rc` record. Explicit solver exit checks remain effective.
- **CORRECT — meas6 collection:** `cluster/collect.sh:13` makes cell archival conditional; `:18` collects probe outputs.
- **CORRECT — local checksum verification:** `cluster/collect.sh:18` generates remote hashes; `:19` verifies pulled files before deletion at `:21`.
- **CORRECT — recertification path:** `cluster/RELEASE-PLAN.md:19` now supplies `runs/cells/L_wide_S1/out/cell`, matching `resources.py:26`.
- **CORRECT — ECSW fit records:** `run_cell.py:315–320` persists individual duration, termination metadata and support sizes; `pod.py:147–148` supplies termination reason and support reached.
- **CORRECT — meas6 allowance:** `cluster/RELEASE-PLAN.md:9` requests nine minutes; `cluster/bodies/smoke6.sh:2` limits each probe to 220 seconds. Combined requested allocation is 11.65 GPU-h.
- **CORRECT — dirty-tree guard:** `cluster/submit.sh:16` excludes `HEARTBEAT.json`. The exact guard passes with the current heartbeat-only modification.

New blocking issue:

- **Collection can delete unarchived full-cell arrays:** `cluster/collect.sh:13` treats SSH failure/timeout exactly like an absent `out/cell`. If connectivity recovers, `:18` pulls only files below 100 MB, then `:21` deletes the entire job directory without archiving larger arrays. Distinguish confirmed absence from transport failure; require successful archival for the full-cell job.

CODE: NEEDS-FIX (subshell/command-substitution failure propagation)  
JOBS(step 1): NOT-READY (collection must abort on archive-detection transport failure)