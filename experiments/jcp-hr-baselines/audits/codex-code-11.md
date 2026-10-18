1. **MAJOR — CORRECT: round-10 query-failure blocker is fixed.**  
   `experiments/jcp-hr-baselines/cluster/collect.sh:14` now captures `sacct` in a standalone assignment under `set -euo pipefail` (`:4`). SSH or scheduler failure aborts execution. The explicit terminal-state whitelist (`:15–16`) rejects empty, unknown and running states before transfer or deletion. Read-only shell reproductions confirmed these outcomes; `COMPLETED` and `FAILED` reached collection. **Fix:** none for the reported failure.

2. **MINOR — NEEDS-RESTATEMENT: “any ambiguous query refuses” is broader than the implementation.**  
   `collect.sh:11–14` selects the first log filename and first accounting row without validating a unique numeric job ID or matching an accounting `JobID`. There is also no terminality recheck before deletion (`:25`). This does not reproduce round 10’s failed-query bypass under the registered one-job-per-directory workflow. **Fix:** validate one numeric ID, request and match `JobIDRaw,State`, reject multiple allocation records, and repeat verification before deletion.

3. **MINOR — WRONG: remote manifest pipeline still lacks explicit failure propagation.**  
   At `collect.sh:17`, local `pipefail` does not govern the remote `find | sort | xargs` pipeline. Upstream remote failures can produce an incomplete manifest despite successful downstream execution. Successful `rsync` and local checksum verification remain mandatory (`:19–20`). **Fix:** execute manifest generation inside explicit remote `bash -o pipefail`.

**Overall: READY for smoke `smk2`, within this targeted round-11 review of `7d89c5bf3`.** The round-10 blocker is resolved; no remaining smoke blocker was established. This is not a fresh certification of the numerical implementation or phase-II drivers.

No files changed, jobs submitted, or GPU work performed.