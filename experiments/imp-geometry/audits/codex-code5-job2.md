Read-only audit of `9efb19514`; no writes, jobs, or array copies. Paths below are relative to `experiments/imp-geometry/`.

**Part 1 — four code items**

- **CORRECT — FD/SEM Q1 comparison gate:** `finalize.py:126–131` includes comparison validity in `passed`. In-memory probes reject failed comparison solves for both families.
- **CORRECT — box test eligibility:** `decide.py:110–115` checks test-reference eligibility before attribution, including validation losses.
- **CORRECT — pooled validity:** `ladder.py:63–65` pools counts across cases. Probe `[11,0]` failures over `[1000,1000]` steps correctly passes.
- **WRONG — inherited failure handling:** `cluster/job.sbatch.tmpl:23–25` fixes functions but not propagation from subshells/command substitutions. Reproduction: `(false; echo continued)` leaves parent `FAILED=0`; likewise `x=$(false; echo continued)`. Inherited traps update only the child’s counter. Explicit solver exit checks in these two bodies remain effective.

**Part 2 — step-1 blockers**

- **meas6 cannot be collected:** `cluster/collect.sh:11` unconditionally copies `out/cell`, but `cluster/bodies/smoke6.sh:3–4` produces only `out/probe`. Collection exits before pulling logs/results or deleting the job directory. Add a probe-job collection path.
- **Pulled files lack checksum verification before deletion:** `cluster/collect.sh:11` verifies the remote cell archive, but lines `13–15` pull files and delete the job without checking local hashes. Logs/probe outputs outside the cell archive consequently lack verified preservation.
- **Recertification command points at the wrong directory:** collection preserves `out/cell/` beneath `runs/cells/L_wide_S1` (`cluster/collect.sh:13`), whereas `cluster/RELEASE-PLAN.md:19` passes its parent. `resources.py:26` then cannot find `stage_seconds.jsonl`.
- **A8’s per-fit ECSW measurements are absent:** `run_cell.py:315` discards fit metadata and records no fit duration. `pod.py:147–148` returns termination information but no elapsed time. Aggregate ladder timing at `run_cell.py:887–889` cannot establish completed-fit costs or distinguish censored fits. Persist duration and termination/support metadata for each fit.
- **meas6’s stated individual cap is unenforced:** `cluster/RELEASE-PLAN.md:9` retains ≤0.15 GPU-h, but requests 12 minutes and `smoke6.sh:2–4` permits two sequential 300-second probes, already 10 minutes before overhead. The combined requests are 11.7 GPU-h, within A8’s 12-hour step cap; the individual allowance still needs reconciliation.
- **Submission currently refuses the lane:** tracked `HEARTBEAT.json` is modified; `cluster/submit.sh:16` rejects all tracked lane changes. Resolve this before submitting.

**Closed or appropriately deferred**

- Committed template/body staging is repaired (`cluster/submit.sh:24–25`); GPU preflight, precision, directory isolation, locked concurrency caps and queue checks are present.
- Bank measurement now uses within-run synchronized intervals and retains repetition arrays (`probe.py:162–170`; `bank.py:122–124`).
- Separate stage processes record device/host peaks (`run_cell.py:884–889`); A8 makes this first cell the measurement step. Later-panel scaling is not accepted by this audit.
- Frozen-artifact restoration exists (`cluster/bodies/test.sh.in:3–5`); test-job qualification is outside these two submissions.

CODE: NEEDS-FIX (subshell/command-substitution failure propagation)
JOBS(step 1): NOT-READY (meas6 collection; local checksum verification; recertification path; ECSW fit records; meas6 allowance; dirty-tree submission guard)