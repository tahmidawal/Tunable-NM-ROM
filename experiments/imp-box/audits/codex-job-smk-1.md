**DO-NOT-SUBMIT yet.** The core submission mechanics are sound, but failed-job collection has concrete gaps, and the pilot can report `JOB DONE` without running controls.

Audited HEAD `d7206bcf7`, including the round-6 audit and subsequent fixes. No files changed, jobs submitted, large arrays allocated, or lab-log writes made.

1. **Sbatch — CORRECT for execution; NEEDS-FIX for complete failure handling.**

   [job.sbatch.tmpl](experiments/imp-box/cluster/job.sbatch.tmpl) requests `gpu`, one specified GPU, and checks for H200 at runtime. It uses the absolute cluster interpreter, sets `JAX_DEFAULT_MATMUL_PRECISION=highest`, and exits 42 on GPU-preflight failure. `cell.py:29` enables x64 before computation.

   Logs and results resolve under `/cluster/tufts/paralab/tawal01/imp-box/<job>/`; the execution `flock` prevents simultaneous execution in that directory. Stage exit codes, `FAILED`, and the inherited ERR trap provide normal failure accounting.

   **Gap:** hardware/preflight failures occur before the failure-file machinery and before `out/cell` exists. Slurm still records failure, but the collector currently refuses these smoke jobs—see item 6. SIGKILL/OOM also cannot rely on Python’s `finally` for memory evidence.

2. **submit.sh — CORRECT for these proposed submissions.**

   [submit.sh](experiments/imp-box/cluster/submit.sh:18) stages the recorded commit using `git archive`; template and body also come from that commit. Approval checks bind name, body SHA-256, GPU, memory, and wall limit. Both body hashes exactly match those supplied.

   Both reservation and submission acquire the shared campaign lock and recheck account `<8` and lane `<3`. They are **separate lock acquisitions**, despite the introductory comment; rechecking immediately before `sbatch` preserves the caps. Directory reservation and the `submitted` sentinel prevent ordinary duplicate submissions. Queue queries occur before and after submission.

   Live read-only check found **one account job, zero imp-box jobs**, H200 resources in the GPU partition, and neither smoke directory present.

   `RELEASE.json` is currently empty, so submission correctly refuses until the planned approval is committed. Approval does **not itself calculate GPU-hour consumption**; that remains the release decision’s responsibility.

3. **Bodies — CORRECT configuration; NEEDS-FIX for pilot completion semantics.**

   Both bodies specify the correct W40/A2 or W40/A4 cell and all seven stages, including controls. Neither uses `--smoke`, overrides training size/reference resolution, or sets debug variables. Defaults select 96 training cases, full bank training, and the TRANS arm. A nonzero stage exit increments `FAILED`, stops subsequent stages, and fails the job.

   **Exception:** [cell.py:432](experiments/imp-box/cell.py:432) skips `ladder`, `time`, and `controls` when references are unscorable, returning success each time. The shell can consequently print **`JOB DONE` without controls or all stage markers**. The resource certificate rejects this later, but the smoke has not fulfilled its purpose.

   **Fix:** require all seven completion markers and successful controls before declaring the pilot successful, or propagate an explicit nonzero pilot-incomplete status.

   Also reject or clear inherited debug overrides before work starts. In particular, `ladder.py:23` consumes `IGEO_NCAP` even without `--smoke`. The current certificate detects this only after spending the allocation.

4. **Smoke evidence — CORRECT if completed; NEEDS-FIX for robust failure evidence.**

   The intended successful outputs cover the pilot’s central questions:

   - `controls.json` has an explicit mandatory registry; failure exits nonzero.
   - `stage_seconds.jsonl` records elapsed time, host peak RSS, and JAX allocator device peaks per stage.
   - A4 uses **N_ref=768**, with spatial, temporal, padding and retry gates.
   - Cohorts, bank metadata, stage markers, coverage/termination files, frozen policy/source identity and accounting supply certificate inputs.

   Boolean/selector controls appropriately use fixtures; the numerical controls use generated cell data. “Every control on real data” should not imply every registry entry is a physical-data experiment.

   **Limitations:** device peak retrieval silently becomes zero if unavailable; peaks are written only when the stage exits through Python. A timeout/OOM can therefore erase the most relevant stage’s peak evidence. Preserve external sampling/accounting, and distinguish unavailable peaks from measured zero.

   Runtime remains unmeasured. Bank training logs progress but saves its model only after training and ordering; a bank timeout leaves progress evidence but no completed bank. ECSW’s 1,800-second budget is checked **after matrix construction** (`podsine.py:222–251`), so it is not a hard stage bound. I found no defensible basis to claim either wall limit will necessarily be exceeded—or that both jobs will finish.

   Round-6 certificate hardening has improved. However, `resources.py:56–67` still does not reconcile expected coverage against all termination records, and it does not validate memory peaks. These are **production-release follow-ups**, not reasons to demand completed pilot measurements before running the pilot.

5. **Budget — CORRECT for the pilot pair; full-campaign projection unestablished.**

   With the stated zero spend:

   | Reservation | GPU-h |
   |---|---:|
   | smk-A2 wall limit | 12 |
   | smk-A4 wall limit | 24 |
   | Pair maximum | **36** |
   | Remaining from 87 | **51** |

   Even applying 15% headroom to both limits and reserving one additional largest-job failure gives **65.4 GPU-h**, below 87.

   Thus the pair has a defensible **bounded pilot exposure**, independent of runtime estimates. It does not establish affordability of the remaining campaign. Keep subsequent releases gated on collected costs and revised scope.

6. **collect.sh / ledger.py — NEEDS-FIX for reliable collection within one hour.**

   [collect.sh](experiments/imp-box/cluster/collect.sh:14) checks terminal state, verifies archive and local checksums, and deletes only after verification. `ledger.py` correctly uses allocated GPU count × elapsed seconds and replaces duplicate job IDs.

   Concrete gaps:

   - **Early failures cannot be collected under their required names.** The no-cell exception accepts `meas*|smoke*`, not `smk-*` (`collect.sh:19`). A preflight failure therefore blocks collection of its logs.
   - **OOM state recognition is fragile.** `sacct -o State` uses display formatting without an explicit sufficient width; a truncated `OUT_OF_MEMORY` will not match the case statement. Request an adequately wide state field.
   - **Collection retries can self-hash `.pull-sums`.** After an interrupted attempt, the existing checksum file enters the next `.pull-list`, then is overwritten while being hashed (`collect.sh:27`). Explicitly exclude both pull bookkeeping files.
   - The sequential timeouts permit **more than one hour**, and neither script establishes completion monitoring or automatically updates the ledger. Archive transfer size/throughput has not been measured.

   Arrange terminal-state monitoring and immediate ledger/log collection, with bulk archival following. Preserve scheduler failure evidence before deleting the job directory.

**Required before submission:** repair failed-job collection and retry bookkeeping; enforce explicit pilot completion; prevent inherited debug settings; establish failure-memory evidence and the one-hour collection procedure. Then commit approval for the exact final specs and recheck the changed execution path. **DO-NOT-SUBMIT.**