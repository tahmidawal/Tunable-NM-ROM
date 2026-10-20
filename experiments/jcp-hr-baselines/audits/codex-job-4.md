1. **CORRECT — bundle integrity.** Recorded commit: `dd65560426af6334af8b2e962af54c7c6f9d177e`. All 61 manifest entries match their SHA-256 hashes; no missing files, extras, symlinks or bytecode. The manifest itself is necessarily excluded. All 57 unchanged provenance payloads equal their Git blobs; the remaining payload is the config with only the recorded commit injected. Generated metadata and `run.sbatch` are staging products, not verbatim source blobs.

2. **CORRECT — batch script.** `runs/t3d64b/run.sbatch` requests `gpu`, `gpu:h200:1`, `200G`, and `12:00:00`, matching `wall_hours=12`. Logs, output and caches use the prescribed attempt directory. The prescribed venv, `highest`, x64, GPU preflight with exit 42, and `JCPREF_STORE` are present. Driver timeout is 42,960 seconds, with a 30-second kill grace; output checksumming follows into `OUTPUTS.sha256`. The four-minute margin also absorbs preceding shell/preflight time.

3. **CORRECT — configuration.** Staged config equals the committed `configs/tune_3d64.json` plus `commit`. It contains `ecsw_path_seconds=2400`, all nine comparators, ranks `[64,128,256,512,768,1024]`, and 12 scheme/timestep options. Budget/wall/search allocation are respectively `11.75`, `12`, and `12.216077777777778` hours. Neither `dry` nor `allow_cpu` is present.

4. **CORRECT — audit coverage and gates.** Staged `nnls.py`, `build.py`, `pipeline.py`, and `tune.py` are byte-identical to READY commit `7f5bb352e`. Relative to audit-23 commit `b08690fdb`, differences are:
   - `nnls.py`: blocked deletion, retained Givens reference, sign-normalized refactorization, cooperative deadlines and elapsed/budget evidence.
   - `build.py`: forwards the deadline budget and records timing evidence.
   - `pipeline.py`: forwards `max_seconds`.
   - `tune.py`: supplies `ecsw_path_seconds`.

   These are the changes reviewed through audits 28–30. Committed/staged `checks/test_core.json` records **15/15 PASS**, `all_pass=true`; every recorded source pin matches the staged bytes. The suite was inspected, not rerun.

5. **CORRECT — ledger.** Exactly one unbound `t3d64b` reservation exists, for 12 hours. `t3d64a` is reconciled to `0.8875` GPU-hours, matching archived `sacct.txt`: **00:53:15 × one GPU**. Measured expenditure plus outstanding reservations is **19.3995 hours**; adding the 7.5-hour Stage-B reserve gives **26.8995 ≤ 30**.

6. **CORRECT — submission controls.** `cluster/submit.sh` places both account-count `<8` and lane-count `<2` checks, together with `sbatch`, inside `/cluster/tufts/paralab/tawal01/.campaign-sbatch.lock`. It includes manifest hash/coverage checks, reservation `submit-check`, job-ID `bind`, and `--no-requeue`.

7. **CORRECT — nominal wall-time room, not guaranteed completion.** The cancelled attempt reached `ecsw_states` completion at **1,008.4 seconds**. Six 2,400-second NNLS budgets total **4 hours**, or **8 hours** if every path retries. Against the driver timeout, that leaves approximately **7.65 hours**, or **3.65 hours with all retries**, for subsequent work before additional overhead. Thus the registered Stage-A grid has nominal room. These are cooperative budgets: contribution assembly, deadline overshoot, terminal evidence and artifact writes consume additional time. Neither the logs nor the budget arithmetic proves that all 1,008 configurations will finish.

No files changed, scratch created, jobs submitted, or GPU work run.

**SUBMIT-t3d64b**