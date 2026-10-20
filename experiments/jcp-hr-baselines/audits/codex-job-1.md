1. **WRONG — manifest coverage, both attempts.** Both record commit `40fd986bafde069c063bc97744aabd993053bef7`. All 58 provenance entries validate; all 61 manifest hashes match. Source blobs match `git show`, allowing the intended config `commit` injection. However, **each bundle contains 34 unmanifested `.pyc` files**, including `experiments/jcp-hr-baselines/__pycache__/tune.cpython-312.pyc`. Thus the manifest does not cover the staged tree. `cluster/submit.sh` transfers these through unrestricted `rsync -a`. Remove the bytecode from both bundles and recheck exact manifest coverage before submission.

2. **CORRECT — batch scripts, both attempts.** `run.sbatch:3–36` requests `gpu`, `gpu:h200:1`, `200G`, and respectively `12:00:00` / `05:00:00`. Logs, output and caches use each attempt’s paralab directory. The prescribed venv, highest precision, x64, GPU preflight exiting 42, and `JCPREF_STORE` are present. Driver timeouts are 716 / 296 minutes, with a further 30-second kill grace; `OUTPUTS.sha256` follows. Submission uses `sbatch --no-requeue`.

3. **CORRECT — configurations.** Both equal their committed configurations after removing the injected commit:
   - **t3d64a:** ranks 64/128/256/512/768/1024; BE/CN/BDF2 × four timesteps; five diagnostic batch-1 and four qualified batch-2 comparators; allocation **12.2160777778 h**, budget **11.75 h**, wall **12 h**.
   - **t2d1024a:** ranks 32/64/128/256/384, extension rank 512; BE/CNR at 0.005; eight P1 batch-1 comparators; allocation **1.9797222222 h**, budget **4.75 h**, wall **5 h**.
   
   These match DESIGN’s amended contracts, including A26.4, A28.4 and A29. Neither enables `dry` or `allow_cpu`.

4. **CORRECT — tuning dependency inventory.** Both bundles include the driver, `ours`, `s3common/qn3/fs3p`, `p1core/p1common`, `c3common`, transitive vendored runtime modules, banks, rules, input JSONs, and E0 loader/metadata. E0 arrays are intentionally external through `JCPREF_STORE`; their remote availability was not verified.

5. **CORRECT — reservations and budget.** `runs/GPU-LEDGER.json` contains exactly one unbound reservation per attempt: **12 h** and **5 h**. Measured smoke expenditure is **1.512 h**. Total including Stage-B reserve: **1.512 + 12 + 5 + 7.5 = 26.012 GPU-h ≤ 30**. Including the separately planned 2-hour 256² tuning job gives **28.012 GPU-h**.

6. **CORRECT — submission safeguards, with scope limitation.** `cluster/submit.sh` invokes `submit-check`, checks `squeue` before submission and again under its remote lock, refuses lane counts ≥2 or account counts ≥8, binds the returned job ID, and checks `squeue` afterward. `ledger.py:51–58` prevents reservation reuse. The campaign check is a snapshot; its lane-local lock cannot prevent another lane submitting concurrently.

7. **CORRECT — audit applicability to tuning sources.** All provenance-listed committed files match `b08690fdb`, covered by code audit 23’s **READY** verdict. Relative to design audit 21’s `47bc19088`, differences are:
   - `tune.py`: eligibility masks, batch labels, evidence persistence, inventories and accounting fixes.
   - `checks/test_core.json`: refreshed source pins and NNLS regression evidence.
   - `stageb.py`: replay, timing, checkpoint and memory changes; outside these tuning jobs.
   - DESIGN: acceptance record and A32–A34.
   
   These tuning changes are covered by code audit 23 and preserve the accepted ordering. Since that code revision, `cluster/collect.sh` also gained missing-output handling and validated accounting with an absolute Python path; these are nonblocking collection changes. Stage B remains separately gated.

8. **CORRECT — resource evidence, with unresolved production memory risk.** Expected GPU: **NVIDIA H200**. Smoke job **5422908** reports H200 with **143771 MiB**, GPU backend, x64, and successful completion. `RESOURCES-tune.json` records a **107.181 GB** device peak for 2D1024; the 3D peak is masked by that process-cumulative measurement. Higher-rank production memory is therefore unproven; the design’s 120 GB criterion is retrospective. Projected durations are **9.806 h / 3.151 h**, not completion guarantees. Original smk2 acceptance failed one ECSW gate; `ACCEPTANCE-A23.json` passes under the disclosed amended rule.

No files changed, jobs submitted, GPU work run, or scratch created. The submission blocker is item 1.

**DO-NOT-SUBMIT-t3d64a**  
**DO-NOT-SUBMIT-t2d1024a**