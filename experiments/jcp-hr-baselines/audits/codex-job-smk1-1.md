**DO NOT SUBMIT as staged.** The resource request and numerical inventory are correct, but `wall_seconds=3300` does **not** enforce a deadline or protect the checksum window. Completing the full workload within 55 minutes is also unsubstantiated.

Audited HEAD: `cc8e435e09f3d5ceebbde10d8c6ab421b5df3d6b`. No files changed, jobs submitted, GPU work run, or scratch created.

| Item | Verdict | Evidence |
|---|---|---|
| Slurm resources | **CORRECT** | `run.sbatch:3–8` requests `gpu`, one H200, 200G host memory, eight CPUs, and 60 minutes. The requested allocation is **1 GPU-h**. |
| Environment and GPU preflight | **CORRECT** | `run.sbatch:13–27` uses the prescribed cluster venv, enables f64 and highest matmul precision, and exits 42 unless JAX selects GPU. `smoke.py:23,664–667` reinforces these settings. |
| Output paths | **CORRECT** | Logs, output arrays, JSON, temporary files, and `OUTPUTS.sha256` target `/cluster/tufts/paralab/tawal01/jcp-hrb/smk1`. E0 is an external **read-only input**. |
| All caches confined there | **NEEDS-RESTATEMENT** | `XDG_CACHE_HOME` and `TMPDIR` are pinned, but the script does not pin or clear other cache overrides, including `CUDA_CACHE_PATH` and `JAX_COMPILATION_CACHE_DIR`. “All caches stay here” is stronger than the script establishes. |
| Manifest integrity | **CORRECT** | Independently hashed all **49 manifest entries**: every digest matches. |
| Staged files equal HEAD blobs | **NEEDS-RESTATEMENT** | **45 of 46 source-derived files** match HEAD byte-for-byte. The config is deliberately transformed by `cluster/stage.py:75–78`: it adds `commit` and serializes JSON. Removing that field produces exactly the committed configuration. The generated wrapper and provenance files are manifest-bound, not HEAD blobs. |
| A5.4 / A11 frozen inventory | **CORRECT** | Every `FROZEN` key matches the staged config: meshes, trajectory counts, maximal ranks, smoke ranks, sampling parameters, fixtures, options, and 1,024-point NNLS pilot support. The additional `ranks` lists also match A5.4. |
| E0 pin | **CORRECT** | `run.sbatch:18–19` unsets `JCPREF_META` and sets the required store. Staged `e0/store/{MANIFEST.sha256,STATUS.json}` match HEAD and `score.py`’s hard-coded pins. `check_store()` verifies the metadata directory actually used by the loader. |
| One job per directory; queue checks | **CORRECT** | `submit.sh` checks the queue before and after submission, refuses an existing attempt directory, verifies the manifest remotely, and serializes this lane’s submissions with `flock`. |
| Account/lane caps | **NEEDS-RESTATEMENT** | The wrapper refuses submission at eight account jobs or two `hrb_` jobs. Its lock is lane-local: simultaneous submissions from another lane can race the account-wide check. It also does not enforce the historical **one-smoke-total** limit. No GPU ledger exists here to establish previous expenditure. |
| Five-minute checksum reserve | **WRONG** | `smoke.py:55–79` checks remaining time **only before a phase starts**. It never interrupts that phase. `T_START` also begins after imports, excluding earlier job startup. The wrapper checksums only after the driver returns, with no timeout or termination trap. Slurm can kill the job before checksumming. |
| Full workload fits 55 minutes | **NEEDS-RESTATEMENT** | No real-size timing certificate establishes this. Treat completion as uncertain and high-risk, not a credible guaranteed fit. Details below. |
| Missing phases fail acceptance | **CORRECT** | `smoke_accept.py:23–25,207–210` requires all 18 phases in both cells. Missing, skipped, or errored phases fail; unavailable required controls remain unfulfilled. A scheduler kill may leave incomplete evidence or prevent the checker from completing, never establish success. |
| Outputs answer Phase I | **CORRECT, with amended scope** | The checker reconstructs control predicates from saved fields, trajectories, certificates, and timing repetitions. All 13 staged CPU gates pass and their source hashes match. Small-fixture controls remain CPU tests; A7.3 explicitly defers G-ours export replay to Phase II. “Every control at real size” must retain those exceptions. |

The timing risks are substantial. The driver executes **all 2D work before 3D**, including:

- 192 and 342 training trajectories, respectively, plus additional training/probe solves.
- State and advection PODs, potentially with randomized fallback.
- Maximal Q-DEIM pivoting and dense QR: the constructed matrices have shapes **1,046,529 × 3,072** and **250,047 × 6,144**.
- Smoke-rank NNLS paths, maximal-support pilots, full-grid solver runs, anchors, derivative checks, under-sampling ladders, and repeated timing calls.

These are long, uninterrupted phases; the `need=240`, `300`, or `600` values are admission thresholds, **not runtime bounds**. Later 2D controls and the entire 3D cell are especially exposed to skipping. An earlier phase can instead run into Slurm’s hard deadline.

Acceptance correctly remains closed on incomplete evidence. However, neither `ALL-DONE` nor a successful checker process exit establishes acceptance: the wrapper does not run the checker, and the checker writes `ALL.ok` without returning failure merely because that value is false.

Before submission, enforce the driver deadline while retaining time to checksum completed evidence, explicitly confine caches, and restate provenance and timing claims accurately. Preserve the frozen inventory; do not count skipped controls as a completed Phase I.