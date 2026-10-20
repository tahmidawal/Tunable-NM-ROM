**Overall: SUBMIT at `5c23c82a2`.** Round 1’s shared cache-containment blocker is resolved in all ten staged jobs.

| Job | Verified time limit | Verdict |
|---|---:|---|
| refN0 | 02:00:00 | SUBMIT |
| refN0s | 01:15:00 | SUBMIT |
| refAp | 08:45:00 | SUBMIT |
| refAm | 07:00:00 | SUBMIT |
| refC1 | 07:00:00 | SUBMIT |
| refC2 | 07:00:00 | SUBMIT |
| refB1 | 02:00:00 | SUBMIT |
| refB2 | 02:00:00 | SUBMIT |
| rom1024 | 01:30:00 | SUBMIT |
| ps | 02:30:00 | SUBMIT |

Verified independently:

- **Containment:** every script sets `CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"`, `PYTHONDONTWRITEBYTECODE=1`, and `JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/cache/jax"` before its first Python invocation, and creates the cache directories.
- **Integrity:** all **1,530 source-file comparisons** match HEAD blobs and working sources. All **1,570 manifest entries** validate; no unlisted files or symlinks. Provenance and commit records match HEAD.
- **Execution:** every complete batch script matches the committed template and differs from the round-1 template only by the intended cache fix. All pass `bash -n`.
- **Plan:** time limits, ordered workload steps, reservations, certificates and configuration hashes agree. Every limit satisfies the required projection margin.
- **Nothing else changed:** relative to round 1’s `717b354b7`, the only executable-source change is the cache block in `cluster/stage.py`; other committed changes are audit records. Scientific code, configurations, precision settings and solver controls are unchanged.

The fix changes cache storage and compilation reuse, with no numerical-method change. CUDA’s variable redirects its JIT cache; Python’s variable suppresses bytecode writes. JAX’s directory setting enables a per-job persistent **compiled-program** cache, keyed by computation, jaxlib version, relevant compilation flags and device configuration. It does not cache solution values or change tolerances or arithmetic precision. Compilation/startup timing can change; this static audit does not establish bitwise GPU reproducibility. [CUDA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html), [Python documentation](https://docs.python.org/3.10/using/cmdline.html#envvar-PYTHONDONTWRITEBYTECODE), [JAX documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

Round 1’s unmeasured peak-memory and refinement-time risks remain unchanged; no new blocker found.

No files changed, jobs submitted, or lab-log entries written.