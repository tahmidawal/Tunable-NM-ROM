**DO-NOT-SUBMIT as staged.** Resource safeguards pass, but the entry point is missing and the probe does not reliably distinguish the proposed causes.

| Item | Verdict | Evidence |
|---|---|---|
| GPU partition / A100 | **CORRECT** | `--partition=gpu`, `--gres=gpu:a100:1`. |
| Python environment | **CORRECT** | Uses the prescribed absolute cluster venv path. |
| Precision | **CORRECT** | Batch exports `JAX_DEFAULT_MATMUL_PRECISION=highest`; imported `spec.py` enables x64 before descriptor creation. |
| GPU preflight | **CORRECT** | Prints `jax_backend=gpu` or exits 42 before the diagnostic. |
| Outputs under paralab | **CORRECT** | Slurm log and configured cache reside under `imp-box-diag`; 3.3 TB available when checked. |
| ≤0.5 GPU-h | **CORRECT** | One GPU with a 30-minute allocation limit. Completion within that limit is unverified. |
| Staged executable | **WRONG** | Remote `code/diag_jobs/` does not exist. Line 15 of `n12.sbatch` therefore cannot execute. Remote `spec.py` and `geom.py` match local SHA256 hashes, but the staged tree is incomplete. |
| Persistent-cache isolation | **WRONG** | “Off” never explicitly disables caching. All three probes share one process, without demonstrated persistent-cache reads. “On2” is not evidence of a disk-cache hit. |
| Reproduction of batching failure | **WRONG** | Tests one descriptor with batch size one. Production `run_cohort` uses `jax.jit(jax.vmap(q))`; the probe does not reproduce its batch shape or wrapper. |
| XLA/cuFFT versus code bug | **WRONG** | Padding and certificate probes provide clues, not separation: certificate replay shares the nonlinear FFT kernel; `M=2N` changes both padding and the implementation path. No independent nonlinear reference or first-failing-stage trace is provided. A passing A100 run cannot rule out an H200-specific failure. |

Before submission:

- Stage and hash-check the diagnostic entry point and dependencies.
- Use separate processes for explicitly disabled cache, fresh-cache population, and verified cache-hit replay.
- Reproduce a failing configuration with the production wrapper and batch size; include a finer fixed-step control.
- Record the first non-finite step/stage and compare its nonlinear evaluation against an independent reference. Retain padding/certificate comparisons as supporting diagnostics.

No files changed, jobs submitted, or LAB-LOG entries written.