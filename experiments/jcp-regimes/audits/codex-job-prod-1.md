**Overall: FIX-FIRST — one shared cache/write-containment gap affects all ten jobs.** The staged scientific configurations and execution paths otherwise pass this audit at `717b354b7`.

| Job | Planned limit | Verdict |
|---|---:|---|
| refN0 | 02:00 | FIX-FIRST |
| refN0s | 01:15 | FIX-FIRST |
| refAp | 08:45 | FIX-FIRST |
| refAm | 07:00 | FIX-FIRST |
| refC1 | 07:00 | FIX-FIRST |
| refC2 | 07:00 | FIX-FIRST |
| refB1 | 02:00 | FIX-FIRST |
| refB2 | 02:00 | FIX-FIRST |
| rom1024 | 01:30 | FIX-FIRST |
| ps | 02:30 | FIX-FIRST |

**Required fix**

1. **Contain runtime caches before the first Python invocation.** Every script sets `XDG_CACHE_HOME` and `TMPDIR`, but omits `CUDA_CACHE_PATH` and Python bytecode controls. CUDA’s documented Linux default is `~/.nv/ComputeCache`; XDG redirection alone does not establish containment. Python can also create missing bytecode beside packages in the shared venv. This is an unclosed write path, not an observed production write. [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)

   Update the staging template to set `CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"` and `PYTHONDONTWRITEBYTECODE=1`, create the CUDA cache directory, and explicitly contain or disable any inherited persistent JAX cache configuration. Commit and regenerate all ten stages and manifests.

**Checks passed**

- **Integrity:** 153 source files per job, **1,530 comparisons**, match both MANIFEST hashes and `git show HEAD:<path>`. No unlisted files or symlinks. Provenance, commit records, reservations and certificate/config hashes agree.
- **Batch scripts:** H200, `gpu`, exact planned limits, 200G host memory, correct venv, x64/highest precision, GPU preflight, unit gates before workload, planned steps in order, and fail-fast shell behavior. Output/log/temp paths are correct. All scripts pass `bash -n`.
- **Submission:** local flock serializes lane submissions; lane cap is three. The campaign flock encloses the account-cap-eight check and `sbatch`. Queue-check failures refuse submission.
- **References:** ascending full populations; N0s has eight cases, others sixteen. A+ allowance is one; all others zero. No production probes, consistent with calibration agreement. Slim sources match their preceding reference outputs.
- **ROM:** dev6 + val32 + all eight regimes, nine arms, `tighter=sub`, `q2=all`. Frozen identities match the actual FRONT-2D file, including SHA `a51ac1a2…b8b77d`. Fverify inventory matches all nine arms.
- **PS:** independently regenerated largest-$s$ indices match **N0 2, A+ 7, A− 5, C1 11, C2 9, B1 0, B2 15**. `mode_ps` constructs regime cohorts and calls their IC-coefficient interface, including two-bump sums. CPU unit gates passed; the independent two-bump coefficient check differed by approximately `3.20e-15`.
- **Analysis contract:** staged ROM config, ROM/fverify outputs, reference metadata and slim arrays provide the required inputs. PS output paths match the analysis glob exactly. Collection must preserve those paths and assemble `store/2d/<regime>/ST/` from the slim outputs; the batch jobs do not assemble that local store.
- **Cohort boundaries:** no production path opens sealed/OOD cohorts. Historical `test64` descriptors are generated for duplicate checks; their reference solutions are not opened.

**Resource limitations**

At refinement $L=12288$, one f64 state occupies **1.125 GiB**, and six retained outputs occupy **6.749 GiB**, before solver vectors, FFT workspaces and temporary arrays. The **1,202-stage** schedule fits the 1,300-entry evidence buffer. I found no deterministic memory failure, but did not measure peak GPU memory.

The low-viscosity plan uses **850 seconds/case**, versus the unrefined smoke’s **350.37 seconds**, then adds retry/compile margin and time-limit headroom. That is a planning allowance, not a measured refinement guarantee. Repeated solver retries and PS stability reruns remain timeout risks; neither warrants a separate blocker on the evidence inspected.

No files changed, jobs submitted, or lab-log entries written.