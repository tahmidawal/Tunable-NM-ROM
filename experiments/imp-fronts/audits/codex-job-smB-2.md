**NEEDS-RESTATEMENT — prior SM-b blockers are resolved; scope wording still needs clarification.**

- **CORRECT:** `gpu` partition, one A100 with `a100-80G` constraint; no preempt.
- **CORRECT:** Required cluster venv, highest matmul precision, and x64 enabled.
- **CORRECT:** GPU preflight prints `jax_backend=` and exits 42 unless GPU.
- **CORRECT:** One GPU × 24-minute wall equals the 0.4 GPU-h ceiling.
- **CORRECT:** Explicit outputs, temporary files, XDG and CUDA caches stay under `imp-fronts/smB/`.
- **CORRECT:** All 27 manifest entries match; no unmanifested files; internal provenance hashes match. Header/body/timing scripts match `COMMIT.txt`.
- **CORRECT:** Amendment 17 permits exactly the staged timing subset: 576 seed-0 + first 576 seed-1000 trajectories per generator.
- **CORRECT:** C-B4 checks both complete parameter tables, unchanged non-viscosity draws, and incumbent must-fail.
- **CORRECT:** C-B1 requires exit 1 and the specific provenance-mismatch message.
- **CORRECT:** Both generators receive untimed warm-up; paired timings include residual validation and host transfer; chunk timing arrays persist.
- **CORRECT:** The cov generator now enforces finite states and Newton residual ≤ 1e-8 on every generated chunk, including production.
- **CORRECT:** Remaining-time dry run chains Stage A → B → rotation; certification requires completion.
- **CORRECT:** Generation, chaining, or timing failure prevents `ALL-DONE` on a fresh job directory; shell syntax checks pass.
- **CORRECT:** `timing.json` matches `resources.train_item()`; both projections use seed-0 ×2, seed-1000 ×7, and K1 = 15,018 s.
- **NEEDS-RESTATEMENT:** “No test-split access” needs qualification: chaining generates two seed-1 training-recipe diagnostic test trajectories and evaluates their truth. Amendment 11 discloses this diagnostic; no sealed/OOD or fresh FC test-cohort access was found.
- **NEEDS-RESTATEMENT:** The 1,296-second work timer starts after setup; outer timeouts allow 30-second kill grace. It is a nominal cutoff, with the 1,440-second Slurm wall enforcing spending.
- **NEEDS-RESTATEMENT:** Completion is plausible, not established: prior K1 rates imply approximately 509 s for paired generation, before low-viscosity overhead, residual checks, compilation, warm-up and chaining.
- **NEEDS-RESTATEMENT:** `run.sbatch` still describes full 576 + 4,032 generation; update that description to Amendment 17’s subset.
- **NEEDS-RESTATEMENT:** SM-r’s H200/dev–val/reference restrictions are outside this staged SM-b audit; this verdict does not certify SM-r or production admission.

Read-only audit completed; no files, jobs, SSH, scratch, or lab-log changes.