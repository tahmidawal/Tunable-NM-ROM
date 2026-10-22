**WRONG — fix before submission.**

- **CORRECT:** `gpu` partition, one A100 with `a100-80G` constraint; no preempt.
- **CORRECT:** Required cluster venv, `JAX_DEFAULT_MATMUL_PRECISION=highest`, and x64 enabled.
- **CORRECT:** GPU preflight prints `jax_backend=` and exits 42 unless GPU.
- **CORRECT:** Wall `00:24:00` equals 0.4 GPU-h; nominal work cutoff is 1,296 s.
- **CORRECT:** All 27 manifest entries match the staged files.
- **CORRECT:** Explicit outputs, temporary files, and XDG cache target the `imp-fronts/smB` namespace.
- **NEEDS-RESTATEMENT:** “Never home” is not fully enforced: CUDA’s cache path is not redirected or disabled.
- **CORRECT:** C-B4 checks both complete parameter tables, including incumbent must-fail and unchanged non-viscosity draws.
- **WRONG:** Full generation is absent. `smoke_datagen.py:105–113` rolls out only 576 seed-0 + 576 seed-1000 trajectories per generator; 3,456 required arm trajectories receive no residual validation.
- **CORRECT:** Paired timing uses matching population indices on the same GPU, following untimed generator burn-in; timed trajectories receive finiteness and Newton-residual checks.
- **NEEDS-RESTATEMENT:** Only aggregate pass timings and residual summaries persist; no timing repetition arrays or outlier counts.
- **CORRECT:** C-B1 corrupts a provenance hash and requires exit 1 plus the specific mismatch message.
- **CORRECT:** Remaining-time `TRAIN2D_DRYRUN` chains Stage A → B → rotation; certification requires successful completion.
- **CORRECT:** Generation/chaining/timing failures prevent `ALL-DONE` and propagate failure on a fresh job directory.
- **NEEDS-RESTATEMENT:** The datagen timer checks between chunks, starts after imports, and has no outer timeout; it can overrun the nominal 90% cutoff.
- **CORRECT:** `timing.json` supplies the schema and population counts consumed by `resources.train_item()`, which correctly scales seed 0 ×2 and seed 1000 ×7.
- **WRONG:** `phase_times.py`’s own increment/projection scales seed 0 ×1, contradicting Amendment 16 and the downstream certificate; it also uses rounded 4.17 h instead of 15,018 s.
- **NEEDS-RESTATEMENT:** The staged reduced workload plausibly fits: K1 rates imply roughly 509 s for four timed chunks, before burn-in, residual checks, compilation and chaining. This is an estimate, not completion evidence.
- **WRONG:** Completing the requested full arm population plus incumbent timing implies roughly 1,285 s at K1 rates alone—almost the entire 1,296 s work budget before overhead. A compliant 24-minute completion is not credibly supported.

Read-only audit completed; no files, jobs, SSH, scratch, or lab-log changes.