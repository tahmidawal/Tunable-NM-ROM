**WRONG — fix the extraction timing before submission.**

- **CORRECT:** `run.sbatch` requests one A100-80G, `gpu` partition, 27 minutes, and the specified `imp-fronts/smC` namespace; no `preempt`.
- **CORRECT:** Absolute cluster venv, x64, `highest`, and GPU preflight with exit 42 are configured.
- **CORRECT:** Manifest checks pass with no missing/unlisted files; staged provenance hashes pass; shell syntax passes.
- **CORRECT:** GPU identity tests include oddness, periodicity, walls, derivatives, NumPy agreement, and polynomial/K1 must-fail controls. Tested bank copies match the staged training copies.
- **CORRECT:** Stage A measures paired incumbent/arm sub/full rates, persists interval repetitions, and excludes compilation intervals from rates.
- **WRONG:** `smoke_bnd_timing.py` extraction projects 3,264 states together and rebuilds features/Gram/Cholesky every repetition. Production uses `PROJ_CHUNK=256` and builds those bank quantities once. This changes the architecture-dependent cost weighting; its ratio is not a validated production extraction multiplier.
- **NEEDS-RESTATEMENT:** Burn-in precedes Stage A, but extraction gets only one warm-up per bank. Demonstrate clock stability or add sustained burn-in immediately before extraction timing.
- **CORRECT:** `TRAIN2D_DRYRUN=1` chains the staged bnd Stage A → extraction/head → rotation, requiring completion.
- **CORRECT:** Projection arithmetic implements Amendments 21–22, including conservative remainder/extraction factors and the 16,440-second cutoff; `resources.train_c_item` accepts its schema.
- **CORRECT:** Identity, timing, chain, and projection failures prevent `ALL-DONE`; an over-budget projection remains uncertified.
- **WRONG:** Remaining-time values reach `timeout` without a positive guard. `LEFT=0` disables its timeout; guard every timed invocation to enforce the 90% self-timer.
- **NEEDS-RESTATEMENT:** Completion is plausible, not established: K1 rates imply about 292 seconds for training loops at equal arm cost, leaving roughly 1,166 seconds for identities, compilation, extraction, chaining, and setup.
- **CORRECT:** Audit remained read-only: no jobs, SSH, scratch, file changes, or lab-log writes.