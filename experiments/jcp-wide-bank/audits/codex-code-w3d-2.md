Read-only review completed; no files changed or GPU jobs run.

1. **CLOSED** — Refined-error normalization and lattice indexing remain correct.
2. **CLOSED** — The coefficient-space distance still reproduces the full-mesh field metric.
3. **CLOSED** — Shell completion and test-table prefixes remain aligned.
4. **CLOSED** — Tensor construction is capped at 512 columns with sufficient test rows, and wider settings cannot deploy it.
5. **CLOSED** — G2 now checks all `Ms` rows, and G4 retains its separate solver comparison.
6. **NOT-CLOSED** — Certification records and availability semantics are fixed, but check-rule certification trajectories are still discarded.
7. **CLOSED** — Continuum targets now use 64-state batches, and p95 is recorded.
8. **CLOSED** — Both projection and reached-state Jacobian singular-value arrays are retained.
9. **NOT-CLOSED** — Coefficient-based field retention is accepted, but field determinism remains inadequately checked and job-wide timing withdrawal is missing.
10. **ACCEPTED-AS-DOCUMENTED** — A5 explicitly defers floor exclusion to reporting, and `check_passed` implements the required flag, although the future report must enforce it.
11. **CLOSED** — Cross-mesh distances remain correct for the registered two meshes.
12. **ACCEPTED-AS-DOCUMENTED** — The capped tensor, chunked targets, synchronized sampler and boundary peaks improve accounting, while actual J4 feasibility remains conditional on its mandated smoke.
13. **ACCEPTED-AS-DOCUMENTED** — Config generation now fails closed on missing or mismatched extra rules, but the current smoke config still has a null hash and cannot run until artifacts and configs are regenerated and committed.

Blocking findings from the edits:

- **Field checks can falsely pass:** [w3d.py:518](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:518) compares only sums and squared sums; a field permutation preserves both despite large pointwise error, and the final panel checks only squared sums for ROMs and nothing for FOMs.
- **K-time enforcement is incomplete:** [w3d.py:572](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:572) correctly implements A5’s diagnostic smaller-\(m\) fallback, but never emits `timing_valid_jobwide` or propagates final-panel failures into timing withdrawal.
- **Certification evidence remains incomplete:** persist check certification trajectories alongside converged trajectories.

Other requested checks: the extra `fchk` runs are untimed and add overhead; `MemSampler.now()` introduces no recursive locking; `S_rep_start_mem` is defined correctly, although sampled maxima exclude preceding lean-table preparation; assigning `d = s_ = o = None` is safe; the separate tensor table duplicates mesh-bank storage, which must count toward J4’s smoke memory budget.

Before submission, also finish rule generation, regenerate the pinned configs, and commit the staged inputs.

OK to run the 3D smoke: **NO**.