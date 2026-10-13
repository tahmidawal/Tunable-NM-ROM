**The four previous findings are RESOLVED, but the smokes still have two MAJOR evidence-collection gaps.**

No files were modified or project modules imported. Verification used source inspection, read-only hashing, syntax checks and isolated predicates; I did not rerun the manufactured suite or GPU workloads.

| Previous finding | Status | Evidence |
|---|---|---|
| Missing `timing.py` | **RESOLVED** | Included in all staged trees; `stage.py` now performs an isolated staged-import check and rejects imports from other experiment trees. |
| G-FS1 accepting non-finite cases | **RESOLVED** | Both acceptance branches require finite trajectories and distances. The original `[0, NaN]` counterexample now fails. Non-finite CN-control distance is flagged as non-discriminating, consistent with A5.7. |
| NumPy oracle/non-finite Jacobian | **RESOLVED** | Both solve phases catch `ValueError`. The regenerated certificate includes the fixture: vendor and NumPy both report **21 iterations, reason 3**; `all_pass: true`. All seven certificate hashes match checkout and HEAD. |
| Submission-cap query failing open | **RESOLVED** | Failed `squeue` aborts before counting/submission. A failing-query shell probe exits through the refusal branch. |

All staged manifests verify: **46 entries for smk2d; 33 each for smk3esc and smk3grid**. Provenance matches commit `79431cb54cceb0e15342ce7b30e8d5673c4dbf46`; staged source matches the checkout. Config differences from committed originals are exactly the reference-path rewrites.

| Sbatch check | smk2d | smk3esc | smk3grid |
|---|---|---|---|
| Partition/GPU | `gpu`, A100 with `a100-80G` constraint | `gpu`, H200 | `gpu`, H200 |
| Host memory/time | 200G / 6 hours | 200G / 6 hours | 200G / 6 hours |
| Cluster venv activated; absolute interpreter | Pass | Pass | Pass |
| Highest matmul precision; x64 | Pass | Pass | Pass |
| GPU preflight exits 42 for non-GPU backend | Pass | Pass | Pass |
| Logs/output/cache/tmp under attempt directory | Pass | Pass | Pass |
| Input manifest verification; output SHA manifest | Pass | Pass | Pass |
| Reference paths rewritten correctly | Pass | Pass | Pass |

These checks confirm requested resources, **not demonstrated memory fit or six-hour completion**.

The configurations cover the requested work:

- **smk2d:** all six settings; both anchors and tolerance levels; G1a; all four FOM schemes through order/calibration/evaluation; per-setting and common timing panels.
- **smk3esc:** W1024 at \(R'=1024\); Gauss \(48^3,56^3,64^3\), conditional \(80^3\), all four escalation arms, operator check, and a reduced ladder containing `lat256`. Ladder/panel execution is conditional—finding 1.
- **smk3grid:** cases `923801/0` and `/1`; G-FS1 and CN control; every configured fixed-sweep/adaptive/GAL/anchor family; setting panels and W512 common baseline.

All three preserve `config.smoke: true` in results. The staged drivers do not produce H1–H5 scientific verdicts. However, `analyze2d.py` currently lacks a smoke rejection guard; that must be included in the deferred analysis audit before consuming these outputs.

1. **MAJOR — escalation can report completion without exercising the ladder or family panels.**  
   In [c3run.py:307](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:307), all ladder rollouts—including `lat256`—and family timing are inside `if conv is not None`. If neither rule pair passes selection eligibility/convergence, the job nevertheless reaches `complete=True` and writes `COMPLETE`. Thus a nominally successful smoke can omit precisely the phases its resource evidence needs.

   **Fix:** for smoke only, exercise these phases with an explicitly diagnostic fallback baseline, preserving `converged=None` and withholding selection claims. Alternatively, mark phase coverage incomplete and require a supplemental smoke before resource certification.

2. **MAJOR — smk2d does not capture the whole-job resource evidence.**  
   [c2grid.py:156](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2grid.py:156) samples device peaks only during ROM accuracy, **before FOM order, calibration, evaluation and both timing panels**. There is no final device-peak sample. FOM order/calibration/evaluation also lack per-invocation timing records; later warmed timing measurements cannot recover their compilation and tight-solver costs. Reworking `inventory.py` afterward cannot recover unrecorded measurements.

   **Fix:** record a final whole-job device peak and structured timings for FOM phases/invocations, including scheme, timestep, tolerances and case. Preserve first-call and subsequent-call costs separately. Missing measurements must prevent certification.

An additional unresolved preflight risk remains: the staged 2D references pass field checksums, but regenerating `dev6` locally produces a different cohort hash, which would trigger `c2common.load_refs`’ assertion. Target-cluster compatibility is **unverified**, not established broken: SSH closed the connection during my read-only check. Verify the descriptor hash with the target venv before submission; do not waive the identity check. Both staged 3D references pass checksum, acceptance, metadata, selected-field shape and finiteness checks.

**DO NOT SUBMIT.**