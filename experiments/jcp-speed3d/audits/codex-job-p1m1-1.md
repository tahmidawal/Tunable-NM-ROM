**Verdicts: p1 — SUBMIT; m1 — SUBMIT.** I found no deterministic execution blocker or missing selected pair. Several resource, smoke-coverage, and recovery claims need qualification; these are not guarantees of successful completion.

Read-only audit completed. No repository/main/LAB-LOG writes, submissions, or GPU execution. Python checks used the specified interpreter with `JAX_PLATFORMS=cpu` and bytecode writes disabled.

| Audit item | p1 | m1 |
|---|---|---|
| 1. Required configurations and execution paths | **CORRECT**, qualifications below | **CORRECT**, qualifications below |
| 2. Staging, provenance, batch requirements | **CORRECT** | **CORRECT** |
| 3. Resource estimates and timeout recovery | **NEEDS-RESTATEMENT** | **NEEDS-RESTATEMENT** |
| 4. Memory and smoke coverage | **NEEDS-RESTATEMENT** | **NEEDS-RESTATEMENT** |
| 5. Cohort integrity | **CORRECT** | **CORRECT** |

**1. Configuration and coverage**

For **p1**, I mechanically compared the configuration against `SELECTION-SCREEN.json`:

- All **52 label-to-configuration mappings match exactly**.
- All **31 confirmation pairs match exactly**: nine H2/decomposition pairs plus 22 H1.4 pairs. They execute on cases **0–7 × three repetitions**.
- Profile and mesh-free populations contain `C_old`, `C_W1024`, and both distinct H2 winners.
- All five distinct winner/B* configurations receive their specified converged-rule sensitivity runs and reduced-rule ADT tight/tighter checks.
- K5 uses `gl12`, with rank/condition checks. G4, K6 and K6+ are executed for the configured batches.

The execution order is explicit in [s3modes.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:201): mesh accuracy, mesh-free/ADT/K5, profiling, batching, confirmations, converged-rule sensitivity, reduced-rule ADT.

For **m1**:

- Four selected ROMs run at **n=65,129,257,385**, each on all 64 validation cases, with ADT tight on all 64 and tighter on cases 0–7.
- G5/K7 cover M2_512, W256 and W1024.
- Same-mesh UB/CC use Δt=0.01; calibration is **cases 0–3 at every mesh**, correctly incorporating A9.
- Same-mesh FOM accuracy uses eight cases; matched-FOM reruns use four timing cases. These are not fresh 64-case FOM qualification panels; matching comes from f1.
- All **eight ROM/FOM associations match `fom-match.json` exactly**.
- Four A6 `match|<label>` panels use the **n=65 mesh-path ROM**, four cases and two repetitions. Cross-mesh diagnostic panels use the mesh-free ROM.

See [route-4 execution](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:385).

Config-consumption qualifications:

- `dts` does **not** drive these modes; each `spec` supplies its step.
- `settings.reduced`/`extra` load available rules; they do **not** independently schedule runs.
- `driver` and `attempt` are staging/metadata fields, not runtime mode selectors.
- m1’s `selection_sha256` and `fom_match_sha256` are **not runtime assertions**. Both pins currently match, and the staged config itself is manifest-protected.
- Batch repetitions are hard-coded to **three**, independently of `timing.reps`.
- **T1 is not rerun in p1/m1**; its evidence is inherited from smoke/grid panels. The design’s “checked … at one real size before any verdict” permits that, but these jobs must not be described as containing T1.
- G3 and H3.2/K5 qualification verdicts remain downstream analysis obligations under A4.5. These jobs collect their evidence.

Failure paths are recorded, but completion is not acceptance: condition failures can skip mesh-free work; failed G5 suppresses larger-mesh ROM work; exceptions can record infeasible configurations or failed panels; missing hashes skip timing. `complete=true` alone does not establish required coverage.

**2. Staging and provenance — CORRECT**

For **each job**, I verified:

- **34 source/config blobs**, byte-identical to commit `57c09fac5d751b82e11796a06bdcbd2287d6211a`.
- **38 manifest entries**, all hashes valid, with no uncovered staged files other than the manifest itself.
- Staged configurations equal the lane configurations; resource-certificate config hashes match.
- `gpu` partition, one H200, 24 hours; p1 **200G**, m1 **240G**.
- GPU preflight exits **42** on a non-GPU backend.
- x64 and `JAX_DEFAULT_MATMUL_PRECISION=highest`.
- Output, logs, temporary files and caches target their separate paralab attempt directories.

The submit wrapper uses locking and refuses an existing attempt directory, supporting one job per directory. **I did not check live remote directory occupancy, queue state, free disk, or E0-store availability.**

**3. Resources and recovery — NEEDS-RESTATEMENT**

I independently reproduced both certificates:

| Job | Projection | With ×1.5 margin | Allocation |
|---|---:|---:|---:|
| p1 | 7.0375 h | 10.5562 h | 24 h |
| m1 | 10.2025 h | 15.3037 h | 24 h |

**Both allocations plausibly cover their jobs.** Even the historical g3 underestimate, approximately 3.6/2.7, fits within the prescribed 1.5 factor.

However, A8’s **12.678-second L384 proxy for L256 is an estimate, not an upper bound**. Nonlinear iteration counts, tolerances, cases and compilation can defeat monotonic wall-time assumptions. The estimator pools FOM entries and uses coarse compile/setup allowances. Its docstring also still describes the superseded L⁴ interpolation. See [resources.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/resources.py:175).

The lane budget also needs monitoring:

- Spent plus central projections: **26.1458 GPU-h**.
- Spent plus both ×1.5 projections: **34.7657 GPU-h**, exceeding the **34-hour** cap.
- Neither job implements budget-triggered deferral or stopping.

**Timeout exposure differs:**

- **p1 confirmations follow profiling and batching**, including un-smoked B=64. A timeout or fatal OOM there can lose every confirmation. Moving confirmations earlier would reduce this risk.
- **m1 match panels execute during L64**, before L128/L256/L384, so they are not stranded at the job’s end.

Partial persistence exists, but is coarse. JSON replacement is atomic; timing panels save after completion. `rollouts()` does not checkpoint each case or configuration itself. The initial p1 accuracy sweep has no intermediate save. NPZ files are rewritten non-atomically, with no resume mechanism; interruption during a rewrite can damage the coefficient archive. See [s3ctx.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3ctx.py:90).

**4. Memory and smoke — NEEDS-RESTATEMENT**

The L384 pilot supports feasibility: it completed in **353.3 seconds**, reported **24.59 GB peak device memory**, and passed all **15 acceptance gates**. W1024 n=385 table construction took **42.08 seconds**. This is device-memory evidence, **not host MaxRSS evidence proving 240G sufficient**.

The streaming builder avoids a full mesh-by-bank array and passes large arrays explicitly into JIT functions. Its largest slab still scales with `(n−2)² × R′`, and mode construction uses mesh-sized host arrays. See [bigtables.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/bigtables.py:34).

Two coverage corrections matter:

- **p1 contains no W-bank B=64 batch.** Its three batch labels all resolve to **M2_512**. Smoke covered B=1 and B=3, including a W1024 QN example; it did not establish B=64 feasibility.
- The **A6 mesh-path match block was added after the accepted smoke commit**. Its underlying rollout/timing machinery was exercised, but that exact block was not.

The relevant profile/mesh smoke logs contain no captured-constants warning. The identified warning occurs in replay, which p1/m1 never invoke; inspected production query/profile/table paths pass large tables as arguments. Full production memory remains unproven, and Python exception handlers cannot recover from a host OOM kill.

**5. Integrity — CORRECT**

Both jobs regenerate only **923801, cases 0–63**. E0 lookup selects `val64`; the staged continuity reference is `ref_923801.npz`. Runtime guards restrict 923901 to replication mode with a pinned freeze. The reference loader rejects sealed/OOD cohorts. I found no p1/m1 path opening test, sealed or OOD fields.

**Final submission decisions: p1 — SUBMIT; m1 — SUBMIT.** Preserve the qualifications above, especially mandatory-panel coverage checks, coarse recovery, untested large batches, and aggregate budget monitoring.