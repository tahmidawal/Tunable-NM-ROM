**Verdict: WRONG — E7 is not ready for production.** The core resampling and blocked ECSW arithmetic passed small CPU checks, but qualification, timing acceptance, controls, cost accounting, and job handling have blocking defects.

I made no file changes, submitted no jobs, and did not write to the lab log or main. All Python checks used the specified interpreter with `JAX_PLATFORMS=cpu` and bytecode writing disabled. No large arrays were copied.

Validation performed:

- Tiny resampling, code transformation, node-mapping, and mutation tests passed.
- Blocked versus dense ECSW: relative differences **6.14e-15 for G**, **9.33e-16 for b**; dropping the advection Jacobian changed G by **2.87e-2**.
- The compile listener detected a backend compilation on CPU.
- Imported dependencies of `mmcore` and `p1acc` were all present in the staging file list.
- Both historical probe manifests resolved by hash.
- Synthetic control-8 fixtures passed, but an additional missing-status test exposed a false **FIRES** result.
- Shell syntax checks passed. This does not validate runtime behavior or GPU memory.

**1. `mmcore.py` — WRONG overall; core arithmetic CORRECT**

[Lines 42–90, 139–150, 178–187, 226–298](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmcore.py:42): DST normalization, prolongation scaling, the rank-specific code transform, node mapping, and blocked residual/Jacobian arithmetic are consistent with the amended design. The combined sign-fixed QR factor at line 127 is correct. No large basis is captured by the blocked evaluator’s JIT closures.

Defects:

- **Full-size basis prefixes still allocate.** [Lines 312–318](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmcore.py:312) return `self.Phi[:, :r]`. A small CPU check confirmed a distinct buffer. At 4096, a resident rank-384 basis is approximately 51.5 GB; rank-128/256 prefixes add approximately 17.2/34.3 GB. Avoiding `stencil_rows` concatenation does not eliminate these copies. **Fix:** pass the full basis explicitly and implement rank-prefix initialization/decoding inside fused kernels, or use independently stored column blocks. Revalidate peak memory with the sentinel and inputs resident.
- **Obsolete rank criterion remains binding.** Lines 128–134 include the full-factor diagonal ratio in `orth_ok`, before `mmbuild` performs the registered per-prefix singular-value condition check. A bad high-rank tail can abort otherwise admissible lower ranks. **Fix:** separate orthonormality from rank acceptance; use A2.4’s condition number for each deployed prefix.
- **Missing val32 tighter anchors.** [Lines 355–365](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmcore.py:355) take the first two cases of the combined list, both dev6. Section 3.4 requires each cohort’s first two. **Fix:** collect dev6 and val32 anchors and apply the cohort-specific `q1_config` contract, preserving the required pooled stationarity calculation.

The `Cell.run`, `ecsw_weights`, `Refs`, and `q1_config` argument conventions otherwise match the imported APIs. The compile monitor at lines 401–418 works in the CPU check.

**2. `mmbuild.py` — WRONG**

[Lines 117, 212, 235–278](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmbuild.py:117):

- Basis-certificate failures raise assertions instead of persisting the registered `NOT-CONSTRUCTIBLE` states. Source construction also aborts when any requested rank fails its certificate, discarding usable ranks.
- Successfully measured configurations receive `status='ok'`, rather than `MEASURED-Q1` or `MEASURED-FAIL`.
- **Fix:** initialize the complete registered configuration universe as `UNRESOLVED`; persist certificates and transition each affected configuration explicitly. Continue with certified ranks.

[Lines 261–285](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmbuild.py:261): no κ=4 versus κ=8 rollout-agreement calculation is produced. Consequently the registered rollout sensitivity has no measured input. **Fix:** compare paired trajectories on every case using the registered lattice norm and persist the result.

[Lines 296–303](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmbuild.py:296): the PM trajectory timing block has no GPU burn-in. **Fix:** burn before each timed trajectory and retain the explicit first-call compilation label.

Correct: source rule keys are already `k4`/`k8`; direct `ecsw_weights` returns integer keys, which `contrib_and_nnls` correctly converts. Missing NNLS checkpoints remain `UNRESOLVED`, consistent with A4.

**3. `mmsmoke.py` — WRONG**

[Lines 29–32, 159–163, 187–189](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmsmoke.py:29):

- Failed controls merely record `passed=false`; the program still declares completion and exits successfully.
- The registered unchanged-state fallback for control 4 is absent.
- **Fix:** implement the fallback, require all mandatory controls to pass, and emit a nonzero exit plus an explicit acceptance record otherwise.

[Lines 106–131](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmsmoke.py:106): memory peaks are recorded but never compared with A4.5’s bound. The separate pilot of our 4096 bank table required by A1.7 is absent. **Fix:** assert production-shaped dimensions, validate the NNLS-phase peak against the certificate, and measure our table pilot sequentially.

[Lines 141–150](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmsmoke.py:141): control 5 never checks identical kept-state masks. `i0.get('kept_states')` also reads a nonexistent dense-builder field; the dense builder reports `dropped_blocks` and `block_norms`. **Fix:** derive/persist both masks and compare them explicitly.

Controls 8 and 9 are not implemented by this driver or its configured follow-up commands. The docstring’s `cost.py --selftest` command does nothing because `cost.py` has no CLI.

**4. `mmtime.py` — WRONG**

[Lines 36–50, 100–116, 174–184](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmtime.py:36): timed outputs are discarded; identity checks use additional, untimed executions. A2 is never identity-checked. This violates A2.5’s requirement for **every timed execution**.

**Fix:** retain each timed result until timing stops, then check its outputs, states, iteration counts, and reasons against the accuracy record. Resolve cases by saved identifiers rather than positional assumption.

[Lines 102–103, 153, 224](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmtime.py:102): missing J5 evidence is permitted, basis identity is only recorded, and `valid` depends solely on timing drift. An identity failure can therefore produce a valid timing panel. **Fix:** require the evidence and include all identity predicates in acceptance.

[Lines 117–118, 200–225](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmtime.py:117): ours lacks the required outlier counts, and a retry overwrites the first attempt’s observations. **Fix:** retain both attempts and compute the same statistics for both methods.

The arrays selected for the replay checks—ours’ `W`, internal states and step statistics, competitors’ `C` and step statistics—are appropriate. The defect is which executions are checked and whether failure blocks acceptance.

**5. `oursq.py` — WRONG**

[Lines 47–61](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/oursq.py:47):

- `gcert_ok` is collected but omitted from Q1 acceptance.
- Successor/check-rule distance is collected but omitted from Q2 acceptance.
- `qpp_valid` is ignored.
- Missing/invalid finer-rule evidence is collapsed into Boolean qualification rather than preserving the inherited “unverified” state.
- Python `max` does not reliably reject a later NaN; JSON-cleaned nonfinite values can also be `None` and crash comparisons.

**Fix:** require certifier agreement; require all six dev6 finer-rule checks and valid check-rule runs; handle undefined ρ indicators explicitly; reject nonfinite distances before reductions; preserve pass/fail/unverified distinctions.

[Lines 24–43, 52–61](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/oursq.py:24): counts are checked, but exact unique case coverage, row/NPZ identity, mesh agreement, and full descriptor identity are not. **Fix:** validate these against the registered cohort and configuration before scoring.

The referenced `p1acc` row-field names are real. These are missing predicates, not misspelled API fields.

**6. `cost.py` — WRONG**

[Lines 127–132](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cost.py:127): missing statuses are treated as neither qualified nor unresolved.

**Reproduced:** deleting X’s mesh-256 status changes an otherwise unresolved case to **FIRES**; with every status absent, the function returns **FIRES** with 21 vacuous workloads.

**Fix:** construct the full registered universe and default every missing status to `UNRESOLVED`. Vacuity requires explicit measured nonqualification.

[Lines 93–105](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cost.py:93): missing configurations within a nonempty mesh dictionary are invisible. This permits an incomplete search to prove `NO-MATCH`. **Fix:** validate all registered configuration identifiers and populate omitted ones as unresolved.

[Lines 43–59](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cost.py:43): `var` changes ours’ offline cost only. Competitor cost has no accounting-variant argument, so A4.4’s A_sel construction and qualification costs are not implemented.

The documented `load_inputs` does not exist. Therefore there is no implemented:

- PM/CP source-mesh canonicalization or CPU-1024→CPR-1024 coarse-mesh alias;
- dependency-key construction from measured records;
- timing/identity acceptance gate;
- cache-suspected compile inclusion/exclusion accounting;
- smoke-derived fixture input path.

**Fix:** implement and validate the adapter and variant-dependent ledger before evaluating real results.

[Lines 167–282](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cost.py:167): control 8f2 silently changes its expected outcome when no second X exists; 8e disappears entirely when no witness exists. These do not enforce the registered fixture preconditions. `stop_rule_fixture` also uses a cumulative unresolved flag when deciding later workloads’ vacuity.

**Fix:** require the prescribed fixtures and report missing preconditions as control failures; test absent keys as well as explicit unresolved values.

Correct: joint Cartesian enumeration, union-of-dependencies charging, separate admissibility enumeration, witness precedence, and `NO-MATCH` precedence over unresolved evidence work for complete synthetic inputs.

**7. `cluster/stage.py` — WRONG**

[Lines 78–83, 118–120](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cluster/stage.py:78): failed commands set `FAIL=1`, but the script never exits with it. Slurm can report success after failed steps. **Fix:** preserve output checksums, then `exit "$FAIL"`; skip dependent steps after prerequisites fail.

Arbitrary command strings are appended directly after `timeout`. Shell operators in those strings execute outside the timeout wrapper; the CPU-preflight command consequently loses its original exit status after its `echo`. **Fix:** execute a command script under timeout and handle expected exit codes explicitly.

[Lines 49, 76–83](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cluster/stage.py:49): no validation connects allocation length, sequential command budgets, and the smoke-derived projection. The default allocation is shorter than the smoke command’s own timeout.

No staging omission was found in the exercised import graph. However, importing `mmtime` does not execute its function-local `p1common` import; add explicit driver/deferred-import checks.

**8. `cluster/stage_probe.py` — WRONG**

[Lines 99–130](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cluster/stage_probe.py:99):

- Manifest failure does not stop execution.
- Early bank-process failure and failure to reach the required log marker do not invalidate the job.
- Extraction/head failures are printed but do not fail the job.
- The bank polling loop alone allows 5,400 seconds against a default 4,500-second allocation, before extraction/head work.
- Full producing-process wall times are not recorded as explicit measurements.
- Outputs use `out/`, and no `OUTPUTS.sha256` is generated; the supplied collector expects `output/` and that manifest.

**Fix:** require manifest and marker success, capture/wait for subprocess statuses, enforce an allocation-compatible total budget, measure complete invocation wall times, and adopt the collector’s output/checksum contract.

The historical source resolution itself succeeded: 26 bank-code files and 33 head-code files, plus the checkpoint.

**9. `cluster/submit.sh` — WRONG**

[Lines 13–26](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cluster/submit.sh:13):

- No mandatory S1/control-acceptance gate exists.
- The projection is an unchecked command-line number, not a configuration-bound smoke certificate.
- Lane-cap and budget checks occur outside the submission lock; concurrent submitters can both pass.
- Reservation binding occurs after submission and is not transactional.

**Fix:** validate accepted controls and the hashed resource certificate, then perform lane-cap checking and reservation atomically with submission.

**10. `cluster/collect.sh` — WRONG**

[Lines 8–11](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cluster/collect.sh:8):

- Despite its documented behavior, it neither archives large files nor deletes the job directory. Target configs expect source artifacts in that absent archive.
- Missing files are silently skipped during checksum validation.
- If the manifest’s last file was omitted by the size filter, the `while` loop can exit nonzero under `pipefail` even when selected files verified.
- It does not establish that the job has finished.
- Probe output layout is incompatible.

**Fix:** explicitly enumerate and verify selected files, verify archival completeness and hashes, confirm terminal job state, then remove the remote attempt.

**11. `cluster/ledger.py` — WRONG**

[Lines 19–30, 49–50](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cluster/ledger.py:19): running elapsed time enters `spent_gpu_hours`, while the same job’s full limit remains reserved. Running work is double-counted after `settle`.

**Fix:** use terminal actual hours plus full active reservations, or actual hours plus remaining active reservations.

[Lines 28–35](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/cluster/ledger.py:28): negative/nonfinite projections, duplicate bindings, and concurrent writes are not guarded. Active states beyond pending/running can lose their reservation. **Fix:** validate inputs, recognize all nonterminal states, and lock atomic updates.

**12. Configurations and tests**

- `configs/smoke.json` — **WRONG**, lines 5–6 and 19–22: CPU preflight never asserts exit 42; replication merely scores outputs and never compares against historical R128/R256 errors. Add actual positive/negative replication checks and control-8 invocation.
- `configs/ours.json` — **CORRECT as a command inventory**, lines 4–31. Its execution remains blocked by checker and staging defects; validate the total resource budget and explicitly import-check `p1acc`.
- `configs/ours256.json`, `ours1024.json`, `ours4096.json` — **CORRECT**, lines 13–54: six registered settings, all38 Q1/Q2 populations, correct successor/check rules.
- `configs/repl1024.json` — **CORRECT**, lines 13–25: the registered R128-BE/dev6 fixture. The missing comparator is elsewhere.
- `configs/sources.json` — **CORRECT**, lines 8–27: source meshes, snapshot recipes, ranks, and κ values agree with the design.
- `configs/target256.json`, `target1024.json`, `target4096.json` — **CORRECT as physical-job inventories**, lines 21 onward. Required aliases must be supplied by the missing cost adapter; archive paths require the collector fix.
- `test_mmcore.py` — **CORRECT for its stated tiny-mesh scope**, lines 24–87; all tests passed. It does not establish large-mesh memory safety or driver integration. Add regressions for the qualification, missing-status, and job-failure defects above.

| File | Verdict |
|---|---|
| `mmcore.py` | WRONG |
| `mmbuild.py` | WRONG |
| `mmsmoke.py` | WRONG |
| `mmtime.py` | WRONG |
| `oursq.py` | WRONG |
| `cost.py` | WRONG |
| `cluster/stage.py` | WRONG |
| `cluster/stage_probe.py` | WRONG |
| `cluster/submit.sh` | WRONG |
| `cluster/collect.sh` | WRONG |
| `cluster/ledger.py` | WRONG |
| `configs/smoke.json` | WRONG |
| `configs/ours.json` | CORRECT |
| `configs/ours256.json` | CORRECT |
| `configs/ours1024.json` | CORRECT |
| `configs/ours4096.json` | CORRECT |
| `configs/repl1024.json` | CORRECT |
| `configs/sources.json` | CORRECT |
| `configs/target256.json` | CORRECT |
| `configs/target1024.json` | CORRECT |
| `configs/target4096.json` | CORRECT |
| `test_mmcore.py` | CORRECT |

**Blocking fixes:**

1. Restore complete Q1/Q2 predicates, case identity, and cohort anchors.
2. Verify every timed execution and reject missing/failed identity evidence.
3. Make controls 1–9 executable, mandatory, and submission-gating.
4. Implement the complete cost-input adapter, registered universe, aliases, and A4 accounting variants.
5. Fix missing-status logic and persist configuration terminal states.
6. Remove full-grid prefix copies and enforce the measured memory certificate.
7. Propagate job failures and enforce realistic probe/command budgets.
8. Make submission reservations atomic and correct ledger double-counting.
9. Complete checksum-verified archival and collection.