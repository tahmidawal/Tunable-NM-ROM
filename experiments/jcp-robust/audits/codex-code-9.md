**Overall verdict: NEEDS-FIX. Acceptance code should not yet be used to accept smoke outputs or release production.**

Audited HEAD `504e33ba5`. No files changed, scratch created, smoke outputs read, jobs submitted, or SSH/cluster commands executed. Checks used small in-memory CPU fixtures with bytecode writing disabled.

Paths below are relative to `experiments/jcp-robust/`.

1. **Round 8 #2 — NOT RESOLVED completely — NEEDS-FIX — BLOCKING.**

   The original overflow-of-distance example is now rejected, but **non-finite decoded trajectories can still pass**.

   Evidence: `smoke_accept.py:261–271` checks finite coefficients, finite decoder matrix, positive finite normalizers, and finite decoded **differences**. It never checks the decoded trajectories themselves. At `:264–266`, subtraction occurs before decoding, allowing common overflow-producing components to cancel.

   **CPU counterexample using the extracted `dd()` function:** identical finite arrays of shape `(8,6,256)`, filled with `1e308`, and a finite all-ones decoder matrix produced:
   ```
   decoded trajectories finite: False
   recomputed distances: [0, 0, 0, 0, 0, 0, 0, 0]
   ```
   All newly added numerical predicates pass. With otherwise complete evidence, the pilot predicate at `:278–284` can return a defined label and acceptance can proceed.

   **Fix:** Decode and require finite values for each of `p`, `s`, `t`, and `tt`, including the initial saved state. Then compute distances. Preserve acceptance of finite, numerically unqualified pilot outcomes.

2. **Round 8 #4 — RESOLVED — CORRECT — formerly BLOCKING.**

   Evidence:
   
   - `e2bank.py:27–38` now compares local-file hashes against configured hashes and recorded runtime hashes, including the incumbent branch.
   - `e2analyze.py:209–214` binds contributing accuracy jobs to the requested bank.
   - `e2analyze.py:215–216` binds the saved selection to the supplied selection job ID.

   These close the reported conflicting-recorded-hash and unrelated-selection-job paths.

   **Fix:** None for the original finding.

3. **Round 8 #5 — RESOLVED for mixed-bank job halves — CORRECT — formerly BLOCKING.**

   Evidence:
   
   - `select3d.py:43–58,67–71` checks each input job’s bank name, seed and registry checkpoint against the requested bank.
   - `e2analyze3d.py:66–83,95–100` performs the corresponding checks and binds selection provenance to both jobs.
   - `e2analyze3d.py:212–215` also checks the timing job.

   A seed-1 first half combined with a seed-2 second half now fails before merging. **The separate score-file binding hole remains below.**

   **Fix:** None for the original mixed-job-halves counterexample.

4. **Round 8 #6 — RESOLVED — CORRECT — formerly BLOCKING.**

   Evidence: `e2verdicts.py:28–40,100–112` fixes the population to registered seeds `(1,2,3)` and represents missing files as unavailable.

   CPU checks confirmed:
   
   - One or two agreeing available seeds → `inconclusive`.
   - Two distinct disagreeing available seeds → `seed-dependent`.
   - Three agreeing definitive seeds → `seed-robust`.

   **MINOR:** The documented `--seeds` option at `:13` is now silently ignored. Remove it from usage or explicitly reject it.

5. **NEW — 3D scores remain unbound to their source job and population — WRONG — BLOCKING.**

   `e2analyze3d.py:44–50` loads the supplied score JSON and retains only `['scores']`, discarding its provenance. Although `score3.py:111–114` records `run`, `job_id`, cohort seed and reference metadata, none is checked.

   The job-bank checks at `e2analyze3d.py:95–100` therefore do **not** validate `--sa` or `--sb`. Scores are merged at `:101–106` and used for errors, qualification comparisons and fronts at `:125–127,179–207`.

   **CPU counterexample:** the actual extracted `load_job()` accepted result job `111` with score metadata identifying job `222`. Identical configuration keys across seeds make accidental cross-seed score substitution possible.

   Population checking is also absent: `:180` intersects available score cases, so omitted score rows can silently reduce the comparison population.

   **Fix:** Before merging, validate score job identity, cohort/reference identity and exact case coverage against the corresponding result and coefficient mappings. Bind scores to the coefficient archive and bank hashes, or regenerate them from the supplied job. Apply the same provenance check to `e1analyze3d.py:59`, which also consumes external scores without checking their source.

6. **Smoke control wiring — CORRECT subject to finding 1 — MINOR limitations.**

   Source-level assessment, **not acceptance of actual outputs**:

   | Smoke | Can complete passing evidence reach ACCEPT? | Does every required failed/missing control or gate prevent ACCEPT? |
   |---|---|---|
   | **smk1b** | Yes; no positive-path blocker established. | Yes for the explicit E1 controls, reproduction, target certification and integrity gates (`smoke_accept.py:86–128`). |
   | **smk2** | Yes, including A6.3’s legitimate “verdict not reached” path. | No additional smoke-specific bypass established (`:142–193`). |
   | **smk3** | Yes, with fetched pilot bank, replay archive and accounting evidence. | **No:** non-finite decoded pilot trajectories can pass; finding 1. |
   | **smk4** | Yes. | Explicit E5 controls, fixtures, reproduction, populations and P-cross gates are mandatory; CPU drift test is also required (`:315–323`). |
   | **smk5** | Yes. | Same E5 job gates are mandatory (`:304–311`); the shared CPU drift test is exercised by smk4. |

   E5’s conjunction is explicit in `e5analyze.py:722–764`; job-level population, equivalence and identity failures block at `:123–163`. G-E5.2 correctly remains descriptive. The extracted C-E5.3 drift test passed; population checks rejected missing, duplicate and empty evidence.

   No additional blocking certificate/collection defect was established. Malformed evidence can sometimes terminate the wrapper instead of producing a complete rejection report; that is **MINOR**, since it does not produce ACCEPT.

**Two blocking findings remain: 1 and 5.** All 17 requested Python files parsed successfully; filesystem-writing tests and smoke wrappers were not executed.