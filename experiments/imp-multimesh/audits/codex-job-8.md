**J0: FIX-FIRST. J5: FIX-FIRST.** Passing `s1c/ACCEPT.json` remains an additional submission gate; it would not resolve the blockers below.

Paths below are relative to the worktree. Checked staged implementations match the working copies.

| Previous blocking item | Finding |
|---|---|
| Both: smoke-derived resource certificates | **NOT FIXED.** `checks/resources-j0.json:5` mixes one smoke snapshot rate with historical ECSW estimates and an unexplained 256 estimate. `checks/resources-j5.json:5` still uses historical whole-job times—not measured phases × current workload counts × 1.5. |
| J0: incremental artifacts and terminal states | **NOT FIXED fully.** Basis, codes, lattice basis and trust now persist before ECSW (`mmbuild.py:131–138`), and normal completion with no certified ranks produces an empty rules archive (`:169–170`). However, each rank rewrites the existing archive directly (`:165–166`): interruption during that write can destroy earlier rules. The active and pending ranks also lack persisted `UNRESOLVED` entries before construction starts (`:144–157`). |
| J0: reconcile path and outer budgets | **NOT FIXED.** `path_budget` computes remaining time (`mmbuild.py:70–73`), but that allowance is calculated **before contribution construction** and reused independently for initial NNLS and retry. See `pipeline.py:181–185`, `build.py:350–352,389–392` under `experiments/jcp-hr-baselines/`. |
| J0: construction burn-ins | **FIXED.** Explicit burn-ins precede snapshot, POD, projection-code, state and ECSW timers (`mmbuild.py:110,115,119,140,153`). |
| J5: finer-rule non-convergence | **FIXED.** `oursq.py:96–103` returns `UNVERIFIED` / `UNRESOLVED` when only the finer-rule convergence bar fails. A small in-memory fixture reproduced this. |
| J5: independent acceptance and descriptor identity | **NOT FIXED fully.** Descriptor identity is fixed, but acceptance remains incomplete (`oursq.py:73–101`). Only aggregate stationarity and aggregate production/tight validity are cross-checked; distances, tighter/successor/check validity, certificate agreement and Q2 indicators still rely on JSON. |

The remaining budget defect is material. With 4,500 seconds elapsed, J0 grants 1,500 seconds. Initial NNLS plus retry can then consume 3,000 seconds, **plus contribution construction**, exceeding the 6,600-second outer timeout. Recompute remaining time after contribution construction and share one deadline across both attempts and persistence.

Two persistence issues also need attention:

- Use atomic replacement for `rules.npz`, preserving the previous complete archive until its replacement is ready.
- The new budget-exhausted record at `mmbuild.py:149` has no `seconds`, but [`analysis.py:104`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/analysis.py:104) accesses `e['seconds']` unconditionally. This newly introduced path raises `KeyError` instead of propagating an unresolved construction.

The J5 acceptance gap is reproducible. In a small in-memory fixture, replacing saved tight/tighter coefficients with NaNs and setting saved `valid_qpp=False`, while leaving JSON unchanged, still returned:

```text
Q1 PASS  Q2 PASS  QUALIFIED
```

Recompute the qualification predicates from saved evidence and check per-case agreement, including array shapes and populations. The existing `experiments/jcp-p1-scaling/p1score.py:99–223` demonstrates the fuller checks.

The cohort-hash convention is **correct**: CPU checks confirmed byte-identical descriptor tables and matching hashes for both `dev6` and `val32`. `oursq.py:35` matches `p1common.sha` / `cohort_hashes` at `p1common.py:36–37,87–88`.

Other checks passed:

- S1b supplies empirical memory evidence: device peak **107,248,696,832 bytes**, recorded host RSS **29,024,944,128 bytes**, and the 4096 bank pilot. Accepted controls remain gated on **s1c**, correctly selected by `checks/S1_ATTEMPT` and `cluster/submit.sh:33–35`.
- Q3 is disabled for every configuration on all three meshes.
- Both certificate config hashes match. All checked manifest files ≤2 MB matched: **137 for J0, 142 for J5**.
- Staged `p1acc`, `p1common`, and `p1core` remain byte-identical; batch-script syntax passed.
- J2’s pulled log records GPU execution and successful extraction/head-fit completion. Its training-cost probe does not establish J0/J5 runtime feasibility.

No files changed, jobs submitted, large arrays loaded, or lab-log/main writes performed.