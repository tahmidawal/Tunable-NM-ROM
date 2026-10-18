**NEEDS-FIX.** Five round-3 findings are resolved. Invocation coverage remains incomplete, and the 2D timing smoke still violates the registered table-lifetime limit.

Read-only audit: no files changed, jobs submitted, GPU used, or scratch created. **100/100 CPU checks passed**, run with bytecode writes disabled and the file-writing test entry point bypassed. No reproducible smoke crash was found; real-size GPU behavior remains untested.

References below are relative to `experiments/jcp-dims/`.

| Round-3 finding | Verdict | Evidence |
|---|---|---|
| 1. Certification links accept failed numerical certificates | **CORRECT** | `d3run.py:448–490` now preserves identity, validity, finite fields and certificate counts separately. `s3ctx.py:157–159` retains certificate values and denominators; adaptive blocks retain their counts and steps at `d3run.py:206`. |
| 2. FS/QN nonfinite decoded fields accepted | **CORRECT** | `d3run.py:455` requires present, finite scoring fields; `:490` includes that condition in acceptance. Missing/infinite-field regressions pass. |
| 3. Final 2D fixgate memory unchecked | **CORRECT** | `dacc.py:303–305` checks the job peak after fixgate. `resources.py:59–60` independently rejects incomplete, memory-rejected or over-limit accuracy smokes. The 3D reader checks its peak at `:114`. |
| 4. Previous 3D ladder rung survives successor allocation | **CORRECT** | Previous-rung deletion now precedes successor construction: adaptive `d3run.py:127–129` before `:189`; FS/QN `:236–238` before `:253`. Final-rung cleanup is present. |
| 5. Round manifest schema incomplete | **CORRECT** | `manifest_cells.py:119–126` supplies mode hash, tolerances, weighting and solver metadata. All **17** stored `q3r4` entries contain the previously missing fields. |
| 6. Unexpected panel invocations accepted | **NEEDS-FIX** | `dstats.py:286–288` fixes unexpected **subjects**, but `:282–295` still validates only aggregate counts, without examining invocation case/repetition coverage. |

The certification separation **does not contradict A1.1** (`DESIGN.md:427–438`): certification produces evidence; numerical qualification applies the registered population-level predicate. Timing identity acceptance need not reject each individual case with a noncertified step. However, panel `valid=True` must remain distinct from numerical qualification, and subsequent analysis must enforce the timing population’s qualification before using its costs. I do not re-raise the deferred analysis implementation as a smoke blocker.

Two remaining defects:

1. **The configured `smk2t` holds three subject-rule tables simultaneously, contrary to A1.4.**

   [dtime.py:99](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/dtime.py:99) passes both `qp` and `qpp` into `P.Setting`. Its constructor eagerly builds every role at `p1common.py:237–241`.

   For the actual heavy smoke block (`configs/smk2t.json:65–98`), this means **gauss256 + gauss320 + gauss384**, alongside persistent A. All three survive certification; deletion occurs only afterward at `dtime.py:289–291`.

   This contradicts `DESIGN.md:486–487`, which specifies at most the current and next rung. It is a distinct defect from the repaired 3D previous-rung retention. The smoke therefore tests a different memory schedule from the registered one.

   **Fix:** stage certification so the current rule is released before allocating the second successor, preserving the small outputs needed for comparisons. I have not established that the present allocation exceeds 120 GB.

2. **Duplicate panel invocations can hide missing cases or repetitions.**

   Direct CPU reproduction against [dstats.py:272](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/dstats.py:272):

   - Required certification keys: A and B on cases 0 and 1, all true.
   - Expected B invocation count: 2.
   - Actual invocations: **case 0, repetition 0 twice**, with case 1 absent.
   - Result: **`valid=True`**.

   The current driver constructs the intended Cartesian product (`d3run.py:410–417`), so this is an acceptance-boundary defect, not a demonstrated normal-driver failure. Nevertheless, the claimed coverage check cannot detect corrupted or duplicated evidence.

   **Fix:** validate the exact expected `(subject, case, repetition)` inventory, rejecting duplicates, omissions and unexpected cases/repetitions. Add this counterexample to the regression tests.

**Overall: NEEDS-FIX. Ranked remaining work:**

1. Correct `dtime`’s three-rung allocation to satisfy the smoke’s A1.4 memory contract.
2. Complete `panel_accept`’s exact case/repetition coverage validation.