**NEEDS-FIX.** Both round-4 fixes are correct. Two remaining issues affect the 3D smoke’s memory evidence.

Read-only audit: no files changed, jobs submitted, GPU used, or scratch created. **102/102 CPU checks passed**, with bytecode writes disabled and the file-writing test entry point bypassed.

Paths below are relative to `experiments/jcp-dims/`.

| Round-4 finding | Verdict | Evidence |
|---|---|---|
| `dtime` retains main + q+ + q++ simultaneously | **CORRECT** | `dtime.py:99–102` initially builds only main + q+. At `:132–146`, certification returns before q+ is dropped and q++ allocated. The q+ lattice fields survive at `:227`; `:124–130` computes the finer-rule distance and saves `S_<label>_qpp.npz`. |
| Duplicate invocations can hide missing cases/repetitions | **CORRECT** | `dstats.py:289–292` compares the exact invocation multiset. `d3run.py:436–440` constructs and supplies the expected `(subject, case, repetition)` inventory. The duplicate-case counterexample and positive inventory regression both pass. |

Ranked remaining findings:

1. **`smk3` does not pilot the actual three-table rounds path.**

   `configs/smk3.json:327–334` has no round members; its control has `qpp=null`. Its heavy lattice pilot instead uses `run_ladder`, with only lat131072 + lat262144 live (`:130–148`).

   Production round members include **W1024 lat65536 + lat131072 + lat262144** (`configs/q3r4.json:149–157`). `q3run.member` retains all three: q and q+ are referenced at `q3run.py:54`, then q++ is allocated at `:62`.

   A tiny CPU orchestration check confirmed:
   ```text
   lat65536
   lat65536 + lat131072
   lat65536 + lat131072 + lat262144
   ```

   Thus the smoke cannot demonstrate A1.4’s **actual heaviest footprint for each job type** (`DESIGN.md:489–492`). This is a coverage defect, not a demonstrated OOM.

   **Fix:** add a representative heavy rounds member with q++ to the smoke, using the production allocation path and recording its peak.

2. **The rounds control leaves obsolete tables live through subsequent smoke blocks.**

   Ordinary round members receive cleanup at `d3run.py:521–524`; the control call at `:527–528` does not. `q3run.member` returns without dropping its tables.

   In `smk3`, this leaves W512 lat8192 and lat16384 resident when the fixgate starts (`configs/smk3.json:327–347`). The subsequent timing baseline therefore also includes obsolete allocations; lat8192 remains unused through the panels. This violates A1.4’s release-after-use requirement and contaminates the intended footprint measurement.

   **Fix:** apply the member cleanup to the control too, before entering the next block.

No reproducible smoke crash was established. Deferred analysis implementation is not included as a blocker.

**Overall: NEEDS-FIX — (1) exercise the heavy rounds/q++ path; (2) release control tables after use.**