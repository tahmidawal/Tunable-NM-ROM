**NEEDS-FIX.** Paths relative to `experiments/jcp-qualify/`.

| codex-code-7 blocker | Verdict | Evidence |
|---|---|---|
| Baseline reset bypass | **PARTIALLY** | Baseline mutation is removed, but `time256: 0` still validates with recorded fallback subjects. Staging prices the submitted job correctly; other jobs’ pending projections remain unchecked (`ledger.py:87–89,129–135`; `cluster/stage.py:43–49`). A3.1/A5.1 remain bypassable. |
| Unverified coverage / placeholders | **PARTIALLY** | `time256coversfb: 0` no longer works. `time256rplaceholder: 0.000001` still passes and rounds to zero pending hours (`ledger.py:103–115`). |
| Enlarged-panel lifecycle stalls | **RESOLVED** | With covering staged subjects, enlarged first panels and repeats pass in both SUBMITTED and COMPLETED states. Failed repeats cease counting and require replacement (`ledger.py:85–108`). |
| Subject counts charged as hours | **PARTIALLY** | Counts are no longer required, but legacy `fbsubjects256: 36` remains accepted and adds 36 fictitious GPU-hours (`ledger.py:87–89,115`). |
| Failed fallback can disappear | **PARTIALLY** | `nofb256: 0` and boolean values are rejected as bypasses. However, any older completed `fb256*` satisfies newly recorded fallback obligations—even when the job covering the new configuration failed (`ledger.py:96–102`). No fallback configuration coverage is checked. |

Additional blockers:

- **Staging-to-submission gap:** stage an ordinary `time256` before recording fallback state, then record fallbacks and reserve that already-staged panel. Validation passes: coverage checks inspect only existing live/completed panels, not the candidate being reserved. Submission does not rerun staging coverage (`cluster/stage.py:50–60`; `ledger.py:103–106,129`; `cluster/submit.sh:25`). An undersized, non-covering first panel can therefore launch.
- **Unverified Q2 state:** `fbstate 256 '[]'` can release the allowance without evidence that Q2 needs no fallback. State recording and pending replacement are separate transactions, contrary to A4.1/A5.1 (`ledger.py:96–98,140–153`).

Reproduced using extracted pure functions on copied dictionaries, with in-memory staged-coverage fixtures. No file changes, ledger commands, cluster calls, or GPU execution.