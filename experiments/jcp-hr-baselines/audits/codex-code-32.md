**READY** for re-submitting the 2D 1024² Stage-B job within this focused code-audit scope. No new blocking defect found.

Actual HEAD is `4fef85bf7` (heartbeat); `stageb.py` is identical to requested `c820d3c4e`. Paths below are under `experiments/jcp-hr-baselines/`.

1. **CORRECT — nonblocking:** Comparator cases now force saves immediately after recording evidence (`stageb.py:210–214`).

2. **CORRECT — nonblocking:** Completed/failed timing pairs force saves (`stageb.py:375–378`); budget-skipped pairs force saves before continuing (`:347–350`). Whole-comparator budget skips also force persistence (`:336–340`).

3. **CORRECT — nonblocking:** Successful timing comparators now force a save (`stageb.py:382–384`); failed comparators retain their forced save (`:385–389`).

4. **CORRECT — nonblocking:** Against `1876181dc`, the `stageb.py` diff contains only checkpoint changes and the closing `try/finally`. Numerical computation, qualification, ordering, budget predicates, warmups, output checks and timing rules are unchanged (`stageb.py:292–314,327–378`). Checkpoint overhead can affect how much work fits within the existing budget.

5. **CORRECT — nonblocking:** The `finally` closes the evidence file on normal and exceptional timing-phase exits, including failures in phase checkpointing or final saving (`stageb.py:392–397`). Earlier failures before entering this block remain outside its protection; this is an existing limitation, not a new defect.

Read-only inspection only; no files changed, jobs submitted or GPU work performed.