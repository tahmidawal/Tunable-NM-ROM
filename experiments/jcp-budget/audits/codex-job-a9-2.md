Audited **28ea508d9**, read-only. No files changed, submissions, SSH, or scratch files.

1. **CORRECT — blocker closed.** [anaref.py:142](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/anaref.py:142) excludes every window when `tight_after_refine=false`, `refined=true`, and whole-case outcome is `unresolved-floor`. The flag survives case-record construction. Pre-A9 and accepted-case behavior is unchanged. In-memory regression assertions match the recorded check; **5,184 gate combinations passed**.

   `ana7.py` consumes eligibility in window rows, growth, front ordering, B8 classification and ranking. `ana3d.py` uses those same eligibility and ranking functions. No bypass found. Separate W1 scoring against E0 remains intact.

2. **CORRECT — both stages match the pinned HEAD.** Each job has **157/157 source files** matching Git blobs and provenance, and **161/161 manifest hashes** passing. `COMMIT.txt` and reservation commits match HEAD; config hashes match resource certificates.

   | Job | Time limit | GPU | Host memory |
   |---|---|---|---|
   | ref2r | 05:15:00 | H200 | 240G |
   | ref3b | 06:30:00 | H200 | 240G |

   Against round-1 commit `405e43c9e`, staged-source differences are only `DESIGN.md`, `anaref.py`, and the added regression test. Solver, dependencies, configs and staging code are unchanged. The entire bundles are therefore not byte-identical; the job computation is unchanged.

3. **CORRECT — no additional submission blocker found in local evidence.** Live queue, remote disk and remote staging were outside this audit.

**ref2r: SUBMIT. ref3b: SUBMIT.**

Blocking fixes: none.