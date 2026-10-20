**FIX-FIRST — retry reservation succeeds, but its rollback leaves the ledger invalid.** Audited HEAD `f50910855`.

In-memory execution reproduced:

1. `validate`: passes.
2. `reserve f2br1 2.75`: succeeds, consumes pending `f2b`, and preserves total commitment at **33.915 / 55 GPU-h**.
3. Duplicate reservation: correctly refused.
4. `release f2br1`: restores pending **`f2br1`**, rather than **`f2b`**. Validation then fails: `never-dropped job f2b missing from the pending inventory`.

Correct restoration to use `base_of(attempt)` in [ledger.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/ledger.py:169), including the equivalent `settle --confirmed-absent` and failed-job `reconcile` paths.

Everything else checked passes:

- All **167 manifest entries** and **163 staged source/input files** verify against HEAD.
- Configuration unchanged; certificate hash matches.
- Projection reproduced: **1.7504056963 h**, giving **2.75 h** after margin and rounding.
- Certificate note corrected; four superseded certificates absent. Their replacements remain required before staging those jobs.
- Batch script and reservation agree: H200, GPU partition, dedicated directory, GPU preflight, f64/highest.

No files changed, submissions made, or LAB-LOG writes performed.