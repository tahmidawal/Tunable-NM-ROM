# PROGRESS — burgers2d-speed (kept current)

Lane `exp/2026-09-23-burgers2d-speed` (local commits only, never pushed). Namespace `/cluster/tufts/paralab/tawal01/b2speed_20260923/`.
Design, candidate set and pre-registered rule: `DESIGN.md` (committed before any dev6 job; amendment A0 after the Codex audit).

## Milestones
- 2026-09-23 ~16:15 EDT — engineering E1–E3 + driver written; local smoke 128² (GB10): parity 3e-15 with identical integers; the
  exact-first-step arm 4× faster (0.18 → 0.045 s at 128², local).
- ~16:25 — cluster smoke `smk` (job 4240333, A100, 128², not a result): pipeline incl. graphs compile mode runs on the cluster;
  parity ≤ 6e-15, identical integers.
- ~16:30 — component micro-benchmarks `bench1`/`bench2` (jobs 4240687/4240807, A100 40GB, synthetic arrays, diagnostics only):
  per call at R'=128 (default / graphs): separable evalJ 108 / 85 µs vs dense-Pq evalJ 214 / 179 µs; Cholesky factor+solve
  186 / 120 µs (cuSOLVER potrf alone 76 µs); LU 313 µs; three predictor residuals 54 / 36 µs. At R'=384: evalJ 193 vs 501 µs,
  Cholesky 310 µs. No faster Cholesky found among batched potrf, blocked (nb 32/64) and column-Crout variants. The per-step floor
  of the LM step (2 evalJ + 1 Cholesky + predictor) is ~0.35 ms at R'=128 in graphs mode, i.e. ~17 ms per 50-step query.
- ~16:40 — Codex read-only design audit (checks/codex-design-audit.md): no algebra errors; 5 blockers fixed before any real job (A0).
- ~16:50 — dev6 jobs b1024 (4241031), b256 (4241033), b512 (4241035) submitted, A100.

## Jobs
| attempt | job | mesh | GPU | state |
|---|---|---|---|---|
| smk | 4240333 | 128² smoke | A100 | done, smoke only |
| bench1 / bench2 | 4240687 / 4240807 | synthetic | A100 40GB | done, diagnostics only |
| b1024 | 4241031 | 1024² dev6 | A100 | submitted |
| b256 | 4241033 | 256² dev6 | A100 | submitted |
| b512 | 4241035 | 512² dev6 | A100 | submitted |
