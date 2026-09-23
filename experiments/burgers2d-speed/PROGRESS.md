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

- ~17:40 — dev6 panels complete, pulled with checksums, audited (`checks/b*-summary.json`): **all gates pass at all three
  meshes** (parity incl. 51 internal states ≤ 5e-15 vs parent text, graphs vs default bitwise 0; FOM modes bitwise identical;
  drift 0.97–1.02; case-controlled neighbour ≤ 1.015; ρ NumPy spot check; controls rejected).
  Selection (DESIGN §6), `selection-<L>.json` committed:
  256²: accurate R'=384 linear without the exact first step 0.166 %, 45.7 ms, 0.37× (parent 125.6 ms, 0.13× in the same job);
        fast R'=128 linear 1.60 %, 17.4 ms, **0.97×** vs FOM lean_nt3e-3_l3e-3_dt005 (graphs) 16.9 ms — just short of 1×.
  512²: accurate 0.195 %, 45.9 ms, 0.60× (parent 233 ms); fast R'=128 linear 1.75 %, 17.4 ms, **1.59×** (FOM 27.7 ms).
  1024²: accurate R'=384 linear 0.211 %, 49.5 ms, **1.46×** (parent 81.9 ms, 0.88×); fast 1.83 %, 21.3 ms, 3.39× (FOM 72.1 ms).
  Dropping the exact first step costs nothing in worst error at 256²/512² (identical worst error; ties go to the faster knob).
  The LM cap-1 knob fails every certificate except at 1024² and is 2–4× less accurate: never selected.
  §6.3: general path 58.5 ms vs fast path 31.6 ms at 1024² (R'=128, parent text): X = 1.85, errors 1.8278 % vs 1.8281 %.
- ~17:50 — held-out jobs h1024 (4242036), h256 (4242040), h512 (4242063) submitted from the committed selection; remote dev
  directories deleted after the verified pulls (smk/bench outputs were diagnostics, logs read, not archived).

- ~18:00 — Codex results audit of dev6 (`checks/codex-results-audit-dev6.md`): every selection, speedup, certificate,
  parity and gate independently confirmed; one bug found in an auxiliary before/after factor (LM-budget-1 parent arms matched
  to uncapped engineered arms) — fixed, summaries regenerated, selection unaffected.
- ~17:35 — held-out hold64 jobs complete, pulled, audited: **all gates pass** at all three meshes (parity at 1024² 3e-15).
  256²: accurate 0.875 % worst / 0.026 % median, 45.1 ms, 0.38×; fast 2.547 % / 0.268 %, 18.2 ms, **0.94×** (FOM 17.1 ms).
  512²: accurate 1.128 %, 46.3 ms, 0.61×; fast 2.781 %, 18.2 ms, **1.56×** (FOM 28.4 ms).
  1024²: accurate 1.250 %, 53.1 ms, **1.56×** (parent code 89.1 ms, 0.93×); fast 2.960 %, 22.1 ms, 3.75× (FOM 82.7 ms).
  Remote namespace emptied and removed. Report `reports/2026-09-23-burgers2d-speed-small-meshes.md`; lane summary
  `checks/lane-summary.json`.

## Jobs
| attempt | job | mesh | GPU | state |
|---|---|---|---|---|
| smk | 4240333 | 128² smoke | A100 | done, smoke only |
| bench1 / bench2 | 4240687 / 4240807 | synthetic | A100 40GB | done, diagnostics only |
| b1024 | 4241031 | 1024² dev6 | A100 80GB (pax049) | done, pulled, audited, remote deleted |
| b256 | 4241033 | 256² dev6 | A100 80GB (pax106) | done, pulled, audited, remote deleted |
| b512 | 4241035 | 512² dev6 | A100 80GB (pax105) | done, pulled, audited, remote deleted |
| h1024 / h256 / h512 | 4242036 / 4242040 / 4242063 | hold64 | A100 (pax003 / pax049 / pax049) | done, pulled, audited, remote deleted |
