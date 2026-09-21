# bank-floor — HANDOFF (kept current)

**State 2026-09-20 ~23:00 EDT.** 7/8 GPU jobs used. Phase 1 done for both PDEs; Phase 2 Poisson done; Phase 2 Burgers running.

| attempt | job | outcome |
|---|---|---|
| bfp01 / bfb01 | 4052480 / 4052482 | FAILED 3 min: single-Gram POD sigma-ratio assert. Logs kept. |
| bfb02 | 4053195 | FAILED 3 min: eigh OOM. Logs kept. |
| bfp02 | 4053735 | PARTIAL (POD + ft512 good; died scoring cat1024 on the 1023 mesh). Collected, NumPy audit PASS, remote removed. |
| bfb03 | 4056956 | COMPLETED 1:36. Burgers Phase 1, all arms. Collected, audit PASS (`checks/bfb03-audit.json`), remote removed. |
| bfsp01 | 4059578 | COMPLETED 1:51. Poisson cat1024/cat2048 + Phase-2 solve/timing. Collected, rep + solve audits PASS, remote removed. |
| bfsb01 | 4071262 | SUBMITTED: Burgers Phase-2 (full-bank q=R dense solve for inc512/ft512/pod512/cat1024/pod1024/cat2048/pod2048 + FOM rows). Remote `.../bankfloor_20260920/bfsb01`. |

**When bfsb01 ends:** `python cluster/collect.py bfsb01` → write/ run a NumPy field audit (fields in `ckpt/bfsb01/fields`, truth = `fom_tight` fields) →
`python reports/gen_report.py` → delete remote dir → final lab-log entry (flock) → commit. One job left in the budget.

Generated tables: `reports/tables.generated.md`, `reports/summary.json`. Checkpoints: `ckpt/` (ignored) + `CKPT-MANIFEST.json`.

**Landmines found so far**
- `p-bank-head/checkpoints/head_K32_w0_s0.pkl` is on the WITHDRAWN `bank_R512_S192` bank (floor 0.8688 %). The right Poisson
  incumbent (`new_K32`) is `p-linear/checkpoints/primary_K32.pkl` → `incumbents/poisson_primary_K32.pkl`
  (source `exp/2026-09-17-p-linear` @ aa9b55ee). Caught by the fidelity gate (0.7458918 % reproduced after the swap).
- Codex CLI sandbox cannot spawn a shell here (bwrap RTM_NEWADDR); audits must inline the files into the prompt via stdin.
- Codex r2 found a real bug before any job: SVD-rotated POD basis broke nested-prefix ranks (fixed, known-answer check added).
- GB10 is heavily contended (load 40–60); the "sub-minute" smoke took ~10 min. `jaxrun` output does not reach a redirected
  file when backgrounded with `&`; pipe it instead.
