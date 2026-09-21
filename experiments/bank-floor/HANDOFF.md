# bank-floor — HANDOFF (kept current)

**State 2026-09-20 ~20:40.** 5/8 GPU jobs used. Phase 1 (representation) running:

| attempt | job | outcome |
|---|---|---|
| bfp01 | 4052480 | FAILED at 3 min: single-Gram POD assert (sigma_2048/sigma_1 = 1.1e-8). Logs in `runs/bfp01/`. Remote dir deleted. |
| bfb01 | 4052482 | FAILED at 3 min: same assert (8.0e-9). Logs in `runs/bfb01/`. Remote dir deleted. |
| bfb02 | 4053195 | FAILED at 3 min: OOM in 26k eigh (two snapshot arrays resident). Remote dir deleted. |
| bfp02 | 4053735 | RUNNING (A100 80GB pax050): Poisson all arms; POD + ft512 done, cat1024/cat2048 training |
| bfb03 | 4056956 | RUNNING: Burgers all arms (deflated POD, one resident snapshot array) |

Remote dirs: `/cluster/tufts/paralab/tawal01/bankfloor_20260920/{bfp02,bfb03}`. Log markers: `ARM <tag>`, `REP-DONE`, `ALL-DONE`.

Interim Poisson (from the bfp02 log, NOT yet pulled/audited): held-out dev12 worst floor inc512 0.746 %; POD 512 / 1024 / 2048 =
0.220 % / 0.0207 % / 0.0003 %; ft512 (6 min varpro fine-tune) 0.317 %.

**Protocol deviation (recorded):** at the first submission the account had 5 jobs running and I submitted two (7 > 6).
Since then every submit goes through a waiter that submits only when the account has < 6 jobs and refuses duplicates.

**After completion:** checksum-verified pull (`OUTPUTS.sha256`) into `runs/<attempt>/output/` (git-ignored; copy `ckpt/*` to
`experiments/bank-floor/ckpt/`), run `audit_rep_np.py`, commit `result.json` + audit + logs, delete the remote dir, apply gate
P1, then Phase 2: `bf_solve_poisson.py` (CPU-smoked OK) and `bf_solve_burgers.py` (smoke in progress) — 3 jobs left.

**Landmines found so far**
- `p-bank-head/checkpoints/head_K32_w0_s0.pkl` is on the WITHDRAWN `bank_R512_S192` bank (floor 0.8688 %). The right Poisson
  incumbent (`new_K32`) is `p-linear/checkpoints/primary_K32.pkl` → `incumbents/poisson_primary_K32.pkl`
  (source `exp/2026-09-17-p-linear` @ aa9b55ee). Caught by the fidelity gate (0.7458918 % reproduced after the swap).
- Codex CLI sandbox cannot spawn a shell here (bwrap RTM_NEWADDR); audits must inline the files into the prompt via stdin.
- Codex r2 found a real bug before any job: SVD-rotated POD basis broke nested-prefix ranks (fixed, known-answer check added).
- GB10 is heavily contended (load 40–60); the "sub-minute" smoke took ~10 min. `jaxrun` output does not reach a redirected
  file when backgrounded with `&`; pipe it instead.
