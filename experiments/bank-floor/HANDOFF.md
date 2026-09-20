# bank-floor — HANDOFF (kept current)

**State 2026-09-20 19:40.** Phase 1 (representation) running, 2/8 jobs used:

| attempt | job | node | what | remote dir |
|---|---|---|---|---|
| bfp01 | 4052480 | pax106 | Poisson: inc512, random, POD{128..2048}(+raw,+sub), ft512, cat1024, cat2048 | `/cluster/tufts/paralab/tawal01/bankfloor_20260920/bfp01` |
| bfb01 | 4052482 | pax106 (a100-80G) | Burgers: same arms | `.../bfb01` |

Source commit staged: see `runs/<attempt>/COMMIT.txt`. `result.json` is rewritten after every arm; checkpoints land in
`output/ckpt/`. Log markers: `jax_backend=gpu`, `ARM <tag> ...`, `REP-DONE`, `ALL-DONE`.

**Protocol deviation (recorded):** the account had 5 jobs running when I submitted; the second submit made it 7 (> 6).
Both started at once on idle GPUs, so nothing was queued behind them; later submissions wait until < 6.

**After completion:** checksum-verified pull (`OUTPUTS.sha256`) into `runs/<attempt>/output/` (git-ignored; move `ckpt/*` to
`experiments/bank-floor/ckpt/`), run `audit_rep_np.py`, commit `result.json` + audit + logs, delete the remote dir,
apply gate P1 (DESIGN), then Phase 2 for promoted banks.

**Landmines found so far**
- `p-bank-head/checkpoints/head_K32_w0_s0.pkl` is on the WITHDRAWN `bank_R512_S192` bank (floor 0.8688 %). The right Poisson
  incumbent (`new_K32`) is `p-linear/checkpoints/primary_K32.pkl` → `incumbents/poisson_primary_K32.pkl`
  (source `exp/2026-09-17-p-linear` @ aa9b55ee). Caught by the fidelity gate (0.7458918 % reproduced after the swap).
- Codex CLI sandbox cannot spawn a shell here (bwrap RTM_NEWADDR); audits must inline the files into the prompt via stdin.
- Codex r2 found a real bug before any job: SVD-rotated POD basis broke nested-prefix ranks (fixed, known-answer check added).
- GB10 is heavily contended (load 40–60); the "sub-minute" smoke took ~10 min. `jaxrun` output does not reach a redirected
  file when backgrounded with `&`; pipe it instead.
