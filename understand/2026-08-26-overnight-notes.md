# Overnight run notes — 2026-08-26 (exact linear terms → gradient-target EQ → learned nodes)

Running log of the autonomous session executing
`understand/2026-08-26-autonomous-run-handoff.md`. Newest entries at the bottom. Every number
here is copied from generated output (JSONs / script stdout), never typed from memory.

---

## 00:19 — Task 0: dn1024 / dn256b pulled, verified, committed

Both `burgacc/` jobs finished COMPLETED 0:0. Pulled with
`runs/pull_burgacc.sh <dir> <jid> --delete`; all five checks passed for both
(RESULTS.sha256, `jax_backend=gpu`, stage MANIFEST, ALL-DONE, no captured-constant
warning); remote dirs deleted. Committed on `exp/2026-08-25-sepdec-consolidated`
(commit `b97d86c`), pushed.

**What they say** (from `sep_speed_r5_dense_mid_*.json`):

- **dn1024** — dense_mid recipe trained at full density, N=1024, K=32, R=512 (H200,
  job 2837430): rollout err mean **9.69e-3**, matched-accuracy paired timing
  **33.2 ms ROM vs 63.2 ms classical → 1.90×**. Gates: step-variant vs incumbent 0.0,
  EQ bank vs meshfree 1.4e-15, IC Gram vs full 4.5e-11.
- **dn256b** — same recipe, N=256, K=16 (A100, job 2837431): rollout err mean
  **8.96e-3** (matches round 5's dense_mid), paired **43.9 ms vs 16.4 ms → 0.37×**.
  ROM still loses at N=256, as expected from the crossover.

Verdict: the full-density dense_mid recipe **confirms at N=1024** — same ~1e-2 error class
as round 5 and a 1.90× paired speedup with the optimized solve path (vs 1.61× for the
previous K=16 champion). The K=32 timing gap flagged in the lab log ("timed only on the
un-optimized path") is now closed: this IS the optimized path.

## 00:19 — worktree `2026-08-26-eq-learned` created

Branch `exp/2026-08-26-eq-learned` cut from `exp/2026-08-25-sepdec-consolidated` at
`b97d86c`, pushed `-u`. Copied from the ladder archive branch: `sep_eq_ladder.py`,
`runs/gen_eq_ladder.py`, `runs/pull_eqladder.sh`, and the four
`cluster/*eqladder*/run_lad*.sbatch` scripts (to be cloned as `*exlin*`).
Cluster namespace for this session: `/cluster/tufts/paralab/tawal01/exlin/`.

Next: Stage 1 — write `sep_burgers_exlin.py` (exact linear terms), gates L/A/F, N=64 smoke.
