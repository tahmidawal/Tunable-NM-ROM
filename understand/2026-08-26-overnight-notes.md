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

## 00:32 — Stage 1 code written; N=64 ladder smoke PASSED (all gates green)

Code (commit `633b2ae` + cluster scripts `_exlin`, on `exp/2026-08-26-eq-learned`):

- `exlin_common.py` — `eq_fit_burgers_adv`: the NNLS node fit restricted to the
  advection row blocks (the u rows are gone because the linear terms are now exact).
  Same seed, same subsample size, same padding, same final refit as the incumbent fit.
- `sep_eq_ladder.py` gained `EXLIN=1` (sampled residual computes the linear terms
  exactly as `A h(z)` with `A = Φᵀ G_int` precomputed) and `EQ_ADV_ONLY=1`.
- `sep_burgers_exlin.py` — the round-4/5 speed protocol (`sep_speed_r5.py`) on the
  exact-linear residual; everything else byte-identical.

**Gate rule change (the one the report §3 announced).** Gate 0 = bit-identity of the
whole sampled residual to `make_weak_ops` cannot apply to the exlin residual BY
DESIGN (the linear part is now a different — exact — computation). It still runs on
the incumbent-form ops built on the same node set (code identity), and two new gates
cover the change: **gate L** (exlin linear part vs full-grid linear part, exactness,
≤1e-12) and **gate A** (exlin advection part vs incumbent advection part, ≤1e-12).

**N=64 ladder smoke** (local GB10, jaxrun, ckpt `sep_burgers_N64_K16_R64.pkl`,
`runs/smoke_exlin/`): gate 0 = 4.15e-16, gate F = 3.07e-16, **gate L = 4.36e-16,
gate A = 0.00e+00**. And the structural claim measured: `b_lin` collapsed to
1.3e-14–2.1e-13 (numerically zero) at every time bucket while (b) = `b_adv` —
the residual error is now purely the advection term, as designed. The
advection-only NNLS fit at M=64/m=256: rel fit 4.98e-3 on advection rows
(same support/pad shape as the incumbent fit). Speed-driver smoke reproduced the
fit to the digit (determinism across drivers confirmed).
