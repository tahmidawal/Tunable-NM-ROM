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

## 00:52 — stage-1 cluster wave submitted; first N=1024 results pulled

Six jobs in namespace `exlin/`: ladder arms xl256r3a (2844813), xl256dm (2844814),
xl1024k16 (2844815), xl1024k32 (2844816); speed A/B arms xs256dm (2844824, same
checkpoint+protocol as dn256b) and xs1024dm (2844825, partner of dn1024).
One near-miss caught before submission: `exlin_common.py` was not matched by the stage
script's `sep_*.py` glob — fixed and restaged before anything went up.

**xl1024k16 and xl1024k32 already finished and pulled clean** (gpu preflight, MANIFEST,
RESULTS.sha256, no captured-constant warnings; remote dirs deleted). On the K=32 r4a6
checkpoint, exlin + advection-only refit vs the 2026-08-25 incumbent ladder, same states:

| quantity (N=1024, K=32) | old (incumbent) | new (exlin + adv-only) |
|---|---|---|
| oracle (b), fine set M256 | 0.214 | **0.095** |
| oracle c1 cos, fine set | 0.819 | **0.902** |
| solver-path c3 cos, fine set | 0.659 | 0.805 |
| rollout err, ctrl (m=256) | 3.03e-2 | **2.47e-2** |
| rollout err, fine (m=1024) | 1.435e-2 | 1.435e-2 (unchanged) |

Reading: the quadrature is measurably more faithful at fixed m (the residual error is now
purely advection, and it is smaller), and the coarse-set rollout improves ~20%. The
fine-set rollout does NOT move — consistent with "the binding rung is h's generalisation"
from the lab log: at m=1024 the quadrature was already not the limiter for this checkpoint.
Numbers above are from the two JSONs (commit on the experiment branch); the generated
tables will be the report source.

## 00:52 — stage-2 smoke: off-manifold + gradient rows are a big fidelity lever

`sep_eq_gradfit.py` compares four node/weight sets at fixed m=256 (M=64), all online with
the exact-linear residual, certified on HELD-OUT solver iterates (never in any fit).
N=64 smoke (`runs/smoke_exlin/sep_eq_gradfit_N64_smoke_gradfit.json`):

| set (what the NNLS sees) | held-out (b) | held-out c1 | c1 cos | c3 cos |
|---|---|---|---|---|
| inc — u+N rows at training codes (incumbent) | 5.1e-2 | 0.57 | 0.802 | 0.846 |
| adv — N rows at training codes (stage 1) | 4.4e-2 | 0.45 | 0.595 | 0.648 |
| path — N rows at off-manifold LM iterates | 7.8e-3 | 0.105 | 0.990 | 0.989 |
| grad — path + frozen-J_f gradient rows | 9.0e-3 | **0.054** | **0.997** | 0.990 |

WHERE the rows are evaluated (solver-path states instead of training codes) is worth
~5× on the residual rung and ~4× on the gradient rung; the gradient-teacher rows halve
the gradient error again. Test rollouts at N=64 are unchanged (~2.9e-2) — rollout impact
is what the cluster arms at N=256/1024 will measure. Gradfit wave submitted:
gf256m64 (2844866), gf256m256 (2844868), gf1024m64 (2844869), gf1024m256 (2844870).

## 01:02 — STAGE-1 HEADLINE: −18% rollout error at N=1024, cost unchanged

The A/B speed arms are back (same checkpoint, same protocol, same GPU type as the
incumbent runs pulled in Task 0):

| arm | rollout err (base variant) | matched-accuracy paired timing |
|---|---|---|
| N=256 incumbent (dn256b) | 8.96e-3 | ROM 43.9 ms vs FOM 16.4 ms → 0.37× |
| **N=256 exlin (xs256dm, 2844824)** | **8.02e-3 (−11%)** | 46.9 vs 16.9 → 0.36× |
| N=1024 incumbent (dn1024) | 9.69e-3 | 33.2 vs 63.2 → 1.90× |
| **N=1024 exlin (xs1024dm, 2844825)** | **7.98e-3 (−18%)** | 33.6 vs 62.9 → 1.87× |

Exact linear terms + advection-only node refit is a pure accuracy gain at zero cost:
the paired speedup is unchanged within noise, and every accuracy variant in the sweep
improved. Gates in both jobs: gate 0 ≤5.6e-15 (incumbent-form ops), gate L ≤3.4e-15,
gate A ≤3.0e-14. This confirms the ladder's #1 fix prediction with the round-4 protocol.

## 01:02 — stage-2 first cluster arm (gf1024m64): the gain is monotone in row quality

N=1024, K=32, coarse set m=256 (job 2844869, pulled+deleted): rollout error
inc 2.61e-2 → adv 2.47e-2 → path 2.19e-2 → **grad 2.07e-2** (−21% vs incumbent),
monotone in what the NNLS rows see. Held-out fidelity: (b) 0.39→0.15, c1 cos
0.46→0.95 (adv→grad). BUT the test-solver-path numbers move much less
((b) 0.39→0.34) — there is a real generalisation gap between fit-side (training
trajectory) iterates and test-trajectory iterates. Late-time test-path (b) is even
slightly worse for path/grad than adv at this m. The other three gradfit arms
(N=256 both m, N=1024 m=1024) are still running.

## 01:02 — stage-3 smoke passed; 4 overnight arms submitted

`sep_eq_nodefit.py` N=64 smoke: gate C (continuous machinery at grid init ≡ grid ops)
8.4e-15; gate L [node] 2.2e-16; node optimization loss 5.67e-5 → 1.29e-5 with mean node
move 2.8e-3 (~0.18 dx); learned nodes beat the stage-2 grad baseline on held-out (b)
5.1e-3 vs 9.0e-3 (−43%) at c1 cos 0.998 / c3 cos 0.998. Overnight arms (VARIANTS=
adv,grad,node, STEPS=3000/2000): nf256m64 (2844909), nf256m256 (2844910),
nf1024m64 (2844911), nf1024m256 (2844912). Pull with
`./runs/pull_exlin.sh <dir> <jobid> --delete` from the worktree's
`experiments/separable-decoder/`.

## 01:07 — stage-1 COMPLETE: all six jobs pulled and verified

xl256r3a (2844813) and xl256dm (2844814) pulled; `EQ-LADDER-EXLIN.md` generated on the
branch. The four-checkpoint picture, incumbent → exlin+adv-only at the same m:

- Fine set (m=1024): (b) on the solver path improves 1.9–3.3× (r3a 0.056→0.030,
  dense_mid 0.076→0.023, r3d 0.067→0.036, r4a6 0.112→0.082); oracle (b) improves up to
  4.8× (dense_mid 0.14→0.029); oracle gradient cosines now 0.90–0.99 everywhere.
- Coarse set (m=256): rollout error improves 7–20% on every checkpoint.
- Fine-set rollout error: UNCHANGED on every checkpoint (r3a 2.24e-2, dense_mid 1.11e-2,
  r3d 2.33e-2, r4a6 1.44e-2) — at m=1024 the quadrature is no longer the binding error
  for these checkpoints; h's generalisation is (consistent with the lab log).
- The speed-protocol A/B (T-X1) already showed the −11%/−18% rollout-error gain at
  unchanged cost with the default (ctrl) EQ set.

All six stage-1 cluster dirs deleted from `exlin/` after sha256+marker verification.

## 01:12 — stage 3 first arm: learned nodes BEAT the convex baseline (N=1024 K=32, m=256)

nf1024m64 (2844911, H200, pulled+deleted): optimization loss 1.32e-2 → 4.86e-4, mean node
move 6.9e-3 (~7 grid spacings), all 256 weights positive, gate C 8.6e-14, gate L 2e-16-ish.
Certification vs the convex sets, same m:

| set | held-out (b) | held-out (c1) | (c1) cos | (c3) cos | test-path (b) | oracle (b) | rollout |
|---|---|---|---|---|---|---|---|
| adv (stage 1) | 0.406 | 0.677 | 0.458 | 0.469 | 0.389 | 1.315 | 2.467e-2 |
| grad (stage 2) | 0.146 | 0.217 | 0.950 | 0.906 | 0.338 | 0.773 | 2.067e-2 |
| **node (stage 3)** | **0.071** | **0.170** | **0.977** | **0.985** | **0.262** | **0.527** | **2.005e-2** |

The success bar (beat grad on held-out c1/c3 AND on rollout at the same m) is MET on this
arm. The gain also generalizes to the test path (0.338→0.262), unlike most of the stage-2
convex gains.

## 01:12 — but stage 2 does NOT transfer on the N=256 dense_mid checkpoint

gf256m64 (2844866, pulled+deleted): held-out improves as always ((b) 0.19→0.05, c1 cos
0.61→0.97 adv→grad) but the TEST-path (b) degrades 0.21→0.39 and the test rollouts get
WORSE: inc 1.517e-2 / adv 1.505e-2 / path 1.714e-2 / grad 1.706e-2 (+13%). Same-target
refits can overfit the fit-state distribution; on this checkpoint they do. (Cross-check:
the adv arm's rollout 1.505e-2 reproduces xl256dm's ctrl rollout to the digit.) The
remaining arms (gf256m256, gf1024m256, nf256*, nf1024m256) will say whether this splits
by checkpoint/K or by node budget.

## 01:44 — the fine-budget arms: quadrature learning has nothing to win there

- **gf1024m256** (2844870): at m=1024 all four quadratures TIE on rollout
  (1.433–1.440e-2) and `inc` even has the best test-path (b) (0.051). The same-target
  lever only matters at coarse budgets.
- **nf1024m256** (2844912): a clean reportable negative for node learning at the fine
  budget — the init (grad set) loss was already 1.28e-5 (vs 1.3e-2 on the coarse arm,
  three orders lower), the optimizer wandered (final refit 2.42e-5, worse than init),
  the learned set certifies worse than both baselines, rollouts tied (1.44e-2). Where
  quadrature is not the binding error, there is nothing for learned nodes to win, and
  the optimizer can only overfit.

Emerging synthesis (pending the last two N=256 M=256 arms): (1) exact linear terms +
advection-only refit = unambiguous default, gains everywhere it can gain, zero cost;
(2) same-target/learned quadratures pay only at COARSE budgets and are
checkpoint-dependent (N=1024 K=32: node −3% rollout vs grad and −19% vs adv; N=256
dense_mid: the whole same-target family is worse than adv on rollout, node recovers
part of it); (3) at fine budgets the quadrature is simply not the limiter — `h`'s
generalisation is, as the lab log already says.
