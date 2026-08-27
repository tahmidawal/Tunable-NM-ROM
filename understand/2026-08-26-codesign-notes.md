# Running notes — co-design experiment (exp/2026-08-26-codesign)

Session log for the quadrature-aware co-training experiment. Design:
`understand/2026-08-26-codesign-design.md`. Worktree
`worktrees/2026-08-26-codesign`, branch `exp/2026-08-26-codesign`.

## 2026-08-26 — build + N=64 wiring

- Worktree created from `exp/2026-08-26-eq-learned` (has exlin residual + stage-3
  machinery), pushed.
- **Warm-start checkpoint for the mini-pilot:** the committed
  `runs/sepdec_r1/out/sep_burgers_N64_K16_R64.pkl` (r1 trainer, K=16, R=64,
  M=64, 8192 codes, uniform snapshot pick with seed 0 — mapping reproduced and
  gate-checked). Training snapshots regenerate via `bc.build_data(64)` (~60 s);
  cached locally at `runs/cd_smoke/data_n64.npz` (local convenience only, key
  (N, seed) checked; cluster runs regenerate).
- **`sep_codesign.py` written** (commit 6eba037): h-track (`params["h"]`,
  `h_lin`) + continuous node positions trained jointly against the moving
  full-grid advection teacher; bank frozen; weights NNLS-re-solved every
  REFIT_EVERY on the exact loss-form rows; four-term loss, each normalized by
  its warm-start value; in-driver certification of base AND co-trained with
  the same instrument (gate L, held-out b/c1/c3, r5-rule rollouts).
- **Two wiring bugs caught by gates, both fixed:**
  1. Gate C failed at 8e-7: the NNLS support at N=64 contains nodes exactly on
     the boundary-adjacent ring; the sigmoid-box logit clip shifted them ~1e-6.
     Fix: box widened by dx/16 → gate C 8.5e-15.
  2. Gate D (FD vs autodiff) failed two ways: (a) stop-gradient normalization
     denominators make FD disagree by construction → denominators frozen at
     warm-start values (loss now genuinely smooth in that respect, refit rows
     share the identical quadratic form); (b) the remaining mid-eps
     disagreement is REAL micro-nonsmoothness: `L_jac` contains the upwind
     Jacobian, which jumps ~1e-7 whenever a stencil sign flips. Measured: FD
     matches autodiff to 7 digits at eps=1e-7 on both hp and theta rays; at
     eps=1e-5/1e-6 the kinks dominate. Gate D now probes the kink-free window.
     Consequence for training: loss is piecewise-smooth with 1e-7-scale jumps —
     standard subgradient territory, fine for Adam.
- **Base certification at N=64, m=256** (`runs/cd_smoke/out/sep_codesign_smoke0.json`):
  rollout err mean 5.185e-2 over 4 test traj (traj 1 has a bad IC fit,
  t0 err 3.6e-1 — decoder limitation shared by all variants), held-out
  b 3.06e-2, c1 0.239 / cos 0.9595, c3 cos 0.9915, held recon 2.73e-2.
- **First training observation (100 steps, LR=3e-4, arm ii):** fit-side samp
  mismatch −87%, held-out samp only −11%, reconstruction drift +5.4% →
  **tripwire fired as designed** (the FREEZE_WDEC guard works). LR was too
  hot for a fine-tune; dropped to 3e-5, smoke re-running.
- Runner for the arms: `runs/cd_minipilot.sh` (arms n/i/ii/iii × m 256/64,
  STEPS=2000, ≤3 concurrent local jaxrun).

## 2026-08-26 evening — smoke2 verdict: mechanism works, trade negative at generous m

`runs/cd_smoke/out/sep_codesign_smoke2.json` (N=64, m=256, arm ii, 500 steps,
LR=3e-5, REC_W=1):

- **Held-out mismatch rungs collapse** — the co-design mechanism genuinely
  works: (b) 3.062e-2 → 1.369e-2 (−55%), (c1) 0.2393 → 0.0311 (−87%), cos
  0.9595 → 0.9980, (c3) cos 0.9915 → 0.9993, on states excluded from every
  gradient and NNLS row.
- **But the trade is net-negative at this budget:** held recon 2.733e-2 →
  2.966e-2 (+8.5%; drift tripwire fired from step 100 on) and test rollout
  5.185e-2 → 5.539e-2 (+6.8%). At m=256 = 4×M the quadrature is not the
  binding error at N=64, so paying decoder quality for mismatch is a loss —
  the same shape as stage-2's negative transfer and FREEZE_WDEC.
- Ops: refits cost ~10 min each (full S·K jac row assembly, 139k×256 NNLS)
  — measured from the step-100→200 gap (583 s) repeating at 300→400.

Consequences applied (commit on exp/2026-08-26-codesign): arms run with
**REC_W=10** (anchor 10× so the optimizer cannot cannibalize recon),
REFIT_EVERY=500, jac refit rows subsampled to 16 states. The decisive budget
for the pilot is **m=64** (=M), where quadrature binds.

Mini-pilot wave 1 launched (detached; harness background tasks were being
killed, so runs are setsid-detached with rc files + Monitor): arms n/i/ii at
m=64, 2000 steps. Wave 2: same at m=256 + iii both m.
