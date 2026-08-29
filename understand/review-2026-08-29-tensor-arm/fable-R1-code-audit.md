# R1 — code audit of the B1D tensor (sample-free advection) result

Reviewer angle: leakage, cheating, mismeasurement. Read-only; one CPU check run.
Paths below are relative to `worktrees/2026-08-29-b1d-tensor/experiments/separable-decoder/`.

## Verdict

**Legitimate with corrections** (all minor; nothing blocking, nothing major). The tensor is built only
from the frozen bank and the sine modes, the oracle is the committed baseline's code path, the two arms
share everything except the advection term, error is measured against regenerated FOM truth on a
held-out seed, and I confirmed by direct perturbation that the two arms are genuinely different code
paths. The corrections are to wording in the report (timing-fairness phrasing, the "vs oracle" ratio,
the strength of the stop-reason evidence, and one unmentioned gate maximum).

## What I checked, with evidence

### 1. Is T built only from the frozen bank and the test modes? — Yes.
`b1d_tensor_common.py:36-53` `build_T(Phi, G, dx)` consumes only `Phi` (sine modes,
`b1d_common.py:test_modes_1d`, a function of N and M), `G` (= `su.G_int`, `features(params, coords)` at
`b1d_fast_common.py:88-89`, the frozen bank from the committed checkpoint) and `dx`. Called at
`sep_b1d_tensor.py:293-296` with `Phi_np, G_np, dx`. No `U_test`, `Z_tr`, `nu_test` or any oracle output
enters. `Z_tr` (training latents) is used only in gates T0/TA/TQ and in the shared solver constants
(`TR_DELTA`, `Z0S`, `b1d_fast_common.py:95-104`) that every arm uses. Gate TX
(`sep_b1d_tensor.py:303-308`) shows the in-job Q equals the locally built Q to 8.9e-16; my own rebuild
from the checkpoint reproduces `runs/b1dtensor/audit/Q_n128.npy` exactly (diff 0.0).

### 2. Is the oracle a legitimate reference and the baseline's code path? — Yes.
`b1d_fast_common.py:150-161` `make_full_rw` computes `Nu = b1.upwind_adv_field_1d(G_int @ hz, N)` and
projects with `Phi_j.T`. `upwind_adv_field_1d` (`b1d_common.py:123-131`) is the sign-upwind stencil used
inside the FOM residual (`fom_residual_int`, `b1d_common.py:133-139`), which is what the tridiagonal
truth generator Newton-solves (`make_rollout_1d_tri`). It is textually the same as `full_r_w` in the
committed `sep_b1d_scale.py:234-240`. Gate F (`sep_b1d_tensor.py:255-273`) asserts the oracle weak
residual equals `Phi^T` of the FOM residual (2.6e-15). The re-run oracle matches the committed baseline
JSON to 2.3e-8 (N=128) … 3.9e-10 (N=512), same stop histograms.

### 3. Do tensor and oracle share everything except advection? — Yes.
Same `Setup` instance (`sep_b1d_tensor.py:173`), same `ic_ref`/`ic_fast` (lines 379-380, 468-470), same
`decode_all` (476), same `prev_of`, `TR_DELTA`, `Z0S`, tolerances, LM logic. The only branch is
`make_device_fast` `if tensor: Pq = 0.5*((Qj@hz)@hz) elif X_v is None: upwind…` (`b1d_fast_common.py:
420-428`) and the equivalent in the lean fold (`:459-491`); the reference path differs only in
`make_tensor_rw` vs `make_full_rw` (`:150-176`). Arm dispatch `sep_b1d_tensor.py:440-453` passes
`Q=Qarm` only for `tensor`/`tensor_nolean`; the `assert not (tensor and X_v is not None)` guards
mixing. In the lean fold `Mstack = concat([A2, eyeR])` (`:466`) so `y[Mloc:]` is exactly `h` — no
hidden approximation.

### 4. Is error measured against FOM truth, not the oracle/decoder? — Yes.
`sep_b1d_tensor.py:498-499`: `rel(Ff[t], U_test[ti, t][interior])`, `U_test` from `fc.gen_test`
(`b1d_fast_common.py:63-72`: `sample_params_1d(TEST_SEED=1, 8)`, `make_rollout_1d_tri`, FOM residual
asserted <= 1e-8). Gate T2 cross-checks the tri generator against the dense one (3e-16).

### 5. Are test trajectories disjoint from training? — Yes.
Training data: `DATA_SEED=0` (`sep_b1d_scale.py:53`, baseline JSON `data_seed: 0`); test: `TEST_SEED=1`.
Independent `default_rng` streams. My check: minimum distance between any of the 512 training
parameter tuples (c, w, a, log nu) and the 8 test tuples is 0.060 — no duplicate.

### 6. Are the code paths really different? — Confirmed by perturbation (CPU, N=128 checkpoint).
| state | min u | frac u<=0 | r rel (tensor vs oracle) | J rel |
|---|---|---|---|---|
| training latent | -7.9e-5 | 1.6 % | 1.4e-7 | 4.2e-9 |
| 3x latent | -2.0e-2 | 14 % | 9.6e-7 | 9.1e-6 |
| z + 0.5·N(0,I) | -1.41 | 21 % | **4.2e-3** | **7.9e-3** |
| field negated (h -> -h) | — | ~100 % | q rel **9.4e-2** | — |

The mismatch grows monotonically with the negative region, exactly as a backward-difference tensor
vs a sign-switching oracle should; there is no shared cache or accidental call into the oracle path.
Fast-lean tensor vs reference tensor rollout: 1.5e-13 latent; tensor vs oracle: 1.6e-9 (traj 0). The
in-job gate TQ shows the same (r rel max 5.6e-4, J rel max 1.4e-3 on perturbed states, log line
`GATE TQ`).

### 7. Timing fairness.
- Same GPU per job; every arm timed with `perf_counter` + `block_until_ready` (`sep_b1d_tensor.py:
  466-481`); burn-in 2, 5 timed reps, all persisted (`:486-490`, `times=` in JSON); accuracy from the
  last timed invocation (`:491-493`). e2e = ic_fast + roll_fast + dec (`:541-545`), identical
  components for every arm; `decode_all` (O(n)) is inside the timed region for all arms.
- Per-trajectory folds (`Mfold`, `Afold`, …) are computed **inside** the jitted `rollout_fn` for both
  the tensor lean path (`b1d_fast_common.py:455-474`) and the sampled lean path (`:493-505`) — both
  arms pay them inside the timed region. Q itself (32^3, 256 KiB) is a compile-time constant, the
  analogue of the sampled arms' precomputed `G3`/`PhiqT`. Nothing O(n) is hidden for the tensor arm.
- **Minor (wording):** "arms interleaved" (report Ops record; notes line 58) is inaccurate. What is
  interleaved is ref-vs-fast *within* an arm (`one_rep`); the arms themselves run sequentially
  (`for arm in ARMS:` `:437`, oracle first, tensor fourth). Risk is small (jobs ~1 min, dedicated GPU,
  baseline column agrees 1–3 %), but say "ref/fast interleaved, arms sequential".
- **Minor (ratio choice):** the oracle fast path has no lean/nodot variant (`use_lean = … and (X_v is
  not None or tensor)`, `:379`), so "0.80–0.90x the oracle" (F3) compares a lean tensor to a non-lean
  oracle. The like-for-like number is `tensor_nolean`/oracle = 0.92 / 0.95 / 0.95 / 0.88. Report both.

### 8. Do the gates test what the report says?
- T0 (`sep_b1d_tensor.py:325-340`): tensor `q_T` vs oracle `q_or` on training states with `min u > 0`
  — yes, exactly as described, asserted < 1e-12. Note it is on *training* states (8192), which is fine
  for an identity test.
- TA (`:314-324`): `q_T` vs an independent numpy `(U * D^-U) @ Phi` — independent algebra, good.
- TB/TX: two summation orders / two machines. Fine.
- TQ is "recorded, not asserted" and its perturbed-half maximum (J rel 1.4e-3, min u -0.30) is
  disclosed in the notes (line 260) **but not in the report**, whose F2 states "Jacobian relative
  mismatch <= 5e-5" (the E1 candidate value). **Minor:** add one clause that off-manifold states
  reach 1e-3 and that the trust region (0.017) is what keeps rollouts away from them.
- Gate V (`:638-652`): device tensor rollout vs host-loop tensor rollout, 1e-13. Fine.
- E1 audit (`b1d_tensor_audit.py:205-330`) probes tensor-vs-oracle at every LM candidate **of the
  oracle's rollout**, not the tensor's own; acceptable because the two latent paths stay within
  2e-4 (N=128) / <=1.2e-5 (N>=256) of each other, but the report should say "along the oracle path".

### 9. Is the stop-reason histogram comparison meaningful? — Weakly.
Every arm, including NNLS-32, ends 98–99.5 % of steps on reason 2 (stall, rel decrease < 1e-3) and
the rest on budget; NNLS differs from the oracle by 1–2 counts. "Identical histograms" therefore has
little discriminating power. The JSON contains a stronger per-trajectory statistic that the report
does not use: `mean_njac_fast` (accepted Jacobian evaluations per step) is bit-identical between
tensor and oracle on 31/32 trajectories (only N=128 traj 7: 5.72 vs 5.74; `comparison.tensor.
per_traj[*].njac_equal`). **Minor:** cite that instead of, or beside, the histogram.

### 10. Could the <=1.2e-6 agreement be an artefact? — No.
It is a per-trajectory absolute difference on errors of ~5e-3 (2e-4 relative), with latent
deviations 2e-4 → 6e-6 and a matching ref-path difference (`err_abs_diff_max_ref` equals the fast
one). Gate TQ and my perturbation check show the arms diverge when the field goes negative; along
the rollouts the field only undershoots to -8e-3 (E1), so the agreement is the expected consequence
of F2, not of shared code.

## Findings by severity

| # | severity | finding | fix |
|---|---|---|---|
| 1 | minor | "arms interleaved" is wrong; arms are sequential, only ref/fast interleave | reword Ops record / notes |
| 2 | minor | "0.80–0.90x the oracle" is lean-tensor vs non-lean-oracle; like-for-like is 0.88–0.95x | give `tensor_nolean`/oracle in F3 |
| 3 | minor | report's "J rel <= 5e-5" omits TQ perturbed max 1.4e-3 (min u -0.3) | add one sentence + trust-region caveat |
| 4 | minor | stop-reason histogram is ~all-stall for every arm; low evidential weight | cite `mean_njac` equality (31/32) |
| 5 | nit | E1 sign audit is along the oracle's path, not the tensor's | say so |
| 6 | nit | error is over interior points only (walls are zero) — harmless, but state it in the glossary | wording |

No blocking or major issues. No leakage of test data, truth, or oracle outputs into T; no hidden
O(n) work; no shared residual cache; the reference is the committed baseline's path and matches its
JSON. The result is legitimate as a single-seed, 8-trajectory, N=128–1024 claim once the four minor
wording corrections are applied.
