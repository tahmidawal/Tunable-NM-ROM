# PROGRESS — burgers-bank-knob (kept current)

Lane `exp/2026-09-23-burgers-bank-knob` (local commits only, never pushed). Namespace `/cluster/tufts/paralab/tawal01/bbank_20260923/`.
Design + pre-registered setting rule: `DESIGN.md` (committed before any mesh job).

## Milestones
- 2026-09-23 02:25 EDT — rotation computed locally (`inputs/rotation_R512.npz`, sha256 51149166…, training codes only; ||LT−I|| 8.2e-12).
- 2026-09-23 02:30 — cluster smoke `smoke1` (job 4197296, A100, 128², 2 cases, not a result): whole pipeline ran; parity rotated R'=512 vs unrotated 4.5e-14 / 5.8e-14 with identical iterations; bad0 control fails (ρ 2.0); NumPy audit: all gates pass, injected controls detected, ρ recomputed to 4e-11. Remote dir deleted.

- bk1024b (1024², A100-80G): parity 4.6e-14/4.9e-14, identical iterations; order-effect gate PASSES (pooled 0.0 %, worst arm 0.13 %).
  Parent q256 re-measured deployed ρ = 0.0958 / 0.1157, exactly eqcert's bc1024 certificate.
  Rule (DESIGN §6): accurate = R'=384 q=256 (0.522 %, 168 ms, 0.44×); fast = R'=128 linear rung (1.83 %, 32 ms, 1.21×).
  lat64 FAILS for the linear rung at R'=512/384/256 (ρ 0.173/0.129/0.122 conf) → q NOT droppable at 1024² under certificates
  (uncertified linear R'=384 would be 0.21 % at 83 ms).
  Audit amendment A1 (restricted-grid comparator undercounts shock-front errors by ≤14 %; now one-sided ratio gate).

- bk2048b (2048², A100-80G): parity 5.3e-14/7.8e-14 identical iterations; order effect pooled 0.0 %, worst arm 0.3 %.
  Rule: accurate = R'=384 q=256 (0.534 %, 178 ms, 1.67× vs lean_nt3e-3_l3e-3_dt005 297 ms; current R'=512 q=256 0.598 %, 1.46×);
  fast = R'=128 linear (1.87 %, 35 ms, 4.40×; current q=0 2.37 %, 47 ms, 3.31×). q not droppable (linear R'≥256 fails lat64
  certificate: ρ 0.206/0.161 cert, R'=256 0.078 cert but 0.119 confirmation). ρ failures sit at the initial state k=0.

- bk512b (512², A100; accurate variant has an exact first step, x1): parity 5.3e-14/6.6e-14; order effect 0.0 % / 0.4 %.
  With x1 the linear rung CERTIFIES up to R'=384 (ρ 0.068/0.069); R'=512 fails (0.137). Rule: accurate = R'=384 LINEAR
  (0.195 %, 229 ms, 0.12× vs lean_nt3e-3 28.6 ms); fast = R'=512 q=0 (2.14 %, 35.5 ms, 0.42×). q droppable at 512²: YES.

- bk256b (256², x1): parity 4.8e-14/2.4e-13; order 0.0 %/0.8 %. Rule: accurate = R'=384 LINEAR (0.166 %, 124 ms, 0.15×);
  fast = R'=128 linear (1.60 %, 31.8 ms, 0.30×). q droppable: YES.

- Independent read-only subagent review (no blockers). M1: the certificate includes state k=j, which the EQ residual never
  touches (backward Euler evaluates advection at the NEW state); re-thresholded at k>=j+1 (labelled SENSITIVITY) the
  linear rung R'<=384 passes at 1024²/2048² and flips accurate -> R'=384 linear (0.211 %, 83 ms, 0.89× at 1024²;
  0.219 %, 90 ms, 3.30× at 2048²) and q-droppable -> yes. M2: rho spot-check did not cover the selected arms -> audit
  A3 now covers every arm (84 states, worst 9e-10) and recomputes every arm's coefficient map in NumPy (1.3e-14).
  Order statistic switched to means (medians degenerate); gaps <= 1.1 %. All four meshes re-audited, all gates pass.
- bx2048 (exploratory A2): M=2R' does NOT certify (ρ 0.120/0.126) and costs accuracy (R'=384: 0.433 % vs 0.219 %);
  exact first step (x1) certifies (ρ 0.049) but costs 2.46 s (R'=384) / 1.18 s (R'=256) -> unusable. Cross-job repeat
  of R'=384 linear / R'=384 q=256 / R'=128 linear: 89.4/176.3/34.6 ms vs 90.1/177.7/35.2 ms in bk2048b; errors identical.

- bk4096b (4096², H200): rule: accurate = R'=384 q=256 (0.545 %, 110.0 ms, 4.76× vs lean_nt3e-3_l3e-3_dt005 523 ms;
  current R'=512 q=256 0.604 %, 127.5 ms, 4.10×); fast = R'=128 linear (1.894 %, 24.1 ms, 11.15× vs lean_nt1e-2_l1e-2_dt01
  268 ms; current q=0 2.415 %, 41.5 ms, 6.47× vs the same comparator). q not droppable (pre-registered certificate);
  sensitivity k>=j+1: R'=384 linear 0.224 %, 59.6 ms, 8.78×. Current q=0/q=256 errors reproduce hb4k04 (2.4150/0.6043).
  At 4096² the paper's accurate rule (lat64, R'=512 q=256) is CONFIRMED on deployed states (ρ 0.107/0.108).
- Selection frozen: selection-4096.json (from checks/bk4096-summary.json sha256 3fd1537e…); hold64 config generated.

- bkh64 (4096² hold64, job 4210077, H200 pax011): FAILED at 2 h 20 min, after main-phase rep 4 of 5, when the paralab group
  share hit 100 % (0 bytes free; not this lane's data — the lane held 4.9 GB). result.json intact (2816 invocations = 64 cases
  × 11 arms × 4 reps; slow phase lean_tight / lean_nt1e-4 never timed). Pulled directly with remote sha256 of every file
  (verified), remote deleted. Audit: every gate passes except complete and retained_repetitions (4 < 5). Numbers (4 reps):
  chosen accurate R'=384 q=256 1.409 % / 106.6 ms / 5.05×; current R'=512 q=256 1.330 % / 113.5 ms / 4.74×; chosen fast R'=128
  linear 3.111 % / 24.3 ms / 11.20×; current q=0 9.030 % / 41.0 ms / 6.65×; sensitivity R'=384 linear 1.350 % / 60.1 ms / 8.95×.
  Current-arm errors reproduce hb4kh64 (1.3297 %, 9.0302 %). Rerun as bkh64b (same committed config) once disk freed (280 GB).
  COORDINATOR: the group disk was 100 % full ~10:30–11:10 EDT (another lane's dump); bkh64's last result.json write (11:01)
  falls in that window. The partial is SUPERSEDED and not reported as a result; only bkh64b (clean rerun) is. No dev-panel
  job overlapped the window (bk4096b ended ~08:05, bx2048 ~05:25, the rest earlier).

- bkh64b (clean rerun): all 64-case held-out errors identical to bkh64 to every printed digit — chosen accurate R'=384 q=256
  1.4094 %; current R'=512 q=256 1.3297 %; chosen fast R'=128 linear 3.1107 %; current q=0 9.0302 %; sensitivity R'=384 linear
  1.3498 % (median 0.0297 %). Timing (1 rep): 106.6 / 113.9 / 24.7 / 42.2 / 60.9 ms → 5.02× / 4.70× / 10.99× / 6.42× / 8.80×.

## Jobs
| attempt | job | mesh | GPU | state |
|---|---|---|---|---|
| smoke1 | 4197296 | 128² | A100 | done, smoke only, remote deleted |
| bk4096 | 4197441 | 4096² | H200 240G | cancelled while pending (same bug), never ran; remote removed |
| bk1024 | 4197443 | 1024² | A100-80G | FAILED in 2 s: SyntaxError (repeated keyword `family`, introduced after smoke1); fixed f4c950a0; remote removed |
| bk4096b | 4197473 | 4096² | H200 240G | COMPLETED 80.5 min (pax009); collected, audited (all gates pass), remote deleted |
| bk1024b | 4197475 | 1024² | A100-80G | COMPLETED 17.6 min; collected, audited (all gates pass after amendment A1), remote deleted |
| bk2048b | 4198172 | 2048² | A100-80G | COMPLETED 38.7 min; collected, audited (all gates pass), remote deleted |
| bk512b | 4200506 | 512² | A100 | COMPLETED 19.8 min; collected, audited (all gates pass), remote deleted |
| bk256b | 4202294 | 256² | A100 | COMPLETED 15.2 min; collected, audited (all gates pass), remote deleted |
| bkh64 | 4210077 | 4096² hold64 | H200 | FAILED (group disk full) after 4/5 main reps; partial pulled + audited; remote deleted |
| bkh64b | 4218386 | 4096² hold64 | H200 pax008 | quick phase (all 64-case errors) complete 13:11 (after the disk window); pax008 host ~3× slower (≈9 s host per timed invocation) → rep 0 took 1.9 h; CANCELLED after 1 rep (could not finish 5 in the limit); pulled with remote checksums, audited, remote deleted |
| bkh64c/d | 4233906/4234407 | — | — | cancelled while pending (never ran), superseded by leaner configs |
| bkh64e | 4235338 | 4096² hold64 | H200 | timed loop lean for held-out: no per-invocation re-score, subsample SHA each + full SHA every 8th, slow lean FOMs untimed; submitted 15:20 |
| bx2048 | 4202862 | 2048² | A100-80G | exploratory (A2) COMPLETED 31.5 min; audited (all gates pass), remote deleted |
