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

## Jobs
| attempt | job | mesh | GPU | state |
|---|---|---|---|---|
| smoke1 | 4197296 | 128² | A100 | done, smoke only, remote deleted |
| bk4096 | 4197441 | 4096² | H200 240G | cancelled while pending (same bug), never ran; remote removed |
| bk1024 | 4197443 | 1024² | A100-80G | FAILED in 2 s: SyntaxError (repeated keyword `family`, introduced after smoke1); fixed f4c950a0; remote removed |
| bk4096b | 4197473 | 4096² | H200 240G | submitted 02:45 EDT |
| bk1024b | 4197475 | 1024² | A100-80G | COMPLETED 17.6 min; collected, audited (all gates pass after amendment A1), remote deleted |
| bk2048b | 4198172 | 2048² | A100-80G | COMPLETED 38.7 min; collected, audited (all gates pass), remote deleted |
| bk512b | 4200506 | 512² | A100 | COMPLETED 19.8 min; collected, audited (all gates pass), remote deleted |
| bk256b | 4202294 | 256² | A100 | COMPLETED 15.2 min; collected, audited (all gates pass), remote deleted |
| bx2048 | 4202862 | 2048² | A100-80G | exploratory (A2), submitted |
