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

## Jobs
| attempt | job | mesh | GPU | state |
|---|---|---|---|---|
| smoke1 | 4197296 | 128² | A100 | done, smoke only, remote deleted |
| bk4096 | 4197441 | 4096² | H200 240G | cancelled while pending (same bug), never ran; remote removed |
| bk1024 | 4197443 | 1024² | A100-80G | FAILED in 2 s: SyntaxError (repeated keyword `family`, introduced after smoke1); fixed f4c950a0; remote removed |
| bk4096b | 4197473 | 4096² | H200 240G | submitted 02:45 EDT |
| bk1024b | 4197475 | 1024² | A100-80G | COMPLETED 17.6 min; collected, audited (all gates pass after amendment A1), remote deleted |
| bk2048b | 4198172 | 2048² | A100-80G | submitted |
