# PROGRESS — burgers-bank-knob (kept current)

Lane `exp/2026-09-23-burgers-bank-knob` (local commits only, never pushed). Namespace `/cluster/tufts/paralab/tawal01/bbank_20260923/`.
Design + pre-registered setting rule: `DESIGN.md` (committed before any mesh job).

## Milestones
- 2026-09-23 02:25 EDT — rotation computed locally (`inputs/rotation_R512.npz`, sha256 51149166…, training codes only; ||LT−I|| 8.2e-12).
- 2026-09-23 02:30 — cluster smoke `smoke1` (job 4197296, A100, 128², 2 cases, not a result): whole pipeline ran; parity rotated R'=512 vs unrotated 4.5e-14 / 5.8e-14 with identical iterations; bad0 control fails (ρ 2.0); NumPy audit: all gates pass, injected controls detected, ρ recomputed to 4e-11. Remote dir deleted.

## Jobs
| attempt | job | mesh | GPU | state |
|---|---|---|---|---|
| smoke1 | 4197296 | 128² | A100 | done, smoke only, remote deleted |
| bk4096 | 4197441 | 4096² | H200 240G | submitted 02:35 EDT |
| bk1024 | 4197443 | 1024² | A100-80G | submitted 02:35 EDT |
