# bank-floor — HANDOFF (kept current)

**State 2026-09-20 19:05:** DESIGN.md written + Codex audit r2 applied (Amendment A1). Code: `bf_core.py`, `bf_rep.py`,
`cluster/stage.py`. Local smoke in progress. No GPU job submitted yet (0/8 used).

**Next:** commit → `cluster/stage.py bfb01 rep-burgers` and `bfp01 rep-poisson` → rsync to
`/cluster/tufts/paralab/tawal01/bankfloor_20260920/<attempt>` → squeue check → sbatch → squeue check.

**Landmine found:** `p-bank-head/checkpoints/head_K32_w0_s0.pkl` is on the WITHDRAWN `bank_R512_S192` bank (floor 0.8688 %),
not the selected S3072 bank. The right Poisson incumbent (`new_K32`) is `p-linear/checkpoints/primary_K32.pkl`
(copied to `incumbents/poisson_primary_K32.pkl`, source `exp/2026-09-17-p-linear` @ aa9b55ee). Caught by the fidelity gate.
