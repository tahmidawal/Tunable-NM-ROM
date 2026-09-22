# final256 (job 4171513) — FAILED after 4 min, no numbers

`jax.errors.JaxRuntimeError: INTERNAL: Autotuning failed for HLO: %gemm_fusion_dot_general = f64[6,16581375] fusion(...)` — the 256³
decode is one f64 GEMM against the 42 GB bank. Kernel/memory, not model. Fixed by blocked encode/decode (`core.row_blocks`,
`bank_project`, `bank_expand`, ~2M rows per block): 1 block at ≤128³ (identical path to every earlier run), 8 at 256³.
Parity vs the single GEMM with the real R=320 bank: project 1.0e-16, expand ≤1.3e-15 relative (`diagnostics/block_parity.py`).
Resubmitted as `final256b`. Remote dir of 4171513 removed.
