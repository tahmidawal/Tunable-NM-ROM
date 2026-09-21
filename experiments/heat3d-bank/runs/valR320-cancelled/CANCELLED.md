# valR256 (job 4141159) — cancelled by me 2026-09-21 17:23 EDT, no panel numbers

Bank training completed (worst validation floor 0.1256 % at 32^3, 0.1222 % at 64^3 and 128^3; `partial/job.out`), then the job
stalled in ptxas for >17 min with the GPU at 0 % while compiling `train.train_head`'s validation function (V x S x R broadcast,
1536 x 12288 x 256). Reproduced locally (`diagnostics/head_compile_smoke.py old`: still compiling at the 5-min timeout; `new`: 28 s)
and fixed with a matmul + top_k nearest-code search, bit-identical on a parity test (`diagnostics/head_validate_parity.py`).
The trained bank (`partial/bank.pkl`, sha256 5b6bc519…f34d, gitignored) is NOT used: the rerun retrains from scratch.
valR320 (job 4142297) was cancelled while still PENDING for the same reason (never started). Remote dirs removed.
