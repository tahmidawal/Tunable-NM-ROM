# hb4k02 — FAILED after 10 min, no ROM number (job 4059827, H200 pax010, 4096², source 77db1832)

The in-place 64 GiB bank built (14.6 s) and every full-order arm ran once (untimed `QUICK` lines
in the Slurm stdout; e.g. `fft_tight` 6 cases in 32.5 s, coarse `c2048` 0.2146 % same-grid).
Every ROM arm was then dropped and the job died in `hops.dense_targets`:
`INTERNAL: Autotuning failed for HLO: gemm_fusion_dot = f64[6,16769025]` — XLA's Triton gemm
cannot autotune a product with more than 2^31 elements (16 769 025 × 512 = 8.6e9). The driver's
OOM filter matched the message, so the ROM arms were recorded as dropped instead of raising.
Fix: the bank is a tuple of row blocks, each below the limit (`hops.build_bank`, `bank_apply`).
The untimed FOM lines are NOT results (no repetitions, no audit). Counts as job 3 of 8.
