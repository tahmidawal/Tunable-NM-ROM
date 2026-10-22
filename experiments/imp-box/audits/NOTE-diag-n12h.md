# Note: H200 variant of the N=12 diagnostic (2026-10-22)

`diag_jobs/n12h.sbatch` is `diag_jobs/n12.sbatch` (audited SUBMIT in `codex-job-diag-n12-3.md`) with only the GPU type
changed from a100 to h200 and the job name. Reason: the A100 run 6188855 (JAX 0.10.2) showed no N=12 failure with the
cache off, populated and replayed (35 entries, replay wrote nothing), while both H200 smokes failed at N=12; the failure is
therefore probed on the GPU type where it occurs. Budget ≤ 0.5 GPU-h. No result of these jobs is used scientifically.

## H200 result (job 6188970) and the loop-variant probe
6188970 reproduced the failure on H200 with the cache off, populated and replayed (so the cache is not the cause): the fused
fixed-step loop at N = 12 is non-finite from step 1 on every val case, while eager single steps are finite and the
nonlinear term equals an independent dense evaluation to 2.5e-15 — a Hopper code-generation fault in the fused loop.
`diag_jobs/n12v.sbatch` (same job template; ≤ 0.5 GPU-h; no scientific use) compares loop-structure variants (production
dynamic-bound fori_loop, static-bound fori_loop, scan over steps, unrolled blocks) against an eager reference at
N ∈ {10, 12, 14, 16, 24, 48}, so the production query can be restructured to a variant that compiles correctly.
