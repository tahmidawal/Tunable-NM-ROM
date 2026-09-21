# hb4k01 — FAILED, no number produced (job 4055954, H200 pax010, 4096², source c0c007de)

Died 30 s in, after the six `fft_tight` truth solves, at the bank upload:
`RESOURCE_EXHAUSTED: Out of memory while trying to allocate 63.97GiB [executable_name='jit_stage']`.
`hops.build_bank` gathered the 64 GiB bank on the host and uploaded it with `jnp.asarray`, whose
staged host-to-device copy needs a second 64 GiB buffer. Fix: the bank is filled block by block
into one donated device buffer (`dynamic_update_slice`, peak = G + one block) and is built before
the FOM truth phase so the allocation lands in an unfragmented pool. Counts as job 2 of 8.
Kept: the Slurm stdout, the tail of stderr, the partial `result.json` (truth-phase timings only).
