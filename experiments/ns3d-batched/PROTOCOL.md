# PROTOCOL — ns3d-batched (written 2026-11-13, 01:00 EST, before any batched timing exists)

Question. All timings of the "harder" 3D Navier-Stokes comparison (lane ns3d-hard-rank,
fresh cohort, replicate 0, `results/fresh_s0b/compare.json`, job 7145062) are batch-1: one
query per call. Does the ROM's speed advantage over the accuracy-matched GPU pseudo-spectral
solver survive when BOTH sides process many independent queries per call (batched
throughput, same GPU, same job)? The result is reported whichever way it goes.

Nothing in the source lane is changed or re-selected. Its frozen arms, banks, heads, cohort
and matched solver configurations are read (read-only) from its committed files and its
cluster directories.

## 1. Arms

ROM (frozen seed-0 arms of `frozen_rank/frozen_{n}.json`, unchanged settings):
`r96_head`, `r128_head_fast`, `r192_head_fast` at meshes n = 64, 128, and 96.

Full-order solver: for each arm and mesh, the configuration recorded as that arm's
`any_mesh_solve` comparator in `fresh_s0b/compare.json` (fastest configuration, among the
shortlist timed in that job, of the 450-configuration ladder whose worst-case error over the
same 128 fresh cases is <= the arm's worst-case error). Distinct configurations:
64: cnab2 100^3 x100, cnab2 108^3 x100; 96: cnab2 108^3 x100, cnab2 112^3 x120,
cnab2 120^3 x120; 128: cnab2 108^3 x120, rk4 108^3 x30, rk4 112^3 x30.

## 2. How each side is batched (same effort on both sides)

- `rom_vmap`: `jax.jit(jax.vmap(run, in_axes=(0, 0, None, ...)))` of the UNCHANGED frozen
  query function (`head_rom.make_head_run`), batched over (initial field, viscosity); bank,
  operators, head and candidate codes are shared (unbatched).
- `rom_shared`: the same query with one algebraically equivalent rearrangement so the
  quadratic operator is shared across the batch: the viscosity-dependent preconditioner
  division `(T + T^T) / pre` is applied after the contraction instead of before it, so the
  per-sweep contraction is one matrix product over the batch instead of B matrix-vector
  products against B private copies of the tensor. Everything else is the unchanged code.
- `fom_vmap`: `jax.jit(jax.vmap(run, in_axes=(0, 0, None)))` of the UNCHANGED solver
  (`ns3d_fom.make_solver`, complex FFTs), batched over (initial field, viscosity); the
  geometry is shared. The FFTs become batched FFTs.
- `fom_rfft`: the same CNAB2/RK4 scheme, same 2/3 mask, same Leray projection, written with
  real-to-complex FFTs (`rfftn`/`irfftn`; about half the FFT work and memory), vmapped the
  same way. This is the solver-side counterpart of `rom_shared`: a code improvement the
  original solver did not have, given to it so the solver is not handicapped.
- Both variants on each side must reproduce the unchanged batch-1 code (Section 4) or they
  are reported as failed and excluded from the headline.

Headline per mesh, arm and B: fastest ROM variant vs fastest solver variant at that B.
All variants are reported.

## 3. What is timed

- ROM end-to-end (PRIMARY ROM time): initial field on mesh n -> centroid, centring shift,
  bank projection, initial-code fit -> latent rollout -> reconstruction of all six output
  fields (3 x n^3 each) on mesh n. Same scope as the batch-1 ROM time in fresh_s0b.
- ROM solve-only (`rom_shared` only): latent rollout from the encoded initial state to the
  latent states and frame centres at the six output times; encode and decode are also timed
  separately.
- Solver solve-only (PRIMARY solver time, as in fresh_s0b): initial field already on the
  solver mesh m -> six output fields on mesh m.
- Solver end-to-end: storage-mesh (128^3) initial field -> spectral truncation to m ->
  solve -> spectral transfer of the six fields to mesh n.
- Primary ratio (as fresh_s0b): solver solve-only throughput-time / ROM end-to-end time,
  i.e. ratio = t_solver(B) / t_ROM(B) at equal B; > 1 means the ROM has the higher throughput.
  Also reported: e2e/e2e and solve/solve.
- Batch sizes B in {1, 8, 32, 128} for every callable; ROM additionally B = 512 where the
  compiled program fits in memory (informational; never chunked). If a B does not fit in
  device memory (by the compiler's buffer assignment), it is executed as consecutive calls
  of the largest fitting power-of-two chunk, and marked `chunked` in the table.
- Inputs: the 128 fresh-cohort initial conditions (seed 2026111151, genF_0 + genF_1), case
  i of a batch = cohort case (i mod 128). Both codes run a fixed number of iterations
  (fixed steps, fixed Gauss-Newton sweeps), so time does not depend on the content.
  Inputs are on the device before timing; outputs stay on the device; host transfers are
  excluded on both sides.
- Timing: one job per mesh, A100 80 GB, f64, matmul precision highest, `jax_backend=gpu`
  asserted. Every callable is compiled ahead of time (excluded) and called once untimed.
  Per B, one randomised interleaved block holding every ROM and solver callable at that B,
  preceded by 2 s of GPU burn-in; 5 timed repetitions for B <= 32, 3 for B >= 128; median
  reported with min/max; every repetition array is stored.
- Derived: samples/s = B / t(B); time per 1,000 queries = 1000 * t(B) / B.
- Peak memory: the compiler's buffer assignment of each compiled callable (arguments +
  outputs + temporaries - aliased), and the process peak `peak_bytes_in_use` after each mesh.

## 4. Equal accuracy and output checks

Equal accuracy is inherited from fresh_s0b (the solver configuration's worst-case error over
the 128 fresh cases is <= the ROM arm's). Batching must not change either side:
- Field agreement: outputs at B = 8 and the largest B, batch slots 0 and B-1 (and two more),
  against the unchanged batch-1 code on the same case: max |difference| / max |reference|
  <= 1e-9 (ROM) and <= 1e-10 (solver).
- Cohort agreement: all 128 fresh cases through each batched callable (B = 32 chunks);
  per-case evolved-worst errors against the truth on mesh n must reproduce fresh_s0b
  (ROM arm errors) and the merged ladder fomF_0 + fomF_64 (solver per-case worst, via the
  end-to-end transfer to n) to 1e-8 (ROM) and 1e-7 (solver), the source lane's tolerances.

## 5. Re-match sensitivity (B = 32 only)

The comparator was chosen on batch-1 time; batching may favour a different configuration.
At B = 32, every configuration in the arm's fresh_s0b shortlist (all have worst error <= the
arm's) is timed (solve-only, both solver variants, 3 repetitions), and the arm's ratio is
also reported against the fastest of them at B = 32. The headline stays the fresh_s0b
configuration unless the re-matched one is faster, in which case the re-matched ratio is
the headline (it only lowers the ROM's ratio).

## 6. Reported whatever happens

Per mesh, arm and B: ROM and solver times (both variants), samples/s, time per 1,000
queries, peak memory, ratios, chunking, agreement checks, failures, job ids, GPU model.
