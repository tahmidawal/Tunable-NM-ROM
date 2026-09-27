# Two-PDE k sweep with m = 4M

This experiment measures how accuracy and online cost change with latent dimension for the
saved coordinate NM-ROM architecture on Poisson-2D and Burgers-2D. Results are provisional
until the cluster job finishes, is pulled with checksums, and passes the result audit.

## Frozen scientific contract

- PDEs: Poisson-2D and Burgers-2D.
- Mesh: N = 64 for both PDEs.
- Latent dimensions: k = 4, 6, 8, 12, 16, 24, 32, 48, 64.
- Weak test dimension: M = 4k.
- NNLS-EQ quadrature count: m = 4M = 16k.
- EQ weights are refit for every PDE and every `(k, M, m)` from decoder-output snapshots.
- Poisson uses weak alpha-1 modes and a one-training-cloud-radius trust region.
- Burgers keeps the FOM-exact upwind advection in the weak term and uses the accepted
  trust factor 0.01 times the flattened training-trajectory latent radius.
- Primary stopping rule: relative weak-objective reduction tau = 1e-3, matching the
  earlier inconsistent k study. A secondary tau = 1e-2 deployment point shares the same EQ fit.
- Frozen held-out panels: 16 cases per PDE from each existing driver's recorded test draw.
- Existing seed-0 checkpoints are reused for k through 32. The missing k = 48 and 64
  checkpoints are trained with the same recorded architecture, seed, data generator, and
  training recipe before evaluation.

## Timing contract

Both PDEs and every k run sequentially in one Slurm job on one physical GPU. Every timed block
gets a GPU burn, three warm-ups, then nine synchronized repetitions. Raw repetition arrays are
persisted.

- `time_ms_decode`: isolated final full-field decode. Poisson decodes one field; Burgers decodes
  all 51 trajectory slices.
- `time_ms_solve`: isolated trust-region latent solve. Poisson includes only the weak EQ LM;
  Burgers includes only the 50-step weak EQ rollout.
- `time_ms_cold_start`: Burgers-only hyper-reduced initial-condition fit.
- `time_ms`: the whole deployable path whose returned state is graded for relative L2 error.
  Poisson includes source preprocessing, solve, and decode. Burgers includes nearest-table
  lookup, cold start, solve, and trajectory decode.

Component timings are isolated measurements and are not expected to sum exactly to the fused
whole-path timing. The whole-path cost and relative L2 error come from the same final timed
solver invocation.

## Acceptance and reporting

The run must record GPU backend, f64, highest matmul precision, seed, checkpoint and source
hashes, node, GPU model, Slurm job, exact `(k, M, m)` values, EQ global and worst-row fits,
per-case relative L2 values, solver reasons, censoring/blow-ups, and every timing repetition.
Plots and any speed claim will use medians plus outlier counts; means alone are not sufficient.

The cluster namespace is `/cluster/tufts/paralab/tawal01/k64-eq4m/`. The one job directory is
`ctol_k64_eq4m_r1`, and its Slurm name begins with `ctol_` so the repository's guarded numeric
cancel script can be used if cancellation is ever required.
