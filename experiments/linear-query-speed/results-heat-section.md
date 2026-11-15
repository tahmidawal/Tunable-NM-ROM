## Heat 2D: six output fields in 0.26 to 1.47 ms on A100 80GB at unchanged accuracy (2.2x to 14.2x faster than the paper path in the same job)

Timed quantity (column F): initial field of the whole mesh on the device -> six output fields of the whole
mesh on the device, blocked; median over the paper's 16-case sealed test cohort x 5 repetitions (the
paper's heat scope).  Arms are the paper's: accurate `lin_R128_cn`, fast `lin_R32_cn`.  The configuration
was chosen on the development cohort only at 2048^2 (job 7228875), frozen (`T_tau1e-06_f64`, file SHA256
1e4220ef...), and the same frozen file was used for 4096^2 (job 7228913).  Worst test error is identical
to six digits before and after (0.1332 % and 1.0823 %); the largest field deviation from the paper path is
3.0e-8 (accurate) and 2.1e-7 (fast).  The coarser truncation 1e-4 that Poisson 2D uses failed the heat
development gate (deviation 2.1e-5 > 1e-5) and was not used.

| mesh | arm | worst err, paper path | worst err, F | paper stored | paper path, this job (A1/A2) | F, this job | F vs paper path | cuDSS 20 solves, factor reused (est.) | AmgX setup + 20 solves (est.) | cuDSS est. / F | coefficients only (no field) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024^2 | accurate (`lin_R128_cn`) | 0.001332 | 0.001332 | 0.636 | 1.263 (1.263/1.262) | **0.255** | 4.9x | 99 | 486 | 389x | 0.164 |
| 1024^2 | fast (`lin_R32_cn`) | 0.010823 | 0.010823 | 0.336 | 0.525 (0.524/0.525) | **0.238** | 2.2x | 99 | 486 | 417x | 0.164 |
| 2048^2 | accurate (`lin_R128_cn`) | 0.001332 | 0.001332 | 1.964 | 4.635 (4.633/4.636) | **0.528** | 8.8x | 334 | 1455 | 633x | 0.241 |
| 2048^2 | fast (`lin_R32_cn`) | 0.010823 | 0.010823 | 0.829 | 1.759 (1.758/1.759) | **0.490** | 3.6x | 334 | 1455 | 682x | 0.242 |
| 4096^2 | accurate (`lin_R128_cn`) | 0.001332 | 0.001332 | 8.421 | 20.789 (20.763/20.796) | **1.468** | 14.2x | 1323 | 5370 | 901x | 0.461 |
| 4096^2 | fast (`lin_R32_cn`) | 0.010823 | 0.010823 | 3.966 | 8.159 (8.141/8.162) | **1.301** | 6.3x | 1323 | 5370 | 1017x | 0.463 |

"paper stored" is the paper's stored value from H200 (job 4197350), a different and faster GPU type than
this measurement, so the paper's heat 2D timings came from H200; the before/after pair is "paper path,
this job" and F, both on the same A100 80GB in the same job.  Against the stored H200 values the tuned
A100 query is still faster (accurate 0.636 -> 0.255, 1.964 -> 0.528, 8.421 -> 1.468 ms; fast 0.336 ->
0.238, 0.829 -> 0.490, 3.966 -> 1.301 ms).  The cuDSS and AmgX columns are the sparse-solvers lane's
A100 arithmetic estimates for 20 Crank-Nicolson solves, not measured trajectories.  "Coefficients only"
is a separate measurement with no field output.

Stage medians at 4096^2, accurate arm (first test case, separately compiled): paper path moments by full
sine transform 2.69 ms, evolve 0.07 ms, six-field bank product 18.09 ms; tuned path moments + coefficients
0.43 ms, core contraction 0.07 ms, field products 1.06 ms.  Gates: stored-error reproduction passed at all
three meshes (largest absolute difference 8.2e-17), development deviation gate passed (2.7e-7), moment
identity 5e-16, `jax_backend=gpu` in both logs.  Offline compression 6 to 8 s per mesh; peak device memory
51.5 GB at 4096^2.  Full tables: `tables/heat2d-a100-80G.md`, `tables/heat2d-a100-80G-4096.md`.

The H200 heat 2D job (7221755, amendment A3) was still pending on queue priority when this was written (submitted 17:48 EST); H200 heat 2D is therefore not measured.
