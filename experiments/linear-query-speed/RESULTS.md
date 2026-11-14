# Linear-problem query speed: results

Lane `exp/2026-11-14-linear-query-speed`.  Protocol and dated amendments: `PROTOCOL.md`.  Every number
below is read from the JSON under `runs/`; the full tables (all variants, stage medians, gates) are in
`tables/`.  Comparator numbers (cuDSS, AmgX) are quoted from the sparse-solvers lane for the same GPU type
and were not rerun or altered.

## Poisson 2D: the tuned query is faster than the reused-factorisation cuDSS solve at every mesh, at unchanged accuracy

Timed quantity (headline, column F): nodal source of the whole mesh on the device in float64 -> nodal
solution field of the whole mesh on the device in float64, synchronised; median over the paper's 32-case
test cohort.  This is the like-for-like scope with the cuDSS solve.  The configuration was chosen on
development cases only, frozen, and then run once on the test cohort; the paper path was timed in the same
job on the same GPU (before/after).  Worst test error is identical to six digits before and after.

### NVIDIA H200 (job 7220234)

| mesh | arm | worst err, paper path | worst err, F | paper stored (A100) | paper path, this job | F, this job | cuDSS solve (quoted) | AmgX solve (quoted) | F vs paper path | F vs cuDSS | S (sampled input) | solve only (no field) | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256^2 | accurate | 0.008379 | 0.008379 | 0.565 | 1.335 | **0.175** | 0.448 | 13.900 | 7.6x | 2.6x | - | 0.126 | 7220234 |
| 256^2 | fast | 0.040668 | 0.040668 | 0.381 | 1.014 | **0.171** | 0.448 | 13.900 | 5.9x | 2.6x | - | 0.124 | 7220234 |
| 512^2 | accurate | 0.008356 | 0.008356 | - | 1.541 | **0.183** | 1.000 | 27.200 | 8.4x | 5.5x | 0.177 | 0.157 | 7220234 |
| 512^2 | fast | 0.040594 | 0.040594 | - | 1.315 | **0.175** | 1.000 | 27.200 | 7.5x | 5.7x | 0.175 | 0.125 | 7220234 |
| 1024^2 | accurate | 0.008350 | 0.008350 | 3.103 | 2.213 | **0.199** | 2.960 | 23.900 | 11.1x | 14.9x | 0.186 | 0.164 | 7220234 |
| 1024^2 | fast | 0.040575 | 0.040575 | 1.177 | 1.399 | **0.188** | 2.960 | 23.900 | 7.5x | 15.8x | 0.180 | 0.158 | 7220234 |
| 2048^2 | accurate | 0.008348 | 0.008348 | 11.663 | 5.408 | **0.254** | 9.360 | 48.400 | 21.3x | 36.8x | 0.213 | 0.197 | 7220234 |
| 2048^2 | fast | 0.040571 | 0.040571 | 3.405 | 2.327 | **0.235** | 9.360 | 48.400 | 9.9x | 39.9x | 0.202 | 0.188 | 7220234 |
| 4096^2 | accurate | 0.008348 | 0.008348 | 41.916 | 17.872 | **0.449** | 36.500 | 140.0 | 39.8x | 81.2x | 0.310 | 0.316 | 7220234 |
| 4096^2 | fast | 0.040570 | 0.040570 | 12.110 | 5.593 | **0.408** | 36.500 | 140.0 | 13.7x | 89.5x | 0.295 | 0.297 | 7220234 |

### NVIDIA A100 80GB (jobs 7220235 for 256^2 to 2048^2, 7221131 for 4096^2)

| mesh | arm | worst err, paper path | worst err, F | paper stored (A100) | paper path, this job | F, this job | cuDSS solve (quoted) | AmgX solve (quoted) | F vs paper path | F vs cuDSS | S (sampled input) | solve only (no field) | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256^2 | accurate | 0.008379 | 0.008379 | 0.565 | 1.042 | **0.152** | 0.623 | 10.700 | 6.9x | 4.1x | - | 0.114 | 7220235 |
| 256^2 | fast | 0.040668 | 0.040668 | 0.381 | 0.791 | **0.136** | 0.623 | 10.700 | 5.8x | 4.6x | - | 0.111 | 7220235 |
| 512^2 | accurate | 0.008356 | 0.008356 | - | 1.615 | **0.171** | 1.520 | 19.000 | 9.5x | 8.9x | 0.160 | 0.131 | 7220235 |
| 512^2 | fast | 0.040594 | 0.040594 | - | 1.070 | **0.149** | 1.520 | 19.000 | 7.2x | 10.2x | 0.141 | 0.125 | 7220235 |
| 1024^2 | accurate | 0.008350 | 0.008350 | 3.103 | 3.703 | **0.210** | 4.960 | 34.600 | 17.6x | 23.6x | 0.181 | 0.152 | 7220235 |
| 1024^2 | fast | 0.040575 | 0.040575 | 1.177 | 1.633 | **0.170** | 4.960 | 34.600 | 9.6x | 29.2x | 0.154 | 0.134 | 7220235 |
| 2048^2 | accurate | 0.008348 | 0.008348 | 11.663 | 11.927 | **0.346** | 16.500 | 96.600 | 34.5x | 47.7x | 0.237 | 0.249 | 7220235 |
| 2048^2 | fast | 0.040571 | 0.040571 | 3.405 | 3.673 | **0.295** | 16.500 | 96.600 | 12.5x | 56.0x | 0.203 | 0.221 | 7220235 |
| 4096^2 | accurate | 0.008348 | 0.008348 | 41.916 | 44.348 | **0.778** | 66.800 | 333.0 | 57.0x | 85.9x | 0.429 | 0.541 | 7221131 |
| 4096^2 | fast | 0.040570 | 0.040570 | 12.110 | 12.955 | **0.667** | 66.800 | 333.0 | 19.4x | 100.1x | 0.364 | 0.471 | 7221131 |

Columns: "paper stored (A100)" is the value in the paper's stored results (A100 80GB jobs 4319394 to
4319401, so the paper's Poisson 2D timings came from A100 80GB); "paper path, this job" is the paper's
query function timed in this lane's driver in the same job; S reads the source on a 256-interval
sub-lattice (separate measurement, relies on smooth sources); "solve only" returns reduced coefficients
and no field (separate measurement, not comparable with cuDSS).

Stage medians at 4096^2 on H200, accurate arm (separately compiled and synchronised, so they do not sum
to the fused time): paper path projection + solve 0.996 ms, field reconstruction 17.04 ms; tuned path
projection + solve 0.341 ms, core contraction 0.086 ms, field products 0.247 ms.  All meshes and both
arms: `tables/`.

Gates: stored-error reproduction passed at every mesh with a stored value (largest absolute difference
below 1e-16); the development deviation gate (1e-5) passed for every listed variant; every job log shows
`jax_backend=gpu`.

### Paper-path timing spread at small meshes

The paper path timed in this lane's driver is slower and noisier at small meshes than the stored paper
values, while the tuned variants are tight (H200 256^2 accurate: paper path median 1.335 ms with phase
medians 1.259/1.409 and minimum 0.35; F 0.175).  To check, the unmodified paper driver was run in the
same H200 job:

| mesh | arm | unmodified paper driver, H200 (median ms) | paper path in this lane's driver, H200 | stored paper value, A100 80GB | worst test error (matches stored) |
|---|---|---|---|---|---|
| 256^2 | accurate | 0.734 | 1.335 | 0.565 | 0.008379 |
| 256^2 | fast | 0.482 | 1.014 | 0.381 | 0.040668 |
| 1024^2 | accurate | 1.774 | 2.213 | 3.103 | 0.008350 |
| 1024^2 | fast | 0.961 | 1.399 | 1.177 | 0.040575 |

Both drivers call the same query function; the cause of the extra, variable ~0.5 ms in this lane's driver
was not identified.  The unmodified driver's own timing-stability gate ("neighbour", limit 1.1) failed on
H200 at both meshes (worst ratios 1.185 and 1.105); its error audit passed.  Consequence: at 256^2 to
1024^2 the "F vs paper path" ratios in the tables overstate the gain.  The conservative ratios use the
fastest available paper number as "before": H200 256^2 accurate 0.734 -> 0.175 (4.2x), fast 0.482 ->
0.171 (2.8x); H200 1024^2 accurate 1.774 -> 0.199 (8.9x), fast 0.961 -> 0.188 (5.1x); A100 256^2
accurate 0.565 -> 0.152 (3.7x), fast 0.381 -> 0.136 (2.8x).  At 2048^2 and 4096^2 the in-job paper path
agrees with the stored A100 values within 2 to 7 %.  The comparison with cuDSS does not involve the
paper path and is unaffected.

## What changed, in plain words

The trained model, the reduced basis, the reduced solve and the answer are the same as in the paper.  Only
the way the online query is evaluated changed.

1. **The bank is stored in separated form.**  The paper path keeps the mesh-evaluated bank as a dense
   array with one column per reduced coordinate and one row per mesh node, so writing the output field is a
   product whose cost grows with (mesh nodes) x (reduced size).  That product is 95 % of the paper query at
   4096^2.  The bank is smooth along each mesh direction, so offline it is compressed once into a factor for
   the x direction, a factor for the y direction and a small core (ranks 65 x 66 for the accurate arm,
   50 x 51 for the fast arm, relative truncation 1e-4 in float64).  Online the field is two small
   matrix products, `U (C a) V^T`.
2. **The source is projected through the same factors**, so reading the full nodal source costs two small
   products instead of a product with the dense bank.
3. **The reduced operator is assembled and factorised offline** and the whole query (project, reduced
   solve, core contraction, field products) is one compiled function with one device synchronisation.
4. Heat 2D: the same separated bank; in addition the initial-condition moments are taken with a small sine
   table instead of a full sine transform of the mesh, and the moment fit, the composed Crank-Nicolson
   maps for the six output times and the core are folded offline into one small matrix, so the six output
   fields come from two products.

Offline cost added: one randomised compression of the bank per mesh (6 to 15 s on the GPU for Poisson 2D;
see the tables), no retraining.

## What this supports and what it does not

Supports:

- On Poisson 2D the tuned query returns the same field as the paper query (largest deviation 1.4e-6 of the
  field norm on the accurate arm, 8.8e-6 on the fast arm; worst test error identical to six digits) and is
  faster than the quoted cuDSS reused-factorisation solve at every mesh from 256^2 to 4096^2, on both GPU
  types, with full-field output and the full nodal source as input.
- The query time is now nearly flat in the mesh size on H200 (0.18 ms at 256^2 to 0.45 ms at 4096^2 on the
  accurate arm), because the only mesh-sized work left is reading the source and writing the field.

Does not support:

- cuDSS and AmgX solve an arbitrary right-hand side to solver tolerance.  The reduced model is accurate
  only to its model error (0.84 % worst case accurate arm, 4.06 % fast arm) and only on the trained source
  family.  The comparison is time-to-answer at the reduced model's accuracy, not equivalence.
- The separated evaluation relies on the tensor-product mesh and on the bank being smooth along each
  direction.  The same structure is what makes sine-transform solvers fast on this problem; those were not
  targeted and no claim is made against them.
- Configuration S (source read on a 256-interval sub-lattice) relies on the smoothness of the trained
  source family.  It is a separate column and never the headline.
- "Solve only" (reduced coefficients, no field) is a separate column and never compared with cuDSS.
- The fast-arm deviation (8.8e-6) is close to the 1e-5 development gate; it passed, with little margin.
- 512^2 has no stored paper timing; its "before" is only the in-job paper path.
- At 256^2 to 1024^2 the paper path timed inside this lane's driver is slower and noisier than the stored
  paper values (see "Paper-path timing spread"), so the conservative speed-up at small meshes uses the
  stored paper value as "before".  The cuDSS comparison does not depend on this.
- Heat 2D comparators are the sparse-solvers lane's arithmetic estimates for 20 Crank-Nicolson solves, not
  measured trajectories.
- Poisson 3D and heat 3D were not attempted (three-direction compression and new paper-path drivers were
  out of the time budget; both already beat cuDSS by large margins in the paper).

## Jobs and GPU-hours

| job | GPU | purpose | wall | GPU-h | state |
|---|---|---|---|---|---|
| 7220222 | H200 (requested) | selection only (superseded) | 0 | 0 | cancelled while pending, never started |
| 7220223 | H200 (requested) | paper driver only (superseded) | 0 | 0 | cancelled while pending, never started |
| 7220234 | H200 (pax009) | dev selection 256^2, 4096^2 -> freeze -> test 256^2..4096^2; then the unmodified paper driver at 256^2 and 1024^2 (its own timing-stability 'neighbour' gate failed at both, ratios 1.185 and 1.105 against a limit of 1.1; errors match stored) | 01:36:44 | 1.61 | completed |
| 7220235 | A100 80GB (pax050) | dev selection 256^2, 2048^2 -> freeze -> test 256^2..2048^2; 4096^2 test failed (compression out of device memory) | 01:06:10 | 1.10 | completed |
| 7221131 | A100 80GB (pax106) | test 4096^2 only, frozen pair of 7220235, `--host-compress` (amendment A2) | 00:24:15 | 0.40 | completed |
| 7221749 | H200 (requested) | heat 2D with the wrong fast arm (see A3a) | 0 | 0 | cancelled while pending, never started |
| 7221755 | H200 (requested) | heat 2D: selection at 4096^2 on the development cohort, freeze, test 4096^2/2048^2/1024^2 (A3, A3a) | pending | | |
