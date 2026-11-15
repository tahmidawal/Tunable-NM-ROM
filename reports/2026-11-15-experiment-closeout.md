# Closing the Burgers quadrature, sparse-solver and neural-operator experiments

All three experiments are final. The tables report the frozen Burgers quadrature decisions and their classical controls, the held-out operator errors, and the recovered sparse-solver measurements.

## Sealed Burgers quadrature confirmation

The freeze committed the configurations for 128 two-dimensional and 64 three-dimensional in-family cases before a single authorized opening. The Git identifier of the accepted freeze is `6d5cef5ebf19`. All runs used float64 and the highest matrix-product precision.

The primary reference refines both space and time (ST). The secondary reference refines space (S). A pass requires the selected rule and every larger tested rule in its family to keep both median and worst trajectory error within the registered uncertainty-adjusted margin of the dense reference rule on every mesh. The registered margin is five percent. A failure on one qualified mesh falsifies sufficiency.

| Dimension | Frozen quadrature | Points | Primary ST across meshes | Secondary S across meshes | Eligible ST median error across meshes | Eligible ST worst error across meshes |
|---|---|---:|---|---|---:|---:|
| 2D | Fibonacci lattice | 987 | PASS (5/5 pass) | FAIL (0/5 pass) | 0.9660–0.9683% | 3.4118–3.4152% |
| 2D | Tensor Gauss | 2,025 | PASS (5/5 pass) | FAIL (2/5 pass) | 0.9507–0.9550% | 3.3946–3.3976% |
| 3D | CBC lattice | 4,096 | PASS (3/3 pass) | PASS (3/3 pass) | 1.6044–1.6068% | 5.4221–5.4546% |
| 3D | Tensor Gauss | 8,000 | PASS (3/3 pass) | PASS (3/3 pass) | 1.6053–1.6064% | 5.5595–5.5842% |

| Dimension | Physical cases | Eligible ST / S | Meshes passing both lattice timing checks |
|---|---:|---:|---:|
| 2D | 128 | 123 / 107 | 2 / 5 |
| 3D | 64 | 64 / 64 | 3 / 3 |

Native POD controls used their previously selected native meshes. Median and worst errors below use the eligible ST cases with finite fields. The valid cases column counts numerical terminations over the entire physical cohort.

| Control | Valid cases | Eligible ST median error | Eligible ST worst error |
|---|---:|---:|---:|
| 2D POD | 128 / 128 | 0.2019% | 4.5341% |
| 3D POD | 64 / 64 | 0.2078% | 3.5696% |

On the native 2D mesh, POD had lower ST error than each selected quadrature rule on 119 cases, and each rule had lower error on 4.

On the native 3D mesh, POD had lower ST error than each selected quadrature rule on 64 cases, and each rule had lower error on 0.

The paired comparisons require differences to exceed twice the case's reference uncertainty.

The full-order pseudospectral solver with $N=32$ appears in both native panels. Each table entry uses the previously declared solver setting. The numerical convergence count covers all physical cases.

| Native panel | Converged full-order solves | Eligible ST median error | Eligible ST worst error |
|---|---:|---:|---:|
| 2D | 128 / 128 | 0.0021% | 1.5914% |
| 3D | 64 / 64 | 0.0024% | 0.2114% |

### Comparing the lattice query with an accuracy-matched full-order solver

The spectral solver had to converge on every case and match or improve both the median and worst eligible ST error within reference uncertainty. The initial timing block selected the fastest qualifying full-order configuration. The table uses a separate confirmation block on the same GPU, timing six prepared inputs with five repetitions each. Both solvers returned fields on the same scoring lattice. The ratio divides the lattice median by the full-order median.

| Mesh | Full-order solver | Lattice median | Full-order median | Time ratio |
|---|---|---:|---:|---:|
| 512² | PS_N48 | 41.98 ms | 6.41 ms | 6.55× |
| 2048² | PS_N48 | 41.84 ms | 6.41 ms | 6.52× |
| 64³ | PS_N16 | 39.52 ms | 8.78 ms | 4.50× |
| 128³ | PS_N20 | 46.53 ms | 8.77 ms | 5.31× |
| 256³ | PS_N20 | 42.19 ms | 5.55 ms | 7.60× |

These are prepared-input query times. The full-order solver's input preparation was measured separately. The ROM's corresponding preparation time was unmeasured, so the table supports no end-to-end timing ratio. The other 2D meshes failed the timing stability criterion.

Independent scoring reproduced 15,472 saved case–reference comparisons across the mesh panels. The maximum relative discrepancy was 7.12e-15. Checks used saved model and full-solver fields on selected 2D cases and decoded all 3D model coefficients. The historical 3D protocol saved no full-solver fields for independent reconstruction. POD and quadrature ran in separate allocations. Custody used one shared research account. The out-of-family populations remain sealed.

[Committed freeze](https://github.com/tahmidawal/Tunable-NM-ROM/blob/138cc10a8f50caeca3791f0353a3ba7b60bb8b8e/experiments/jcp-sealed-closeout/F1.json) · [Mesh-by-mesh decisions and controls](https://github.com/tahmidawal/Tunable-NM-ROM/tree/0751cea90eca0801afbce26b5073d98fd2cf3f52/experiments/jcp-sealed-closeout/results)

## Held-out neural operators

Job `7243849` evaluated 21 frozen checkpoints on 64 Burgers cases. 0 met the 0.0264% test-median target. Networks ran in the registered float32 precision with TF32 disabled. Errors were scored in float64.

| Validation-selected operator | Median error | Worst error | Median query time |
|---|---:|---:|---:|
| FNO | 0.06538% | 0.17531% | 11.030 ms |
| U-Net | 0.17668% | 0.37483% | 10.031 ms |

Errors are per-case maxima over evolved output times. Query time is the median across cases and repetitions on one A100 80 GB. The scored output came from the final timed invocation for each case. Reporting includes surviving best checkpoints from interrupted training. Training budgets differ between the selected models. The historical ROM ran in a different allocation.

[All checkpoints and training budgets](https://github.com/tahmidawal/Tunable-NM-ROM/blob/eb699ee396b23fdf3ba44f101221f14bfe033c83/experiments/operator-scaling/reports/tables.md) · [Numerical audit](https://github.com/tahmidawal/Tunable-NM-ROM/blob/eb699ee396b23fdf3ba44f101221f14bfe033c83/experiments/operator-scaling/closeout/audit.json)

## Sparse GPU solver and ROM measurements

AmgX cases are timed full Burgers trajectories on one H200. Each row uses the fastest converged, finite setting whose observed worst error passes the historical accurate ROM threshold. Selection uses all repetitions. The paired time below uses the first timed repetition of each case, the invocation that supplied the scored field. The thresholds come from a larger cohort.

| Mesh | Cases | Worst-error threshold | Median error | Worst error | Paired median time |
|---|---:|---:|---:|---:|---:|
| 256² | 16 | 0.8749% | 0.3035% | 0.6572% | 76.08 ms |
| 512² | 16 | 1.1284% | 0.4564% | 0.8148% | 113.50 ms |
| 1024² | 16 | 1.2499% | 0.4947% | 0.9220% | 185.05 ms |

The ROM panels ran in separate allocations. Each admitted timing below passed its lane’s timing controls and reproduced the paper’s errors. The smaller Poisson panel failed the timing-order check.

| Problem | Mesh | Cases | Accurate / fast median time | Accurate / fast worst error |
|---|---|---:|---:|---:|
| Poisson | 1024² | 32 | 1.673 / 0.861 ms | 0.8350% / 4.0575% |
| Poisson | 2048² | 32 | 4.846 / 1.778 ms | 0.8348% / 4.0571% |
| Burgers | 2048² | 64 | 48.725 / 19.568 ms | 1.3162% / 3.0591% |

[Sparse collection and archive verification](https://github.com/tahmidawal/Tunable-NM-ROM/tree/4b3a989e9b2926a7e5ff341a5cdb06f3b3f06465/experiments/sparse-solvers/ledger/closeout-2026-11-15) · [Complete AmgX settings](https://github.com/tahmidawal/Tunable-NM-ROM/blob/4b3a989e9b2926a7e5ff341a5cdb06f3b3f06465/experiments/sparse-solvers/RESULTS-burgers-stepper.md)

## Glossary

- **Quadrature** approximates a spatial integral using weighted points. Fibonacci and component-by-component (CBC) lattice rules distribute points using structured sequences. Tensor Gauss combines one-dimensional Gauss rules across coordinates. The dense rule uses many points on the same reduced model.
- **Burgers, POD and full-order solvers.** Burgers combines nonlinear transport and diffusion. Proper orthogonal decomposition (POD) builds reduced coordinates from snapshots. A full-order pseudospectral solver approximates the PDE in spectral modes without a trained reduced basis. $N$ sets its spatial resolution.
- **ST and S** denote the space-and-time and space-only reference procedures. Each case and reference kind has an original numerical quality check and uncertainty estimate. Only individually eligible cases enter formal accuracy rankings. A mesh passes the lattice timing checks when timed outputs agree with the scored outputs and the registered timing stability tests pass.
- **Mesh** gives the spatial resolution in one direction. A square denotes a two-dimensional grid.
- **FNO and U-Net** are neural operators. An FNO uses Fourier transforms to mix spatial features. A U-Net uses an encoder and a decoder with connections between corresponding resolutions.
- **Checkpoint** means saved trained model weights. The validation cases selected one checkpoint per family before the held-out test cases were scored.
- **ROM** means reduced-order model. Accurate and fast denote the previously selected solver configurations.
- **AmgX** is the NVIDIA sparse linear-solver library used inside the implicit Burgers solver. The trajectory time includes all time steps.
- **Median and worst error** summarize the middle and largest case error. For a Burgers trajectory, the case error takes the maximum over the evolved output times. A stationary Poisson solve yields one spatial error. The threshold is an existing accuracy target. A worst-error threshold applies to the largest case error in that panel.
- **Median query time** measures elapsed time for one solve. A paired time uses the call that produced the scored solution. The case count gives the number of physical inputs in each comparison.
- **Float32, float64 and TF32** describe floating-point precision. Disabling TF32 keeps the operator's float32 matrix products from using reduced-precision TF32 inputs.
