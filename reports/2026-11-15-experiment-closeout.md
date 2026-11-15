# Closing the Burgers quadrature, sparse-solver and neural-operator experiments

The sparse-solver and held-out operator measurements are final. The separate sealed Burgers quadrature confirmation is pending.

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

- **Mesh** gives the spatial resolution in one direction. A square denotes a two-dimensional grid.
- **FNO and U-Net** are neural operators. An FNO uses Fourier transforms to mix spatial features. A U-Net uses an encoder and a decoder with connections between corresponding resolutions.
- **Checkpoint** means saved trained model weights. The validation cases selected one checkpoint per family before the held-out test cases were scored.
- **ROM** means reduced-order model. Accurate and fast denote the previously selected solver configurations.
- **AmgX** is the NVIDIA sparse linear-solver library used inside the implicit Burgers solver. The trajectory time includes all time steps.
- **Median and worst error** summarize the middle and largest case error. The error of an individual case is the maximum relative spatial error over its evolved output times. The threshold is an existing accuracy target. A worst-error threshold applies to the largest case error in that panel.
- **Median query time** measures elapsed time for one solve. A paired time uses the call that produced the scored solution. The case count gives the number of physical inputs in each comparison.
- **Float32, float64 and TF32** describe floating-point precision. Disabling TF32 keeps the operator's float32 matrix products from using reduced-precision TF32 inputs.
