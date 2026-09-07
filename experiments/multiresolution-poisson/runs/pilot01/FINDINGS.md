# Frozen Poisson mesh-transfer development findings

These generated results are provisional development evidence from one checkpoint and one source cohort. They compare complete queries against direct transform solvers; no final-cohort claim is made.

Source `82d2c3261126cb150bb83220ec3edbcf0dd5f489`, job `3350079`, GPU `NVIDIA A100 80GB PCIe`. The checkpoint was trained on 256 nodes per axis; the new meshes use intervals. The fixed development seed is 7090703 with 6 sources and 4 timing repetitions.

## Accuracy and complete query time

| Intervals | Arm | Tau | Median ms | Median physical error | Worst physical error | Worst same-grid error | Invalid sources | Nonstationary sources | Timing outliers |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | dst | — | 2.0337 | 5.59264e-05 | 0.0001546 | 0 | 0 | 0 | 0 |
| 256 | dst_coarse128 | — | 2.5583 | 0.000207491 | 0.000552816 | 0.000567057 | 0 | 0 | 0 |
| 256 | rom | 0.1 | 4.1148 | 0.0627 | 0.243429 | 0.24346 | 0 | 6 | 0 |
| 256 | rom | 0.01 | 4.6996 | 0.00892707 | 0.0736259 | 0.0737238 | 0 | 5 | 0 |
| 256 | rom | 0.001 | 5.7511 | 0.00764969 | 0.0731865 | 0.0732828 | 0 | 0 | 0 |
| 256 | rom | 0.0 | 6.0676 | 0.00764969 | 0.0731865 | 0.0732828 | 0 | 0 | 0 |
| 512 | dst | — | 2.3913 | 1.33138e-05 | 3.67666e-05 | 0 | 0 | 0 | 0 |
| 512 | dst_coarse128 | — | 2.8021 | 0.000207491 | 0.000552816 | 0.000466725 | 0 | 0 | 0 |
| 512 | rom | 0.1 | 4.3314 | 0.0627006 | 0.243386 | 0.243393 | 0 | 6 | 0 |
| 512 | rom | 0.01 | 4.9578 | 0.00892788 | 0.0736296 | 0.0736528 | 0 | 5 | 0 |
| 512 | rom | 0.001 | 5.8078 | 0.00764839 | 0.0731891 | 0.073212 | 0 | 0 | 0 |
| 512 | rom | 0.0 | 5.9342 | 0.00764839 | 0.0731891 | 0.073212 | 0 | 0 | 0 |

### Complete-query components

| Intervals | Arm | Tau | Input ms | Projection/init ms | Solver ms | Output ms |
|---:|---|---:|---:|---:|---:|---:|
| 256 | dst | — | 0.5621 | 0.0000 | 0.1700 | 1.3386 |
| 256 | dst_coarse128 | — | 0.5883 | 0.0000 | 1.5702 | 0.3833 |
| 256 | rom | 0.1 | 0.6067 | 1.0153 | 1.9995 | 0.6439 |
| 256 | rom | 0.01 | 0.5790 | 1.0559 | 2.4746 | 0.6442 |
| 256 | rom | 0.001 | 0.5771 | 1.0454 | 3.4832 | 0.6471 |
| 256 | rom | 0.0 | 0.5973 | 1.0012 | 3.7062 | 0.6991 |
| 512 | dst | — | 0.8452 | 0.0000 | 0.1959 | 1.3438 |
| 512 | dst_coarse128 | — | 0.8564 | 0.0000 | 1.2495 | 0.6468 |
| 512 | rom | 0.1 | 0.8727 | 0.7184 | 1.8981 | 0.8360 |
| 512 | rom | 0.01 | 0.8624 | 0.7292 | 2.3386 | 0.8830 |
| 512 | rom | 0.001 | 0.8349 | 0.7140 | 3.3966 | 0.8556 |
| 512 | rom | 0.0 | 0.8480 | 0.6664 | 3.4964 | 0.8315 |

Component medians do not necessarily sum to the median total. Raw synchronized timestamps are retained for each invocation.

Physical errors use the same nested observation nodes and independently refined reference. Qualification below adds empirical reference uncertainty; every source must meet the target. Intentional tau stops may pass physical qualification before stationarity. Other terminal ROM solves require measured stationarity. The complete query includes synchronized input, source projection, solve and full output.

## Cheapest qualifying development configurations

| Requested intervals | Target | Selected ROM tau | Selected FOM | ROM ms | FOM ms | Median paired FOM/ROM ratio |
|---:|---:|---:|---|---:|---:|---:|
| 256 | 0.1 | 0.01 | dst | 4.69963 | 2.03371 | 0.421264 |
| 256 | 0.05 | unattained | dst | unattained | 2.03371 | unattained |
| 256 | 0.01 | unattained | dst | unattained | 2.03371 | unattained |
| 256 | 0.001 | unattained | dst | unattained | 2.03371 | unattained |
| 512 | 0.1 | 0.01 | dst | 4.95778 | 2.3913 | 0.499445 |
| 512 | 0.05 | unattained | dst | unattained | 2.3913 | unattained |
| 512 | 0.01 | unattained | dst | unattained | 2.3913 | unattained |
| 512 | 0.001 | unattained | dst | unattained | 2.3913 | unattained |

The displayed ratio first takes the median FOM/ROM time ratio over paired repetitions within each source, then the median over sources. A ratio above unity favors the reduced model. The cost columns are separate aggregate medians, so their quotient need not equal this paired statistic. Unattained targets have no qualifying ratio. The classical envelope searches only the declared same-grid and coarse-grid options. Selection and evaluation use this development cohort; independent confirmation is still required.

## Reference and mesh setup

| Source | Refinement gap, first pair | Refinement gap, final pair | Ratio | Empirical uncertainty |
|---:|---:|---:|---:|---:|
| 0 | 1.58502e-05 | 3.96215e-06 | 4.00042 | 3.96216e-06 |
| 1 | 6.66508e-06 | 1.66624e-06 | 4.00008 | 1.66624e-06 |
| 2 | 9.7802e-06 | 2.44498e-06 | 4.00012 | 2.44498e-06 |
| 3 | 7.04516e-06 | 1.76124e-06 | 4.00012 | 1.76124e-06 |
| 4 | 1.15221e-05 | 2.88035e-06 | 4.00023 | 2.88036e-06 |
| 5 | 2.94153e-05 | 7.35118e-06 | 4.00145 | 7.35123e-06 |

Reference intervals: [512, 1024, 2048]; common observation intervals: 256. The last difference is retained without optimistic Richardson reduction. This empirical evidence is not a rigorous continuum error bound.

| Intervals | Nodes per axis | Interior unknowns | Retained modes | Retained bank rank | Bank bytes | Bank setup s | Weak assembly s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 257 | 65025 | 64 | 64 | 33292800 | 2.40073 | 1.07807 |
| 512 | 513 | 261121 | 64 | 64 | 133693952 | 1.44213 | 1.09566 |

Both network weights remain identical across meshes. New-grid bank evaluation and weak operator assembly recur. Setup timings include their first compilation; warmup timings are retained in JSON. Training cost is inherited and is not amortized by this pilot. Dense projection and requested output depend on grid size, so no flat end-to-end complexity claim follows.

## Limits and open work

The result covers a compact inherited checkpoint and the fixed smooth single-Gaussian source family on a square, constant-coefficient, zero-Dirichlet problem. It does not establish performance on variable coefficients, new geometries, other checkpoints, broader input families or final cohorts. The tolerance ladder is a deployment study of frozen weights; no per-resolution retraining has run. Future performance improvements need separate paired validation and must preserve the input/output contract.

## Plain-language glossary

- **Intervals / nodes / interior unknowns:** cells along an axis / coordinates including boundary walls / values solved inside the walls.
- **Arm / tau:** measured algorithm / requested reduction of its initial weak residual. A zero tau disables that early stop.
- **ROM / FOM / DST:** reduced model / full discrete model / fast sine transform direct solver.
- **Median / worst:** middle measured value / largest source error. Raw repeated times remain in JSON.
- **Physical / same-grid error:** relative discrepancy from refined reference / from the full solver on the identical mesh.
- **Invalid / nonstationary sources:** failed solver validity checks / normalized gradient above the configured stationary threshold. An intentional tau stop may remain valid.
- **Timing outlier:** repetition taking more than three times its configuration median; retained, never discarded.
- **Target / speedup:** required error ceiling / full-model cost divided by reduced-model cost after qualification.
- **Coarse DST / prolongation:** source restriction to fewer grid points followed by direct solve / interpolation back to requested output; its cost is charged.
- **Reference gap / ratio / uncertainty:** discrepancy between consecutive fine meshes / first gap divided by final gap / conservative empirical residual reference error.
- **Mode / bank rank / setup:** smooth weak test function / independently retained spatial directions / one-time mesh preparation.
- **Bytes / ms / s:** stored memory units / milliseconds / seconds.
- **Checkpoint / seed / cohort / commit:** saved trained weights / reproducible random generator setting / fixed source group / immutable code revision.
- **Development / qualification / amortization:** preliminary selection evidence / meeting every declared accuracy and validity condition / repaying setup cost with repeated-query savings.
