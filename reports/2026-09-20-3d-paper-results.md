# Three-dimensional NM-ROM comparisons: generated results

These tables contain retained, audited development experiments from the overnight campaign. They are provisional paper material: tuning, operator comparisons and independent final evaluation must be assessed per attempt before a row supports a manuscript claim.

The explicit input manifest controls which attempts appear; no best run is selected automatically. Errors and timings are recomputed from the same invocation records. Every table retains unsuccessful methods and reports failure counts.

## Burgers 3D — b3d001

**Provisional:** Development diagnostic: retained network and smaller POD/direction training subset are unmatched; physical refinement and operator comparisons are absent. Dense initialization and dense nonlinear evaluation are charged. Final cohort remains unopened.

Source `1d2010aacff49a77ff7f8fe2c3221b4c6c739dc6`; job `3989410`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d001/collected/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d001/collected/out/audit-local.json).

Errors use the initial-field norm. The evolved error excludes the initial compression; all-times and initial errors are also shown. GPU timing begins with the dense input already on device and ends with every requested dense output on device. Host transfers are unmeasured in this attempt. Mesh size counts nodes per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 33 | `fom_nt1e-02` | 8 | 0.0382 | 0.6543 | 0.6543 | 0.0000 | — | 17.930 | — | 0 / 0 | 0 / 24 |
| 33 | `fom_nt1e-04` | 8 | 0.0188 | 0.0207 | 0.0207 | 0.0000 | — | 38.131 | — | 0 / 0 | 0 / 24 |
| 33 | `fom_nt1e-06` | 8 | 0.0000 | 0.0003 | 0.0003 | 0.0000 | — | 91.753 | — | 0 / 0 | 0 / 24 |
| 33 | `free_R128` | 8 | 2.0228 | 5.9695 | 7.0334 | 7.0334 | — | 143.348 | — | 0 / 0 | 0 / 24 |
| 33 | `pod_128` | 8 | 1.8758 | 5.3034 | 6.0555 | 6.0555 | — | 142.684 | — | 0 / 0 | 0 / 24 |
| 33 | `pod_32` | 8 | 10.4499 | 19.1398 | 20.8596 | 20.8596 | — | 91.625 | — | 0 / 0 | 0 / 24 |
| 33 | `pod_64` | 8 | 4.9116 | 10.0613 | 10.8955 | 10.8955 | — | 109.704 | — | 0 / 0 | 0 / 24 |
| 33 | `pod_96` | 8 | 2.8234 | 6.5582 | 7.5205 | 7.5205 | — | 138.375 | — | 0 / 0 | 0 / 24 |
| 33 | `rom_q0` | 8 | 4.5930 | 12.9762 | 14.5016 | 14.5016 | — | 162.856 | — | 0 / 1 | 0 / 24 |
| 33 | `rom_q32` | 8 | 3.9653 | 10.6624 | 11.9876 | 11.9876 | — | 193.751 | — | 0 / 0 | 0 / 24 |
| 33 | `rom_q64` | 8 | 3.0091 | 8.2111 | 9.3453 | 9.3453 | — | 237.352 | — | 0 / 0 | 0 / 24 |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Glossary

- **NM-ROM / ROM:** a neural-manifold reduced model / a model solving for a smaller state.
- **Bank, head, rank:** learned spatial functions, their nonlinear coefficient map, and the number of spatial functions.
- **q / K / R:** correction rank / nonlinear latent dimension / full learned-bank rank; values in method names identify the saved configuration.
- **POD:** a linear reduced basis computed from training snapshots.
- **FOM / DST:** the full-grid numerical solver / a discrete sine-transform direct solver.
- **Dense / EQ:** full-grid contractions / sampled empirical quadrature; an EQ name alone does not mean its accuracy certificate passed.
- **Free bank / Galerkin / weak:** unrestricted bank coefficients / projection against the basis / residual projection against smooth tests.
- **Mesh / cases / calls:** grid size per spatial axis under the stated convention / distinct inputs / timed solver invocations including repetitions.
- **Error median / worst:** median or maximum over distinct cases of each case's largest evolved error; the largest value across repeated calls is retained.
- **All-times / initial:** maximum including time zero / error from compressing the initial field.
- **Physical error:** discrepancy against the independently refined reference; a refinement test is empirical, not a proved continuum bound.
- **GPU / total median:** median elapsed milliseconds on the device / including recorded host transfers. Timings may only be compared within a job and matching output contract.
- **Nonfinite / nonstationary cases:** inputs with an invalid output in any repeat / a failed declared numerical stopping check in any repeat.
- **Timing outliers:** calls slower than one and a half times that method's median; every measured time remains in the machine-readable output.
- **Provisional / development / final cohort:** not accepted as a final paper claim / data available during selection / independent data reserved until configurations freeze.
- **Source / job / audit:** pinned scientific code revision / cluster allocation identifier / independent validation record.
