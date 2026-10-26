# Frozen POD/ECSW consistency replay

Diagnostic subset evidence against approved E0 references. Numerical errors retain conditional reference uncertainty; these cases cannot qualify the whole validation population.

Both frozen HR supports FAIL the new whole-residual admission contract. Q1 below certifies the sampled objective for HR arms and the full objective for the no-HR arm. Passing sampled Q1 does not establish agreement with the full least-squares problem.

The field-distance bar is 0.00025; the whole-residual gradient bar is 0.001. Full stationarity must pass on at least 99% of pooled steps.

The job exited after its complete checkpoint and all timings were saved: the final log print could not serialize a NumPy boolean. The saved metadata string for the finite flag is quarantined; admission is recomputed numerically from arrays. No scientific replay was repeated.

| arm | median error | worst error | Q1 | cases above one-percent error |
|---|---:|---:|---|---:|
| k8 | 2.661360% | 3.233608% | True | 4 |
| k16 | 2.701023% | 3.220514% | True | 4 |
| none | 2.704708% | 3.220848% | True | 4 |

| case | k8→k16 distance | k16→none distance | k8→none distance |
|---|---:|---:|---:|
| 0 | 0.0017883789 | 0.000132795 | 0.0017852795 |
| 5 | 0.0034407586 | 0.00034619645 | 0.0036697799 |
| 30 | 0.0028835 | 0.00021306656 | 0.0028967855 |
| 43 | 0.0023071332 | 0.00012342446 | 0.0023153046 |

| case | arm | maximum whole-residual gradient discrepancy | maximum full stationarity |
|---|---|---:|---:|
| 0 | k8 | 0.011090202 | 0.011090205 |
| 0 | k16 | 0.0012331558 | 0.0012771384 |
| 5 | k8 | 0.0076513182 | 0.0074435731 |
| 5 | k16 | 0.0011323156 | 0.0012684406 |
| 30 | k8 | 0.0066631517 | 0.0066631095 |
| 30 | k16 | 0.00069517658 | 0.00093802748 |
| 43 | k8 | 0.015423685 | 0.015423685 |
| 43 | k16 | 0.00088058149 | 0.00088058067 |

Independent saved-array checks passed: True. Paired timing coverage complete: True.

This is a qualification failure for the frozen sampled models, not evidence that POD loses to the coordinate bank. Historical observed accuracy/cost domination remains labelled with its original scope.

## Glossary

- arm: one frozen numerical model; none uses every residual row.
- k8 / k16: archived weighted supports with eight / sixteen points per reduced coefficient.
- median / worst error: median / largest relative error among these diagnostic cases.
- Q1: nonlinear validity, fresh stationarity and independently stricter-solve agreement.
- distance: largest output field difference divided by the reference initial-state norm.
- whole-residual gradient discrepancy: sampled versus full least-squares gradient difference, normalized by full Jacobian and residual norms.
- full stationarity: full least-squares gradient norm with that same normalization.
- E0: refined approved reference; its numerical uncertainty is conditional.
