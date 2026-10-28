# Provisional spatial refinement screen

Numerical h-trend screen; raw-array review and resolved noise bounds still required. No reference admission.

| Check | Earlier | Finest | Finest limit | Decreasing | Finest passes |
|---|---:|---:|---:|---|---|
| velocity_relative | 1.29337946e-05 | 1.31698999e-06 | 0.001 | True | True |
| pressure_scaled | 3.87765511e-05 | 5.91725368e-06 | 0.002 | True | True |
| physical_drag_relative | 0.000119148142 | 1.49051762e-05 | 0.002 | True | True |
| physical_lift_absolute | 8.08961215e-05 | 2.04030393e-05 | 0.0001 | True | True |
| scaled_divergence | 9.33769457e-05 | 1.80288702e-05 | 0.001 | True | True |

The threshold applies only to the finest pair. The earlier pair establishes the decreasing trend. Divergence is a per-mesh constraint, not a pair difference.

## Glossary

Earlier/finest: successive mesh differences for fields and forces; second/last mesh values for divergence. Relative: divided by the finer result norm or drag. Scaled pressure: normalized by the frozen physical pressure scale. Lift: transverse force coefficient. Decreasing: numerical trend only; a measured noise floor is still required to resolve it.
