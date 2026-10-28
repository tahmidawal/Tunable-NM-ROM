# Provisional spatial refinement screen

Numerical h-trend screen; raw-array review and resolved noise bounds still required. No reference admission.

| Check | Earlier | Finest | Finest limit | Decreasing | Finest passes |
|---|---:|---:|---:|---|---|
| velocity_relative | 0.000177038425 | 2.56088293e-05 | 0.001 | True | True |
| pressure_scaled | 0.000416722279 | 9.23315407e-05 | 0.002 | True | True |
| physical_drag_relative | 0.000654549629 | 0.000130303254 | 0.002 | True | True |
| physical_lift_absolute | 0.000671419648 | 0.00024542508 | 0.0001 | True | False |
| scaled_divergence | 0.00140247734 | 0.000427975442 | 0.001 | True | True |

The threshold applies only to the finest pair. The earlier pair establishes the decreasing trend. Divergence is a per-mesh constraint, not a pair difference.

## Glossary

Earlier/finest: successive mesh differences for fields and forces; second/last mesh values for divergence. Relative: divided by the finer result norm or drag. Scaled pressure: normalized by the frozen physical pressure scale. Lift: transverse force coefficient. Decreasing: numerical trend only; a measured noise floor is still required to resolve it.
