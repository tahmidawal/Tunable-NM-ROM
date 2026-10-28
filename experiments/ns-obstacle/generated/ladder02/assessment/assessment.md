# Provisional reference qualification screen

provisional per-case screen only; cross-run and raw evidence audit required. No training admission is issued.

| Case | Check | Saved value | Threshold | Result |
|---|---|---:|---:|---|
| dfg_h2 | fluid_area_relative | 5.35194323e-10 | 1e-05 | within threshold |
| dfg_h2 | obstacle_perimeter_relative | 3.05090531e-08 | 1e-05 | within threshold |
| dfg_h2 | independent_free_residual | 1.77245989e-15 | 1e-09 | within threshold |
| dfg_h2 | prescribed_FE_trace_over_U | 1e-12 | 1e-10 | within threshold |
| dfg_h2 | scaled_divergence | 0.00140247734 | 0.001 | fails |
| dfg_h2 | inlet_outlet_mass | 0 | 1e-07 | within threshold |
| dfg_h2 | physical_energy_balance | 1.46224193e-07 | 0.005 | within threshold |
| dfg_h2 | physical_momentum_0 | 8.66023299e-06 | 0.005 | within threshold |
| dfg_h2 | physical_momentum_1 | 4.19948149e-06 | 0.005 | within threshold |
| dfg_h2 | physical_gradient_drag_relative | 0.000121676525 | 0.0002 | within threshold |
| dfg_h2 | physical_gradient_lift_absolute | 5.90134999e-05 | 1e-05 | fails |
| dfg_h2 | reaction_gradient_drag_relative | 3.37321733e-05 | 0.002 | within threshold |
| dfg_h2 | reaction_gradient_lift_absolute | 0.000137475159 | 0.0001 | fails |
| dfg_h2 | physical_DFG_drag_relative | 0.000157145509 | 0.002 | within threshold |
| dfg_h2 | physical_DFG_lift_absolute | 0.000195926939 | 0.0001 | fails |
| dfg_h2 | DFG_pressure_drop_relative | 0.000558933337 | 0.002 | within threshold |
| manufactured_h2 | fluid_area_relative | 5.35194447e-10 | 1e-05 | within threshold |
| manufactured_h2 | obstacle_perimeter_relative | 3.05090532e-08 | 1e-05 | within threshold |
| manufactured_h2 | independent_free_residual | 6.24142254e-12 | 1e-09 | within threshold |
| manufactured_h2 | prescribed_FE_trace_over_U | 1e-12 | 1e-10 | within threshold |
| manufactured_h2 | scaled_divergence | 1.09260438e-07 | 0.001 | within threshold |
| manufactured_h2 | analytic_velocity | 7.14004288e-10 | 1e-06 | within threshold |
| manufactured_h2 | analytic_pressure | 1.43986327e-08 | 1e-06 | within threshold |
| manufactured_h2 | physical_energy_balance | 3.76159837e-10 | 0.005 | within threshold |
| manufactured_h2 | physical_momentum_0 | 1.32005811e-10 | 0.005 | within threshold |
| manufactured_h2 | physical_momentum_1 | 1.18298986e-09 | 0.005 | within threshold |

Cross-run controls not assessed by this screen: two_decreasing_h_differences, p4_g4_crosscheck, p3_g4_geometry, solver_100x, quadrature_plus4, upstream_extension, downstream_extension, symmetric_outlet_long_domain, conditional_uncertainty, shared_mesh_lift_reuse, parameter_population_coverage, independent_raw_evidence_review.

Raw per-case controls not assessed here: positive_mapped_Jacobians_and_fluid_points, normal_and_binary_reload_certificates, surface_rule_12_vs_16, outlet_backflow, manufactured_force_and_derivative_controls, wall_shear_sign_control, manufactured_global_flux.

## Glossary

Case: one saved solver invocation. Saved value: the recorded metric or the explicit force/probe difference derived from it. Threshold: the frozen campaign limit. Within threshold: this one check passes; it does not qualify the reference. FE trace: prescribed finite-element boundary velocity. Divergence: local incompressibility error. DFG: cylinder-in-channel benchmark. Reaction force: force recovered from the discrete equations; gradient and physical forces use different stress conventions. Cross-run: evidence requiring more than one independently retained solve. Conditional uncertainty: a refinement-based estimate, not a rigorous error bound.
