# Provisional reference qualification screen

provisional per-case screen only; cross-run and raw evidence audit required. No training admission is issued.

| Case | Check | Saved value | Threshold | Result |
|---|---|---:|---:|---|
| dfg_h3 | fluid_area_relative | 4.00956009e-11 | 1e-05 | within threshold |
| dfg_h3 | obstacle_perimeter_relative | 2.28496932e-09 | 1e-05 | within threshold |
| dfg_h3 | independent_free_residual | 2.05215747e-15 | 1e-09 | within threshold |
| dfg_h3 | prescribed_FE_trace_bound_over_U | 1e-12 | 1e-10 | within threshold |
| dfg_h3 | scaled_divergence | 0.000427975442 | 0.001 | within threshold |
| dfg_h3 | inlet_outlet_mass | 5.07723944e-16 | 1e-07 | within threshold |
| dfg_h3 | physical_energy_balance | 2.70127012e-09 | 0.005 | within threshold |
| dfg_h3 | physical_momentum_0 | 2.13670197e-06 | 0.005 | within threshold |
| dfg_h3 | physical_momentum_1 | 4.95523634e-07 | 0.005 | within threshold |
| dfg_h3 | physical_gradient_drag_relative | 1.08588933e-05 | 0.0002 | within threshold |
| dfg_h3 | physical_gradient_lift_absolute | 1.41628812e-05 | 1e-05 | fails |
| dfg_h3 | reaction_gradient_drag_relative | 1.56656542e-05 | 0.002 | within threshold |
| dfg_h3 | reaction_gradient_lift_absolute | 3.61649477e-05 | 0.0001 | within threshold |
| dfg_h3 | physical_DFG_drag_relative | 2.68387575e-05 | 0.002 | within threshold |
| dfg_h3 | physical_DFG_lift_absolute | 4.94981416e-05 | 0.0001 | within threshold |
| dfg_h3 | DFG_pressure_drop_relative | 4.21017021e-05 | 0.002 | within threshold |
| manufactured_h3 | fluid_area_relative | 4.00957251e-11 | 1e-05 | within threshold |
| manufactured_h3 | obstacle_perimeter_relative | 2.28496915e-09 | 1e-05 | within threshold |
| manufactured_h3 | independent_free_residual | 3.26523741e-12 | 1e-09 | within threshold |
| manufactured_h3 | prescribed_FE_trace_bound_over_U | 1e-12 | 1e-10 | within threshold |
| manufactured_h3 | scaled_divergence | 1.12318012e-08 | 0.001 | within threshold |
| manufactured_h3 | analytic_velocity | 3.97724619e-11 | 1e-06 | within threshold |
| manufactured_h3 | analytic_pressure | 1.47525594e-09 | 1e-06 | within threshold |
| manufactured_h3 | physical_energy_balance | 4.3176359e-11 | 0.005 | within threshold |
| manufactured_h3 | physical_momentum_0 | 1.68962117e-11 | 0.005 | within threshold |
| manufactured_h3 | physical_momentum_1 | 1.5479568e-10 | 0.005 | within threshold |

Cross-run controls not assessed by this screen: two_decreasing_h_differences, p4_g4_crosscheck, p3_g4_geometry, solver_100x, quadrature_plus4, upstream_extension, downstream_extension, symmetric_outlet_long_domain, conditional_uncertainty, shared_mesh_lift_reuse, parameter_population_coverage, independent_raw_evidence_review.

Raw per-case controls not assessed here: positive_mapped_Jacobians_and_fluid_points, normal_and_binary_reload_certificates, surface_rule_12_vs_16, outlet_backflow, manufactured_force_and_derivative_controls, wall_shear_sign_control, manufactured_global_flux.

## Glossary

Case: one saved solver invocation. Saved value: the recorded metric or the explicit force/probe difference derived from it. Threshold: the frozen campaign limit. Within threshold: this one check passes; it does not qualify the reference. FE trace bound: bound on error against prescribed finite-element boundary velocity, including a fixed roundoff allowance; it is not a raw observed error. Raw polynomial coefficients and independent check residuals are retained in the producer certificate. Divergence: local incompressibility error. DFG: cylinder-in-channel benchmark. Reaction force: force recovered from the discrete equations; gradient and physical forces use different stress conventions. Cross-run: evidence requiring more than one independently retained solve. Conditional uncertainty: a refinement-based estimate, not a rigorous error bound.
