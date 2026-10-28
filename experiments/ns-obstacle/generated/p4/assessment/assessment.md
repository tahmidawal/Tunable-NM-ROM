# Provisional reference qualification screen

provisional per-case screen only; cross-run and raw evidence audit required. No training admission is issued.

| Case | Check | Saved value | Threshold | Result |
|---|---|---:|---:|---|
| dfg_p4_g4_h3 | fluid_area_relative | 2.47089824e-14 | 1e-05 | within threshold |
| dfg_p4_g4_h3 | obstacle_perimeter_relative | 2.64162736e-13 | 1e-05 | within threshold |
| dfg_p4_g4_h3 | independent_free_residual | 2.08910911e-15 | 1e-09 | within threshold |
| dfg_p4_g4_h3 | prescribed_FE_trace_bound_over_U | 1e-12 | 1e-10 | within threshold |
| dfg_p4_g4_h3 | scaled_divergence | 1.80288702e-05 | 0.001 | within threshold |
| dfg_p4_g4_h3 | inlet_outlet_mass | 5.07723944e-16 | 1e-07 | within threshold |
| dfg_p4_g4_h3 | physical_energy_balance | 7.75077193e-10 | 0.005 | within threshold |
| dfg_p4_g4_h3 | physical_momentum_0 | 1.84923969e-07 | 0.005 | within threshold |
| dfg_p4_g4_h3 | physical_momentum_1 | 5.09134023e-08 | 0.005 | within threshold |
| dfg_p4_g4_h3 | physical_gradient_drag_relative | 8.04044603e-08 | 0.0002 | within threshold |
| dfg_p4_g4_h3 | physical_gradient_lift_absolute | 1.90955134e-08 | 1e-05 | within threshold |
| dfg_p4_g4_h3 | reaction_gradient_drag_relative | 2.32087599e-06 | 0.002 | within threshold |
| dfg_p4_g4_h3 | reaction_gradient_lift_absolute | 3.23336883e-06 | 0.0001 | within threshold |
| dfg_p4_g4_h3 | physical_DFG_drag_relative | 2.24055772e-06 | 0.002 | within threshold |
| dfg_p4_g4_h3 | physical_DFG_lift_absolute | 3.21485517e-06 | 0.0001 | within threshold |
| dfg_p4_g4_h3 | DFG_pressure_drop_relative | 3.50271329e-07 | 0.002 | within threshold |
| manufactured_p4_g4_h3 | fluid_area_relative | 2.45848166e-14 | 1e-05 | within threshold |
| manufactured_p4_g4_h3 | obstacle_perimeter_relative | 2.63986039e-13 | 1e-05 | within threshold |
| manufactured_p4_g4_h3 | independent_free_residual | 3.26518498e-12 | 1e-09 | within threshold |
| manufactured_p4_g4_h3 | prescribed_FE_trace_bound_over_U | 1e-12 | 1e-10 | within threshold |
| manufactured_p4_g4_h3 | scaled_divergence | 6.37564845e-11 | 0.001 | within threshold |
| manufactured_p4_g4_h3 | analytic_velocity | 9.48044354e-13 | 1e-06 | within threshold |
| manufactured_p4_g4_h3 | analytic_pressure | 2.39929559e-11 | 1e-06 | within threshold |
| manufactured_p4_g4_h3 | physical_energy_balance | 1.15861865e-12 | 0.005 | within threshold |
| manufactured_p4_g4_h3 | physical_momentum_0 | 6.77740905e-13 | 0.005 | within threshold |
| manufactured_p4_g4_h3 | physical_momentum_1 | 1.08574836e-12 | 0.005 | within threshold |

Cross-run controls not assessed by this screen: two_decreasing_h_differences, p4_g4_crosscheck, p3_g4_geometry, solver_100x, quadrature_plus4, upstream_extension, downstream_extension, symmetric_outlet_long_domain, conditional_uncertainty, shared_mesh_lift_reuse, parameter_population_coverage, independent_raw_evidence_review.

Raw per-case controls not assessed here: positive_mapped_Jacobians_and_fluid_points, normal_and_binary_reload_certificates, surface_rule_12_vs_16, outlet_backflow, manufactured_force_and_derivative_controls, wall_shear_sign_control, manufactured_global_flux.

## Glossary

Case: one saved solver invocation. Saved value: the recorded metric or the explicit force/probe difference derived from it. Threshold: the frozen campaign limit. Within threshold: this one check passes; it does not qualify the reference. FE trace bound: bound on error against prescribed finite-element boundary velocity, including a fixed roundoff allowance; it is not a raw observed error. Raw polynomial coefficients and independent check residuals are retained in the producer certificate. Divergence: local incompressibility error. DFG: cylinder-in-channel benchmark. Reaction force: force recovered from the discrete equations; gradient and physical forces use different stress conventions. Cross-run: evidence requiring more than one independently retained solve. Conditional uncertainty: a refinement-based estimate, not a rigorous error bound.
