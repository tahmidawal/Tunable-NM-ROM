"""Declare existing FD1 reference checks independently of their measured outcomes."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1]
C=B.parents[2]/'2026-10-27-ns-obstacle'/'experiments/ns-obstacle'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
campaign=json.loads((B/'g2/campaign-A6.json').read_text());g=campaign['reference_gates'];rows=[];shared=[]
def add(name,role,op='true',threshold=True,required=True,applies='all',source='DESIGN.md §4',target=rows):
 target.append(dict(name=name,role=role,required=required,operator=op,threshold=threshold,applies_to_re=applies,criterion_source=source))
for name in ['exact_population_state_coverage','source_and_retention_closure','fixed_FD1_geometry_and_units','complete_common_rule_no_dropped_points','all_states_native_algebra_and_reload_valid','direct_five_state_S_Q_R_coverage']:
 add(name,'input_and_control_integrity',source='DESIGN.md §4; RESOLVED-NOISE-PROTOCOL.md; FD1-SCOPE-CONTRACT.json')
for name,key in [('geometry_volume_relative','geometry_relative'),('geometry_perimeter_relative','geometry_relative'),('independent_free_residual','free_dof_residual'),('mass_relative','mass_2d'),('divergence_scaled','divergence_scaled'),('essential_trace_over_U','essential_trace_over_U'),('physical_energy_balance','balance_scaled'),('physical_momentum_balance_max','balance_scaled'),('physical_gradient_drag_relative','physical_gradient_drag_gap_relative'),('physical_gradient_transverse_absolute','physical_gradient_transverse_gap_absolute'),('reaction_gradient_drag_relative','h_drag_relative'),('reaction_gradient_transverse_absolute','h_transverse_absolute_2d')]:
 add(name,'adopted_truth','le',g[key],source='campaign-A6.json/reference_gates/'+key+'; DESIGN.md §4/A3')
add('positive_mapped_jacobian','adopted_truth','gt',0.)
add('no_obstacle_quadrature','adopted_truth');add('correct_boundary_labels_and_normal_orientation','adopted_truth')
add('divergence_decreases_under_h_refinement','adopted_h_trend',source='DESIGN.md §4 divergence row')
for metric,key,solver,delta in [('velocity_relative','h_velocity_relative',1e-5,.002),('pressure_scaled','h_pressure_scaled',2e-5,.004),('drag_relative','h_drag_relative',None,.004),('transverse_absolute','h_transverse_absolute_2d',None,.0004)]:
 add('h_finest_'+metric,'adopted_h_trend','le',g[key],source='campaign-A6.json/reference_gates/'+key)
 add('resolved_decreasing_h_'+metric,'adopted_h_trend',source='RESOLVED-NOISE-PROTOCOL.md')
 add('conditional_delta_'+metric,'conditional_uncertainty','le',delta,source='DESIGN.md §4/A1; campaign-A6.json uncertainty_max_fraction_target=0.2 times pilot target')
 add('p_geometry_evidence_'+metric,'p_geometry_control',source='PROSPECTIVE-P4-REFERENCE-ROLE.md; RESOLVED-NOISE-PROTOCOL.md: both contrasts, direct noise and P3 geometry triangle retained; unresolved sensitivity upper interval must fit existing adequacy budget, no automatic nonzero/p5/g5 requirement')
 if solver is not None:add('solver_change_'+metric,'adopted_tolerance','le',solver,source='DESIGN.md §4 tolerance adequacy')
for component in ('drag','transverse'):add('solver_change_'+component+'_absolute','adopted_tolerance','le',1e-5,source='DESIGN.md §4 tolerance adequacy: each force coefficient absolute')
add('higher_surface_quadrature_retained','adopted_force_control',source='DESIGN.md §4 force integration: repeat surface rule at higher order; retain12/16 and Q16/20, no invented additional scalar threshold')
add('outlet_flux_backflow_and_stress_audited','adopted_boundary_control',source='DESIGN.md §4 domain/outlet audit retained under FD1; no invented backflow cutoff')
for name,key in [('dfg_drag_relative','dfg_drag_relative'),('dfg_pressure_drop_relative','dfg_pressure_drop_relative'),('dfg_lift_absolute','dfg_lift_absolute')]:add(name,'benchmark_anchor','le',g[key],applies=[20],source='campaign-A6.json/reference_gates/'+key+'; canonical DFG Re20 only')
for name in ['refinement_and_P3_physical_diagnostics_retained','P3_fixed_order_geometry_diagnostic_retained','P4_g5_interaction_unmeasured_disclosed']:
 add(name,'diagnostic_disclosure',source='PROSPECTIVE-P4-REFERENCE-ROLE.md; no P3 physical gate imported into P4 admission')
add('original_domain_sensitivity_pass','original_broader_G2',required=False,source='AMENDMENT-FD1-USER-APPROVED-SCOPE.md: original failed gate retained, not waived or relabeled')
for name,source in [('analytic_channel_controls','DESIGN.md §4 analytic control'),('manufactured_obstacle_force_and_flux_controls','DESIGN.md A3'),('wall_shear_sign_control','DESIGN.md A3'),('derivative_orientation_and_native_batch_controls','DESIGN.md A6; CACHE-PRIMARY01-RESULT-REVIEW.json'),('portable_coefficient_primary_reconstruction_controls','CACHE-PRIMARY01-RESULT-REVIEW.json'),('common_immutable_mesh_layout_and_Stokes_scaling','DESIGN.md A6; retained g2-crossp02 immutable_lift')]:add(name,'shared_operator_or_geometry_control',source=source,target=shared)
record={'schema':'ns-obstacle-FD1-check-contract-v1','status':'prospective_requires_independent_review','qualification_gate':'G2_FIXED_FINITE_CHANNEL_V1','per_parameter':rows,'shared':shared,'role_policy':'Only p4_g4_h3 baseline supplies admitted physical truth; h1/h2 and P3 states supply declared convergence/order/noise diagnostics. All15 states require actual native algebra/reload and source integrity. Diagnostic physical failures remain visible. No parameter inherits another parameter physics/noise.','source_sha256':{'campaign-A6.json':sha(B/'g2/campaign-A6.json')}}
for name in ['DESIGN.md','reviews/RESOLVED-NOISE-PROTOCOL.md','reviews/PROSPECTIVE-P4-REFERENCE-ROLE.md','reviews/FD1-SCOPE-CONTRACT.json']:
 record['source_sha256'][name]=sha(C/name)
record['scientific_contracts']={'design_sha256':sha(C/'DESIGN.md'),'campaign_sha256':sha(B/'g2/campaign-A6.json'),'scope_contract_sha256':sha(C/'reviews/FD1-SCOPE-CONTRACT.json'),'noise_protocol_sha256':sha(C/'reviews/RESOLVED-NOISE-PROTOCOL.md')}
(B/'g2/FD1-CHECK-CONTRACT.json').write_text(json.dumps(record,indent=2)+'\n');print(len(rows),len(shared))
