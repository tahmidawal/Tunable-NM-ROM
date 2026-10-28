"""Prospective resource-only h3 review from retained actual h2 measurements."""
from pathlib import Path
import hashlib,json
import numpy as np
B=Path(__file__).resolve().parents[1];rows=[]
for kind in ('dfg','manufactured'):
 p=B/'artifacts/g2-ladder02/results'/f'{kind}_h2';r=json.loads((p/'result.json').read_text());memory=json.loads((p/'memory_checks.json').read_text())
 with np.load(p/'full_rule.npz',allow_pickle=False) as a:nq=len(a['weights'])
 ne=r['geometry']['element_count']*4;nd=r['geometry']['ndof']*4;nz=max(m['nnz'] for m in memory)*4;npoints=nq*4
 factor=nz*16*100+nd*8*32
 # Includes Python mapped-point objects, raw diagnostic arrays, parent/child replay,
 # compression and comparison workspace; direct factor fill remains separately counted.
 mapping_workspace=npoints*2048
 jacobian_workspace=17*nd*8
 measured_peak_allowance=r['peak_rss_bytes']*4
 estimate=factor+mapping_workspace+jacobian_workspace+measured_peak_allowance
 phases=r['reference_phase_seconds'];nonlinear=sum(phases[k] for k in ('geometry_spaces_fields_forms','stokes_lift','stokes_checkpoint','newton_query','independent_residual_and_Jacobian'))
 seconds=nonlinear*8+(phases['reference_diagnostics_and_export']+phases['trace_certificate_and_reload'])*4
 rows.append({'case':kind+'_h3','baseline_result_sha256':hashlib.sha256((p/'result.json').read_bytes()).hexdigest(),'baseline_memory_sha256':hashlib.sha256((p/'memory_checks.json').read_bytes()).hexdigest(),'baseline_points':nq,'projected_elements':ne,'projected_dofs':nd,'projected_nnz':nz,'projected_points':npoints,'factor_estimate_bytes':factor,'mapped_export_replay_comparison_workspace_bytes':mapping_workspace,'Jacobian_evidence_workspace_bytes':jacobian_workspace,'measured_peak_times_four_allowance_bytes':measured_peak_allowance,'total_working_envelope_bytes':estimate,'planning_seconds_not_guarantee':seconds})
record={'status':'prospective_resource_review_only_no_release','spatial_extension':'single h3 only; h1/h2/h3 differences remain required','proposed_mesh':{'maxh':.01,'obstacle_h':.0025,'velocity_order':3,'pressure_order':2,'geometry_order':3},'producer_caps_proposed':{'elements':{'old':20000,'new':30000},'dofs':{'old':100000,'new':300000},'volume_points':{'old':500000,'new':1300000}},'unchanged_caps':{'nnz':12000000,'factor_fill_estimate':100,'memory_guard_GiB':48,'allocation_GiB':64,'archive_bytes':2**30,'boundary_points':30000,'gpus':1,'cpus':16,'wall_seconds':1200,'per_case_timeout_seconds':300},'cases':rows,'planning_total_seconds_not_guarantee':sum(r['planning_seconds_not_guarantee'] for r in rows),'conditions':['comparison repair validated before h3 execution release','complete fresh native replay at h3','full positive quadrature unchanged; no artificial point deletion or reduced integration','source-bound independent capacity and method review','one level only, no automatic h4','producer caps separate from ROM fitting caps'],
'p_geometry_followup':{'p3_g4_same_h3_dofs_estimate':rows[0]['projected_dofs'],'p4_g4_same_h3_dofs_estimate':round(rows[0]['projected_dofs']*(16+4.5)/(9+2)),'p4_nnz_estimate':round(rows[0]['projected_nnz']*((2*15+10)/(2*10+6))**2),'requires_separate_capacity_review':True},'physical_admission':False,'training_released':False}
(B/'g2/H3-RESOURCE-PROPOSAL.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
