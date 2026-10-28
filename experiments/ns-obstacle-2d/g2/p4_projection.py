"""Prospective existing-mesh p refinement; h3 counts remain preliminary until retention."""
from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parents[1];rows=[]
for level in (1,2,3):
 if level==3:
  p=B/'staging/g2-h3/DFG-H3-PRELIMINARY.json';mp=B/'staging/g2-h3/DFG-H3-MEMORY-PRELIMINARY.json'
 else:
  p=B/f'artifacts/g2-ladder02/results/dfg_h{level}/result.json';mp=p.parent/'memory_checks.json'
 r=json.loads(p.read_text());m=json.loads(mp.read_text());nnz=max(x['nnz'] for x in m)
 for order in ((4,) if level<3 else (3,4,5)):
  scale=(order**2+.5*(order-1)**2)/(9+2)
  local=lambda k:(k+1)*(k+2)/2
  matrixscale=((2*local(order)+local(order-1))/(2*local(3)+local(2)))**2
  nd=round(r['geometry']['ndof']*scale);nz=round(nnz*matrixscale);nq=r['geometry']['element_count']*49
  factor=nz*16*100+nd*8*32;peak=round(r['peak_rss_bytes']*scale);workspace=nq*2048+17*nd*8
  phase=r['reference_phase_seconds'];solver=sum(phase[k] for k in ('geometry_spaces_fields_forms','stokes_lift','stokes_checkpoint','newton_query','independent_residual_and_Jacobian'))
  wall=solver*scale**1.5+(phase['reference_diagnostics_and_export']+phase['trace_certificate_and_reload'])*scale
  rows.append({'case':f'p{order}_g4_h{level}','role':'separate optional p5 independent adequacy proposal; not released' if order==5 else 'prospective complete P4 h trend or separate P3 geometry control','baseline_result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'baseline_memory_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'baseline_retention':'h3 preliminary immutable result copied after case completion; full job still running' if level==3 else 'full checksum verified archive retained','projected_cells':r['geometry']['element_count'],'projected_dofs':nd,'projected_nnz':nz,'projected_volume_points':nq,'factor_accounting_bytes':factor,'working_envelope_bytes':factor+peak+workspace,'planning_seconds_not_guarantee':wall})
record={'status':'prospective_resource_plan_no_execution_release','scientific_role':'P4 cannot inherit P3 metrics or h-trend. Evaluate P4/g4 at existing h1,h2,h3. P3/g4 at h3 isolates geometry effects against retained P3/g3. Independent p adequacy policy must be reviewed before interpreting P4 as primary reference; optional P5/g4 estimate is included, not authorized.','new_h_levels':0,'rows':rows,'p4_proposed_caps':{'elements':30000,'dofs':500000,'nnz':30000000,'volume_points':1300000,'archive_bytes':2**30,'working_guard_GiB':48,'allocation_GiB':64},'optional_p5_requires_separate_review':{'dofs':800000,'nnz':60000000,'working_guard_GiB':96,'allocation_GiB':128,'per_process_seconds':600},'unchanged_physical_thresholds':True,'physical_admission':False}
(B/'g2/P4-RESOURCE-PROPOSAL.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(rows,indent=2))
