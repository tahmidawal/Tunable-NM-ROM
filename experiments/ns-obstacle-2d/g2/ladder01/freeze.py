"""Freeze a source-only initial h-ladder and measured resource projection."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
B=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((D/'PLAN.json').read_text());native=json.loads((B/'artifacts/g2-native01/results/dfg/result.json').read_text())
with np.load(B/'artifacts/g2-native01/results/dfg/full_rule.npz',allow_pickle=False) as saved_rule:
    baseline_point_count=len(saved_rule['weights'])
base=native['configuration'];memory=json.loads((B/'artifacts/g2-native01/results/dfg/memory_checks.json').read_text());nnz=max(x['nnz'] for x in memory)
rows=[]
for case in plan['cases']:
 cfg=json.loads((B/case['configuration']).read_text());scale=(base['maxh']/cfg['maxh'])**2
 changed={k:{'native':base.get(k),'candidate':cfg.get(k)} for k in set(base)|set(cfg) if base.get(k)!=cfg.get(k)}
 assert set(changed)<={'kind','maxh','obstacle_h','purpose'}
 ne=round(native['geometry']['element_count']*scale);nd=round(native['geometry']['ndof']*scale);nq=round(baseline_point_count*scale);nz=round(nnz*scale)
 # Original per-point Python objects, parent arrays, child replay arrays, compression
 # and evaluation workspace are conservatively budgeted together, separately from FE factor.
 export_workspace=nq*2048
 factor=nz*16*cfg['factor_fill_estimate']+nd*8*32
 projected_memory=factor+native['peak_rss_bytes']+export_workspace
 phases=native['reference_phase_seconds'];nonlinear=sum(phases[k] for k in ('geometry_spaces_fields_forms','stokes_lift','stokes_checkpoint','newton_query','independent_residual_and_Jacobian'))
 diagnostic=phases['reference_diagnostics_and_export']+phases['trace_certificate_and_reload']
 predicted_seconds=nonlinear*scale**1.5+diagnostic*scale
 rows.append({'case':case['name'],'config_delta':changed,'projected_elements':ne,'projected_dofs':nd,'projected_volume_points':nq,'projected_nnz':nz,'factor_accounting_bytes':factor,'export_parent_child_workspace_envelope_bytes':export_workspace,'projected_working_bytes':projected_memory,'planning_seconds_not_guarantee':predicted_seconds})
free=(__import__('os').statvfs(B).f_bavail*__import__('os').statvfs(B).f_frsize)
resource={'status':'prospective_resource_only_amendment_requires_review','native_result_sha256':sha(B/'artifacts/g2-native01/results/dfg/result.json'),'measured_native_seconds':native['elapsed_seconds'],'measured_native_rss_bytes':native['peak_rss_bytes'],'projection':'h^-2 mesh scaling; factor accounting unchanged; 2048 bytes per point combined export/replay workspace; solver time N^1.5 plus linear export time, not a guarantee','cases':rows,'producer_volume_point_cap':{'previous':100000,'proposed':500000},'unchanged_caps':{'dofs':100000,'elements':20000,'nnz':12000000,'boundary_points':30000,'archive_bytes':2**30,'memory_stop_GiB':48,'allocation_GiB':64,'CPUs':16,'GPUs':1,'wall_seconds':1200},'downstream_ROM_caps':'unchanged; not exercised by this job','comparison_planning_seconds_unmeasured':180,'total_planning_seconds_not_guarantee':sum(x['planning_seconds_not_guarantee'] for x in rows)+180,'per_process_timeout_seconds':300,'runner_internal_budget_seconds':1100,'retention_working_copies_GiB_envelope':12,'local_free_bytes':free,'required_local_reserve_bytes':10*2**30,'retention_plus_reserve_fits':free>22*2**30,'physical_admission':False}
(D/'RESOURCE-PROJECTION.json').write_text(json.dumps(resource,indent=2)+'\n')
names=['fom.py','reference_export.py','shared_lift.py','preflight.py','scalars.py','oriented_normals.py','benchmark.json','manufactured_witness.json','dependencies.py','DEPENDENCIES.json','g2/campaign-A6.json','g2/DESIGN-A6.md','g2/compare_fields.py','g2/ladder01/PLAN.json','g2/ladder01/run.py','g2/ladder01/summary.py','g2/ladder01/ladder.sh']+[x['configuration'] for x in plan['cases']]
files=[{'source':n,'destination':n,'sha256':sha(B/n),'bytes':(B/n).stat().st_size} for n in names]
n='g2/ladder01/PROVISION.json';files.append({'source':n,'destination':'REPAIR.json','sha256':sha(B/n),'bytes':(B/n).stat().st_size})
assert sha(B/'oriented_normals.py')=='3a13ea7ca6c9110997b1c44fd04006bdf9d5f089121beb91b2988a3ca6d7779b'
(D/'SOURCE-CANDIDATE.json').write_text(json.dumps({'status':'review_candidate_not_staged_or_released','executed_native_base':'b29448620cabd7efe957364eec99a83e7f73368c','parent_at_freeze':subprocess.check_output(['git','rev-parse','HEAD'],cwd=B,text=True).strip(),'files':files,'physical_admission':False},indent=2)+'\n')
parts=[]
for name in ('fom.py','reference_export.py'):
 old=subprocess.check_output(['git','show','b29448620:experiments/ns-obstacle-2d/'+name],cwd=B,text=True)
 import difflib
 parts.extend(difflib.unified_diff(old.splitlines(True),(B/name).read_text().splitlines(True),fromfile='native01/'+name,tofile='ladder01/'+name))
(D/'PRODUCER-DELTA.patch').write_text(''.join(parts))
print(json.dumps({'inputs':len(files),'max_projected_points':max(x['projected_volume_points'] for x in rows),'max_projected_working_GiB':max(x['projected_working_bytes'] for x in rows)/2**30,'planning_seconds':resource['total_planning_seconds_not_guarantee'],'reserve_fits':resource['retention_plus_reserve_fits']},indent=2))
