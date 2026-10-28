"""Freeze source-only P4 trend and analytical/geometry-control packages."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
common=['fom.py','reference_export.py','shared_lift.py','scalars.py','oriented_normals.py','preflight.py','dependencies.py','DEPENDENCIES.json','benchmark.json','manufactured_witness.json','g2/campaign-A6.json','g2/DESIGN-A6.md','g2/compare_fields.py','g2/exact_lookup.py']
projection=json.loads((B/'g2/P4-RESOURCE-PROPOSAL.json').read_text())
progress_path=B/'staging/g2-h3/H3-PROGRESS-PRELIMINARY.json'
progress=json.loads(progress_path.read_text())
comparison_seconds=next(row['seconds'] for row in progress['completed'] if row['name']=='comparison_dfg')
for slug,batch,configs in [('p4trend','p4trend.sh',['dfg_h1.json','dfg_h2.json','dfg_h3.json']),('p4controls','controls.sh',['dfg_p3_g4_h3.json','manufactured_p4_g4_h3.json'])]:
 d=B/'g2'/slug;names=common+[f'g2/{slug}/{n}' for n in ['run.py','summary.py',batch]+configs]
 rows=[{'source':n,'destination':n,'sha256':sha(B/n),'bytes':(B/n).stat().st_size} for n in names]
 p=d/'PROVISION.json';rows.append({'source':str(p.relative_to(B)),'destination':'REPAIR.json','sha256':sha(p),'bytes':p.stat().st_size})
 solves=sum(r['planning_seconds_not_guarantee'] for r in projection['rows'] if r['case'] in (['p4_g4_h1','p4_g4_h2','p4_g4_h3'] if slug=='p4trend' else ['p3_g4_h3','p4_g4_h3']))
 record={'status':'source_review_candidate_no_execution_release','files':rows,'resource':{'allocation_GiB':64,'working_guard_GiB':48,'gpus':1,'cpus':16,'wall_seconds':1200,'runner_seconds':1100,'per_process_seconds':600,'solves_planning_seconds_not_guarantee':solves,'comparison_hard_budget_seconds':600 if slug=='p4trend' else 0,'solves_planning_plus_comparison_hard_limit_seconds':solves+(600 if slug=='p4trend' else 0),'projection_sha256':sha(B/'g2/P4-RESOURCE-PROPOSAL.json'),'measured_P3_h3_comparison_seconds':comparison_seconds,'comparison_timing_provenance_sha256':sha(progress_path),'note':'P4 per-process limit600s prospective capacity-only change. No equation, threshold, quadrature, initialization or solver change. Exact counts and memory checked before factorization.'},'scope':'P4/g4 ownhtrend onexistingh1/h2/h3' if slug=='p4trend' else 'fixed-P3geometry and nativeP4manufactured controls; cross-p comparisons follow after retained P4trend inputs','physical_admission':False,'training_released':False,'no_new_h_level':True}
 (d/'SOURCE-CANDIDATE.json').write_text(json.dumps(record,indent=2)+'\n');print(slug,len(rows),sha(d/'SOURCE-CANDIDATE.json'))
