"""Generate audit-facing counts/masks and geometric budgets from actual collected mesh data."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();root=HERE/'results'/a.attempt;raw=root/'raw/output';rp=raw/'mesh_result.json'
    z=json.loads(rp.read_text());production=0;anchors=0;masks=[];geometry=[]
    for qid,query in z['queries'].items():
        cases=z['case_results'].get(qid,{})
        for j,row in cases.items():
            production+=1
            with np.load(raw/row['raw']['path'],allow_pickle=False) as arrays:
                anchors+=sum(k in arrays for k in ('tight_W','tighter_W'))
                if 'projection_budget' in row:
                    pb=row['projection_budget'];span=pb['span_id'];oracle=z['projection_budgets'][span]['cases'][j]
                    with np.load(raw/oracle['raw']['path'],allow_pickle=False) as data:Cbest=data['best_coefficients']
                    with np.load(raw/'data/operators/W1024.npz',allow_pickle=False) as metric:L=metric['L65']
                    for level,key in [('production','W'),('tight','tight_W'),('tighter','tighter_W')]:
                        sc=row['score'] if level=='production' else row.get('anchor_scores',{}).get(level,{}).get('score')
                        if sc is None:continue
                        departure=np.linalg.norm((arrays[key]-Cbest)@L,axis=1)/oracle['normalisation']
                        errors=np.asarray(sc['e_t']);floor=np.asarray(oracle['floor_e_t']);defect=errors**2-floor**2-departure**2
                        assert np.max(np.abs(defect))<=1e-8*max(float(np.max(errors**2)),1e-16)+1e-12
                        i=int(np.argmax(errors[1:])+1)
                        geometry.append(dict(query=qid,case=int(j),level=level,physical_max_output_index=i,time=.05*i,
                            error_at_max=float(errors[i]),floor_at_same_time=float(floor[i]),in_span_departure_at_same_time=float(departure[i]),
                            max_Pythagorean_squared_defect=float(np.max(np.abs(defect))),scope='common65 orthogonal geometry; no causal dynamical attribution'))
        if query.get('Q1') or query.get('Q2') or query.get('anchor_Q2'):
            masks.append(dict(query=qid,case_ids=query['cases'],case_count=len(query['cases']),production_Q1=query.get('Q1'),
                production_Q2=query.get('Q2'),adaptive_Q2=query.get('anchor_Q2',{}),
                anchor_usable_by_case={j:r['anchor_usable'] for j,r in cases.items()},
                production_bad_steps_by_case={j:r['nonstationary'] for j,r in cases.items()}))
    owning=[x for x in z['incremental_setup'] if x['bank']=='W1024']
    release=z['operator_differences'].get('W1024',{}).get('native_basis_lifetime')
    out=dict(complete=z['execution']['complete'],source=dict(path=str(rp.relative_to(HERE)),sha256=sha(rp)),generator_sha256=sha(Path(__file__)),
        actual_counts=dict(production_cells=production,adaptive_anchor_trajectories=anchors,total_trajectories=production+anchors),
        planned_counts=dict(production_cells=108,adaptive_anchor_trajectories=216,total_trajectories=324),
        memory=dict(native_basis_release=release,owning_W_caches=owning,whole_job=z['execution'].get('whole_job_memory',z['execution'].get('checkpoint_memory')),
            interpretation='all peaks are cumulative whole-job counters; release delta is measured within this allocation'),
        masks=masks,critical_time_projection_geometry=geometry,cost_summaries=z.get('cost_summaries',{}),
        scientific_status='independent results audit required; incomplete allocations cannot support publication or cross-job timing')
    if out['complete']:assert out['actual_counts']==out['planned_counts']
    (root/'SUMMARY.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(dict(complete=out['complete'],counts=out['actual_counts'],masks=len(masks))))
if __name__=='__main__':main()
