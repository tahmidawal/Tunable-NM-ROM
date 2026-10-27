"""Conservative job-planning proxy from measured diagnostic wall times; never a speed comparison."""
from pathlib import Path
import hashlib
import json
HERE=Path(__file__).resolve().parent
def main():
    cfg=json.loads((HERE/'configs/front64.json').read_text());ledger=json.loads((HERE/'JOB-LEDGER.json').read_text())
    entries={e['attempt']:e for e in ledger['entries']}
    tp=HERE/'results/tdiag02/raw/output/result.json';t=json.loads(tp.read_text())
    actual_weak_steps=sum(round(.25/float(k.split('|')[3])) for k,v in t['runs'].items() if not v.get('reused'))
    new=[s for s in cfg['subjects'] if s.get('successors')];step_sum=sum(round(.25/s['dt']) for s in new)
    weak_once=(3*64+64+len(cfg['finer_check_cases']))*step_sum
    weak_proxy=entries['tdiag02']['elapsed_seconds']*weak_once/actual_weak_steps
    pp=HERE/'results/podtune02/raw/output/result.json';p=json.loads(pp.read_text())
    pilot_native_steps=2*sum(v['steps'] for v in p['runs'].values())
    final_native_steps=2*64*sum(s['steps'] for s in cfg['subjects'] if s['kind']=='POD')
    pod_proxy=entries['podtune02']['elapsed_seconds']*final_native_steps/pilot_native_steps
    ds=json.loads((HERE/'results/sdiag01/raw/output/result.json').read_text())
    observed=max(r['seconds'] for k,trips in ds['timing'].items() if k.startswith('W1024|refresh|') for trip in trips for r in trip if r['arm']=='refresh')
    plans={}
    for reps in [cfg['timing_reps'],cfg['max_timing_reps']]:
        extra_weak=(reps-1)*64*sum(observed*(56/40)**3*.025/s['dt'] for s in new)
        burn=len(cfg['subjects'])*64*reps*3*cfg['burn_seconds']
        other=900*reps/3  # explicit allowance for other warmed queries and anchor calls, not an observed cross-job ratio
        setup_serialization=600
        plans[reps]=dict(weak_control_proxy_seconds=weak_proxy,POD_validation_proxy_seconds=pod_proxy,
            additional_new_bank_query_allowance_seconds=extra_weak,burn_seconds=burn,other_query_allowance_seconds=other,
            setup_serialization_allowance_seconds=setup_serialization,total_gpu_hours=(weak_proxy+pod_proxy+extra_weak+burn+other+setup_serialization)/3600)
    spent=sum(e.get('gpu_hours',0) for e in ledger['entries'])
    out=dict(scope='planning proxy only; measured difficult-case controls include setup/compilation and overlapping query/certification costs, not a prediction or speed comparison',
        config_sha256=hashlib.sha256((HERE/'configs/front64.json').read_bytes()).hexdigest(),ordered_subjects=[s['id'] for s in cfg['subjects']],
        measured_sources=dict(time=dict(path=str(tp.relative_to(HERE)),sha256=hashlib.sha256(tp.read_bytes()).hexdigest(),Slurm_seconds=entries['tdiag02']['elapsed_seconds'],new_steps=actual_weak_steps),
            POD=dict(path=str(pp.relative_to(HERE)),sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),Slurm_seconds=entries['podtune02']['elapsed_seconds'],native_steps=pilot_native_steps)),
        proxies_by_max_reps=plans,spent_before_job_gpu_hours=spent,hard_final_gpu_hours=3,planned_mesh_hard_gpu_hours=1,lane_cap_gpu_hours=8,
        reserved_total_gpu_hours=spent+3+1,device_peak_limit_bytes=cfg['memory_peak_limit_bytes'],
        memory_projection='Measured D-T86.14GB plus shared POD/M2 tables and streamed rule changes;120GB late checkpoint cap. Peak attribution remains whole job.')
    assert out['reserved_total_gpu_hours']<=8
    (HERE/'FRONT-BUDGET.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out['proxies_by_max_reps']))
if __name__=='__main__':main()
