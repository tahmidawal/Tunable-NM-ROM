"""Audit exact fixed-seed membership separately from platform exp rounding."""
import argparse,decimal,hashlib,json,subprocess
from pathlib import Path
import numpy as np


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.run)
    local=json.loads((run/'parameter_probe_local.json').read_text());cluster=json.loads((run/'parameter_probe_cluster.json').read_text())
    raw=json.loads((run/'collected/output/result.json').read_text())
    with np.load(run/'collected/output/dev_data.npz') as f: actual=f['parameters']
    probe=Path('experiments/ns3d/audit_parameter_runtime_probe.py')
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(probe)],text=True).strip()
    blob=subprocess.check_output(['git','show',commit+':'+str(probe)]);assert blob==probe.read_bytes()
    independent=json.loads(subprocess.check_output(['/home/tahmid/Dev/.venv/bin/python',str(probe)],text=True))
    checks={}
    checks['independent_local_probe_reproduction']=local==independent
    checks['fixed_seed_and_count']=all(x['seed']==202609203 and x['count']==32 for x in (local,cluster)) and raw['cohort_seed']==202609203 and raw['cohort_count']==32
    checks['exact_uniform_draws_and_log_bounds']=np.array_equal(local['log_bounds'],cluster['log_bounds']) and np.array_equal(local['exponents'],cluster['exponents']) and np.array_equal(np.asarray(local['parameters'])[:,:5],np.asarray(cluster['parameters'])[:,:5])
    checks['exact_cluster_parameter_reproduction']=np.array_equal(actual,np.asarray(cluster['parameters'],dtype=np.float64))
    actual_hash=hashlib.sha256(actual.tobytes()).hexdigest()
    checks['recorded_parameter_hash']=actual_hash==cluster['parameter_sha256']==raw['dev_data']['parameter_sha256']
    difference=actual-np.asarray(local['parameters']);indices=np.argwhere(difference!=0);rows=[]
    with decimal.localcontext() as ctx:
        ctx.prec=100
        for i,j in indices:
            assert j==5,'Only the scalar exponential may differ'
            rounded=float(decimal.Decimal.from_float(local['exponents'][i]).exp())
            rows.append(dict(case=int(i),component=int(j),recorded=float(actual[i,j]),local=float(local['parameters'][i][j]),
                signed_difference=float(difference[i,j]),correctly_rounded_decimal_exp=rounded,
                cluster_minus_correctly_rounded_ulps=int(actual[i,j].view(np.int64)-np.float64(rounded).view(np.int64))))
    result=dict(passed=all(checks.values()),complete=True,scope=__doc__,checks=checks,
        acceptance='Exact independent cluster-runtime parameter replay and recorded hash, plus bitwise identical independent RNG draws and log/exponent inputs; no floating-point acceptance tolerance is introduced',
        strict_local_bitwise_passed=np.array_equal(actual,np.asarray(local['parameters'])),
        qualification='The original strict local parameter gate is retained as failed. Two NumPy CPU implementations round scalar exp differently; all original data and final query results are retained unchanged.',
        differences=rows,parameter_sha256=actual_hash,probe_source_commit=commit,probe_source_sha256=hashlib.sha256(blob).hexdigest(),
        probe_files_sha256={name:digest(run/name) for name in ('parameter_probe_local.json','parameter_probe_cluster.json')},
        runtimes={name:{k:x[k] for k in ('platform','machine','python','numpy')} for name,x in (('local',local),('cluster',cluster))})
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__': main()
