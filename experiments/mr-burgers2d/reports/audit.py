"""Recompute physical errors from each retained invocation field, without JAX."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run);d=json.loads((run/'out/pilot.json').read_text())
assert d['complete'] and d['backend']=='gpu' and d['x64'] and d['matmul_precision']=='highest'
assert d['checkpoint_sha256']==d['checkpoint_sha256_after']
for line in (run/'COLLECT.sha256').read_text().splitlines():
    h,file=line.split(maxsplit=1);assert hashlib.sha256((run/file).read_bytes()).hexdigest()==h,file
stderr='\n'.join(x.read_text() for x in (run/'logs').glob('*.err'))
assert not stderr.strip(),stderr
stdout='\n'.join(x.read_text() for x in (run/'logs').glob('*.out'))
assert 'jax_backend=gpu' in stdout and 'ALL-DONE' in stdout
ref={}
for i in range(d['config']['cases']):
    name=f"ref_L{d['config']['reference_mesh']}_dt{d['config']['reference_dt']}_case{i}.npz"
    ref[i]=np.load(run/'out'/name)['fields']
worst_difference=0.;counts={};fom_failures=rom_bad=0;initial_budget=0;initial_stalls=0
for r in d['invocations']:
    z=np.load(run/'out'/r['observation_artifact']);f=z['fields'];truth=ref[r['case']]
    e=np.linalg.norm((f-truth).reshape(len(f),-1),axis=1)/np.linalg.norm(truth[0])
    delta=np.max(np.abs(e-r['physical_error']['fixed_initial_per_time']));worst_difference=max(worst_difference,float(delta))
    assert delta<1e-13,(r['name'],r['case'],delta)
    if r['output_intervals']==d['observation_intervals']:
        assert hashlib.sha256(f.tobytes()).hexdigest()==r['field_sha256']
    assert np.array_equal(z['iterations'],r['iterations']) and np.array_equal(z['residuals'],r['residuals'])
    key=(r['name'],r['case']);counts[key]=counts.get(key,0)+1
    if r['method']=='fom':
        good=np.max(z['residuals'])<=r['newton_tolerance']*(1+1e-9)
        assert bool(good)==r['nonlinear_tolerance_satisfied'];fom_failures+=int(not good)
    else:
        initial_budget+=r['ic_reason']==0;initial_stalls+=r['ic_reason']==2
        rom_bad+=sum(x==3 for x in r['stop_reasons'])
assert all(n==d['config']['reps'] for n in counts.values())
assert max(r['max_relative_residual'] for r in d['reference'])<2e-11
out=dict(source_sha256=hashlib.sha256((run/'out/pilot.json').read_bytes()).hexdigest(),
    output_checksums_verified=True,closed_logs_clean=True,reference_solvers_converged=True,
    invocations_verified=len(d['invocations']),physical_error_recompute_max_difference=worst_difference,
    fom_nonlinear_tolerance_failures=fom_failures,rom_rejected_nonfinite_step_stops=rom_bad,
    ic_budget_exits_across_all_rom_invocations=initial_budget,ic_small_step_or_improvement_exits=initial_stalls,
    note='ROM relative-improvement/small-step stopping is not a stationarity certificate; reference refinement qualification is separate.')
(run/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
