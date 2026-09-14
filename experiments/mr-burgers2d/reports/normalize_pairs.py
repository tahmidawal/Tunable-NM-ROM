"""Export primary predeclared fixed-settings pairs for the coordinator accounting audit.

Reference refinement is empirical, so the strict shared uncertainty-bound field
stays null. Native tables may separately discuss empirical-margin pilot selection.
"""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run);src=run/'out/pilot.json';d=json.loads(src.read_text())
for L in map(int,d['config']['meshes'].split(',')):
    rom=f'rom_L{L}_dt0.005_stall0.001'+('_starts1' if 'ic_starts' in d['config'] else '')
    fom=f'fom_L{L}_out{L}_dt0.005_ntol0.003'
    contract=f'burgers_dense_host_L{L}_outputs_'+','.join(map(str,d['output_times']))
    selected=[r for r in d['invocations'] if r['name'] in [rom,fom]]
    out=dict(schema_version=1,stage='development',reference_kind='refined_physical',configuration_selection='predeclared',
        source_artifact_sha256={'out/pilot.json':hashlib.sha256(src.read_bytes()).hexdigest()},
        provenance=dict(backend=d['backend'],x64=d['x64'],matmul_precision=d['matmul_precision'],job_id=d['job_id'],
            gpu_identity=d['gpu']+' / single Slurm GPU allocation '+d['job_id'],source_commit=d['commit']),
        case_ids=[str(x) for x in range(d['config']['cases'])],repetitions=d['config']['reps'],
        required_error_metrics=['max_observation_error_initial_norm'],output_contract_id=contract,
        reference_uncertainty_bound=None,empirical_reference_difference_max=max(r['conservative_difference_sum'] for r in d['reference_uncertainty']),
        reference_uncertainty_fraction=.1,targets=[.1,.05,.01,.001],independent_final_cohort=False,invocations=[])
    for r in selected:
        uid=f"{d['job_id']}_{r['name']}_{r['case']}_{r['rep']}"
        valid=r['finite'] and (r['nonlinear_tolerance_satisfied'] if r['method']=='fom' else r['ic_reason']!=3 and 3 not in r['stop_reasons'])
        out['invocations'].append(dict(case_id=str(r['case']),repeat=r['rep'],method='nmrom' if r['method']=='rom' else 'fom',
            invocation_id=uid,metric_invocation_id=uid,job_id=d['job_id'],gpu_identity=out['provenance']['gpu_identity'],
            output_contract_id=contract,query_includes_input_and_output=True,query_seconds=r['seconds'],
            warmup_and_burnin_complete=True,device_synchronized=True,completed=True,numerically_valid=bool(valid),
            errors={'max_observation_error_initial_norm':r['physical_error']['fixed_initial_max'] if r['physical_error'] else None},
            native_stop_reasons=r.get('stop_reasons'),native_ic_reason=r.get('ic_reason'),
            budget_policy='Finite predeclared budget/improvement exits are early-stopped approximations; physical errors are retained, not stationary-fit claims.'))
    assert len(selected)==2*d['config']['cases']*d['config']['reps']
    path=run/f'PAIRED_L{L}.json';path.write_text(json.dumps(out,indent=2)+'\n');print(path)
