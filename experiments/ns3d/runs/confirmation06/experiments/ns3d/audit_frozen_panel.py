"""Independent complete-field, reference and frozen-model NS3D panel audit."""
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
from scipy.signal import resample
from audit_pilot import independently_evaluate_bank,head,rel
from audit_comparison import file_sha,array_sha,leaves
import ns3d_independent as I


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=Path(a.collected);out=root if a.smoke else root/'output';raw=json.loads((out/'result.json').read_text());cfg=raw['config']
    assets=out/'assets' if cfg['evaluation_cohort']=='development' else root/cfg['frozen_source_directory']/'assets'
    result=dict(passed=True,checks={},source_commit=raw['source_commit'],job_id=raw['job_id'],scope=__doc__)
    def gate(name,passed,**values):
        result['checks'][name]=dict(passed=bool(passed),**values);result['passed'] &= bool(passed)
        Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(name,json.dumps(result['checks'][name]),flush=True)
    if not a.smoke:
        failures=[];count=0
        for line in (root/'OUTPUTS.sha256').read_text().splitlines():
            expected,name=line.split(None,1);count+=1
            if file_sha(root/name.strip())!=expected:failures.append(name)
        gate('all_output_checksums',not failures,files=count,failures=failures)
        stdout=(root/'logs'/f"{raw['job_id']}.out").read_text();stderr=(root/'logs'/f"{raw['job_id']}.err").read_text()
        gate('gpu_provenance',raw['backend']=='gpu' and raw['x64'] and raw['precision']=='highest' and 'jax_backend=gpu' in stdout and 'PILOT_EXIT=0' in stdout and not any(x in (stdout+stderr).lower() for x in ['out of memory','captured-large-constant','resource_exhausted','no space left']),stderr_bytes=len(stderr))
    gate('complete_cohort_record',raw['complete'] and raw['cohort_count']==cfg['timed_cases'] and raw['final_cohort_opened']==(cfg['evaluation_cohort']=='final'))
    if cfg['evaluation_cohort']=='final':
        frozen=raw['freeze'];gate('final_freeze_and_real_development_replay',raw['verified_before_final_parameter_generation'] and raw['development_replay']['passed'] and frozen['final_fields_generated'] is False and raw['cohort_seed']==202609203 and raw['cohort_count']==32)
    manifest=json.loads((assets/'manifest.json').read_text());bad=[name for name,value in manifest['files'].items() if file_sha(assets/name)!=value]
    gate('all_frozen_asset_hashes',not bad,files=len(manifest['files']),failures=bad)
    with (assets/'frozen_bank.pkl').open('rb') as f:bank=pickle.load(f)
    with (assets/'head.pkl').open('rb') as f:hp=pickle.load(f)
    bad_dtype=[str(x.dtype) for model in (bank,hp) for x in leaves(model) if x.dtype.kind in 'fc' and x.dtype not in (np.float64,np.complex128)]
    for spec in cfg['operators']:
        with (assets/spec['kind']/'best.pkl').open('rb') as f:model=pickle.load(f)
        bad_dtype.extend(str(x.dtype) for x in leaves(model) if x.dtype.kind in 'fc' and x.dtype not in (np.float64,np.complex128))
    gate('all_model_parameter_precision',not bad_dtype,bad_dtypes=bad_dtype)
    G=bank['extra']['bank'];Q=bank['extra']['qr_Q'];Rb=bank['extra']['qr_R'];computed=independently_evaluate_bank(bank['params'],cfg['n'])
    gate('independent_coordinate_bank',rel(computed,G)<1e-10,relative=rel(computed,G));del computed
    gate('bank_whitening',rel(Q@Rb,G)<1e-10 and rel(Q.T@Q,np.eye(cfg['r']))<1e-10)
    with np.load(assets/'dense_weak_operators.npz') as f:ops={key:f[key] for key in f};C=ops['C']
    metric=Rb@C;gate('correction_physical_orthogonality',np.max(abs(metric.T@metric-np.eye(cfg['r'])))<1e-7)
    with np.load(assets/'pod.npz') as f:P=f['basis']
    gate('POD_orthogonality',np.max(abs(P.T@P-np.eye(P.shape[1])))<1e-7)
    ids=json.loads((assets/'dense_weak_test_modes.json').read_text());waves=np.asarray([x['wave'] for x in ids]);index=waves%cfg['n'];pol=np.asarray([x['polarization'] for x in ids]);cosine=np.asarray([x['kind']=='cos' for x in ids]);spec=I.setup(cfg['n'])
    operator_errors=[];probe_rng=np.random.default_rng(202609321)
    for basis,A,L in ((G,ops['A'],None),(P,ops['pod_A'],ops['pod_L']),(Q,None,ops['free_L'])):
        for _ in range(8):
            c=probe_rng.normal(size=basis.shape[1]);field=(basis@c).reshape(3,cfg['n'],cfg['n'],cfg['n']);spectrum=I.transform(field)
            if A is not None:
                value=spectrum[:,index[:,0],index[:,1],index[:,2]];scalar=np.einsum('cm,mc->m',value,pol)
                expected=np.sqrt(2*cfg['n']**3)*np.where(cosine,scalar.real,-scalar.imag);operator_errors.append(rel(A@c,expected))
            if L is not None:
                expected=basis.T@I.physical(-spec[1]*spectrum).ravel();operator_errors.append(rel(L@c,expected))
    gate('independent_weak_and_Galerkin_linear_probes',max(operator_errors)<1e-10 and np.allclose(ops['lam'],4*np.pi**2*np.sum(waves*waves,axis=1),rtol=1e-13,atol=1e-13),
        maximum_relative=max(operator_errors),checks=len(operator_errors),seed=202609321,scope='Eight random full-coefficient probes per saved linear map, and every Fourier Laplacian eigenvalue')
    with np.load(out/'dev_data.npz') as f:states=f['states'];parameters=f['parameters']
    rng=np.random.default_rng(raw['cohort_seed']);draws=[]
    for _ in range(len(states)):
        center=rng.uniform(0,1,3);draws.append([*center,rng.uniform(.12,.24),rng.uniform(.6,1.4),np.exp(rng.uniform(np.log(.002),np.log(.01)))])
    gate('independent_parameter_draw_and_membership',np.array_equal(parameters,np.asarray(draws)) and raw['dev_data']['states_sha256']==array_sha(states),seed=raw['cohort_seed'],cases=len(states))
    with np.load(out/'timing_references.npz') as f:refs=f['same_grid'];fine=f['fine']
    refinement_errors=[]
    for case in range(len(states)):
        with np.load(out/'refinement'/f'case{case}.npz') as f:larger=f['fine']
        lifted=fine[case]
        for axis in (-3,-2,-1):lifted=resample(lifted,cfg['refinement_n'],axis=axis)
        error=np.linalg.norm((lifted-larger).reshape(6,-1),axis=1)/np.linalg.norm(larger[0]);record=raw['reference_refinement'][case]
        refinement_errors.append(float(np.max(abs(error-record['errors']))))
        assert array_sha(larger)==record['fine_sha256']
    gate('all_reference_refinement_metrics',max(refinement_errors)<1e-10,maximum_disagreement=max(refinement_errors),failed_refinement_cases=sum(not x['passed'] for x in raw['reference_refinement']))
    rows=json.loads((out/'timing_rows.json').read_text());cache={};bad=[];worst=0.;fields_count=0
    with np.load(out/'timed_fields.npz') as fields:
        for name in fields:
            field=fields[name];fields_count+=1;parts=name.split('__');case=int(parts[1][4:]);finite=bool(np.isfinite(field).all());error=physical=None
            if finite:
                error=np.linalg.norm((field-refs[case]).reshape(6,-1),axis=1)/np.linalg.norm(states[case,0]);lifted=field
                for axis in (-3,-2,-1):lifted=resample(lifted,cfg['fine_reference_n'],axis=axis)
                physical=np.linalg.norm((lifted-fine[case]).reshape(6,-1),axis=1)/np.linalg.norm(fine[case,0])
                if not np.isfinite(error).all() or not np.isfinite(physical).all():finite=False;error=None;physical=None
            cache[(parts[0],case,array_sha(field))]=(error,physical,finite)
    for row in rows:
        key=(row['method'],row['case'],row['field_sha256'])
        if key not in cache:bad.append('missing saved invocation');continue
        error,physical,finite=cache[key]
        if finite:
            difference=max(float(np.max(abs(error-row['same_grid_errors']))),float(np.max(abs(physical-row['physical_errors']))));worst=max(worst,difference)
            if difference>1e-10:bad.append('metric mismatch')
        elif row['same_grid_errors'] is not None or row['physical_errors'] is not None:bad.append('nonfinite field not labelled')
        if finite!=row['finite'] or not 0<row['gpu_seconds']<=row['with_host_seconds']:bad.append('paired invocation metadata')
    gate('every_timed_field_and_paired_cost',not bad,invocations=len(rows),fields=fields_count,maximum_metric_disagreement=worst,failures=bad)
    expected={(method,case,repetition) for method in raw['method_names'] for case in range(cfg['timed_cases']) for repetition in range(cfg['repetitions'])}
    observed=[(row['method'],row['case'],row['repetition']) for row in rows]
    gate('full_case_method_repetition_coverage',len(observed)==len(expected) and set(observed)==expected,expected=len(expected))
    stopping_bad=[];nonstationary=0
    for row in rows:
        if row['kind']!='weak':continue
        reason=np.asarray([row['cold'][2],*row['steps'][2]],dtype=float);gradient=np.asarray([row['cold'][3],*row['steps'][3]],dtype=float)
        nonstationary+=int(np.sum(reason!=4))
        if np.any((reason==4)&((gradient>cfg['gtol']+1e-10)|~np.isfinite(gradient))):stopping_bad.append((row['method'],row['case'],row['repetition']))
    gate('recorded_stopping_consistency',not stopping_bad,nonstationary_solves=nonstationary,failures=stopping_bad,qualification='Independent residual/Jacobian probes are reported separately')
    probes=sorted({0,len(states)//2,len(states)-1});reference_disagreement=[]
    for case in probes:
        prediction=I.solve_cnab2(states[case,0],parameters[case,-1],cfg['reference_dt'],round(cfg['horizon']/cfg['reference_dt']),round(cfg['horizon']/cfg['reference_dt'])//5)
        reference_disagreement.append(rel(prediction,refs[case]))
    gate('independent_advective_form_reference_probes',max(reference_disagreement)<1e-9,cases=probes,maximum_relative=max(reference_disagreement))
    if not a.smoke:gate('all_four_operator_families',all(any(name.startswith(kind+'_') for name in raw['method_names']) for kind in ('fno3d','unet3d','deeponet3d','transolver3d')))
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
