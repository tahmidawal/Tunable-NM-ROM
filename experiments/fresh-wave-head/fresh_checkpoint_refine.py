"""Numerical continuation of frozen checkpoints whose original refinement failed.

No training, new fitting, data-family change or original-verdict replacement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from fresh_fom import Grid,localized_initial,integrate_balance,energy,damping_ratio,provenance
from fresh_models import tree_from_npz
from fresh_learning import parameter_rows
from fresh_evaluate import nonlinear_rollouts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def relative_max(actual,expected):
    return float(np.max(abs(actual-expected))/max(float(np.max(abs(expected))),1e-30))


def completed(store,prefix):
    fields=('coefficients','physical_velocity_coefficients','displacement_error','velocity_error','energy_state_error','rom_energy')
    return bool(np.all(store[prefix+'_completed'])) and all(prefix+'_'+name in store and np.all(np.isfinite(store[prefix+'_'+name])) for name in fields)


def clean_json(value):
    if isinstance(value,dict):
        return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value,np.generic):
        return clean_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def regenerate_validation(config,grid,manifest,bank,reconstruction):
    expected=manifest['splits']['validation']
    pars=parameter_rows(config['validation_seed'],config['validation_count'])
    np.testing.assert_array_equal(pars,np.asarray(expected['parameters']))
    stride=int(np.ceil(config['observation_dt']/(config['fom_cfl']*grid.h/1.15)))
    dt=config['observation_dt']/stride
    if dt!=expected['dt']:
        raise RuntimeError('Changed FOM time step')
    observations=int(round(config['end_time']/config['observation_dt']))
    us,vs,scales,energies,audits=[],[],[],[],[]
    for i,p in enumerate(pars):
        u0,v0=localized_initial(grid,p)
        u,v,flux=integrate_balance(u0,v0,p[5],dt,grid=grid,steps=observations*stride,stride=stride)
        ee=np.asarray(energy(u,v,grid,p[5]))
        u,v,flux=np.asarray(u),np.asarray(v),np.asarray(flux)
        if not all(np.all(np.isfinite(x)) for x in (u,v,ee,flux)):
            raise RuntimeError('Nonfinite regenerated validation truth')
        e0=float(ee[0])
        uscale=float(np.sqrt(np.sum(grid.mass()*np.asarray(u0)**2)))
        if not np.isfinite(e0) or e0<=0 or not np.isfinite(uscale) or uscale<=0:
            raise RuntimeError('Invalid regenerated initial scales')
        balance=float(np.max(abs(ee+flux-e0))/e0)
        invariant_drift=0.
        if grid.bx=='absorbing':
            invariant=np.sum(grid.mass()*(v+np.asarray(damping_ratio(grid,p[5]))*u),axis=(-2,-1))
            invariant_drift=float(np.max(abs(invariant-invariant[0])))
        old_audit=expected['truth_audits'][i]
        energy_parity=relative_max(ee,np.asarray(old_audit['energy_trace']))
        if balance>=1e-5 or invariant_drift>=1e-10 or energy_parity>1e-10:
            raise RuntimeError('Regenerated FOM balance/invariant/energy parity failed')
        audits.append({'case':i,'max_balance_relative':balance,'invariant_drift':invariant_drift,'energy_trace_relative_parity':energy_parity})
        us.append(u.reshape(observations+1,-1));vs.append(v.reshape(observations+1,-1))
        scales.append((uscale,np.sqrt(2*e0)));energies.append(e0)
    data={'parameters':pars,'u':np.stack(us),'v':np.stack(vs),'scales':np.asarray(scales),'initial_energy':np.asarray(energies),'dt':dt}
    parity={'u_sha256':array_sha(data['u']),'v_sha256':array_sha(data['v']),'original_u_sha256':expected['u_sha256'],'original_v_sha256':expected['v_sha256'],'scales_relative_parity':relative_max(data['scales'],np.asarray(expected['scales'])),'audits':audits}
    parity['bit_identical_truth']=parity['u_sha256']==parity['original_u_sha256'] and parity['v_sha256']==parity['original_v_sha256']
    gmass=bank['mass'][:,None]*bank['g']
    a=(data['u']@gmass).reshape(-1,config['rank'])
    b=(data['v']@gmass).reshape(a.shape)
    parity['projected_displacement_relative_parity']=relative_max(a,reconstruction['projected_truth'])
    parity['projected_velocity_relative_parity']=relative_max(b,reconstruction['projected_velocity'])
    if max(parity['scales_relative_parity'],parity['projected_displacement_relative_parity'],parity['projected_velocity_relative_parity'])>1e-10:
        raise RuntimeError('Regenerated initial scales/projected truth parity failed')
    return data,parity


def difference(a,b,aa,bb,bank,c,e0,uscale,vscale):
    da,db=a-aa,b-bb
    du=np.linalg.norm(da,axis=-1)/uscale
    dv=np.linalg.norm(db,axis=-1)/vscale
    de=np.sqrt(np.maximum(0.,(np.sum(db*db,axis=-1)+np.einsum('tr,rs,ts->t',da,c*c*bank['stiffness_unit'],da))/(2*e0)))
    return du,dv,de


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--inputs',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--kind',choices=('mlp','quadratic'),required=True)
    ap.add_argument('--bc',choices=('reflective','absorbing'),required=True)
    ap.add_argument('--seed',type=int,required=True)
    args=ap.parse_args()
    directory=Path(__file__).resolve().parent
    frozen=json.loads((directory/'FROZEN-MATH.json').read_text())
    for name,expected in frozen['sha256'].items():
        if sha(directory/name)!=expected:
            raise RuntimeError('Frozen mathematical source changed: '+name)
    inputs=Path(args.inputs)
    required=('bank_tables.npz','head.npz','reconstruction.npz','rollouts.npz','data_manifest.json','arm_result.json')
    hashes={name:sha(inputs/name) for name in required}
    original_result=json.loads((inputs/'arm_result.json').read_text())
    if original_result['kind']!=args.kind or original_result['optimizer_seed']!=args.seed or original_result['rollout']['refinement_passed']:
        raise RuntimeError('Wrong arm metadata or original refinement already passed')
    origin=None
    if (inputs/'ORIGIN.json').exists():
        origin=json.loads((inputs/'ORIGIN.json').read_text())
        hashes['ORIGIN.json']=sha(inputs/'ORIGIN.json')
    original_config=json.loads(Path(args.config).read_text())
    meta=provenance()
    if meta['jax_backend']!='gpu' or not meta['x64'] or meta['matmul_precision']!='highest':
        raise RuntimeError('GPU/f64/highest required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    grid=Grid(original_config['n'],'dirichlet' if args.bc=='reflective' else 'absorbing','dirichlet' if args.bc=='reflective' else 'absorbing')
    manifest=json.loads((inputs/'data_manifest.json').read_text())
    if manifest['bc']!=grid.bx or manifest['n']!=grid.n or manifest['final_test_opened']:
        raise RuntimeError('Wrong checkpoint boundary/mesh/cohort')
    with np.load(inputs/'bank_tables.npz') as data:
        bank={name:data[name] for name in ('g','mass','stiffness_unit','damping_unit')}
    np.testing.assert_array_equal(bank['mass'],grid.mass().ravel())
    restored=tree_from_npz(inputs/'head.npz')
    p,f=restored['p'],restored['frozen']
    with np.load(inputs/'reconstruction.npz') as data:
        rec={name:data[name] for name in data.files}
    count=original_config['validation_count']
    observations=int(round(original_config['end_time']/original_config['observation_dt']))+1
    zs=rec['fitted_z'].reshape(count,observations,original_config['latent'])
    ws=rec['tangent_w'].reshape(zs.shape)
    with np.load(inputs/'rollouts.npz') as data:
        original={name:data[name] for name in data.files}
    for i in range(count):
        for dt in original_config['rom_dts']:
            np.testing.assert_array_equal(zs[i,0],original[f'dt{dt}_case{i}_z'][0])
            np.testing.assert_array_equal(ws[i,0],original[f'dt{dt}_case{i}_w'][0])
    data,truth_parity=regenerate_validation(original_config,grid,manifest,bank,rec)
    old_refinements=[]
    for i,par in enumerate(data['parameters']):
        coarse,fine=f'dt0.0025_case{i}',f'dt0.00125_case{i}'
        pair_completed=completed(original,coarse) and completed(original,fine)
        row={'case':i,'both_completed':pair_completed,'passed':False}
        if pair_completed:
            errors=difference(original[coarse+'_coefficients'],original[coarse+'_physical_velocity_coefficients'],original[fine+'_coefficients'],original[fine+'_physical_velocity_coefficients'],bank,par[5],data['initial_energy'][i],*data['scales'][i])
            row.update(max_displacement_difference=float(errors[0].max()),max_velocity_difference=float(errors[1].max()),max_energy_state_difference=float(errors[2].max()),passed=bool(max(x.max() for x in errors)<=original_config['rom_refinement_target']))
        old_refinements.append(row)
    config={**original_config,'rom_dts':[.00125,.000625,.0003125],'rom_primary_dt':.000625}
    run=nonlinear_rollouts(config,grid,bank,data,p,f,zs,ws,args.kind,out)
    # The helper's index1 primary is diagnostic only; preserve original .0025.
    run['diagnostic_middle_dt']=run.pop('primary_dt')
    run['selection_rule']='Separate numerical continuation; original primary step and verdict are preserved.'
    with np.load(out/'rollouts.npz') as store:
        refined={name:store[name] for name in store.files}
    old_fine_parity,adjacent,raw_differences=[],[],{}
    for i,par in enumerate(data['parameters']):
        old=f'dt0.00125_case{i}'
        old_completed=completed(original,old)
        new_completed=completed(refined,old)
        parity={'case':i,'original_completed':old_completed,'repeat_completed':new_completed,'passed':False}
        if old_completed==new_completed and not old_completed:
            parity['passed']=bool(np.array_equal(original[old+'_completed'],refined[old+'_completed']))
        elif old_completed and new_completed:
            errors=difference(original[old+'_coefficients'],original[old+'_physical_velocity_coefficients'],refined[old+'_coefficients'],refined[old+'_physical_velocity_coefficients'],bank,par[5],data['initial_energy'][i],*data['scales'][i])
            parity.update(max_displacement_difference=float(errors[0].max()),max_velocity_difference=float(errors[1].max()),max_energy_state_difference=float(errors[2].max()),passed=bool(max(x.max() for x in errors)<=1e-8))
        old_fine_parity.append(parity)
        for first,second in ((.00125,.000625),(.000625,.0003125)):
            ka,kb=f'dt{first}_case{i}',f'dt{second}_case{i}'
            complete=completed(refined,ka) and completed(refined,kb)
            row={'case':i,'coarse_dt':first,'fine_dt':second,'both_completed':complete,'passed':False}
            if complete and ka+'_coefficients' in refined and kb+'_coefficients' in refined:
                errors=difference(refined[ka+'_coefficients'],refined[ka+'_physical_velocity_coefficients'],refined[kb+'_coefficients'],refined[kb+'_physical_velocity_coefficients'],bank,par[5],data['initial_energy'][i],*data['scales'][i])
                row.update(max_displacement_difference=float(errors[0].max()),max_velocity_difference=float(errors[1].max()),max_energy_state_difference=float(errors[2].max()),passed=bool(max(x.max() for x in errors)<=original_config['rom_refinement_target']))
                for label,value in zip(('displacement','velocity','energy_state'),errors):
                    raw_differences[f'case{i}_{first}_{second}_{label}']=value
            adjacent.append(row)
    contraction=[]
    for i in range(count):
        pair=[r for r in adjacent if r['case']==i]
        if all(r['both_completed'] and 'max_energy_state_difference' in r for r in pair):
            ratio=pair[1]['max_energy_state_difference']/max(pair[0]['max_energy_state_difference'],1e-30)
            contraction.append({'case':i,'energy_difference_ratio':ratio,'observed_order':float(-np.log2(max(ratio,1e-300)))})
    np.savez_compressed(out/'adjacent_differences.npz',**raw_differences)
    result={'provenance':meta,'status':'separate_frozen_checkpoint_numerical_continuation','name':original_result['name'],'kind':args.kind,'boundary':args.bc,'optimizer_seed':args.seed,'input_sha256':hashes,'input_origin':origin,'frozen_mathematics':frozen,'original_config':original_config,'diagnostic_config':config,'original_primary_dt':original_config['rom_primary_dt'],'original_accuracy_passed':original_result['accuracy_passed'],'original_refinement':old_refinements,'original_refinement_passed':original_result['rollout']['refinement_passed'],'truth_parity':truth_parity,'old_fine_physical_parity':old_fine_parity,'old_fine_parity_passed':all(x['passed'] for x in old_fine_parity),'adjacent_refinement':adjacent,'observed_contraction':contraction,'fine_diagnostic':run,'final_test_opened':False,'retraining_performed':False,'initial_fitting_performed':False}
    (out/'result.json').write_text(json.dumps(clean_json(result),indent=2,allow_nan=False)+'\n')
    if not result['old_fine_parity_passed']:
        raise SystemExit('Old-fine physical parity failed; finer diagnostics are not promoted')
    print('fresh_checkpoint_refinement_complete',flush=True)


if __name__=='__main__':
    main()
