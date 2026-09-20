"""Independent NumPy audit of membership, reconstructions and every timed field."""
from __future__ import annotations
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
from audit_comparison import file_sha,array_sha,leaves
from audit_pilot import independently_evaluate_bank,head,rel,stats


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=Path(a.collected);out=root if a.smoke else root/'output';raw=json.loads((out/'result.json').read_text());cfg=raw['config']
    result=dict(passed=True,source_commit=raw.get('source_commit'),job_id=raw.get('job_id'),checks={},scope='all membership bytes and saved development fields; independent model reconstruction and invocation metrics')
    def gate(name,passed,**details):
        result['checks'][name]=dict(passed=bool(passed),**details);result['passed'] &= bool(passed)
        Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(name,json.dumps(result['checks'][name]),flush=True)
    if not a.smoke:
        bad=[];count=0
        for line in (root/'OUTPUTS.sha256').read_text().splitlines():
            expected,name=line.split(None,1);count+=1
            if file_sha(root/name.strip())!=expected:bad.append(name)
        gate('raw_checksums',not bad,count=count,failures=bad)
        stdout=(root/'logs'/f"{raw['job_id']}.out").read_text();stderr=(root/'logs'/f"{raw['job_id']}.err").read_text()
        gate('gpu_provenance',raw['backend']=='gpu' and raw['x64'] and raw['precision']=='highest' and 'jax_backend=gpu' in stdout and 'PILOT_EXIT=0' in stdout and not any(x in (stdout+stderr).lower() for x in ['out of memory','captured-large-constant','resource_exhausted','no space left']),stderr_bytes=len(stderr))
    gate('complete_and_final_sealed',raw['complete'] and not raw['final_cohort_opened'])
    with np.load(out/'train_data.npz') as f:base=f['states'];params=f['parameters']
    with np.load(out/'dev_data.npz') as f:dev=f['states']
    manifest=json.loads((out/'membership.json').read_text());table=np.asarray(manifest['membership'],dtype=np.int64)
    rng=np.random.default_rng(cfg['augmentation_seed']);expected=[]
    for case in range(len(base)):
        offsets=[(0,0,0)]
        while len(offsets)<cfg['augmentation_copies']:
            offset=tuple(int(v) for v in rng.integers(0,cfg['n'],3))
            if offset not in offsets:offsets.append(offset)
        expected.extend((case,*offset) for offset in offsets)
    stream=hashlib.sha256()
    for case,dx,dy,dz in table:
        shifted=np.roll(base[case],(dx,dy,dz),axis=(-3,-2,-1))
        stream.update(np.ascontiguousarray(shifted).tobytes())
    gate('exact_shared_membership',np.array_equal(table,np.asarray(expected)) and array_sha(table)==manifest['membership_sha256'] and array_sha(base)==manifest['base_states_sha256'] and array_sha(params)==manifest['base_parameter_sha256'] and stream.hexdigest()==raw['augmentation']['augmented_states_sha256'],base_cases=len(base),augmented_cases=len(table))
    cap=out/'capacity';screen=json.loads((cap/'screen.json').read_text())
    with (cap/'frozen_bank.pkl').open('rb') as f:bank=pickle.load(f)
    G=bank['extra']['bank'];Q=bank['extra']['qr_Q'];Rb=bank['extra']['qr_R']
    computed=independently_evaluate_bank(bank['params'],cfg['n'])
    gate('learned_coordinate_bank',rel(computed,G)<1e-10,relative=rel(computed,G));del computed
    gate('bank_QR',rel(Q@Rb,G)<1e-10 and rel(Q.T@Q,np.eye(Q.shape[1]))<1e-10)
    singular=np.linalg.svd(Rb,compute_uv=False)
    measured_rank=int(np.sum(singular>singular[0]*1e-10))
    gate('projected_bank_numerical_rank',measured_rank==G.shape[1] and np.allclose(singular,screen['whitening']['singular_values'],rtol=1e-8,atol=1e-12),
        numerical_rank=measured_rank,requested_rank=G.shape[1],relative_threshold=1e-10,condition=float(singular[0]/singular[-1]))
    if cfg.get('reuse_path'):
        reuse=Path(cfg['reuse_path']) if a.smoke else root/'reuse'
        manifest=json.loads((reuse/'REUSE.json').read_text())
        failures=[entry['path'] for entry in manifest['files'] if file_sha(reuse/entry['path'])!=entry['sha256']]
        prior=json.loads((reuse/'result.json').read_text())
        membership_keys=['membership_sha256','base_states_sha256','base_parameter_sha256','augmented_states_sha256']
        same=all(raw['augmentation'][key]==prior['augmentation'][key] for key in membership_keys)
        same &= raw['dev_data']['states_sha256']==prior['dev_data']['states_sha256']
        for spec in cfg['operators']:
            if file_sha(out/spec['kind']/'best.pkl')!=file_sha(reuse/spec['kind']/'best.pkl'):failures.append(spec['kind'])
        gate('same_cohort_operator_checkpoint_reuse',not failures and same and manifest==raw['reuse'],failures=failures,source_job=prior['job_id'])
        with (reuse/'capacity/frozen_bank.pkl').open('rb') as f:previous=pickle.load(f)
        with (cap/'initial_expanded_bank.pkl').open('rb') as f:initial=pickle.load(f)
        initial_G=independently_evaluate_bank(initial['params'],cfg['n']);old_G=previous['extra']['bank'];old_rank=old_G.shape[1]
        difference=rel(initial_G[:,:old_rank],old_G)
        initial_R=np.linalg.qr(initial_G,mode='r');initial_singular=np.linalg.svd(initial_R,compute_uv=False)
        initial_rank=int(np.sum(initial_singular>initial_singular[0]*1e-10));saved=screen['warm_initialization']['initial_whitening']
        rank_agreement=(initial_rank==initial_G.shape[1])==saved['passed']
        if saved['passed']:rank_agreement &= np.allclose(initial_singular,saved['singular_values'],rtol=1e-8,atol=1e-12)
        gate('warm_expansion_physical_columns_and_rank',difference<1e-10 and rank_agreement,
            old_column_relative=difference,old_rank=old_rank,new_rank=initial_G.shape[1],measured_initial_rank=initial_rank,
            initial_condition=float(initial_singular[0]/initial_singular[-1]),initial_full_rank=initial_rank==initial_G.shape[1],
            interpretation='initialization rank is measured; only the final trained full-rank bank is eligible for capacity acceptance')
        del initial_G,initial_R,previous,initial
    X=dev.reshape(-1,G.shape[0]);den=np.repeat(np.linalg.norm(dev[:,0].reshape(len(dev),-1),axis=1),6)
    bank_prediction=(X@Q)@Q.T;floor=np.linalg.norm(X-bank_prediction,axis=1)/den
    with np.load(cap/'bank_fields.npz') as f:
        gate('bank_development_fields',rel(bank_prediction,f['development_prediction'].reshape(X.shape))<1e-10 and np.max(abs(floor-f['development_floor']))<1e-10)
    gate('bank_development_summary',all(abs(screen['development_bank_initial_normalized'][k]-v)<1e-10 for k,v in stats(floor).items()),summary=stats(floor))
    train_floors=[];coefficient_error=0.;train_norm=[];stored_coeff=bank['extra']['train_coefficients']
    for index,(case,dx,dy,dz) in enumerate(table):
        Xtr=np.roll(base[case],(dx,dy,dz),axis=(-3,-2,-1)).reshape(6,-1);scale=np.linalg.norm(Xtr[0])
        rotated=Xtr@Q;fitted=rotated@Q.T;start=6*index
        coefficient_error=max(coefficient_error,rel(stored_coeff[start:start+6]@Rb.T,rotated))
        train_floors.extend(np.linalg.norm(Xtr-fitted,axis=1)/scale);train_norm.extend([scale]*6)
    train_floors=np.asarray(train_floors);train_norm=np.asarray(train_norm)
    with np.load(cap/'bank_fields.npz') as f:train_difference=float(np.max(abs(train_floors-f['train_floor'])))
    gate('all_augmented_training_projections',coefficient_error<1e-10 and train_difference<1e-10,
        snapshots=len(train_floors),maximum_coefficient_relative_error=coefficient_error,maximum_floor_error=train_difference)
    for name,record in screen['heads'].items():
        with (cap/(name+'.pkl')).open('rb') as f:model=pickle.load(f)
        pred=head(model['params'],np.asarray(record['fits']['z']))@G.T
        error=np.linalg.norm(pred-X,axis=1)/den
        with np.load(cap/(name+'_fields.npz')) as f:saved=f['prediction'].reshape(X.shape)
        gate(name+'_development_reconstruction',rel(pred,saved)<1e-10 and np.max(abs(error-np.asarray(record['error_by_case_time']).ravel()))<1e-10,
            summary=stats(error),stationary_count=int(np.sum(np.asarray(record['fits']['reasons'])==4)))
        if 'training_stored_code_initial_normalized' in record:
            coefficient_residual=(head(model['params'],np.asarray(model['codes']))-stored_coeff)@Rb.T
            training_error=np.sqrt(train_floors**2+np.sum(coefficient_residual**2,axis=1)/train_norm**2)
            gate(name+'_stored_training_codes',all(abs(record['training_stored_code_initial_normalized'][k]-v)<1e-10 for k,v in stats(training_error).items()),summary=stats(training_error))
        if not record['configuration']['free_codes']:
            # Independent weighted PCA reconstruction fixes its subspace, while
            # eigenvector signs are immaterial. Verify code Gram/score spectrum.
            Y=bank['extra']['train_coefficients']@Rb.T
            norm=np.repeat(np.linalg.norm(base[table[:,0],0].reshape(len(table),-1),axis=1),6)
            weight=1/norm**2;center=np.sum(weight[:,None]*Y,axis=0)/np.sum(weight)
            _,basis=np.linalg.eigh((Y-center).T@(weight[:,None]*(Y-center)))
            k=record['configuration']['k'];scores=(Y-center)@basis[:,-k:]
            scale=record['initialization']['code_scale'];given=np.asarray(model['codes'])*scale
            # Orthogonal PCA rotations/signs disappear from pairwise Gram probes.
            ids=np.linspace(0,len(given)-1,min(24,len(given)),dtype=int)
            difference=rel(given[ids]@given[ids].T,scores[ids]@scores[ids].T)
            gate(name+'_fixed_PCA_codes',difference<1e-8,relative_Gram_probe=difference,count=len(ids))
    precision_bad=[]
    for spec in cfg['operators']:
        with (out/spec['kind']/'best.pkl').open('rb') as f:model=pickle.load(f)
        precision_bad.extend(str(v.dtype) for v in leaves(model['params']) if v.dtype.kind in 'fc' and v.dtype not in (np.float64,np.complex128))
        gate(spec['kind']+'_shared_recipe',model['spec']==spec and spec['output_residual_initial'])
    gate('operator_precision',not precision_bad,bad=precision_bad)
    with np.load(out/'timing_references.npz') as f:refs=f['same_grid']
    rows=json.loads((out/'timing_rows.json').read_text());cache={};bad=[];worst=0.
    with np.load(out/'timed_fields.npz') as fields:
        for name in fields:
            field=fields[name];parts=name.split('__');case=int(parts[1][4:]);finite=bool(np.isfinite(field).all())
            error=np.linalg.norm((field-refs[case]).reshape(6,-1),axis=1)/np.linalg.norm(refs[case,0]) if finite else None
            cache[(parts[0],case,array_sha(field))]=(error,finite)
    for row in rows:
        key=(row['method'],row['case'],row['field_sha256'])
        if key not in cache:bad.append('unretained invocation');continue
        error,finite=cache[key]
        if finite:
            difference=float(np.max(abs(error-row['same_grid_errors'])));worst=max(worst,difference)
            if difference>1e-10:bad.append('metric mismatch')
        elif row['same_grid_errors'] is not None:bad.append('nonfinite error not labelled')
        if finite!=row['finite'] or not 0<row['gpu_seconds']<=row['with_host_seconds']:bad.append('invocation metadata')
    gate('every_timed_field',not bad,invocations=len(rows),fields=len(cache),max_metric_disagreement=worst,failures=bad)
    result['derived_summary']={}
    for name in raw['method_names']:
        arm=[r for r in rows if r['method']==name];t=np.asarray([r['gpu_seconds'] for r in arm]);finite=[r for r in arm if r['finite']]
        result['derived_summary'][name]=dict(invocations=len(arm),nonfinite_invocations=len(arm)-len(finite),device_ms_median=float(np.median(t)*1000),
            worst_evolved_same_grid=max(max(r['same_grid_errors'][1:]) for r in finite) if len(finite)==len(arm) else None)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
