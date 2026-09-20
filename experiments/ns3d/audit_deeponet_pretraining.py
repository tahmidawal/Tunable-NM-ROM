"""NumPy audit of train-only approximate teacher and vector-aware branch loss."""
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024**2):h.update(block)
    return h.hexdigest()


INPUT_FILES=('train_data.npz','confirmation_statistics.pkl','membership.json',
    'deeponet_candidate/pretraining/pretraining.json',
    'deeponet_candidate/pretraining/training_teacher.npz',
    'deeponet_candidate/pretraining/branch_teacher.npz',
    'deeponet_candidate/pretraining/branch_selected_coefficients.npz',
    'deeponet_candidate/pretraining/trunk_selected.pkl')


def audit(pretraining,cases,points,components=3):
    pretraining=Path(pretraining);info=json.loads((pretraining/'pretraining.json').read_text());spec=info['spec'];rank=spec['rank']
    with np.load(pretraining/'training_teacher.npz') as f:basis=f['spatial_basis'];values=f['eigenvalues'];dn=f['denominators'];teacher_error=f['projection_errors']
    with np.load(pretraining/'branch_teacher.npz') as f:target=f['coefficients'];gram=f['gram'];learned_error=f['learned_trunk_projection_errors']
    with np.load(pretraining/'branch_selected_coefficients.npz') as f:prediction=f['coefficients']
    with (pretraining/'trunk_selected.pkl').open('rb') as f:model=pickle.load(f)['params']
    n=round(points**(1/3));axis=np.arange(n)/n;xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(points,3)
    features=[xyz]
    for frequency in spec.get('trunk_frequencies',[1.,2.,4.]):features.extend((np.sin(np.pi*frequency*xyz),np.cos(np.pi*frequency*xyz)))
    trunk=np.concatenate(features,axis=-1)
    for layer in model['trunk'][:-1]:trunk=np.tanh(trunk@layer['w']+layer['b'])
    trunk=trunk@model['trunk'][-1]['w']+model['trunk'][-1]['b'];matrix=trunk/np.sqrt(rank)
    trunk_loss=float(np.mean(np.sum((trunk-basis*np.sqrt(points))**2*(values/np.sum(values)),axis=-1)))
    covariance=np.zeros((rank,rank));h=hashlib.sha256();maximum_teacher=0.;maximum_learned=0.;maximum_stationarity=0.;maximum_denominator=0.;offset=0;branch_errors=[];actual_teacher=[];actual_learned=[]
    for case,(channels_last,denominator) in enumerate(cases):
        h.update(np.ascontiguousarray(channels_last).tobytes());snapshot=np.moveaxis(channels_last,-1,0).reshape(-1,points);channels=len(snapshot);den=np.repeat(denominator,components)
        maximum_denominator=max(maximum_denominator,float(np.max(abs(dn[offset:offset+channels]-den))))
        scores=snapshot@basis;residual=snapshot-scores@basis.T;te=np.sqrt(np.sum(residual*residual,axis=1)/den)
        maximum_teacher=max(maximum_teacher,float(np.max(abs(te-teacher_error[offset:offset+channels]))));covariance+=(scores/np.sqrt(den[:,None])).T@(scores/np.sqrt(den[:,None]))
        residual=target[case]@matrix.T+model['bias'][:,None]-snapshot;le=np.sqrt(np.sum(residual*residual,axis=1)/den)
        maximum_learned=max(maximum_learned,float(np.max(abs(le-learned_error[offset:offset+channels]))))
        maximum_stationarity=max(maximum_stationarity,float(np.linalg.norm(residual@matrix)/(np.linalg.norm(residual)*np.linalg.norm(matrix)+1e-300)))
        difference=(prediction[case]-target[case])@np.linalg.cholesky(gram);e=np.sum(difference*difference,axis=1)/den
        branch_errors.extend(e.reshape(-1,components).sum(axis=1));actual_teacher.extend(np.sqrt((te*te).reshape(-1,components).sum(axis=1)));actual_learned.extend(np.sqrt((le*le).reshape(-1,components).sum(axis=1)));offset+=channels
    checks=dict(training_array_hash=h.hexdigest()==info['training_array_sha256'],all_training_snapshots=offset==info['training_snapshots'],
        denominator_max_error=maximum_denominator,teacher_error_max_disagreement=maximum_teacher,learned_error_max_disagreement=maximum_learned,
        teacher_orthogonality=float(np.max(abs(basis.T@basis-np.eye(rank)))),teacher_projected_covariance_relative=float(np.linalg.norm(covariance-np.diag(values))/np.linalg.norm(values)),
        actual_trunk_gram_relative=float(np.linalg.norm(matrix.T@matrix-gram)/np.linalg.norm(gram)),learned_projection_max_normalized_gradient=maximum_stationarity,
        trunk_training_loss_disagreement=abs(trunk_loss-info['trunk']['best_training_loss']),branch_vector_loss_disagreement=abs(float(np.mean(branch_errors))-info['branch']['best_training_relative_mse']),
        teacher_vector_summary_disagreement=max(abs(float(np.mean(actual_teacher))-info['teacher_projection_error']['mean']),abs(float(np.max(actual_teacher))-info['teacher_projection_error']['worst'])),
        learned_vector_summary_disagreement=max(abs(float(np.mean(actual_learned))-info['learned_trunk_projection_error']['mean']),abs(float(np.max(actual_learned))-info['learned_trunk_projection_error']['worst'])))
    passed=checks['training_array_hash'] and checks['all_training_snapshots'] and all(value<1e-7 for key,value in checks.items() if not isinstance(value,bool))
    return dict(passed=passed,checks=checks,scope=__doc__,teacher_status='Seeded randomized training aid; projection is not an optimal rank floor',development_targets_used=False,final_targets_used=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True)
    p.add_argument('--verify-existing',help='Reuse a completed numerical audit only after every immutable input byte matches the final checksum-collected files')
    a=p.parse_args();out=Path(a.collected)/'output'
    raw=json.loads((out/'result.json').read_text());cfg=raw['config'];membership=json.loads((out/'membership.json').read_text())['membership']
    hashes={name:file_hash(out/name) for name in INPUT_FILES}
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if a.verify_existing:
        result=json.loads(Path(a.verify_existing).read_text());assert result['passed'] and result['complete']
        assert result['input_sha256']==hashes and result['configuration_sha256']==config_hash
        assert raw['complete'],'Only a completed panel can accept the earlier audit'
        result['verified_against_complete_collection']=True
        Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print('Every audited input matches the complete collection');return
    with np.load(out/'train_data.npz') as f:base=f['states']
    with (out/'confirmation_statistics.pkl').open('rb') as f:stats=pickle.load(f)
    def cases():
        for case,dx,dy,dz in membership:
            states=np.roll(base[case],(dx,dy,dz),axis=(-3,-2,-1));increment=(states[1:]-states[0])/stats['output_scale']
            y=np.ascontiguousarray(np.moveaxis(increment.reshape(15,cfg['n'],cfg['n'],cfg['n']),0,-1));den=np.repeat(np.sum(states[0]**2)/stats['output_scale']**2,5)
            yield y,den
    result=audit(out/'deeponet_candidate/pretraining',cases(),cfg['n']**3)
    result.update(complete=True,input_sha256=hashes,configuration_sha256=config_hash,verified_against_complete_collection=bool(raw['complete']))
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
