"""Independent streamed NumPy audit of expanded fine-grid QR targets and emitted head."""
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import scipy.linalg
import audit_iterative as ai


def digest(p):return ai.sha(Path(p))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outdir',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--result',type=Path,required=True);parser.add_argument('--chunk',type=int,default=4096)
    a=parser.parse_args();begin=time.perf_counter();target_path=a.outdir/'training_targets.npz';train_path=a.outdir/'training.json';new_path=a.outdir/'trained_checkpoint.pkl'
    tar=dict(np.load(target_path));training=json.loads(train_path.read_text());cfg=training['config'];old=pickle.loads(a.checkpoint.read_bytes());new=pickle.loads(new_path.read_bytes());R=tar['R'];Y=tar['Y'];L=cfg['mesh'];n=cfg['cases']
    assert L==1024 and R.shape==(512,512) and Y.shape==(n,512) and cfg['steps']==30000
    singular=scipy.linalg.svdvals(R);rank=int(np.sum(singular>max(R.shape)*np.finfo(float).eps*singular[0]));condition=float(singular[0]/singular[-1])
    assert rank==512 and np.isfinite(condition)
    def leaves(obj):
        if isinstance(obj,dict):return [x for v in obj.values() for x in leaves(v)]
        if isinstance(obj,(list,tuple)):return [x for v in obj for x in leaves(v)]
        return [np.asarray(obj)]
    for k in ['B','g','out_scale']:assert all(np.array_equal(x,y) for x,y in zip(leaves(old['params'][k]),leaves(new['params'][k])))
    assert np.array_equal(new['Z_tr'][:len(old['Z_tr'])],old['Z_tr']) and new['Z_tr'].shape==(len(old['Z_tr'])+n,16)
    assert all(v.dtype==np.float64 for v in leaves(new['params']))
    assert cfg['physical_draws']==[dict(seed=0,cases=576),dict(seed=1000,cases=4032)] and n==4608
    def draw(seed,count):
        rng=np.random.default_rng(seed)
        return np.stack([rng.uniform(.15,.85,count),rng.uniform(.15,.85,count),rng.uniform(.05,.2,count),rng.uniform(.5,2.,count),np.exp(rng.uniform(np.log(.01),np.log(.1),count))],axis=1)
    physical=np.concatenate([draw(item['seed'],item['cases']) for item in cfg['physical_draws']])
    assert training['per_update_head_initial_batch']==training['per_update_head_replay_batch']==256
    assert training['optimized_code_scalars']==4608*16
    assert all(training[k]>0 for k in ['feature_qr_seconds','field_projection_seconds','code_fit_seconds','optimizer_loop_seconds'])
    assert np.allclose(physical,tar['physical'],rtol=2e-15,atol=0)
    physical=tar['physical'];assert np.array_equal(physical,np.array(training['physical_cases']))
    ids=np.sort(np.random.default_rng(cfg['seed']).choice(len(old['Z_tr']),cfg['replay_cases'],replace=False));assert np.array_equal(ids,tar['replay_ids'])
    assert np.array_equal(tar['Z_replay'],old['Z_tr'][ids])
    expected_replay=ai.head(old['params'],tar['Z_replay'])@R.T
    replay_target_delta=float(np.linalg.norm(expected_replay-tar['Y_replay'])/np.linalg.norm(expected_replay));assert replay_target_delta<1e-11
    before=ai.head(old['params'],tar['Z_init']);after=ai.head(new['params'],new['Z_tr'][-n:]);proj=scipy.linalg.solve_triangular(R,Y.T,lower=False)
    norm=np.zeros(n);floor=np.zeros(n);old_error=np.zeros(n);new_error=np.zeros(n);gram=np.zeros(R.shape);moment=np.zeros(Y.T.shape)
    coefficients=np.concatenate((proj,before.T,after.T),axis=1)
    # Complete interior domain, regenerated analytically in physical-coordinate batches.
    for start in range(0,(L-1)**2,a.chunk):
        end=min(start+a.chunk,(L-1)**2);index=np.arange(start,end);xy=np.stack((index//(L-1)+1,index%(L-1)+1),axis=1)/L
        G=ai.features(old['params'],xy)/L
        cx,cy,w,amp,nu=physical.T
        U=amp[None,:]*np.exp(-((xy[:,0,None]-cx)**2+(xy[:,1,None]-cy)**2)/(2*w*w))/L
        predictions=G@coefficients;fp,bp,ap=np.split(predictions,3,axis=1)
        norm+=np.sum(U*U,axis=0);floor+=np.sum((fp-U)**2,axis=0);old_error+=np.sum((bp-U)**2,axis=0);new_error+=np.sum((ap-U)**2,axis=0)
        gram+=G.T@G;moment+=G.T@U
        if start==0 or end==(L-1)**2 or end//a.chunk%32==0:print('DIRECT_TRAINING_AUDIT',end,(L-1)**2,'seconds',time.perf_counter()-begin,flush=True)
    norm_delta=float(np.max(abs(norm-tar['norm2'])/norm));floor2_delta=float(np.max(abs(floor-tar['floor2'])/norm));gram_delta=float(np.linalg.norm(gram-R.T@R)/np.linalg.norm(gram));moment_delta=float(np.linalg.norm(moment-R.T@Y.T)/np.linalg.norm(moment))
    old_relative=np.sqrt(old_error/norm);new_relative=np.sqrt(new_error/norm);floor_relative=np.sqrt(floor/norm)
    before_delta=float(np.max(abs(old_relative-np.array(training['initial_relative_before_training']))));after_delta=float(np.max(abs(new_relative-np.array(training['final_initial_relative']))))
    floor_relative_delta=float(np.max(abs(floor_relative-np.array(training['bank_relative_floor']))))
    new_replay=ai.head(new['params'],tar['Z_replay'])@R.T
    replay_relative=np.linalg.norm(new_replay-expected_replay,axis=1)/np.linalg.norm(expected_replay,axis=1)
    replay_error_delta=float(np.max(abs(replay_relative-np.array(training['final_replay_relative']))))
    assert norm_delta<1e-11 and floor2_delta<1e-11 and gram_delta<1e-11 and moment_delta<1e-11
    assert before_delta<1e-8 and after_delta<1e-8 and floor_relative_delta<1e-8 and replay_error_delta<1e-10
    trace=training['loss_trace'];assert len(trace)==61 and trace[-1]['step']==cfg['steps']
    assert abs(trace[-1]['initial_relative_squared_loss']-np.mean(new_error/norm))<1e-10
    assert abs(trace[-1]['replay_relative_squared_loss']-np.mean(replay_relative**2))<1e-10
    def stats(x):return dict(median=float(np.median(x)),mean=float(np.mean(x)),maximum=float(np.max(x)))
    result=dict(passed=True,status='independent full N1024 NumPy training audit',training_examples=n,interior_nodes=(L-1)**2,spatial_bank_byte_identical=True,
        R_rank=rank,R_condition2=condition,R_min_singular_value=float(singular[-1]),R_max_singular_value=float(singular[0]),direct_gram_relative_delta=gram_delta,
        direct_projection_moment_relative_delta=moment_delta,direct_field_norm_relative_max_delta=norm_delta,perpendicular_squared_error_normalized_max_delta=floor2_delta,
        perpendicular_relative_max_delta=floor_relative_delta,initial_before_relative_max_delta=before_delta,initial_after_relative_max_delta=after_delta,
        replay_target_relative_delta=replay_target_delta,replay_error_relative_max_delta=replay_error_delta,
        floor=stats(floor_relative),before=stats(old_relative),after=stats(new_relative),replay_after=stats(replay_relative),
        per_case=dict(floor=floor_relative.tolist(),before=old_relative.tolist(),after=new_relative.tolist()),
        replay_note='teacher decoded training states; does not verify true trajectory generalization',elapsed_seconds=time.perf_counter()-begin,
        source_sha256={str(p):digest(p) for p in [target_path,train_path,new_path,a.checkpoint,Path(__file__)]})
    a.result.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['per_case','source_sha256']}),flush=True)
if __name__=='__main__':main()
