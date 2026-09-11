"""Independent training-only average-optimal span diagnostic; no learned model."""
from audit_iterative import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();run=args.run.resolve();out=run/'cluster/out/pilot';d=json.loads((out/'result.json').read_text());cfg=d['config'];n=cfg['training_nodes']-1
    train=np.asarray(d['training']['parameters']);dev=np.asarray(d['cohort']['parameters']);U=np.array([truth(n,p)[1:-1,1:-1].ravel() for p in train]);V=np.array([truth(n,p)[1:-1,1:-1].ravel() for p in dev]);U/=np.linalg.norm(U,axis=1)[:,None]
    eig,vec=np.linalg.eigh(U@U.T);order=np.argsort(eig)[::-1];eig=eig[order];vec=vec[:,order];panels=[]
    for rank in [64,128]:
        assert eig[rank-1]>eig[0]*len(train)*np.finfo(float).eps
        P=np.linalg.qr(U.T@vec[:,:rank]/np.sqrt(eig[:rank]),mode='reduced')[0]
        te=np.linalg.norm(U-(U@P)@P.T,axis=1);ve=np.linalg.norm(V-(V@P)@P.T,axis=1)/np.linalg.norm(V,axis=1)
        trace_error=abs(np.sum(te**2)-np.sum(eig[rank:]));assert trace_error<1e-9
        item=dict(rank=rank,basis_sha256=array_sha(P),orthogonality_error=float(np.linalg.norm(P.T@P-np.eye(rank))),training_relative_errors=te.tolist(),training_mean_squared_relative_error=float(np.mean(te**2)),training_median_relative_error=float(np.median(te)),training_worst_relative_error=float(max(te)),average_optimal_trace_disagreement=float(trace_error),development_relative_errors=ve.tolist(),groups=[])
        for group in ['all','existing_development','new_development']:
            ids=[i for i,g in enumerate(d['cohort']['groups']) if group=='all' or g==group];errors=ve[ids];item['groups'].append(dict(group=group,cases=len(ids),median_relative_error=float(np.median(errors)),worst_relative_error=float(max(errors)),worst_case=ids[int(np.argmax(errors))]))
        if rank==64:
            delta=float(np.max(np.abs(ve-d['training']['pod_diagnostic']['development_relative_errors'])));assert delta<1e-9;item['native_rank64_error_disagreement']=delta
        panels.append(item);print('POD',rank,item['training_worst_relative_error'],item['groups'],flush=True)
    result=dict(status='independent CPU diagnostic, training-only span construction; no additional neural training or PDE benchmark',source_result_sha256=file_sha(out/'result.json'),script_sha256=file_sha(Path(__file__)),training_parameter_sha256=d['training']['parameter_sha256'],development_parameter_sha256=d['cohort']['parameter_sha256'],training_intervals=n,normalized_training_snapshots=len(train),eigenvalues=eig.tolist(),ranks=panels,interpretation='POD minimizes average squared relative reconstruction error on these normalized training snapshots. Its development worst errors diagnose capacity potential but are not certified minimax lower bounds, learned-coordinate-bank guarantees, or online NMROM results. Development truth never constructs the span; final cohorts remain unopened.')
    (run/'pod-capacity-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
