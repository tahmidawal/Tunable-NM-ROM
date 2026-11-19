"""Named development-family reconstruction and actual4096 output; no PDE rollout."""
import argparse
from common import *
import data
from neural import stable as S,fields as F
from neural.build_data import resolve
from training_io import read_checkpoint,checkpoint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    rp=resolve(ns/'b6namedrefs/output/result.json',ns);refs=json.loads(rp.read_text())
    check=resolve(ns/'b6namedfd/output/result.json',ns);assert json.loads(check.read_text())['passed']
    source=resolve(ns/'b7parentrepair/output/result.json',ns);rr=json.loads(source.read_text());assert rr['status']=='complete_stable_parent'
    cp=source.parent/'stable_parent.pkl';assert digest(cp)==rr['selected_sha256'];p=read_checkpoint(cp)
    network=jax.tree.map(jnp.asarray,p['network']);codes=jnp.asarray(p['training_codes']);enc=jnp.asarray(p['training_encoder'])
    xy,w=data.grid(128,True);xf,wf=data.grid(256,True);xe,_=data.grid(16,True)
    xy,w,xf,wf=map(jnp.asarray,(xy,w,xf,wf));ex=jnp.asarray(np.unique(xe[:,0]))
    fit=F.make_fit(S.apply,60);curv=F.make_fit(S.apply,120,newton=True);decode=jax.jit(S.apply)
    result=dict(meta=meta,status='running',parent_sha256=digest(cp),source_sha256=digest(source),reference_sha256=digest(rp),crosscheck_sha256=digest(check),
        classification='Offline reconstruction diagnostic, including failed-checkpoint transfer if primary gate fails; not a PDE solve or method advancement',
        parent_primary_gate=rr['parent_quality_pass'],final_test_opened=False,trained_on_named_cases=False,cases=[])
    for ref in refs['cases']:
        path=rp.parent/ref['levels'][-1]['file'];assert digest(path)==ref['levels'][-1]['sha256']
        b=jnp.asarray(np.load(path)['coefficients']);norm=jnp.linalg.norm(b.reshape(11,-1),axis=1)/2
        y=data.synth_grid(b,jnp.unique(xy[:,0])).reshape(11,-1);yf=data.synth_grid(b,jnp.unique(xf[:,0])).reshape(11,-1)
        ey=data.synth_grid(b,ex).reshape(11,-1);rows=[];saved=[]
        for i in range(11):
            near,interp=S.nearest_starts(ey[i],enc,codes)
            candidates=[fit(network,z,xy,y[i],w,norm[i]) for z in [near,interp,jnp.zeros(32)]]
            z,st=min(candidates,key=lambda v:float(v[1][0]))
            if float(st[-1])==0:z,st=curv(network,z,xy,y[i],w,norm[i])
            pred=decode(network,z,xf);error=float(jnp.sqrt(jnp.sum(wf*(pred-yf[i])**2))/norm[i])
            row=dict(time=float(data.TIMES[i]),relative_error=error,fit_status=np.asarray(st),scoring_change=error-float(st[0]))
            if i%2==0:
                np.savez_compressed(out/f"{ref['case']}_time{i:02d}_independent256.npz",xy=np.asarray(xf),reference=np.asarray(yf[i]),prediction=np.asarray(pred))
            if i%2==0 and abs(ref['nu']-.005)<1e-12:
                axis=np.linspace(0,1,4096);prediction=np.empty((4096,4096),np.float64)
                for lo in range(0,4096,16):
                    points=np.stack(np.meshgrid(axis[lo:lo+16],axis,indexing='ij'),-1).reshape(-1,2)
                    prediction[lo:lo+16]=np.asarray(decode(network,z,jnp.asarray(points))).reshape(16,4096)
                truth=np.asarray(data.synth_grid(b[i],jnp.asarray(axis)))
                file=out/f"{ref['case']}_time{i:02d}_actual4096.npz"
                np.savez_compressed(file,reference=truth,prediction=prediction,time=data.TIMES[i])
                row['actual4096']=dict(file=file.name,shape=[4096,4096],relative_error=float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth)))
            rows.append(row);saved.append(np.asarray(z))
        case=dict(case=ref['case'],nu=ref['nu'],definition=ref['definition'],rows=rows,
            worst_error=max(v['relative_error'] for v in rows),unfinished=sum(int(v['fit_status'][-1]==0) for v in rows),
            max_scoring_change=max(abs(v['scoring_change']) for v in rows))
        case['codes_sha256']=checkpoint(out/(ref['case']+'_codes.pkl'),dict(codes=saved))
        result['cases'].append(case);save(out/'result.json',result);print('named_parent_diagnostic',ref['case'],case['worst_error'],case['unfinished'],flush=True)
    result.update(status='complete',representation_admitted=False,pde_admitted=False);save(out/'result.json',result)


if __name__=='__main__':main()
