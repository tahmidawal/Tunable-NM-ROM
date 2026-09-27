"""Bounded Burgers3D development panel, with incremental artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import time
import traceback
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import core as c
b3=c.b3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    temp=Path(str(path)+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def host(tree):
    return jax.tree_util.tree_map(lambda x:np.asarray(x),tree)


def burn(seconds=.25):
    x=jnp.eye(384,dtype=jnp.float64)+.001
    f=jax.jit(lambda a:a@a.T)
    end=time.perf_counter()+seconds
    while time.perf_counter()<end:
        jax.block_until_ready(f(x))


def reference_audit(fields,nu,n,dt):
    maximum=0.
    for i in range(1,len(fields)):
        a,l=c.operators_np(fields[i],n)
        r=fields[i]-fields[i-1]+dt*(a-nu*l)
        maximum=max(maximum,float(np.linalg.norm(r)/max(np.linalg.norm(fields[i-1]),1e-300)))
    return maximum


def starts_for(target,H,Z,count=4):
    distance=np.sum(H**2,axis=1)[None,:]-2*target@H.T
    nearest=np.argmin(distance,axis=1)
    starts=np.zeros((len(target),count,Z.shape[1]))
    starts[:,0]=Z[nearest]
    for j,index in enumerate(np.linspace(0,len(Z)-1,count-2,dtype=int),start=2):
        starts[:,j]=Z[index]
    return starts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--checkpoint',default='inputs/refined_checkpoint.pkl')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report=dict(config=cfg,job_id=os.environ.get('SLURM_JOB_ID'),commit=os.environ.get('SOURCE_COMMIT'),
                gpu=jax.devices()[0].device_kind,backend=jax.default_backend(),
                x64=bool(jax.config.jax_enable_x64),precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
                checkpoint_sha256=sha(args.checkpoint),stage='starting',complete=False,
                references=[],representation=[],invocations=[],setup=[],
                final_cohort_unopened=True,comparison_scope='development, POD training subset unmatched to inherited network',
                timing_contract='dense initial interior field on GPU to all 51 dense interior fields on GPU; initial fitting included',
                vendor=json.loads(Path('VENDOR.json').read_text()))
    save=lambda:dump(out/'result.json',report)
    if 'training' in cfg:
        report['comparison_scope']='development, all newly learned components use identical original 512 training trajectories and eight saved training times'
    save()
    start=time.perf_counter()
    try:
        assert report['backend']=='gpu' and report['x64'] and report['precision']=='highest'
        print('jax_backend=gpu f64=True precision=highest',flush=True)
        ck=pickle.loads(Path(args.checkpoint).read_bytes())
        params=jax.tree_util.tree_map(jnp.asarray,ck['params'])
        hp={key:params[key] for key in ['h','h_lin']}
        Z=np.asarray(ck['Z_tr']);K=Z.shape[1];R=ck['cfg']['r']
        report['verification']=c.verify(params,Z)
        report['inherited_checkpoint_config']=ck['cfg']
        report['stage']='parameters';save()
        n=cfg['nodes'];steps=cfg['steps'];dt=cfg['dt']
        assert b3.DT==dt and b3.NUM_STEPS==steps
        train_rows=list(range(cfg['train_trajectories']))
        val_rows=cfg['validation_rows'];rows=train_rows+val_rows
        assert len(set(rows))==len(rows) and max(train_rows)<512 and min(val_rows)>=512
        raw=b3.draw_param_table(cfg['seed'],max(rows)+1)
        tab={key:(value[rows] if isinstance(value,np.ndarray) else value) for key,value in raw.items()}
        tab['m']=len(rows)
        tab['s_star']=b3.peak_on_reference_grid(tab)
        np.savez_compressed(out/'parameters.npz',**tab,original_rows=np.asarray(rows))
        coords=b3.grid_coords_3d(n);idx=b3.interior_indices_3d(n)
        G=np.concatenate([np.asarray(b3.features(params,jnp.asarray(coords[idx[s:s+2048]])))
                          for s in range(0,len(idx),2048)])
        Q,Rb=np.linalg.qr(G,mode='reduced')
        singular=np.linalg.svd(Rb,compute_uv=False)
        rank=int(np.sum(singular>np.finfo(float).eps*max(G.shape)*singular[0]))
        assert rank==R
        report['bank']=dict(rank=rank,singular_values=singular.tolist(),condition=float(singular[0]/singular[-1]),
                            qr_identity=float(np.linalg.norm(G-Q@Rb)/np.linalg.norm(G)))
        assert report['bank']['qr_identity']<1e-12
        H=np.asarray(b3.head(hp,jnp.asarray(Z)))@Rb.T
        del G
        _,_,_,Phi,lam=b3.test_modes_3d(n,cfg['test_modes'])
        report['actual_test_modes']=len(lam)
        assert len(lam)>2*max(R,K+max(cfg['q_ladder']))
        report['stage']='references';save()
        fom=b3.make_newton_tol_rollout(n,'fft')
        Utrain=[];truths=[];initials=[];viscosities=[]
        for local,row in enumerate(rows):
            u=b3.blob_ic_3d(n,tab,local,coords)[idx]
            nu=float(tab['nu'][local])
            f,it,rn=host(fom(jnp.asarray(u),nu,1e-10,1e-11))
            independent=reference_audit(f,nu,n,dt)
            assert np.isfinite(f).all() and independent<2e-9,(row,independent)
            record=dict(row=row,max_relative_residual=float(np.max(rn)),numpy_max_relative_residual=independent,
                        iterations=it.tolist(),nu=nu,role='train' if local<len(train_rows) else 'validation')
            if local<len(train_rows):
                Utrain.append(f[cfg['train_steps']])
            else:
                case=local-len(train_rows)
                truths.append(f);initials.append(u);viscosities.append(nu)
                name=f'reference_case{case}.npz';np.savez_compressed(out/name,fields=f,iterations=it,residuals=rn,u0=u,nu=nu)
                record['artifact']=name
            report['references'].append(record)
            if local%8==0 or local==len(rows)-1:
                save();print('REFERENCE',local+1,len(rows),'elapsed',round(time.perf_counter()-start,1),flush=True)
        U=np.concatenate(Utrain);del Utrain
        targets=U@Q;floor=np.sum((U-targets@Q.T)**2,axis=1)
        np.savez_compressed(out/'training_fields.npz',fields=U,steps=cfg['train_steps'],rows=train_rows)
        report['stage']='directions';save()
        selected=np.linspace(0,len(targets)-1,min(cfg['direction_states'],len(targets)),dtype=int)
        target=targets[selected];ff=floor[selected]
        initial_starts=starts_for(target,H,Z)
        fit0=c.make_fit(K,0,cfg['fit_budget'],cfg['gradient_tolerance'])
        selected_fit,all_fit=host(fit0(jnp.asarray(target),jnp.asarray(ff),jnp.asarray(initial_starts),
                                    jnp.zeros((R,0)),jnp.asarray(Rb),hp))
        zfit=selected_fit[0][:,:K]
        rho=target-np.asarray(b3.head(hp,jnp.asarray(zfit)))@Rb.T
        _,s,Ct=np.linalg.svd(rho,full_matrices=False)
        C=Ct.T
        assert C.shape==(R,R) and np.linalg.norm(C.T@C-np.eye(R))<1e-10
        np.savez_compressed(out/'directions.npz',C=C,singular_values=s,selected=selected,fit_states=selected_fit[0],
                            fit_errors=selected_fit[1],fit_iterations=selected_fit[2],fit_reasons=selected_fit[3],
                            fit_gradients=selected_fit[4],all_errors=all_fit[1],all_gradients=all_fit[4])
        print('DIRECTIONS complete',round(time.perf_counter()-start,1),flush=True)
        report['directions']=dict(training_states=len(selected),stationary=int(np.sum(selected_fit[4]<=cfg['gradient_tolerance'])),
                                  total=len(selected),singular_values=s.tolist())
        report['stage']='POD';save()
        maxrank=max(cfg['pod_ranks'])
        width=min(maxrank+24,min(U.shape))
        uj=jnp.asarray(U);omega=jnp.asarray(np.random.default_rng(20260920).normal(size=(len(U),width)))
        podq=jnp.linalg.qr(uj.T@omega,mode='reduced')[0]
        for _ in range(2):
            podq=jnp.linalg.qr(uj.T@(uj@podq),mode='reduced')[0]
        _,ps,pvt=jnp.linalg.svd(uj@podq,full_matrices=False)
        POD=np.asarray(podq@pvt.T[:,:maxrank])
        assert np.linalg.norm(POD.T@POD-np.eye(maxrank))<1e-9
        np.savez_compressed(out/'bases.npz',bank=Q,bank_R=Rb,pod=POD,pod_singular=np.asarray(ps),lam=lam)
        del uj,omega,podq,pvt,U
        report['stage']='representation';save()
        vals=np.concatenate([x[[0,10,25,50]] for x in truths])
        at=vals@Q;fl=np.sum((vals-at@Q.T)**2,axis=1);norm=np.linalg.norm(vals,axis=1)
        st=starts_for(at,H,Z)
        for q in cfg['q_ladder']:
            fit=c.make_fit(K,q,cfg['fit_budget'],cfg['gradient_tolerance'])
            best,all_results=host(fit(jnp.asarray(at),jnp.asarray(fl),jnp.asarray(st),jnp.asarray(C[:,:q]),jnp.asarray(Rb),hp))
            errors=best[1]/np.maximum(norm,1e-300)
            name=f'representation_q{q}.npz'
            np.savez_compressed(out/name,states=best[0],errors=errors,iterations=best[2],reasons=best[3],gradients=best[4],
                                all_states=all_results[0],all_errors=all_results[1],all_gradients=all_results[4],
                                target=at,bank_floor=np.sqrt(fl)/norm)
            report['representation'].append(dict(q=q,mean=float(np.mean(errors)),median=float(np.median(errors)),
                 worst=float(np.max(errors)),stationary=int(np.sum(best[4]<=cfg['gradient_tolerance'])),n=len(errors),artifact=name))
            save();print('REPRESENTATION',q,report['representation'][-1],flush=True)
        report['bank_floor']=dict(mean=float(np.mean(np.sqrt(fl)/norm)),worst=float(np.max(np.sqrt(fl)/norm)))
        report['refinement']=[]
        for case in cfg.get('refinement_cases',[]):
            local=len(train_rows)+case;nu=viscosities[case]
            for label,nn,dd,ss in [('time_half',n,dt/2,steps*2),('space_fine',2*(n-1)+1,dt,steps),
                                  ('space_time_fine',2*(n-1)+1,dt/2,steps*2)]:
                xx=b3.grid_coords_3d(nn);ii=b3.interior_indices_3d(nn)
                uu=b3.blob_ic_3d(nn,tab,local,xx)[ii]
                fn=c.make_fom(nn,dd,ss)
                ff,its,rns=host(fn(jnp.asarray(uu),nu,1e-10,1e-11))
                defect=reference_audit(ff,nu,nn,dd)
                assert defect<2e-9 and np.isfinite(ff).all()
                physical=ff.reshape((ss+1,)+(nn-2,)*3)
                temporal=physical[::int(round(dt/dd))]
                restricted=temporal if nn==n else temporal[:,1::2,1::2,1::2]
                restricted=restricted.reshape(truths[case].shape)
                artifact=f'refinement_{label}_case{case}.npz'
                np.savez_compressed(out/artifact,fields=ff,restricted=restricted,nu=nu,iterations=its,residuals=rns)
                report['refinement'].append(dict(case=case,label=label,nodes=nn,dt=dd,steps=ss,artifact=artifact,
                    numpy_max_relative_residual=defect,**c.metrics(truths[case],restricted)))
                save();print('REFINEMENT',report['refinement'][-1]['label'],case,report['refinement'][-1]['worst_evolved'],flush=True)
        report['stage']='timing';save()
        subjects=[]
        for q in cfg['q_ladder']:
            subjects.append((f'rom_q{q}',False,K,q,Q,C[:,:q]))
        subjects.append((f'free_R{R}',True,0,R,Q,np.zeros((R,0))))
        for rank in cfg['pod_ranks']:
            subjects.append((f'pod_{rank}',True,0,rank,POD[:,:rank],np.zeros((rank,0))))
        rng=np.random.default_rng(cfg['timing_seed']);rng.shuffle(subjects)
        # Controls run in the same allocation and return the identical dense time grid.
        for ntol,ltol in cfg.get('fom_controls',[(1e-2,5e-3),(1e-4,1e-6),(1e-6,1e-8)]):
            name=f'fom_nt{ntol:.0e}' if 'fom_controls' not in cfg else f'fom_nt{ntol:.0e}_lt{ltol:.0e}'
            for _ in range(2):
                jax.block_until_ready(fom(jnp.asarray(initials[0]),viscosities[0],ntol,ltol))
            for case in rng.permutation(len(truths)):
                case=int(case);u0=jnp.asarray(initials[case]);nu=viscosities[case]
                for rep in range(cfg['repetitions']):
                    burn();before=time.perf_counter();result=fom(u0,nu,ntol,ltol);jax.block_until_ready(result)
                    gpu_ms=(time.perf_counter()-before)*1000
                    f,it,rn=host(result);met=c.metrics(f,truths[case]);artifact=f'{name}_case{case}_rep{rep}.npz'
                    canonical=out/f'{name}_case{case}_rep0.npz'
                    if rep and canonical.exists():
                        previous=np.load(canonical)
                        if all(np.array_equal(previous[key],value) for key,value in [('fields',f),('iterations',it),('residuals',rn)]):
                            artifact=canonical.name
                    if not (out/artifact).exists():np.savez_compressed(out/artifact,fields=f,iterations=it,residuals=rn)
                    report['invocations'].append(dict(method=name,case=case,row=val_rows[case],repetition=rep,gpu_ms=gpu_ms,
                           artifact=artifact,iterations=it.tolist(),nonlinear_tolerance=ntol,linear_tolerance=ltol,
                           converged=bool(np.max(rn)<=ntol*(1+1e-6)),**met))
                    save()
        for name,linear,k,q,B,Cq in subjects:
            began=time.perf_counter()
            data=c.data_for_basis(B,Phi,lam,Cq,Rb,Z,H)
            query=c.make_query(n,k,q,linear,steps,dt,cfg['fit_budget'],cfg['step_budget'],cfg['gradient_tolerance'])
            # Two untimed calls compile and warm the exact full-query contract.
            for _ in range(2):
                jax.block_until_ready(query(jnp.asarray(initials[0]),viscosities[0],data,hp))
            report['setup'].append(dict(method=name,seconds=time.perf_counter()-began))
            order=[(case,rep) for case in range(len(truths)) for rep in range(cfg['repetitions'])]
            rng.shuffle(order)
            for case,rep in order:
                u0=jnp.asarray(initials[case]);nu=viscosities[case]
                burn()
                before=time.perf_counter();result=query(u0,nu,data,hp);jax.block_until_ready(result)
                gpu_ms=(time.perf_counter()-before)*1000
                f,w,it,reason,grad,rn,ic=host(result)
                met=c.metrics(f,truths[case]);artifact=f'{name}_case{case}_rep{rep}.npz'
                arrays=dict(fields=f,states=w,iterations=it,reasons=reason,gradients=grad,residuals=rn,initial_fit=ic)
                previous_path=next(iter(out.glob(f'{name}_case{case}_rep*.npz')),None)
                if previous_path is not None:
                    previous=np.load(previous_path)
                    if all(np.array_equal(previous[key],value) for key,value in arrays.items()):artifact=previous_path.name
                if not (out/artifact).exists():np.savez_compressed(out/artifact,**arrays)
                row=dict(method=name,case=case,row=val_rows[case],repetition=rep,gpu_ms=gpu_ms,artifact=artifact,
                         iterations=it.tolist(),reasons=reason.tolist(),gradients=grad.tolist(),initial_fit=ic.tolist(),
                         stationary=bool(np.all(grad<=cfg['gradient_tolerance']) and ic[2]<=cfg['gradient_tolerance']),**met)
                report['invocations'].append(row);save()
                print('TIMING',name,case,rep,round(gpu_ms,2),'err',round(met['worst_evolved'],5),'stationary',row['stationary'],flush=True)
            del query,data
            jax.clear_caches()
        report['stage']='complete';report['complete']=True
        report['seconds']=time.perf_counter()-start;save()
        print('COMPLETE',report['seconds'],flush=True)
    except BaseException as error:
        report['error']=repr(error);report['traceback']=traceback.format_exc();report['seconds']=time.perf_counter()-start
        save();raise


if __name__=='__main__':
    main()
