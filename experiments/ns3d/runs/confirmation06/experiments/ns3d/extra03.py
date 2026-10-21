"""Add frozen shared DeepONet/Transolver; remeasure the entire NS3D panel.

Data regenerate from the original seeds and must match hashes of comparison02.
Reused checkpoints/operators are content-addressed and device resident before
compilation/timing. This driver does not select or draw a final cohort.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,subprocess,time,traceback
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.signal import resample
import ns3d_fom as F
import ns3d_model as D
import ns3d_rom as R
import dataset_adapter as A
import operator_adapter as O
from operators import training as OT
from pilot import generate,write,sha
from comparison import burn


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(path):
    with Path(path).open('rb') as f:return pickle.load(f)


def with_coordinates(x):
    n=x.shape[1];axis=np.arange(n,dtype=np.float64)/n
    xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    return np.concatenate((x,np.broadcast_to(xyz,(len(x),*xyz.shape))),axis=-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True);result=out/'result.json';reuse=Path(cfg['reuse_path'])
    report=dict(complete=False,config=cfg,kind='ns3d_four_operator_development_comparison',source_commit=os.environ.get('SOURCE_COMMIT'),
        job_id=os.environ.get('SLURM_JOB_ID'),backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
        final_cohort_opened=False,limitations=['one training seed, opened development cohort','R512 bank and K16 head failed representation target; preserved as negative controls',
        'dense initial projection and dense output charged; no grid-independent claim','training capped, convergence not claimed'])
    def stage(name):report['stage']=name;write(report,result);print('STAGE',name,flush=True)
    try:
        report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
        assert report['backend']=='gpu' and report['x64'] and report['precision']=='highest'
        assert cfg['initial_amplitude']==F.INITIAL_AMPLITUDE
        stage('verified_reuse')
        reuse_manifest=json.loads((reuse/'REUSE.json').read_text());report['reuse']=reuse_manifest
        for entry in reuse_manifest['files']:
            assert file_hash(reuse/entry['path'])==entry['sha256'],entry['path']
        previous=json.loads((reuse/'result.json').read_text());assert previous['complete']
        prior_cfg=previous['config']
        for key in ['n','initial_amplitude','dt','horizon','train_cases','dev_cases','train_seed','dev_seed','k','rom_dt','test_modes','q_values']:
            assert cfg[key]==prior_cfg[key],key
        stage('regenerate_matched_data')
        n=cfg['n'];geom=F.geometry(n);h=cfg['horizon'];dt=cfg['rom_dt'];steps=round(h/dt);every=steps//5
        Utr,report['train_data']=generate(n,cfg['dt'],h,cfg['train_cases'],cfg['train_seed'],out/'train_data.npz')
        Udev,report['dev_data']=generate(n,cfg['dt'],h,cfg['dev_cases'],cfg['dev_seed'],out/'dev_data.npz')
        for key in ['train_data','dev_data']:
            assert report[key]['parameter_sha256']==previous[key]['parameter_sha256']
            # Roundoff from GPU models may change exact bytes; enforce bounded
            # field replay below against retained seed records if hashes differ.
            assert report[key]['states_sha256']==previous[key]['states_sha256'], 'data replay differs: '+key
        statistics=load(reuse/'operator_statistics.pkl')
        trds=dict(initial=Utr[:,0],viscosity=F.parameters(cfg['train_seed'],len(Utr))[:,-1:],targets=Utr[:,1:])
        dvds=dict(initial=Udev[:,0],viscosity=F.parameters(cfg['dev_seed'],len(Udev))[:,-1:],targets=Udev[:,1:])
        computed=A.training_statistics(trds)
        assert all(np.allclose(statistics[key],computed[key],rtol=1e-13,atol=1e-13) for key in statistics)
        tx,ty=A.common_model_arrays(trds,statistics);vx,vy=A.common_model_arrays(dvds,statistics);tx=with_coordinates(tx);vx=with_coordinates(vx)
        td=np.repeat(np.sum(trds['initial']**2,axis=(1,2,3,4))[:,None]/statistics['output_scale']**2,5,1)
        vd=np.repeat(np.sum(dvds['initial']**2,axis=(1,2,3,4))[:,None]/statistics['output_scale']**2,5,1)
        stage('larger_bank_head_screen')
        from capacity_screen import run as capacity_run
        try:
            report['capacity_screen']=capacity_run(Utr,Udev,cfg,out/'capacity')
        except Exception as capacity_error:
            report['capacity_screen_failure']=dict(error=str(capacity_error),traceback=traceback.format_exc())
            print('CAPACITY_SCREEN_FAILED',report['capacity_screen_failure'],flush=True)
        write(report,result)
        stage('new_operator_training');op_models=[];report['operators']=[]
        for index,spec in enumerate(cfg['extra_operators']):
            training=dict(cfg['operator_train'],seed=cfg['operator_train']['seed']+index)
            model,info=OT.train(tx,ty,vx,vy,spec,training,out/spec['kind'],lambda path,obj:write(obj,path),O.checkpoint,td,vd)
            op_models.append((spec,model));report['operators'].append(info);write(report,result)
        del tx,ty,vx,vy,td,vd,Utr,trds
        for spec in previous['config']['operators']:
            frozen=load(reuse/spec['kind']/'best.pkl');assert frozen['spec']==spec
            op_models.append((spec,jax.tree_util.tree_map(jax.device_put,frozen['params'])))
        checkpoint=load(reuse/'checkpoint.pkl');params=checkpoint['params'];z=checkpoint['codes'];G=checkpoint['extra']['bank'];Qb=checkpoint['extra']['qr_Q'];Rb=checkpoint['extra']['qr_R']
        with np.load(reuse/'weak_operators.npz') as f:Amat=f['A'];T=f['T'];lam=f['lam'];Cq=f['C']
        with np.load(reuse/'pod.npz') as f:P=f['basis']
        stage('assemble_reused_complete_queries');methods=[]
        def add(name,fn,extra,kind):methods.append(dict(name=name,fn=fn,extra=jax.tree_util.tree_map(jax.device_put,extra),kind=kind))
        shared=(G,Qb,Rb,Amat,T,lam);theta={name:params[name] for name in ('h','h_lin')}
        # Device-put the shared tensor exactly once; every rung receives same objects.
        shared=jax.tree_util.tree_map(jax.device_put,shared);theta=jax.tree_util.tree_map(jax.device_put,theta);zc=jax.device_put(z)
        for q in cfg['q_values']:
            add(f'nmrom_q{q}',R.make_run(dt,steps,every,cfg['k'],q,budget=cfg['rom_budget'],retain_states=True),shared+(jax.device_put(Cq[:,:q]),theta,zc),'weak_lm')
        Phi,_,ids=R.test_modes(n,cfg['test_modes']);rank=max(cfg['pod_weak_ranks']);PT=R.build_tensor(P[:,:rank],n,ids)
        np.savez_compressed(out/'pod_weak_operators.npz',A=Phi.T@P[:,:rank],T=PT,lam=lam)
        for r in cfg['pod_weak_ranks']:
            basis=P[:,:r]
            add(f'pod_weak_{r}',R.make_run(dt,steps,every,0,r,linear=True,budget=cfg['rom_budget'],retain_states=True),
                (basis,basis,np.eye(r),Phi.T@basis,PT[:,:r,:r],lam,np.empty((r,0)),{},np.empty((1,0))),'weak_lm')
        with np.load(reuse/'pod_galerkin.npz') as f:PL=f['L'];PG=f['T']
        for r in cfg['pod_galerkin_ranks']:
            add(f'pod_galerkin_{r}',R.make_galerkin_run(dt,steps,every),(P[:,:r],PL[:r,:r],PG[:r,:r,:r]),'galerkin')
        with np.load(reuse/'bank_galerkin.npz') as f:BL=f['L'];BG=f['T']
        add('free_bank_galerkin',R.make_galerkin_run(dt,steps,every),(Qb,BL,BG),'galerkin')
        for fdt in cfg['fom_dts']:
            ns=round(h/fdt);add(f'fom_dt{fdt}',F.make_solver(fdt,ns,ns//5),(geom,),'fom')
        for spec,model in op_models:
            for projected in (False,True):
                add(spec['kind']+('_projected' if projected else '_raw'),O.make_query(spec,n,projected),
                    (model,np.asarray(statistics['input_mean']),np.asarray(statistics['input_std']),np.asarray(statistics['output_scale']),geom),'operator')
        report['method_names']=[m['name'] for m in methods]
        report['device_resident_parameters']=all(isinstance(x,jax.Array) for m in methods for x in jax.tree_util.tree_leaves(m['extra']) if hasattr(x,'dtype'))
        assert report['device_resident_parameters'];stage('timed_reference_fields')
        count=cfg['timed_cases'];parameters=F.parameters(cfg['dev_seed'],len(Udev))[:count];refs=[];fine=[]
        rs=round(h/cfg['reference_dt']);rf=F.make_solver(cfg['reference_dt'],rs,rs//5);fs=round(h/cfg['fine_reference_dt']);ff=F.make_solver(cfg['fine_reference_dt'],fs,fs//5);fg=F.geometry(cfg['fine_reference_n'])
        for par in parameters:
            refs.append(np.asarray(rf(jnp.asarray(F.initial(n,par)),par[-1],geom)))
            fine.append(np.asarray(ff(jnp.asarray(F.initial(cfg['fine_reference_n'],par)),par[-1],fg)))
        refs=np.stack(refs);fine=np.stack(fine);np.savez_compressed(out/'timing_references.npz',same_grid=refs,fine_grid=fine,parameters=parameters)
        stage('compile_complete_queries')
        for method in methods:
            st=time.perf_counter();value=method['fn'](jnp.asarray(Udev[0,0]),parameters[0,-1],*method['extra']);jax.block_until_ready(value)
            method['compile_warm_seconds']=time.perf_counter()-st;print('COMPILED',method['name'],method['compile_warm_seconds'],flush=True)
        stage('matched_timing');rows=[];saved={};histories={};burn(3.)
        for case in range(count):
            u0=jnp.asarray(Udev[case,0]);nu=parameters[case,-1]
            for repetition in range(cfg['repetitions']):
                burn(1.);invocations=[];order=np.roll(np.arange(len(methods)),case+repetition)
                if repetition%2:order=order[::-1]
                for oi,j in enumerate(order):
                    method=methods[j];st=time.perf_counter();value=method['fn'](u0,nu,*method['extra']);jax.block_until_ready(value);device=time.perf_counter()-st
                    host=jax.tree_util.tree_map(np.asarray,value);total=time.perf_counter()-st;field=(host[0] if method['kind']=='weak_lm' else host).reshape(6,3,n,n,n)
                    row=dict(case=case,repetition=repetition,order_index=oi,method=method['name'],gpu_seconds=device,with_host_seconds=total,field_sha256=sha(field),finite=bool(np.isfinite(field).all()))
                    key=f'{method["name"]}__case{case}';saved_key=key if repetition==0 or sha(saved[key])==row['field_sha256'] else key+f'__rep{repetition}'
                    if method['kind']=='weak_lm':
                        row['cold']=host[1];row['steps']=host[2];row['state_sha256']=sha(host[3]);histories[f'{key}__rep{repetition}']=host[3]
                    row['same_grid_errors']=np.linalg.norm((field-refs[case]).reshape(6,-1),axis=1)/np.linalg.norm(Udev[case,0]);rows.append(row);invocations.append((row,field));saved[saved_key]=field
                for row,field in invocations:
                    lift=field
                    for axis in (-3,-2,-1):lift=resample(lift,cfg['fine_reference_n'],axis=axis)
                    row['fine_grid_errors']=np.linalg.norm((lift-fine[case]).reshape(6,-1),axis=1)/np.linalg.norm(fine[case,0])
                write(rows,out/'timing_rows.json');print('TIMED',case,repetition,flush=True)
        np.savez_compressed(out/'timed_fields.npz',**saved);np.savez_compressed(out/'latent_histories.npz',**histories)
        report['timing_summary']={}
        for method in methods:
            arm=[r for r in rows if r['method']==method['name']];times=np.array([r['gpu_seconds'] for r in arm]);median=float(np.median(times))
            report['timing_summary'][method['name']]=dict(kind=method['kind'],invocations=len(arm),device_seconds_median=median,
                timing_outliers_above_three_median=int(np.sum(times>3*median)),worst_evolved_same_grid=max(max(r['same_grid_errors'][1:]) for r in arm),
                worst_evolved_fine_grid=max(max(r['fine_grid_errors'][1:]) for r in arm),compile_warm_seconds=method['compile_warm_seconds'])
        report['complete']=True;stage('complete');print('ALL-DONE',flush=True)
    except Exception as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc();write(report,result);print(report['traceback'],flush=True);raise


if __name__=='__main__':main()
