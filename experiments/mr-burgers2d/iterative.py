"""Current frozen Burgers NMROM against named same-grid iterative FOMs."""
import argparse, hashlib, json, os, pickle, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import engines as e
import iterative_paths as ip


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def host(x):return jax.tree_util.tree_map(np.asarray,x)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    begin=time.perf_counter();physical=e.params_draw(cfg['seed'],cfg['cases']);meshes=cfg['meshes'];largest=max(meshes)
    ck=pickle.load(open(a.checkpoint,'rb'));params=jax.tree_util.tree_map(jnp.asarray,ck['params']);Z=ck['Z_tr'];K=Z.shape[1];R=params['h_lin'].shape[1]
    report=dict(config=cfg,commit=os.environ.get('COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),backend=jax.default_backend(),gpu=jax.devices()[0].device_kind,
        x64=jax.config.jax_enable_x64,matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],jax_version=jax.__version__,checkpoint_sha256=sha(a.checkpoint),
        checkpoint_cfg=ck['cfg'],K=K,R=R,M=4*K,m=16*K,network_weights_frozen=True,physical_cases=physical.tolist(),cohort_role='existing development seed; final cohort unopened',
        output_times=[0,.05,.1,.15,.2,.25],timing_contract='supplied full initial field on GPU to six full GPU output fields; same-invocation host transfers also measured',
        reference=[],reference_metrics={},mesh_setup=[],invocations=[],declared_subjects=[],verification=ip.verify(),complete=False)
    # Config sometimes contains arrays used only in past training.
    report['checkpoint_cfg']={k:v for k,v in ck['cfg'].items() if k not in ['hfit_pick','train']}
    save=lambda:dump(out/'result.json',report);save()
    rf=cfg['reference_mesh'];rt=cfg['reference_dt']
    settings=[(rf//2,rt),(rf,2*rt),(rf,rt),(rf//4,2*rt),(rf//2,2*rt),(rf,4*rt)]
    refs={}
    for L,dt in settings:
        print('REFERENCE',L,dt,flush=True);q,_=e.make_fom(L,dt)
        for case,phys in enumerate(physical):
            t=time.perf_counter();f,it,rn=host(q(jnp.asarray(e.initial(L,phys)),float(phys[4]),1e-11,1e-9))
            assert np.isfinite(f).all() and np.max(rn)<2e-11
            f=np.array(f[:,::L//largest,::L//largest],copy=True);refs[L,dt,case]=f
            name=f'ref_L{L}_dt{dt}_case{case}.npz';np.savez_compressed(out/name,fields=f,iterations=it,residuals=rn)
            report['reference'].append(dict(intervals=L,dt=dt,case=case,artifact=name,max_relative_residual=float(np.max(rn)),seconds=time.perf_counter()-t));save()
        del q;jax.clear_caches()
    for L in meshes:
        rows=[]
        for case in range(cfg['cases']):
            def diff(L1,t1,L2,t2):return e.errors(refs[L1,t1,case][:,::largest//L,::largest//L],refs[L2,t2,case][:,::largest//L,::largest//L],L)['fixed_initial_max']
            space=diff(rf//2,rt,rf,rt);temporal=diff(rf,2*rt,rf,rt);sc=diff(rf//4,2*rt,rf//2,2*rt);sf=diff(rf//2,2*rt,rf,2*rt);tc=diff(rf,4*rt,rf,2*rt)
            rows.append(dict(case=case,space_difference=space,time_difference=temporal,margin=space+temporal,spatial_coarse=sc,spatial_fine=sf,temporal_coarse=tc,
                observed_space_order=float(np.log2(sc/sf)),observed_time_order=float(np.log2(tc/temporal)),decrease=bool(sf<sc and temporal<tc)))
        report['reference_metrics'][str(L)]=rows
    save();rng=np.random.default_rng(cfg['order_seed'])
    for L in meshes:
        print('ASSEMBLE',L,flush=True)
        data,info=e.build_rom(params,Z,L,4*K,16*K,candidate_cap=cfg['candidate_cap'],fit_states=cfg['fit_states'])
        cold,cinfo=e.build_gauss_cold(params,data[7]);info['cold']=cinfo
        # Save only small preassembled operators. The full learned bank is reproducible from the frozen checkpoint.
        np.savez_compressed(out/f'operators_L{L}.npz',A=np.asarray(data[1]),lam=np.asarray(data[2]),G5=np.asarray(data[3]),Pq=np.asarray(data[4]),
            candidate_Z=np.asarray(data[7]),cold_xy=np.asarray(cold[0]),cold_w=np.asarray(cold[1]),cold_Q=np.asarray(cold[2]),cold_R=np.asarray(cold[3]))
        rom=ip.make_rom(params,L,cfg['dt'],info['trust_radius']);foms={}
        for pc in ['fft','dense']:foms[pc]=ip.make_fom(L,cfg['dt'],pc)
        subjects=[dict(name='nmrom',method='rom',dt=cfg['dt'])]+[dict(method='fom',dt=cfg['dt'],**s) for s in cfg['fom_settings']]
        report['declared_subjects'] += [dict(intervals=L,**s) for s in subjects]
        inputs=[e.initial(L,phys) for phys in physical];gpu_inputs=[jnp.asarray(u) for u in inputs];jax.block_until_ready(gpu_inputs)
        def invoke(s,u,case):
            nu=float(physical[case,4])
            if s['method']=='rom':return rom(u,nu,data,cold)
            q,predata=foms[s['preconditioner']];return q(u,nu,s['ntol'],s['ltol'],*predata)
        t=time.perf_counter()
        for s in subjects:jax.block_until_ready(invoke(s,gpu_inputs[0],0))
        info['compile_warmup_seconds']=time.perf_counter()-t
        # Verify the instrumented ROM preserves the previously audited complete query.
        old=e.make_gauss_rom(params,L,cfg['dt'],info['trust_radius'],stall=.01,ic_budget=180)
        oldvalue=host(old(gpu_inputs[0],float(physical[0,4]),data,cold));newvalue=host(invoke(subjects[0],gpu_inputs[0],0))
        info['instrumented_rom_relative_parity']=float(np.linalg.norm(oldvalue[0]-newvalue[0])/np.linalg.norm(oldvalue[0]))
        info['instrumented_rom_counter_parity']=bool(all(np.array_equal(oldvalue[j],newvalue[j]) for j in [1,3,5,6]))
        assert info['instrumented_rom_relative_parity']<1e-11 and info['instrumented_rom_counter_parity']
        del old,oldvalue,newvalue;report['mesh_setup'].append(info);save()
        artifacts={}
        for rep in range(cfg['reps']):
            for case in range(cfg['cases']):
                truth=refs[rf,rt,case][:,::largest//L,::largest//L]
                for i in rng.permutation(len(subjects)):
                    s=subjects[i];e.burn(cfg['burn_seconds'])
                    ht=time.perf_counter();u=jax.device_put(np.array(inputs[case],copy=True));jax.block_until_ready(u)
                    gt=time.perf_counter();value=invoke(s,u,case);jax.block_until_ready(value);gs=time.perf_counter()-gt
                    # All full diagnostic fields are blocked above; only public fields are transferred inside the host timer.
                    f=np.asarray(value[0]);hs=time.perf_counter()-ht;v=host(value)
                    assert np.isfinite(f).all() and all(np.isfinite(x).all() for x in v)
                    h=hashlib.sha256(f.tobytes()).hexdigest();key=(s['name'],case,h)
                    if key not in artifacts:
                        filename=f'L{L}_{s["name"]}_case{case}_rep{rep}.npz'
                        extra=dict(internal_latents=v[7]) if s['method']=='rom' else dict(previous_output_steps=v[4])
                        np.savez_compressed(out/filename,fields=f,**extra);artifacts[key]=filename
                    row=dict(intervals=L,nodes_per_axis=L+1,case=case,rep=rep,**s,gpu_seconds=gs,host_seconds=hs,output_bytes=f.nbytes,
                        field_sha256=h,artifact=artifacts[key],error=e.errors(f,truth,L),iterations=v[1].tolist(),residuals=v[2].tolist(),finite=True)
                    if s['method']=='rom':row.update(stop_reasons=v[3].tolist(),latent_states=v[4].tolist(),ic_iterations=int(v[5]),ic_reason=int(v[6]),internal_latents=v[7].tolist())
                    else:row.update(linear_relative_residuals=v[3].tolist(),nonlinear_converged=bool(np.max(v[2])<=s['ntol']*(1+1e-9)),
                        linear_converged=bool(all(np.max(x[:int(n)],initial=0.)<=s['ltol']*(1+1e-7) for x,n in zip(v[3],v[1]))))
                    report['invocations'].append(row);save()
            print('TIMED',L,rep,'elapsed',time.perf_counter()-begin,flush=True)
        del data,cold,rom,foms,gpu_inputs;jax.clear_caches()
    report['checkpoint_sha256_after']=sha(a.checkpoint);assert report['checkpoint_sha256']==report['checkpoint_sha256_after']
    report['elapsed_seconds']=time.perf_counter()-begin;report['complete']=True;save();(out/'COMPLETE').write_text('complete\n')
    print('ITERATIVE BURGERS COMPLETE',flush=True)

if __name__=='__main__':main()
