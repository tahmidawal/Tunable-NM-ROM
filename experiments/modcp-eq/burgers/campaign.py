"""Resumable train/validate/frozen-test Burgers2D architecture campaign.

No final-test parameter draw occurs until selections.json has been frozen.
Full-cohort validation is measured once, with a predeclared seven-repeat case0
selection timing proxy. Frozen final evaluation retains seven repetitions/case.
"""
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from common.decoders import DecoderConfig,init_decoder,add_modulation,initial_codes,decode_grid
from common.training import train,save_checkpoint
from common.seal import verify_validation_seal
from burgers import fom,kernels as k


def clean(value):
    if isinstance(value,dict):return {str(a):clean(b) for a,b in value.items()}
    if isinstance(value,(tuple,list)):return [clean(x) for x in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def host(tree):return jax.tree_util.tree_map(np.asarray,tree)


def checkpoint_load(path):
    with Path(path).open('rb') as f:p=pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray,p['params']),jnp.asarray(p['Z']),DecoderConfig(**p['config'])


def stage_complete(path,steps,total_updates=None):
    if not Path(path).exists():return False
    with Path(path).open('rb') as f:extra=pickle.load(f).get('extra',{})
    return extra.get('steps')==steps or (total_updates is not None and extra.get('updates')==total_updates)


class Campaign:
    def __init__(self,args):
        self.args=args;self.out=Path(args.out);self.out.mkdir(parents=True,exist_ok=True)
        self.smoke=args.smoke
        self.meshes=[16,32] if args.smoke else [256,512]
        self.L=self.meshes[0];self.counts={'train':2 if args.smoke else 64,'validation':1 if args.smoke else 16,'evaluation':1 if args.smoke else 16}
        self.seeds={'train':2609101701,'validation':2609101702,'evaluation':2609101703}
        self.cfg=DecoderConfig(intervals=self.L,k=4 if args.smoke else 16,rank=8 if args.smoke else 64,
                               head_width=24 if args.smoke else 256,width=12 if args.smoke else 64,inr_width=24 if args.smoke else 128)
        self.reps=1 if args.smoke else 7;self.split_reps={'validation':1,'evaluation':self.reps};self.target=[.01,.05]
        self.provenance=dict(commit=os.environ.get('COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),
          gpu=jax.devices()[0].device_kind,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,
          matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),jax_version=jax.__version__,
          training_seed=self.seeds['train'],smoke=args.smoke)
        assert self.provenance['backend']=='gpu' and self.provenance['x64'] and self.provenance['matmul_precision']=='highest'
        self.config=dict(decoder=self.cfg.to_dict(),meshes=self.meshes,counts=self.counts,seeds=self.seeds,
          validation_case_ids=list(range(self.counts['validation'])),evaluation_case_ids=list(range(self.counts['evaluation'])),
          repetitions=self.reps,split_repetitions=self.split_reps,selection_proxy_case=0,selection_proxy_repetitions=self.reps,
          validation_repetitions=1,evaluation_repetitions=self.reps,discarded_warmups=2,targets=self.target,output_times=[0,.05,.10,.15,.20,.25],
          training_updates=dict(cp_initialization=20 if args.smoke else 6000,comparison=20 if args.smoke else 10000,film=40 if args.smoke else 16000),
          training_batch_size=64,point_batch=256,lr_initial=.001,lr_final=.0001,stage_cosine_schedule='reset for each6k/10k stage; paired CP/modCP share6k initialization',
          timing_contract='GPU resident supplied dense field+viscosity to six GPU dense output fields; initialization+evolution+decoding included; transfer/setup/compile excluded',
          boundary_transfer='clipped training-interior coordinates and linear boundary strip; exact training-node predictions, no masked endpoint leakage')
        cp=self.out/'config.json'
        if cp.exists():assert json.loads(cp.read_text())==clean(self.config)
        else:write_json(cp,self.config)
        self.rows=[];self.finished=set();self.log=self.out/'invocations.jsonl'
        self.physical_cases=json.loads((self.out/'physical_cases.json').read_text()) if (self.out/'physical_cases.json').exists() else {}
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                row=json.loads(line);self.rows.append(row);self.finished.add(self.row_id(row))
        self.selections=json.loads((self.out/'selections.json').read_text()) if (self.out/'selections.json').exists() else []
        self.selection_timings=[];self.selection_log=self.out/'selection_timings.jsonl'
        if self.selection_log.exists():self.selection_timings=[json.loads(line) for line in self.selection_log.read_text().splitlines()]
        self.timing_archives=json.loads((self.out/'timing_archives.json').read_text()) if (self.out/'timing_archives.json').exists() else []
        current=(self.provenance['job_id'],self.provenance['gpu'])
        identities={(r.get('provenance',{}).get('job_id'),r.get('provenance',{}).get('gpu')) for r in self.rows+self.selection_timings}
        if any(identity!=current for identity in identities):
            # A new GPU allocation may reuse trained models, reference fields and
            # quadrature. It must never splice old wall clocks into a new panel.
            number=len(self.timing_archives);archived={}
            for label,path in [('invocations',self.log),('selection_timings',self.selection_log)]:
                if path.exists():
                    dest=path.with_name(f'{path.stem}_prior_allocation_{number}.jsonl')
                    assert not dest.exists();path.replace(dest)
                    archived[label]=str(dest.relative_to(self.out))
            self.timing_archives.append(dict(artifacts=archived,allocation_identities=[list(i) for i in identities],
                reason='new allocation: timing panels restarted; frozen selection, if already created, remains an offline validation choice'))
            write_json(self.out/'timing_archives.json',self.timing_archives)
            self.rows=[];self.finished=set();self.selection_timings=[]
        self.status='initialized';self.save()

    @staticmethod
    def row_id(row):return (row['split'],row['configuration'],row['intervals'],row['case'],row['rep'])

    def save(self):
        checkpoints={p.stem:sha(p) for p in self.out.glob('checkpoints/*.pkl') if not p.stem.endswith('trainstate')}
        write_json(self.out/'handoff.json',dict(case_name='burgers2d',status=self.status,config=self.config,
          provenance=self.provenance,checkpoint_hashes=checkpoints,invocations=self.rows,selections=self.selections,selection_timings=self.selection_timings,timing_archives=self.timing_archives,
          physical_cases=self.physical_cases,physical_parameter_columns=['center_x','center_y','width','amplitude','viscosity'],
          artifacts=dict(raw_invocations='invocations.jsonl',training='checkpoints',rules='rules',fields='fields',references='references')))

    def append(self,row):
        row=clean(row)
        with self.log.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.rows.append(row);self.finished.add(self.row_id(row))

    def dataset(self,split,L,dt):
        assert split!='evaluation' or (self.out/'selections.json').exists(),'final cohort stays sealed until frozen selection'
        dest=self.out/'references';dest.mkdir(exist_ok=True)
        path=dest/f'{split}_L{L}_dt{dt}.npz'
        if path.exists():
            loaded=np.load(path);self.physical_cases[split]=loaded['physical'].tolist()
            write_json(self.out/'physical_cases.json',self.physical_cases)
            return loaded['fields'],loaded['physical'],loaded['iterations'],loaded['residuals']
        physical=fom.params_draw(self.seeds[split],self.counts[split]);q,_=fom.make_fom(L,dt)
        self.physical_cases[split]=physical.tolist();write_json(self.out/'physical_cases.json',self.physical_cases)
        fields=[];iterations=[];residuals=[]
        for case,p in enumerate(physical):
            print(f'DATA {split} L={L} dt={dt} case={case}',flush=True)
            a,it,r=host(q(jnp.asarray(fom.initial(L,p)),float(p[4]),1e-10,1e-8))
            assert np.isfinite(a).all() and np.max(r)<2e-10,(split,L,case,np.max(r))
            fields.append(a);iterations.append(it);residuals.append(r)
        fields=np.asarray(fields);iterations=np.asarray(iterations);residuals=np.asarray(residuals)
        np.savez_compressed(path,fields=fields,physical=physical,iterations=iterations,residuals=residuals)
        return fields,physical,iterations,residuals

    def training(self):
        self.status='training';self.save()
        ck=self.out/'checkpoints';ck.mkdir(exist_ok=True)
        presteps=self.config['training_updates']['cp_initialization'];steps=self.config['training_updates']['comparison']
        filmsteps=self.config['training_updates']['film']
        if (all(stage_complete(ck/f'{arm}.pkl',steps,presteps+steps) for arm in ('cp','modcp'))
            and stage_complete(ck/'film.pkl',filmsteps,filmsteps)):
            return
        fields,physical,_,_=self.dataset('train',self.L,.00125)
        states=jnp.asarray(fields.reshape((-1,(self.L+1)**2,1)))
        # Fixed physical normalization: one initial-field RMS per trajectory,
        # retained at all six times. No true parameter is a model input.
        scale=np.sqrt(np.mean(fields[:,0]**2,axis=(1,2)))
        scales=jnp.asarray(np.repeat(scale,6)[:,None]);xy=jnp.asarray(fom.coords(self.L,False))
        codes=initial_codes(len(states),self.cfg.k,self.seeds['train']+11)
        if stage_complete(ck/'cp_pretrain.pkl',presteps):p,preZ,_=checkpoint_load(ck/'cp_pretrain.pkl')
        else:
            p=init_decoder(jax.random.PRNGKey(self.seeds['train']+7),self.cfg)
            p,preZ,_=train(p,codes,states,xy,self.cfg,steps=presteps,seed=self.seeds['train']+101,
                          scales=scales,out_dir=ck,stage_name='cp_pretrain')
        for arm in ('cp','modcp'):
            if stage_complete(ck/f'{arm}.pkl',steps,presteps+steps):continue
            cfg=replace(self.cfg,architecture=arm)
            params=p if arm=='cp' else add_modulation(p,cfg,jax.random.PRNGKey(self.seeds['train']+13))
            trained,Z,hist=train(params,preZ,states,xy,cfg,steps=steps,seed=self.seeds['train']+102,
                                scales=scales,out_dir=ck,stage_name=arm)
            save_checkpoint(ck/f'{arm}.pkl',trained,Z,cfg,dict(training_cases=len(physical),updates=presteps+steps,parent='cp_pretrain',history=hist))
        if not stage_complete(ck/'film.pkl',filmsteps,filmsteps):
            cfg=replace(self.cfg,architecture='film');p=init_decoder(jax.random.PRNGKey(self.seeds['train']+7),cfg)
            p,Z,hist=train(p,codes,states,xy,cfg,steps=self.config['training_updates']['film'],seed=self.seeds['train']+103,
                          scales=scales,out_dir=ck,stage_name='film')
            save_checkpoint(ck/'film.pkl',p,Z,cfg,dict(training_cases=len(physical),updates=self.config['training_updates']['film'],history=hist))
        self.status='trained';self.save()

    def rule(self,arm,L,multiplier):
        p,Z,cfg=checkpoint_load(self.out/'checkpoints'/f'{arm}.pkl')
        dest=self.out/'rules';dest.mkdir(exist_ok=True);path=dest/f'{arm}_L{L}_q{multiplier}.pkl'
        if path.exists():
            with path.open('rb') as f:s=pickle.load(f)
            assert s['checkpoint_hash']==sha(self.out/'checkpoints'/f'{arm}.pkl')
            return p,Z,cfg,k.precontract_mass(jax.tree_util.tree_map(jnp.asarray,s['data']),cfg),s['info']
        print(f'RULE {arm} L={L} multiplier={multiplier}',flush=True)
        data,info=k.build_rule(p,Z,cfg,L,4*cfg.k,multiplier*4*cfg.k,
                              fit_states=4 if self.smoke else 32,candidate_cap=8192)
        with path.open('wb') as f:pickle.dump(dict(data=host(data),info=info,checkpoint_hash=sha(self.out/'checkpoints'/f'{arm}.pkl')),f,protocol=5)
        write_json(path.with_suffix('.json'),info)
        return p,Z,cfg,k.precontract_mass(data,cfg),info

    def references(self,split,L):
        # Same-grid tight discrete reference is primary for a solver architecture
        # test; a nested finer reference makes physical accuracy uncertainty visible.
        fine_dt=.000625
        ref,physical,_,_=self.dataset(split,L,fine_dt)
        fine,_,_,_=self.dataset(split,2*L,fine_dt/2)
        coarse_dt,_,_,_=self.dataset(split,L,2*fine_dt)
        truth=fine[:, :, ::2, ::2]
        rows=[]
        for case in range(len(physical)):
            space_time=k.errors(ref[case],truth[case])['displacement']
            temporal=k.errors(coarse_dt[case],ref[case])['displacement']
            rows.append(dict(case=case,intervals=L,nested_space_time_difference=space_time,
                             temporal_difference=temporal,empirical_uncertainty=space_time+temporal,
                             strict_continuum_bound=False,provisional_targets=[t for t in self.target if space_time+temporal>.1*t]))
        write_json(self.out/'references'/f'{split}_L{L}_uncertainty.json',rows)
        return ref,truth,physical

    def configs(self):
        if self.smoke:return [dict(q=4,dt=.005,cap=10,tol=1e-4)]
        return [dict(q=q,dt=dt,cap=cap,tol=tol) for q,dt,cap,tol in itertools.product([4,8],[.005,.0025,.00125],[10,30],[1e-4,1e-6])]

    @staticmethod
    def name(arm,settings):return arm+'_'+('_'.join(f'{a}{settings[a]}' for a in sorted(settings)))

    def run_split(self,split):
        if split=='evaluation' and not self.smoke:
            self.provenance.update(verify_validation_seal(self.out,'burgers2d',self.seeds['evaluation']))
        self.status=split;self.save();fielddir=self.out/'fields';fielddir.mkdir(exist_ok=True)
        reps=self.split_reps[split]
        for L in self.meshes:
            same,truth,physical=self.references(split,L)
            inputs=[jnp.asarray(fom.initial(L,p)) for p in physical]
            subjects=[];objects={};compiled={};rule_cache={}
            selected=set(s['configuration'] for s in self.selections if s['intervals']==L and s['configuration'] is not None)
            for arm in ('cp','modcp','film'):
                p,Z,cfg=checkpoint_load(self.out/'checkpoints'/f'{arm}.pkl')
                ic=k.build_initial_data(p,Z,cfg,L)
                for settings in self.configs():
                    name=self.name(arm,settings)
                    if split=='evaluation' and name not in selected:continue
                    rulekey=(arm,settings['q'])
                    if rulekey not in rule_cache:rule_cache[rulekey]=self.rule(arm,L,settings['q'])
                    _,_,_,data,info=rule_cache[rulekey]
                    kernelkey=(arm,settings['q'],settings['dt'],settings['cap'])
                    if kernelkey not in compiled:compiled[kernelkey]=k.make_rom(cfg,L,settings['dt'],settings['cap'])
                    fun,parts=compiled[kernelkey]
                    subjects.append((name,arm,settings,fun,(p,data,ic),parts))
                    objects[name]=(p,Z,cfg,data,info,ic)
            fconfigs=[dict(dt=.005,ntol=1e-4,ltol=.1)] if self.smoke else [dict(dt=dt,ntol=tol,ltol=.1) for dt,tol in itertools.product([.005,.0025,.00125],[1e-2,1e-4,1e-6])]
            for settings in fconfigs:
                name=self.name('newton_bicgstab',settings)
                if split=='evaluation' and name not in selected:continue
                fun,_=fom.make_fom(L,settings['dt'])
                subjects.append((name,'newton_bicgstab',settings,fun,(),None))
            def invoke(sub,case):
                name,arm,settings,fun,packed,_=sub;nu=float(physical[case,4])
                if arm=='newton_bicgstab':return fun(inputs[case],nu,settings['ntol'],settings['ltol'])
                return fun(inputs[case],nu,*packed,settings['tol'])
            # Compile and two explicit discarded calls for every configuration.
            for sub in subjects:
                if all((split,sub[0],L,c,r) in self.finished for c in range(len(physical)) for r in range(reps)):continue
                t=time.perf_counter()
                for _ in range(2):jax.block_until_ready(invoke(sub,0))
                print(f'WARM {split} L={L} {sub[0]} seconds={time.perf_counter()-t:.3f}',flush=True)
            for rep in range(reps):
                k.burn(.25 if self.smoke else .75)
                order=subjects if rep%2==0 else subjects[::-1]
                for case in range(len(physical)):
                    for sub in order:
                        name,arm,settings,fun,packed,parts=sub
                        if (split,name,L,case,rep) in self.finished:continue
                        # Host artifact compression and diagnostics between calls
                        # may lower clocks; burn in before each measured region.
                        k.burn(.05 if self.smoke else .75)
                        jax.block_until_ready(inputs[case]);start=time.perf_counter()
                        result=jax.block_until_ready(invoke(sub,case));elapsed=time.perf_counter()-start
                        h=host(result);fields=h[0];finite=bool(np.isfinite(fields).all())
                        if arm=='newton_bicgstab':
                            it,rn=h[1:];completed=bool(finite and np.isfinite(rn).all() and np.max(rn)<=settings['ntol']*(1+1e-8))
                            stop='converged' if completed else 'nonlinear_failure'
                            detail=dict(iterations=it.tolist(),residuals=rn.tolist(),converged=completed)
                        else:
                            Zo,(Zsteps,it,reason,stat,rn),initial=h[1:]
                            numerical_full_horizon=bool(finite and np.isfinite(stat).all() and not np.isin(reason,[3,4,5]).any() and int(initial[1]) not in [3,4,5])
                            stationary=bool(numerical_full_horizon and np.all(reason==1) and int(initial[1])==1)
                            completed=numerical_full_horizon
                            stop='stationary' if stationary else ('budget_or_small_step' if numerical_full_horizon else 'solver_failure')
                            detail=dict(iterations=it.tolist(),reasons=reason.tolist(),stationarity=stat.tolist(),residuals=rn.tolist(),initial_fit=list(initial),
                                        converged=stationary,numerical_full_horizon=numerical_full_horizon,latent_max_norm=float(np.max(np.linalg.norm(Zsteps,axis=1))),latent_output_states=Zo.tolist())
                        row=dict(split=split,method=arm,configuration=name,settings=settings,intervals=L,case=case,rep=rep,seconds=elapsed,
                                 errors=k.errors(fields,truth[case]) if finite else None,same_grid_errors=k.errors(fields,same[case]) if finite else None,
                                 finite=finite,completed=completed,stop_reason=stop,solver=detail,
                                 field_sha256=hashlib.sha256(fields.tobytes()).hexdigest(),output_bytes=fields.nbytes,
                                 provenance=self.provenance)
                        row['stationary']=bool(detail['converged'])
                        row['linear_mass_precontracted']=arm=='cp'
                        if rep==0:
                            # Full output/truth retained once for each deterministic
                            # configuration/case; all repetitions retain output hash.
                            artifact=fielddir/f'{split}_{name}_L{L}_case{case}.npz'
                            np.savez_compressed(artifact,u=fields,truth_u=truth[case],same_grid_u=same[case])
                            row.update(field_artifact=str(artifact.relative_to(self.out)),field_artifact_kind='self_contained_full_grid')
                            if arm!='newton_bicgstab' and finite and np.isfinite(Zo).all() and np.isfinite(Zsteps).all():
                                p,Z,cfg,data,info,ic=objects[name]
                                prev=jnp.asarray(Zo[-2]);last=jnp.asarray(Zo[-1])
                                # Last physical output-to-output interval is not one
                                # solver step: use the actual preceding latent step.
                                prev=jnp.asarray(Zsteps[-2]) if len(Zsteps)>1 else jnp.asarray(Zo[0])
                                prevm,_=k.moments(p,prev,data,cfg)
                                full=np.asarray(k.full_weak(last,prev,float(physical[case,4]),p,cfg,L,4*cfg.k,settings['dt'],float(initial[4])))
                                sampled=np.asarray(k.weak(last,prevm,float(physical[case,4]),p,data,cfg,L,settings['dt'],float(initial[4])))
                                J=np.asarray(jax.jacfwd(lambda z:k.weak(z,prevm,float(physical[case,4]),p,data,cfg,L,settings['dt'],float(initial[4])))(last))
                                try:singular=np.linalg.svd(J,compute_uv=False) if np.isfinite(J).all() else np.array([])
                                except np.linalg.LinAlgError:singular=np.array([])
                                row['weak_audit']=dict(full_norm=float(np.linalg.norm(full)),sampled_norm=float(np.linalg.norm(sampled)),difference_norm=float(np.linalg.norm(full-sampled)),
                                   relative_difference=float(np.linalg.norm(full-sampled)/max(np.linalg.norm(full),1e-30)),tangent_singular_values=singular.tolist(),
                                   tangent_rank=int(np.sum(singular>max(singular[0]*1e-10,1e-24))) if len(singular) else None,
                                   audit_latent_step=len(Zsteps),finite_jacobian=bool(np.isfinite(J).all()))
                            elif arm!='newton_bicgstab':
                                row['weak_audit']=dict(status='not_evaluated_nonfinite_latent_or_fields',tangent_rank=None)
                        self.append(row)
                    print(f'TIMED {split} L={L} rep={rep} case={case} rows={len(self.rows)}',flush=True);self.save()
            if split=='validation':
                existing={(x['intervals'],x['configuration'],x['rep']) for x in self.selection_timings}
                for rep in range(self.reps):
                    for sub in (subjects if rep%2==0 else subjects[::-1]):
                        name,arm,settings,_,_,_=sub
                        if (L,name,rep) in existing:continue
                        k.burn(.05 if self.smoke else .75)
                        jax.block_until_ready(inputs[0]);t=time.perf_counter()
                        result=jax.block_until_ready(invoke(sub,0));seconds=time.perf_counter()-t
                        fields=np.asarray(result[0]);finite=bool(np.isfinite(fields).all())
                        row=clean(dict(intervals=L,configuration=name,method=arm,case=0,rep=rep,seconds=seconds,
                            errors=k.errors(fields,truth[0]) if finite else None,finite=finite,
                            role='predeclared validation case0 selection timing proxy; not final claim',provenance=self.provenance,
                            field_sha256=hashlib.sha256(fields.tobytes()).hexdigest()))
                        with self.selection_log.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        self.selection_timings.append(row)
                    self.save()
            # Explicit component timings of the same staged query, never used in
            # the complete-query time or cross-job speedup denominator.
            if split=='evaluation':
                profile=[]
                for sub in subjects:
                    name,arm,settings,_,packed,parts=sub
                    if parts is None:continue
                    p,data,ic=packed
                    rfun=jax.jit(parts['residual']);jfun=jax.jit(jax.jacfwd(parts['residual']))
                    def staged():
                        ini=jax.block_until_ready(parts['initialize'](inputs[0],p,ic))
                        zz,detail=jax.block_until_ready(parts['evolve'](ini[0],float(physical[0,4]),ini[1],p,data,settings['tol']))
                        f=jax.block_until_ready(parts['reconstruct'](zz,p))
                        return ini,zz,detail,f
                    for _ in range(2):ini,zz,detail,f=staged()
                    z=zz[-1];previous=detail[0][-2] if len(detail[0])>1 else zz[0]
                    prevm,_=k.moments(p,previous,data,objects[name][2])
                    rargs=(z,prevm,float(physical[0,4]),p,data,ini[1])
                    for _ in range(2):jax.block_until_ready((rfun(*rargs),jfun(*rargs)))
                    repetitions=[]
                    for rep in range(self.reps):
                        k.burn(.05 if self.smoke else .75)
                        t=time.perf_counter();ini=jax.block_until_ready(parts['initialize'](inputs[0],p,ic));t1=time.perf_counter()
                        zz,detail=jax.block_until_ready(parts['evolve'](ini[0],float(physical[0,4]),ini[1],p,data,settings['tol']));t2=time.perf_counter()
                        f=jax.block_until_ready(parts['reconstruct'](zz,p));t3=time.perf_counter()
                        rr=jax.block_until_ready(rfun(*rargs));t4=time.perf_counter()
                        jj=jax.block_until_ready(jfun(*rargs));t5=time.perf_counter()
                        repetitions.append(dict(rep=rep,initialization_s=t1-t,evolution_s=t2-t1,reconstruction_s=t3-t2,
                           residual_s=t4-t3,jacobian_s=t5-t4,physical_error=k.errors(np.asarray(f),truth[0])))
                    profile.append(dict(configuration=name,intervals=L,case=0,repetitions=repetitions,
                                        includes_component_compilation=False,interpretation='separately synchronized warmed components; never replace fused query timing'))
                write_json(self.out/f'profiles_L{L}.json',profile)
                reconstruction=[]
                for arm in ('cp','modcp','film'):
                    representative=next((sub for sub in subjects if sub[1]==arm),None)
                    if representative is None:continue
                    name,_,_,_,(p,data,ic),parts=representative
                    cfg=objects[name][2]
                    decode=jax.jit(lambda parameters,z:decode_grid(parameters,z,L,cfg)[...,0])
                    for case in range(len(physical)):
                        norm=max(float(np.linalg.norm(truth[case,0])),1e-30)
                        per_time=[];fit_reasons=[];fit_stationarity=[]
                        for frame in truth[case]:
                            fit=jax.block_until_ready(parts['initialize'](jnp.asarray(frame),p,ic))
                            output=np.asarray(decode(p,fit[0]))
                            per_time.append(float(np.linalg.norm(output-frame)/norm))
                            fit_reasons.append(int(fit[3]));fit_stationarity.append(float(fit[4]))
                        reconstruction.append(dict(method=arm,intervals=L,case=case,error_max=max(per_time),errors=per_time,
                            reasons=fit_reasons,stationarity=fit_stationarity,
                            interpretation='offline snapshot-wise sampled reconstruction fit; upper bound on best manifold approximation, never used to initialize or tune a query'))
                write_json(self.out/f'reconstruction_L{L}.json',reconstruction)
            del subjects,objects,inputs;same=truth=None;jax.clear_caches()
        self.save()

    def freeze(self):
        if (self.out/'selections.json').exists():return
        selections=[]
        for L in self.meshes:
            for arm in ('cp','modcp','film','newton_bicgstab'):
                rows=[r for r in self.rows if r['split']=='validation' and r['intervals']==L and r['method']==arm]
                groups={name:[r for r in rows if r['configuration']==name] for name in sorted(set(r['configuration'] for r in rows))}
                expected=self.counts['validation']*self.split_reps['validation']
                assert groups and all(len(g)==expected for g in groups.values()),'cannot freeze incomplete validation grid'
                timing={};proxies={}
                for name in groups:
                    proxy=[r for r in self.selection_timings if r['intervals']==L and r['configuration']==name]
                    assert len(proxy)==self.reps,'cannot freeze incomplete predeclared timing proxy'
                    timing[name]=float(np.median([r['seconds'] for r in proxy]))
                    proxies[name]=proxy
                for target in self.target:
                    passed=[(timing[name],name) for name,g in groups.items()
                            if all(r['finite'] and r['completed'] and r['errors']['displacement']<=target for r in g)
                            and all(r['finite'] and r['errors']['displacement']<=target for r in proxies[name])]
                    chosen=min(passed)[1] if passed else None
                    selections.append(dict(method=arm,intervals=L,target=target,configuration=chosen,validation_passed=bool(passed),role='target'))
                if not any(s['method']==arm and s['intervals']==L and s['configuration'] for s in selections):
                    # Retain a declared best-error diagnostic when no requested
                    # target is attained, without presenting it as target success.
                    scores=[]
                    for name,g in groups.items():
                        error=max(r['errors']['displacement'] if r['errors'] else float('inf') for r in g)
                        scores.append((error,timing[name],name))
                    selections.append(dict(method=arm,intervals=L,target=None,configuration=min(scores)[2],validation_passed=False,role='best_error_diagnostic'))
        self.selections=selections;write_json(self.out/'selections.json',selections)
        write_json(self.out/'selection_freeze.json',dict(selection_sha256=sha(self.out/'selections.json'),validation_invocations_sha256=sha(self.log),
            selection_timings_sha256=sha(self.selection_log),provenance=self.provenance))
        self.status='validation_frozen';self.save()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--phase',choices=['all','train','validate','evaluate'],default='all');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();c=Campaign(args)
    if args.phase in ['all','train']:c.training()
    if args.phase in ['all','validate']:c.run_split('validation');c.freeze()
    if args.phase in ['all','evaluate']:
        assert (c.out/'selections.json').exists();c.run_split('evaluation');c.status='complete_smoke' if args.smoke else 'complete';c.save()
    print('ALL-DONE '+c.status,flush=True)


if __name__=='__main__':main()
