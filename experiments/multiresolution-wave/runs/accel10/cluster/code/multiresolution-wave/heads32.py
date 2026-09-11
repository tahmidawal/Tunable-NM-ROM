"""Fixed-bank K32 reconstruction-only heads with independent time-step controls."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import dynamics as d
from dynamics import base,jax,jnp,Grid,parameter_rows,localized_initial,restrict,metrics,save_json,clean
from fresh_learning import train_head
from fresh_models import tree_from_npz,head_apply,count_parameters
from fresh_fom import provenance


def train_endpoints(inputs,bc,out,cfg,linear,center):
    original=json.loads((inputs/'campaign-config.json').read_text())
    original['latent']=cfg['new_latent_dimension']
    for key,value in cfg['training'].items():assert original[key]==value,(key,original[key],value)
    assert cfg['training_velocity_weight']==0 and not cfg['bank_retrained']
    with np.load(out/f'training_ladder_{bc}.npz') as f:
        a,b,scales=f['a'],f['b'],f['case_scales']
        coordinates=(a.reshape(-1,64)-center)@f['basis'][:,:32]
        codes=coordinates/f['scales'][:32]
    expected=json.loads((inputs/'data_manifest.json').read_text())['splits']['train']
    np.testing.assert_allclose(scales,expected['scales'],rtol=1e-12,atol=1e-14)
    output_scale=float(np.sqrt(np.mean(np.sum(a.reshape(-1,64)**2,axis=1))/64))
    common=(linear[:,:32],center,codes,output_scale)
    projected=dict(a=a,b=b,u_scale=np.repeat(scales[:,0],a.shape[1]),v_scale=np.repeat(scales[:,1],a.shape[1]))
    endpoints={};records=[]
    for seed in cfg['training']['optimizer_seeds']:
        name=f'new_mlp32_seed{seed}';dest=out/f'head_{bc}_{name}';dest.mkdir()
        start=time.perf_counter()
        p,frozen,z,history=train_head(original,projected,common,'mlp',0.,seed,dest)
        jax.block_until_ready((p,frozen,z));seconds=time.perf_counter()-start
        if not all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves((p,frozen,z))):raise RuntimeError('Nonfinite trained endpoint')
        endpoints[name]=dict(p=p,frozen=frozen,codes=z)
        records.append(dict(boundary=bc,name=name,optimizer_seed=seed,configuration_dimension=32,phase_dimension=64,
            training=cfg['training'],velocity_weight=0.,head_parameter_count=count_parameters(p),
            training_seconds_including_first_compile=seconds,history=history,
            checkpoint_path=str((dest/'head.npz').relative_to(out)),checkpoint_sha256=base.sha(dest/'head.npz'),
            initializer_linear_sha256=base.array_sha(common[0]),initializer_center_sha256=base.array_sha(center),
            initializer_codes_sha256=base.array_sha(codes),output_scale=output_scale,
            data_kind='Fresh regenerated original training coefficients and initial physical scales only.'))
    return endpoints,records


def adapt_model(bank,endpoint,linear,center,name):
    q=dict(bank);transform=jnp.asarray(bank['transform'])
    q.update(p=base.transform_head(endpoint['p'],transform),frozen=endpoint['frozen'],
        common_inverse=jnp.linalg.pinv(transform@jnp.asarray(linear[:,:32])),
        common_center=transform@jnp.asarray(center),
        fixed_codes=jnp.asarray(endpoint['codes'][np.linspace(0,len(endpoint['codes'])-1,6,dtype=int)]),model_name=name)
    return q


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out;out.mkdir(parents=True,exist_ok=False)
    meta=provenance();print(json.dumps(meta),flush=True)
    assert meta['jax_backend']=='gpu' and meta['x64'] and meta['matmul_precision']=='highest' and not cfg['final_test_opened']
    assert cfg['new_latent_dimension']<cfg['weak_bank_equations']==cfg['frozen_bank_rank']
    frozen=json.loads((base.FRESH/'FROZEN-MATH.json').read_text())
    for name,expected in frozen['sha256'].items():assert base.sha(base.FRESH/name)==expected
    result=dict(config=cfg,provenance=meta,frozen_mathematics=frozen,training=[],head_training=[],references=[],meshes=[],
        invocations=[],accuracy_controls=[],warmups=[],diagnostics=[],final_test_opened=False,
        input_sha256={str(p.relative_to(args.inputs)):base.sha(p) for p in args.inputs.rglob('*') if p.is_file()})
    pars=parameter_rows(cfg['validation_seed'],max(cfg['validation_indices'])+1)
    for bc in cfg['boundaries']:
        linear,center,manifest=d.regenerate_ladder(args.inputs/bc,bc,out,cfg)
        assert manifest['data_hashes_match'] and manifest['saved_initialization_matched']
        result['training'].append(manifest);save_json(out/'result.json',result)
        endpoints,headrecords=train_endpoints(args.inputs/bc,bc,out,cfg,linear,center)
        result['head_training'].extend(headrecords);save_json(out/'result.json',result)
        refs={}
        for ci in cfg['validation_indices']:
            refs[ci],audit=base.build_reference(bc,pars[ci],cfg)
            result['references'].append(dict(boundary=bc,case=ci,parameters=pars[ci],**audit))
            np.savez_compressed(out/f'reference_{bc}_{ci}.npz',u=refs[ci][0],v=refs[ci][1])
        coarse={}
        for n in cfg['meshes']:
            grid=Grid(n,bc,bc);common=Grid(cfg['comparison_intervals'],bc,bc)
            bank=base.rebuild(args.inputs/bc,grid);bank['model_name']='frozen_mlp16_seed691200'
            models={bank['model_name']:bank}
            models.update({name:adapt_model(bank,head,linear,center,name) for name,head in endpoints.items()})
            affine=next(a for a in d.arms_for(bank,args.inputs/bc,linear,center) if a['name']=='affine32')
            full=next(a for a in d.arms_for(bank,args.inputs/bc,linear,center) if a['name']=='full64')
            result['meshes'].append(dict(boundary=bc,intervals=n,audits=bank['audits'],assembly_seconds_including_first_compile=bank['assembly_seconds_including_first_compile']))
            arrays=dict(g=np.asarray(bank['g']),mass=np.asarray(bank['mass']),stiffness=np.asarray(bank['k']),damping=np.asarray(bank['d']),transform=bank['transform'])
            for arm in (affine,full):
                for key in ('basis','center','inverse'):arrays[arm['name']+'_'+key]=np.asarray(arm[key])
            np.savez_compressed(out/f'mesh_{bc}_{n}.npz',**arrays)
            methods=[('frozen_mlp16_seed691200',cfg['primary_dt'])]
            methods += [(name,dt) for name in endpoints for dt in cfg['nonlinear_dts']]
            methods += [('affine32',0.)]+([('dst',0.)] if bc=='dirichlet' else [('rk4',cf) for cf in cfg['fom_cfls']])
            for ci in cfg['validation_indices']:
                par=pars[ci];u0,v0=map(np.asarray,localized_initial(grid,par))
                su,sv,_,_=base.query('dst' if bc=='dirichlet' else 'rk4',0. if bc=='dirichlet' else cfg['reference_cfl'],u0,v0,par[5],grid,cfg)
                np.savez_compressed(out/f'samegrid_{bc}_{n}_{ci}.npz',u=su,v=sv,u0=u0,v0=v0)
                same_common=tuple(restrict(x,n,common.n,bc) for x in (su,sv))
                refrow=dict(boundary=bc,case=ci,same_grid_intervals=n,same_grid_to_physical_reference=metrics(*same_common,*refs[ci],common,par[5],cfg))
                if n==cfg['meshes'][0]:coarse[ci]=same_common
                elif bc=='absorbing':refrow['adjacent_coarse_to_this_mesh']=metrics(*coarse[ci],*same_common,common,par[5],cfg)
                result['references'].append(refrow)
                def call(name,setting):
                    if name=='affine32':return d.linear_query(affine,u0,v0,par[5],grid,cfg,bank)
                    if name in models:
                        u,v,rec,aux=base.query('rom',setting,u0,v0,par[5],grid,cfg,models[name]);rec['method']=name
                        return u,v,rec,aux
                    return base.query(name,setting,u0,v0,par[5],grid,cfg)
                def record_query(name,setting,rep,order,role):
                    print(role,bc,n,ci,rep,name,setting,flush=True);base.burn()
                    u,v,rec,aux=call(name,setting)
                    invocation=f'{bc}_{n}_{ci}_{rep}_{name}_{setting}'
                    if role=='accuracy_refinement_only':invocation+='_'+role
                    rec.update(case=ci,repetition=rep,order=order,parameters=par,invocation_id=invocation,
                        measurement_role=role,comparison_eligible=role=='timed_comparison',
                        warmup_performed=role=='timed_comparison',
                        timing_note='Full charged query; eligible repeated comparison.' if role=='timed_comparison' else 'Accuracy-only single query; may include uncached compilation. Latency is ineligible for speed comparison or selection.',
                        same_grid_discrepancy=metrics(u,v,su,sv,grid,par[5],cfg),
                        physical_reference_error=metrics(*(restrict(x,n,common.n,bc) for x in (u,v)),*refs[ci],common,par[5],cfg))
                    if name in models:
                        fits=aux['fits'];selected=int(fits['selected'])
                        rec['cold_fit']={k:clean(x) for k,x in fits.items() if k not in ('projected_u','projected_v')}
                        rec['fit_stationary']=bool(fits['finite'][selected] and fits['rank_ratio'][selected]>1e-8 and fits['gradient'][selected]<=1e-7 and (fits['stationarity'][selected]<=1e-6 or fits['objective'][selected]<=1e-20))
                        rec['minimum_dynamic_rank_ratio']=float(np.min(aux['rollout']['rank_ratio']))
                        rec['configuration_dimension']=models[name]['p']['linear'].shape[1];rec['phase_dimension']=2*rec['configuration_dimension']
                    if rep==0:
                        arrays=dict(u=restrict(u,n,common.n,bc),v=restrict(v,n,common.n,bc))
                        if name in models:
                            arrays.update({k:aux[k] for k in ('coefficients','velocity_coefficients')})
                            arrays.update({'rollout_'+k:vv for k,vv in aux['rollout'].items()})
                        elif name=='affine32':arrays.update(aux)
                        np.savez_compressed(out/(invocation+'.npz'),**arrays)
                    result['invocations' if role=='timed_comparison' else 'accuracy_controls'].append(rec);save_json(out/'result.json',result)
                for name,setting in methods:
                    print('warmup',bc,n,ci,name,setting,flush=True);_,_,rec,_=call(name,setting)
                    result['warmups'].append(dict(case=ci,**rec))
                for rep in range(cfg['repetitions']):
                    for order,(name,setting) in enumerate(methods if rep%2==0 else list(reversed(methods))):record_query(name,setting,rep,order,'timed_comparison')
                for order,name in enumerate(endpoints):record_query(name,cfg['accuracy_only_dt'],0,order,'accuracy_refinement_only')
                for name,model in models.items():
                    print('diagnostics',bc,n,ci,name,flush=True)
                    diag=d.fitted_diagnostics(model,[affine,full],su,sv,grid,par[5],cfg,out/f'diagnostic_{bc}_{n}_{ci}_{name}.npz')
                    result['diagnostics'].append(dict(boundary=bc,intervals=n,case=ci,model=name,**diag));save_json(out/'result.json',result)
            del bank,models;jax.clear_caches()
    assert len(result['invocations'])==cfg['expected_timed_invocations']
    assert len(result['accuracy_controls'])==cfg['expected_accuracy_only_queries']
    result['complete']=True
    result['output_sha256']={str(p.relative_to(out)):base.sha(p) for p in out.rglob('*.npz')}
    save_json(out/'result.json',result);print('wave_heads32_complete',flush=True)

if __name__=='__main__':main()
