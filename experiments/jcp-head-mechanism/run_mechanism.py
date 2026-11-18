"""Prospective matched head/span NS3D development experiment; no training or heldout data."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
os.environ.setdefault('JAX_ENABLE_X64', '1')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import ns3d_fom as F
import shift_rom as SR
import head_rom as H
import bankio as BI
from shared_geometry import shared_state_geometry

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    return x

def dump(path, obj):
    Path(path).write_text(json.dumps(plain(obj), indent=2, allow_nan=False)+'\n')

def log(s): print(s, flush=True)

@jax.jit
def lm_direction(res, jac, damping):
    scales = jnp.maximum(jnp.linalg.norm(jac,axis=0),1e-30)
    js = jac/scales
    mat = js.T@js + damping*jnp.eye(jac.shape[1])
    return jnp.linalg.solve(mat,-js.T@res)/scales

def stationarity(res,jac):
    den=np.maximum(np.linalg.norm(jac,axis=0),1e-30)*max(np.linalg.norm(res),1e-30)
    return float(np.max(np.abs(jac.T@res)/den))

def solve(evaluate, x0, args, gtol, budget, record=True):
    """Adaptive LM. Report failure on stagnation/cap; no objective-only convergence."""
    x=np.asarray(x0,dtype=np.float64).copy(); damping=1e-4; history=[]
    resj,jacj=evaluate(jnp.asarray(x),*args)
    res,jac=np.asarray(resj),np.asarray(jacj)
    cost=float(.5*np.dot(res,res)); reason='budget'; accepted=0
    for it in range(budget):
        gn=stationarity(res,jac)
        if not np.isfinite(cost+gn) or not np.all(np.isfinite(jac)):
            reason='nonfinite';break
        if gn<=gtol:
            reason='stationarity';break
        step=np.asarray(lm_direction(jnp.asarray(res),jnp.asarray(jac),jnp.asarray(damping)))
        xt=x+step
        rt,jt=evaluate(jnp.asarray(xt),*args)
        rt,jt=np.asarray(rt),np.asarray(jt)
        trial=float(.5*np.dot(rt,rt))
        finite_trial=bool(np.isfinite(trial) and np.all(np.isfinite(rt)) and np.all(np.isfinite(jt)))
        trial_gn=stationarity(rt,jt) if finite_trial else np.inf
        roundoff_band=1024*np.finfo(np.float64).eps*max(cost,trial if np.isfinite(trial) else cost,1e-300)
        roundoff_accept=bool(finite_trial and np.isfinite(trial_gn) and gn<=1e-5 and trial<=cost+roundoff_band
                             and trial_gn<=.5*gn and np.linalg.norm(step)<=1e-6*(1+np.linalg.norm(x)))
        strict_accept=bool(finite_trial and np.isfinite(trial_gn) and trial<cost)
        accept=strict_accept or roundoff_accept
        acceptance_reason='strict_decrease' if strict_accept else ('roundoff_gradient_improvement' if roundoff_accept else 'rejected')
        if record:
            history.append({'iteration':it,'cost':cost,'trial_cost':trial if np.isfinite(trial) else None,
                            'gradient':gn,'damping':damping,'step_norm':float(np.linalg.norm(step)),
                            'accepted':accept,'acceptance_reason':acceptance_reason,
                            'trial_gradient':trial_gn if np.isfinite(trial_gn) else None,'roundoff_band':roundoff_band})
        if accept:
            x,res,jac,cost=xt,rt,jt,trial;accepted+=1;damping=max(damping/3.,1e-14)
        else:
            damping=min(damping*10.,1e18)
        if np.linalg.norm(step)<=1e-15*(1+np.linalg.norm(x)) or damping>=1e18:
            reason='stagnation';break
    gn=stationarity(res,jac)
    if gn<=gtol: reason='stationarity'
    return x, {'reason':reason,'gradient':gn,'cost':cost,'residual_norm':float(np.linalg.norm(res)),
               'iterations':len(history),'accepted':accepted,'history':history}

@jax.jit
def fit_eval(z,target,p):
    return H.head_apply(p,z)-target,H.head_jacobian(p,z)

def select_fit_candidate(infos):
    """A stationary candidate may replace a nonstationary winner only inside an f64 cost tie."""
    costs=np.asarray([q['cost'] for q in infos]);raw=int(np.argmin(costs))
    band=1024*np.finfo(np.float64).eps*max(float(costs[raw]),1e-300)
    selected=raw
    if infos[raw]['reason']!='stationarity':
        eligible=[i for i,q in enumerate(infos) if q['reason']=='stationarity' and costs[i]<=costs[raw]+band]
        if eligible:selected=min(eligible,key=lambda i:costs[i])
    return selected,raw,band

def fit_head(target,p,Z,Hcand,starts=4,extra=None,gtol=1e-7,budget=250):
    dist=np.sum((Hcand-np.asarray(target)[None,:])**2,axis=1)
    seeds=[np.asarray(Z[i]) for i in np.argsort(dist)[:starts]]
    if extra is not None: seeds.extend(np.atleast_2d(extra))
    results=[solve(fit_eval,z,(jnp.asarray(target),p),gtol,budget) for z in seeds]
    idx,raw,band=select_fit_candidate([q[1] for q in results])
    coefficient_gap=float(np.linalg.norm(np.asarray(H.head_apply(p,jnp.asarray(results[idx][0])))-np.asarray(H.head_apply(p,jnp.asarray(results[raw][0]))))) if idx!=raw else 0.0
    return results[idx][0], {'best_index':idx,'raw_minimum_cost_index':raw,'selection_tie_band':band,
                           'selected_stationary_tie':idx!=raw,'all_final_codes':[q[0] for q in results],
                           'coefficient_gap_selected_vs_raw':coefficient_gap,
                           'raw_minimum_found_distance':float(np.sqrt(2*results[raw][1]['cost'])),
                           'starts':[q[1] for q in results],
                           'best_found_fixed_frame_distance':float(np.sqrt(2*results[idx][1]['cost']))}

def make_eval(dt,n,r,k,kind):
    _,prepare,resjac=SR.make_frozen_step(dt,n,r,1,1e-6)
    @jax.jit
    def ev(w,prev,ops,p):
        if kind=='head':
            a=H.head_apply(p,w[:k]); rr,jj=resjac(jnp.concatenate((a,w[k:])),prev,ops)
            return rr,jnp.concatenate((jj[:,:r]@H.head_jacobian(p,w[:k]),jj[:,r:]),axis=1)
        return resjac(w,prev,ops)
    return prepare,ev

def coefficients(w,p,k,kind):
    return np.asarray(H.head_apply(p,jnp.asarray(w[:k]))) if kind=='head' else np.asarray(w[:-3])

@jax.jit
def reconstruct(basis,a,center):
    n=round((basis.shape[0]/3)**(1/3))
    return SR.shift_field((basis@a).reshape(3,n,n,n),center*n)

@jax.jit
def field_metrics(pred,truth,geom):
    n=pred.shape[-1]; pc=SR.grid_centroid(pred);tc=SR.grid_centroid(truth)
    phase=(pc-tc+.5)%1-.5
    aligned=SR.shift_field(pred,-phase*n)
    uh=F.fft(pred);k,k2,_=geom
    div=jnp.linalg.norm(jnp.sum(1j*k*uh,axis=0))/jnp.maximum(jnp.sqrt(jnp.sum(k2*jnp.sum(jnp.abs(uh)**2,axis=0))),1e-300)
    return (jnp.linalg.norm(pred-truth),jnp.linalg.norm(aligned-truth),pc,tc,phase,
            .5*jnp.mean(jnp.sum(pred*pred,axis=0)),.5*jnp.mean(jnp.sum(truth*truth,axis=0)),div)

@jax.jit
def tangent_gram(basis,a,dc,geom):
    n=round((basis.shape[0]/3)**(1/3))
    field=(basis@a).reshape(3,n,n,n)
    deriv=-jnp.stack([F.ifft(1j*geom[0][d]*F.fft(field)).ravel() for d in range(3)],axis=1)
    mixed=dc.T@(basis.T@deriv)
    top=jnp.concatenate((dc.T@dc,mixed),axis=1)
    bottom=jnp.concatenate((mixed.T,deriv.T@deriv),axis=1)
    return jnp.concatenate((top,bottom),axis=0)/(n**3)

def action_check(ev,w,args,seed):
    rng=np.random.default_rng(seed);d=rng.normal(size=len(w));d/=np.linalg.norm(d)
    r,j=ev(jnp.asarray(w),*args);ad=np.asarray(j)@d
    epsilons=[1e-4,1e-5,1e-6]
    errors=[]
    for eps in epsilons:
        plus=ev(jnp.asarray(w+eps*d),*args)[0];minus=ev(jnp.asarray(w-eps*d),*args)[0]
        fd=np.asarray((plus-minus)/(2*eps))
        errors.append(float(np.linalg.norm(fd-ad)/max(np.linalg.norm(ad),1e-30)))
    return {'epsilons':epsilons,'relative_errors':errors,'minimum':min(errors)}

def physical_spectrum(jac,gram):
    vals,vecs=np.linalg.eigh((gram+gram.T)/2)
    threshold=max(float(vals[-1])*1e-12,1e-30);keep=vals>threshold
    normalized=jac@(vecs[:,keep]/np.sqrt(vals[keep]))
    return {'gram_eigenvalues':vals,'gram_threshold':threshold,'rank':int(keep.sum()),
            'excluded':int((~keep).sum()),'raw_jacobian_singular_values':np.linalg.svd(jac,compute_uv=False),
            'physical_jacobian_singular_values':np.linalg.svd(normalized,compute_uv=False)}

def rollout(kind,a0,z0,c0,p,ops,opswide,dt,n,k,steps,gtol,budget,basis,truth,geom,norm0,Z,Hcand,diagnose,independent_start):
    r=len(a0); prepare,ev=make_eval(dt,n,r,k,kind)
    w=np.r_[z0 if kind=='head' else a0,np.zeros(3)];old=w.copy();center=c0.copy()
    states=[w.copy()];centers=[center.copy()];coeffs=[coefficients(w,p,k,kind)];records=[]
    for t in range(steps):
        prev=coeffs[-1];guess=w+(w-old)
        new,rec=solve(ev,guess,(jnp.asarray(prev),ops,p),gtol,budget)
        old,w=w,new;center=(center+w[-3:])%1.;a=coefficients(w,p,k,kind)
        rw,jw=ev(jnp.asarray(w),jnp.asarray(prev),opswide,p)
        rec['expanded_residual_norm']=float(jnp.linalg.norm(rw));rec['expanded_only_residual_norm']=float(jnp.linalg.norm(rw[ops[0].shape[0]:]));rec['fixed_independent_shell_norm']=float(jnp.linalg.norm(rw[independent_start:]));rec['expanded_residual_vector']=np.asarray(rw);rec['step_index']=t
        if diagnose:
            rec['jacobian_action']=action_check(ev,w,(jnp.asarray(prev),ops,p),700+t)
            dc=np.asarray(H.head_jacobian(p,jnp.asarray(w[:k]))) if kind=='head' else np.eye(r)
            gram=np.asarray(tangent_gram(basis,jnp.asarray(a),jnp.asarray(dc),geom))
            jj=np.asarray(ev(jnp.asarray(w),jnp.asarray(prev),ops,p)[1])
            rec['physical_tangent']=physical_spectrum(jj,gram)
        records.append(rec);states.append(w.copy());centers.append(center.copy());coeffs.append(a)
    metrics=[];fields=[];fits=[]
    fit_extra=z0
    for t,(a,c) in enumerate(zip(coeffs,centers)):
        pred=reconstruct(basis,jnp.asarray(a),jnp.asarray(c));met=field_metrics(pred,jnp.asarray(truth[t]),geom)
        metrics.append({'relative_error':float(met[0])/norm0,'shape_aligned_relative_error':float(met[1])/norm0,
                        'physical_centroid':np.asarray(met[2]),'truth_centroid':np.asarray(met[3]),
                        'phase_displacement':np.asarray(met[4]),'energy':float(met[5]),'truth_energy':float(met[6]),
                        'relative_divergence':float(met[7])})
        fields.append(np.asarray(pred))
        if diagnose and kind=='span':
            zfit,info=fit_head(a,p,Z,Hcand,extra=fit_extra,gtol=1e-7,budget=budget)
            fit_extra=zfit; info['relative_distance']=info['best_found_fixed_frame_distance']/norm0
            info['code']=zfit;fits.append(info)
    return {'states':np.asarray(states),'centers':np.asarray(centers),'coefficients':np.asarray(coeffs),
            'records':records,'metrics':metrics,'span_head_fits':fits,'fields':np.asarray(fields)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out;out.mkdir(parents=True,exist_ok=True)
    if jax.default_backend()!='gpu': raise RuntimeError('GPU required')
    if cfg['eval_seed'] in cfg['closed_seeds']:raise RuntimeError('closed seed forbidden')
    n=cfg['n'];k=cfg['k'];horizon=cfg['horizon'];t_start=time.time()
    manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    for rel,meta in manifest['files'].items():
        if sha(HERE/rel)!=meta['sha256']:raise RuntimeError('source mismatch: '+rel)
    if cfg.get('reproducibility_reference'):
        for rel,expected in json.loads((HERE/'REPRO_MANIFEST.json').read_text()).items():
            if sha(HERE/rel)!=expected:raise RuntimeError('reproducibility fixture mismatch: '+rel)
    report={'schema':'jcp-head-mechanism-result-v1','config':cfg,'source_manifest':manifest,
            'source_commit':os.environ.get('SOURCE_COMMIT'),'job_id':os.environ.get('SLURM_JOB_ID'),
            'backend':jax.default_backend(),'precision':str(jnp.ones(1).dtype),'matmul_precision':str(jax.config.jax_default_matmul_precision),
            'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],text=True).strip(),
            'driver_sha256':sha(__file__),'cases':[],'qualification':{}}
    def save():report['elapsed_seconds']=time.time()-t_start;dump(out/'summary.json',report)
    save()
    log('Sampling frozen bank')
    pbank=BI.load_params(HERE/'assets/bank_selected.npz');ck=np.load(HERE/'assets/bank_selected.npz')
    G,binfo=BI.mesh_bank(pbank,ck['T'],n,chunk=32768,abort=.05);lowdin=binfo.pop('lowdin_factor')
    report['bank']=binfo;report['bank']['rank']=G.shape[1];basis=jnp.asarray(G)
    h=np.load(HERE/'assets/head_k8_n64.npz');p={key:jnp.asarray(h['p_'+key]) for key in ('W1','b1','W2','b2','W3','b3','Ws')}
    Z=np.asarray(h['Z']);Hcand=np.asarray(H.head_apply(p,jnp.asarray(Z)))
    log('Building exact weak operator with independent dense checks')
    opsall,checks=SR.build_operators_fast(G,n,cfg['expanded_test_rows'],check_modes=8)
    report['operator_checks']=checks;report['actual_expanded_scalar_rows']=opsall['A'].shape[0]
    np.savez_compressed(out/'operator_arrays.npz',**{key:opsall[key] for key in ('A','T','lam','Dd')},lowdin=lowdin)
    save()
    geom=F.geometry(n);nout=round(cfg['truth_output_dt']/cfg['dt_truth'])
    truth_run=F.make_solver(cfg['dt_truth'],round(horizon/cfg['dt_truth']),nout,scheme=cfg['truth_scheme'])
    fine_run=F.make_solver(cfg['dt_truth']/2,round(2*horizon/cfg['dt_truth']),nout*2,scheme=cfg['truth_scheme'])
    original_run=F.make_solver(cfg['dt_truth'],round(horizon/cfg['dt_truth']),round(horizon/cfg['dt_truth']/5),scheme=cfg['truth_scheme'])
    params=F.parameters(cfg['eval_seed'],cfg['eval_cases']);report['parameter_sha256']=hashlib.sha256(params.tobytes()).hexdigest()
    for ci,par in enumerate(params):
        if ci not in cfg.get('initial_case_indices',list(range(len(params)))):continue
        casepath=out/f'case_{ci:03d}';casepath.mkdir(exist_ok=True)
        u0=F.initial(n,par);truth=np.asarray(truth_run(jnp.asarray(u0),float(par[-1]),geom));norm0=np.linalg.norm(u0)
        fine=np.asarray(fine_run(jnp.asarray(u0),float(par[-1]),geom))
        truth_gap=np.linalg.norm((truth-fine).reshape(len(truth),-1),axis=1)/norm0
        oldtruth=np.asarray(original_run(jnp.asarray(u0),float(par[-1]),geom))
        original_gap=np.linalg.norm((truth[::4]-oldtruth).reshape(len(oldtruth),-1),axis=1)/norm0
        del fine,oldtruth
        center=np.asarray(SR.grid_centroid(jnp.asarray(u0)));centered=SR.shift_field(jnp.asarray(u0),-jnp.asarray(center)*n)
        projected=np.asarray(basis.T@centered.ravel())
        zlo,icfitlo=fit_head(projected,p,Z,Hcand,starts=cfg['head_fit_starts'],gtol=max(cfg['solver']['gtols']),budget=cfg['solver']['budget'])
        z0,icfit=fit_head(projected,p,Z,Hcand,starts=cfg['head_fit_starts'],gtol=min(cfg['solver']['gtols']),budget=cfg['solver']['budget'])
        ahead=np.asarray(H.head_apply(p,jnp.asarray(z0)))
        icgap=float(np.linalg.norm(ahead-np.asarray(H.head_apply(p,jnp.asarray(zlo))))/norm0)
        icfit['tight_tighter_gap']=icgap
        icfit['raw_to_selected_physical_relative_gap']=icfit['coefficient_gap_selected_vs_raw']/norm0
        icfit['looser_fit']=icfitlo
        icfit['qualified']=bool(icgap<=1e-6 and icfit['starts'][icfit['best_index']]['reason']=='stationarity' and icfitlo['starts'][icfitlo['best_index']]['reason']=='stationarity')
        field1=np.asarray(reconstruct(basis,jnp.asarray(ahead),jnp.asarray(center)))
        # Separate coefficient and head decoding paths establish common-state equality.
        field2=np.asarray(SR.shift_field((basis@H.head_apply(p,jnp.asarray(z0))).reshape(3,n,n,n),jnp.asarray(center)*n))
        parity=np.linalg.norm(field1-field2)/norm0
        case={'index':ci,'parameters':par,'norm_u0':norm0,'shared_ic_parity':parity,'head_ic_fit':icfit,
              'reference_check':{'dt':cfg['dt_truth'],'dt_refined':cfg['dt_truth']/2,'scope':'same mesh and discrete spatial operator, not continuum error','normalization':'norm original physical u0','per_time_difference':truth_gap,'max_difference':float(truth_gap.max()),'original_output_grid_parity':float(original_gap.max())},
              'head_ic_relative_error':np.linalg.norm(field1-u0)/norm0,'span_ic_relative_error':np.linalg.norm(np.asarray(reconstruct(basis,jnp.asarray(projected),jnp.asarray(center)))-u0)/norm0,'runs':[]}
        report['cases'].append(case);save()
        if parity>cfg['qualification']['shared_ic_relative']:raise RuntimeError('shared IC parity failed')
        np.savez_compressed(casepath/'initial.npz',parameters=par,head_coefficients=ahead,span_native_coefficients=projected,head_code=z0,center=center)
        # Only compact reproducible state is retained; truth is regenerated from seed and checked by SHA256.
        case['truth_sha256']=hashlib.sha256(truth.tobytes()).hexdigest()
        for setting in cfg['test_settings']:
            rows,dt=setting['rows'],setting['dt'];steps=round(horizon/dt);stride=round(dt/cfg['truth_output_dt']);ref=truth[::stride]
            _,prepare,_=SR.make_frozen_step(dt,n,len(ahead),1,1e-6)
            oa=tuple(jnp.asarray(opsall[key]) for key in ('A','T','lam','Dd'))
            wide=prepare(float(par[-1]),*oa)
            selected=prepare(float(par[-1]),oa[0][:rows],oa[1][:rows],oa[2][:rows],oa[3][:,:rows])
            cache={}
            for gtol in cfg['solver']['gtols']:
                for arm in cfg['arms']:
                    kind='head' if arm.startswith('head') else 'span';aa=projected if arm=='span_native' else ahead
                    tag=f'{arm}_m{rows}_dt{dt:g}_g{gtol:g}'
                    log(f'case={ci} {tag}')
                    diag=gtol==min(cfg['solver']['gtols'])
                    started=time.time()
                    ans=rollout(kind,aa,z0,center,p,selected,wide,dt,n,k,steps,gtol,cfg['solver']['budget'],basis,ref,geom,norm0,Z,Hcand,diag,max(x['rows'] for x in cfg['test_settings']))
                    elapsed=time.time()-started
                    errors=np.asarray([m['relative_error'] for m in ans['metrics']]);recs=ans['records']
                    item={'tag':tag,'arm':arm,'rows':rows,'actual_scalar_rows':int(selected[0].shape[0]),'unknowns':k+3 if kind=='head' else len(ahead)+3,
                          'dt':dt,'gtol':gtol,'elapsed_diagnostic_seconds':elapsed,'timing_claim':False,
                          'evolved_worst':float(errors[1:].max()),'final_error':float(errors[-1]),'metrics':ans['metrics'],
                          'all_steps_stationary':all(r['reason']=='stationarity' for r in recs),'max_gradient':max(r['gradient'] for r in recs),
                          'termination_counts':{key:sum(r['reason']==key for r in recs) for key in sorted(set(r['reason'] for r in recs))},
                          'records_file':tag+'.json','state_file':tag+'.npz'}
                    if diag:
                        item['jacobian_action_worst_best_epsilon']=max(r['jacobian_action']['minimum'] for r in recs)
                        item['head_fit_distance']=[f['relative_distance'] for f in ans['span_head_fits']]
                    if ci in cfg.get('reproducibility_case_indices',[]):
                        prior=np.load(HERE/'reproducibility'/cfg['reproducibility_reference']/f'case_{ci:03d}'/(tag+'.npz'))
                        field_gaps=[]
                        for olda,oldc,newfield in zip(prior['coefficients'],prior['centers'],ans['fields']):
                            oldfield=np.asarray(reconstruct(basis,jnp.asarray(olda),jnp.asarray(oldc)))
                            field_gaps.append(float(np.linalg.norm(oldfield-newfield)/norm0))
                        item['pilot_reproducibility']={'coefficient_gap':float(np.max(np.linalg.norm(prior['coefficients']-ans['coefficients'],axis=1))/norm0),'wrapped_frame_gap':float(np.max(np.abs((prior['centers']-ans['centers']+.5)%1-.5))),'physical_field_gaps':field_gaps,'passed':max(field_gaps)<=cfg['qualification']['pilot_field_reproducibility']}
                    cache[(arm,gtol)]=ans
                    np.savez_compressed(casepath/(tag+'.npz'),states=ans['states'],centers=ans['centers'],coefficients=ans['coefficients'],time=np.arange(steps+1)*dt)
                    dump(casepath/(tag+'.json'),{'records':recs,'span_head_fits':ans['span_head_fits']})
                    case['runs'].append(item);save()
            for arm in cfg['arms']:
                loose=cache[(arm,max(cfg['solver']['gtols']))];tight=cache[(arm,min(cfg['solver']['gtols']))]
                gap=np.linalg.norm((loose['fields']-tight['fields']).reshape(steps+1,-1),axis=1)/norm0
                err=np.asarray([m['relative_error'] for m in tight['metrics']]);bound=min(cfg['qualification']['tight_tighter_field_difference'],cfg['qualification']['tight_tighter_fraction_of_physical_error']*max(float(err[1:].max()),1e-30))
                case.setdefault('tight_tighter',[]).append({'arm':arm,'rows':rows,'dt':dt,'per_time_gap':gap,'max_gap':float(gap.max()),'bound':bound,'passed':bool(gap.max()<=bound)})
            # One-step objective-containment control: both solves share the reached head previous state.
            base=cache[('head_common',min(cfg['solver']['gtols']))];_,sev=make_eval(dt,n,len(ahead),k,'span')
            reset=[]
            for t in range(steps):
                start=np.r_[base['coefficients'][t+1],base['states'][t+1,-3:]]
                previous=base['coefficients'][t]
                r0,jspan=sev(jnp.asarray(start),jnp.asarray(previous),selected,p)
                opt,info=solve(sev,start,(jnp.asarray(previous),selected,p),min(cfg['solver']['gtols']),cfg['solver']['budget'])
                pred=reconstruct(basis,jnp.asarray(opt[:-3]),jnp.asarray((base['centers'][t]+opt[-3:])%1))
                reset.append({'step':t,'head_seed_residual_norm':float(jnp.linalg.norm(r0)),'span_result':info,
                              'head_physical_error':base['metrics'][t+1]['relative_error'],
                              'span_physical_error':float(jnp.linalg.norm(pred-jnp.asarray(ref[t+1])))/norm0,
                              'span_state':opt})
            if cfg.get('shared_state_geometry',False):
                _,hev=make_eval(dt,n,len(ahead),k,'head')
                for t,probe in enumerate(reset):
                    start=np.r_[base['coefficients'][t+1],base['states'][t+1,-3:]]
                    previous=base['coefficients'][t]
                    _,js=sev(jnp.asarray(start),jnp.asarray(previous),selected,p)
                    _,jh=hev(jnp.asarray(base['states'][t+1]),jnp.asarray(previous),selected,p)
                    dc=np.asarray(H.head_jacobian(p,jnp.asarray(base['states'][t+1,:k])))
                    gram=np.asarray(tangent_gram(basis,jnp.asarray(start[:-3]),jnp.eye(len(ahead)),geom))
                    geometry=shared_state_geometry(np.asarray(js),dc,gram,np.asarray(probe['span_state'])-start)
                    chained=np.c_[np.asarray(js)[:,:len(ahead)]@dc,np.asarray(js)[:,len(ahead):]]
                    geometry['analytic_chain_relative']=float(np.linalg.norm(chained-np.asarray(jh))/max(np.linalg.norm(jh),1e-300))
                    geometry['containment_probe_stationary']=probe['span_result']['reason']=='stationarity'
                    probe['shared_state_geometry']=geometry
            dump(casepath/f'reset_m{rows}_dt{dt:g}.json',reset);save()
            del cache
        np.savez_compressed(casepath/'truth_slices.npz',times=np.arange(len(truth))*cfg['truth_output_dt'],velocity_zmid=truth[:,:,:,n//2,:])
        reference_bound=min(cfg['qualification']['reference_absolute'],cfg['qualification']['reference_fraction_of_rom_error']*min(a['evolved_worst'] for a in case['runs']))
        case['reference_check']['bound']=reference_bound
        case['reference_check']['passed']=bool(truth_gap.max()<=reference_bound and original_gap.max()<=1e-12)
        save()
    allruns=[a for c in report['cases'] for a in c['runs']]
    report['qualification']={'all_stationary':all(a['all_steps_stationary'] for a in allruns),
        'all_tight_tighter':all(a['passed'] for c in report['cases'] for a in c['tight_tighter']),
        'all_action_checks':all(a.get('jacobian_action_worst_best_epsilon',0)<=cfg['qualification']['jacobian_action_relative'] for a in allruns),
        'common_ic':all(c['shared_ic_parity']<=cfg['qualification']['shared_ic_relative'] for c in report['cases']),
        'ic_fit_passed':all(c['head_ic_fit']['qualified'] for c in report['cases']),'pilot_reproducibility':all(a.get('pilot_reproducibility',{'passed':True})['passed'] for a in allruns),'operator_passed':True,'reference_passed':all(c['reference_check']['passed'] for c in report['cases']),'cases_completed':len(report['cases'])}
    report['qualification']['passed']=all(v for key,v in report['qualification'].items() if key!='cases_completed')
    save();log('DONE '+json.dumps(report['qualification']))

if __name__=='__main__':main()
