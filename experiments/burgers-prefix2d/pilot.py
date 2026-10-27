"""Approved dev6 early gate. No validation/reference/sealed population is loaded."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/jcp-qualify'))
import p1common as P
from p1common import Q, QS, T2
import prefixcore as C
from test_prefix import actual_evaluators, run as synthetic_checks
import jax
import jax.numpy as jnp

ARMS=('direct','affine2','pure2')


def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if hasattr(x,'tolist'): return clean(x.tolist())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def timed(fn):
    t=time.perf_counter(); v=fn(); jax.block_until_ready(v)
    return time.perf_counter()-t,v


def hash_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    P.check_env({})
    started=time.perf_counter()
    rep={'complete':False,'scope':'dev6 early gate only','arms':ARMS,
         'commit':os.environ.get('SOURCE_COMMIT'),'job_id':os.environ.get('SLURM_JOB_ID'),
         'backend':jax.default_backend(),'gpu':P.gpu_info(),'jax_version':jax.__version__,
         'precision':os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),'x64':bool(jax.config.jax_enable_x64),
         'settings':{'L':256,'r':384,'coarse_r':128,'M':1536,'rule':'gauss128','policy':'frozen',
                     'scheme':'CNR','dt':.005,'steps':50,'startup_BE_steps':2,'gtol':1e-3,'tolf':1e-9,
                     'coarse_max_corrections':2,'repetitions':7,'timing_order_seed':20261027},
         'checkpoint_sha256':QS.sha_file(T2.CKPT),'rotation_sha256':QS.sha_file(Q.ROTATION),
         'setup':{},'records':[],'timings':[],'microtimings':[],'parity':[],'correction_capture':[],
         'certificate_convention':'independent AD and tight/tighter are offline research validation; timed online acceptance unchanged'}
    def save():
        tmp=out/'result.json.tmp'; tmp.write_text(json.dumps(clean(rep),indent=2,allow_nan=False)+'\n'); tmp.replace(out/'result.json')
    rep['synthetic']=synthetic_checks(); save()
    print('SYNTHETIC_PASS',flush=True)
    t=time.perf_counter()
    bank=P.Bank(256,[0,128,256,384,512])
    setting=P.Setting(bank,'R384',384,'gauss128',policy='frozen',forms=('LSPG',),nrec=51,nstep=50)
    setting.check('main',bank)
    data=setting.data[('main','LSPG')]
    assert data['A'].shape==(1536,384) and data['Gq'].shape==(16384,384)
    rep['setup']['bank_and_setting_seconds']=time.perf_counter()-t
    rep['setup']['trust']=setting.trust; rep['table_meta']=setting.meta
    rep['setup']['memory']=P.mem()
    rep['table_sha256']={k:hash_array(data[k]) for k in ('A','lam','Gq','Gs','Psi')}
    # Only the allowed development cohort is generated; no case_list() call.
    cases=Q.cohort('dev6'); assert cases.shape==(6,5)
    rep['cases']=cases; rep['cohort_sha256']=hash_array(cases)
    rep['seeds']={'dev4':7090702,'dev2':911702,'order':20261027}
    U0=[jnp.asarray(Q.e.initial(256,ph)) for ph in cases]
    rep['initial_hashes']=[hash_array(u) for u in U0]
    sch=T2.sched('CNR',.005,1e-3,1e-9)
    qrec={'direct':setting.q['LSPG']}; qtim={'direct':setting.q0['LSPG']}
    for arm in ARMS[1:]:
        qrec[arm]=C.make_split_query(Q,'point','LSPG',384,256,setting.trust,nrec=51,nstep=50,prefix_mode=arm)
        qtim[arm]=C.make_split_query(Q,'point','LSPG',384,256,setting.trust,nrec=0,nstep=50,prefix_mode=arm)
    qnone=C.make_split_query(Q,'point','LSPG',384,256,setting.trust,nrec=51,nstep=50,prefix_mode='none')
    cert=T2.make_certifier(qrec['direct']['nl'],384)
    records={}; timed_identity={}
    def call(q,ci,ss=sch):
        return q['fused'](U0[ci],float(cases[ci,4]),data,setting.cold,ss)
    # Compile and warm EVERY signature before timed production calls.
    rep['compile_warm_seconds']={}
    for arm in ARMS:
        dur,v=timed(lambda:call(qrec[arm],0)); rep['compile_warm_seconds'][arm+'_recorded']=dur
        dur,v=timed(lambda:call(qtim[arm],0)); rep['compile_warm_seconds'][arm+'_fused']=dur
    print('SETUP',round(time.perf_counter()-started,3),P.mem(),flush=True); save()
    for ci in range(6):
        for arm in ARMS:
            dur,v=timed(lambda:call(qrec[arm],ci)); records[(arm,ci)]=v
            st=v['stats']; states=st['states']; nu=float(cases[ci,4])
            cd,(gam,rn)=timed(lambda:cert(states,nu,data,sch))
            _,scale=qrec[arm]['init'](U0[ci],setting.cold)
            tol=float(scale)*1e-9
            g=np.asarray(gam); r=np.asarray(rn)
            gc=P.gcert_ok(g,np.asarray(st['step_gn']))
            fresh=bool(np.all(np.isfinite(g)) and np.all((g<=1e-3)|(r<=tol)))
            valid=bool(P.run_valid(v) and fresh and gc and int(st['nfail'])==0)
            entry={'case':ci,'arm':arm,'valid':valid,'fresh_stationarity':fresh,'gradient_agreement':gc,
                   'gamma_max':float(np.max(g)),'research_record_seconds':dur,'offline_certificate_seconds':cd,
                   'fine_corrections':int(st['it_sum']),'fine_jacobian_pairs':50+int(st['it_sum']),
                   'fine_factorizations':int(st['it_sum']),'predictor_residuals':150,'history_nonlinear_values':48,
                   'fine_rejections':int(st['rej']),'termination_counts':np.asarray(st['exits']),
                   'coarse_corrections':int(jnp.sum(st.get('coarse_it',jnp.array([0])))),
                   'coarse_accepted':int(jnp.sum(st.get('coarse_accept',jnp.array([0])))),
                   'coarse_numerical_failures':int(jnp.sum(st.get('coarse_reason',jnp.array([-1]))==3)),
                   'W_sha256':hash_array(v['W']),'states_sha256':hash_array(states)}
            entry['coarse_jacobian_pairs']=0 if arm=='direct' else 50+entry['coarse_corrections']
            entry['coarse_factorizations']=entry['coarse_corrections']
            entry['fine_guard_residuals']=0 if arm=='direct' else 100
            rep['records'].append(entry)
            np.savez_compressed(out/f'{arm}_case{ci}.npz',W=v['W'],fields=v['fields'],states=states,
                                gamma_fresh=gam,residual_norm=rn,**{k:np.asarray(x) for k,x in st.items() if k!='states'})
            _,vt=timed(lambda:call(qtim[arm],ci))
            # Fused timer returns identical coefficients/decisions; recorded states are outside timing.
            np.testing.assert_allclose(np.asarray(vt['W']),np.asarray(v['W']),rtol=1e-12,atol=1e-13)
            assert np.array_equal(vt['stats']['step_it'],st['step_it'])
            assert np.array_equal(vt['stats']['step_reason'],st['step_reason'])
            np.savez_compressed(out/f'timed_identity_{arm}_case{ci}.npz',W=vt['W'],fields=vt['fields'],
                                **{k:np.asarray(x) for k,x in vt['stats'].items()})
            timed_identity[(arm,ci)]={'W':np.asarray(vt['W']), 'it':np.asarray(vt['stats']['step_it']),
                                     'reason':np.asarray(vt['stats']['step_reason']),'fields':np.asarray(vt['fields'])}
        _,vn=timed(lambda:call(qnone,ci)); vd=records[('direct',ci)]
        err=float(jnp.max(jnp.abs(vn['stats']['states']-vd['stats']['states'])))
        original_stats=vd['stats']
        decisions=all(np.array_equal(vn['stats'][k],original_stats[k]) for k in ('step_reason','step_it','exits','rej'))
        parity=err<=1e-12 and decisions
        rep['parity'].append({'case':ci,'state_max_abs':err,'original_decisions_equal':decisions,'pass':parity})
        assert parity,rep['parity'][-1]
        print('RECORDED',ci,[(x['arm'],x['fine_corrections'],x['coarse_corrections'],x['valid']) for x in rep['records'][-3:]],flush=True)
        save()
    # Per-call burns and a fixed randomized ordering avoid clock-ramp/order selection.
    rng=np.random.default_rng(20261027)
    for block in range(7):
        for ci in rng.permutation(6):
            for ai in rng.permutation(3):
                arm=ARMS[ai]; ci=int(ci)
                QS.burn(.1)
                dt,v=timed(lambda:call(qtim[arm],ci))
                want=timed_identity[(arm,ci)]
                identity=bool(np.array_equal(v['W'],want['W']) and np.array_equal(v['stats']['step_it'],want['it'])
                              and np.array_equal(v['stats']['step_reason'],want['reason']) and np.array_equal(v['fields'],want['fields']))
                rep['timings'].append({'block':block,'case':ci,'arm':arm,'seconds':dt,'identity':identity,
                                       'W_sha256':hash_array(v['W']),'fields_sha256':hash_array(v['fields']),
                                       'array_witness':f'timed_identity_{arm}_case{ci}.npz',
                                       'fine_corrections':int(v['stats']['it_sum'])})
        save(); print('TIMING_BLOCK',block,flush=True)
    # Fixed diagnostic steps: never use these component timers in place of fused timers.
    full,coarse=actual_evaluators(qrec['direct']['nl'],Q.tested_jac('point',256),384,128,setting.trust)
    fjit,cjit=jax.jit(full),jax.jit(coarse)
    def residual(w,args):
        hist,nu,d,co,dt=args
        return T2.lspg_residual(qrec['direct']['nl'],co,dt,nu,d,hist)(w)
    rjit=jax.jit(residual)
    for ci in (0,5):
        st=records[('direct',ci)]['stats']['states']; nu=float(cases[ci,4])
        for k in (0,10,49):
            cn,cm,cm2=st[k],st[max(k-1,0)],st[max(k-2,0)]
            co=jnp.asarray(T2.BE_CO) if k<2 else sch['co']
            Fn=jnp.zeros(1536) if k<2 else qrec['direct']['nl'](cn,data)+nu*data['lam']*(data['A']@cn)
            args=((data['A']@cn,data['A']@cm,Fn),nu,data,co,.005)
            cand=jnp.stack((cn,2*cn-cm,3*cn-3*cm+cm2 if k>=2 else 2*cn-cm))
            wi=cand[jnp.argmin(jnp.stack([jnp.linalg.norm(rjit(w,args)) for w in cand]))]
            bundle=(args,data['Gq']@wi,data['Gs']@wi,data['A']@wi)
            zz=jnp.zeros(128)
            for name,fn in [('residual',lambda:rjit(wi,args)),('fine_residual_jacobian',lambda:fjit(wi,args)),
                            ('coarse_residual_jacobian',lambda:cjit(zz,bundle))]:
                timed(fn)
                times=[]
                for _ in range(5): QS.burn(.05); times.append(timed(fn)[0])
                rep['microtimings'].append({'case':ci,'step':k,'operation':name,'seconds':times})
            rv,Jf=fjit(wi,args); rc,Jc=cjit(zz,bundle)
            err=float(jnp.linalg.norm(Jc-Jf[:,:128])/(1+jnp.linalg.norm(Jf[:,:128])))
            assert err<1e-11
            # Field-metric best prefix projection of the actual direct correction.
            errc=np.asarray(st[k+1]-wi)
            G=np.concatenate([np.concatenate([np.asarray(x) for x in row],axis=1) for row in data['G']],axis=0)
            field=G@errc; Gc=G[:,:128]
            proj=np.linalg.lstsq(Gc,field,rcond=1e-12)[0]
            frac=float(1-np.linalg.norm(field-Gc@proj)**2/(np.linalg.norm(field)**2+1e-300))
            rep['correction_capture'].append({'case':ci,'step':k,'field_energy_fraction':frac,
                'restricted_jacobian_relative_error':err,'correction_field_norm':float(np.linalg.norm(field))})
    # Literal same-bank injection on independent off-grid coordinates with real checkpoint.
    X=np.random.default_rng(20261028).uniform(.01,.99,(37,2))
    B,Bx,By=bank.mdl.values_grads(X,384); z=jnp.asarray(np.random.default_rng(1).normal(size=128))
    pad=jnp.pad(z,(0,256))
    rep['transfer']={name:float(jnp.max(jnp.abs(mat@pad-mat[:,:128]@z))) for name,mat in [('field',B),('dx',Bx),('dy',By)]}
    rep['transfer']['pass']=all(v<1e-10 for v in rep['transfer'].values())
    assert rep['transfer']['pass']
    # Gate depends only on predeclared dev6 complete production queries, all failures retained.
    arr={(arm,ci):np.array([x['seconds'] for x in rep['timings'] if x['arm']==arm and x['case']==ci]) for arm in ARMS for ci in range(6)}
    rep['gate']={}
    for arm in ARMS[1:]:
        ratios=[float(np.median(arr[(arm,ci)]/arr[('direct',ci)])) for ci in range(6)]
        valid=all(x['valid'] for x in rep['records'] if x['arm'] in ('direct',arm))
        ident=all(x['identity'] for x in rep['timings'] if x['arm'] in ('direct',arm))
        rep['gate'][arm]={'per_case_paired_median_ratios':ratios,'median_ratio':float(np.median(ratios)),
                         'all_numerically_valid':valid,'all_timing_identities':ident,
                         'pass':valid and ident and float(np.median(ratios))<=.85}
    rep['continue_validation']=any(x['pass'] for x in rep['gate'].values())
    rep['strong_accuracy_checks']='not run in cost screen: tight/tighter, physical E0 and successor quadrature remain required if continuing'
    rep['elapsed_program_seconds']=time.perf_counter()-started
    rep['memory']=P.mem(); rep['complete']=True; save()
    print('FINAL',json.dumps(clean(rep['gate'])),flush=True)


if __name__=='__main__': main()
