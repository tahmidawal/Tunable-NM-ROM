"""Prospective dev6 replay cost screen; no reference or sealed cohort access."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/jcp-heads'))
import p1common as P
from p1common import Q, QS, T2
import hcommon as HC
import hcore as H
import harm
import replaycore as C
from replay import make_query, INFO
import jax
import jax.numpy as jnp

ARMS = ('direct', 'replay')
SEED = 20261028
BANK_SHA = '18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589'
ROTATION_SHA = '51149166b53dad386c93fa0682079aec96b61276801d9a6d4f622453d6426772'


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    if hasattr(x, 'tolist'): return clean(x.tolist())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def timed(fn):
    t = time.perf_counter(); v = fn(); jax.block_until_ready(v)
    return time.perf_counter() - t, v


def dump_npz(path, v, **extra):
    """Split previews so no uncompressed array bundle exceeds 50 MiB."""
    arrays = {k: np.asarray(x) for k, x in v.items() if k not in ('stats', 'coarse_fields')}
    arrays.update({k: np.asarray(x) for k, x in v['stats'].items()})
    arrays.update(extra)
    if 'coarse_fields' in v:
        companion = path.with_name(path.stem + '_coarse_fields.npz')
        np.savez_compressed(companion, fields=v['coarse_fields'])
        arrays['coarse_fields_file'] = companion.name
    np.savez_compressed(path, **arrays)


def coarse_view(v):
    return dict(W=v['coarse_W'], fields=v['coarse_fields'], w0=v['coarse_w0'],
                stats={k[7:]: x for k, x in v['stats'].items() if k.startswith('coarse_')})


def same_witness(v, want):
    return all(np.array_equal(np.asarray(v[k]), x, equal_nan=True) for k, x in want['arrays'].items()) and all(
        np.array_equal(np.asarray(v['stats'][k]), x, equal_nan=True) for k, x in want['stats'].items())


def integer_parity(a, b):
    for k in ('step_it','step_reason','it_sum','it_max','rej','exits','nfail','ic_it','ic_reason','ic_idx'):
        assert np.array_equal(a[k], b[k]), k


def make_guard_check(hf, hc, res):
    """Rebuild each original predictor and its own fine history independently."""
    def check(states, coarse_states, nu, data, sch):
        def one(k):
            zn, zp, zp2=states[k],states[jnp.maximum(k-1,0)],states[jnp.maximum(k-2,0)]
            hist=(data['A']@hf(zn),data['A']@hf(zp),jnp.zeros(256))
            args=(hist,nu,data,sch['co'],sch['dt'])
            ze=zn+(zn-zp); zqd=jnp.where(k>=2,3.*zn-3.*zp+zp2,ze)
            cand=jnp.stack((zn,ze,zqd)); rs=jax.vmap(lambda z:jnp.linalg.norm(res(z,args)))(cand)
            warm=cand[jnp.argmin(jnp.where(jnp.isfinite(rs),rs,jnp.inf))]
            promoted=jnp.pad(coarse_states[k+1],(0,48))
            r0,r1=res(warm,args),res(promoted,args)
            gap=jnp.linalg.norm(hf(promoted)-hc(coarse_states[k+1]))
            return warm,jnp.array([jnp.vdot(r0,r0),jnp.vdot(r1,r1)]),gap
        return jax.lax.map(one,jnp.arange(50))
    return jax.jit(check)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--model-dir', required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True, exist_ok=False); model_dir=Path(a.model_dir)
    P.check_env({}); started=time.perf_counter()
    model=json.loads((model_dir/'MODEL.json').read_text())
    assert model['schema']==1 and model['complete'] and model['source_ckpt_sha256']==BANK_SHA
    assert model['training']['steps']==200000 and model['training']['prefixes']==[16,32,64]
    assert model['training']['seed']==SEED
    rep=dict(complete=False, scope='dev6 standalone D16 trajectory and guarded D64 replay',
        commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
        backend=jax.default_backend(), gpu=P.gpu_info(), jax_version=jax.__version__,
        precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), x64=bool(jax.config.jax_enable_x64),
        model=model, model_manifest_sha256=HC.sha_file(model_dir/'MODEL.json'),
        settings=dict(L=1024, K_fine=64, K_coarse=16, R=512, M=256, rule='gauss128',
                      scheme='BE', dt=.005, steps=50, budget=600, gtol=1e-3, tolf=1e-9,
                      repetitions=7, order_seed=SEED, coarse_preview_fields=6, fine_fields=6),
        checks={}, parity=[], records=[], timings=[], setup={}, diagnostics=[])
    def save():
        temp=out/'result.json.tmp'; temp.write_text(json.dumps(clean(rep),indent=2,allow_nan=False)+'\n'); temp.replace(out/'result.json')
    save(); setup_start=time.perf_counter()
    assert HC.sha_file(T2.CKPT)==BANK_SHA and HC.sha_file(Q.ROTATION)==ROTATION_SHA
    raw={}
    for k in (16,32,64):
        spec=model['heads'][str(k)]; path=model_dir/spec['file']
        assert spec['file']==f'h{k}.npz' and spec['k']==k and HC.sha_file(path)==spec['sha256']
        with np.load(path) as z: raw[k]={n:np.asarray(z[n]) for n in z.files}
        assert int(raw[k]['k'])==k and int(raw[k]['steps'])==200000
    for k in (16,32):
        assert np.array_equal(raw[k]['h0_w'],raw[64]['h0_w'][:k])
        assert np.array_equal(raw[k]['h_lin'],raw[64]['h_lin'][:k])
        assert np.array_equal(raw[k]['Zsub'],raw[64]['Zsub'][:,:k])
        for key in ('h0_b','h1_w','h1_b','h2_w','h2_b','n_layers'):
            assert np.array_equal(raw[k][key],raw[64][key]), key
    bank=P.Bank(1024,[0,128,256,384,512])
    heads={k:HC.Head(model_dir/f'h{k}.npz',bank.mdl) for k in (16,32,64)}
    c,f=heads[16],heads[64]
    assert all(h.K==k and len(h.Zsub)==8192 and np.isfinite(h.trust) and h.trust>0 for k,h in heads.items())
    assert all(h.trust==model['heads'][str(k)]['trust'] for k,h in heads.items())
    fine=HC.HeadSetting(bank,'shared64',f,'gauss128',M=256,nrec=51,nstep=50)
    fine.check('main',bank); d=fine.data['main']; cold_c=c.cold(bank.mdl)
    assert d['A'].shape==(256,512) and d['Gq'].shape==(16384,512)
    jax.block_until_ready((d,cold_c,fine.cold))
    rep['setup']=dict(seconds=time.perf_counter()-setup_start, memory=P.mem(),
                      trust={str(k):h.trust for k,h in heads.items()})
    rep['table_meta']=fine.meta; rep['table_hashes']={k:sha(d[k]) for k in ('A','lam','Gq','Gs','Psi')}
    rep['bank_sha256']=BANK_SHA; rep['rotation_sha256']=ROTATION_SHA
    hf,hc=H.head_fn(f.pf),H.head_fn(c.pf); nl=Q.tested_value('point',1024)
    qrec={'direct':fine.query(record=True), 'replay':make_query(c,f,record=True)}
    qtim={'direct':fine.query(record=False), 'replay':make_query(c,f,record=False)}
    qnone=C.make_head_query(f.pf,64,1024,f.trust,nrec=51,nstep=50)
    qfail0=make_query(c,f,record=True,force_failure_step=0)
    qfailmid=make_query(c,f,record=True,force_failure_step=19)
    # Only exposed development descriptors; no case_list/reference loader.
    cases=Q.cohort('dev6'); assert cases.shape==(6,5)
    U0=[jnp.asarray(Q.e.initial(1024,ph)) for ph in cases]
    rep['cases']=cases; rep['seeds']=dict(dev4=7090702,dev2=911702,order=SEED)
    rep['initial_hashes']=[sha(u) for u in U0]
    sch=T2.sched('BE',.005,1e-3,1e-9)
    def call(q,ci,replay=False):
        if replay: return q['fused'](U0[ci],float(cases[ci,4]),d,fine.cold,cold_c,sch)
        return q['fused'](U0[ci],float(cases[ci,4]),d,fine.cold,sch)
    # Upward identity, spatial derivative identity, and nonzero expansion directions.
    X,_=Q.offmesh_rule('gauss128'); X=X[::129][:32]; G,Gx,Gy=bank.mdl.values_grads(X,512)
    for k in (16,32):
        hk=H.head_fn(heads[k].pf); z=jnp.asarray(heads[k].Zsub[:8]); zp=jnp.pad(z,((0,0),(0,64-k)))
        lo=jax.vmap(hk)(z); hi=jax.vmap(hf)(zp)
        np.testing.assert_allclose(lo,hi,rtol=1e-12,atol=1e-13)
        for B in (G,Gx,Gy): np.testing.assert_allclose(lo@B.T,hi@B.T,rtol=1e-12,atol=1e-13)
        rep['checks'][f'upward_{k}_64']=dict(coeff_max_abs=float(jnp.max(jnp.abs(lo-hi))), field_and_derivative_identity=True)
    Jhead=jax.vmap(jax.jacfwd(hf))(jnp.asarray(f.Zsub[:8]))
    colnorm=np.asarray(jnp.linalg.norm(Jhead,axis=1))
    assert np.isfinite(colnorm).all() and np.all(colnorm[:,16:]>0)
    rep['checks']['expansion']=dict(min_new_column_norm=float(colnorm[:,16:].min()),max_new_column_norm=float(colnorm[:,16:].max()))
    for name,q,h in [('fine',qrec['direct'],f),('coarse',qrec['replay']['small'],c)]:
        z=jnp.asarray(h.Zsub[0]); cc=hf(jnp.asarray(f.Zsub[0])); hist=(d['A']@cc,d['A']@cc,jnp.zeros(256))
        args=(hist,float(cases[0,4]),d,sch['co'],sch['dt'])
        _,J=q['evalJ'](z,args); Ja=jax.jacfwd(q['res'])(z,args)
        gap=float(jnp.linalg.norm(J-Ja)/(1+jnp.linalg.norm(Ja))); assert gap<1e-11
        rep['checks'][name+'_jacobian']=dict(columns=J.shape[1],relative_gap=gap)
    rep['compile_warm_seconds']={}
    for arm in ARMS:
        for name,qs in [('recorded',qrec),('timed',qtim)]:
            sec,_=timed(lambda:call(qs[arm],0,arm=='replay')); rep['compile_warm_seconds'][arm+'_'+name]=sec
    cert_f=H.make_head_certifier(hf,nl); cert_c=H.make_head_certifier(hc,nl)
    guard_check=make_guard_check(hf,hc,qrec['direct']['res'])
    records={}; witnesses={}; did_forced=False
    for ci in range(6):
        _,scale_f,_=qrec['direct']['init'](U0[ci],fine.cold)
        _,scale_c,_=qrec['replay']['small']['init'](U0[ci],cold_c)
        tf=float(scale_f)*1e-9; tc=float(scale_c)*1e-9
        for arm in ARMS:
            sec,v=timed(lambda:call(qrec[arm],ci,arm=='replay')); records[arm,ci]=v; st=v['stats']
            _,(g,rn)=timed(lambda:cert_f(st['states'],float(cases[ci,4]),d,sch))
            fresh=bool(np.isfinite(g).all() and np.all((np.asarray(g)<=1e-3)|(np.asarray(rn)<=tf)))
            agree=P.gcert_ok(g,st['step_gn'])
            valid=bool(harm.run_valid(v) and fresh and agree and int(st['nfail'])==0)
            entry=dict(arm=arm,case=ci,valid=valid,fine_stationarity=fresh,gradient_agreement=agree,
                       fine_gamma_max=float(np.max(g)),recorded_seconds=sec,fine_corrections=int(st['it_sum']),
                       fine_physics_J=50+int(st['it_sum']),fine_predictor_residual_calls=150,
                       guard_residual_calls=100 if arm=='replay' else 0,fine_ic_iterations=int(st['ic_it']))
            extras=dict(gamma_fresh=g,residual_norm_fresh=rn,residual_tolerance=tf)
            if arm=='replay':
                cv=coarse_view(v); cs=cv['stats']; count=int(cs['valid_steps']); mask=np.arange(50)<count
                _,(cg,crn)=timed(lambda:cert_c(cs['states'],float(cases[ci,4]),d,sch))
                cstop=bool(count==50 and cs['initial_valid'] and np.isfinite(cg).all() and
                           np.all((np.asarray(cg)<=1e-3)|(np.asarray(crn)<=tc)))
                cagree=P.gcert_ok(np.asarray(cg)[mask],np.asarray(cs['step_gn'])[mask])
                cvalid=bool(cstop and cagree and harm.run_valid(cv) and int(cs['nfail'])==0)
                entry.update(coarse_valid=cvalid,coarse_valid_steps=count,coarse_stationarity=cstop,
                    coarse_gradient_agreement=cagree,coarse_corrections=int(cs['it_sum']),
                    coarse_physics_J=int(cs['steps'])+int(cs['it_sum']),coarse_predictor_residual_calls=3*int(cs['steps']),
                    coarse_ic_iterations=int(cs['ic_it']),accepted_candidates=int(np.asarray(st['replay_info'])[:,0].sum()),
                    unavailable_guesses=int(np.sum(~np.asarray(st['coarse_replay_valid']))))
                entry['valid']=entry['valid'] and cvalid
                extras.update(coarse_gamma_fresh=cg,coarse_residual_norm_fresh=crn,coarse_residual_tolerance=tc)
                info=np.asarray(st['replay_info']); good=(info[:,1].astype(bool)&info[:,4].astype(bool)&
                    (info[:,3]<info[:,2])&(info[:,3]<=(1-1e-8)*info[:,2]))
                assert np.array_equal(info[:,0].astype(bool),good)
                _,(warm_codes,objectives,promote_gap)=timed(lambda:guard_check(st['states'],cs['states'],float(cases[ci,4]),d,sch))
                np.testing.assert_allclose(objectives,info[:,2:4],rtol=2e-11,atol=1e-22,equal_nan=True)
                assert np.all(np.asarray(promote_gap)[mask]<1e-12)
                _,lifted_fields=timed(lambda:qrec['direct']['decode'](jnp.pad(v['coarse_W'],((0,0),(0,48))),d))
                np.testing.assert_allclose(lifted_fields,v['coarse_fields'],rtol=1e-12,atol=1e-13,equal_nan=True)
                np.savez_compressed(out/f'guard_case{ci}.npz',fine_warm_codes=warm_codes,objectives=objectives,
                                    promoted_coefficient_gap=promote_gap,coarse_valid_steps=count)
                entry['independent_guard_check']=True
                entry['reached_upward_field_identity']=True
                np.testing.assert_allclose(v['W'][0],records['direct',ci]['W'][0],rtol=1e-12,atol=1e-13)
                np.testing.assert_allclose(v['fields'][0],records['direct',ci]['fields'][0],rtol=1e-12,atol=1e-13)
                for key in ('ic_it','ic_reason','ic_idx'):
                    assert np.array_equal(st[key],records['direct',ci]['stats'][key])
                entry['fine_initial_state_matches_direct']=True
            rep['records'].append(entry)
            dump_npz(out/f'{arm}_case{ci}.npz',v,**extras)
            _,vt=timed(lambda:call(qtim[arm],ci,arm=='replay')); ts=vt['stats']
            np.testing.assert_allclose(vt['W'],v['W'],rtol=1e-12,atol=1e-13)
            np.testing.assert_allclose(vt['fields'],v['fields'],rtol=1e-12,atol=1e-13)
            assert P.gcert_ok(g,ts['step_gn'])
            np.testing.assert_allclose(ts['step_tolratio'],np.asarray(rn)/(tf+1e-300),rtol=1e-7,atol=1e-12)
            integer_parity(ts,st)
            tstop=bool(int(ts['nfail'])==0 and np.all((np.asarray(ts['step_gn'])<=1e-3)|(np.asarray(ts['step_tolratio'])<=1)))
            if arm=='replay':
                for k in ('coarse_W','coarse_fields'): np.testing.assert_allclose(vt[k],v[k],rtol=1e-12,atol=1e-13,equal_nan=True)
                cts=coarse_view(vt)['stats']; integer_parity(cts,cs)
                assert P.gcert_ok(np.asarray(cg)[mask],np.asarray(cts['step_gn'])[mask])
                np.testing.assert_allclose(np.asarray(cts['step_tolratio'])[mask],np.asarray(crn)[mask]/(tc+1e-300),rtol=1e-7,atol=1e-12)
                assert np.array_equal(ts['replay_info'][:,:2],st['replay_info'][:,:2])
                assert np.array_equal(ts['coarse_replay_valid'],st['coarse_replay_valid'])
            entry['timed_production_stationarity']=tstop; entry['valid']=entry['valid'] and tstop
            dump_npz(out/f'timed_identity_{arm}_case{ci}.npz',vt)
            witnesses[arm,ci]=dict(arrays={k:np.asarray(x) for k,x in vt.items() if k!='stats'},
                                   stats={k:np.asarray(x) for k,x in ts.items()})
        vd=records['direct',ci]
        _,vn=timed(lambda:call(qnone,ci))
        err=float(jnp.max(jnp.abs(vn['stats']['states']-vd['stats']['states'])))
        integer_parity(vn['stats'],vd['stats']); assert err<=1e-12
        rep['parity'].append(dict(case=ci,check='disabled',state_max_abs=err,passed=True))
        if not did_forced and entry['valid']:
            for step,q in ((0,qfail0),(19,qfailmid)):
                _,vbad=timed(lambda:call(q,ci,True)); bs=vbad['stats']
                assert int(bs['coarse_steps'])==step+1 and int(bs['coarse_valid_steps'])==step
                assert int(bs['coarse_step_reason'][step])==3 and np.isnan(np.asarray(bs['coarse_states'])[step+1:]).all()
                assert not np.any(np.asarray(bs['coarse_replay_valid'])[step:])
                assert not np.any(np.asarray(bs['replay_info'])[step:,0])
                np.testing.assert_allclose(bs['coarse_states'][:step+1],records['replay',ci]['stats']['coarse_states'][:step+1],rtol=1e-12,atol=1e-13)
                if step==0:
                    err=float(jnp.max(jnp.abs(bs['states']-vd['stats']['states'])))
                    integer_parity(bs,vd['stats']); assert err<=1e-12
                    np.testing.assert_allclose(vbad['fields'],vd['fields'],rtol=1e-12,atol=1e-13)
                _,(bg,br)=timed(lambda:cert_f(bs['states'],float(cases[ci,4]),d,sch))
                assert np.all((np.asarray(bg)<=1e-3)|(np.asarray(br)<=tf)) and P.gcert_ok(bg,bs['step_gn'])
                dump_npz(out/f'forced_failure_case{ci}_step{step}.npz',vbad,gamma_fresh=bg,residual_norm_fresh=br,residual_tolerance=tf)
                rep['parity'].append(dict(case=ci,check='forced_coarse_failure',step=step,passed=True))
            did_forced=True
        print('RECORDED',ci,[(x['arm'],x['valid']) for x in rep['records'][-2:]],flush=True); save()
    QS.burn(.1); rng=np.random.default_rng(SEED)
    for block in range(7):
        for ci_ in rng.permutation(6):
            ci=int(ci_)
            for ai in rng.permutation(2):
                arm=ARMS[ai]; QS.burn(.1)
                sec,v=timed(lambda:call(qtim[arm],ci,arm=='replay'))
                identity=same_witness(v,witnesses[arm,ci]); name=f'timed_identity_{arm}_case{ci}.npz'
                if not identity:
                    name=f'timed_variant_{arm}_case{ci}_block{block}.npz'; dump_npz(out/name,v)
                good=bool(harm.run_valid(v) and int(v['stats']['nfail'])==0)
                if arm=='replay': good=good and bool(harm.run_valid(coarse_view(v)) and int(v['stats']['coarse_nfail'])==0 and int(v['stats']['coarse_valid_steps'])==50)
                rep['timings'].append(dict(arm=arm,case=ci,block=block,seconds=sec,identity=identity,online_valid=good,
                    array_witness=name,W_sha256=sha(v['W']),fields_sha256=sha(v['fields']),
                    coarse_fields_sha256=sha(v['coarse_fields']) if arm=='replay' else None,
                    fine_corrections=int(v['stats']['it_sum']),coarse_corrections=int(v['stats']['coarse_it_sum']) if arm=='replay' else 0))
        print('TIMING_BLOCK',block,flush=True); save()
    arr={(a,c):np.array([t['seconds'] for t in rep['timings'] if t['arm']==a and t['case']==c]) for a in ARMS for c in range(6)}
    ratios=[float(np.median(arr['replay',c]/arr['direct',c])) for c in range(6)]
    rep['checks']['forced_failure_controls_completed']=did_forced
    quality=did_forced and all(x['valid'] for x in rep['records']); identity=all(t['identity'] and t['online_valid'] for t in rep['timings'])
    rep['gate']=dict(per_case_paired_median_ratios=ratios,median_ratio=float(np.median(ratios)),maximum_case_ratio=max(ratios),
        all_numerically_valid=quality,all_timing_identities=identity,passed=bool(quality and identity and np.median(ratios)<=.85 and max(ratios)<=1.10))
    rep['continue_accuracy']=rep['gate']['passed']
    rep['strong_accuracy_checks']='not run in phase1; cost pass requires separately reviewed tight/tighter and Gauss160/192 qualification'
    rep['complete']=True; rep['elapsed_program_seconds']=time.perf_counter()-started; rep['memory']=P.mem(); save()
    print('FINAL',json.dumps(clean(rep['gate'])),flush=True)


if __name__=='__main__': main()
