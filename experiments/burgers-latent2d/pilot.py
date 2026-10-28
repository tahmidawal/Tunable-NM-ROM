"""Approved dev6 cost screen: no validation, references or sealed cohorts."""
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
import latentcore as C
from transfer import make_candidate, make_fit, INFO
import jax
import jax.numpy as jnp

ARMS = ('direct', 'physics', 'transfer')
SEED = 20261028
HEAD_HASH = {'coarse': '764633e87de101243bf8154fb3352ccda4018249ae19926728850794358dab8f',
             'fine': '63f287d1f907bc3fe779a86999dfaec9a2029c73e34e8fce743cfbaa55380031'}


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if hasattr(x, 'tolist'): return clean(x.tolist())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def timed(fn):
    t = time.perf_counter(); v = fn(); jax.block_until_ready(v)
    return time.perf_counter() - t, v


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def dump_npz(path, v, **extra):
    np.savez_compressed(path, W=v['W'], fields=v['fields'],
                        **{k: np.asarray(x) for k, x in v['stats'].items()}, **extra)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    P.check_env({}); started = time.perf_counter()
    rep = dict(complete=False, scope='dev6 bounded latent initialization cost screen',
        commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
        backend=jax.default_backend(), gpu=P.gpu_info(), jax_version=jax.__version__,
        precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), x64=bool(jax.config.jax_enable_x64),
        settings=dict(L=256, fine_K=32, coarse_K=16, bank_R=512, M=128, rule='gauss96',
                      scheme='BE', dt=.005, steps=50, gtol=1e-3, tolf=1e-9,
                      fit_budget=4, coarse_budget=2, fine_budget=600, repetitions=7, order_seed=SEED),
        records=[], timings=[], parity=[], diagnostics=[], setup={}, checks={})
    def save():
        t = out/'result.json.tmp'; t.write_text(json.dumps(clean(rep), indent=2, allow_nan=False)+'\n')
        t.replace(out/'result.json')
    save()
    t = time.perf_counter()
    bank = P.Bank(256, [0, 128, 256, 384, 512])
    c = HC.Head(ROOT/'experiments/jcp-heads/heads/head_mid_k16_s0.npz', bank.mdl)
    f = HC.Head(ROOT/'experiments/jcp-heads/heads/head_mid_k32_s1.npz', bank.mdl)
    assert c.sha256 == HEAD_HASH['coarse'] and f.sha256 == HEAD_HASH['fine']
    assert c.K == 16 and f.K == 32 and len(c.Zsub) == 8192
    fine = HC.HeadSetting(bank, 'H32s1', f, 'gauss96', M=128, nrec=51, nstep=50)
    fine.check('main', bank)
    d = fine.data['main']; cold_c = c.cold(bank.mdl)
    assert d['A'].shape == (128, 512) and d['Gq'].shape == (9216, 512)
    assert fine.mode_hash == '6628446b0ddf2fe13810382e38e7bca0d80d17312654309135246c0876cffcb7'
    R = fine.cold[3]
    # Common spatial bank means both QR factors describe precisely the same metric.
    np.testing.assert_allclose(R, cold_c[3], rtol=1e-12, atol=1e-13)
    hf, hc = H.head_fn(f.pf), H.head_fn(c.pf)
    coarse_Hrot = jax.jit(jax.vmap(hc))(cold_c[6]) @ R.T
    data = dict(d, transfer_R=R, coarse_Hrot=coarse_Hrot,
                coarse_Hnorm=jnp.sum(coarse_Hrot*coarse_Hrot,axis=1), coarse_Zcand=cold_c[6])
    nl, nlJ = Q.tested_value('point', 256), Q.tested_jac('point', 256)
    qrec = {'direct': fine.query(record=True)}; qtim = {'direct': fine.query(record=False)}
    for arm in ARMS[1:]:
        seed = make_candidate(hc, hf, nl, nlJ, c.trust, arm)
        qrec[arm] = C.make_head_query(f.pf, 32, 256, f.trust, nrec=51, nstep=50, seed=seed)
        qtim[arm] = C.make_head_query(f.pf, 32, 256, f.trust, nrec=0, nstep=50, seed=seed)
    qnone = C.make_head_query(f.pf, 32, 256, f.trust, nrec=51, nstep=50)
    qfail = C.make_head_query(f.pf, 32, 256, f.trust, nrec=51, nstep=50,
                            seed=make_candidate(hc, hf, nl, nlJ, c.trust, 'physics', force_failure=True))
    rep['setup'] = dict(seconds=time.perf_counter()-t, heads=dict(coarse=c.sha256, fine=f.sha256),
                        trust=dict(coarse=c.trust, fine=f.trust), memory=P.mem())
    rep['table_meta'] = fine.meta
    rep['table_hashes'] = {k: sha(d[k]) for k in ('A','lam','Gq','Gs','Psi')}
    rep['transfer_hashes'] = {k: sha(data[k]) for k in ('transfer_R','coarse_Hrot','coarse_Hnorm','coarse_Zcand')}
    rep['bank_sha256'] = HC.sha_file(T2.CKPT); rep['rotation_sha256'] = HC.sha_file(Q.ROTATION)
    # No case_list(), validation generation, references or oracle data.
    cases = Q.cohort('dev6'); assert cases.shape == (6, 5)
    U0 = [jnp.asarray(Q.e.initial(256, ph)) for ph in cases]
    rep['cases'] = cases; rep['seeds'] = dict(dev4=7090702, dev2=911702, order=SEED)
    rep['initial_hashes'] = [sha(u) for u in U0]
    sch = T2.sched('BE', .005, 1e-3, 1e-9)
    def call(q, ci, use_aux=False):
        return q['fused'](U0[ci], float(cases[ci,4]), data if use_aux else d, fine.cold, sch)
    # Correctness checks on actual bank/heads before timing.
    B48, _, _ = bank.mdl.values_grads(np.asarray(fine.cold[0]), 512)
    WB = fine.cold[1][:, None] * B48
    vectors = jnp.asarray(np.random.default_rng(SEED).normal(size=(512, 5)))
    a48 = np.asarray(WB @ vectors); aR = np.asarray(R @ vectors)
    metric_gap = float(np.max(np.abs(np.sum(a48*a48, axis=0)-np.sum(aR*aR, axis=0)) / (1+np.sum(a48*a48, axis=0))))
    sv = np.linalg.svd(np.asarray(R), compute_uv=False)
    assert metric_gap < 1e-11 and sv[-1] > 1e-12 * sv[0]
    rep['checks']['metric'] = dict(relative_squared_norm_gap=metric_gap, rank=int(np.sum(sv>1e-12*sv[0])), condition=sv[0]/sv[-1])
    for name, hz, z in [('fine',hf,jnp.asarray(f.Zsub[0])),('coarse',hc,jnp.asarray(c.Zsub[0]))]:
        fit = make_fit(hz)
        vv = fit(z, R@hz(z), R); jax.block_until_ready(vv)
        assert np.array_equal(vv[0], z) and int(vv[2]) == 0
        # Retained fit solver is the independent positive arithmetic control.
        ref = H.A_.make_stationary_lm(lambda zz,tgt,RR: RR@hz(zz)-tgt,4,gtol=1e-6,linear='gj')
        target = R@hz(z+0.001)
        ours = jax.jit(fit)(z,target,R); theirs = jax.jit(ref)(z,(target,R),0.)
        np.testing.assert_allclose(ours[0],theirs[0],rtol=1e-12,atol=1e-13)
        assert int(ours[2]) == int(theirs[2]) and int(ours[3]) == int(theirs[3])
        _, res, ev = H.make_head_evolve(hz, nl, nlJ, len(z), f.trust)
        co = sch['co']; cc = hf(jnp.asarray(f.Zsub[0])); hist=(d['A']@cc,d['A']@cc,jnp.zeros(128))
        args = (hist,float(cases[0,4]),d,co,sch['dt'])
        rv,J = ev(z,args); Ja = jax.jacfwd(res)(z,args)
        err = float(jnp.linalg.norm(J-Ja)/(1+jnp.linalg.norm(Ja)))
        assert err < 1e-11
        rep['checks'][name] = dict(identity_transfer=True,fit_matches_retained_solver=True,analytic_AD_gap=err)
    rep['compile_warm_seconds'] = {}
    for arm in ARMS:
        for label, q in [('recorded',qrec[arm]),('timed',qtim[arm])]:
            dur,_ = timed(lambda:call(q,0,arm!='direct')); rep['compile_warm_seconds'][arm+'_'+label]=dur
    print('SETUP_PASS',round(time.perf_counter()-started,3),flush=True); save()
    cert = H.make_head_certifier(hf,nl)
    records, witnesses = {}, {}
    for ci in range(6):
        for arm in ARMS:
            dur,v = timed(lambda:call(qrec[arm],ci,arm!='direct')); records[(arm,ci)] = v
            st = v['stats']; states = st['states']
            _, (g,rn) = timed(lambda:cert(states,float(cases[ci,4]),d,sch))
            _, scale, _ = qrec['direct']['init'](U0[ci],fine.cold)
            fresh = bool(np.all(np.isfinite(g)) and np.all((np.asarray(g)<=1e-3)|(np.asarray(rn)<=float(scale)*1e-9)))
            agree = P.gcert_ok(np.asarray(g),np.asarray(st['step_gn']))
            valid = bool(harm.run_valid(v) and fresh and agree and int(st['nfail'])==0)
            info = np.asarray(st.get('seed_info',np.zeros((50,16))))
            entry = dict(case=ci,arm=arm,valid=valid,fresh_stationarity=fresh,gradient_agreement=agree,
                         gamma_max=float(np.max(g)),recorded_seconds=dur,
                         fine_corrections=int(st['it_sum']),fine_J=50+int(st['it_sum']),
                         fine_rejections=int(st['rej']),fine_guard_residuals=0 if arm=='direct' else 100,
                         coarse_code_searches=0 if arm=='direct' else 50,
                         auxiliary_sums=dict(zip(INFO,info.sum(axis=0).tolist())),
                         auxiliary_failures=int(np.sum(np.any(info[:,[2,6,9]]==3,axis=1))))
            entry['coarse_J'] = (50+int(info[:,5].sum())) if arm=='physics' else 0
            rep['records'].append(entry)
            dump_npz(out/f'{arm}_case{ci}.npz',v,gamma_fresh=g,residual_norm_fresh=rn)
            _, vt = timed(lambda:call(qtim[arm],ci,arm!='direct'))
            np.testing.assert_allclose(vt['W'],v['W'],rtol=1e-12,atol=1e-13)
            np.testing.assert_allclose(vt['fields'],v['fields'],rtol=1e-12,atol=1e-13)
            assert P.gcert_ok(np.asarray(g),np.asarray(vt['stats']['step_gn']))
            # Same roundoff envelope as gcert: relative1e-7 + absolute1e-12.
            np.testing.assert_allclose(vt['stats']['step_tolratio'],np.asarray(rn)/(float(scale)*1e-9+1e-300),rtol=1e-7,atol=1e-12)
            assert int(vt['stats']['nfail'])==0
            assert np.all((np.asarray(vt['stats']['step_gn'])<=1e-3)|(np.asarray(vt['stats']['step_tolratio'])<=1))
            for k in ('step_it','step_reason','it_sum','it_max','rej','exits','nfail','ic_it','ic_reason','ic_idx'):
                assert np.array_equal(vt['stats'][k],st[k])
            if arm!='direct':
                assert np.array_equal(vt['stats']['seed_info'][:,:12],st['seed_info'][:,:12])
            entry['timed_certificate_matches']=True
            dump_npz(out/f'timed_identity_{arm}_case{ci}.npz',vt)
            witnesses[(arm,ci)] = {k: np.asarray(vt[k]) for k in ('W','fields')}
            witnesses[(arm,ci)]['stats']={k:np.asarray(x) for k,x in vt['stats'].items()}
        vd = records[('direct',ci)]
        for label,q in [('disabled',qnone)]+([('forced_failure',qfail)] if ci==0 else []):
            _,vv = timed(lambda:call(q,ci,label=='forced_failure'))
            err=float(jnp.max(jnp.abs(vv['stats']['states']-vd['stats']['states'])))
            decisions=all(np.array_equal(vv['stats'][k],vd['stats'][k]) for k in ('step_it','step_reason','exits','rej'))
            ok = err<=1e-12 and decisions
            rep['parity'].append(dict(case=ci,check=label,state_max_abs=err,decisions_equal=decisions,passed=ok))
            assert ok
            if label=='forced_failure':
                assert np.all(np.asarray(vv['stats']['seed_info'])[:,2]==3)
                assert not np.any(np.asarray(vv['stats']['seed_info'])[:,0])
        print('RECORDED',ci,[(x['arm'],x['fine_corrections'],x['valid']) for x in rep['records'][-3:]],flush=True); save()
    rng=np.random.default_rng(SEED)
    QS.burn(.1)  # Compile the burn kernel before any measured block.
    for block in range(7):
        for ci_ in rng.permutation(6):
            ci=int(ci_)
            for ai in rng.permutation(3):
                arm=ARMS[ai]; QS.burn(.1)
                elapsed,v=timed(lambda:call(qtim[arm],ci,arm!='direct'))
                want=witnesses[(arm,ci)]
                identity=all(np.array_equal(np.asarray(v[k]),want[k]) for k in ('W','fields')) and all(
                    np.array_equal(v['stats'][k],x) for k,x in want['stats'].items())
                path=f'timed_identity_{arm}_case{ci}.npz'
                if not identity:
                    path=f'timed_variant_{arm}_case{ci}_block{block}.npz'; dump_npz(out/path,v)
                rep['timings'].append(dict(block=block,case=ci,arm=arm,seconds=elapsed,identity=identity,
                    W_sha256=sha(v['W']),fields_sha256=sha(v['fields']),array_witness=path,
                    fine_corrections=int(v['stats']['it_sum']),online_valid=bool(harm.run_valid(v) and int(v['stats']['nfail'])==0)))
        print('TIMING_BLOCK',block,flush=True); save()
    # Transfer diagnostics on every reached step, outside measured invocations.
    X128,w128=Q.offmesh_rule('gauss128'); B128,_,_=bank.mdl.values_grads(X128,512)
    _,w96=Q.offmesh_rule('gauss96')
    hfv=jax.jit(jax.vmap(hf)); hcv=jax.jit(jax.vmap(hc))
    def norm_rows(delta,B,w):
        return np.asarray(jnp.sqrt(jnp.sum((delta@B.T)**2*jnp.asarray(w)[None,:],axis=1)))
    for arm in ARMS[1:]:
        for ci in range(6):
            v=records[(arm,ci)]; s=v['stats']; states=np.asarray(s['states']); codes=s['seed_codes']
            zfit,zc,zu=codes[:,16:32],codes[:,32:48],codes[:,48:]
            warm=[]; staged=[]; nu=float(cases[ci,4])
            for k in range(50):
                zn=jnp.asarray(states[k]); zp=jnp.asarray(states[max(k-1,0)]); zp2=jnp.asarray(states[max(k-2,0)])
                hist=(d['A']@hf(zn),d['A']@hf(zp),jnp.zeros(128))
                args=(hist,nu,d,sch['co'],sch['dt'])
                cand=jnp.stack((zn,2*zn-zp,3*zn-3*zp+zp2 if k>=2 else 2*zn-zp))
                rv=jax.vmap(lambda zz:qrec['direct']['res'](zz,args))(cand)
                zw=cand[jnp.argmin(jnp.linalg.norm(rv,axis=1))]; warm.append(zw)
                rc=T2.lspg_residual(nl,sch['co'],sch['dt'],nu,d,hist)
                staged.append(jnp.stack([jnp.vdot(rc(cc),rc(cc)) for cc in (hf(zw),hc(zfit[k]),hc(zc[k]),hf(zu[k]))]))
            wf=hfv(jnp.stack(warm)); cf=hcv(zfit); cp=hcv(zc); fu=hfv(zu)
            down,disp,up = (np.asarray(jnp.linalg.norm(delta@R.T,axis=1)) for delta in (cf-wf,cp-cf,fu-cp))
            ratio=np.divide(up,disp,out=np.full_like(up,np.nan),where=disp>0)
            fields={}
            for tag,B,w in [('96',d['Gq'],w96),('128',B128,w128)]:
                for name,delta in [('down',cf-wf),('displacement',cp-cf),('up',fu-cp)]:
                    fields[name+tag]=norm_rows(delta,B,w)
            initial_norm=float(jnp.linalg.norm(R@hf(v['W'][0])))
            np.savez_compressed(out/f'diagnostics_{arm}_case{ci}.npz',down=down,displacement=disp,up=up,
                                up_over_displacement=ratio,zero_displacement=disp==0,objectives=jnp.stack(staged),
                                fine_warm_codes=jnp.stack(warm),initial_field_norm=initial_norm,**fields)
            rep['diagnostics'].append(dict(arm=arm,case=ci,up_field_relative_max=float(np.max(fields['up128'])/initial_norm),
                down_field_relative_max=float(np.max(fields['down128'])/initial_norm),zero_displacement_count=int(np.sum(disp==0))))
    arr={(arm,ci):np.array([t['seconds'] for t in rep['timings'] if t['arm']==arm and t['case']==ci]) for arm in ARMS for ci in range(6)}
    rep['gate']={}
    for arm in ARMS[1:]:
        ratios=[float(np.median(arr[(arm,ci)]/arr[('direct',ci)])) for ci in range(6)]
        valid=all(x['valid'] for x in rep['records'] if x['arm'] in ('direct',arm))
        identity=all(x['identity'] and x['online_valid'] for x in rep['timings'] if x['arm'] in ('direct',arm))
        rep['gate'][arm]=dict(per_case_paired_median_ratios=ratios,median_ratio=float(np.median(ratios)),
            maximum_case_ratio=max(ratios),all_numerically_valid=valid,all_timing_identities=identity,
            passed=valid and identity and np.median(ratios)<=.85 and max(ratios)<=1.10)
    rep['physics_vs_transfer_per_case']=[float(np.median(arr[('physics',ci)]/arr[('transfer',ci)])) for ci in range(6)]
    rep['continue_accuracy']=any(g['passed'] for g in rep['gate'].values())
    rep['strong_accuracy_checks']='not run in cost screen; conditional tight/tighter and successor qualification required after a cost pass'
    rep['complete']=True; rep['elapsed_program_seconds']=time.perf_counter()-started; rep['memory']=P.mem(); save()
    print('FINAL',json.dumps(clean(rep['gate'])),flush=True)


if __name__=='__main__': main()
