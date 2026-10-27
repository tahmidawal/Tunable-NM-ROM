"""A3: one archived k16 support at one independently frozen no-HR endpoint."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import numpy as np
import replay as X
import hr_interface as I
from fit import conv

jax,jnp,H,R,P,B,S=X.jax,X.jnp,X.H,X.R,X.P,X.B,X.S
HERE=Path(__file__).resolve().parent


def layers(cases,ids,Q1):
    mm=[cases[str(i)]['k16']['whole_residual'] for i in ids]
    dg=np.concatenate([m['delta_g'] for m in mm]);gg=np.concatenate([m['full_gamma'] for m in mm])
    finite=bool(all(all(m['finite']) for m in mm) and np.isfinite(dg).all() and np.isfinite(gg).all())
    fields={str(i):bool(np.isfinite(cases[str(i)]['k16_to_none']) and cases[str(i)]['k16_to_none']<=.00025) for i in ids}
    gradient={str(i):(np.isfinite(cases[str(i)]['k16']['whole_residual']['delta_g']) &
        (np.asarray(cases[str(i)]['k16']['whole_residual']['delta_g'])<=.001)).tolist() for i in ids}
    bad={str(i):((~np.isfinite(cases[str(i)]['k16']['whole_residual']['full_gamma'])) |
        (np.asarray(cases[str(i)]['k16']['whole_residual']['full_gamma'])>.001)).tolist() for i in ids}
    count=sum(sum(v) for v in bad.values());steps=len(gg)
    common=bool(all(q['q1'] for q in Q1.values()) and all(fields.values()))
    gp=bool(finite and all(all(v) for v in gradient.values()));sp=bool(finite and count<=.01*steps)
    return dict(common_layer_pass=common,shared_field_case_mask=fields,shared_field_bar=.00025,
        extra_gradient_step_mask=gradient,extra_gradient_bar=.001,extra_gradient_pass=gp,
        extra_full_stationarity_bad_step_mask=bad,extra_full_stationarity_bad_steps=count,
        extra_full_stationarity_steps=steps,extra_full_stationarity_fraction_limit=.01,
        extra_full_stationarity_pass=sp,finite=finite,delta_g_max=float(dg.max()),full_gamma_max=float(gg.max()),
        strict_extra_layer_pass=bool(gp and sp),both_layers_pass=bool(common and gp and sp),
        full64_adapter_release=bool(common and gp and sp))


def run(cfg,artifacts,out):
    start=time.perf_counter()
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    assert cfg['attempt']=='k16fine01' and cfg['driver']=='fine_replay'
    ids=cfg['cases'];assert ids==[0,5,30,43] and cfg['tighter_cases']==[0,5]
    assert cfg['rank']==512 and cfg['L']==64 and cfg['cohort_seed']==923801 and cfg['cohort_count']==64
    pin=HERE/cfg['selection_manifest'];assert X.digest(pin)==cfg['selection_sha256']
    sel=json.loads(pin.read_text());assert sel['independent_results_gate']=='READY'
    refp=HERE/sel['reference_fixture'];assert X.digest(refp)==sel['reference_fixture_sha256']
    expected=np.load(refp,allow_pickle=False)
    endpoint=sel['selected'];scheme,dt,steps=endpoint['scheme'],endpoint['dt'],endpoint['steps']
    assert steps in (10,15,20,25,40) and scheme in ('BDF2','CN') and abs(steps*dt-.25)<1e-14
    assert endpoint['gtol'] in (.001,.0001) and endpoint['tolf']==R.LEVELS['prod'][1]
    artifact=json.loads((HERE/'inputs/MANIFEST.json').read_text())
    for name,h in artifact['files'].items():assert X.digest(artifacts/name)==h
    assert not out.exists();out.mkdir();(out/'arrays').mkdir()
    src=json.loads((out.parent/'PROVENANCE.json').read_text())
    rep=dict(config=cfg,selection=sel,selection_sha256=X.digest(pin),source_commit=src['commit'],
        source_manifest_sha256=X.digest(out.parent/'PROVENANCE.json'),job_id=os.environ['SLURM_JOB_ID'],
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip(),
        backend='gpu',precision='f64/highest',artifacts=artifact,cases={},timing=[],complete=False,
        contract='given full63^3 field: full projection, evolution and six full decodes; parameter preparation separate',
        scope='A3 four-case diagnostic only; old failures and constructor stop preserved')
    def save():
        (out/'result.json.tmp').write_text(json.dumps(rep,indent=2,default=conv)+'\n');os.replace(out/'result.json.tmp',out/'result.json')
    def budget():
        if time.perf_counter()-start>=cfg['budget_hours']*3600:
            save();raise TimeoutError('A3 frozen budget reached; partial evidence retained')
    save()
    full,hr,meta=I.archived_data(artifacts/'Phi.npz',artifacts/'HR~ecsw~512~k16.npz')
    data=dict(none=full,k16=hr);trust=meta['trust_radius'];arms=['k16','none']
    common={};native={};cert={};sch={};st={};stt={}
    for a in arms:
        kind='none' if a=='none' else 'ecsw'
        common[a],_,_,_,cert[a],schedule=I.functions_for(kind,trust,steps,'common')
        native[a],_,_,_,_,ns=I.functions_for(kind,trust,steps,'native')
        sch[a]=dict(schedule(scheme,dt),gtol=jnp.float64(endpoint['gtol']),tolf=jnp.float64(endpoint['tolf']))
        st[a],stt[a]=ns(scheme,dt,'tight'),ns(scheme,dt,'tighter')
        assert int(sch[a]['steps'])==steps and len(sch[a]['slot'])==steps+1
    fam=H.Family(3,'val64',count=64);refs=S.Refs(3,'val64',fam,ids)
    rep.update(cohort_sha256=fam.sha256,reference=dict(store_version=S.STORE_VERSION,status_sha256=S.STATUS_SHA,kind='ST',
        cases={i:dict(refs.st[i],n0=refs.n0(i)) for i in ids}),steps=steps,
        actual_schedule={k:np.asarray(v).tolist() for k,v in sch['none'].items()})
    metrics=X.make_metrics(64,512,steps)
    records={a:{} for a in arms};tight={a:{} for a in arms};tighter={a:{} for a in arms}
    warm=jax.block_until_ready(jnp.asarray(fam.ic(64,ids[0])));wnu=jnp.float64(fam.nu(ids[0]))
    for a in arms:jax.block_until_ready(common[a](warm,wnu,data[a],sch[a]))
    for i in ids:
        budget();t0=time.perf_counter();u=jax.block_until_ready(jnp.asarray(fam.ic(64,i)));nu=jnp.float64(fam.nu(i))
        prep=time.perf_counter()-t0;inp=X.array_hash(u)
        assert inp==sel['input_sha256'][str(i)],'selected input binding mismatch'
        rep['cases'][str(i)]={};fields={}
        for a in arms:
            q={}
            for level in ('prod','tight')+(('tighter',) if i in cfg['tighter_cases'] else ()):
                budget();ss={'prod':sch[a],'tight':st[a],'tighter':stt[a]}[level]
                fn=common[a] if level=='prod' else native[a]
                if level=='prod':P.burn(cfg['burn_seconds'])
                t0=time.perf_counter();o=jax.block_until_ready(fn(u,nu,data[a],ss));seconds=time.perf_counter()-t0
                gamma=jax.block_until_ready(cert[a](o['C'],nu,data[a],ss));cc=R.certify(o,gamma,ss,level)
                extra_fields=dict(fields=o['fields']) if level=='prod' else {}
                np.savez_compressed(out/'arrays'/f'{a}_c{i:03}_{level}.npz',case=i,W=o['W'],C=o['C'],**extra_fields,
                    gamma=gamma,reason=o['stats']['reason'],it=o['stats']['it'],lam=o['stats']['lam'],
                    scale=o['scale'],c0=o['c0'],times=np.arange(6)*.05,dt=dt)
                q[level]=dict(out=o,cert=cc,seconds=seconds)
            prod=q['prod'];o=prod['out'];F=np.asarray(o['fields']);fields[a]=F
            row=dict(e0=refs.score_safe(F,i),query_seconds=prod['seconds'],cert=prod['cert'],
                input_prepare_seconds=prep,input_sha256=inp,W_sha256=X.array_hash(o['W']),fields_sha256=X.array_hash(F),
                tight=q['tight']['cert'],tight_distance=S.lattice_distance(F,q['tight']['out']['fields'],refs.n0(i)))
            if 'tighter' in q:row.update(tighter=q['tighter']['cert'],
                tight_tighter_distance=S.lattice_distance(q['tight']['out']['fields'],q['tighter']['out']['fields'],refs.n0(i)))
            if a=='k16':row['whole_residual']=X.metric_summary(jax.block_until_ready(metrics(o['C'],nu,full,hr,sch[a])))
            else:
                Cref,Wref,Fref=[expected[f'{k}_c{i:03}'] for k in ('C','W','fields')]
                c0=np.asarray(o['c0']);c0ref=Cref[0]
                dc0=float(np.linalg.norm(c0-c0ref)/np.linalg.norm(c0ref))
                dc0field=float(jnp.linalg.norm(full['Phi']@(o['c0']-jnp.asarray(c0ref))))/refs.n0(i)
                cdist=float(np.linalg.norm(np.asarray(o['C'])-Cref)/np.linalg.norm(Cref))
                fdist=S.lattice_distance(F,Fref,refs.n0(i))
                row['endpoint_reproduction']=dict(C_bit_equal=np.array_equal(o['C'],Cref),W_bit_equal=np.array_equal(o['W'],Wref),
                    fields_bit_equal=np.array_equal(F,Fref),c0_bit_equal=np.array_equal(c0,c0ref),
                    C_relative=cdist,field_distance=fdist,c0_relative=dc0,c0_field_distance=dc0field,
                    pass_=bool(cdist<=1e-10 and fdist<=1e-10 and dc0<=1e-12 and dc0field<=1e-12))
            rep['cases'][str(i)][a]=row;ck=f'val64|{i}'
            records[a][ck]=dict(cert=prod['cert'],fields_lat=F)
            tight[a][ck]=dict(cert=q['tight']['cert'],fields_lat=np.asarray(q['tight']['out']['fields']))
            if 'tighter' in q:tighter[a][ck]=dict(cert=q['tighter']['cert'],fields_lat=np.asarray(q['tighter']['out']['fields']))
            save()
        rep['cases'][str(i)]['k16_to_none']=S.lattice_distance(fields['k16'],fields['none'],refs.n0(i));save()
        print('completed_case',i,flush=True)
    keys=[f'val64|{i}' for i in ids];anchors=keys[:2];norms={f'val64|{i}':refs.n0(i) for i in ids}
    rep['Q1']={a:X.q1(records[a],tight[a],tighter[a],norms,anchors,keys) for a in arms}
    rep['admission']=layers(rep['cases'],ids,rep['Q1'])
    rep['endpoint_reproduction_pass']=all(rep['cases'][str(i)]['none']['endpoint_reproduction']['pass_'] for i in ids)
    rep['admission']['full64_adapter_release'] &= rep['endpoint_reproduction_pass']
    panels=[(i,r) for i in cfg['timing_cases'] for r in range(cfg['timing_reps'])];np.random.default_rng(20261026).shuffle(panels)
    for i,r in panels:
        budget();u=jax.block_until_ready(jnp.asarray(fam.ic(64,i)));nu=jnp.float64(fam.nu(i));row=dict(case=i,rep=r,invocations=[])
        for a in ('k16','none','k16'):
            P.burn(cfg['burn_seconds']);t0=time.perf_counter();o=jax.block_until_ready(common[a](u,nu,data[a],sch[a]));seconds=time.perf_counter()-t0
            pin=np.load(out/'arrays'/f'{a}_c{i:03}_prod.npz');identical=X.scalar_check(o['W'],pin['W']) and X.scalar_check(o['fields'],pin['fields'])
            row['invocations'].append(dict(arm=a,seconds=seconds,identity=identical,W_sha256=X.array_hash(o['W']),
                fields_sha256=X.array_hash(o['fields']),e0=refs.score_safe(o['fields'],i)))
        ts=[v['seconds'] for v in row['invocations']];row.update(drift=ts[0]/ts[2],drift_accepted=abs(ts[0]/ts[2]-1)<=.25)
        rep['timing'].append(row);save()
    rep['timing_coverage_complete']=all(v['identity'] and v['e0']['eligible'] for r in rep['timing'] for v in r['invocations'])
    rep['admission']['full64_adapter_release'] &= rep['timing_coverage_complete']
    rep.update(complete=True,elapsed_seconds=time.perf_counter()-start,peak_device_bytes=P.device_peak(),peak_host_bytes=P.host_rss())
    save();print('A3_complete',rep['admission']['full64_adapter_release'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--artifacts',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),Path(a.artifacts),Path(a.out))
