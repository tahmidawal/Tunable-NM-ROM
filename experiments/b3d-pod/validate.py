"""Full-population no-HR fallback and single frozen k32 admission; no support search."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import replay as X
from fit import conv

jax,jnp,H,R,P,B,S=X.jax,X.jnp,X.H,X.R,X.P,X.B,X.S


def run(cfg,artifacts,out):
    start=time.perf_counter()
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    assert cfg['driver']=='validate' and cfg['arms'] in (['none'],['k32','none'])
    ids=cfg['cases'];arms=cfg['arms']
    assert ids in (list(range(64)),[0,5,30,43]) and cfg['tighter_cases']==ids[:2]
    assert cfg['cohort_seed']==923801 and cfg['cohort_count']==64 and cfg['rank']==512 and cfg['L']==64
    assert cfg['scheme']=='BDF2' and cfg['dt_numerator']==1 and cfg['dt_denominator']==60
    assert not out.exists();out.mkdir(parents=True);(out/'arrays').mkdir()
    source=json.loads((out.parent/'PROVENANCE.json').read_text())
    manifest=json.loads((X.HERE/cfg['artifact_manifest']).read_text())
    for name,h in manifest['files'].items():assert X.digest(artifacts/name)==h
    rep=dict(config=cfg,source_commit=source['commit'],source_manifest_sha256=X.digest(out.parent/'PROVENANCE.json'),
             job_id=os.environ['SLURM_JOB_ID'],backend='gpu',precision='f64/highest',artifacts=manifest,
             gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip(),
             cases={},timing=[],complete=False)
    def save():
        (out/'result.json.tmp').write_text(json.dumps(rep,indent=2,default=conv)+'\n')
        os.replace(out/'result.json.tmp',out/'result.json')
    def budget():
        if time.perf_counter()-start>=cfg['budget_hours']*3600:
            save();raise TimeoutError('registered validation budget exhausted; retained partial evidence')
    save()
    z=np.load(artifacts/'Phi.npz');trust=float(z['trust'][list(z['ranks']).index(512)])
    ph=np.asarray(z['Phi'][:,:512]);orth=float(np.max(np.abs(ph.T@ph-np.eye(512))))
    assert 512 in z['constructible'] and orth<=1e-10 and np.isfinite(ph).all()
    Phi=jax.block_until_ready(jnp.asarray(ph));del ph,z
    cell=P.Cell(3,64);cell.set_basis(Phi,{512:trust});cell.constructible_r={512}
    data={'none':cell.arm_data('none',512,'-')[0]}
    if 'k32' in arms:
        w=np.load(artifacts/'HR~ecsw~512~k32.npz')
        assert len(w['nodes'])==manifest['points'] and 0<len(w['nodes'])<=16384
        assert np.isfinite(w['weights']).all() and np.all(w['weights']>0)
        dat,_=B.ecsw_data(Phi,w['nodes'],w['weights'],3,64);data['k32']=jax.block_until_ready(dict(dat,Phi=Phi))
    fam=H.Family(3,'val64',count=64);refs=S.Refs(3,'val64',fam,ids)
    rep.update(cohort_sha256=fam.sha256,reference=dict(store_version=S.STORE_VERSION,status_sha256=S.STATUS_SHA,kind='ST',
             cases={i:dict(refs.st[i],n0=refs.n0(i)) for i in ids}),orthonormality=orth,trust_radius=trust,
             setup_seconds=time.perf_counter()-start)
    opt=('BDF2',1/60);sch=R.schedule(3,*opt,'prod')
    assert int(sch['steps'])==15 and int(sch['nstart'])==1
    assert np.array_equal(np.flatnonzero(np.asarray(sch['slot'])>=0),np.arange(0,16,3))
    queries={a:cell.query('none' if a=='none' else 'ecsw',512,'common') for a in arms}
    for a in arms:
        # Warm each exact numerical signature before any deployable query measurement.
        jax.block_until_ready(queries[a](jnp.asarray(fam.ic(64,0)),jnp.float64(fam.nu(0)),data[a],sch))
    metrics=X.make_metrics(64,512,15) if 'k32' in arms else None
    recs={a:{} for a in arms};tight={a:{} for a in arms};tighter={a:{} for a in arms}
    for i in ids:
        ck=f'val64|{i}';rep['cases'][str(i)]={}
        t_input=time.perf_counter();u0=jax.block_until_ready(jnp.asarray(fam.ic(64,i)));nu=jnp.float64(fam.nu(i))
        input_seconds=time.perf_counter()-t_input
        for a in arms:
            budget();kind='none' if a=='none' else 'ecsw';rr={}
            for level in ('prod','tight')+(('tighter',) if i in cfg['tighter_cases'] else ()):
                rr[level]=cell.run(kind,512,data[a],'common' if level=='prod' else 'native',opt,level,
                                   u0,nu,burn_s=cfg['burn_seconds'] if level=='prod' else 0.)
                q=rr[level]
                np.savez(out/'arrays'/f'{a}_c{i:03}_{level}.npz',case=i,W=q['W'],C=q['C'],gamma=q['gamma'],
                         reason=q['reason'],it=q['it'],lam=q['lam'],scale=q['scale'],dt=1/60,times=np.arange(6)*.05)
            r=rr['prod'];score=refs.score(r['fields_lat'],i);assert score['eligible']
            row=dict(e0=score,query_seconds=r['seconds'],W_sha256=X.array_hash(r['W']),cert=r['cert'],
                     input_prepare_seconds=input_seconds,input_sha256=X.array_hash(u0),
                     tight=rr['tight']['cert'],tight_distance=S.lattice_distance(r['fields_lat'],rr['tight']['fields_lat'],refs.n0(i)))
            if 'tighter' in rr:
                row.update(tighter=rr['tighter']['cert'],tight_tighter_distance=S.lattice_distance(rr['tight']['fields_lat'],rr['tighter']['fields_lat'],refs.n0(i)))
            if a=='k32':row['whole_residual']=X.metric_summary(jax.block_until_ready(metrics(jnp.asarray(r['C']),jnp.float64(fam.nu(i)),data['none'],data[a],sch)))
            rep['cases'][str(i)][a]=row
            recs[a][ck]=dict(cert=r['cert'],fields_lat=r['fields_lat'])
            tight[a][ck]=dict(cert=rr['tight']['cert'],fields_lat=rr['tight']['fields_lat'])
            if 'tighter' in rr:tighter[a][ck]=dict(cert=rr['tighter']['cert'],fields_lat=rr['tighter']['fields_lat'])
            del rr,r;save()
        if 'k32' in arms:
            rep['cases'][str(i)]['k32_to_none']=S.lattice_distance(recs['k32'][ck]['fields_lat'],recs['none'][ck]['fields_lat'],refs.n0(i))
        save();print('completed_case',i,flush=True)
    expected=[f'val64|{i}' for i in ids];subset=[f'val64|{i}' for i in cfg['tighter_cases']];n0={f'val64|{i}':refs.n0(i) for i in ids}
    rep['Q1']={a:X.q1(recs[a],tight[a],tighter[a],n0,subset,expected) for a in arms}
    if 'k32' in arms:
        mm=[rep['cases'][str(i)]['k32']['whole_residual'] for i in ids]
        dg=np.concatenate([m['delta_g'] for m in mm]);gg=np.concatenate([m['full_gamma'] for m in mm])
        finite=bool(all(all(m['finite']) for m in mm) and np.isfinite(dg).all() and np.isfinite(gg).all())
        dist=[rep['cases'][str(i)]['k32_to_none'] for i in ids]
        field_mask={str(i):bool(np.isfinite(rep['cases'][str(i)]['k32_to_none']) and
                                rep['cases'][str(i)]['k32_to_none']<=.00025) for i in ids}
        gradient_mask={str(i):bool(all(rep['cases'][str(i)]['k32']['whole_residual']['finite']) and
                                   np.isfinite(rep['cases'][str(i)]['k32']['whole_residual']['delta_g']).all() and
                                   max(rep['cases'][str(i)]['k32']['whole_residual']['delta_g'])<=.001) for i in ids}
        full_bad_masks={str(i):(np.asarray(rep['cases'][str(i)]['k32']['whole_residual']['full_gamma'])>.001).tolist() for i in ids}
        own_q1=bool(all(q['q1'] for q in rep['Q1'].values()))
        common_pass=bool(own_q1 and all(field_mask.values()))
        gradient_pass=bool(finite and all(gradient_mask.values()))
        full_stationarity_pass=bool(finite and (gg>.001).sum()<=.01*len(gg))
        rep['HR']=dict(delta_g_max=float(dg.max()),full_gamma_max=float(gg.max()),full_nonstat=int((gg>.001).sum()),
                       steps=len(gg),field_max=max(dist),finite=finite,
                       own_objective_Q1_pass=own_q1,
                       shared_field_checks=dict(bar=.00025,case_mask=field_mask,pass_=all(field_mask.values())),
                       common_layer_pass=common_pass,
                       extra_POD_gradient_checks=dict(bar=.001,case_mask=gradient_mask,pass_=gradient_pass),
                       extra_POD_full_stationarity=dict(bar=.001,pooled_bad_fraction_limit=.01,
                           per_case_bad_step_mask=full_bad_masks,pass_=full_stationarity_pass),
                       pass_=bool(common_pass and gradient_pass and full_stationarity_pass))
    # All repeated costs and errors belong to the same actual invocation; no cross-job ratio is formed.
    panels=[(i,r) for i in cfg['timing_cases'] for r in range(cfg['timing_reps'])]
    np.random.default_rng(20261026).shuffle(panels)
    for i,r in panels:
        budget();row=dict(case=i,rep=r,invocations=[])
        sequence=arms if len(arms)==1 else [arms[0],arms[1],arms[0]]
        input_start=time.perf_counter();u0=jax.block_until_ready(jnp.asarray(fam.ic(64,i)));nu=jnp.float64(fam.nu(i))
        row['input_prepare_seconds']=time.perf_counter()-input_start
        for a in sequence:
            P.burn(cfg['burn_seconds']);t=time.perf_counter()
            o=jax.block_until_ready(queries[a](u0,nu,data[a],sch))
            sec=time.perf_counter()-t;pin=np.load(out/'arrays'/f'{a}_c{i:03}_prod.npz')['W']
            identical=X.scalar_check(o['W'],pin)
            row['invocations'].append(dict(arm=a,seconds=sec,W_sha256=X.array_hash(o['W']),identity=identical,
                                            e0=refs.score(cell.lattice(o['fields']),i)))
            assert identical
        if len(sequence)>1:
            ts=[x['seconds'] for x in row['invocations']]
            row.update(drift=ts[0]/ts[2],ratio=ts[1]/((ts[0]+ts[2])/2),drift_accepted=abs(ts[0]/ts[2]-1)<=.25)
        rep['timing'].append(row);save()
    rep.update(complete=True,full_population=ids==list(range(64)),elapsed_seconds=time.perf_counter()-start,
               peak_device_bytes=P.device_peak(),peak_host_bytes=P.host_rss())
    save();print('validation_complete',rep['full_population'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--artifacts',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();run(json.loads(Path(a.config).read_text()),Path(a.artifacts),Path(a.out))
