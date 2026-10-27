"""Same-sample 2D reconstruction. Separate development/frozen-validation phases.

Timed methods accept only y. Scoring and reference-driven oracle fits live outside the
reconstruction call. Every repetition's actual returned field is scored and hashed.
"""
from __future__ import annotations
import argparse,gc,json,time,os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import adapter as A
import reconstruction_core as C
import data_manifest as DM
RANKS=(128,256,384,512)

def arrsave(path,**arrays):
    np.savez(path,**{k:C.host(v) for k,v in arrays.items()})
    return A.sha_file(path)

def make_plans(data,out,L):
    pdir=out/f'plans/L{L}'; pdir.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter(); plans=C.interpolation_plans(L,A.LF,2); oracle={}
    for p in plans:
        p.metadata.update(group='ordinary',representation='geometry_only',rank=None)
        path=pdir/(p.name+'.npz'); p.metadata['array_file']=str(path.relative_to(out)); p.metadata['array_sha256']=arrsave(path,**p.arrays)
    for bank in (['base','coarse'] if L==128 else ['base']):
        for family in ['bank','pod']:
            name=f'{family}_{bank}'
            bc=np.load(data/'bases'/f'{name}_L{L}.npy'); bf=np.load(data/'bases'/f'{name}_L{A.LF}.npy')
            prior=None
            if family=='pod':
                pi=json.loads((data/'bases'/f'pod_{bank}.json').read_text())
                prior=np.asarray(pi['captured_energy'])/16384
            for r in RANKS:
                start=time.perf_counter(); label=f'{name}_r{r}'
                geo=C.prepare_basis(bc[:,:r],bf[:,:r],L,A.LF,2,label)
                gpath=pdir/(label+'_geometry.npz')
                gh=arrsave(gpath,fine_basis=geo.fine,coarse_basis=geo.coarse,transform=geo.transform)
                oracle[label]=geo.fine
                for cov in ([False,True] if family=='pod' else [False]):
                    for lam in C.RIDGES:
                        if cov and lam==0: continue # mathematically identical to the unregularized POD row
                        plan=geo.plan(lam,prior_variances=prior[:r] if cov else None)
                        plan.metadata.update(group=name,representation=family,rank=r,training='256_node_fine_informed' if bank=='base' else '129_node_solution_data_only_inherited_recipe',strict_coarse_control=bool(bank=='coarse' and L==128 and r==384 and lam==0 and not cov),covariance_note='uncentered training second moments, no mean mode' if cov else None,geometry_file=str(gpath.relative_to(out)),geometry_sha256=gh)
                        path=pdir/(plan.name+'.npz')
                        plan.metadata['array_file']=str(path.relative_to(out)); plan.metadata['array_sha256']=arrsave(path,fit_matrix=plan.arrays['fit_matrix'])
                        plans.append(plan)
                A.log('prepared',L,label,'physicalrank',geo.metadata['physical_rank'],'seconds',time.perf_counter()-start)
            del bc,bf; gc.collect()
    meta=dict(L=L,prepare_seconds=time.perf_counter()-t0,plans=[p.metadata for p in plans])
    A.write_json(pdir/'plans.json',meta)
    return plans,oracle,meta

def run_case(plan,ys,levels,meta,idx,out,source,cohort,case,L):
    ref=levels[-1]; n0=meta['n0']; off=np.ones(ref.shape[-1],dtype=bool); off[idx]=False
    score=[]; coefficients=[]; repetition_fields=[]; timing=[]; initial={}
    # One uninterrupted method/case/source block: five single-field invocations,
    # with no host error norms, persistence or reference access between them.
    results=[C.benchmark(plan,ys[ti:ti+1],repetitions=3,burn_seconds=.1 if ti==1 else 0.) for ti in range(1,6)]
    for ti,result in enumerate(results,start=1):
        fields=np.stack(result['field_repetitions'])[:,0,:]
        coeff=np.stack(result['coefficient_repetitions'])[:,0,:]
        if not (np.isfinite(fields).all() and np.isfinite(coeff).all()): raise FloatingPointError('nonfinite timed fields or coefficients')
        errors=np.linalg.norm(fields-ref[ti],axis=1)/n0
        offerr=np.linalg.norm(fields[:,off]-ref[ti,off],axis=1)/n0
        cres=np.linalg.norm(fields[:,idx]-ys[ti],axis=1)/(n0*L/A.LF)
        timing.append(dict(time=float(ti*.05),compile_warm_seconds=result['compile_warm_seconds'],repeated_fields_identical=result['repeated_fields_identical'],repetitions=result['repetitions'],error_by_repetition=errors.tolist(),off_node_error_by_repetition=offerr.tolist(),coarse_sample_residual_by_repetition=cres.tolist()))
        score.append(float(errors[-1])); coefficients.append(coeff); repetition_fields.append(fields[-1])
    primary=np.stack(repetition_fields)
    # t0 is an untimed sanity diagnostic, never mixed into evolved-output headline errors.
    f0,c0=C.ready(plan.query(jnp.asarray(ys[:1]))); f0=C.host(f0); c0=C.host(c0)
    if not (np.isfinite(f0).all() and np.isfinite(c0).all()): raise FloatingPointError('nonfinite initial reconstruction')
    initial=dict(error=float(np.linalg.norm(f0[0]-ref[0])/n0),coarse_sample_residual=float(np.linalg.norm(f0[0,idx]-ys[0])/(n0*L/A.LF)),field_sha256=A.sha_array(f0))
    # Sensitivity is untimed and uses the very same frozen reconstruction map.
    alternatives=[]; alt_coefs=[]
    if source=='A':
        for li in [0,1]:
            alt_y=levels[li][:,idx]
            f,c=C.ready(plan.query(jnp.asarray(alt_y[1:]))); f=C.host(f); c=C.host(c)
            if not (np.isfinite(f).all() and np.isfinite(c).all()): raise FloatingPointError('nonfinite reference-level sensitivity')
            alternatives.append(dict(reference_level=li,output_change_vs_primary=(np.linalg.norm(f-primary,axis=1)/n0).tolist(),error_against_own_reference=(np.linalg.norm(f-levels[li,1:],axis=1)/n0).tolist(),input_change_physical=(np.linalg.norm(alt_y[1:]-ys[1:],axis=1)/(n0*L/A.LF)).tolist(),field_sha256=A.sha_array(f)))
            alt_coefs.append(c)
    else:
        alt_y=meta['_penultimate']
        f,c=C.ready(plan.query(jnp.asarray(alt_y[1:]))); f=C.host(f); c=C.host(c)
        if not (np.isfinite(f).all() and np.isfinite(c).all()): raise FloatingPointError('nonfinite time-level sensitivity')
        alternatives.append(dict(coarse_time_level='penultimate',output_change_vs_primary=(np.linalg.norm(f-primary,axis=1)/n0).tolist(),error_against_fixed_reference=(np.linalg.norm(f-ref[1:],axis=1)/n0).tolist(),input_change_physical=(np.linalg.norm(alt_y[1:]-ys[1:],axis=1)/(n0*L/A.LF)).tolist(),field_sha256=A.sha_array(f)))
        alt_coefs.append(c)
    target=out/f'cases/{cohort}/L{L}/{source}'; target.mkdir(parents=True,exist_ok=True)
    stem=target/f'c{case:03d}_{plan.name}'
    coefhash=arrsave(stem.with_suffix('.npz'),coefficient_repetitions=np.stack(coefficients),initial_coefficients=c0,alternative_coefficients=np.stack(alt_coefs))
    record=dict(status='complete',cohort=cohort,case=case,L=L,source=source,method=plan.name,group=plan.metadata['group'],rank=plan.metadata['rank'],ridge=plan.metadata.get('ridge'),strict_coarse_control=plan.metadata.get('strict_coarse_control',False),error_t=score,error_max=max(score),ref_eps=meta['ref_eps'],n0=n0,coarse_input_admitted=meta['accepted'],input_sha256=A.sha_array(ys),reference_sha256=A.sha_array(ref),timing=timing,initial=initial,sensitivity=alternatives,coefficient_file=str(stem.with_suffix('.npz').relative_to(out)),coefficient_file_sha256=coefhash,plan_arrays_file=plan.metadata['array_file'],plan_arrays_sha256=plan.metadata['array_sha256'],geometry_file=plan.metadata.get('geometry_file'),geometry_sha256=plan.metadata.get('geometry_sha256'),output_field_hashes=[v['repetitions'][-1]['field_sha256'] for v in timing])
    A.write_json(stem.with_suffix('.json'),record)
    return {k:record[k] for k in ['status','cohort','case','L','source','method','group','rank','ridge','strict_coarse_control','error_t','error_max','ref_eps','coarse_input_admitted']}|dict(record=str(stem.with_suffix('.json').relative_to(out)))

def failed_case(metadata,error,out,source,cohort,case,L,case_meta):
    """A failed declared method remains in coverage and cannot enter selection."""
    rec=dict(status='failed',reason=type(error).__name__+': '+str(error),cohort=cohort,case=case,L=L,source=source,method=metadata['name'],group=metadata['group'],rank=metadata.get('rank'),ridge=metadata.get('ridge'),strict_coarse_control=metadata.get('strict_coarse_control',False),error_t=None,error_max=None,ref_eps=case_meta['ref_eps'],coarse_input_admitted=case_meta['accepted'])
    target=out/f'cases/{cohort}/L{L}/{source}'; target.mkdir(parents=True,exist_ok=True)
    path=target/f"c{case:03d}_{metadata['name']}.json"; A.write_json(path,rec)
    A.log('failed candidate',L,source,cohort,case,metadata['name'],rec['reason'])
    return rec|dict(record=str(path.relative_to(out)))

def freeze_choices(records,source_hashes):
    """Frozen selector uses development only and retains the complete declared population."""
    if {r['cohort'] for r in records}!={'dev6'}: raise ValueError('selector accepts only dev6')
    choices=[]
    for L in [32,64,128]:
        for source in ['A','B']:
            groups=sorted({r['group'] for r in records if r['L']==L and r['source']==source})
            for group in groups:
                rows=[r for r in records if (r['L'],r['source'],r['group'])==(L,source,group)]
                methods=sorted({r['method'] for r in rows}); candidates=[]
                for m in methods:
                    rr=sorted([r for r in rows if r['method']==m],key=lambda r:r['case'])
                    if [r['case'] for r in rr]!=list(range(6)): raise ValueError('incomplete dev population '+m)
                    if any(r.get('status')!='complete' or r['error_max'] is None or not np.isfinite(r['error_max']) for r in rr): continue
                    e=np.asarray([r['error_max'] for r in rr]); margin=2*max(r['ref_eps'] for r in rr) if source=='B' else 1e-14
                    candidates.append(dict(method=m,median=float(np.median(e)),worst=float(e.max()),rank=rr[0]['rank'],ridge=rr[0]['ridge'],margin=margin))
                if not candidates:
                    choices.append(dict(L=L,source=source,group=group,selected=None,candidates=[],status='no_complete_finite_candidate',selection_information='fine_development_selected')); continue
                best=min(c['median'] for c in candidates)
                eligible=[c for c in candidates if c['median']<=best+c['margin']]
                # Exact worst error then representation size then ridge then identifier; no validation input.
                pick=min(eligible,key=lambda c:(c['worst'],c['rank'] or 0,c['ridge'] or 0,c['method']))
                choices.append(dict(L=L,source=source,group=group,selected=pick,candidates=candidates,selection_information='fine_development_selected',strict_control_method=(f'{group}_r384_ridge0' if group.endswith('_coarse') else None)))
    return dict(version=1,selection_cohort='dev6',primary=dict(L=32,source='B'),choices=choices,source_hashes=source_hashes,strict_coarse_control=dict(L=128,rank=384,ridge=0,svd_rtol=C.SVD_RTOL,methods=['bank_coarse_r384_ridge0','pod_coarse_r384_ridge0']),selector='min median; B within 2max eps allowed, A ties1e-14; then min worst,rank,ridge,name; exposed developmental')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['dev','val'],required=True); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True); ap.add_argument('--frozen'); args=ap.parse_args()
    out=Path(args.out); data=Path(args.data)
    if out.exists() and any(out.iterdir()): raise RuntimeError('output directory must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    A.verify_inputs(); runtime=A.runtime(); A.write_json(out/'runtime.json',runtime)
    hashes={fn:A.sha_file(A.HERE/fn) for fn in ['screen.py','adapter.py','reconstruction_core.py','data_manifest.py','INPUTS.json','CORE_SOURCE.json']}
    hashes['consumed_data_manifest']=DM.verify(data)
    if hashes['consumed_data_manifest']!=os.environ.get('EXPECTED_PREP_MANIFEST_SHA256'): raise RuntimeError('unexpected prepared-data manifest')
    hashes['bases_manifest']=A.sha_file(data/'bases/bases.json'); hashes['coarse_manifest']=A.sha_file(data/'coarse/summary.json')
    if args.phase=='val':
        if not args.frozen: raise ValueError('validation requires frozen development choices')
        frozen=json.loads(Path(args.frozen).read_text())
        for key,val in frozen['source_hashes'].items():
            if hashes.get(key)!=val: raise RuntimeError('frozen source/input mismatch '+key)
        A.write_json(out/'FROZEN.json',frozen); hashes['frozen_file']=A.sha_file(args.frozen)
    A.write_json(out/'SOURCES.json',hashes)
    cohort='dev6' if args.phase=='dev' else 'val32'; summary=[]; setup=[]
    for L in [32,64,128]:
        plans,oracles,meta=make_plans(data,out,L); setup.append(meta)
        idx=A.indices(L)
        for case in range(A.COHORTS[cohort]):
            stem=data/'coarse'/f'{cohort}_c{case:03d}_L{L}'
            md=json.loads(stem.with_suffix('.json').read_text())
            with np.load(stem.with_suffix('.npz')) as z:
                levels=np.asarray(z['reference_levels']); coarse=np.asarray(z['fields'])
            if not (np.isfinite(levels).all() and np.isfinite(coarse).all() and np.isfinite(md['n0']) and md['n0']>0 and np.isfinite(md['ref_eps']) and md['ref_eps']>=0): raise RuntimeError('invalid prepared input/reference metadata')
            md['_penultimate']=coarse[-2]
            # A-level input and B temporal input sensitivities are retained independently.
            inputs={'A':levels[-1][:,idx],'B':coarse[-1]}
            # Sustained GPU burn before the case; later method blocks have their own burn.
            burn_y=jnp.asarray(inputs['B'][1:2]); deadline=time.perf_counter()+1.
            while time.perf_counter()<deadline: C.ready(plans[-1].query(burn_y))
            order=np.random.default_rng(7201+case+L).permutation(len(plans))
            for ii in order:
                plan=plans[int(ii)]
                for source in (['A','B'] if case%2==0 else ['B','A']):
                    try:
                        summary.append(run_case(plan,inputs[source],levels,md,idx,out,source,cohort,case,L))
                    except Exception as exc:
                        summary.append(failed_case(plan.metadata,exc,out,source,cohort,case,L,md))
            oracle_rec={}
            for label,qf in oracles.items():
                c=C.ready(jnp.asarray(levels[-1,1:])@qf*(A.LF**-2)); f=C.host(C.ready(c@qf.T))
                oracle_rec[label]=dict(error_t=(np.linalg.norm(f-levels[-1,1:],axis=1)/md['n0']).tolist(),reference_access='fine oracle only, never a reconstruction input')
            op=out/f'oracle/L{L}'; op.mkdir(parents=True,exist_ok=True); A.write_json(op/f'{cohort}_c{case:03d}.json',oracle_rec)
            A.write_json(out/'summary.json',summary)
            A.log('completed',args.phase,L,case,'records',len(summary))
        del plans,oracles; jax.clear_caches(); gc.collect()
    A.write_json(out/'setup.json',setup)
    if args.phase=='dev': A.write_json(out/'FROZEN.json',freeze_choices(summary,hashes))
    A.write_json(out/'COMPLETE.json',dict(phase=args.phase,cases=A.COHORTS[cohort],records=len(summary),source_hashes=hashes,runtime=runtime))
if __name__=='__main__': main()
