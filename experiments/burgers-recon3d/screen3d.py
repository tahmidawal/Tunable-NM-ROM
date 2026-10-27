"""Frozen coarse-sample reconstruction; references are visible only to scorers.

Development and validation are separate allocations. Validation requires the
committed development selection file and exact immutable preparation manifest.
No PDE solver or case descriptor is passed to any reconstruction method.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from prepare3d import HERE, LF, RANKS, RS, dump, digest, guard, log
sys.path.insert(0, str(HERE/'vendor'))
import reconstruction_core as K

SEED = 2026102704
PREFIXES = [256, 512, 1024]
TIMES = np.arange(6)*.05


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(16*1024*1024), b''): h.update(b)
    return h.hexdigest()


def source_contract():
    paths=['screen3d.py','export3d.py','prepare3d.py','vendor/PROVENANCE.json','configs/frozen_inputs.json']
    paths+=['vendor/'+p for p in json.loads((HERE/'vendor/PROVENANCE.json').read_text())]
    return {p:sha(HERE/p) for p in sorted(paths)}


def verify_prepared(prepared, expected):
    manifest = prepared.parent/'OUTPUTS.sha256'
    assert sha(manifest) == expected['manifest_sha256'], 'preparation manifest changed'
    for line in manifest.read_text().splitlines():
        checksum, rel = line.split(None, 1)
        assert sha(prepared.parent/rel.strip()) == checksum, rel
    meta = json.loads((prepared/'COMPLETE.json').read_text())
    assert meta['complete'] and not meta['smoke']
    assert meta['execution']['commit'] == expected['source_commit']
    assert RS.store_version() == json.loads((HERE/'configs/frozen_inputs.json').read_text())['reference_store_sha256']
    vendor = json.loads((HERE/'vendor/PROVENANCE.json').read_text())
    for rel, info in vendor.items(): assert sha(HERE/'vendor'/rel) == info['sha256'], rel
    return meta


def build_plans(prepared, out, lc, phase):
    """Cache geometry and factors. All setup times are reported separately."""
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    plans = {p.name:p for p in K.interpolation_plans(lc, LF, 3)}
    descriptors = {n:dict(category='ordinary', rank=0, power=2, prior='geometry', ridge=0.) for n in plans}
    for name,plan in plans.items():
        arrays={key:K.host(value) for key,value in plan.arrays.items()}
        np.savez(out/f'{name}_interpolation.npz',**arrays)
        plan.metadata['replay_arrays_sha256']={key:digest(value) for key,value in arrays.items()}
    geometries, failures, setup = {}, {}, {}
    ix = K.restriction_indices(lc, LF, 3)
    specs = [('bank',2,r) for r in PREFIXES]
    specs += [(n,q,r) for n in ('pod_matched','pod_stronger')
              for q in ([1,2] if phase=='dev' else [2]) for r in RANKS]
    for category, power, rank in specs:
        label = f'{category}_q{power}_r{rank}'
        started = time.perf_counter()
        original_path = prepared/('bank_L64.npy' if category=='bank' else f'{category}_q{power}_basis.npy')
        bf = np.load(original_path, mmap_mode='r')[:,:rank]
        bc = (np.load(prepared/f'bank_L{lc}.npy', mmap_mode='r')[:,:rank]
              if category=='bank' else np.asarray(bf[ix]))
        intended=[]
        for prior in ('field',) if category=='bank' else ('field','diagonal_second_moment'):
            for ridge in K.RIDGES:
                if prior!='field' and ridge==0: continue
                name=f'{label}_'+('ridge' if prior=='field' else 'covridge')+f'{ridge:g}'
                intended.append((name,dict(category=category,rank=rank,power=power,prior=prior,ridge=ridge,geometry=label)))
        try:
            geom = K.prepare_basis(bc, bf, lc, LF, 3, label)
        except (ValueError,np.linalg.LinAlgError) as err:
            for name,d in intended: failures[name]=dict(descriptor=d,failure=str(err),admissible=False)
            continue
        geometries[label] = geom
        setup[label] = dict(geometry_seconds=time.perf_counter()-started, plans={},
                           original_basis=str(original_path.relative_to(prepared)),
                           original_basis_sha256=sha(original_path), **geom.metadata)
        # Actual whitened fine basis is common to every ridge in this prefix.
        # Its hash plus original basis and transform permit numerical replay;
        # output hashes refer to the actual timed floating-point arrays.
        np.savez(out/f'{label}_geometry.npz', transform=K.host(geom.transform),
                 singular=K.host(geom.singular), right=K.host(geom.right))
        setup[label]['whitened_fine_basis_array_sha256'] = digest(K.host(geom.fine))
        var = None if category=='bank' else np.load(prepared/f'{category}_q{power}_variances.npy')[:rank]
        for prior in ('field',) if category=='bank' else ('field','diagonal_second_moment'):
            for ridge in K.RIDGES:
                if prior!='field' and ridge==0: continue
                name = f'{label}_'+('ridge' if prior=='field' else 'covridge')+f'{ridge:g}'
                started = time.perf_counter()
                desc = dict(category=category, rank=rank, power=power, prior=prior, ridge=ridge, geometry=label)
                try:
                    plan = geom.plan(ridge, None if prior=='field' else var)
                except (ValueError, np.linalg.LinAlgError) as err:
                    failures[name] = dict(descriptor=desc, failure=str(err), admissible=False)
                    continue
                assert plan.name==name
                plans[name], descriptors[name] = plan, desc
                setup[label]['plans'][name] = dict(seconds=time.perf_counter()-started, **plan.metadata)
        log(f'prepared L{lc} {label} elapsed={time.perf_counter()-started:.2f}')
        del bf, bc
    # Warm every executable before the balanced measurement blocks.
    for name, plan in list(plans.items()):
        started=time.perf_counter()
        try:
            values=K.ready(plan.query(jnp.zeros((1,(lc-1)**3),jnp.float64)))
            if not all(np.isfinite(K.host(v)).all() for v in values): raise ValueError('nonfinite zero-input warmup')
            plan.metadata['initial_compile_warm_seconds']=time.perf_counter()-started
        except (ValueError,np.linalg.LinAlgError) as err:
            failures[name]=dict(descriptor=descriptors[name],failure=str(err),admissible=False)
            del plans[name]
    dump(out/'setup.json', dict(total_seconds=time.perf_counter()-t0, geometries=setup,
         plans={n:dict(descriptor=descriptors[n],**p.metadata) for n,p in plans.items()}, failures=failures,
         covariance_note='Diagonal actual training column second moments after QR; approximate POD need not diagonalize full training covariance.'))
    return plans, descriptors, geometries, failures


def metrics(field, ref, coarse, ix, n0, lc):
    error = field-ref
    total2 = float(np.dot(error,error))
    on2 = float(np.dot(error[ix],error[ix]))
    values=dict(error=float(np.sqrt(max(0.,total2))/n0),
                coincident_error=float(np.sqrt(max(0.,on2))/n0),
                off_grid_error=float(np.sqrt(max(0.,total2-on2))/n0),
                physical_coarse_residual=float(np.linalg.norm(field[ix]-coarse)*(LF/lc)**1.5/n0))
    if not all(np.isfinite(v) for v in values.values()): raise ValueError('nonfinite field scoring metric')
    return values


def failed_time(row, coefficients, ti, reason):
    row.setdefault('failures',[]).append(dict(time_index=ti,message=reason))
    row['e_t'].append(None);row['scores'].append([]);row['repetitions'].append([])
    row['field_sha256'].append([]);row['coefficient_sha256'].append([])
    row['repetition_deduplicated'].append(False);row['sensitivity'].append([])
    coefficients.append(np.empty((0,0)))


def warm_block(plan, sample):
    deadline=time.perf_counter()+1.
    y=jnp.asarray(sample[None,:])
    while time.perf_counter()<deadline: K.ready(plan.query(y))


def one_query(plan, sample, timed):
    if timed:
        result=K.benchmark(plan,sample[None,:],repetitions=3,burn_seconds=0.)
        fields=[f[0] for f in result['field_repetitions']]
        coeff=[c[0] for c in result['coefficient_repetitions']]
        timing=result['repetitions']
    else:
        field,c=K.ready(plan.query(jnp.asarray(sample[None,:])))
        fields,coeff=[K.host(field)[0]],[K.host(c)[0]]
        timing=[]
        if not all(np.isfinite(a).all() for a in fields+coeff):
            raise ValueError('nonfinite reconstruction or coefficients')
    # Exact deduplication is allowed only after checking every repeated byte.
    same=all(np.array_equal(fields[0],f) and np.array_equal(coeff[0],c) for f,c in zip(fields,coeff))
    return fields,coeff,timing,same


def oracle_case(geometries, ref, n0, out):
    """Fine target projection is an explicitly inadmissible diagnostic."""
    rows,coeffs={},{}
    target=jnp.asarray(ref)
    for name,g in geometries.items():
        c=target@g.fine*LF**-3
        pred=K.host(c@g.fine.T)
        cc=K.host(c)
        coeffs[name]=cc
        rows[name]=dict(e_t=(np.linalg.norm(pred-ref,axis=1)/n0).tolist(),
                        predicted_field_sha256=digest(pred),coefficient_sha256=digest(cc),
                        admissible_reconstruction=False,information='full fine reference target')
    np.savez(out.with_suffix('.npz'),**coeffs)
    dump(out.with_suffix('.json'),rows)


def run_cohort(prepared,out,lc,phase,plans,desc,geometries,failures,selected=None):
    cohort,count=('dev4',4) if phase=='dev' else ('val64',64)
    ix=K.restriction_indices(lc,LF,3)
    cohort_rows=[]
    primary_selected=set() if selected is None else set(selected.values())
    burn_plan=plans.get('bank_q2_r1024_ridge0',plans['sine_full'])
    for case in range(count):
        levels,rm=RS.load(3,cohort,case,'ST',field='levels')
        ref=levels[-1]; n0=rm['judge']['n0']; eps=rm['judge']['eps_ref']
        assert RS.reference_status(3,cohort,case,'ST')['eligible']
        assert np.isfinite(n0) and n0>0 and np.isfinite(eps) and eps>=0
        assert all(a.shape==(6,(LF-1)**3) and np.isfinite(a).all() for a in levels)
        inputs=prepared/'inputs'/cohort/f'c{case:03d}'
        cm=json.loads((inputs/'meta.json').read_text())['meshes'][str(lc)]
        coarse=np.load(inputs/f'coarse_L{lc}_accepted.npz')
        assert all(coarse[k].shape==(6,(lc-1)**3) and np.isfinite(coarse[k]).all() for k in ('fields','previous'))
        case_dir=out/cohort/f'c{case:03d}';case_dir.mkdir(parents=True,exist_ok=True)
        np.savez(case_dir/'input_fields.npz',reference=ref,reference_level0=levels[0],
                 reference_level1=levels[1],A=ref[:,ix],B=coarse['fields'],B_previous=coarse['previous'])
        oracle_case(geometries,ref,n0,case_dir/'oracle')
        for source,y in [('A',ref[:,ix]),('B',coarse['fields'])]:
            records={name:dict(method=name,descriptor=desc[name],source=source,Lc=lc,case=case,
                       cohort=cohort,eps_ref=eps,input_eligible=(source=='A' or cm['eligible']),
                       e_t=[],scores=[],repetitions=[],field_sha256=[],coefficient_sha256=[],
                       repetition_deduplicated=[],sensitivity=[],approximation_gap=[])
                     for name in plans}
            coeffs={name:[] for name in plans}
            diagnostic_coefficients={}
            # A contiguous GPU block retains every actual output before scoring.
            # Approximately three GB of host memory is sufficient for this block.
            # Only the common query transfer/hash audit occurs between calls.
            warm_block(burn_plan,y[1])
            actual={}
            for ti in range(6):
                order=list(plans)
                rng=np.random.default_rng(SEED+100000*lc+1000*case+100*ti+(source=='B'))
                rng.shuffle(order)
                for name in order:
                    d=desc[name];plan=plans[name]
                    # Broad curves retain actual single-field outputs. Timed rows
                    # are selected primary methods, every unregularized prefix,
                    # and all ordinary methods. Other ridges are accuracy-only.
                    timed=(phase=='val' and ti>0 and (name in primary_selected or d['category']=='ordinary'
                           or (d['power']==2 and d['prior']=='field' and d['ridge']==0)))
                    try:
                        actual[ti,name]=one_query(plan,y[ti],timed)
                    except (ValueError,np.linalg.LinAlgError) as err:
                        actual[ti,name]=str(err)
            # References and scores are used only after the entire query block.
            for ti in range(6):
                for name,plan in plans.items():
                    d=desc[name]
                    row=records[name]
                    result=actual[ti,name]
                    if isinstance(result,str):
                        failed_time(row,coeffs[name],ti,result)
                        continue
                    fs,cs,timing,same=result
                    try: scored=[metrics(f,ref[ti],y[ti],ix,n0,lc) for f in fs]
                    except ValueError as err:
                        failed_time(row,coeffs[name],ti,str(err));actual[ti,name]=str(err)
                        continue
                    row['e_t'].append(scored[-1]['error']);row['scores'].append(scored)
                    row['repetitions'].append(timing)
                    row['field_sha256'].append([digest(f) for f in fs])
                    row['coefficient_sha256'].append([digest(c) for c in cs])
                    row['repetition_deduplicated'].append(same)
                    coeffs[name].append(np.stack(cs[:1] if same else cs))
                    # Diagnostics cannot influence the deployed map. Each input
                    # perturbation uses exactly the same prepared coefficients.
                    if phase=='dev' or name in primary_selected or (d['power']==2 and d['ridge']==0):
                        alternatives=[a[ti,ix] for a in levels[:2]] if source=='A' else [coarse['previous'][ti]]
                        sens=[]
                        for k,alt in enumerate(alternatives):
                            altf,altc=K.ready(plan.query(jnp.asarray(alt[None,:])))
                            altf=K.host(altf)[0];altc=K.host(altc)
                            diagnostic_coefficients[f'{name}__t{ti}__level{k}']=altc
                            if not np.isfinite(altf).all() or not np.isfinite(altc).all():
                                sens.append(dict(status='failed-nonfinite',prediction_gap=None,error_vector_gap=None,
                                    error_against_finest=None,error_against_own_reference=None,input_physical_gap=None))
                                continue
                            er=(altf-levels[k][ti])-(fs[-1]-ref[ti]) if source=='A' else altf-fs[-1]
                            values=dict(prediction_gap=float(np.linalg.norm(altf-fs[-1])/n0),
                                error_vector_gap=float(np.linalg.norm(er)/n0),
                                error_against_finest=float(np.linalg.norm(altf-ref[ti])/n0),
                                error_against_own_reference=float(np.linalg.norm(altf-(levels[k][ti] if source=='A' else ref[ti]))/n0),
                                input_physical_gap=float(np.linalg.norm(alt-y[ti])*(LF/lc)**1.5/n0))
                            good=all(np.isfinite(v) for v in values.values())
                            sens.append(dict(status='complete' if good else 'failed-nonfinite-metric',
                                **{key:value if good else None for key,value in values.items()},
                                field_sha256=digest(altf),coefficient_sha256=digest(altc)))
                        row['sensitivity'].append(sens)
                    else: row['sensitivity'].append([])
                if phase=='dev':
                    for name,d in desc.items():
                        if d['power']!=2 or not d['category'].startswith('pod_') or name not in plans: continue
                        counterpart=name.replace('_q2_','_q1_')
                        pair=[actual.get((ti,n)) for n in (name,counterpart)]
                        available=all(isinstance(p,tuple) for p in pair)
                        gap=float(np.linalg.norm(pair[0][0][-1]-pair[1][0][-1])/n0) if available else None
                        available=available and np.isfinite(gap)
                        records[name]['approximation_gap'].append(dict(time_index=ti,
                            status='paired' if available else 'unavailable',gap=gap if available else None,
                            consequential=bool(gap>2*eps or gap>1e-5) if available else None))
            packed={}
            for name,row in records.items():
                row['e']=max(row['e_t'][1:]) if all(e is not None for e in row['e_t'][1:]) else None
                row['status']='complete' if all(e is not None for e in row['e_t']) else 'query-failure'
                row['initial_error']=row['e_t'][0]
                row['timing_policy']='three repetitions of each one-field query' if any(row['repetitions']) else 'accuracy-only'
                # Repetition count can differ only if actual fields differed.
                for ti,cc in enumerate(coeffs[name]): packed[f'{name}__t{ti}']=cc
                cohort_rows.append(row)
            for name,failure in failures.items():
                row=dict(method=name,descriptor=failure['descriptor'],source=source,Lc=lc,case=case,
                    cohort=cohort,eps_ref=eps,input_eligible=(source=='A' or cm['eligible']),
                    e=None,e_t=[None]*6,initial_error=None,status='setup-failure',
                    failures=[failure['failure']],scores=[],repetitions=[],sensitivity=[],approximation_gap=[],
                    timing_policy='unavailable: failed numerical setup')
                records[name]=row;cohort_rows.append(row)
            np.savez(case_dir/f'{source}_coefficients.npz',**packed)
            np.savez(case_dir/f'{source}_sensitivity_coefficients.npz',**diagnostic_coefficients)
            dump(case_dir/f'{source}_scores.json',list(records.values()))
            log(f'scored {cohort}/{case} L{lc} {source} methods={len(plans)} eligible={source=="A" or cm["eligible"]}')
            del actual
        del levels,ref,coarse
    return cohort_rows


def select(rows):
    assert rows and all(r['cohort']=='dev4' for r in rows), 'selection accepts development cohort only'
    choices,details={},{}
    for lc in (16,32):
        for source in ('A','B'):
            for cat in ('ordinary','bank','pod_matched','pod_stronger'):
                pool=[r for r in rows if r['Lc']==lc and r['source']==source and r['descriptor']['category']==cat
                      and r['descriptor']['power']==2]
                groups={}
                for r in pool: groups.setdefault(r['method'],[]).append(r)
                stats=[]
                for method,rr in groups.items():
                    assert len(rr)==4 and sorted(r['case'] for r in rr)==list(range(4)), (lc,source,method,'missing/duplicate development cases')
                    if not all(r.get('status')=='complete' and r['e'] is not None and np.isfinite(r['e']) for r in rr): continue
                    vals=np.array([r['e'] for r in rr]);d=rr[0]['descriptor']
                    stats.append(dict(method=method,median=float(np.median(vals)),worst=float(max(vals)),
                                      rank=d['rank'],ridge=d['ridge'],prior=d['prior'],
                                      qualified_spatial_input=all(r['input_eligible'] for r in rr)))
                key=f'{lc}/{source}/{cat}'
                if not stats:
                    choices[key]=None;details[key]=dict(status='no-finite-complete-candidate',all_candidates=[])
                    continue
                delta=1e-14 if source=='A' else 2*max(r['eps_ref'] for r in pool)
                best=min(s['median'] for s in stats)
                tied=[s for s in stats if s['median']<=best+delta]
                winner=min(tied,key=lambda s:(s['worst'],s['rank'],s['ridge'],s['prior'],s['method']))
                choices[key]=winner['method']
                details[key]=dict(status='qualified' if winner['qualified_spatial_input'] else 'descriptive-input-not-admitted',
                                 winner=winner,tie_width=delta,all_candidates=stats)
    return choices,details


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--phase',choices=['dev','val'],required=True)
    ap.add_argument('--prepared-contract',default=str(HERE/'configs/prepared.json'))
    ap.add_argument('--selection',default=str(HERE/'configs/selection.json'))
    args=ap.parse_args();guard()
    prepared=Path(args.prepared);out=Path(args.out)
    assert not out.exists() or not any(out.iterdir()), 'output is not fresh'
    out.mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path(args.prepared_contract).read_text())
    prepmeta=verify_prepared(prepared,contract)
    selection=None
    if args.phase=='val':
        selection=json.loads(Path(args.selection).read_text())
        assert selection['prepared_contract']==contract
        assert selection['reference_store']==RS.store_version()
        assert selection['cohort']=='dev4' and selection['frozen_before_validation']
        assert selection['source_contract']==source_contract(), 'reconstruction source differs from development freeze'
        (out/'selection.json').write_bytes(Path(args.selection).read_bytes())
    execution=dict(job_id=os.environ.get('SLURM_JOB_ID'),commit=os.environ.get('SOURCE_COMMIT'),
        gpu=jax.devices()[0].device_kind,jax=jax.__version__,f64=True,precision='highest',phase=args.phase,
        prepared=prepmeta,prepared_contract=contract,seed=SEED,times=TIMES.tolist(),
        selection_sha256=None if selection is None else sha(args.selection),cost_note='Only within this allocation; cached geometry/compile separate from query.')
    execution['source_contract']=source_contract()
    dump(out/'execution.json',execution)
    allrows=[]
    for lc in (16,32):
        sub=out/f'L{lc}';sub.mkdir(exist_ok=True)
        plans,desc,geoms,failures=build_plans(prepared,sub/'plans',lc,args.phase)
        choices=None if selection is None else {k:v for k,v in selection['selected'].items() if k.startswith(f'{lc}/')}
        allrows+=run_cohort(prepared,sub,lc,args.phase,plans,desc,geoms,failures,choices)
        del plans,desc,geoms;gc.collect();jax.clear_caches()
    dump(out/'all_scores.json',allrows)
    if args.phase=='dev':
        choices,details=select(allrows)
        dump(out/'selection.json',dict(cohort='dev4',frozen_before_validation=True,
             selected=choices,details=details,prepared_contract=contract,reference_store=RS.store_version(),
             dev_commit=execution['commit'],dev_job=execution['job_id'],
             source_contract=source_contract(),dev_scores_sha256=sha(out/'all_scores.json'),
             policy='All complete finite dev4 cases; minimum median case-maximum. A fixed-array tie width 1e-14; B tie width twice maximum reference eps. Within tie choose worst-case, rank, ridge, prior, lexical method. Input admission qualifies interpretation, never deletes cases.'))
    dump(out/'COMPLETE.json',dict(complete=True,execution=execution,rows=len(allrows)))


if __name__=='__main__': main()
