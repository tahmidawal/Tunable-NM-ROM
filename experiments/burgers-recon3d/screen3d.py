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
    for name, plan in plans.items():
        started=time.perf_counter(); K.ready(plan.query(jnp.zeros((1,(lc-1)**3),jnp.float64)))
        plan.metadata['initial_compile_warm_seconds']=time.perf_counter()-started
    dump(out/'setup.json', dict(total_seconds=time.perf_counter()-t0, geometries=setup,
         plans={n:dict(descriptor=descriptors[n],**p.metadata) for n,p in plans.items()}, failures=failures,
         covariance_note='Diagonal actual training column second moments after QR; approximate POD need not diagonalize full training covariance.'))
    return plans, descriptors, geometries, failures


def metrics(field, ref, coarse, ix, n0, lc):
    error = field-ref
    total2 = float(np.dot(error,error))
    on2 = float(np.dot(error[ix],error[ix]))
    return dict(error=float(np.sqrt(max(0.,total2))/n0),
                coincident_error=float(np.sqrt(max(0.,on2))/n0),
                off_grid_error=float(np.sqrt(max(0.,total2-on2))/n0),
                physical_coarse_residual=float(np.linalg.norm(field[ix]-coarse)*(LF/lc)**1.5/n0))


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
        inputs=prepared/'inputs'/cohort/f'c{case:03d}'
        cm=json.loads((inputs/'meta.json').read_text())['meshes'][str(lc)]
        coarse=np.load(inputs/f'coarse_L{lc}_accepted.npz')
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
                pair_predictions={}
                for name,plan in plans.items():
                    d=desc[name]
                    row=records[name]
                    result=actual[ti,name]
                    if isinstance(result,str):
                        row.setdefault('failures',[]).append(dict(time_index=ti,message=result))
                        row['e_t'].append(None);row['scores'].append([]);row['repetitions'].append([])
                        row['field_sha256'].append([]);row['coefficient_sha256'].append([])
                        row['repetition_deduplicated'].append(False);row['sensitivity'].append([])
                        coeffs[name].append(np.empty((0,0)))
                        continue
                    fs,cs,timing,same=result
                    scored=[metrics(f,ref[ti],y[ti],ix,n0,lc) for f in fs]
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
                            altf=K.host(altf)[0]
                            er=(altf-levels[k][ti])-(fs[-1]-ref[ti]) if source=='A' else altf-fs[-1]
                            sens.append(dict(prediction_gap=float(np.linalg.norm(altf-fs[-1])/n0),
                                error_vector_gap=float(np.linalg.norm(er)/n0),
                                error_against_finest=float(np.linalg.norm(altf-ref[ti])/n0),
                                error_against_own_reference=float(np.linalg.norm(altf-(levels[k][ti] if source=='A' else ref[ti]))/n0),
                                input_physical_gap=float(np.linalg.norm(alt-y[ti])*(LF/lc)**1.5/n0),
                                field_sha256=digest(altf),coefficient_sha256=digest(K.host(altc))))
                        row['sensitivity'].append(sens)
                    else: row['sensitivity'].append([])
                    counterpart=name.replace('_q1_','_q2_') if d['power']==1 else name.replace('_q2_','_q1_')
                    if (phase=='dev' and d['category'].startswith('pod_') and counterpart in plans
                            and not isinstance(actual[ti,counterpart],str)):
                        key=name.replace('_q1_','_q2_')
                        if key in pair_predictions:
                            f0=pair_predictions.pop(key)
                            gap=float(np.linalg.norm(fs[-1]-f0)/n0)
                            row['approximation_gap'].append(dict(time_index=ti,gap=gap,
                                consequential=bool(gap>max(2*eps,1e-5))))
                        else: pair_predictions[key]=fs[-1]
                assert not pair_predictions, 'unpaired power-iteration comparator'
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
            dump(case_dir/f'{source}_scores.json',list(records.values()))
            log(f'scored {cohort}/{case} L{lc} {source} methods={len(plans)} eligible={source=="A" or cm["eligible"]}')
            del actual
        del levels,ref,coarse
    return cohort_rows


def select(rows):
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
                    if len(rr)!=4 or not all(r['input_eligible'] and r.get('status')=='complete' and r['e'] is not None for r in rr): continue
                    vals=np.array([r['e'] for r in rr]);d=rr[0]['descriptor']
                    stats.append(dict(method=method,median=float(np.median(vals)),worst=float(max(vals)),
                                      rank=d['rank'],ridge=d['ridge'],prior=d['prior']))
                assert stats, (lc,source,cat,'no complete eligible development candidate')
                delta=2*float(np.median([r['eps_ref'] for r in next(iter(groups.values()))]))
                best=min(s['median'] for s in stats)
                tied=[s for s in stats if s['median']<=best+delta]
                winner=min(tied,key=lambda s:(s['worst'],s['rank'],s['ridge'],s['prior'],s['method']))
                key=f'{lc}/{source}/{cat}'
                choices[key]=winner['method']
                details[key]=dict(winner=winner,tie_width=delta,all_candidates=stats)
    return choices,details


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--phase',choices=['dev','val'],required=True)
    ap.add_argument('--prepared-contract',default=str(HERE/'configs/prepared.json'))
    ap.add_argument('--selection',default=str(HERE/'configs/selection.json'))
    args=ap.parse_args();guard()
    prepared=Path(args.prepared);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path(args.prepared_contract).read_text())
    prepmeta=verify_prepared(prepared,contract)
    selection=None
    if args.phase=='val':
        selection=json.loads(Path(args.selection).read_text())
        assert selection['prepared_contract']==contract
        assert selection['reference_store']==RS.store_version()
        assert selection['cohort']=='dev4' and selection['frozen_before_validation']
    execution=dict(job_id=os.environ.get('SLURM_JOB_ID'),commit=os.environ.get('SOURCE_COMMIT'),
        gpu=jax.devices()[0].device_kind,jax=jax.__version__,f64=True,precision='highest',phase=args.phase,
        prepared=prepmeta,prepared_contract=contract,seed=SEED,times=TIMES.tolist(),
        selection_sha256=None if selection is None else sha(args.selection),cost_note='Only within this allocation; cached geometry/compile separate from query.')
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
             dev_scores_sha256=sha(out/'all_scores.json'),policy='minimum median case-maximum; within two median reference eps choose smallest worst-case, rank, ridge, prior and lexical method'))
    dump(out/'COMPLETE.json',dict(complete=True,execution=execution,rows=len(allrows)))


if __name__=='__main__': main()
