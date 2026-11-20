"""Offline frozen-parent child-span discriminator; never an online algorithm."""
import argparse
from common import *
from training_io import read_checkpoint, checkpoint
from neural.build_data import resolve
from neural.parent_provenance import load_parent
from neural import residual_tree as R, stable as S, fields as F, allocations as A
import data


def child_apply(p, z, xy):
    c = R.coefficients(p['head'], z.reshape(-1, p['head'][0]['w'].shape[0]), p['scale'])
    return p['parent'] + jnp.einsum('pnr,nr->p', p['B'], c)


def child_jac(p, z, xy):
    local = z.reshape(-1, p['head'][0]['w'].shape[0])
    d = jax.vmap(jax.jacfwd(R.coefficients, argnums=1), in_axes=(None, 0, None))(
        p['head'], local, p['scale'])
    return jnp.einsum('pnr,nrk->pnk', p['B'], d).reshape(len(xy), -1)


@jax.jit
def span_fit(matrix, residual, weight):
    """Column-scaled SVD, fixed relative cutoff; no normal-equation inversion."""
    weighted = matrix*jnp.sqrt(weight)[:, None]
    scale = jnp.maximum(jnp.linalg.norm(weighted, axis=0), 1e-14)
    u, s, vh = jnp.linalg.svd(weighted/scale, full_matrices=False)
    keep = s > s[0]*1e-10
    coef = (vh.T @ (jnp.where(keep, 1/jnp.maximum(s, 1e-300), 0.)*(u.T@(residual*jnp.sqrt(weight)))))/scale
    return coef, s, keep, scale


def error(pred, truth, weights, norm):
    return float(jnp.sqrt(jnp.sum(weights*(pred-truth)**2))/norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--model-job', required=True)
    ap.add_argument('--root-code-source',choices=['initial','selected-joint'],default='initial')
    ap.add_argument('--time-indices', default='0'); ap.add_argument('--max-seconds', type=int, default=1740)
    a = ap.parse_args(); assert 0 < a.max_seconds <= 1740
    times = [int(x) for x in a.time_indices.split(',')]; assert len(set(times)) == len(times) and all(t in [0,5,10] for t in times)
    started = time.perf_counter(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    metadata = preflight(); ns = Path(os.environ['STUDY_NAMESPACE'])
    rp = resolve(ns/a.model_job/'output/result.json', ns); record = json.loads(rp.read_text())
    assert record['status'].startswith('complete_nonlinear_child') and not record['final_test_opened']
    mp = rp.parent/'nonlinear_tree.pkl'; assert digest(mp) == record['selected_sha256']
    model = read_checkpoint(mp); p = jax.tree.map(jnp.asarray, model['params']); root_k, child_k = R.dimensions(p)
    parent_path,parent_record,parent,parent_proof=load_parent(record,ns)
    assert all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(p['root']),jax.tree.leaves(parent['network'])))
    dp = resolve(ns/'b6parent/output/coordinate_data.npz', ns); assert digest(dp) == parent['data_sha256'] == record['primary_data_sha256']
    arrays = dict(np.load(dp)); xy,w = jnp.asarray(arrays['xy']),jnp.asarray(arrays['w'])
    xf,wf = map(jnp.asarray,data.grid(256,True)); assert len(xy) == 128**2
    dr = parent['data_record']['groups']['dev']; labels=list(dr['labels']); sources=list(dr['sources'])
    ys=[arrays['dev_y']]; norms=[arrays['dev_norm']]; source_records=[]
    for source in record['expanded_sources']:
        if source['cohort'] != 'dev': continue
        erp=resolve(ns/source['job']/'output/result.json',ns); assert digest(erp)==source['record_sha256']
        er=json.loads(erp.read_text()); assert er['cohort']=='dev' and er['all_ladders_pass'] and not er['final_test_opened']
        cache=erp.parent/'coordinate_data.npz'; assert digest(cache)==source['cache_sha256']==er['coordinate_data']['sha256']
        v=np.load(cache); assert np.array_equal(v['xy'],arrays['xy']) and np.array_equal(v['w'],arrays['w'])
        assert np.array_equal(v['case_ids'],np.repeat([r['case'] for r in er['cases']],11))
        assert np.array_equal(v['times'],np.tile(data.TIMES,len(er['cases'])))
        ys.append(v['y']);norms.append(v['norm']);source_records.append(source)
        for row in er['cases']:
            assert row['provisional_ladder_pass']; labels.append(row['case'])
            level=row['levels'][-1];sources.append(dict(file=str(erp.parent/level['file']),sha256=level['sha256']))
    assert labels==record['dev_labels'] and len(set(labels))==len(labels)
    y=jnp.asarray(np.concatenate(ys));norm=jnp.asarray(np.concatenate(norms))
    initpath=rp.parent/'root_initialization.pkl'
    if initpath.exists():
        assert digest(initpath)==record['root_initialization_sha256']; init=read_checkpoint(initpath)
    else:
        initpath=rp.parent/'initialization.pkl'; assert digest(initpath)==record['initialization_sha256'];init=read_checkpoint(initpath)
    rz=jnp.asarray(init['dev_root_codes']); assert rz.shape==(len(y),root_k)
    selected=[labels.index(label) for label in ['dev000','dev003','dev013','align004']]
    expanded=list(range(len(dr['labels']),len(labels))); assert len(expanded)==8
    selected+=expanded
    joint_rows={};joint_proof=None
    if a.root_code_source=='selected-joint':
        manifest_path=HERE/'manifests'/f'retention-{a.model_job}.json';manifest=json.loads(manifest_path.read_text())
        assert manifest['name']==a.model_job and manifest['terminal_state']=='COMPLETED' and manifest['local_and_remote_verified']
        assert str(manifest['job'])==str(record['meta']['job']) and manifest['source_commit']==record['meta']['commit']
        comparison_path=rp.parent/'allocation_comparison.pkl'
        assert digest(comparison_path)==record['comparison_sha256']
        entries={v['file']:v for v in manifest['files']}
        for path in [rp,mp,comparison_path]:
            entry=entries['output/'+path.name]
            assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256']
        comparison=read_checkpoint(comparison_path)
        allowed={(labels[ci],ti,layout) for ci in selected for ti in [0,5,10] for layout in ['coarse4','uniform20','prescribed20']}
        layout_names={'coarse4':'coarse4','uniform':'uniform20','prescribed':'prescribed20'}
        for retained_row in comparison:
            assert retained_row['layout'] in layout_names
            ti=round(retained_row['time']/.025);assert abs(retained_row['time']-data.TIMES[ti])<1e-12
            key=(retained_row['case'],ti,layout_names[retained_row['layout']])
            assert key in allowed and key not in joint_rows
            ci=labels.index(key[0]);assert retained_row['index']==11*ci+ti
            n=4 if key[2]=='coarse4' else 20
            assert np.asarray(retained_row['codes']).shape==(root_k+n*child_k,)
            assert np.asarray(retained_row['cells']).shape==(n,3)
            A.validate(np.asarray(retained_row['cells']))
            assert np.isfinite(np.asarray(retained_row['codes'])).all() and np.isfinite(np.asarray(retained_row['status'])).all()
            if n==20:
                matching=[v for v in model['evaluation'] if v['case']==key[0] and v['time']==retained_row['time'] and v['layout']==retained_row['layout']]
                assert len(matching)==1
                for field in ['codes','cells','status','independent_error']:
                    assert np.array_equal(np.asarray(matching[0][field]),np.asarray(retained_row[field]))
            joint_rows[key]=retained_row
        required={(labels[ci],ti,layout) for ci in selected for ti in times for layout in ['coarse4','uniform20','prescribed20']}
        assert required<=set(joint_rows)
        joint_proof=dict(model_sha256=digest(mp),comparison_sha256=digest(comparison_path),record_sha256=digest(rp),
            retention_manifest_sha256=digest(manifest_path),required_rows=len(required),retained_rows=len(joint_rows),
            score_consistency_absolute_tolerance=1e-10,
            interpretation='Within-layout conditional controls at retained jointly fitted parent codes; different layouts may use different parent codes. Unfinished retained fits do not establish a capacity floor.')
    result=dict(meta=metadata,status='running_offline_span_diagnostic',parent_job=parent_proof['parent_job'],parent_provenance=parent_proof,model_job=a.model_job,
        source_result_sha256=digest(rp),model_sha256=digest(mp),parent_sha256=digest(parent_path),
        root_initialization_sha256=digest(initpath),primary_data_sha256=digest(dp),expanded_sources=source_records,
        selected_cases=[labels[i] for i in selected],time_indices=times,max_seconds=a.max_seconds,
        root_code_source=a.root_code_source,selected_joint_provenance=joint_proof,
        protocol='Frozen parent weights AND per-field parent code; shared checkpoint features; full64 span / zero16 tangent / actual nonlinear child fits',
        svd_relative_cutoff=1e-10,final_test_opened=False,representation_admitted=False,pde_admitted=False,
        offline_truth_used=True,online_method_claim=False,rows=[])
    save(out/'result.json',result)
    fit=F.make_fit(child_apply,60,jacobian=child_jac); more=F.make_fit(child_apply,120,jacobian=child_jac)
    D=jax.jacfwd(R.coefficients,argnums=1)(p['head'],jnp.zeros(child_k),p['scale'])
    for ci in selected:
        src=resolve(sources[ci]['file'],ns); assert digest(src)==sources[ci]['sha256']
        fine=data.synth_grid(jnp.asarray(np.load(src)['coefficients']),jnp.unique(xf[:,0])).reshape(11,-1)
        for layout in ['coarse4','uniform20','prescribed20']:
            if a.root_code_source=='selected-joint':
                first=joint_rows[labels[ci],times[0],layout]
                cells=jnp.asarray(first['cells'])
                assert all(np.array_equal(first['cells'],joint_rows[labels[ci],ti,layout]['cells']) for ti in times)
            else:
                cells=jnp.asarray(A.allocate('prescribed' if layout=='prescribed20' else 'uniform',np.asarray(xy),np.asarray(y[11*ci]),record['seed']+ci,nodes=4 if layout=='coarse4' else 20,max_depth=3))
            B=R.feature_matrix(p['feature'],cells,xy);Bf=R.feature_matrix(p['feature'],cells,xf)
            full=B.reshape(len(xy),-1);fullf=Bf.reshape(len(xf),-1)
            tangent=jnp.einsum('pnr,rk->pnk',B,D).reshape(len(xy),-1)
            tangentf=jnp.einsum('pnr,rk->pnk',Bf,D).reshape(len(xf),-1)
            for ti in times:
                if time.perf_counter()-started>a.max_seconds:
                    result.update(status='partial_time_cap',elapsed_seconds=time.perf_counter()-started);save(out/'result.json',result);return
                i=11*ci+ti;stored=joint_rows.get((labels[ci],ti,layout))
                parent_code=rz[i] if stored is None else jnp.asarray(stored['codes'][:root_k])
                base=S.apply(p['root'],parent_code,xy);basef=S.apply(p['root'],parent_code,xf);res=y[i]-base
                row=dict(case=labels[ci],cohort='dev',time=float(data.TIMES[ti]),layout=layout,
                    parent_fit_error=error(base,y[i],w,norm[i]),parent_independent_error=error(basef,fine[ti],wf,norm[i]),controls={})
                payload=dict(cells=cells,parent_code=parent_code,root_code_source=a.root_code_source,source_sha256=digest(src))
                coeffs={}
                for name,m,mf in [('full_feature_span',full,fullf),('zero_tangent_span',tangent,tangentf)]:
                    (coef,s,keep,scale),elapsed=timer(lambda:span_fit(m,res,w)); coeffs[name]=coef
                    singular=np.asarray(s); mask=np.asarray(keep); rank=int(mask.sum())
                    row['controls'][name]=dict(fit_error=error(base+m@coef,y[i],w,norm[i]),independent_error=error(basef+mf@coef,fine[ti],wf,norm[i]),
                        variables=m.shape[1],rank=rank,condition_retained=float(singular[0]/singular[rank-1]) if rank else None,
                        singular_values=singular,seconds_including_compile=elapsed)
                    payload[name]=dict(coefficients=coef,column_scales=scale)
                cp=dict(p,parent=base,B=B,child_k=child_k); starts=[jnp.zeros(len(cells)*child_k),coeffs['zero_tangent_span']]
                # Trust-limited tangent start avoids starting deeply saturated in the head.
                zz=starts[1].reshape(-1,child_k);starts.append((zz*jnp.minimum(1.,2/jnp.maximum(jnp.linalg.norm(zz,axis=1,keepdims=True),1e-30))).reshape(-1))
                start_names=['zero','tangent','clipped_tangent']
                if stored is not None:
                    stored_child=jnp.asarray(stored['codes'][root_k:])
                    untouched_fit=error(child_apply(cp,stored_child,xy),y[i],w,norm[i])
                    untouched_score=error(child_apply(dict(cp,B=Bf,parent=basef),stored_child,xf),fine[ti],wf,norm[i])
                    assert abs(untouched_fit-float(stored['status'][0]))<=1e-10
                    assert abs(untouched_score-stored['independent_error'])<=1e-10
                    row['untouched_selected_joint']=dict(fit_error=untouched_fit,independent_error=untouched_score,
                        retained_status=np.asarray(stored['status']),retained_independent_error=stored['independent_error'],
                        retained_fit_error=float(stored['status'][0]),retained_converged=bool(stored['status'][-1]))
                    payload['retained_joint_codes']=np.asarray(stored['codes'])
                    starts.append(stored_child);start_names.append('stored_selected_joint_child')
                attempts=[];best=None
                for name,z0 in zip(start_names,starts):
                    ts=time.perf_counter();z,st=fit(cp,z0,xy,y[i],w,norm[i])
                    if not bool(st[-1]):z,st=more(cp,z,xy,y[i],w,norm[i])
                    jax.block_until_ready(st);score=error(child_apply(dict(cp,B=Bf,parent=basef),z,xf),fine[ti],wf,norm[i])
                    attempts.append(dict(start=name,status=np.asarray(st),independent_error=score,seconds_including_compile=time.perf_counter()-ts))
                    if best is None or float(st[0])<best[0]:best=(float(st[0]),z,len(attempts)-1)
                z=best[1];row['nonlinear_fixed_parent']=dict(attempts=attempts,selected_by_fit=best[2],variables=len(z))
                payload['nonlinear_codes']=z
                # Spatial residual energy in each retained cell's core, explicitly overlapping across depths.
                remaining=y[i]-child_apply(cp,z,xy); localization=[]
                for cell in np.asarray(cells):
                    d,cx,cy=cell;h=2.**(-int(d));center=(np.array([cx,cy])+.5)*h
                    inside=jnp.all(jnp.abs(xy-center)<=.5*h,axis=1)
                    localization.append(dict(cell=cell,initial_residual_energy=float(jnp.sum(w*res**2*inside)/norm[i]**2),
                        remaining_residual_energy=float(jnp.sum(w*remaining**2*inside)/norm[i]**2)))
                row['node_core_energy_overlaps_across_depths']=localization
                filename=f'{labels[ci]}_{layout}_t{ti:02d}.pkl';row['payload']=filename;row['payload_sha256']=checkpoint(out/filename,payload)
                result['rows'].append(row);result['elapsed_seconds']=time.perf_counter()-started;save(out/'result.json',result)
                print('span_diagnostic',labels[ci],layout,ti,row['controls']['full_feature_span']['independent_error'],attempts[best[2]]['independent_error'],flush=True)
    result.update(status='complete_offline_span_diagnostic',elapsed_seconds=time.perf_counter()-started);save(out/'result.json',result)


if __name__=='__main__':main()
