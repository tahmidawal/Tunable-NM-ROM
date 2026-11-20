"""All-case B8 reconstruction checks; no PDE rollout or online timing claim.

Field-assisted starts and fitting are offline representation diagnostics. Extended
reference ladders remain provisional until the separate independent FD check.
"""
import argparse
from common import *
from neural import stable as S, fields as F, allocations as A, residual_tree as R
from neural import expanded_cases as E
from neural.build_data import resolve
from training_io import checkpoint, read_checkpoint
import data


def owned_job(value):
    assert value and value.replace('-', '').isalnum(), value
    return value


def job_list(value):
    jobs = [owned_job(x) for x in value.split(',') if x]
    assert len(jobs) == len(set(jobs)), 'Duplicate source jobs'
    return jobs


def equal_tree(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    return all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)))


def load_group(ns, group, jobs, arrays, parent):
    record = parent['data_record']['groups'][group]
    result = dict(labels=list(record['labels']), sources=list(record['sources']),
                  family=['primary']*len(record['labels']), provenance=[],
                  y=[arrays[group+'_y']], encoder=[arrays[group+'_encoder']],
                  norm=[arrays[group+'_norm']])
    for job in jobs:
        rp = resolve(ns/job/'output/result.json', ns)
        source = json.loads(rp.read_text())
        assert source['cohort'] == group and not source['final_test_opened']
        assert source['status'] == 'complete_expanded_ladders_pending_independent_crosscheck'
        assert source['all_ladders_pass']
        spec = source['spec']
        expected = E.definitions(group, spec['start'], spec['count'])
        assert spec['definitions'] == expected
        assert [row['case'] for row in source['cases']] == [row['case'] for row in expected]
        cache = rp.parent/'coordinate_data.npz'
        assert digest(cache) == source['coordinate_data']['sha256']
        with np.load(cache) as v:
            for key in ['xy', 'w', 'encoder_xy']:
                assert np.array_equal(v[key], arrays[key])
            assert np.array_equal(v['case_ids'], np.repeat([r['case'] for r in expected], 11))
            assert np.array_equal(v['times'], np.tile(data.TIMES, len(expected)))
            for key in ['y', 'encoder', 'norm']:
                result[key].append(np.array(v[key]))
        for row, definition in zip(source['cases'], expected):
            assert all(row[key] == value for key, value in definition.items())
            assert row['complete'] and row['provisional_ladder_pass']
            level = row['levels'][-1]
            result['labels'].append(row['case']); result['family'].append(row['family'])
            result['sources'].append(dict(case=row['case'], file=str(rp.parent/level['file']),
                                          sha256=level['sha256'], allowance=row['allowance']))
        result['provenance'].append(dict(job=job, cohort=group, record_sha256=digest(rp),
                                         cache_sha256=digest(cache)))
    assert len(set(result['labels'])) == len(result['labels'])
    for key in ['y', 'encoder', 'norm']:
        result[key] = jnp.asarray(np.concatenate(result[key]))
        assert len(result[key]) == 11*len(result['labels'])
    return result


def make_initializer(child_k, nodes):
    """Same residual-tangent initialization/backtracking as B8 training."""
    @jax.jit
    def initialize(p, rz, B, y, norm, xy, w):
        D = jax.jacfwd(R.coefficients, argnums=1)(p['head'], jnp.zeros(child_k), p['scale'])
        J = jnp.einsum('pnr,rk->pnk', B, D).reshape(len(xy), -1)
        Jw = J*jnp.sqrt(w)[:, None]; H = Jw.T@Jw
        scale = jax.lax.rsqrt(jnp.maximum(jnp.diag(H), 1e-14))
        pred = S.apply(p['root'], rz[:, None, :], xy[None])
        rhs = (y-pred)@(w[:, None]*J)
        z = jnp.linalg.solve(H*scale[:, None]*scale[None, :]+1e-3*jnp.eye(H.shape[0]),
                             (rhs*scale).T).T*scale
        z = z.reshape(len(y), nodes, child_k)
        z = z*jnp.minimum(1., 2/jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-30))
        alphas = jnp.array([0., 1., .5, .25, .125])
        def loss(alpha):
            c = R.coefficients(p['head'], alpha*z, p['scale'])
            reconstructed = pred+jnp.einsum('pnr,tnr->tp', B, c)
            return jnp.sum(w*(reconstructed-y)**2, axis=1)/norm**2
        losses = jax.vmap(loss)(alphas)
        best = jnp.argmin(losses, axis=0)
        z = z*alphas[best, None, None]
        return jnp.concatenate([rz, z.reshape(len(y), -1)], axis=1), jnp.sqrt(jnp.min(losses, axis=0))
    return initialize


def zero_child_audit(params, code, B, xy, weight):
    """New directions after removing the physical weighted parent tangent."""
    root_k, child_k = R.dimensions(params)
    zero = jnp.concatenate([code[:root_k], jnp.zeros(len(code)-root_k)])
    pp = dict(params, B=B)
    difference = R.cached(pp, zero, xy)-S.apply(params['root'], code[:root_k], xy)
    J = R.cached_jacobian(pp, zero, xy)*jnp.sqrt(weight)[:, None]
    parent, child = J[:, :root_k], J[:, root_k:]
    parent_norm = jnp.sqrt(jnp.sum(parent**2, axis=0))
    child_norm = jnp.sqrt(jnp.sum(child**2, axis=0))
    U, singular, _ = jnp.linalg.svd(parent/jnp.maximum(parent_norm, 1e-30), full_matrices=False)
    keep = singular > jnp.maximum(singular[0], 1e-30)*1e-6
    U = U*keep[None, :]
    normalized = child/jnp.maximum(child_norm, 1e-30)
    orthogonal = normalized-U@(U.T@normalized)
    gram = orthogonal.T@orthogonal
    eig = jnp.maximum(jnp.linalg.eigvalsh((gram+gram.T)/2), 0.)
    sig = jnp.sqrt(eig); ratio = sig/jnp.maximum(sig[-1], 1e-30)
    rank = int(jnp.sum(ratio > 1e-6))
    preservation = float(jnp.max(jnp.abs(difference)))
    return dict(zero_child_max_error=preservation, zero_child_preserved=preservation <= 1e-13,
                root_directions=int(jnp.sum(keep)), child_directions=len(code)-root_k,
                independent_child_directions_after_parent=rank, relative_singular_cutoff=1e-6,
                minimum_normalized_singular_ratio=float(ratio[0]),
                parent_column_norms=np.asarray(parent_norm), child_column_norms=np.asarray(child_norm),
                residualized_normalized_column_norms=np.asarray(jnp.sqrt(jnp.sum(orthogonal**2, axis=0))),
                residualized_singular_values=np.asarray(sig), child_k=child_k,
                useful_independent_directions=bool(rank == len(code)-root_k and float(child_norm.min()) > 1e-12))


def actual_witness(out, kind, params, row, coefficients):
    """All4096² values; bounded tile feature allocation rather than full-grid B."""
    axis = np.linspace(0., 1., 4096)
    z = jnp.asarray(row['codes']); cells = jnp.asarray(row['cells'])
    root_k, _ = R.dimensions(params)
    root_decode = jax.jit(S.apply)
    decode = jax.jit(R.tree)
    prediction = np.empty((4096, 4096), np.float64)
    parent_prediction = np.empty_like(prediction)
    for lo in range(0, 4096, 4):
        points = jnp.asarray(np.stack(np.meshgrid(axis[lo:lo+4], axis, indexing='ij'), -1).reshape(-1, 2))
        prediction[lo:lo+4] = np.asarray(decode(params, z, points, cells)).reshape(4, 4096)
        parent_prediction[lo:lo+4] = np.asarray(root_decode(params['root'], z[:root_k], points)).reshape(4, 4096)
    truth = np.asarray(data.synth_grid(jnp.asarray(coefficients), jnp.asarray(axis)))
    error = prediction-truth
    relative = float(np.linalg.norm(error)/np.linalg.norm(truth))
    path = out/f'{kind}_worst_actual4096.npz'
    np.savez_compressed(path, reference=truth, parent_prediction=parent_prediction,
                        tree_prediction=prediction, child_correction=prediction-parent_prediction,
                        error=error, cells=np.asarray(cells), codes=np.asarray(z),
                        case=row['case'], time=row['time'],
                        parent_role='Parent component at jointly fitted tree code, not separately optimal parent fit')
    return dict(file=path.name, sha256=digest(path), case=row['case'], time=row['time'],
                shape=[4096, 4096], relative_error=relative, values_computed=4096**2,
                reference_norm='current-time full4096 reference norm', role='offline reconstruction, not PDE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--model-job', required=True)
    ap.add_argument('--train-jobs', default=''); ap.add_argument('--dev-jobs', default='')
    ap.add_argument('--layouts', default='uniform,prescribed'); ap.add_argument('--limit-cases', type=int)
    ap.add_argument('--nodes', type=int, default=20, choices=[4, 20])
    a = ap.parse_args(); owned_job(a.model_job)
    layouts = a.layouts.split(',')
    assert layouts and len(set(layouts)) == len(layouts) and set(layouts) <= {'uniform', 'prescribed'}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); metadata = preflight()
    ns = Path(os.environ['STUDY_NAMESPACE'])
    mp = resolve(ns/a.model_job/'output/nonlinear_tree.pkl', ns)
    mrp = mp.parent/'result.json'; model_record = json.loads(mrp.read_text())
    assert model_record['status'] in ('complete_nonlinear_child_subset','complete_nonlinear_child_multidepth_subset')
    assert digest(mp) == model_record['selected_sha256'] and not model_record['final_test_opened']
    model = read_checkpoint(mp); params = jax.tree.map(jnp.asarray, model['params'])
    pp = resolve(ns/'b7parentrepair/output/stable_parent.pkl', ns)
    pr = json.loads((pp.parent/'result.json').read_text())
    assert digest(pp) == pr['selected_sha256'] == model['parent_sha256']
    parent = read_checkpoint(pp)
    assert equal_tree(params['root'], parent['network']), 'Frozen parent weights changed'
    dp = resolve(ns/'b6parent/output/coordinate_data.npz', ns)
    assert digest(dp) == parent['data_sha256']
    with np.load(dp) as v: arrays = {k: np.array(v[k]) for k in v.files}
    root_k, child_k = R.dimensions(params)
    trained_nodes = model_record['nodes']; nodes = a.nodes
    assert nodes <= trained_nodes and root_k == 32 and child_k == model_record['child_k']
    declared = model.get('expanded_sources', [])
    train_jobs = job_list(a.train_jobs) if a.train_jobs else [r['job'] for r in declared if r['cohort'] == 'train']
    dev_jobs = job_list(a.dev_jobs) if a.dev_jobs else [r['job'] for r in declared if r['cohort'] == 'dev']
    train = load_group(ns, 'train', train_jobs, arrays, parent)
    dev = load_group(ns, 'dev', dev_jobs, arrays, parent)
    assert not set(train['labels']) & set(dev['labels'])
    assert train['labels'] == model_record['train_labels'], 'Training cohort must match selected checkpoint'
    for entry in declared:
        matches = [p for p in train['provenance']+dev['provenance'] if p['job'] == entry['job']]
        assert len(matches) == 1 and matches[0] == entry, 'Expanded cache changed since training'
    total_cases = len(dev['labels']); count = total_cases if a.limit_cases is None else a.limit_cases
    assert 0 < count <= total_cases
    limited = count < total_cases
    xy, w = jnp.asarray(arrays['xy']), jnp.asarray(arrays['w'])
    xf, wf = data.grid(256, True); xf, wf = jnp.asarray(xf), jnp.asarray(wf)
    fit = F.make_fit(R.cached, 60, jacobian=R.cached_jacobian)
    more = F.make_fit(R.cached, 120, jacobian=R.cached_jacobian)
    rootfit = F.make_fit(S.apply, 60); rootmore = F.make_fit(S.apply, 120, newton=True)
    initializer = make_initializer(child_k, nodes)
    devfit_path = resolve(ns/'b7parentrepair/output/development_fits.pkl', ns)
    assert digest(devfit_path) == pr['all_fits_sha256']
    original = read_checkpoint(devfit_path)
    original_codes = jnp.asarray(original['codes']); original_status = np.asarray(original['status'])
    assert len(original_codes) == len(parent['data_record']['groups']['dev']['labels'])*11
    result = dict(meta=metadata, status='running', model_job=a.model_job, model_sha256=digest(mp),
                  model_record_sha256=digest(mrp), parent_sha256=digest(pp), primary_data_sha256=digest(dp),
                  expanded_sources=train['provenance']+dev['provenance'], final_test_opened=False,
                  parent_weights_verified_identical=True, expected_cases=total_cases, evaluated_cases=count,
                  diagnostic_subset=limited, train_labels=train['labels'], dev_labels=dev['labels'],
                  normalization='current-time reference physical L2', root_k=root_k, child_k=child_k,
                  trained_nodes=trained_nodes, evaluated_nodes=nodes, nodes=nodes, active_variables=root_k+nodes*child_k, layouts={}, representation_admitted=False,
                  pde_admitted=False, reconstruction_role='offline independently fitted fields, not PDE trajectories',
                  reference_status='Primary references qualified; expanded references pending separate independent FD checks',
                  model_training_coverage='expanded' if train_jobs else 'primary only')
    save(out/'result.json', result)
    root_cache = {}
    for kind in layouts:
        records = []; statuses = []; errors = []; changes = []; worst = None
        for ci in range(count):
            source = dev['sources'][ci]; path = resolve(source['file'], ns)
            assert digest(path) == source['sha256']
            with np.load(path) as v: coefficients = jnp.asarray(v['coefficients'])
            truth = data.synth_grid(coefficients, jnp.unique(xf[:, 0])).reshape(11, -1)
            cell = jnp.asarray(A.allocate(kind, np.asarray(xy), np.asarray(dev['y'][11*ci]),
                                         model_record['seed']+ci, nodes=trained_nodes, max_depth=3)[:nodes])
            A.validate(cell)
            B = R.feature_matrix(params['feature'], cell, xy)
            Bf = R.feature_matrix(params['feature'], cell, xf)
            cp, fp = dict(params, B=B), dict(params, B=Bf)
            rows = []
            for ti in range(11):
                i = 11*ci+ti
                if i < len(original_codes):
                    rz, rst = original_codes[i], original_status[i]
                elif i in root_cache:
                    rz, rst = root_cache[i]
                else:
                    near, interp = S.nearest_starts(dev['encoder'][i], jnp.asarray(parent['training_encoder']),
                                                   jnp.asarray(parent['training_codes']))
                    attempts = [rootfit(params['root'], start, xy, dev['y'][i], w, dev['norm'][i])
                                for start in [near, interp, jnp.zeros(root_k)]]
                    rz, rst = min(attempts, key=lambda item: float(item[1][0]))
                    if not bool(rst[-1]):
                        rz, rst = rootmore(params['root'], rz, xy, dev['y'][i], w, dev['norm'][i])
                    root_cache[i] = (rz, np.asarray(rst))
                initial, initerror = initializer(params, rz[None], B, dev['y'][i:i+1], dev['norm'][i:i+1], xy, w)
                z, st = fit(cp, initial[0], xy, dev['y'][i], w, dev['norm'][i])
                attempts = [dict(start='residual_tangent', status=np.asarray(st))]
                if not bool(st[-1]):
                    z, st = more(cp, z, xy, dev['y'][i], w, dev['norm'][i])
                    attempts.append(dict(start='residual_tangent_continuation', status=np.asarray(st)))
                # A second field-preserving start diagnoses unfinished nonlinear fits;
                # it never changes the stopping tolerances or replaces a better fit.
                if not bool(st[-1]):
                    z0 = jnp.concatenate([rz, jnp.zeros(nodes*child_k)])
                    alt, ast = fit(cp, z0, xy, dev['y'][i], w, dev['norm'][i])
                    attempts.append(dict(start='zero_child', status=np.asarray(ast)))
                    if not bool(ast[-1]):
                        alt, ast = more(cp, alt, xy, dev['y'][i], w, dev['norm'][i])
                        attempts.append(dict(start='zero_child_continuation', status=np.asarray(ast)))
                    if float(ast[0]) < float(st[0]): z, st = alt, ast
                prediction = R.cached(fp, z, xf)
                independent = float(jnp.sqrt(jnp.sum(wf*(prediction-truth[ti])**2))/dev['norm'][i])
                parent_error = float(jnp.sqrt(jnp.sum(wf*(S.apply(params['root'], rz, xf)-truth[ti])**2))/dev['norm'][i])
                row = dict(index=i, case=dev['labels'][ci], family=dev['family'][ci], time=float(data.TIMES[ti]),
                           layout=kind, codes=np.asarray(z), cells=np.asarray(cell), status=np.asarray(st),
                           root_start_status=np.asarray(rst), root_start_code=np.asarray(rz),
                           initialization_error=float(initerror[0]), fit_attempts=attempts,
                           parent_independent_error=parent_error, independent_error=independent,
                           scoring_change=abs(independent-float(st[0])), reference_sha256=source['sha256'])
                rows.append(row); statuses.append(np.asarray(st)); errors.append(independent); changes.append(row['scoring_change'])
                if worst is None or independent > worst['independent_error']: worst = row
                print('tree_qualification', kind, row['case'], ti, independent, bool(st[-1]), flush=True)
            casefile = out/f'{kind}_{dev["labels"][ci]}_fits.pkl'
            records.append(dict(case=dev['labels'][ci], family=dev['family'][ci], file=casefile.name,
                                sha256=checkpoint(casefile, rows), worst=max(r['independent_error'] for r in rows),
                                unfinished=sum(not bool(r['status'][-1]) for r in rows)))
            result['layouts'][kind] = dict(status='running', cases=records, independent_errors=errors,
                                          unfinished_fits=sum(not bool(s[-1]) for s in statuses),
                                          max_scoring_change=max(changes), worst_independent_error=max(errors))
            save(out/'result.json', result)
            # B/Bf are per-trajectory caches; do not retain them in records or closures.
            del cp, fp, B, Bf, coefficients, truth
        wi = worst['index']; wci, wti = divmod(wi, 11)
        source = dev['sources'][wci]; path = resolve(source['file'], ns)
        assert digest(path) == source['sha256']
        with np.load(path) as v: witness_coefficients = np.array(v['coefficients'][wti])
        actual = actual_witness(out, kind, params, worst, witness_coefficients)
        Bw = R.feature_matrix(params['feature'], jnp.asarray(worst['cells']), xy)
        audit = zero_child_audit(params, jnp.asarray(worst['codes']), Bw, xy, w)
        passed = bool(not limited and max(errors) <= .005 and max(changes) <= .0002
                      and all(bool(s[-1]) for s in statuses) and actual['relative_error'] <= .005
                      and audit['zero_child_preserved'] and audit['useful_independent_directions'])
        result['layouts'][kind].update(status='complete', actual4096=actual, zero_child_audit=audit,
                                       numerical_representation_gate=passed,
                                       representation_admitted=passed and not dev_jobs)
        save(out/'result.json', result)
        del Bw
    result.update(status='complete_residual_tree_representation',
                  numerical_representation_gate=all(r['numerical_representation_gate'] for r in result['layouts'].values()),
                  representation_admitted=all(r['representation_admitted'] for r in result['layouts'].values()),
                  independent_expanded_reference_crosscheck_required=bool(dev_jobs))
    save(out/'result.json', result)


if __name__ == '__main__':
    main()
