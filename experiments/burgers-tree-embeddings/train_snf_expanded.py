"""Native SNF expanded-data training adapter; no projection or PDE qualification."""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse, json, shutil, sys, time, traceback, tomllib
from pathlib import Path
import numpy as np


def descriptor(parameters=None, definition=None, nu=.005):
    """Physical metadata for OFFLINE hyper training only; two canonical slots."""
    if definition is None:
        kind, cx, cy, width, amplitude, background = map(float, parameters)
        gaussians = [] if kind == 1 else [dict(center=[cx, cy], sigma_t=width,
                            sigma_n=width, amplitude=amplitude, angle_degrees=0.)]
    else:
        background = definition['background']; gaussians = definition['gaussians']
    assert len(gaussians) <= 2
    # Canonical ordering makes the descriptor invariant to Gaussian list order.
    gaussians = sorted(gaussians, key=lambda g: tuple(g['center'])+(g['amplitude'],))
    result = [float(background), float(nu)]
    for g in gaussians:
        angle = 2*np.deg2rad(g['angle_degrees'])
        result.extend([g['amplitude'], *g['center'], g['sigma_t'], g['sigma_n'],
                       float(np.sin(angle)), float(np.cos(angle))])
    result.extend([0.]*(16-len(result)))
    value = np.asarray(result, dtype='<f8'); assert value.shape == (16,) and np.isfinite(value).all()
    return value


def write_binary(path, array):
    array = np.asarray(array, dtype='<f8')
    array.ravel(order='F').tofile(path)
    return list(array.shape)


def fixture():
    import tempfile
    single = descriptor([0., .3, .4, .05, 1.2, .15])
    assert np.array_equal(single[:7], [.15, .005, 1.2, .3, .4, .05, .05])
    assert np.array_equal(single[9:], np.zeros(7))
    assert np.array_equal(descriptor([1., 0., 0., 1., 0., .6])[2:], np.zeros(14))
    from neural.expanded_cases import definitions
    rows = definitions('train'); assert len(rows) == 32
    for row in rows:
        d = row['definition']; reverse = dict(d, gaussians=list(reversed(d['gaussians'])))
        assert np.array_equal(descriptor(definition=d), descriptor(definition=reverse))
    with tempfile.TemporaryDirectory() as tmp:
        a = np.arange(120, dtype=np.float64).reshape(1, 5, 8, 3)
        p = Path(tmp)/'shape.f64'; shape = write_binary(p, a)
        assert np.array_equal(np.fromfile(p, dtype='<f8').reshape(shape, order='F'), a)
    print('SNF_DESCRIPTOR_AND_COLUMN_MAJOR_SHAPE_PASS')


def patch_scientific_progress(project):
    """Exact scientific-only observer insertion; native updates/stopping stay intact."""
    import difflib, hashlib
    from qualify_snf_training import replace_once
    path=project/'src/train.jl';original=path.read_text()
    modified=replace_once(original,
        '        kwargs = (;lossfun, opt_st, cb, io, early_stopping, patience, schedule)',
        '        kwargs = (;lossfun, opt_st, cb, io, early_stopping, patience, schedule,\n'
        '            b8_context=(dir=dir,metadata=metadata,stats=STATS,stage=iopt))')
    modified=replace_once(modified,'    for epoch in 1:nepoch\n',
        '    b8_context = get(kwargs, :b8_context, nothing)\n'
        '    for epoch in 1:nepoch\n        b8_epoch_begin = time_ns()\n')
    modified=replace_once(modified,'        for batch in _loader\n',
        '        b8_batch_begin = time_ns()\n        for batch in _loader\n')
    modified=replace_once(modified,
        '        # callback, early stopping\n        _, l_, ifbreak_cb = cb(p, st; epoch, nepoch, io)',
        '        b8_batch_seconds = (time_ns()-b8_batch_begin)/1e9\n'
        '        b8_callback_begin = time_ns()\n'
        '        # callback, early stopping\n        _, l_, ifbreak_cb = cb(p, st; epoch, nepoch, io)')
    modified=replace_once(modified,
        '        minconfig, ifbreak_mc = update_minconfig(minconfig, l_, p, st, opt_st; io)\n',
        '        minconfig, ifbreak_mc = update_minconfig(minconfig, l_, p, st, opt_st; io)\n'
        '        b8_callback_seconds = (time_ns()-b8_callback_begin)/1e9\n'
        '        b8_scientific_progress(b8_context, NN, p, st, minconfig, epoch, nepoch,\n'
        '            num_batches, b8_batch_seconds, b8_callback_seconds,\n'
        '            (time_ns()-b8_epoch_begin)/1e9, l_, ifbreak_cb | ifbreak_mc)\n')
    modified='include(joinpath(ENV["B8_SNF_OUTPUT"], "snf_progress.jl"))\n'+modified
    path.chmod(path.stat().st_mode|0o200);path.write_text(modified)
    sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
    return dict(file='src/train.jl',original_sha256=sha(original),patched_sha256=sha(modified)), ''.join(
        difflib.unified_diff(original.splitlines(True),modified.splitlines(True),
                             fromfile='train.jl',tofile='train.jl.scientific-progress'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out'); ap.add_argument('--fixture', action='store_true')
    ap.add_argument('--qualification-job', default='b8snftrainfix')
    ap.add_argument('--boundary-envelope', action='store_true')
    ap.add_argument('--boundary-qualification-job', default='b8snfboundary')
    ap.add_argument('--progress-qualification-job', default='b8snfprogress')
    ap.add_argument('--timeout-seconds', type=int, default=6900)
    a = ap.parse_args()
    if a.fixture:
        fixture(); return
    assert a.out and 0 < a.timeout_seconds <= 6900
    assert a.qualification_job.replace('-', '').isalnum()
    # Imports below this point may initialize accelerators; tiny fixture does not.
    from common import HERE, save, digest, preflight, jax, jnp
    from neural.build_data import resolve
    from neural import expanded_cases as E
    from qualify_native_competitors import command
    import data
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'result.json').exists()
    result = dict(status='starting', final_test_opened=False, representation_admitted=False,
        pde_comparator_admitted=False, exact_resume_supported=False,
        classification='Pinned native SNF training loop, documented Float64/new-data/physical-metadata port',
        online_initialization='Future online decoder codes must be fitted from common initial field; physical metadata and hypernetwork are not online initializers',
        evaluation='No projection, actual4096, or PDE accuracy claim from native training loss')
    started = time.monotonic()
    save(out/'result.json', result)
    try:
        assert os.environ.get('SLURM_JOB_ID'), 'Scientific synthesis/training requires a cluster allocation'
        ns = Path(os.environ['STUDY_NAMESPACE'])
        assert str(ns).startswith('/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-')
        result['meta'] = preflight()
        qpath = resolve(ns/a.qualification_job/'output/result.json', ns)
        qualification = json.loads(qpath.read_text())
        retention = HERE/'manifests'/('retention-'+a.qualification_job+'.json')
        retained = json.loads(retention.read_text()); assert retained['local_and_remote_verified']
        rhashes = {row['file']: row['sha256'] for row in retained['files']}
        assert digest(qpath) == rhashes['output/result.json']
        assert qualification['passed'] and qualification['fixture']['passed']
        assert qualification['fixture']['training_only_normalization_invariant_to_development_shift']
        assert qualification['fixture']['native_training_steps'] == 7
        assert qualification['fixture']['directional_gradient_relative_error'] < 1e-6
        native = resolve(ns/'b6native/output/result.json', ns); evidence = json.loads(native.read_text())
        assert digest(native) == qualification['native_source_result_sha256']
        oldproject = qpath.parent/'sources/snf'
        assert digest(oldproject/'Manifest.toml') == qualification['resolved_manifest_sha256']
        patched = {row['file']: row['patched_sha256'] for row in qualification['patches']}
        for row in evidence['sources']['snf']['files']:
            name = row['path']; assert digest(oldproject/name) == patched.get(name, row['sha256'])
        binary = native.parent/'runtime/julia-1.10.10/bin/julia'
        assert digest(binary) == evidence['julia']['binary_sha256']
        compat = resolve(ns/'b7snfcompatfix/output/result.json', ns)
        lux_rule = compat.parent/'julia-depot/packages/Lux/68T9j/src/chainrules.jl'
        assert digest(lux_rule) == qualification['lux_patched_rule_sha256']
        project = out/'sources/snf'; shutil.copytree(oldproject, project)
        if a.boundary_envelope:
            from qualify_snf_boundary import patch_boundary
            assert a.boundary_qualification_job.replace('-', '').isalnum()
            bqp = resolve(ns/a.boundary_qualification_job/'output/result.json', ns)
            bq = json.loads(bqp.read_text()); assert bq['passed'] and bq['fixture']['boundary_wrapper_passed']
            brp = HERE/'manifests'/('retention-'+a.boundary_qualification_job+'.json')
            br = json.loads(brp.read_text()); assert br['local_and_remote_verified']
            bh = {row['file']:row['sha256'] for row in br['files']}
            assert digest(bqp)==bh['output/result.json']
            assert bq['native_training_qualification_sha256']==digest(qpath)
            assert digest(HERE/'baselines/snf_boundary.jl')==bq['boundary_wrapper_sha256']
            boundary_patch, patchtext = patch_boundary(project)
            assert boundary_patch==bq['boundary_patch']
            (out/'boundary-port.patch').write_text(patchtext)
            assert digest(out/'boundary-port.patch')==bq['boundary_patch_sha256']
            shutil.copy2(HERE/'baselines/snf_boundary.jl', out/'snf_boundary.jl')
            result.update(boundary_envelope=True, boundary_qualification_sha256=digest(bqp),
                boundary_retention_sha256=digest(brp), boundary_wrapper_sha256=bq['boundary_wrapper_sha256'],
                boundary_patch=boundary_patch, classification='Native SNF training with continuous physical Dirichlet decoder; modified new-PDE port, not unchanged published reproduction')
        else:
            result['boundary_envelope']=False
            result['boundary_scope']='Unconstrained native-network control; not an eligible boundary-conforming PDE comparator'
        assert a.progress_qualification_job.replace('-', '').isalnum()
        pqp=resolve(ns/a.progress_qualification_job/'output/result.json',ns)
        pq=json.loads(pqp.read_text());assert pq['passed'] and pq['scientific_progress'] and pq['fixture']['scientific_progress_passed']
        prm=HERE/'manifests'/('retention-'+a.progress_qualification_job+'.json')
        pm=json.loads(prm.read_text());assert pm['local_and_remote_verified']
        ph={row['file']:row['sha256'] for row in pm['files']};assert digest(pqp)==ph['output/result.json']
        assert pq['native_training_qualification_sha256']==digest(qpath)
        assert pq['scientific_progress_helper_sha256']==digest(HERE/'baselines/snf_progress.jl')
        result.update(progress_qualification_sha256=digest(pqp),progress_retention_sha256=digest(prm))
        progress_patch, progress_diff = patch_scientific_progress(project)
        assert progress_patch==pq['scientific_progress_patch']
        (out/'scientific-progress.patch').write_text(progress_diff)
        assert digest(out/'scientific-progress.patch')==pq['scientific_progress_patch_sha256']
        shutil.copy2(HERE/'baselines/snf_progress.jl', out/'snf_progress.jl')
        result.update(scientific_progress_patch=progress_patch,
            scientific_progress_patch_sha256=digest(out/'scientific-progress.patch'),
            scientific_progress_helper_sha256=digest(out/'snf_progress.jl'))
        adapter = out/'snf_expanded_training.jl'; shutil.copy2(HERE/'baselines/snf_expanded_training.jl', adapter)
        depot = out/'julia-depot'; depot.mkdir()
        lookup = [depot, qpath.parent/'julia-depot'] + [resolve(ns/job/'output/julia-depot', ns)
                  for job in ['b7snff64', 'b7snfgpu', 'b7snfcompatfix', 'b6snfnocache', 'b6native']]
        assert all(p.is_dir() for p in lookup)
        result.update(qualification_job=a.qualification_job, qualification_sha256=digest(qpath),
            retention_sha256=digest(retention), source_commit=qualification['source_commit'],
            source_patches=qualification['patches'], resolved_manifest_sha256=digest(project/'Manifest.toml'),
            lux_patched_rule_sha256=digest(lux_rule), adapter_sha256=digest(adapter))
        save(out/'result.json', result)
        original_path = resolve(ns/'b6parent/output/coordinate_data.json', ns)
        original = json.loads(original_path.read_text())
        groups, provenance = {}, []
        for cohort, job, expected in [('train', 'b8trainrefs', 160), ('dev', 'b8devrefs', 52)]:
            rows = [dict(row, family='primary') for row in original['groups'][cohort]['sources']]
            rp = resolve(ns/job/'output/result.json', ns); record = json.loads(rp.read_text())
            assert record['cohort'] == cohort and record['all_ladders_pass'] and not record['final_test_opened']
            definitions = E.definitions(cohort)
            assert record['spec']['definitions'] == definitions
            assert len(record['cases']) == len(definitions)
            for row, definition in zip(record['cases'], definitions):
                assert all(row[k] == v for k, v in definition.items())
                assert row['complete'] and row['provisional_ladder_pass']
                last = row['levels'][-1]; assert last['N'] == 512 and last['dt'] == .00015625
                rows.append(dict(case=row['case'], family=row['family'], definition=row['definition'], nu=row['nu'],
                                 file=str(rp.parent/last['file']), sha256=last['sha256']))
            assert len(rows) == expected and len({r['case'] for r in rows}) == expected
            groups[cohort] = rows; provenance.append(dict(job=job, sha256=digest(rp)))
        assert not ({r['case'] for r in groups['train']} & {r['case'] for r in groups['dev']})
        axis = np.linspace(0., 1., 128)
        xy = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2).T
        shape = (1, 16384, 212, 11)
        fieldpath = out/'u.f64'
        fields = np.memmap(fieldpath, mode='w+', dtype='<f8', shape=shape, order='F')
        descriptors = []; synth = jax.jit(data.synth_grid)
        for index, row in enumerate(groups['train']+groups['dev']):
            path = resolve(row['file'], ns); assert digest(path) == row['sha256']
            with np.load(path) as v:
                assert np.array_equal(v['times'], data.TIMES)
                b = v['coefficients']
                d = descriptor(definition=row['definition'], nu=row['nu']) if 'definition' in row else descriptor(v['parameters'])
                y = np.asarray(synth(jnp.asarray(b), jnp.asarray(axis))).reshape(11, 16384)
            assert np.isfinite(y).all(); fields[0, :, index, :] = y.T; descriptors.append(d)
            if index % 16 == 0: print('SNF_SOURCE_SYNTHESIS', index+1, 212, flush=True)
        fields.flush(); del fields
        raw = np.stack(descriptors, axis=1)
        mean = raw[:, :160].mean(axis=1); scale = raw[:, :160].std(axis=1)
        scale = np.where(scale < 1e-12, 1., scale)
        normalized = (raw-mean[:, None])/scale[:, None]
        shapes = dict(x=write_binary(out/'x.f64', xy), u=list(shape),
                      t=write_binary(out/'t.f64', data.TIMES), mu=write_binary(out/'mu.f64', normalized))
        # Julia TOML stdlib reads explicit shapes; arrays are little-endian Float64,
        # column-major. No NPZ, Python package, or implicit reshape convention.
        (out/'raw-data.toml').write_text('dtype = "Float64"\nbyte_order = "little"\norder = "F"\ntrain_count = 160\ndev_count = 52\n'+
            ''.join(f'{k}_shape = {json.dumps(v)}\n' for k, v in shapes.items()))
        save(out/'descriptor_statistics.json', dict(mean=mean, scale=scale, raw=raw,
             schema=['background', 'nu']+[f'gaussian{n}_{k}' for n in [1,2]
                     for k in ['amplitude','cx','cy','sigma_t','sigma_n','sin2theta','cos2theta']],
             fitting_cases=160, time='Native training-only time mean/std; 17 total hyper inputs',
             constant_dimension_scale=1., online_use=False))
        save(out/'matched_sources.json', dict(primary_record_sha256=digest(original_path), expanded=provenance, groups=groups))
        result.update(status='native_expanded_training', train_trajectories=160, dev_trajectories=52,
            raw_files={k: dict(file=k+'.f64', shape=s, sha256=digest(out/(k+'.f64'))) for k,s in shapes.items()},
            descriptor_statistics_sha256=digest(out/'descriptor_statistics.json'),
            matched_sources_sha256=digest(out/'matched_sources.json'),
            config=dict(latent=32, hh=0, hd=5, hyper_width=64, decoder_width=128, requested_epochs=210, native_early_stopping=True,
                schedule="Seven requested30-epoch stages; actual completed epochs may be fewer",
                warmup=False, train_batch=262144, validation_batch=262144, hyper_l2=.001,
                lipschitz_penalty=0., decoder_weight_decay=.01, seed=2026111991),
            checkpoint_policy='Retain native stages/final plus current and best model at epoch1/every5epochs/stage-end/early-stop; model/STATS only, no exact optimizer/RNG resume',
            initialization_caveat='Native rand32 bias values are converted to Float64; optimizer arithmetic and gradients are Float64')
        save(out/'result.json', result)
        env = dict(os.environ, JULIA_DEPOT_PATH=':'.join(map(str, lookup)), JULIA_PKG_PRECOMPILE_AUTO='0',
            JULIA_NUM_THREADS='4', JULIA_NUM_PRECOMPILE_TASKS='4', GKSwstype='100',
            CUDA_CACHE_PATH=str(out/'cuda-cache'), B8_SNF_OUTPUT=str(out), B8_SNF_PROJECT=str(project), B8_SNF_BOUNDARY='1' if a.boundary_envelope else '0')
        remaining = int(a.timeout_seconds-(time.monotonic()-started)); assert remaining > 0
        result['execution'] = command([str(binary), '--startup-file=no', '--project='+str(project), str(adapter)],
            out/'native_expanded_training.log', env, timeout=remaining)
        metrics = out/'expanded-training.toml'
        if metrics.exists(): result['training'] = tomllib.loads(metrics.read_text())
        result['status'] = 'native_training_complete' if result['execution']['returncode'] == 0 else 'native_training_incomplete'
        result['native_training_completed'] = result['execution']['returncode'] == 0 and result.get('training', {}).get('completed', False)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed_wall_seconds'] = time.monotonic()-started
        result['checkpoints'] = [dict(file=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=digest(p))
                                 for p in sorted((out/'native_model').glob('*.jld2'))]
        progress_file = out/'native_model/epoch_progress.tsv'
        if progress_file.exists(): result['epoch_progress_sha256'] = digest(progress_file)
        log = out/'native_expanded_training.log'
        if log.exists(): result['log_sha256'] = digest(log)
        save(out/'result.json', result)


if __name__ == '__main__':
    main()
