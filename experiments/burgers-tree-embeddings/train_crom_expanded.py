"""Matched expanded-data CROM native-network training with projected validation.

This is a documented float64/manual-loop port of pinned author networks and
optimizers. It does not implement or claim an eligible CROM PDE comparator.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse, sys, shutil, difflib, random
from common import *
from neural.build_data import resolve
from neural import expanded_cases as E
import data


def job_names(text):
    jobs = text.split(',') if text else []
    assert len(jobs) == len(set(jobs))
    assert all(job and job.replace('-', '').isalnum() for job in jobs)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-jobs', default='b8trainrefs')
    ap.add_argument('--dev-jobs', default='b8devrefs')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--resume-job', default='')
    ap.add_argument('--latent', type=int, default=32)
    ap.add_argument('--lr-multipliers', type=float, nargs=2, default=[10.,5.])
    a = ap.parse_args()
    assert 0 < a.epochs <= 200 and a.latent > 0
    assert all(np.isfinite(v) and v>0 for v in a.lr_multipliers)
    if a.resume_job:
        job_names(a.resume_job)
        assert ',' not in a.resume_job
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'result.json').exists(), 'Use a new owned job directory for each attempt'
    metadata = preflight(); ns = Path(os.environ['STUDY_NAMESPACE'])
    native = resolve(ns/'b6native/output/result.json', ns)
    evidence = json.loads(native.read_text())
    assert evidence['crom_native_online_fixture']['passed']
    dependencies = resolve(ns/'b6cromdeps/output/python-packages', ns)
    retention = json.loads((HERE/'manifests/retention-b6cromdeps.json').read_text())
    assert retention['local_and_remote_verified']
    verified_dependencies = 0
    for row in retention['files']:
        prefix = 'output/python-packages/'
        if row['file'].startswith(prefix) and not row['file'].endswith('.pyc'):
            assert digest(dependencies/row['file'][len(prefix):]) == row['sha256']
            verified_dependencies += 1
    assert verified_dependencies > 0
    port = out/'crom-port'; port.mkdir()
    hashes = {v['path']: v['sha256'] for v in evidence['sources']['crom_offline']['files']}
    patch = []
    for name in ['cromnet.py', 'simulation.py', 'util.py']:
        source = native.parent/'sources/crom_offline'/name
        assert digest(source) == hashes[name]
        original = source.read_text()
        modified = original.replace('.float()', '.double()') if name == 'cromnet.py' else original
        (port/name).write_text(modified)
        patch.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                          fromfile=name, tofile=name+'.float64'))
    (out/'precision-port.patch').write_text(''.join(patch))
    online = native.parent/'sources/crom_online/online'
    online_hashes = {v['path']: v['sha256'] for v in evidence['sources']['crom_online']['files']}
    assert digest(online/'NonlinearSolver.py') == online_hashes['online/NonlinearSolver.py']
    sys.dont_write_bytecode = True  # Archived dependencies/native sources remain read-only.
    sys.path[:0] = [str(dependencies), str(port), str(online)]
    import torch, cromnet
    from NonlinearSolver import NonlinearSolver
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    # The native constructor owns initialization; retain its authored seed behavior.
    parent_record = resolve(ns/'b6parent/output/coordinate_data.json', ns)
    original = json.loads(parent_record.read_text())
    axis = np.linspace(0., 1., 128)
    xy = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    one_w = np.full(128, 1/127); one_w[[0, -1]] *= .5
    physical_w = np.outer(one_w, one_w).ravel()
    arrays, sources, provenance = {}, {}, []
    synth = jax.jit(data.synth_grid)
    for group, jobs, expected in [('train', a.train_jobs, 32), ('dev', a.dev_jobs, 8)]:
        rows = [dict(row, family='primary') for row in original['groups'][group]['sources']]
        extra_labels = []
        for job in job_names(jobs):
            rp = resolve(ns/job/'output/result.json', ns); record = json.loads(rp.read_text())
            assert record['status'] == 'complete_expanded_ladders_pending_independent_crosscheck'
            assert record['cohort'] == group and record['all_ladders_pass'] and not record['final_test_opened']
            spec = record['spec']; definitions = E.definitions(group, spec['start'], spec['count'])
            assert record['spec']['definitions'] == definitions
            assert [row['case'] for row in record['cases']] == [row['case'] for row in definitions]
            cache = rp.parent/record['coordinate_data']['file']
            assert cache.parent == rp.parent and digest(cache) == record['coordinate_data']['sha256']
            with np.load(cache) as v:
                assert str(v['cohort']) == group
                assert np.array_equal(v['case_ids'], np.repeat([row['case'] for row in definitions], 11))
                assert np.array_equal(v['times'], np.tile(data.TIMES, len(definitions)))
            provenance.append(dict(job=job, cohort=group, record_sha256=digest(rp), cache_sha256=digest(cache)))
            for row, definition in zip(record['cases'], definitions):
                assert all(row[k] == value for k, value in definition.items())
                assert row['complete'] and row['provisional_ladder_pass']
                last = row['levels'][-1]
                assert last['N'] == 512 and last['dt'] == .00015625
                rows.append(dict(case=row['case'], family=row['family'], file=str(rp.parent/last['file']), sha256=last['sha256']))
                extra_labels.append(row['case'])
        assert len(extra_labels) == expected and len(set(extra_labels)) == expected
        assert set(extra_labels) == {row['case'] for row in E.definitions(group)}
        assert len({row['case'] for row in rows}) == len(rows)
        fields, norms = [], []
        for row in rows:
            path = resolve(row['file'], ns); assert digest(path) == row['sha256']
            with np.load(path) as payload:
                coefficients = payload['coefficients']
                assert np.array_equal(payload['times'], data.TIMES)
            fields.append(np.asarray(synth(jnp.asarray(coefficients), jnp.asarray(axis))).reshape(11, -1))
            norms.append(np.linalg.norm(coefficients.reshape(11, -1), axis=1)/2)
        arrays[group] = np.concatenate(fields)
        arrays[group+'_norm'] = np.concatenate(norms)
        sources[group] = rows
    assert not ({row['case'] for row in sources['train']} & {row['case'] for row in sources['dev']})
    np.savez_compressed(out/'matched_data.npz', xy=xy, physical_w=physical_w, **arrays)
    save(out/'matched_sources.json', dict(primary_record_sha256=digest(parent_record), expanded=provenance, groups=sources))
    # Semantic source fingerprint survives archive moves and zip timestamps.
    fingerprint = hashlib.sha256(json.dumps({g: [(r['case'], r['sha256']) for r in rows]
                                             for g, rows in sources.items()}, sort_keys=True).encode()).hexdigest()
    config = dict(latent=a.latent, width=128, base_lr=1e-4, lr_multipliers=list(a.lr_multipliers),
                  scheduler_epochs=[100, 100], batch_size=4, order_seed=2026111987,
                  selection_primary_cases=[0, 3, 13, 20], selection_time_indices=[0, 5, 10],
                  projection_iterations=[20, 50], native_diff_threshold=1e-15,
                  native_step_threshold=1e-15, source_fingerprint=fingerprint)
    fmt = dict(i_dim=2, o_dim=1, npoints=len(xy), data_path='matched-owned-expanded-training-references')
    stats = dict(mean_x=xy.mean(0), std_x=xy.std(0), min_x=xy.min(0), max_x=xy.max(0),
                 mean_q=np.array([arrays['train'].mean()]), std_q=np.array([arrays['train'].std()]))
    model = cromnet.CROMnet(fmt, stats, torch.zeros((1, len(xy), 3)), 1e-4, 4, a.latent, 64, 6, 4,
                           False, True, .3, 30, False, list(a.lr_multipliers), [100, 100], 'explicit')
    model.setup('fit'); model = model.cuda().double()
    assert all(p.dtype == torch.float64 for p in model.parameters())
    assert all(b.dtype == torch.float64 for b in model.buffers())
    optimizers, schedulers = model.configure_optimizers(); optimizer, scheduler = optimizers[0], schedulers[0]
    tx = torch.as_tensor(xy, device='cuda')
    train = torch.as_tensor(arrays['train'], device='cuda'); dev = torch.as_tensor(arrays['dev'], device='cuda')
    tw = torch.as_tensor(physical_w, device='cuda')
    rng = np.random.default_rng(config['order_seed'])
    def inputs(q):
        return torch.cat([q[:, :, None], tx[None].expand(len(q), -1, -1)], 2)
    # Keep the native objective/gradient and decoder-Jacobian parity gates.
    batch = dict(encoder_input=inputs(train[:2]), q=train[:2, :, None])
    authored = model.training_step(batch, 0)['loss']; ga = torch.autograd.grad(authored, tuple(model.parameters()))
    manual = 1000*model.criterion(model(inputs(train[:2]))[0], train[:2, :, None])
    gb = torch.autograd.grad(manual, tuple(model.parameters()))
    gradient_error = max(float(torch.max(abs(x-y))) for x, y in zip(ga, gb)); assert gradient_error < 1e-12
    probe = torch.randn((19, a.latent+2), device='cuda')
    analytic, _ = model.decoder.computeJacobianFullAnalytical(probe)
    autodiff = torch.func.vmap(torch.func.jacrev(lambda x: model.decoder(x[None])[0]))(probe)
    jacobian_error = float(torch.max(abs(analytic-autodiff))); assert jacobian_error < 1e-10
    del authored, manual, ga, gb, probe, analytic, autodiff
    result = dict(meta=metadata, status='native_expanded_training', config=config,
                  source_commits={key: evidence['sources'][key]['commit'] for key in ['crom_offline', 'crom_online']},
                  native_source_result_sha256=digest(native), precision_patch_sha256=digest(out/'precision-port.patch'),
                  dependency_files_verified=verified_dependencies,
                  torch=dict(version=torch.__version__, device=torch.cuda.get_device_name(), dtype='float64'),
                  matched_data_sha256=digest(out/'matched_data.npz'), matched_sources_sha256=digest(out/'matched_sources.json'),
                  train_trajectories=len(sources['train']), dev_trajectories=len(sources['dev']),
                  normalization='Training normalization fitted on training fields only; scoring uses current-time reference physical L2',
                  classification='Pinned author CROM networks/optimizer/native GN; documented float64/manual-loop and new-data port',
                  reference_status='Primary qualified; expanded spectral ladders passed, independent FD tracked separately',
                  port_fixtures=dict(training_gradient_max_difference=gradient_error, native_analytic_jacobian_difference=jacobian_error),
                  final_test_opened=False, representation_admitted=False, pde_comparator_admitted=False,
                  history=[], selected_epoch=None, selected_sha256=None)
    # Fixed small diagnostic batch: pulse, smooth, ridge, two-pulse at t=0.
    diagnostic_cases=[0,3]+[next(i for i,row in enumerate(sources['train']) if row['family']==family)
                            for family in ['ridge','two_pulse']]
    diagnostic_indices=[11*i for i in diagnostic_cases]
    constant_mean_loss=1000*float(stats['std_q'][0])**2
    result.update(training_constant_mean_native_loss=constant_mean_loss,training_diagnostics=[],
        diagnostic_snapshot_indices=diagnostic_indices,
        diagnostic_cases=[sources['train'][i]['case'] for i in diagnostic_cases],
        projection_execution_is_not_convergence=True,physical_stationarity_threshold=1e-8,
        native_stop_reason_observability='Unmodified author solve returns codes only; stopping reason is not exposed')

    def training_diagnostic(epoch):
        """Read-only gradient/variation probe; no optimizer update or random draw."""
        was_training=model.training;model.eval()
        q=train[diagnostic_indices]
        with torch.enable_grad():
            prediction,latent,_=model(inputs(q))
            loss=1000*model.criterion(prediction,q[:,:,None])
            named=list(model.named_parameters())
            gradients=torch.autograd.grad(loss,tuple(p for _,p in named))
        with torch.no_grad():
            constant=1000*torch.mean((q-float(stats['mean_q'][0]))**2)
            record=dict(epoch=epoch,effective_lr=float(optimizer.param_groups[0]['lr']),
                batch_native_loss=float(loss.detach()),batch_constant_mean_native_loss=float(constant),
                full_training_constant_mean_native_loss=constant_mean_loss,
                output_mean=float(prediction.detach().mean()),output_std=float(prediction.detach().std(unbiased=False)),
                output_spatial_std_by_field=prediction.detach()[:,:,0].std(dim=1,unbiased=False).cpu().tolist(),
                latent_norm_by_field=torch.linalg.vector_norm(latent.detach(),dim=1).cpu().tolist(),
                latent_between_field_std_mean=float(latent.detach().std(dim=0,unbiased=False).mean()),
                gradient_l2=float(torch.sqrt(sum(torch.sum(g*g) for g in gradients))),
                all_gradients_finite=all(bool(torch.isfinite(g).all()) for g in gradients),
                layers={name:dict(weight_l2=float(p.norm()),max_abs=float(p.abs().max()),gradient_l2=float(g.norm()))
                        for (name,p),g in zip(named,gradients) if name.endswith('weight')})
        model.train(was_training)
        assert np.isfinite(record['batch_native_loss']) and record['all_gradients_finite']
        return record
    start_epoch = 0; best = np.inf
    if a.resume_job:
        rp = resolve(ns/a.resume_job/'output/result.json', ns); previous = json.loads(rp.read_text())
        assert previous['config'] == config and previous['source_commits'] == result['source_commits']
        cp = rp.parent/previous['last_checkpoint']; assert digest(cp) == previous['last_checkpoint_sha256']
        state = torch.load(cp, map_location='cuda', weights_only=False)
        assert state['config'] == config and state['epoch'] < a.epochs
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state = state['numpy_rng']; random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        start_epoch = state['epoch']; best = state['best_projected_error']
        result['history'] = [dict(row, origin_output=row.get('origin_output', str(rp.parent)))
                             for row in previous['history']]
        result['training_diagnostics'] = [dict(row,origin_output=row.get('origin_output',str(rp.parent)))
                                           for row in previous.get('training_diagnostics',[])]
        result['resume'] = dict(job=a.resume_job, result_sha256=digest(rp), checkpoint_sha256=digest(cp), epoch=start_epoch)
        if previous['selected_sha256']:
            selected = rp.parent/'crom_selected.pt'; assert digest(selected) == previous['selected_sha256']
            shutil.copy2(selected, out/'crom_selected.pt')
            result.update(selected_epoch=previous['selected_epoch'], selected_sha256=digest(out/'crom_selected.pt'))
    selection = [11*c+t for c in config['selection_primary_cases'] for t in config['selection_time_indices']]
    selection += [11*c+t for c in range(44, len(sources['dev'])) for t in config['selection_time_indices']]
    xf, wf = data.grid(256, True); fine_points = torch.as_tensor(xf, device='cuda'); fine_weights = torch.as_tensor(wf, device='cuda')
    fine_truth = {}
    for ci in sorted({i//11 for i in selection}):
        row = sources['dev'][ci]; path = resolve(row['file'], ns); assert digest(path) == row['sha256']
        with np.load(path) as payload:
            coefficients = payload['coefficients']
        truth = np.asarray(synth(jnp.asarray(coefficients), jnp.asarray(np.unique(xf[:, 0])))).reshape(11, -1)
        fine_truth[ci] = torch.as_tensor(truth.copy(), device='cuda')
    class Problem:
        def updateStateSample(self, z, decoder, points):
            return model.decoder(torch.cat([z.reshape(1, a.latent).expand(len(points), -1), points], 1))
        def getJacobianSample(self, z, decoder, points):
            self.jacobian_calls=getattr(self,'jacobian_calls',0)+1
            inp = torch.cat([z.reshape(1, a.latent).expand(len(points), -1), points], 1)
            jac, _ = model.decoder.computeJacobianFullAnalytical(inp)
            return jac[:, 0, :a.latent]
    problem = Problem(); solver = NonlinearSolver(problem, None, 1e-15, 1e-15)
    def evaluate():
        model.eval(); encoded_errors, projected = [], []
        with torch.no_grad():
            for lo in range(0, len(dev), 4):
                pred = model(inputs(dev[lo:lo+4]))[0][:, :, 0]
                error = torch.sqrt(torch.sum(tw*(pred-dev[lo:lo+4])**2, dim=1))
                error = error/torch.as_tensor(arrays['dev_norm'][lo:lo+4], device='cuda')
                assert torch.isfinite(error).all()
                encoded_errors.extend(error.cpu().tolist())
            for i in selection:
                row = dict(index=i, case=sources['dev'][i//11]['case'], time=float(data.TIMES[i%11]))
                try:
                    z = model.encoder(dev[i:i+1, :, None]).reshape(1, 1, a.latent)
                    errors, fits, native_exits = [], [], []
                    for iterations in [0, 20, 50]:
                        if iterations:
                            before=getattr(problem,'jacobian_calls',0)
                            z = solver.solve(z, dev[i, :, None], tx, step_size=1, max_iters=iterations, print_info=False)
                            used=problem.jacobian_calls-before
                            native_exits.append(dict(iteration_budget=iterations,jacobian_evaluations=used,
                                returned=True,returned_before_iteration_budget=used<iterations,
                                stop_reason='Not exposed by unmodified native solver; return alone does not certify physical stationarity'))
                        pred = problem.updateStateSample(z, None, fine_points)[:, 0]
                        error = torch.sqrt(torch.sum(fine_weights*(pred-fine_truth[i//11][i%11])**2))/arrays['dev_norm'][i]
                        assert torch.isfinite(error) and torch.isfinite(z).all()
                        errors.append(float(error)); fits.append(z.cpu().numpy())
                    jac = problem.getJacobianSample(z, None, tx)
                    residual = problem.updateStateSample(z, None, tx)-dev[i, :, None]
                    scale = torch.linalg.vector_norm(jac, dim=0).clamp(min=1e-14)
                    stationarity = float(torch.max(abs((jac.T@residual)[:, 0]/scale))/torch.linalg.vector_norm(dev[i]))
                    assert np.isfinite(stationarity)
                    row.update(passed_execution=True, independent_errors=errors, codes=np.stack(fits),
                               scaled_stationarity=stationarity,native_solve_observations=native_exits,
                               physical_stationarity_threshold=1e-8,physical_stationarity_pass=stationarity<=1e-8,
                               convergence_certified=False,
                               convergence_note='Physical stationarity is reported separately; native final-iteration update/stop reason is not exposed')
                except (RuntimeError, AssertionError) as exc:
                    row.update(passed_execution=False, error=repr(exc))
                projected.append(row)
        return encoded_errors, projected
    if start_epoch==0:
        initial=out/'crom_initial.pt'
        torch.save(dict(model=model.state_dict(),format=fmt,statistics=stats,config=config,epoch=0,
                        initialization='Authored constructor initialization; no training update'),initial)
        result.update(initial_checkpoint=initial.name,initial_checkpoint_sha256=digest(initial))
    result['training_diagnostics'].append(training_diagnostic(start_epoch))
    print('crom_training_diagnostic',result['training_diagnostics'][-1],flush=True)
    save(out/'result.json', result)
    for epoch in range(start_epoch+1, a.epochs+1):
        model.train(); order = rng.permutation(len(train)); losses = []
        epoch_lr = float(optimizer.param_groups[0]['lr'])
        for lo in range(0, len(order), 4):
            q = train[order[lo:lo+4]]; optimizer.zero_grad(set_to_none=True)
            loss = 1000*model.criterion(model(inputs(q))[0], q[:, :, None])
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step()
        if epoch in [1,2,5,10] or epoch%20==0 or epoch==a.epochs:
            diagnosis=training_diagnostic(epoch);result['training_diagnostics'].append(diagnosis)
            if epoch in [1,2,5,10]:
                path=out/f'crom_diagnostic_epoch{epoch:04d}.pt'
                torch.save(dict(model=model.state_dict(),format=fmt,statistics=stats,config=config,epoch=epoch,
                                role='Early training diagnostic, not optimizer-resumable checkpoint'),path)
                diagnosis.update(checkpoint=path.name,checkpoint_sha256=digest(path))
            print('crom_training_diagnostic',diagnosis,flush=True);save(out/'result.json',result)
        if epoch % 20 and epoch != a.epochs:
            continue
        encoded, projected = evaluate()
        valid = all(row['passed_execution'] for row in projected)
        metric = max(row['independent_errors'][-1] for row in projected) if valid else np.inf
        projection = out/f'projection_epoch{epoch:04d}.pt'
        torch.save(projected, projection)
        row = dict(epoch=epoch, lr=epoch_lr, mean_native_loss=float(np.mean(losses)),
                   native_loss_over_constant_mean_baseline=float(np.mean(losses))/constant_mean_loss,
                   encoded_full_errors=encoded, encoded_full_worst=max(encoded),
                   projected_subset_worst=float(metric) if valid else None,
                   failed_projection_fits=sum(not r['passed_execution'] for r in projected),
                   physical_stationarity_passes=sum(r.get('physical_stationarity_pass',False) for r in projected),
                   physical_stationarity_failures=sum(r['passed_execution'] and not r.get('physical_stationarity_pass',False) for r in projected),
                   failed_projection_fits_meaning='Execution exceptions/nonfinite values only, not a convergence count',
                   projection_file=projection.name, projection_sha256=digest(projection))
        result['history'].append(row)
        improved = metric < best
        if improved:
            best = metric; result['selected_epoch'] = epoch
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     numpy_rng=rng.bit_generator.state, python_rng=random.getstate(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                     format=fmt, statistics=stats, epoch=epoch, config=config,
                     best_projected_error=best, projected_evaluation=projected)
        path = out/f'crom_epoch{epoch:04d}.pt'; tmp = path.with_suffix('.tmp.pt')
        torch.save(state, tmp); tmp.replace(path)
        result.update(last_checkpoint=path.name, last_checkpoint_sha256=digest(path), last_epoch=epoch)
        if improved:
            shutil.copy2(path, out/'crom_selected.pt'); result['selected_sha256'] = digest(out/'crom_selected.pt')
        save(out/'result.json', result)
        print('crom_expanded', {k: v for k, v in row.items() if k != 'encoded_full_errors'}, flush=True)
    result.update(status='complete_native_expanded_training_diagnostic',
                  remaining='All-case native projection, independent scoring, actual4096 output and verified 2D viscous dynamics adapter')
    save(out/'result.json', result)


if __name__ == '__main__':
    main()
