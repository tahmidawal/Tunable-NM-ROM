"""Common validation-only evaluator for separable Burgers-3D head architectures.

See B3D-ARCH-DESIGN.md. Model modules cannot access validation through the
training API. The physical RHS is used for diagnostics only; online ROMs retain
the inherited weak objective. Large arrays are explicit compiled arguments.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

import b3d_common as b3
import b3d_repair as repair


def tree_hash(tree):
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        a = np.asarray(leaf)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def make_shared(bank_params, r, a):
    # Anchor is derived from the already trained neural linear skip, never POD.
    skip_q = np.asarray(bank_params['h_lin']) @ r.T
    u, ru = np.linalg.qr(skip_q.T, mode='reduced')
    assert np.linalg.matrix_rank(ru) == ru.shape[0]
    signs = np.where(np.diag(ru) < 0, -1., 1.)
    u *= signs[None, :]
    center = np.mean(a, axis=0)
    z = (a - center) @ u
    scale = np.sqrt(np.mean(z*z, axis=0))
    assert np.all(scale > 1e-10)
    shared = dict(anchor=u, center=center, latent_scale=scale,
                  output_scale=np.asarray(np.sqrt(np.mean((a-center)**2))))
    return shared, z


def rel_errors(apply, p, frozen, z, target, floor2, norm2):
    residual = apply(p, frozen, z) - target
    return jnp.sqrt((jnp.sum(residual*residual, axis=-1) + floor2) / norm2)


def parse_seeds(raw):
    seeds = [int(x) for x in raw.split(',')]
    assert seeds and len(set(seeds)) == len(seeds) and all(0 <= x < 2**31 for x in seeds)
    return seeds


def train_model(module, shared, z0, train, seed=200, steps=60000, lr=3e-4, batch=4096):
    """Same global relative field-MSE and optimizer schedule across all arms."""
    assert module.MODE in {'codes', 'encoder'} and steps > 0
    kp, kb = jax.random.split(jax.random.PRNGKey(seed))
    p, frozen = module.init(kp, shared)
    frozen = jax.tree_util.tree_map(jnp.asarray, frozen)
    frozen_hash = tree_hash(frozen)
    z = jnp.asarray(z0)
    a, floor, norm = [jnp.asarray(train[k]) for k in ('target', 'perpendicular2', 'norm2')]
    inv = jnp.asarray(1. / np.mean(train['norm2']))
    nb = min(batch, len(z0))
    variables = (p, z) if module.MODE == 'codes' else p
    schedule = optax.warmup_cosine_decay_schedule(0., lr, min(500, steps//10+1), steps, lr*.01)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(variables)

    def loss(v, f, aa, ids, inverse):
        if module.MODE == 'codes':
            pp, zz = v
            zz = zz[ids]
        else:
            pp = v
            zz = module.encode(pp, f, aa)
        residual = module.apply(pp, f, zz) - aa
        return jnp.mean(jnp.sum(residual**2, axis=1)) * inverse

    @jax.jit
    def update(v, state, f, aa, inverse, key):
        ids = jax.random.choice(key, aa.shape[0], shape=(nb,), replace=False)
        value, grad = jax.value_and_grad(loss)(v, f, aa[ids], ids, inverse)
        updates, state = optimizer.update(grad, state, v)
        return optax.apply_updates(v, updates), state, value

    metrics = jax.jit(lambda pp, ff, zz, aa, floor_, norm_: rel_errors(
        module.apply, pp, ff, zz, aa, floor_, norm_))

    def assigned(v):
        if module.MODE == 'codes':
            return v
        return v, module.encode(v, frozen, a)

    initial_p, initial_z = assigned(variables)
    initial_q = np.asarray(module.apply(initial_p, frozen, initial_z))
    expected_q = shared['center'] + z0 @ shared['anchor'].T
    initial_code_difference = float(np.max(np.abs(np.asarray(initial_z)-z0)))
    initial_field_difference = float(np.max(np.abs(initial_q-expected_q)))
    np.testing.assert_allclose(initial_z, z0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(initial_q, expected_q, rtol=1e-12, atol=1e-12)
    initial_errors = np.asarray(metrics(initial_p, frozen, initial_z, a, floor, norm))
    history = []
    begin = time.time()
    for step in range(steps):
        kb, key = jax.random.split(kb)
        variables, opt_state, value = update(variables, opt_state, frozen, a, inv, key)
        if step == 0 or (step + 1) % 5000 == 0 or step + 1 == steps:
            pp, zz = assigned(variables)
            errors = np.asarray(metrics(pp, frozen, zz, a, floor, norm))
            assert np.isfinite(float(value)) and np.all(np.isfinite(errors))
            row = dict(step=step+1, coefficient_loss=float(value),
                       reconstruction=repair.summarize(errors), seconds=time.time()-begin)
            history.append(row)
            b3.log(f"{module.NAME} seed={seed} step={step+1}/{steps} train_mean={errors.mean():.6e}")
    p, z = assigned(variables)
    assert tree_hash(frozen) == frozen_hash
    final = np.asarray(metrics(p, frozen, z, a, floor, norm))
    return p, frozen, np.asarray(z), dict(
        seed=seed, steps=steps, lr=lr, batch=nb, norm='global', mode=module.MODE,
        initial=repair.summarize(initial_errors), final=repair.summarize(final),
        global_mse_initial=float(np.sum(initial_errors**2 * train['norm2'])/np.sum(train['norm2'])),
        global_mse_final=float(np.sum(final**2 * train['norm2'])/np.sum(train['norm2'])),
        trainable_parameter_count=sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(p)),
        optimized_code_count=int(z.size) if module.MODE == 'codes' else 0,
        frozen_hash=frozen_hash, history=history, seconds=time.time()-begin,
        initialization_code_max_difference=initial_code_difference,
        initialization_field_max_difference=initial_field_difference)


def geometry_metrics(jac, residual, field_residual2, velocity, velocity_floor2, velocity_norm2,
                     rank_rtol=1e-10):
    """SVD range projection: no ridge and no dependence on a latent norm."""
    left, singular, _ = np.linalg.svd(jac, full_matrices=False)
    cutoff = rank_rtol * singular[0] if len(singular) else 0.
    rank = int(np.sum(singular > cutoff))
    left = left[:, :rank]
    invariant = np.linalg.norm(left.T @ residual) / np.sqrt(max(field_residual2, 1e-300))
    missing = velocity - left @ (left.T @ velocity)
    absolute = np.sqrt(np.dot(missing, missing) + velocity_floor2)
    return dict(rank=rank, singular_values=singular, rank_cutoff=cutoff,
                singular_ratio=float(singular[-1]/singular[0]) if singular[0] > 0 else 0.,
                invariant_stationarity=float(invariant),
                tangent_absolute=float(absolute), velocity_norm=float(np.sqrt(velocity_norm2)),
                tangent_relative=float(absolute/np.sqrt(velocity_norm2)) if velocity_norm2 > 0 else None,
                bank_velocity_relative=float(np.sqrt(velocity_floor2/velocity_norm2)) if velocity_norm2 > 0 else None)


def evaluate(module, p, frozen, ztrain, val, g, q, r, budgets=(400, 800), on_stage=None):
    apply = lambda pf, z: module.apply(pf[0], pf[1], z)
    start_ids = np.random.default_rng(5).choice(len(ztrain), 7, replace=False)
    starts = np.concatenate([np.zeros((1, ztrain.shape[1])), ztrain[start_ids]])
    stages = repair.fit_states((p, frozen), val['target'], val['perpendicular2'],
                              val['norm2'], starts, budgets=budgets, apply_fn=apply, on_stage=on_stage)
    q_apply = jax.jit(module.apply)
    jac_apply = jax.jit(jax.vmap(jax.jacfwd(module.apply, argnums=2), in_axes=(None, None, 0)))
    for stage in stages:
        zz = jnp.asarray(stage['z'])
        qval = np.asarray(q_apply(p, frozen, zz))
        coeff = np.linalg.solve(r, qval.T).T
        field_error = coeff @ g.T - val['fields']
        direct = np.linalg.norm(field_error, axis=1) / np.sqrt(val['norm2'])
        assert np.max(np.abs(direct-stage['error'])) < 1e-10
        jacs = np.asarray(jac_apply(p, frozen, zz))
        geometric = []
        for i, jac in enumerate(jacs):
            residual = qval[i]-val['target'][i]
            full_r2 = np.dot(residual, residual)+val['perpendicular2'][i]
            geometric.append(geometry_metrics(jac, residual, full_r2, val['velocity_target'][i],
                                             val['velocity_perpendicular2'][i], val['velocity_norm2'][i]))
        # Independent full-field Jacobian/range checks on the hardest local fit.
        idx = int(np.argmax([x['invariant_stationarity'] for x in geometric]))
        jfield = g @ np.linalg.solve(r, jacs[idx])
        eta_field = np.linalg.norm(jfield.T@field_error[idx]) / (np.linalg.norm(jfield)*np.linalg.norm(field_error[idx])+1e-300)
        assert abs(eta_field-stage['optimality'][idx]) < 1e-9
        assert np.linalg.norm(jfield-q@jacs[idx]) / max(np.linalg.norm(jfield), 1e-300) < 1e-10
        stage.update(summary=repair.summarize(stage['error']), geometry=geometric,
                     nonstationary=int(np.sum(stage['optimality']>1e-6)),
                     outliers_above_15pct=int(np.sum(stage['error']>.15)),
                     tangent_summary=repair.summarize([x['tangent_relative'] for x in geometric if x['tangent_relative'] is not None]),
                     direct_error_difference_max=float(np.max(np.abs(direct-stage['error']))))
        stage['groups'] = {}
        for name, mask in [('initial', val['sid'] % 51 == 0), ('later', val['sid'] % 51 > 0)] + [
                (f'blobs_{b}', val['blob_count'] == b) for b in (1,2,3)]:
            stage['groups'][name] = repair.summarize(stage['error'][mask])
            stage['groups'][name]['tangent'] = repair.summarize([
                geometric[i]['tangent_relative'] for i in np.flatnonzero(mask)
                if geometric[i]['tangent_relative'] is not None])
        b3.log(f"{module.NAME} fit={stage['budget']} mean={stage['summary']['mean']:.6e} worst={stage['summary']['worst']:.6e} nonstationary={stage['nonstationary']}")
    return stages, start_ids


def check_model(module, p, frozen, z):
    probe = np.asarray(z[:8])
    direct = np.asarray(module.apply(p, frozen, jnp.asarray(probe)))
    independent = module.apply_np(p, frozen, probe)
    assert direct.dtype == np.float64
    relative = np.linalg.norm(direct-independent)/max(np.linalg.norm(direct), 1e-300)
    assert relative < 1e-11
    single = np.stack([np.asarray(module.apply(p, frozen, jnp.asarray(zz))) for zz in probe])
    np.testing.assert_allclose(single, direct, rtol=1e-11, atol=1e-11)
    direction = np.random.default_rng(41).normal(size=probe[0].shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-5 * max(1., np.linalg.norm(probe[0]))
    finite = (module.apply_np(p, frozen, probe[0]+eps*direction)-module.apply_np(p, frozen, probe[0]-eps*direction))/(2*eps)
    _, tangent = jax.jvp(lambda zz: module.apply(p, frozen, zz), (jnp.asarray(probe[0]),), (jnp.asarray(direction),))
    jvp_error = np.linalg.norm(np.asarray(tangent)-finite)/max(np.linalg.norm(tangent), 1e-12)
    assert jvp_error < 1e-6
    return dict(numpy_relative=relative, jvp_relative=float(jvp_error), batch_single_parity=True)


def main():
    assert jax.default_backend() == 'gpu'
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    module_name = os.environ.get('MODEL', 'b3d_arch_baseline')
    assert module_name.startswith('b3d_arch_') and module_name.replace('_','').isalnum()
    module = importlib.import_module(module_name)
    out = Path(os.environ['OUT']); out.parent.mkdir(parents=True, exist_ok=True)
    source = Path(os.environ['CKPT'])
    bank_params, _, cfg = b3.load_pkl(source)
    assert (cfg['N'], cfg['k'], cfg['r'], cfg['seed']) == (33, 32, 128, 0)
    report = dict(complete=False, kind='b3d_architecture', config=dict(
        model=module_name, name=module.NAME, model_config=getattr(module,'CONFIG',{}),
        source_config=cfg, source_sha256=b3.sha256_file(source),
        model_sha256=b3.sha256_file(module.__file__), commit=os.environ['COMMIT'],
        slurm_job=os.environ['SLURM_JOB_ID'], node=os.environ.get('SLURMD_NODENAME'),
        gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), x64=bool(jax.config.x64_enabled),
        matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
        test_table_opened=False, initialization='common_source_skip_anchor_linear', rank_rtol=1e-10))

    def save():
        tmp = out.with_suffix('.writing')
        tmp.write_text(json.dumps(report, default=repair.jsonable, allow_nan=False, indent=1))
        tmp.replace(out)

    save()
    assert cfg['max_snaps'] == 8192
    seeds = parse_seeds(os.environ.get('OPT_SEEDS', '200,201'))
    report['config']['optimizer_seeds'] = seeds
    tables = b3.get_tables(os.environ['TABLE_DIR'], 576, 8, 0, 1, with_test=False)
    reference = b3.load_param_table(os.environ['REFERENCE_TABLE'])
    report['provenance'] = repair.validate_parameter_manifest(tables['train'], reference, cfg['table_sha256'])
    g, q, r, data, info = repair.extract(bank_params, cfg, tables['train'], 33, out.parent, include_velocity=True)
    report['bank'] = info
    train, val = data['train'], data['validation']
    assert len(train['sid']) == 8192 and len(val['sid']) == 256
    np.testing.assert_array_equal(val['sid'],
        (np.arange(512,576)[:,None]*51+np.asarray([0,10,25,50])).ravel())
    shared, z0 = make_shared(bank_params, r, train['target'])
    report['config']['shared_hash'] = tree_hash(shared)
    report['validation_states'] = {k:v for k,v in val.items() if k != 'fields'}
    report['bank_error'] = repair.summarize(np.sqrt(val['perpendicular2']/val['norm2']))
    report['runs'] = []
    save()
    for seed in seeds:
        p, frozen, codes, training = train_model(module, shared, z0, train, seed=seed,
            steps=int(os.environ.get('ARCH_STEPS','60000')), lr=float(os.environ.get('ARCH_LR','3e-4')),
            batch=int(os.environ.get('ARCH_BATCH','4096')))
        checks = check_model(module, p, frozen, codes)
        checkpoint = out.parent / f'{module.NAME}_seed{seed}.pkl'
        payload = dict(kind='b3d_arch_checkpoint', model=module_name,
                       trainable=p, frozen=frozen, bank_params=bank_params, r=r,
                       codes=codes, cfg=cfg, training=training, source=report['config'])
        with checkpoint.open('wb') as stream:
            pickle.dump(jax.tree_util.tree_map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, payload), stream)
        with checkpoint.open('rb') as stream:
            loaded = pickle.load(stream)
        np.testing.assert_array_equal(module.apply(p,frozen,jnp.asarray(codes[:8])),
                                      module.apply(loaded['trainable'],loaded['frozen'],jnp.asarray(codes[:8])))
        assert tree_hash(loaded['bank_params']) == tree_hash(bank_params)
        row = dict(seed=seed, training=training, model_checks=checks, checkpoint_sha256=b3.sha256_file(checkpoint))
        report['runs'].append(row); save()
        def save_raw_stage(stage):
            row.setdefault('unchecked_fit_stages', []).append(stage)
            save()
        stages, start_ids = evaluate(module, p, frozen, codes, val, g, q, r, on_stage=save_raw_stage)
        row.pop('unchecked_fit_stages', None)
        row.update(fits=stages, start_training_indices=start_ids,
                   start_snapshot_ids=train['sid'][start_ids],
                   budget_change_max=float(np.max(np.abs(stages[-1]['error']-stages[0]['error'])/stages[0]['error'])))
        if module.MODE == 'encoder':
            encoded = module.encode(p, frozen, jnp.asarray(val['target']))
            row['direct_encoder_error'] = np.asarray(rel_errors(module.apply,p,frozen,encoded,
                jnp.asarray(val['target']),jnp.asarray(val['perpendicular2']),jnp.asarray(val['norm2'])))
        if hasattr(module, 'diagnostics'):
            row['architecture_diagnostics'] = module.diagnostics(p, frozen, jnp.asarray(stages[-1]['z']))
        save()
    assert report['config']['source_sha256'] == b3.sha256_file(source)
    report['complete'] = True
    save()
    b3.log(f'DONE {out}')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        output = Path(os.environ.get('OUT', 'architecture_failure.json'))
        if output.exists():
            failed = json.loads(output.read_text())
            failed.update(complete=False, failure_type=type(error).__name__,
                          failure_message=str(error), failure_traceback=traceback.format_exc())
            output.write_text(json.dumps(failed, default=repair.jsonable, allow_nan=False, indent=1))
        raise
