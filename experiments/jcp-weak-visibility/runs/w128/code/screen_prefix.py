"""W0 projection-only screen, frozen representation prefixes and original dev6."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pickle
import subprocess

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import core as C
from core import e
import cohort_guard
import sep_common as sc
from run import save_json, npz, sha

HERE = Path(__file__).resolve().parent


def qualified_reference(u0, nu, cfg, output):
    n, dt = cfg['mesh_intervals'], cfg['dt']
    steps = int(round(cfg['horizon']/dt))
    stable_fom = C.make_reference(n, dt, steps, True)
    old_fom = C.make_reference(n, dt, steps, False)
    refs, records = [], []
    for level, (ntol, ltol) in enumerate(cfg['reference_tolerances']):
        ref, it, residual, trace = jax.block_until_ready(stable_fom(u0, nu, ntol, ltol))
        old, oi, ores, otr = jax.block_until_ready(old_fom(u0, nu, ntol, ltol))
        npz(output/f'reference-{level}.npz', fields=ref, iterations=it, residual=residual, newton_history=trace)
        npz(output/f'original-reference-{level}.npz', fields=old, iterations=oi, residual=ores, newton_history=otr)
        old_r = jax.jit(jax.vmap(lambda u, p: e.residual(u, p, nu, dt, n)))(ref[1:], ref[:-1])
        new_r = jax.jit(jax.vmap(lambda u, p: C.reference_residual(u, p, nu, dt, n, True)))(ref[1:], ref[:-1])
        scale = jnp.linalg.norm(ref[:-1], axis=1)
        old_relative = jnp.linalg.norm(old_r, axis=1)/scale
        primal_difference = jnp.linalg.norm(old_r-new_r, axis=1)/scale
        trajectory_difference = jnp.linalg.norm(ref-old, axis=1).max()/jnp.linalg.norm(ref[0])
        actions = []
        for ti in (1, steps):
            for seed in (1701, 1702, 1703):
                v = jnp.asarray(np.random.default_rng(seed).normal(size=ref[ti].shape))
                v = v/jnp.linalg.norm(v)
                jo = jax.jvp(lambda q: e.residual(q, ref[ti-1], nu, dt, n), (ref[ti],), (v,))[1]
                js = jax.jvp(lambda q: C.reference_residual(q, ref[ti-1], nu, dt, n, True), (ref[ti],), (v,))[1]
                actions.append(dict(time_index=ti, seed=seed, relative_difference=float(jnp.linalg.norm(js-jo)/jnp.linalg.norm(jo))))
        passed = bool(np.all(np.isfinite(ref)) and residual.max() <= 1e-10 and it.max() < 20 and
                      old_relative.max() <= 1e-10 and primal_difference.max() <= 1e-12 and
                      trajectory_difference <= 1e-8 and max(a['relative_difference'] for a in actions) <= 1e-12)
        records.append(dict(tolerances=[ntol, ltol], maximum_iterations=int(it.max()),
            maximum_relative_residual=float(residual.max()), original_residual_max=float(old_relative.max()),
            original_stable_primal_max=float(primal_difference.max()), original_stable_field_difference=float(trajectory_difference),
            action_checks=actions, passed=passed))
        npz(output/f'reference-arithmetic-{level}.npz', original_relative_residual=old_relative, primal_difference=primal_difference)
        refs.append(ref)
    delta = float(jnp.linalg.norm(refs[0]-refs[1], axis=1).max()/jnp.linalg.norm(refs[0][0]))
    info = dict(levels=records, tight_tighter_difference=delta,
                eligible=all(x['passed'] for x in records) and delta <= cfg['reference_max_disagreement'])
    save_json(output/'reference-controls.json', info)
    return refs[1], info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    assert jax.default_backend() == 'gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print('jax_backend=gpu x64=True precision=highest', flush=True)
    assert cfg['prefix_widths'] == [128, 256, 512] and cfg['case_ids'] == list(range(6))
    physical, guard = cohort_guard.regenerate(cfg, e.params_draw)
    save_json(out/'cohort-drawing.json', guard)
    assert guard['guard_passed']
    n = cfg['mesh_intervals']
    ckpt, rotation = HERE/'inputs/frozen.pkl', HERE/'inputs/rotation_R512.npz'
    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(ckpt, 'rb'))['params'])
    T = jnp.asarray(np.load(rotation)['T'][:, :512])
    xy = e.coords(n)
    feature = jax.jit(lambda p, x, t: sc.features(p, x) @ t)
    G = jax.block_until_ready(jnp.concatenate([feature(params, jnp.asarray(xy[k:k+2048]), T) for k in range(0, len(xy), 2048)]))
    assert G.dtype == jnp.float64 and bool(jnp.all(jnp.isfinite(G)))
    for start in range(0, 512, 64):
        npz(out/f'bank-columns-{start:03d}.npz', bank=G[:, start:start+64])
    # One stable ordered QR gives every nested prefix without normal-equation fitting.
    Qmass, Rmass = jax.block_until_ready(jnp.linalg.qr(G/n, mode='reduced'))
    qr_orthogonality = float(jnp.linalg.norm(Qmass.T @ Qmass-jnp.eye(512))/jnp.sqrt(512.))
    npz(out/'qr-factor.npz', Rmass=Rmass, orthogonality_defect=qr_orthogonality)
    rank_info = {}
    for width in cfg['prefix_widths']:
        gram = jax.block_until_ready((G[:, :width].T @ G[:, :width])/n**2)
        eigenvalues, vectors = np.linalg.eigh(np.asarray(gram))
        ranks = {str(t):int(np.sum(eigenvalues > t*eigenvalues[-1])) for t in (1e-10, 1e-12, 1e-14)}
        singular = np.linalg.svd(np.asarray(Rmass[:width, :width]), compute_uv=False)
        qr_ranks = {str(t):int(np.sum(singular**2 > t*singular[0]**2)) for t in (1e-10, 1e-12, 1e-14)}
        rank_info[str(width)] = dict(ranks=ranks, qr_ranks=qr_ranks,
            full_rank_all_thresholds=all(r == width for r in [*ranks.values(), *qr_ranks.values()]),
            qr_orthogonality_defect=float(jnp.linalg.norm(Qmass[:, :width].T @ Qmass[:, :width]-jnp.eye(width))/jnp.sqrt(float(width))),
            eigenvalue_min=float(eigenvalues[0]), eigenvalue_max=float(eigenvalues[-1]),
            physical_basis_condition=float(singular[0]/singular[-1]),
            projection_algorithm='ordered mass QR, triangular solve with every original prefix coefficient; no truncation')
        npz(out/f'gram-r{width}.npz', gram=gram, eigenvalues=eigenvalues, eigenvectors=vectors, qr_singular_values=singular)
    environment = dict(config=cfg, checkpoint_sha256=sha(ckpt), rotation_sha256=sha(rotation),
        source_sha256={str(p.relative_to(HERE)):sha(p) for p in [HERE/'screen_prefix.py', HERE/'core.py', HERE/'run.py', HERE/'cohort_guard.py']},
        job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip(),
        backend=jax.default_backend(), x64=True, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'])
    save_json(out/'environment.json', environment)
    summary = dict(scope='post-pilot frozen-prefix projection-only screen against same-grid same-step target',
                   rank_info=rank_info, qr_orthogonality_defect=qr_orthogonality,
                   cases={}, frozen_widths=cfg['prefix_widths'], gates_provisional_until_R128_reproduction_audit=True)
    for case_id in cfg['case_ids']:
        case_dir = out/f'case-{case_id:02d}'
        case_dir.mkdir()
        print(f'projection case={case_id} start', flush=True)
        u0 = jnp.asarray(e.initial(n, physical[case_id]))
        reference, controls = qualified_reference(u0, float(physical[case_id, -1]), cfg, case_dir)
        norm0 = jnp.linalg.norm(reference[0])
        case = dict(physical_parameters=physical[case_id], reference=controls, prefixes={})
        previous_error = None
        for width in cfg['prefix_widths']:
            basis = G[:, :width]
            rhs = Qmass[:, :width].T @ (reference.T/n)
            coefficients = jax.scipy.linalg.solve_triangular(Rmass[:width, :width], rhs, lower=False).T
            projected = coefficients @ basis.T
            qr_projected = (Qmass[:, :width] @ rhs).T*n
            qr_field_parity = float(jnp.linalg.norm(projected-qr_projected, axis=1).max()/norm0)
            error = jax.block_until_ready(jnp.linalg.norm(projected-reference, axis=1)/norm0)
            orthogonality = jnp.linalg.norm((projected-reference) @ basis/n**2)/(jnp.linalg.norm(reference)*jnp.linalg.norm(basis)/n**2+1e-300)
            nesting = previous_error is None or bool(jnp.max(error-previous_error) <= 1e-10)
            record = dict(initial_error=float(error[0]), maximum_all_times_error=float(error.max()),
                          evolved_maximum_error=float(error[1:].max()), errors=np.asarray(error),
                          orthogonality_defect=float(orthogonality), qr_field_parity=qr_field_parity,
                          nested_projection_check=nesting)
            case['prefixes'][str(width)] = record
            npz(case_dir/f'projection-r{width}.npz', coeff=coefficients, error=error, orthogonality_defect=orthogonality)
            previous_error = error
            print(f'case={case_id} R={width} initial={float(error[0]):.8g} alltime={float(error.max()):.8g} nested={nesting}', flush=True)
        summary['cases'][str(case_id)] = case
        save_json(out/'summary.json', summary)
    all_references = all(c['reference']['eligible'] for c in summary['cases'].values())
    all_nested = all(c['prefixes'][str(w)]['nested_projection_check'] for c in summary['cases'].values() for w in cfg['prefix_widths'])
    prefix_controls = {str(w):bool(rank_info[str(w)]['qr_orthogonality_defect'] <= 1e-10 and
        all(c['prefixes'][str(w)]['orthogonality_defect'] <= 1e-9 and c['prefixes'][str(w)]['qr_field_parity'] <= 1e-10 and
            c['prefixes'][str(w)]['nested_projection_check'] and np.all(np.isfinite(c['prefixes'][str(w)]['errors']))
            for c in summary['cases'].values())) for w in cfg['prefix_widths']}
    all_orthogonal = all(prefix_controls.values())
    summary['aggregates'] = {}
    for width in cfg['prefix_widths']:
        records = [c['prefixes'][str(width)] for c in summary['cases'].values()]
        summary['aggregates'][str(width)] = dict(worst_initial=max(x['initial_error'] for x in records),
            worst_all_times=max(x['maximum_all_times_error'] for x in records),
            median_all_times=float(np.median([x['maximum_all_times_error'] for x in records])))
    base = summary['aggregates']['128']
    baseline_eligible = all_references and rank_info['128']['full_rank_all_thresholds'] and prefix_controls['128']
    summary['provisional_admission'] = {}
    for width in (256, 512):
        rec = summary['aggregates'][str(width)]
        summary['provisional_admission'][str(width)] = bool(baseline_eligible and prefix_controls[str(width)] and
            rank_info[str(width)]['full_rank_all_thresholds'] and
            rec['worst_initial'] <= cfg['projection_admission_ratio']*base['worst_initial'] and
            rec['worst_all_times'] <= cfg['projection_admission_ratio']*base['worst_all_times'])
    summary.update(all_references_eligible=all_references, all_nested=all_nested, all_orthogonal=all_orthogonal,
                   baseline_eligible_before_reproduction=baseline_eligible, per_prefix_projection_controls=prefix_controls,
                   environmental_acceptance_pending=True)
    save_json(out/'summary.json', summary)
    print('projection_completed=True', flush=True)


if __name__ == '__main__':
    main()
