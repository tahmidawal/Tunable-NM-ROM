"""Untimed A10 confirmation; frozen models/cohorts, coefficient-space fits only.

This driver does not call a timed solver or retrain a checkpoint. It preserves the
original eight nearest-code multistart result and a separately labelled safeguarded
best-found value seeded from retained solved latents and the preceding q rung.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import scipy.linalg

import arms as A
import core as C
import pbh_core as K
import plin_core as L
import pbh_audit_np as N
import sep_common as sc
from plin_solve import resolve, training_cohort


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_projected(params, Rg, Z, T, V, cfg, seed_rows, previous):
    """Retain each original start, then independently fit additional feasible starts."""
    R = Rg.shape[0]
    q = V.shape[1]
    Q = np.linalg.qr(V, mode='reduced')[0] if q else np.empty((R, 0))
    orth = float(np.linalg.norm(Q.T @ Q - np.eye(q)))
    assert orth < 1e-8
    Qj, Rj = jnp.asarray(Q), jnp.asarray(Rg)

    # The frozen parameters remain explicit arguments, avoiding captured large arrays.
    def residual(z, target, metric, proj, p):
        raw = metric @ sc.head(p, z) - target
        return raw - proj @ (proj.T @ raw)

    lm = A.make_stationary_lm(residual, cfg['recon_budget'],
                             gtol=cfg['stationarity_tolerance'], linear='gj')
    fit = jax.jit(jax.vmap(lambda z, t, r, v, p: lm(z, (t, r, v, p), 0.),
                          in_axes=(0, 0, None, None, None)))
    res_jac = jax.jit(lambda z, t, r, v, p: (
        residual(z, t, r, v, p), jax.jacfwd(residual)(z, t, r, v, p)))
    H = np.asarray(jax.jit(jax.vmap(sc.head, in_axes=(None, 0)))(params, jnp.asarray(Z)))
    Hr = H @ Rg.T
    Hr -= (Hr @ Q) @ Q.T
    Tp = T - (T @ Q) @ Q.T
    score = np.sum(Hr * Hr, axis=1)[None, :] - 2 * Tp @ Hr.T
    picks = np.argsort(score, axis=1)[:, :cfg['recon_starts']]
    rows, zs = [], []
    for case in range(len(T)):
        initial = [*Z[picks[case]]]
        labels = [f'nearest_code_{i}' for i in picks[case]]
        for row in seed_rows:
            if row['case'] == case:
                initial.append(np.asarray(row['latent']))
                labels.append('retained_solve:' + row['name'])
        if previous is not None:
            initial.append(previous[case])
            labels.append('previous_q_best')
        initial = np.stack(initial)
        targets = np.repeat(T[case:case + 1], len(initial), axis=0)
        out = jax.device_get(fit(jnp.asarray(initial), jnp.asarray(targets), Rj, Qj, params))
        allz, norms, iterations, reasons, gradients = [np.asarray(x) for x in out]
        selected = int(np.argmin(norms))
        original = int(np.argmin(norms[:cfg['recon_starts']]))
        z = allz[selected]
        residual_value, jacobian = jax.device_get(res_jac(jnp.asarray(z), jnp.asarray(T[case]), Rj, Qj, params))
        h = np.asarray(sc.head(params, jnp.asarray(z)))
        delta = T[case] - Rg @ h
        correction = scipy.linalg.solve_triangular(Rg, Q @ (Q.T @ delta)) if q else np.zeros(R)
        coefficient = h + correction
        zs.append(z)
        rows.append(dict(case=case, selected=selected, original_selected=original,
                         start_labels=labels, initial_latents=initial.tolist(),
                         all_latents=allz.tolist(), residual_norms=norms.tolist(),
                         iterations=iterations.tolist(), reasons=reasons.tolist(),
                         normalized_gradients=gradients.tolist(),
                         selected_latent=z.tolist(), selected_coefficient=coefficient.tolist(),
                         selected_residual=np.asarray(residual_value).tolist(),
                         selected_jacobian=np.asarray(jacobian).tolist(),
                         selected_residual_norm=float(norms[selected]),
                         original_residual_norm=float(norms[original])))
    return rows, np.stack(zs), Q, orth


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    here = Path(a.config).resolve().parent
    cfg = json.loads(Path(a.config).read_text())
    history_path = resolve(here, cfg['history'])
    history = json.loads(history_path.read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    assert jax.default_backend() == 'gpu'
    result = dict(schema='p-linear-oracle-confirmation-v1', config=cfg, complete=False,
                  backend=jax.default_backend(), x64=jax.config.x64_enabled,
                  matmul_precision=str(jax.config.jax_default_matmul_precision),
                  source_commit=os.environ.get('SOURCE_COMMIT', 'local-smoke'),
                  job_id=os.environ.get('SLURM_JOB_ID'), jax_version=jax.__version__,
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip(),
                  history_sha256=digest(history_path), smoke=a.smoke,
                  interpretation='Finite multistart optimizer upper bound on best achievable representation error; no global minimum claim.',
                  timed_solver_changed=False, retraining=False, final_cohort_opened=False, panels=[])
    def save():
        result['elapsed_seconds'] = time.monotonic() - start
        K.dump(out / 'result.json', result)
    save()
    original_cfg = history['256']['config']
    dev = np.concatenate((C.source_params(original_cfg['eval_seed'], original_cfg['eval_count']),
                          C.source_params(original_cfg['fresh_seed'], original_cfg['fresh_count'])))
    np.testing.assert_allclose(dev, history['256']['cohort']['parameters'], atol=1e-15, rtol=0)
    if a.smoke:
        dev = dev[:2]
    for model in original_cfg['models']:
        if a.smoke and model['id'] != 'new_K32':
            continue
        mid = model['id']
        ck, bp = resolve(here, model['checkpoint']), resolve(here, model['basis'])
        params, codes, _ = sc.load_pkl(ck)
        params = jax.tree_util.tree_map(jnp.asarray, params)
        codes, basis = np.asarray(codes), dict(np.load(bp))
        original_checkpoint = next(x for x in history['256']['checkpoints'] if x['id'] == mid)
        assert digest(ck) == original_checkpoint['checkpoint_sha256']
        assert digest(bp) == original_checkpoint['basis_sha256']
        if a.smoke:
            # The full-cohort job regenerates the original extension; smoke checks only
            # the frozen retained prefix and a full-bank endpoint, avoiding training data.
            Cfull = basis['coefficient_directions']
            direction_info = dict(smoke_retained_prefix_only=True)
        else:
            Cfull, direction_info = L.extend_basis(params, codes, basis, training_cohort(model['training']),
                                                  original_cfg['training_intervals'])
            assert direction_info['extension_orthonormality_error'] < 1e-7
        rank = int(params['h_lin'].shape[1])
        print('DIRECTIONS', mid, direction_info, flush=True)
        for n in ([64] if a.smoke else cfg['intervals']):
            G = K.bank_of(params, n)
            Rg, rank_info = K.bank_r(G)
            assert rank_info['rank_valid']
            U = K.fields(dev, n)
            T, perp2, nu2 = [np.asarray(x) for x in K.project_targets(G, Rg, U)]
            Rg = np.asarray(Rg)
            old = history[str(n)] if not a.smoke else None
            panel = dict(intervals=n, model=mid, R=rank, K=codes.shape[1],
                         checkpoint_sha256=digest(ck), basis_sha256=digest(bp),
                         cohort_parameters=dev.tolist(), direction_info=direction_info,
                         original_result_sha256=old['result_sha256'] if old else None,
                         original_direction_sha256=(next(x['directions_sha256'] for x in old['directions'] if x['model'] == mid) if old else None),
                         regenerated_directions_sha256=L.sha_array(Cfull), rank=rank_info,
                         bank_floor=np.sqrt(perp2 / nu2).tolist(), rungs=[])
            metric_file = f'n{n}_{mid}_metric.npz'
            np.savez_compressed(out / metric_file, Rg=Rg, T=T, perp2=perp2, nu2=nu2,
                                Cfull=Cfull, gt_u=np.asarray(U @ G), gram=np.asarray(G.T @ G))
            panel['metric_artifact'] = metric_file
            panel['metric_sha256'] = digest(out / metric_file)
            result['panels'].append(panel)
            qvalues = ([0, 32, rank] if a.smoke else [q for q in original_cfg['ladder_q']
                       if q <= rank and (model['role'] != 'control' or q in original_cfg['control_q'])])
            previous = None
            for q in qvalues:
                seeds = [x for x in old['solved_states'] if x['model'] == mid and x['q'] == q
                         and '_ccrule' not in x['name']] if old else []
                replay = []
                if q < rank:
                    for seed in seeds:
                        h = np.asarray(sc.head(params, jnp.asarray(seed['latent'])))
                        coef = h + Cfull[:, :q] @ np.asarray(seed['correction_coefficients'])
                        c = seed['case']
                        projected = Rg @ coef - T[c]
                        error = float(np.sqrt((projected @ projected + perp2[c]) / nu2[c]))
                        replay.append(dict(name=seed['name'], case=c,
                                           original_error=seed['same_grid_error'], regenerated_error=error))
                if q == rank:
                    # The head is redundant at full span. A direct free-coefficient
                    # projection avoids an ill-identified latent fit to QR roundoff.
                    coefficient = scipy.linalg.solve_triangular(Rg, T.T).T
                    rows = [dict(case=c, selected_coefficient=coefficient[c].tolist(),
                                 selected_residual_norm=0., original_residual_norm=0.,
                                 endpoint='direct_free_bank_projection', iterations=[0], reasons=[4])
                            for c in range(len(dev))]
                    Q, orth = np.eye(rank), 0.
                else:
                    rows, previous, Q, orth = fit_projected(params, Rg, codes, T,
                                                           Rg @ Cfull[:, :q], cfg, seeds, previous)
                coefs = np.asarray([x['selected_coefficient'] for x in rows])
                fields = jnp.asarray(coefs) @ G.T
                physical = np.asarray(jnp.linalg.norm(fields - U, axis=1) / jnp.linalg.norm(U, axis=1))
                for c, row in enumerate(rows):
                    row['error'] = float(np.sqrt((row['selected_residual_norm'] ** 2 + perp2[c]) / nu2[c]))
                    row['original_starts_error'] = float(np.sqrt((row['original_residual_norm'] ** 2 + perp2[c]) / nu2[c]))
                    row['decoded_field_error'] = float(physical[c])
                    row['floor'] = float(np.sqrt(perp2[c] / nu2[c]))
                rung = dict(q=q, cases=rows, projector_orthogonality_error=orth,
                            retained_solve_replay=replay,
                            worst=max(x['error'] for x in rows), median=float(np.median([x['error'] for x in rows])),
                            direct_decode_metric_discrepancy=float(np.max(np.abs(physical - [x['error'] for x in rows]))))
                panel['rungs'].append(rung)
                save()
                print('ORACLE', n, mid, q, rung['worst'], 'decoded_defect', rung['direct_decode_metric_discrepancy'], flush=True)
            del G, U
            jax.clear_caches()
            gc.collect()
    result['complete'] = True
    save()
    print('ALL-DONE', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
