"""Stage 1 of the low-viscosity panel job: the checkpoint's OWN correction directions.

The incumbent panel transferred `directions_qtd02.npz`, which is `directions.audited` (the
`old` rule: field-metric POD of the frozen head's static reconstruction residual) evaluated on
the INCUMBENT checkpoint by job 3757505. A fresh checkpoint needs the same rule evaluated on
itself. This script is `b-seeds/seeds_run.py`'s setup up to `DIR.audited`, with the cohort and
training draws through `lv_common` at the config's viscosity bounds, and it writes the file and
the `PROVENANCE.json` entry in exactly the form `lv_panel.py` (= b-panel's panel.py) consumes.

    python lv_directions.py --config <cfg> --checkpoint <pkl> --out <inputs dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip
import arms as A
import ladder as LD
import directions as DIR
import lv_common as LV

host = lambda t: jax.tree_util.tree_map(np.asarray, jax.device_get(t))


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L = int(cfg['intervals'])
    dt = cfg['dt']
    strict = cfg['strict']
    qlad = [int(q) for q in cfg['q_ladder']]
    lo, hi = cfg['nu_bounds']
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
               matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
               checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R, intervals=L, dt=dt, nu_bounds=[lo, hi],
               final_cohort_unopened=True, trained_model_used=True, verification=ip.verify(), gates={},
               complete=False)
    save = lambda: (out / 'directions_result.json').write_text(json.dumps(rep, indent=1) + '\n')

    physical = LV.cohort(cfg, lo, hi)
    rep['physical_sha256'] = LV.sha_array(physical)
    rep['physical_cases'] = physical.tolist()
    want = cfg.get('expected_physical_sha256')
    rep['gates']['evaluation_cohort_matches_gate_job'] = dict(
        expected=want, got=rep['physical_sha256'], passed=(None if want is None else rep['physical_sha256'] == want))
    train_physical = LV.params_draw(cfg['train_seed'], cfg['train_trajectories'], lo, hi)
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
    rep['train_physical_sha256'] = LV.sha_array(train_physical)
    rep['gates']['training_disjoint_from_evaluation'] = dict(passed=True)
    save()

    U, sinfo = LD.generate_snapshots(L, dt, train_physical, cfg['train_state_stride'],
                                     cfg['snapshot_ntol'], cfg['snapshot_ltol'])
    assert sinfo['max_relative_residual'] <= cfg['snapshot_residual_bar'], sinfo
    rep['gates']['snapshot_residual'] = dict(passed=True, bar=cfg['snapshot_residual_bar'],
                                             max_relative_residual=sinfo['max_relative_residual'])
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)
    jax.block_until_ready(Rb)
    Ut = jnp.asarray(U.T)
    coef_truth = jnp.linalg.solve(Rb, Qb.T @ Ut).T
    sinfo['bank_projection_relative_rms'] = float(jnp.linalg.norm(Ut - Qb @ (Qb.T @ Ut)) / jnp.linalg.norm(Ut))
    del Ut, U
    jax.clear_caches()
    rep['snapshots'] = sinfo
    print('SNAPSHOTS', round(sinfo['seconds'], 1), 'bank projection rms', f"{sinfo['bank_projection_relative_rms']:.3e}", flush=True)
    save()

    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    dcfg = {**cfg, 'strict': strict}
    C, Ct, Zstar, rho, dinfo = DIR.audited(params, Rb, coef_truth, Zsub, K, dcfg)
    Cn = np.ascontiguousarray(np.asarray(C))
    Ctn = np.ascontiguousarray(np.asarray(Ct))
    name = cfg['directions_file']
    np.savez_compressed(out / name, C=Cn, Ct=Ctn, singular_values=np.asarray(dinfo['singular_values']))
    prefix = {str(q): LV.sha_array(np.ascontiguousarray(Cn[:, :q])) for q in qlad}
    orth = float(np.max(np.abs(np.asarray(jnp.asarray(Ctn).T @ jnp.asarray(Ctn)) - np.eye(Ctn.shape[1]))))
    dinfo.update(artifact=name, sha256=sha_file(out / name), prefix_sha256=prefix, columns=int(Cn.shape[1]),
                 residual_rows=int(np.asarray(rho).shape[0]), field_orthonormality_deviation=orth,
                 singular_values=[float(x) for x in np.asarray(dinfo['singular_values'])])
    rep['directions'] = dinfo
    rep['gates']['directions_rank_covers_ladder'] = dict(passed=bool(dinfo['available_rank'] >= max(qlad)),
                                                          available_rank=dinfo['available_rank'], need=max(qlad))
    rep['gates']['directions_field_orthonormal'] = dict(passed=bool(orth <= 1e-8), deviation=orth)
    # the PROVENANCE entry in the form panel.py asserts against (file sha + prefix hashes per rung)
    (out / 'PROVENANCE.json').write_text(json.dumps(dict(
        note=('built in THIS job by lv_directions.py: directions.audited (the incumbent `old` rule, '
              'ladder.residual_directions arithmetic unchanged) evaluated on the low-viscosity checkpoint; '
              'the same rule that produced the incumbent panel\'s directions_qtd02.npz (job 3757505)'),
        files={name: dict(sha256=dinfo['sha256'], source_job=os.environ.get('SLURM_JOB_ID'), source_attempt=cfg.get('attempt'),
                          source_path=f'output/directions/{name}', columns=int(Cn.shape[1]),
                          available_rank=int(dinfo['available_rank']), prefix_sha256=prefix,
                          rule='old: field-metric POD of the frozen head\'s static reconstruction residual eta - h(z*), nested in q',
                          checkpoint_sha256=rep['checkpoint_sha256'])}), indent=2) + '\n')
    rep['elapsed_seconds'] = time.perf_counter() - begin
    rep['complete'] = all(g['passed'] is not False for g in rep['gates'].values())
    save()
    print('DIRECTIONS', round(dinfo['seconds'], 1), 'rank', dinfo['available_rank'], 'orth', f'{orth:.2e}',
          'sha', dinfo['sha256'][:12], 'complete', rep['complete'], flush=True)
    assert rep['complete']


if __name__ == '__main__':
    main()
