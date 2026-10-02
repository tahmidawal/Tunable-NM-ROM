"""quadrature-burgers3d refined reference job (DESIGN.md section 4).

The refined reference is the paper-b3d full-order model (sign-upwind advection, 7-point Laplacian, backward Euler,
Newton with Helmholtz-preconditioned BiCGStab; vendor common.make_fom, unchanged) at 513 nodes per axis (512^3 cells,
511^3 = 133M unknowns) with half the time step (dt = 0.0025). Only the fields on the 65-node lattice x = k/64
(63^3 nodes, common to every mesh of the lane) at t = 0, 0.05, ..., 0.25 are kept.

mode 'probe': time one case at each (ntol, ltol) of the config and record its lattice difference from the tightest;
              also the same-grid 257-node reference of that case (reference self-error estimate). Diagnostic only.
mode 'run'  : every cohort of the config, written to output/ref_<seed>.npz (keys c<j>, seed, count, n, dt, ntol,
              ltol) and output/ref_<seed>.done (written last, with the npz SHA256). Per-case records in result.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offmesh as OM  # noqa: E402
from offmesh import C  # noqa: E402
from qpanel import lattice65_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('local_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    t_begin = time.perf_counter()
    log = lambda s: print(f'[{time.perf_counter() - t_begin:.1f}s] {s}', flush=True)
    n, dt = int(cfg['n_ref']), float(cfg['dt_ref'])
    i65 = lattice65_index(n)
    rep = dict(config=cfg, job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'),
               gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), cohorts={}, probe={})
    save = lambda: C.dump(out / 'result.json', C.clean(rep))
    save()

    def solve(fom, tab, j):
        u0 = jnp.asarray(C.initial_interior(n, tab, j))
        t = time.perf_counter()
        f, it, rn = fom(u0, float(tab['nu'][j]))
        f = jax.block_until_ready(f)
        sec = time.perf_counter() - t
        lat = np.asarray(f[:, jnp.asarray(i65)])
        rec = dict(seconds=sec, max_newton=int(jnp.max(it)), total_newton=int(jnp.sum(it)),
                   max_rel_residual=float(jnp.max(rn)), finite=bool(np.isfinite(lat).all()))
        rec['accepted'] = bool(rec['finite'] and rec['max_newton'] < C.MAX_NEWTON and
                               rec['max_rel_residual'] <= fom.ntol)
        del f, u0
        return lat, rec

    def lean(nn, dt_, ntol, ltol):
        f = OM.make_fom_lean(nn, dt_, ntol, ltol)
        f.ntol = ntol
        return f

    # gate R1-1: the lean FOM equals the vendor FOM (fields, Newton counts, residuals) at a small mesh
    tab_g = C.table(cfg.get('probe_seed', 923651), 1)
    u0g = jnp.asarray(C.initial_interior(65, tab_g, 0))
    a_ = C.make_fom(65, 0.005, 1e-8, 1e-9)(u0g, float(tab_g['nu'][0]))
    b_ = lean(65, 0.005, 1e-8, 1e-9)(u0g, float(tab_g['nu'][0]))
    rep['gate_lean_vs_vendor'] = dict(fields=float(jnp.max(jnp.abs(a_[0] - b_[0])) / jnp.max(jnp.abs(a_[0]))),
                                      newton_equal=bool(jnp.all(a_[1] == b_[1])),
                                      residuals=float(jnp.max(jnp.abs(a_[2] - b_[2]))))
    log(f"lean FOM gate {rep['gate_lean_vs_vendor']}")
    assert rep['gate_lean_vs_vendor']['fields'] <= 1e-13 and rep['gate_lean_vs_vendor']['newton_equal']
    save()

    if cfg['mode'] == 'probe':
        tab = C.table(cfg['probe_seed'], 1)
        lats = {}
        for ntol, ltol in cfg['tolerances']:
            fom = lean(n, dt, ntol, ltol)
            lat, rec = solve(fom, tab, 0)                         # includes compilation (blocked)
            lat, rec = solve(fom, tab, 0)
            rec['peak_bytes'] = (jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use')
            lats[(ntol, ltol)] = lat
            rep['probe'][f'{ntol:g}_{ltol:g}'] = rec
            log(f'probe ntol {ntol:g} ltol {ltol:g}: {rec}')
            save()
        tight = lats[tuple(cfg['tolerances'][-1])]
        n0 = float(np.linalg.norm(tight[0]))
        for k, v in lats.items():
            rep['probe'][f'{k[0]:g}_{k[1]:g}']['lattice_diff_vs_tightest'] = float(
                (np.linalg.norm(v - tight, axis=1) / n0).max())
        for nn in cfg.get('compare_meshes', []):
            fom = lean(nn, C.DT, 1e-10, 1e-11)
            u0 = jnp.asarray(C.initial_interior(nn, tab, 0))
            jax.block_until_ready(fom(u0, float(tab['nu'][0])))
            t = time.perf_counter()
            f, _, _ = fom(u0, float(tab['nu'][0]))
            f = jax.block_until_ready(f)
            sec = time.perf_counter() - t
            lat = np.asarray(f[:, jnp.asarray(lattice65_index(nn))])
            rep['probe'][f'same_grid_{nn}'] = dict(seconds=sec, lattice_diff_vs_refined=float(
                (np.linalg.norm(lat - tight, axis=1) / n0)[1:].max()))
            log(f"same-grid {nn}: {rep['probe'][f'same_grid_{nn}']}")
            del f, u0
            save()
    else:
        fom = lean(n, dt, cfg['ntol'], cfg['ltol'])
        for seed, count in cfg['cohorts']:
            tab = C.table(seed, count)
            recs, lats = [], {}
            for j in range(count):
                lats[f'c{j}'], rec = solve(fom, tab, j)
                rec['case'] = j
                recs.append(rec)
                log(f'cohort {seed} case {j}: {rec}')
                rep['cohorts'][str(seed)] = dict(count=count, table_sha256=tab['sha256'], cases=recs)
                save()
            assert all(r['accepted'] for r in recs), [r for r in recs if not r['accepted']]
            p = out / f'ref_{seed}.npz'
            np.savez(p, seed=seed, count=count, n=n, dt=dt, ntol=cfg['ntol'], ltol=cfg['ltol'], **lats)
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            rep['cohorts'][str(seed)].update(npz_sha256=h, worst_rel_residual=max(r['max_rel_residual'] for r in recs))
            save()
            p.with_suffix('.done').write_text(json.dumps(dict(sha256=h, n=n, dt=dt, ntol=cfg['ntol'], ltol=cfg['ltol'],
                                                              seed=seed, count=count, accepted=True)) + '\n')
            log(f'cohort {seed} written ({h[:12]})')
    rep['complete'] = True
    rep['seconds'] = time.perf_counter() - t_begin
    save()
    print('REF COMPLETE', flush=True)


if __name__ == '__main__':
    main()
