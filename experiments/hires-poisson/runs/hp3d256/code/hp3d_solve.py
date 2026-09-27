"""hires-poisson 3D: frozen-weight transfer of the accepted Poisson3D checkpoint (trained at
32^3, R=128, K=16; `experiments/paper-p3d/runs/final08/checkpoints`) to 64^3 and 128^3 (256^3 if
configured), every comparator timed in the same allocation.

The parent query `poisson.engine` is run unchanged as the `retained` baseline. The lean
variants are the same query with (a) the in-kernel full-gradient diagnostic removed from the
timed kernel, (b) the dense M x n^3 source projection replaced by one orthonormal DST-I of the
source and a gather of the M retained modes, (c) an f32 decode, (d) one start instead of three
(a different solver setting, labelled; not a parity arm). Each of (a)-(c) is parity-gated.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pickle
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.fft import dstn
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl

import common as C
import poisson as P
import shared_rom as S
import iterative_cg as CG


def gpu_uuid():
    cuda = ctypes.CDLL('libcuda.so.1')
    assert cuda.cuInit(0) == 0
    dev = ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
    raw = (ctypes.c_ubyte * 16)()
    fn = getattr(cuda, 'cuDeviceGetUuid_v2', cuda.cuDeviceGetUuid)
    assert fn(ctypes.byref(raw), dev) == 0
    return 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def reference(n, f):
    k = np.arange(1, n)
    l = 4. * n * n * np.sin(np.pi * k / (2 * n)) ** 2
    lam = l[:, None, None] + l[None, :, None] + l[None, None, :]
    return dstn(dstn(f, type=1, norm='ortho', workers=8) / lam, type=1, norm='ortho', workers=8)


def lean_engine(model, n, triples, lam, operator, q, cfg, projection=None, starts=None):
    """`poisson.engine` without the in-kernel full-gradient diagnostic. With `projection=None`
    the weak source moments come from one orthonormal DST-I of the forcing:
    (phi^T f / n^3)_k = dst3(f)[k-1] / n^{3/2}, then the same 1/lambda_k row scaling."""
    starts = cfg['initial_starts'] if starts is None else starts
    directions = np.asarray(model['directions'])[:, :q]
    if q:
        qq, rr = np.linalg.qr(operator @ directions, mode='reduced')
        reduced = operator - qq @ (qq.T @ operator)
    else:
        qq = np.zeros((len(operator), 0)); rr = np.zeros((0, 0)); reduced = operator
    p = jax.device_put(model['params'])
    codes = jnp.asarray(model['codes'])
    library = C.head(p, codes) @ jnp.asarray(reduced).T
    fit = S.lm(C.head, cfg['lm_budget'], cfg['lm_tolerance'])
    ti, tj, tk = (jnp.asarray(triples[:, a] - 1) for a in range(3))
    scale = jnp.asarray(1.0 / (n ** 1.5 * np.asarray(lam)))

    @jax.jit
    def query(f, p, bank, a, ap, qq, rr, directions, projection, library, codes, scale):
        if projection is None:
            target = C.dst3(f)[ti, tj, tk] * scale
        else:
            target = projection.T @ f.reshape(-1)
        tp = target - qq @ (qq.T @ target)
        order = jnp.argsort(jnp.sum((library - tp) ** 2, axis=1))[:starts]
        zs, stats = jax.vmap(lambda z: fit(p, ap, tp, z))(codes[order])
        which = jnp.argmin(stats[:, 3]); z = zs[which]; h = C.head(p, z)
        y = jsl.solve_triangular(rr, qq.T @ (target - a @ h), lower=False) if q else jnp.zeros(0)
        coef = h + directions @ y
        field = (bank @ coef.astype(bank.dtype)).astype(jnp.float64)
        return field, stats[which], coef, z, jnp.sum(stats[:, 0])
    return query, (jnp.asarray(operator), jnp.asarray(reduced), jnp.asarray(qq), jnp.asarray(rr),
                   jnp.asarray(directions)), library, codes, scale, p


def trilinear_up(u, nc, n):
    """Zero-boundary coarse interior (nc-1)^3 -> fine interior (n-1)^3, trilinear."""
    v = jnp.pad(u, 1)
    old = jnp.linspace(0., 1., nc + 1)
    new = jnp.linspace(0., 1., n + 1)[1:-1]
    for axis in range(3):
        v = jnp.moveaxis(v, axis, -1)
        v = jax.vmap(jax.vmap(lambda line: jnp.interp(new, old, line)))(v)
        v = jnp.moveaxis(v, -1, axis)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    if a.smoke:
        cfg.update(meshes=[16, 32], dense_projection_max=16, repetitions=1, burn_seconds=0.001, case_count=2,
                   coarse_intervals=[8], cg_tolerances=[1e-2, 1e-6])
    assert jax.default_backend() == 'gpu' and len(jax.devices()) == 1
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    uuid0 = gpu_uuid()
    inventory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader'], text=True)
    assert uuid0 in inventory
    bankck = pickle.loads((here / 'bank.pkl').read_bytes())
    model = pickle.loads((here / 'head_K16.pkl').read_bytes())
    Rw = int(np.asarray(bankck['rotation']).shape[1])
    K = int(np.asarray(model['codes']).shape[1])
    dev = C.family(cfg['cohort_seed'], cfg['cohort_draw'])[:cfg['case_count']]
    train = C.family(cfg['train_seed'], cfg['train_count'])
    assert not any(np.allclose(t, s) for t in train for s in dev)
    R_ = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
              backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, gpu_uuid=uuid0,
              nvidia_smi=inventory, x64=True, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
              jax_version=jax.__version__, smoke=bool(a.smoke), K=K, R=Rw,
              checkpoint_sha256=dict(bank=hashlib.sha256((here / 'bank.pkl').read_bytes()).hexdigest(),
                                     head=hashlib.sha256((here / 'head_K16.pkl').read_bytes()).hexdigest()),
              cohort=dict(parameters=dev.tolist(), seed=cfg['cohort_seed'],
                          note='development (validation-seed) cohort of paper-p3d; the accepted final cohort is not reused'),
              timing_contract=('host f64 interior forcing (n-1)^3 in to host f64 interior field out for every '
                               'subject; upload, device work, download in one synchronised interval'),
              meshes=[], invocations=[], parity=[], diagnostics=[], declared_subjects={}, complete=False)
    save = lambda: C.dump(out / 'result.json', R_)
    save()
    order = np.random.default_rng(cfg['order_seed'])
    for n in cfg['meshes']:
        t0 = time.perf_counter()
        sources = [np.asarray(P.source(n, p)) for p in dev]
        same = [reference(n, f) for f in sources]
        fine = [C.restrict(reference(2 * n, np.asarray(P.source(2 * n, p))), 2 * n, n) for p in dev]
        bank = C.bank_at(bankck['params'], n, cfg['field_chunk']) @ np.asarray(bankck['rotation'])
        dense = n <= cfg.get('dense_projection_max', 128)
        bankj = jnp.asarray(bank)
        # DST assembly (A7): the weak operator from one DST-I per bank column, never the
        # dense (n-1)^3 x M test matrix (68 GB at 256^3). Gated against `poisson.assemble`.
        triples = C.modes(cfg['weak_tests'], n)
        lam = np.asarray(C.mode_eigenvalues(n, triples))
        ti, tj, tk = (jnp.asarray(triples[:, ax] - 1) for ax in range(3))
        col = jax.jit(lambda g: C.dst3(g.reshape((n - 1,) * 3))[ti, tj, tk] / n ** 1.5)
        operator_dst = np.stack([np.asarray(col(bankj[:, r])) for r in range(bankj.shape[1])], axis=1)
        assembly = dict(route='dst')
        if dense:
            test, operator, projection, lam_d, triples_d = P.assemble(bank, n, cfg['weak_tests'])
            del test
            assert np.array_equal(triples_d, triples) and np.allclose(lam_d, lam, rtol=1e-14, atol=0)
            assembly = dict(route='dense (parent) for retained/lean; dst gated against it',
                            dst_vs_dense_operator_relative=rel(operator_dst, operator))
            assert assembly['dst_vs_dense_operator_relative'] <= 1e-11, assembly
            projj = jnp.asarray(projection)
        else:
            operator, projection, projj = operator_dst, None, None
        rfac = np.linalg.qr(bank, mode='r')
        import scipy.linalg as sla
        floor = []
        for u in same:
            t_ = sla.solve_triangular(rfac.T, bank.T @ u.ravel(), lower=True)
            floor.append(float(np.sqrt(max(0.0, 1.0 - float(t_ @ t_) / float(u.ravel() @ u.ravel())))))
        bank32 = bankj.astype(jnp.float32)
        R_['meshes'].append(dict(intervals=n, unknowns=(n - 1) ** 3, weak_tests=int(len(triples)),
                                 assembly=assembly, dense_projection=bool(dense),
                                 bank_sha256=sha(bank), bank_floor=floor, bank_floor_worst=max(floor),
                                 discretisation_error=[rel(s_, f_) for s_, f_ in zip(same, fine)],
                                 setup_seconds=time.perf_counter() - t0))
        print('MESH', n, 'floor worst', max(floor), round(time.perf_counter() - begin, 1), flush=True)
        subjects = []
        indices = np.arange((n - 1) ** 3)
        for q in cfg['q_ladder']:
            if dense:
                fn = P.engine(model, bank, operator, projection, indices, q, cfg)
                subjects.append(dict(name=f'rom_q{q}_retained', family='nm-rom', q=q, kind='rom',
                                     fn=lambda f, fn=fn: (lambda v: (v[0], v[1], v[2], v[1][5], v[1][7]))(fn(f))))
            for vname, proj, st in ((('lean', projj, None),) if dense else ()) + (('leandst', None, None), ('onestart', None, 1)):
                query, mats, library, codes, scale, p = lean_engine(model, n, triples, lam, operator, q, cfg,
                                                                    projection=proj, starts=st)
                for prec, bk in (('64', bankj), ('32', bank32)):
                    if prec == '32' and vname != 'leandst':
                        continue

                    def run(f, query=query, mats=mats, library=library, codes=codes, scale=scale, p=p,
                            bk=bk, proj=proj):
                        aa, ap_, qq, rr, dd = mats
                        return query(f, p, bk, aa, ap_, qq, rr, dd, proj, library, codes, scale)
                    subjects.append(dict(name=f'rom_q{q}_{vname}{prec}', family='nm-rom', q=q, kind='lean',
                                         fn=run, operator=operator, starts=st))
        if dense:
            lin, linfo = P.linear_weak(bank, operator, projection)
            subjects.append(dict(name=f'rom_q{Rw}_linear', family='linear-rom', q=Rw, kind='fn', fn=lin))
        qq_l, rr_l = np.linalg.qr(operator, mode='reduced')
        scale_l = jnp.asarray(1.0 / (n ** 1.5 * lam))
        lin_dst = jax.jit(lambda f, bank, qq, rr, scale: bank @ jsl.solve_triangular(
            rr, qq.T @ (C.dst3(f)[ti, tj, tk] * scale), lower=False))
        subjects.append(dict(name=f'rom_q{Rw}_lineardst', family='linear-rom', q=Rw, kind='fn',
                             fn=lambda f, a_=(bankj, jnp.asarray(qq_l), jnp.asarray(rr_l), scale_l): lin_dst(f, *a_)))
        lamj = C.eigenvalues(n)
        subjects.append(dict(name='dst_direct', family='direct-control', kind='fn',
                             fn=lambda f, lamj=lamj: P.solve_dst(f, lamj)))
        for tol in cfg['cg_tolerances']:
            eng = CG.engine(n, float(tol), int(cfg['cg_iterations_per_interval']) * n, retain_history=False)
            subjects.append(dict(name=f'cg_{tol:g}', family='cg', kind='cg', tolerance=tol, fn=eng))
        for nc in cfg['coarse_intervals']:
            if nc >= n:
                continue
            assert n % nc == 0
            s_ = n // nc
            clam = C.eigenvalues(nc)
            cd = jax.jit(lambda f, clam, s_=s_, nc=nc: trilinear_up(
                P.solve_dst(f[s_ - 1::s_, s_ - 1::s_, s_ - 1::s_], clam), nc, n))
            subjects.append(dict(name=f'coarse{nc}_dst', family='coarse-dst', kind='fn', coarse_intervals=nc,
                                 fn=lambda f, cd=cd, clam=clam: cd(f, clam)))
            for tol in cfg['coarse_cg_tolerances']:
                ceng = CG.engine(nc, float(tol), int(cfg['cg_iterations_per_interval']) * nc, retain_history=False)

                def ccg(f, ceng=ceng, s_=s_, nc=nc):
                    x, stats = ceng(f[s_ - 1::s_, s_ - 1::s_, s_ - 1::s_])
                    return trilinear_up(x, nc, n), stats
                subjects.append(dict(name=f'coarse{nc}_cg_{tol:g}', family='coarse-cg', kind='cg',
                                     coarse_intervals=nc, tolerance=tol, fn=jax.jit(ccg)))
        R_['declared_subjects'][str(n)] = [s['name'] for s in subjects]

        def invoke(sub, f):
            start = time.perf_counter()
            fj = jax.device_put(f); fj.block_until_ready()
            up = time.perf_counter()
            value = sub['fn'](fj); jax.block_until_ready(value)
            done = time.perf_counter()
            value = jax.device_get(value)
            end = time.perf_counter()
            row = dict(total_seconds=end - start, input_seconds=up - start,
                       fused_device_seconds=done - up, output_seconds=end - done)
            if sub['kind'] in ('rom', 'lean'):
                field, stats, coef, z, its = value
                full_ok = True
                if sub['kind'] == 'rom':          # parent kernel: its full-gradient diagnostic is gated
                    row['full_gradient'] = float(z)
                    full_ok = bool(float(z) <= cfg['lm_tolerance'])
                row.update(attempts=int(stats[0]), accepted=int(stats[1]), reason=int(stats[2]),
                           weak_residual=float(stats[3]), lm_gradient=float(stats[4]),
                           total_attempts_all_starts=int(its),
                           stationary=bool(int(stats[2]) == 1 and stats[4] <= cfg['lm_tolerance'] and full_ok),
                           coefficients=np.asarray(coef).tolist())
            elif sub['kind'] == 'cg':
                field, stats = value
                c = CG.counters(stats, sub['tolerance'], 0)
                row.update(iterations=c['iterations'], cg_converged=c['cg_converged'],
                           true_relative_residual=c['cg_true_relative_residual'])
            else:
                field = value
            return np.asarray(field, dtype=np.float64).reshape((n - 1,) * 3), row

        t = time.perf_counter()
        for sub in subjects:
            invoke(sub, sources[0])
        print('WARMUP', n, len(subjects), round(time.perf_counter() - t, 1), flush=True)
        saved = {}
        for rep in range(cfg['repetitions']):
            for case in range(len(dev)):
                for i in order.permutation(len(subjects)):
                    sub = subjects[int(i)]
                    C.burn(cfg['burn_seconds'])
                    assert gpu_uuid() == uuid0
                    field, row = invoke(sub, sources[case])
                    assert gpu_uuid() == uuid0
                    assert np.isfinite(field).all(), sub['name']
                    h = sha(field)
                    key = (n, sub['name'], case)
                    if key not in saved:
                        fname = f"N{n}_{sub['name']}_case{case}.npy"
                        np.save(out / 'fields' / fname, field)
                        saved[key] = (fname, h)
                    if rep:
                        row.pop('coefficients', None)
                    R_['invocations'].append(dict(
                        intervals=n, case=case, rep=rep, name=sub['name'], family=sub['family'],
                        q=sub.get('q'), tolerance=sub.get('tolerance'),
                        coarse_intervals=sub.get('coarse_intervals'), field_sha256=h,
                        saved_field=saved[key][0], matches_saved_field=bool(h == saved[key][1]),
                        same_grid_error=rel(field, same[case]), physical_error=rel(field, fine[case]), **row))
            print('TIMED', n, rep, round(time.perf_counter() - begin, 1), flush=True)
            save()
        # parity: every lean variant against the unchanged parent engine
        first = {}
        for x in R_['invocations']:
            if x['intervals'] == n:
                first.setdefault((x['name'], x['case']), x)
        if dense:
            worst = max(rel(np.load(out / 'fields' / first[(f'rom_q{Rw}_lineardst', c)]['saved_field']),
                            np.load(out / 'fields' / first[(f'rom_q{Rw}_linear', c)]['saved_field']))
                        for c in range(len(dev)))
            R_['parity'].append(dict(intervals=n, candidate=f'rom_q{Rw}_lineardst', baseline=f'rom_q{Rw}_linear',
                                     worst_field_relative=worst, integers_identical=True, limit=1e-10,
                                     same_solver=True, passed=bool(worst <= 1e-10)))
        for q in (cfg['q_ladder'] if dense else []):
            for v, limit, same_solver in (('lean64', 1e-12, True), ('leandst64', 1e-10, True),
                                          ('leandst32', 1e-4, True), ('onestart64', None, False)):
                worst, ints = 0.0, True
                for case in range(len(dev)):
                    xa, xb = first[(f'rom_q{q}_{v}', case)], first[(f'rom_q{q}_retained', case)]
                    worst = max(worst, rel(np.load(out / 'fields' / xa['saved_field']),
                                           np.load(out / 'fields' / xb['saved_field'])))
                    ints = ints and all(xa[k] == xb[k] for k in ('attempts', 'accepted', 'reason', 'total_attempts_all_starts'))
                R_['parity'].append(dict(intervals=n, candidate=f'rom_q{q}_{v}', baseline=f'rom_q{q}_retained',
                                         worst_field_relative=worst, integers_identical=ints, limit=limit,
                                         same_solver=same_solver,
                                         passed=(None if limit is None else bool(worst <= limit and ints))))
                print('PARITY', R_['parity'][-1], flush=True)
        for (name, case), x in first.items():
            if 'stationary' in x:
                rows = [v for v in R_['invocations'] if v['intervals'] == n and v['name'] == name and v['case'] == case]
                R_['diagnostics'].append(dict(intervals=n, name=name, case=case,
                                              valid=bool(all(v['stationary'] for v in rows))))
            elif 'cg_converged' in x:
                rows = [v for v in R_['invocations'] if v['intervals'] == n and v['name'] == name and v['case'] == case]
                R_['diagnostics'].append(dict(intervals=n, name=name, case=case,
                                              valid=bool(all(v['cg_converged'] for v in rows))))
        del bankj, bank32, projj, subjects, bank, operator, projection
        jax.clear_caches()
        save()
    stats = jax.devices()[0].memory_stats() or {}
    R_['device_memory'] = {k: int(v) for k, v in stats.items() if k in ('peak_bytes_in_use', 'bytes_limit')}
    gates = dict(parity=bool(R_['parity']) and all(p['passed'] for p in R_['parity'] if p['passed'] is not None),
                 solver_validity=bool(R_['diagnostics']) and all(d['valid'] for d in R_['diagnostics']),
                 deterministic=all(x['matches_saved_field'] for x in R_['invocations']))
    R_['gates'] = gates
    print('GATES', gates, flush=True)
    R_['device_guard_final_uuid'] = gpu_uuid()
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / ('COMPLETE' if all(gates.values()) else 'COMPLETE-WITH-FAILED-GATES')).write_text(json.dumps(gates) + '\n')
    print('HIRES 3D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
