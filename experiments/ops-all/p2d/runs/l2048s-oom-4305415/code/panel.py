"""ops-all p2d panel: one Table-1 cell (Poisson 2D square or L-shape, one mesh) in ONE GPU allocation.

Subjects, all timed in the same process on the same UUID-guarded GPU:
  * NM-ROM accurate and fast Table-1 settings, frozen, built with the source lane's code, copied verbatim from
    the staged `code/` directory of the Table-1 job (square: poisson-bank-knob pbkH/pbkI, `pbk_core`/`hp_core`
    linear rung; L-shape: poisson-bank-knob-3d l2048b, `pbk3_lshape.make_linear`). Square: R'=512 / R'=128
    span; L-shape: R'=128 / R'=64 span. No retraining, no new arm.
  * the lane's unpreconditioned CG grid (square `iterative_core.make_cg` via `hp_core.generic_query`; L-shape
    `lsh_core.make_gpu_cg`), including the Table-1 named FOM;
  * the four trained operator checkpoints (FNO / U-Net / Transolver / DeepONet), or their recorded failure.
Timing scope = the Table-1 scope of both series: GPU query = device work between a synchronised host->device
source copy and the device->host field copy (`fused_device_seconds`); `total_seconds` also recorded.
Design A-B-A as poisson-bank-knob DESIGN A5: phase A1 (ROM + operators, randomised within each case, burn-in
before every invocation), phase B (every CG setting), phase A2 (= A1). Table time = median over A1 u A2.
Error = same-grid relative L2 against the lane's reference (square: DST-I, `pbk_solve.reference`;
L-shape: SuperLU + refinement, `lsh_core.FOM.reference`), worst / median over the Table-1 cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import torch

HERE = Path(__file__).resolve().parent


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))


def dump(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float) + '\n')
    tmp.replace(path)


# ------------------------------------------------------------------ square (poisson-bank-knob code)
def build_square(cfg, n, R_):
    sys.path.insert(0, str(HERE / 'nmrom_sq'))
    import core as C
    import iterative_core as IC
    import pbh_core as K_
    import sep_common as sc
    import hp_core as H
    import pbk_core as P
    from pbk_solve import reference
    d = HERE / 'nmrom_sq'
    params, codes, _ = sc.load_pkl(d / 'primary_K32.pkl')
    basis = dict(np.load(d / 'primary_K32-basis.npz'))
    codes = np.asarray(codes)
    np.testing.assert_array_equal(codes, basis['training_latents'])
    draws = C.source_params(0, 3072)
    fit, _ = K_.fit_validation_split(len(draws), 20260916, 0.15)
    dev = np.concatenate((C.source_params(7090703, 6), C.source_params(7090732, 6)))
    assert not any(np.allclose(t, s) for t in draws[fit] for s in dev)
    R_['cohort'] = dict(parameters=dev.tolist(), sha256=H.sha_array(dev), role='development',
                        note='the twelve opened development sources of p-linear / hires-poisson / poisson-bank-knob')
    K, Rw = codes.shape[1], int(np.asarray(params['h_lin']).shape[1])
    z = np.load(d / 'prep.npz', allow_pickle=True)
    Tm = z['T']
    R_['prep_sha256'] = sha_file(d / 'prep.npz')
    R_['checkpoint_sha256'] = sha_file(d / 'primary_K32.pkl')
    edges = [0, 32, 64, 128, 256, 384, 512]
    assert edges[-1] == Rw
    same = [reference(p_, n) for p_ in dev]
    sources = [C.full_source(n, p_) for p_ in dev]
    arms = [dict(Rp=Rp, q=Rp, kind='linear', name=f'R{Rp}_linear', M=4 * (K + Rp)) for Rp in cfg['nmrom_R']]
    maxmode = 0
    for arm in arms:
        I, J, _ = H.mode_set(n, arm['M'])
        maxmode = max(maxmode, int(max(I.max(), J.max())) + 1)
    truth = np.stack([s_[1:-1, 1:-1] for s_ in same])
    t0 = time.perf_counter()
    bank = P.build_banks(params, n, cfg['rows_per_chunk'], cfg['eval_rows'], maxmode, Tm, edges, truth, keep_orig=False)
    del truth
    R_['bank'] = dict(bank['info'], seconds=time.perf_counter() - t0)
    subjects = []
    for arm in arms:
        ops = H.make_ops(bank, params, codes, n, arm['M'])
        Rp = arm['Rp']
        nb = P.nblocks(edges, Rp)
        Bp = ops['B'] @ jnp.asarray(Tm[:, :Rp])
        kern, Qt, Rr, linfo = P.make_trunc_linear(Bp, n, nb)
        arm.update({k: v for k, v in linfo.items() if isinstance(v, (int, float, str, bool))})

        def call(s, ops=ops, kern=kern, Qt=Qt, Rr=Rr):
            field, row = P.trunc_linear_query(s, ops, kern, Qt, Rr, bank['rot'])
            return np.asarray(field, dtype=np.float64), row
        subjects.append(dict(name=arm['name'], family='nm-rom', arm=arm, call=call))
    cg = IC.make_cg(n, int(cfg['cg_maxiter']))

    def cg_call(tol):
        def call(s):
            field, row, extra = H.generic_query(s, lambda x: cg(x, jnp.asarray(tol)))
            count, true, recursive, converged = extra
            row.update(iterations=int(count), true_relative_residual=float(true), cg_converged=bool(converged))
            return np.asarray(field, dtype=np.float64), row
        return call
    cgs = [dict(name=f'cg_{t:g}', family='cg', tolerance=t, call=cg_call(t)) for t in cfg['cg_tols']]
    return dev, same, sources, subjects, cgs, C.burn, None


# ------------------------------------------------------------------ L-shape (poisson-bank-knob-3d code)
def build_lshape(cfg, n, R_):
    sys.path.insert(0, str(HERE / 'nmrom_ls'))
    import sep_common as sc
    import lsh_core as K_
    import pbk3_core as P3
    from pbk3_lshape import make_linear, query
    d = HERE / 'nmrom_ls'
    models = {m['id']: m for m in json.loads((d / 'models.json').read_text())}
    m = models['head_sdf_R512_K16']
    assert P3.sha_file(d / m['checkpoint']) == m['checkpoint_sha256']
    params, codes, ckcfg = sc.load_pkl(d / m['checkpoint'])
    codes = np.asarray(codes)
    Rw = int(np.asarray(params['h_lin']).shape[1])
    prep = np.load(d / 'prep_lshape.npz', allow_pickle=False)
    Tm = prep['T']
    R_['prep_sha256'] = P3.sha_file(d / 'prep_lshape.npz')
    R_['checkpoint_sha256'] = m['checkpoint_sha256']
    dev, dinfo = K_.cohort(20260917, 64, 32)
    train_draws, _ = K_.cohort(0, 4608, 3072)
    assert not any(np.allclose(t, s) for t in train_draws for s in dev)
    R_['cohort'] = dict(**dinfo, role='development', parameters=dev.tolist(),
                        note='the 32 development sources of the lshape / hires-poisson / poisson-bank-knob-3d lanes')
    t0 = time.perf_counter()
    geom = K_.Geometry(n, 'lshape')
    fom = K_.FOM(geom, build_ic0=False)
    ops = K_.weak_ops(fom, 257, 0)
    same, sources, refs = {}, [], []
    for case, q in enumerate(dev):
        u, resid, floor = fom.reference(K_.source_interior(geom, q))
        assert resid <= max(1e-12, floor), (case, resid, floor)
        same[case] = geom.scatter(u)
        sources.append(K_.source_full(geom, q))
        refs.append(dict(case=case, residual=resid, floor=floor))
    R_['references'] = refs
    R_['fom_setup_seconds'] = time.perf_counter() - t0
    print('FOM + MODES + REFERENCES', round(R_['fom_setup_seconds'], 1), flush=True)
    t0 = time.perf_counter()
    G = K_.bank_of(K_.make_features(ckcfg['factor'], int(ckcfg['n_enrich'])), params, geom)
    Rg, rank = K_.bank_r(G)
    assert rank['rank_valid'], rank
    edges = [0, 32, 64, 128, 256, 384, 512]
    assert edges[-1] == Rw
    # DESIGN A8: hold the bank on the host during reduce_bank (which converts it to host anyway), so the
    # device has room for P (A G) at 2048^2 on an 80 GB card; same functions, same arithmetic.
    Gh = np.asarray(G)
    del G, Rg
    import gc
    gc.collect()
    B = K_.reduce_bank(ops, fom, Gh)
    G = jnp.asarray(Gh)
    del Gh
    Gr = G @ jnp.asarray(Tm)
    del G
    rot = P3.split_blocks(Gr, edges)
    del Gr
    jax.block_until_ready(rot)
    R_['bank'] = dict(rank=rank['rank'], condition_number=rank['condition_number'], seconds=time.perf_counter() - t0)
    idx = jnp.asarray(geom.idx)
    subjects = []
    for Rp in cfg['nmrom_R']:
        arm = dict(Rp=Rp, q=Rp, kind='linear', name=f'R{Rp}_linear', M=257)
        blocks = rot[:P3.nblocks(edges, Rp)]
        Bp = B @ jnp.asarray(Tm[:, :Rp])
        Qm, Rr = jnp.linalg.qr(Bp, mode='reduced')
        sv = np.asarray(jnp.linalg.svd(Rr, compute_uv=False))
        arm.update(qr_condition_number=float(sv[0] / sv[-1]))
        kern, _ = make_linear(geom)
        Qt = Qm.T

        def call(s, kern=kern, Qt=Qt, Rr=Rr, blocks=blocks):
            out_, row = query(lambda x: kern(x, idx, ops['P'], Qt, Rr, blocks), s)
            return np.asarray(out_[0], dtype=np.float64), row
        subjects.append(dict(name=arm['name'], family='nm-rom', arm=arm, call=call))
    cg = K_.make_gpu_cg(geom, int(cfg['cg_maxiter']))

    def cg_call(tol):
        def call(s):
            out_, row = query(lambda x: cg(x, tol), s)
            field, it, res = out_
            row.update(iterations=int(it), final_relative_residual=float(res), cg_converged=bool(float(res) <= tol))
            return np.asarray(field, dtype=np.float64), row
        return call
    cgs = [dict(name=f'cg_{t:g}', family='cg', tolerance=t, call=cg_call(t)) for t in cfg['cg_tols']]
    return dev, same, sources, subjects, cgs, K_.burn, geom


# ------------------------------------------------------------------ operators
def build_operators(cfg, problem, n, R_):
    sys.path.insert(0, str(HERE / 'ops'))
    import pops as OP
    R_['operator_environment'] = OP.configure()
    torch.set_default_dtype(torch.float64)
    mask = OP.domain_mask(problem, n)
    subjects = []
    for fam, info in cfg['operators'].items():
        rec = dict(family=fam, training_job=info.get('job'), training_result=info.get('result'))
        if info.get('checkpoint') is None or not Path(info['checkpoint']).exists():
            rec.update(status='not_available', reason=info.get('failure', 'no checkpoint'))
            R_['operators'].append(rec)
            continue
        try:
            net, norm, ck = OP.load(info['checkpoint'])
            assert ck['mesh'] == n and ck['problem'] == problem and ck['family'] == fam, (ck['mesh'], ck['problem'], ck['family'])
            rec.update(status='loaded', checkpoint_sha256=sha_file(info['checkpoint']), epoch=ck['epoch'], step=ck['step'],
                       config=ck['config'], real_parameter_count=sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters()))
        except Exception as exc:  # noqa: BLE001 - recorded as the cell's result
            rec.update(status='load_failed', reason=f'{type(exc).__name__}: {str(exc)[:500]}')
            R_['operators'].append(rec)
            continue

        def call(s, net=net, norm=norm):
            start = time.perf_counter()
            src = torch.from_numpy(np.asarray(s)).cuda()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                out = OP.predict(net, src[None], *norm, mask)[0]
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            field = out.cpu().numpy()
            end = time.perf_counter()
            return field, dict(total_seconds=end - start, input_seconds=t1 - start, fused_device_seconds=t2 - t1,
                               output_seconds=end - t2)
        # the query must run once to count; an OOM here is the cell's result
        subjects.append(dict(name=f'op_{fam}', family='operator', op_family=fam, call=call, rec=rec))
        R_['operators'].append(rec)
    return subjects


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    problem, n = cfg['problem'], int(cfg['intervals'])
    if cfg.get('xla_mem_fraction'):   # DESIGN A9: cap the JAX pool so the operators' queries fit beside it
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg['xla_mem_fraction'])
    if cfg.get('xla_flags'):     # DESIGN A7: set before the XLA backend initialises
        os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' ' + cfg['xla_flags']).strip()
    assert jax.default_backend() == 'gpu' and len(jax.devices()) == 1, jax.devices()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    class P3u:  # pbk3_core.gpu_uuid, copied (the square panel does not stage nmrom_ls)
        @staticmethod
        def gpu_uuid():
            import ctypes
            import uuid as _uuid
            cuda = ctypes.CDLL('libcuda.so.1')
            assert cuda.cuInit(0) == 0
            dev = ctypes.c_int()
            assert cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
            raw = (ctypes.c_ubyte * 16)()
            fn = getattr(cuda, 'cuDeviceGetUuid_v2', cuda.cuDeviceGetUuid)
            assert fn(ctypes.byref(raw), dev) == 0
            return 'GPU-' + str(_uuid.UUID(bytes=bytes(raw)))
    uuid0 = P3u.gpu_uuid()
    try:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader'], text=True)
        assert uuid0 in inventory, (uuid0, inventory)
    except FileNotFoundError:
        inventory = None
    R_ = dict(config=cfg, problem=problem, intervals=n, job_id=os.environ.get('SLURM_JOB_ID'),
              node=os.environ.get('SLURMD_NODENAME'), gpu=jax.devices()[0].device_kind, gpu_uuid=uuid0,
              nvidia_smi=inventory, jax_version=jax.__version__, torch_version=torch.__version__,
              sources={p.name: sha_file(p) for d in ('.', 'ops', 'nmrom_sq', 'nmrom_ls') for p in sorted((HERE / d).glob('*.py'))},
              xla_flags=os.environ.get('XLA_FLAGS'), xla_mem_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
              timing_contract='GPU query (fused_device_seconds): synchronised host->device source copy to device->host field '
                              'copy; A-B-A phases; burn-in before every invocation',
              operators=[], invocations=[], complete=False)
    save = lambda: dump(out / 'result.json', R_)
    build = build_square if problem == 'square' else build_lshape
    dev, same, sources, subjects, cgs, burn, _ = build(cfg, n, R_)
    save()
    print('NM-ROM + CG BUILT', round(time.perf_counter() - begin, 1), flush=True)
    ops = build_operators(cfg, problem, n, R_)
    saved = {}

    def record(sub, case, rep, phase, field, row, prev):
        finite = bool(np.isfinite(field).all())
        h = sha_array(field)
        key = (sub['name'], case)
        if key not in saved:
            saved[key] = (h, field if sub['family'] == 'operator' else None)
        dev_first = 0.0
        if sub['family'] == 'operator' and saved[key][1] is not None:
            dev_first = rel(field, saved[key][1]) if np.linalg.norm(saved[key][1]) > 0 else 0.0
        keep = {k: v for k, v in row.items() if k.endswith('seconds') or k in (
            'iterations', 'true_relative_residual', 'final_relative_residual', 'cg_converged')}
        return dict(name=sub['name'], family=sub['family'], case=case, rep=rep, phase=phase, previous=prev,
                    tolerance=sub.get('tolerance'), field_sha256=h, matches_first=bool(h == saved[key][0]),
                    relative_to_first=dev_first, finite=finite,
                    same_grid_error=rel(field, same[case]) if finite else float('nan'), **keep)

    # warm-up / compile (untimed). An operator whose query cannot run (OOM) is dropped and recorded.
    t = time.perf_counter()
    live_ops = []
    for sub in subjects + cgs:
        sub['call'](sources[0])
    for sub in ops:
        try:
            sub['call'](sources[0])
            live_ops.append(sub)
        except torch.OutOfMemoryError as exc:
            sub['rec'].update(status='query_out_of_memory', reason=str(exc)[:500])
            torch.cuda.empty_cache()
    R_['compile_warmup_seconds'] = time.perf_counter() - t
    print('WARMUP', round(R_['compile_warmup_seconds'], 1), [s['name'] for s in live_ops], flush=True)
    save()
    fast = subjects + live_ops
    order = np.random.default_rng(int(cfg['order_seed']))

    def phase_break(label):
        jax.block_until_ready(jnp.zeros(()))
        torch.cuda.synchronize()
        time.sleep(cfg['cooldown_seconds'])
        burn(cfg['phase_dummy_seconds'])
        R_.setdefault('phase_breaks', []).append(dict(before=label, at_seconds=time.perf_counter() - begin))

    def run_phase(label, subs, reps):
        prev = None
        for rep in range(reps):
            for case in range(len(dev)):
                for i in order.permutation(len(subs)):
                    sub = subs[int(i)]
                    burn(cfg['burn_seconds'])
                    assert P3u.gpu_uuid() == uuid0
                    field, row = sub['call'](sources[case])
                    R_['invocations'].append(record(sub, case, rep, label, field, row, prev))
                    prev = sub['name']
            print('PHASE', label, rep, round(time.perf_counter() - begin, 1), flush=True)
            save()

    phase_break('A1')
    run_phase('A1', fast, cfg['repetitions'])
    phase_break('B')
    run_phase('B', cgs, cfg['cg_repetitions'])
    phase_break('A2')
    run_phase('A2', fast, cfg['repetitions'])

    # ---------------------------------------------------------------- summary
    inv = R_['invocations']
    summ = {}
    for sub in fast + cgs:
        rows = [x for x in inv if x['name'] == sub['name']]
        per_case = [max(x['same_grid_error'] for x in rows if x['case'] == c) for c in range(len(dev))]
        ms = [x['fused_device_seconds'] * 1e3 for x in rows]
        a1 = [x['fused_device_seconds'] * 1e3 for x in rows if x['phase'] == 'A1']
        a2 = [x['fused_device_seconds'] * 1e3 for x in rows if x['phase'] == 'A2']
        summ[sub['name']] = dict(family=sub['family'], tolerance=sub.get('tolerance'),
                                 worst_error=float(max(per_case)), median_error=float(np.median(per_case)),
                                 gpu_ms=float(np.median(ms)), total_ms=float(np.median([x['total_seconds'] * 1e3 for x in rows])),
                                 gpu_ms_A1=float(np.median(a1)) if a1 else None, gpu_ms_A2=float(np.median(a2)) if a2 else None,
                                 samples=len(rows), deterministic=all(x['matches_first'] for x in rows),
                                 max_relative_to_first=float(max(x['relative_to_first'] for x in rows)),
                                 all_finite=all(x['finite'] for x in rows),
                                 cg_converged=all(x.get('cg_converged', True) for x in rows))
    R_['subjects'] = summ
    drift = {k: v['gpu_ms_A2'] / v['gpu_ms_A1'] for k, v in summ.items() if v['gpu_ms_A1'] and v['gpu_ms_A2']}
    R_['drift'] = dict(ratios=drift, limit=1.10, passed=bool(all(1 / 1.1 <= r <= 1.1 for r in drift.values())))
    exp = cfg.get('expected_nmrom_worst', {})
    R_['reproduction'] = {k: dict(expected=v, got=summ[k]['worst_error'], relative_difference=abs(summ[k]['worst_error'] - v) / v,
                                  passed=bool(abs(summ[k]['worst_error'] - v) / v <= 1e-6)) for k, v in exp.items() if k in summ}
    R_['gates'] = dict(nmrom_reproduces_table1=all(r['passed'] for r in R_['reproduction'].values()) if R_['reproduction'] else None,
                       nmrom_cg_deterministic=all(v['deterministic'] for v in summ.values() if v['family'] != 'operator'),
                       cg_converged=all(v['cg_converged'] for v in summ.values()),
                       device_guard=P3u.gpu_uuid() == uuid0, drift_A2_over_A1=R_['drift']['passed'])
    stats = jax.devices()[0].memory_stats() or {}
    R_['device_memory'] = dict(jax_peak_bytes=int(stats.get('peak_bytes_in_use', 0)), torch_peak_bytes=int(torch.cuda.max_memory_allocated()))
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    print('GATES', R_['gates'], flush=True)
    for k, v in summ.items():
        print(f"{k:14s} worst {v['worst_error']*100:8.3f}% median {v['median_error']*100:8.3f}% gpu {v['gpu_ms']:10.3f} ms", flush=True)
    print('P2D PANEL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
