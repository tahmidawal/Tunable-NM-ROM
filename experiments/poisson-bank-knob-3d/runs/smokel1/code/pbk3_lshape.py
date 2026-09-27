"""poisson-bank-knob-3d, L-shaped Poisson: nested bank truncation R' of the frozen head_sdf_R512_K16
model on ONE mesh per allocation, with the CG grid in the same allocation.

Parent: `experiments/hires-poisson/hpl_solve.py` (its lean ROM kernel, SuperLU same-grid reference,
matrix-free GPU CG, M = 257 L-shape eigenmode weak tests, 32 development sources). Changes:
  * the rotated nested bank (pbk3_core), arms over the R' ladder x q, the linear rung q = R'
    (defined only where M > R'), parent unrotated arms for the parity gate;
  * no 2n fine reference (same-grid error only; Table 1 reports same-grid error);
  * phases as the sibling lane (`poisson-bank-knob/pbk_solve.py`): MAIN (all ROM arms + fast CG,
    randomised, burn-in before every invocation), SLOW (tight CG), NEIGHBOUR (order-effect gate),
    PROFILE (stage split).
Timing contract (the L-shape series' scope in Table 1): host (N+1)^2 f64 nodal source in -> host
(N+1)^2 f64 nodal field out; `total_seconds` = complete query, `fused_device_seconds` = device work.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import sep_common as sc
import arms as A
import lsh_core as K_
import pbk3_core as P3


def timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t


def make_rom(head, geom, trust, budget, gtol, linear, q, nb_blocks):
    """hpl_solve.make_rom_kernel(lean=True) with the decode through the given (rotated) column blocks
    and a = Lr c (Lr = None: unrotated parent)."""
    nfull = (geom.N + 1) ** 2
    lm = A.make_stationary_lm(lambda z, fp, Bp: Bp @ head(z) - fp, budget, trust, gtol, linear)

    def front(source_full, idx, P, Q, predictions):
        f = source_full.reshape(-1)[idx]
        fm = P @ f
        fp = fm - Q @ (Q.T @ fm) if q else fm
        index = jnp.argmin(jnp.sum((predictions - fp[None, :]) ** 2, axis=1))
        return fm, fp, index

    def solve(z0, fp, Bp):
        return lm(z0, (fp, Bp), 0.)

    def elim(z, fm, Bt, Q, Rq, C, Lr):
        h = head(z)
        if q:
            y = jax.scipy.linalg.solve_triangular(Rq, Q.T @ (fm - Bt @ h), lower=False)
            coeff = h + C @ y
        else:
            y = jnp.zeros((0,))
            coeff = h
        return (coeff if Lr is None else Lr @ coeff), y

    def back(a, blocks, idx):
        u = P3.decode(blocks, a)
        return jnp.zeros((nfull,)).at[idx].set(u).reshape(geom.N + 1, geom.N + 1)

    @jax.jit
    def kernel(source_full, idx, P, Bt, Bp, Q, Rq, C, Lr, blocks, predictions, codes):
        fm, fp, index = front(source_full, idx, P, Q, predictions)
        z, rn, it, reason, gn = solve(codes[index], fp, Bp)
        a, y = elim(z, fm, Bt, Q, Rq, C, Lr)
        return back(a, blocks, idx), rn, it, reason, gn, index, y

    stages = dict(front=jax.jit(front), solve=jax.jit(solve), elim=jax.jit(elim), back=jax.jit(back))
    return kernel, stages


def make_linear(geom):
    nfull = (geom.N + 1) ** 2

    def front(source_full, idx, P, Qt, Rr):
        fm = P @ source_full.reshape(-1)[idx]
        return jax.scipy.linalg.solve_triangular(Rr, Qt @ fm, lower=False)

    def back(a, blocks, idx):
        return jnp.zeros((nfull,)).at[idx].set(P3.decode(blocks, a)).reshape(geom.N + 1, geom.N + 1)

    @jax.jit
    def kernel(source_full, idx, P, Qt, Rr, blocks):
        a = front(source_full, idx, P, Qt, Rr)
        return back(a, blocks, idx), a
    return kernel, dict(front=jax.jit(front), back=jax.jit(back))


def query(call, host_source):
    start = time.perf_counter()
    src = jax.device_put(host_source)
    src.block_until_ready()
    t1 = time.perf_counter()
    out = call(src)
    jax.block_until_ready(out)
    t2 = time.perf_counter()
    out = jax.device_get(out)
    end = time.perf_counter()
    return out, dict(total_seconds=end - start, input_seconds=t1 - start,
                     fused_device_seconds=t2 - t1, output_seconds=end - t2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    if a.smoke:
        cfg.update(intervals=64, repetitions=1, slow_repetitions=1, burn_seconds=0.001, case_count=3,
                   neighbour_cases=1, profile_reps=1, lm_budget=60, cg_fast=[0.1, 0.01], cg_slow=[1e-4],
                   neighbour_tolerance=1e-4)
    n = int(cfg['intervals'])
    assert jax.default_backend() == 'gpu' and len(jax.devices()) == 1, jax.devices()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    uuid0 = P3.gpu_uuid()
    try:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                             '--format=csv,noheader'], text=True)
        assert uuid0 in inventory, (uuid0, inventory)
    except FileNotFoundError:
        inventory = None
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    R_ = dict(problem='lshape', config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
              gpu=jax.devices()[0].device_kind, gpu_uuid=uuid0, nvidia_smi=inventory, x64=True,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
              smoke=bool(a.smoke), intervals=n,
              timing_contract=('host (N+1)^2 f64 nodal source in -> host (N+1)^2 f64 nodal field out for every '
                               'subject; total_seconds = complete query (the L-shape Table-1 scope); '
                               'fused_device_seconds = device work; randomised order; burn-in before every '
                               'invocation; slow CG in its own phase; neighbour (order-effect) gate'),
              arms=[], invocations=[], slow_invocations=[], neighbour=[], profile=[], parity=[],
              references=[], complete=False)
    save = lambda: P3.dump(out / 'result.json', R_)

    # ---------------------------------------------------------------- model + rotation
    models = {m['id']: m for m in json.loads((here / 'models.json').read_text())}
    m = models[cfg['model']]
    assert P3.sha_file(here / m['checkpoint']) == m['checkpoint_sha256']
    params, codes, ckcfg = sc.load_pkl(here / m['checkpoint'])
    basis = np.load(here / m['basis'])
    codes = np.asarray(codes)
    np.testing.assert_array_equal(codes, basis['training_latents'])
    Rw, K = int(np.asarray(params['h_lin']).shape[1]), int(codes.shape[1])
    prep = np.load(here / cfg['prep'], allow_pickle=False)
    Tm, Lm = prep['T'], prep['L']
    R_['rotation'] = json.loads(str(prep['rotation_info']))
    R_['prep_sha256'] = P3.sha_file(here / cfg['prep'])
    R_['checkpoint_sha256'] = m['checkpoint_sha256']
    assert R_['rotation']['checkpoint_sha256'] == m['checkpoint_sha256']
    Cfull = np.asarray(basis['coefficient_directions'])

    cc = cfg['cohorts']
    dev, dinfo = K_.cohort(cc['development']['seed'], cc['development']['draw'], cc['development']['count'])
    dev = dev[:cfg['case_count']]
    train_draws, _ = K_.cohort(cc['training']['seed'], cc['training']['draw'], cc['training']['count'])
    assert not any(np.allclose(t, s) for t in train_draws for s in dev)
    R_['cohort'] = dict(**dinfo, used=int(len(dev)), parameters=dev.tolist(), role='development',
                        note='the 32 development sources of the lshape / hires-poisson (hpl32) lanes')
    save()

    # ---------------------------------------------------------------- FOM, references, tests
    geom = K_.Geometry(n, 'lshape')
    fom = K_.FOM(geom, build_ic0=False)
    ops = K_.weak_ops(fom, cfg['requested_modes'], cfg['mode_seed'])
    M = int(cfg['requested_modes'])
    R_['fom'] = dict(unknowns=geom.n, splu_factor_seconds=fom.factor_seconds, splu_lu_nnz=fom.lu_nnz,
                     weak_modes=ops['info'])
    same, sources = {}, []
    for case, q in enumerate(dev):
        u, resid, floor = fom.reference(K_.source_interior(geom, q))
        assert resid <= max(cfg['reference_residual_limit'], floor), (case, resid, floor)
        same[case] = geom.scatter(u)
        np.save(out / 'fields' / f'reference_same_case{case}.npy', same[case])
        sources.append(K_.source_full(geom, q))
        R_['references'].append(dict(case=case, same_residual=resid, same_roundoff_floor=floor))
    save()
    print('FOM + MODES + REFERENCES', round(time.perf_counter() - begin, 1), flush=True)

    # ---------------------------------------------------------------- banks
    t0 = time.perf_counter()
    G = K_.bank_of(K_.make_features(ckcfg['factor'], int(ckcfg['n_enrich'])), params, geom)
    Rg, rank = K_.bank_r(G)
    assert rank['rank_valid'], rank
    Udev = np.stack([geom.gather(same[c]) for c in range(len(dev))])
    GtU = np.asarray(jnp.asarray(Udev) @ G)
    nu2 = np.sum(Udev * Udev, axis=1)
    edges = [0] + sorted(cfg['R_ladder'])
    assert edges[-1] == Rw
    floors = P3.floors_by_prefix(Rg, Tm, GtU, nu2, edges)
    B = K_.reduce_bank(ops, fom, G)                       # P A G  (M x R)
    Gr = G @ jnp.asarray(Tm)
    rot = P3.split_blocks(Gr, edges)
    del Gr
    jax.block_until_ready(rot)
    keep_orig = bool(cfg['keep_orig'])
    orig = (G,) if keep_orig else None
    if not keep_orig:
        del G
    R_['bank'] = dict(rank=rank['rank'], condition_number=rank['condition_number'], column_block_edges=edges,
                      bank_bytes_f64=int(geom.n * Rw * 8), kept_original=keep_orig,
                      seconds=time.perf_counter() - t0)
    R_['floors'] = {str(k): P3.summarise(v) for k, v in floors.items()}
    save()
    print('BANK', {k: round(v['worst'] * 100, 3) for k, v in R_['floors'].items()}, flush=True)

    # ---------------------------------------------------------------- arms
    head = K_.head_of(params)
    trust = float(np.max(np.linalg.norm(codes - codes.mean(0), axis=1)))
    linear = 'gj' if K <= cfg['gauss_jordan_max'] else 'lu'
    idx = jnp.asarray(geom.idx)
    codes_j = jnp.asarray(codes)
    arms = []
    for Rp in sorted(cfg['R_ladder'], reverse=True):
        for q in P3.q_set(cfg['q_ladder'], Rp, K):
            if K + q < M:
                arms.append(dict(Rp=Rp, q=q, kind='trunc', name=f'R{Rp}_q{q}', M=M))
        if Rp < M:
            arms.append(dict(Rp=Rp, q=Rp, kind='linear', name=f'R{Rp}_linear', M=M))
        else:
            R_.setdefault('not_constructible', []).append(
                dict(name=f'R{Rp}_linear', reason=f'linear rung needs M > R\' (M = {M}, R\' = {Rp})'))
    if keep_orig:
        for q in cfg['parity_q']:
            arms.append(dict(Rp=Rw, q=q, kind='orig', name=f'orig_q{q}', M=M))
    subjects = []
    for arm in arms:
        Rp, q, kind = arm['Rp'], arm['q'], arm['kind']
        blocks = rot[:P3.nblocks(edges, Rp)] if kind != 'orig' else orig
        if kind == 'linear':
            Bp = B @ jnp.asarray(Tm[:, :Rp])
            Qm, Rr = jnp.linalg.qr(Bp, mode='reduced')
            sv = np.asarray(jnp.linalg.svd(Rr, compute_uv=False))
            arm.update(qr_condition_number=float(sv[0] / sv[-1]))
            kern, st = make_linear(geom)
            Qt = Qm.T
            call = (lambda s, kern=kern, Qt=Qt, Rr=Rr, blocks=blocks: kern(s, idx, ops['P'], Qt, Rr, blocks))

            def prof(s, st=st, Qt=Qt, Rr=Rr, blocks=blocks):
                aa, t1 = timed(st['front'], s, idx, ops['P'], Qt, Rr)
                fld, t4 = timed(st['back'], aa, blocks, idx)
                return fld, dict(project_and_start=t1, lm_solve=0.0, elimination_and_map=0.0, reconstruction=t4)
        else:
            Bt = B if (kind == 'orig' or Rp == Rw) else B @ jnp.asarray(Tm[:, :Rp] @ Lm[:Rp])
            Lr = None if kind == 'orig' else jnp.asarray(Lm[:Rp])
            if q:
                C = jnp.asarray(Cfull[:, :q])
                Q, Rq = jnp.linalg.qr(Bt @ C, mode='reduced')
                sv = np.asarray(jnp.linalg.svd(Rq, compute_uv=False))
                qrank = int(np.count_nonzero(sv > sv[0] * M * np.finfo(float).eps))
                assert qrank == q, (arm, qrank)
                arm.update(correction_condition_number=float(sv[0] / sv[-1]))
                Bp = Bt - Q @ (Q.T @ Bt)
            else:
                C, Q, Rq, Bp = jnp.zeros((Rw, 0)), jnp.zeros((M, 0)), jnp.zeros((0, 0)), Bt
            pred = jax.jit(jax.vmap(lambda z, Bp: Bp @ head(z), in_axes=(0, None)))(codes_j, Bp)
            kern, st = make_rom(head, geom, trust, cfg['lm_budget'], cfg['stationarity_tolerance'], linear, q,
                                len(blocks))
            args = (idx, ops['P'], Bt, Bp, Q, Rq, C, Lr, blocks, pred, codes_j)
            call = (lambda s, kern=kern, args=args: kern(s, *args))

            def prof(s, st=st, Bt=Bt, Bp=Bp, Q=Q, Rq=Rq, C=C, Lr=Lr, blocks=blocks, pred=pred):
                (fm, fp, index), t1 = timed(st['front'], s, idx, ops['P'], Q, pred)
                sol, t2 = timed(st['solve'], codes_j[index], fp, Bp)
                (aa, _), t3 = timed(st['elim'], sol[0], fm, Bt, Q, Rq, C, Lr)
                fld, t4 = timed(st['back'], aa, blocks, idx)
                return fld, dict(project_and_start=t1, lm_solve=t2, elimination_and_map=t3, reconstruction=t4)
        arm['family'] = {'trunc': 'nm-rom', 'orig': 'nm-rom-parent', 'linear': 'linear-rung'}[kind]
        subjects.append(dict(name=arm['name'], family=arm['family'], arm=arm, call=call, prof=prof))
        R_['arms'].append(arm)
    cg = K_.make_gpu_cg(geom, cfg['cg_maxiter'])

    def cg_call(tol):
        return lambda s: cg(s, tol)
    fast = [dict(name=f'cg_{t:g}', family='cg', tolerance=t, call=cg_call(t)) for t in cfg['cg_fast']]
    slow = [dict(name=f'cg_{t:g}', family='cg', tolerance=t, call=cg_call(t)) for t in cfg['cg_slow']]
    R_['declared_subjects'] = [s['name'] for s in subjects + fast + slow]
    save()

    saved = {}

    def invoke(sub, case):
        out_, row = query(sub['call'], sources[case])
        if sub['family'] == 'cg':
            field, it, res = out_
            row.update(iterations=int(it), final_relative_residual=float(res),
                       cg_converged=bool(float(res) <= sub['tolerance']))
        elif sub['family'] == 'linear-rung':
            field = out_[0]
        else:
            field, rn, it, reason, gn, index, y = out_
            row.update(residual=float(rn), iterations=int(it), reason=int(reason), stationarity=float(gn),
                       stationary=bool(int(reason) == 4), selected_code_index=int(index))
        return np.asarray(field, dtype=np.float64), row

    def record(sub, case, rep, phase, field, row, prev):
        assert np.isfinite(field).all(), sub['name']
        h = P3.sha_array(field)
        key = (sub['name'], case)
        if key not in saved:
            np.save(out / 'fields' / f"{sub['name']}_case{case}.npy", field)
            saved[key] = h
        arm = sub.get('arm', {})
        return dict(name=sub['name'], family=sub['family'], case=case, rep=rep, phase=phase, previous=prev,
                    Rp=arm.get('Rp'), q=arm.get('q'), tolerance=sub.get('tolerance'), field_sha256=h,
                    matches_saved_field=bool(h == saved[key]), same_grid_error=P3.rel(field, same[case]), **row)

    t = time.perf_counter()
    for sub in subjects + fast + slow:
        invoke(sub, 0)
    R_['compile_warmup_seconds'] = time.perf_counter() - t
    print('WARMUP', len(subjects), round(R_['compile_warmup_seconds'], 1), flush=True)

    order = np.random.default_rng(cfg['order_seed'])
    main_set = subjects + fast
    prev = None
    for rep in range(cfg['repetitions']):
        for case in range(len(dev)):
            for i in order.permutation(len(main_set)):
                sub = main_set[int(i)]
                P3.burn(cfg['burn_seconds'])
                assert P3.gpu_uuid() == uuid0
                field, row = invoke(sub, case)
                R_['invocations'].append(record(sub, case, rep, 'main', field, row, prev))
                prev = sub['name']
        print('MAIN', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()
    for rep in range(cfg['slow_repetitions']):
        for case in range(len(dev)):
            for i in order.permutation(len(slow)):
                sub = slow[int(i)]
                P3.burn(cfg['burn_seconds'])
                assert P3.gpu_uuid() == uuid0
                field, row = invoke(sub, case)
                R_['slow_invocations'].append(record(sub, case, rep, 'slow', field, row, None))
        print('SLOW', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()
    # ---- neighbour (order-effect) gate: each ROM arm immediately after a long CG solve
    long_sub = [s for s in slow if s['tolerance'] == cfg['neighbour_tolerance']][0]
    ncase = range(cfg['neighbour_cases'])
    for case in ncase:
        for i in order.permutation(len(subjects)):
            sub = subjects[int(i)]
            invoke(long_sub, case)
            P3.burn(cfg['burn_seconds'])
            field, row = invoke(sub, case)
            R_['neighbour'].append(record(sub, case, 0, 'neighbour', field, row, long_sub['name']))
    gate_rows = []
    for sub in subjects:
        for scope in ('total_seconds', 'fused_device_seconds'):
            base = np.median([x[scope] for x in R_['invocations'] if x['name'] == sub['name'] and x['case'] in ncase])
            after = np.median([x[scope] for x in R_['neighbour'] if x['name'] == sub['name']])
            gate_rows.append(dict(name=sub['name'], scope=scope, main_median=float(base),
                                  after_long_median=float(after), ratio=float(after / base)))
    R_['neighbour_gate'] = dict(rows=gate_rows, limit=cfg['neighbour_limit'], neighbour=long_sub['name'],
                                cases=len(ncase),
                                gate_scope=cfg['gate_scope'],
                                passed=bool(all(r['ratio'] <= cfg['neighbour_limit'] for r in gate_rows
                                                if r['scope'] == cfg['gate_scope'])))
    print('NEIGHBOUR', R_['neighbour_gate']['passed'], max(r['ratio'] for r in gate_rows), flush=True)
    save()
    # ---- stage profile
    for rep in range(cfg['profile_reps']):
        for case in range(len(dev)):
            for i in order.permutation(len(subjects)):
                sub = subjects[int(i)]
                P3.burn(cfg['burn_seconds'])
                src = jax.device_put(sources[case])
                src.block_until_ready()
                fld, stages = sub['prof'](src)
                fld = np.asarray(jax.device_get(fld))
                R_['profile'].append(dict(name=sub['name'], case=case, rep=rep, **stages,
                                          field_relative_to_fused=P3.rel(fld, np.load(
                                              out / 'fields' / f"{sub['name']}_case{case}.npy"))))
        print('PROFILE', rep, round(time.perf_counter() - begin, 1), flush=True)
    save()
    # ---- parity: rotated R' = R arms against the unrotated parent kernel
    if keep_orig:
        for q in cfg['parity_q']:
            worst = max(P3.rel(np.load(out / 'fields' / f'R{Rw}_q{q}_case{c}.npy'),
                               np.load(out / 'fields' / f'orig_q{q}_case{c}.npy')) for c in range(len(dev)))
            R_['parity'].append(dict(candidate=f'R{Rw}_q{q}', baseline=f'orig_q{q}', worst_field_relative=worst,
                                     limit=cfg['parity_limit'], passed=bool(worst <= cfg['parity_limit'])))
            print('PARITY', R_['parity'][-1], flush=True)
    stats = jax.devices()[0].memory_stats() or {}
    R_['device_memory'] = {k: int(v) for k, v in stats.items() if k in ('peak_bytes_in_use', 'bytes_limit')}
    allinv = R_['invocations'] + R_['slow_invocations'] + R_['neighbour']
    R_['gates'] = dict(
        parity=bool(R_['parity']) and all(p['passed'] for p in R_['parity']),
        deterministic=all(x['matches_saved_field'] for x in allinv),
        cg_converged=all(x.get('cg_converged', True) for x in allinv),
        rom_stationary_parent_and_full_R=all(x.get('stationary', True) for x in allinv if x['Rp'] == Rw),
        neighbour=R_['neighbour_gate']['passed'],
        profile_matches_fused=all(x['field_relative_to_fused'] <= 1e-10 for x in R_['profile']),
        device_guard=P3.gpu_uuid() == uuid0)
    R_['stationarity_by_arm'] = {s_['name']: dict(
        stationary=sum(bool(x.get('stationary', True)) for x in allinv if x['name'] == s_['name']),
        invocations=sum(1 for x in allinv if x['name'] == s_['name'])) for s_ in subjects}
    print('GATES', R_['gates'], flush=True)
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / ('COMPLETE' if all(R_['gates'].values()) else 'COMPLETE-WITH-FAILED-GATES')).write_text(
        json.dumps(R_['gates']) + '\n')
    print('PBK3 LSHAPE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
