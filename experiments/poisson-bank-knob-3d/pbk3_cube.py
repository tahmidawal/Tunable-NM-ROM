"""poisson-bank-knob-3d, Poisson 3D: nested bank truncation R' of the frozen paper-p3d K16 model
(R = 128, trained at 32^3) on ONE mesh per allocation, with the CG grid in the same allocation.

Parent: `experiments/hires-poisson/hp3d_solve.py` (lean query with the weak source moments from one
orthonormal DST-I of the forcing, M = 512 weak sine tests, 3 LM starts; parity-gated there against
the unchanged `poisson.engine`) and paper-p3d `iterative_cg.engine(retain_history=False)` (the
efficient CG of the accepted Table-1 rows). New here: the rotated nested bank (pbk3_core), the R'
ladder x q arms, the linear rung q = R', unrotated parent arms for the parity gate, the unchanged
`poisson.engine` (dense projection, the Table-1 32^3/64^3 variant) as a reference arm where the dense
test matrix fits, and the sibling lane's phase structure (MAIN / SLOW / NEIGHBOUR / PROFILE).

Cohort: `development` = paper-p3d validation cohort (seed 920411, 16 cases);
        `final` = the reserved final cohort (seed 920499, 64 cases), frozen settings only.
Timing contract (the cube Table-1 scope): GPU query = `fused_device_seconds`, the device work between
the synchronised host->device copy of the f64 interior forcing and completion; `total_seconds` adds
both host transfers.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.fft import dstn
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl

import common as C
import poisson as PP
import shared_rom as S
import iterative_cg as CG
import pbk3_core as P3


def reference(n, f):
    k = np.arange(1, n)
    l = 4. * n * n * np.sin(np.pi * k / (2 * n)) ** 2
    lam = l[:, None, None] + l[None, :, None] + l[None, None, :]
    return dstn(dstn(np.asarray(f), type=1, norm='ortho', workers=8) / lam, type=1, norm='ortho', workers=8)


def timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t


def make_rom(model, n, triples, lam, q, starts, cfg):
    """hp3d_solve.lean_engine(projection=None) with the decode through column blocks and a = Lr c."""
    fit = S.lm(C.head, cfg['lm_budget'], cfg['lm_tolerance'])
    ti, tj, tk = (jnp.asarray(triples[:, a] - 1) for a in range(3))
    scale = jnp.asarray(1.0 / (n ** 1.5 * np.asarray(lam)))

    def front(f, qq, library):
        target = C.dst3(f)[ti, tj, tk] * scale
        tp = target - qq @ (qq.T @ target)
        order = jnp.argsort(jnp.sum((library - tp) ** 2, axis=1))[:starts]
        return target, tp, order

    def solve(p, ap, tp, order, codes):
        zs, stats = jax.vmap(lambda z: fit(p, ap, tp, z))(codes[order])
        which = jnp.argmin(stats[:, 3])
        return zs[which], stats[which], jnp.sum(stats[:, 0])

    def elim(p, z, target, a, qq, rr, directions, Lr):
        h = C.head(p, z)
        y = jsl.solve_triangular(rr, qq.T @ (target - a @ h), lower=False) if q else jnp.zeros(0)
        coef = h + directions @ y
        return (coef if Lr is None else Lr @ coef), coef

    def back(ac, blocks):
        return P3.decode(blocks, ac)

    @jax.jit
    def kernel(f, p, a, ap, qq, rr, directions, Lr, blocks, library, codes):
        target, tp, order = front(f, qq, library)
        z, st, its = solve(p, ap, tp, order, codes)
        ac, coef = elim(p, z, target, a, qq, rr, directions, Lr)
        return back(ac, blocks), st, coef, its
    return kernel, dict(front=jax.jit(front), solve=jax.jit(solve), elim=jax.jit(elim), back=jax.jit(back))


def make_linear(n, triples, lam):
    ti, tj, tk = (jnp.asarray(triples[:, a] - 1) for a in range(3))
    scale = jnp.asarray(1.0 / (n ** 1.5 * np.asarray(lam)))

    def front(f, Qt, Rr):
        return jsl.solve_triangular(Rr, Qt @ (C.dst3(f)[ti, tj, tk] * scale), lower=False)

    def back(ac, blocks):
        return P3.decode(blocks, ac)

    @jax.jit
    def kernel(f, Qt, Rr, blocks):
        return back(front(f, Qt, Rr), blocks)
    return kernel, dict(front=jax.jit(front), back=jax.jit(back))


def query(call, host_f):
    start = time.perf_counter()
    fj = jax.device_put(host_f)
    fj.block_until_ready()
    t1 = time.perf_counter()
    out = call(fj)
    jax.block_until_ready(out)
    t2 = time.perf_counter()
    out = jax.device_get(out)
    end = time.perf_counter()
    return out, dict(total_seconds=end - start, input_seconds=t1 - start,
                     fused_device_seconds=t2 - t1, output_seconds=end - t2)


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument('--config', required=True)
    ap_.add_argument('--out', required=True)
    ap_.add_argument('--smoke', action='store_true')
    a_ = ap_.parse_args()
    cfg = json.loads(Path(a_.config).read_text())
    here = Path(a_.config).resolve().parent
    if a_.smoke:
        cfg.update(intervals=16, repetitions=1, slow_repetitions=1, burn_seconds=0.001, case_count=3,
                   neighbour_cases=1, profile_reps=1, cg_fast=[0.1, 0.01], cg_slow=[1e-4],
                   neighbour_tolerance=1e-4, dense_engine_max=16)
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
    out = Path(a_.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)

    bankck = pickle.loads((here / 'bank.pkl').read_bytes())
    model = pickle.loads((here / 'head_K16.pkl').read_bytes())
    shas = dict(bank=P3.sha_file(here / 'bank.pkl'), head=P3.sha_file(here / 'head_K16.pkl'))
    assert shas == cfg['accepted_checkpoint_sha256'], shas
    mcfg = bankck['cfg']
    Rw = int(np.asarray(bankck['rotation']).shape[1])
    K = int(np.asarray(model['codes']).shape[1])
    prep = np.load(here / cfg['prep'], allow_pickle=False)
    Tm, Lm = prep['T'], prep['L']
    rinfo = json.loads(str(prep['rotation_info']))
    assert rinfo['checkpoint_sha256'] == shas
    np.testing.assert_array_equal(prep['directions'], np.asarray(model['directions']))

    role = cfg['cohort']
    coh = json.loads((here / 'cohorts.json').read_text())
    train = C.family(mcfg['train_seed'], mcfg['train_count'])
    if role == 'development':
        cases = C.family(mcfg['validation_seed'], mcfg['validation_count'])
        np.testing.assert_array_equal(cases, np.asarray(coh['validation_parameters']))
    else:
        assert role == 'final'
        frozen = json.loads((here / cfg['frozen_settings']).read_text())
        assert frozen['intervals'] == n and frozen['selected_on'] == 'development'
        cases = C.family(mcfg['reserved_final_seed'], cfg['final_count'])
        assert P3.sha_array(cases) == cfg['final_parameters_sha256'], P3.sha_array(cases)
    cases = cases[:cfg['case_count']]
    assert not any(np.allclose(t, s) for t in train for s in cases)
    R_ = dict(problem='cube', config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
              gpu=jax.devices()[0].device_kind, gpu_uuid=uuid0, nvidia_smi=inventory, x64=True,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
              smoke=bool(a_.smoke), intervals=n, K=K, R=Rw, checkpoint_sha256=shas,
              prep_sha256=P3.sha_file(here / cfg['prep']), rotation=rinfo,
              cohort=dict(role=role, parameters=cases.tolist(), sha256=P3.sha_array(cases), count=int(len(cases))),
              frozen_settings=(json.loads((here / cfg['frozen_settings']).read_text()) if role == 'final' else None),
              timing_contract=('host f64 interior forcing (n-1)^3 in -> host f64 interior field out for every '
                               'subject; GPU query = fused_device_seconds (the cube Table-1 scope); '
                               'total_seconds adds host transfers; randomised order; burn-in before every '
                               'invocation; slow CG in its own phase; neighbour (order-effect) gate'),
              arms=[], invocations=[], slow_invocations=[], neighbour=[], profile=[], parity=[], complete=False)
    save = lambda: P3.dump(out / 'result.json', R_)
    save()

    # ------------------------------------------------------------ truth + bank
    t0 = time.perf_counter()
    sources = [np.asarray(PP.source(n, p)) for p in cases]
    same = [reference(n, f) for f in sources]
    bank = C.bank_at(bankck['params'], n, cfg['field_chunk']) @ np.asarray(bankck['rotation'])
    bankj = jnp.asarray(bank)
    triples = C.modes(cfg['weak_tests'], n)
    lam = np.asarray(C.mode_eigenvalues(n, triples))
    ti, tj, tk = (jnp.asarray(triples[:, ax] - 1) for ax in range(3))
    col = jax.jit(lambda g: C.dst3(g.reshape((n - 1,) * 3))[ti, tj, tk] / n ** 1.5)
    operator = np.stack([np.asarray(col(bankj[:, r])) for r in range(Rw)], axis=1)       # M x R
    dense = n <= cfg['dense_engine_max']
    assembly = dict(route='dst')
    if dense:
        test, op_dense, projection, lam_d, triples_d = PP.assemble(bank, n, cfg['weak_tests'])
        del test
        assert np.array_equal(triples_d, triples) and np.allclose(lam_d, lam, rtol=1e-14, atol=0)
        assembly['dst_vs_dense_operator_relative'] = P3.rel(operator, op_dense)
        assert assembly['dst_vs_dense_operator_relative'] <= 1e-11, assembly
    Rg = P3.qr_r(bankj)
    U = np.stack([s_.ravel() for s_ in same])
    GtU = np.asarray(jnp.asarray(U) @ bankj)
    nu2 = np.sum(U * U, axis=1)
    edges = [0] + sorted(cfg['R_ladder'])
    assert edges[-1] == Rw
    floors = P3.floors_by_prefix(Rg, Tm, GtU, nu2, edges)
    rot = P3.split_blocks(bankj @ jnp.asarray(Tm), edges)
    orig = (bankj,)
    jax.block_until_ready(rot)
    R_['mesh'] = dict(intervals=n, unknowns=(n - 1) ** 3, weak_tests=int(len(triples)), assembly=assembly,
                      dense_engine=bool(dense), bank_sha256=P3.sha_array(bank), column_block_edges=edges,
                      bank_bytes_f64=int((n - 1) ** 3 * Rw * 8), setup_seconds=time.perf_counter() - t0)
    R_['floors'] = {str(k): P3.summarise(v) for k, v in floors.items()}
    save()
    print('MESH', n, {k: round(v['worst'] * 100, 4) for k, v in R_['floors'].items()},
          round(time.perf_counter() - begin, 1), flush=True)

    # ------------------------------------------------------------ arms
    p = jax.device_put(model['params'])
    codes = jnp.asarray(model['codes'])
    Dfull = np.asarray(model['directions'])
    M = int(len(triples))
    starts = int(cfg['initial_starts'])
    arms = []
    for Rp in sorted(cfg['R_ladder'], reverse=True):
        for q in P3.q_set(cfg['q_ladder'], Rp, K):
            arms.append(dict(Rp=Rp, q=q, kind='trunc', name=f'R{Rp}_q{q}', M=M))
        arms.append(dict(Rp=Rp, q=Rp, kind='linear', name=f'R{Rp}_linear', M=M))
    for q in cfg['parity_q']:
        arms.append(dict(Rp=Rw, q=q, kind='orig', name=f'orig_q{q}', M=M))
        if dense:
            arms.append(dict(Rp=Rw, q=q, kind='engine', name=f'engine_q{q}', M=M))
    subjects = []
    indices = np.arange((n - 1) ** 3)
    for arm in arms:
        Rp, q, kind = arm['Rp'], arm['q'], arm['kind']
        if kind == 'engine':
            fn = PP.engine(model, bank, op_dense, projection, indices, q, dict(cfg, initial_starts=starts))
            call = (lambda f, fn=fn: (lambda v: (v[0], v[1][:5], v[2], v[1][7]))(fn(f)))
            prof = None
        elif kind == 'linear':
            Bl = operator @ Tm[:, :Rp]
            Qm, Rr = np.linalg.qr(Bl, mode='reduced')
            sv = np.linalg.svd(Rr, compute_uv=False)
            arm.update(qr_condition_number=float(sv[0] / sv[-1]))
            kern, st = make_linear(n, triples, lam)
            blocks = rot[:P3.nblocks(edges, Rp)]
            Qt, Rrj = jnp.asarray(Qm.T), jnp.asarray(Rr)
            call = (lambda f, kern=kern, Qt=Qt, Rrj=Rrj, blocks=blocks: kern(f, Qt, Rrj, blocks))

            def prof(f, st=st, Qt=Qt, Rrj=Rrj, blocks=blocks):
                ac, t1 = timed(st['front'], f, Qt, Rrj)
                fld, t4 = timed(st['back'], ac, blocks)
                return fld, dict(project_and_start=t1, lm_solve=0.0, elimination_and_map=0.0, reconstruction=t4)
        else:
            Bt = operator if (kind == 'orig' or Rp == Rw) else operator @ (Tm[:, :Rp] @ Lm[:Rp])
            Lr = None if kind == 'orig' else jnp.asarray(Lm[:Rp])
            blocks = orig if kind == 'orig' else rot[:P3.nblocks(edges, Rp)]
            directions = Dfull[:, :q]
            if q:
                qq, rr = np.linalg.qr(Bt @ directions, mode='reduced')
                sv = np.linalg.svd(rr, compute_uv=False)
                assert sv[-1] > sv[0] * 1e-12, ('correction rank deficiency', arm, sv)
                arm.update(correction_condition_number=float(sv[0] / sv[-1]))
                reduced = Bt - qq @ (qq.T @ Bt)
            else:
                qq, rr, reduced = np.zeros((M, 0)), np.zeros((0, 0)), Bt
            library = C.head(p, codes) @ jnp.asarray(reduced).T
            kern, st = make_rom(model, n, triples, lam, q, starts, cfg)
            mats = tuple(jnp.asarray(v) for v in (Bt, reduced, qq, rr, directions))
            call = (lambda f, kern=kern, mats=mats, Lr=Lr, blocks=blocks, library=library:
                    kern(f, p, *mats, Lr, blocks, library, codes))

            def prof(f, st=st, mats=mats, Lr=Lr, blocks=blocks, library=library):
                Bt_, red_, qq_, rr_, dd_ = mats
                (target, tp, order), t1 = timed(st['front'], f, qq_, library)
                (z, _, _), t2 = timed(st['solve'], p, red_, tp, order, codes)
                (ac, _), t3 = timed(st['elim'], p, z, target, Bt_, qq_, rr_, dd_, Lr)
                fld, t4 = timed(st['back'], ac, blocks)
                return fld, dict(project_and_start=t1, lm_solve=t2, elimination_and_map=t3, reconstruction=t4)
        arm['family'] = {'trunc': 'nm-rom', 'orig': 'nm-rom-parent', 'engine': 'nm-rom-parent-engine',
                         'linear': 'linear-rung'}[kind]
        subjects.append(dict(name=arm['name'], family=arm['family'], arm=arm, call=call, prof=prof))
        R_['arms'].append(arm)
    cap = int(cfg['cg_iterations_per_interval']) * n
    fast = [dict(name=f'cg_{t:g}', family='cg', tolerance=t, call=CG.engine(n, float(t), cap, retain_history=False))
            for t in cfg['cg_fast']]
    slow = [dict(name=f'cg_{t:g}', family='cg', tolerance=t, call=CG.engine(n, float(t), cap, retain_history=False))
            for t in cfg['cg_slow']]
    R_['declared_subjects'] = [s['name'] for s in subjects + fast + slow]
    save()

    saved = {}

    def invoke(sub, case):
        out_, row = query(sub['call'], sources[case])
        if sub['family'] == 'cg':
            field, stats = out_
            c = CG.counters(stats, sub['tolerance'], cap)
            row.update(iterations=c['iterations'], cg_converged=c['cg_converged'],
                       true_relative_residual=c['cg_true_relative_residual'])
        elif sub['family'] == 'linear-rung':
            field = out_
        else:
            field, stats, coef, its = out_
            stats = np.asarray(stats)
            row.update(attempts=int(stats[0]), accepted=int(stats[1]), reason=int(stats[2]),
                       weak_residual=float(stats[3]), lm_gradient=float(stats[4]),
                       total_attempts_all_starts=int(its),
                       stationary=bool(int(stats[2]) == 1 and stats[4] <= cfg['lm_tolerance']))
        return np.asarray(field, dtype=np.float64).reshape((n - 1,) * 3), row

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

    for c_, f_ in enumerate(same):
        np.save(out / 'fields' / f'reference_same_case{c_}.npy', f_)
    t = time.perf_counter()
    for sub in subjects + fast + slow:
        invoke(sub, 0)
    R_['compile_warmup_seconds'] = time.perf_counter() - t
    print('WARMUP', len(subjects), round(R_['compile_warmup_seconds'], 1), flush=True)

    order = np.random.default_rng(cfg['order_seed'])
    main_set = subjects + fast
    prev = None
    for rep in range(cfg['repetitions']):
        for case in range(len(cases)):
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
        for case in range(len(cases)):
            for i in order.permutation(len(slow)):
                sub = slow[int(i)]
                P3.burn(cfg['burn_seconds'])
                assert P3.gpu_uuid() == uuid0
                field, row = invoke(sub, case)
                R_['slow_invocations'].append(record(sub, case, rep, 'slow', field, row, None))
        print('SLOW', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()
    long_sub = [s for s in slow if s['tolerance'] == cfg['neighbour_tolerance']][0]
    R_['neighbour'], R_['neighbour_gate'] = P3.neighbour_phase(subjects + (fast if cfg.get('neighbour_include_cg') else []), long_sub, invoke, record,
                                                             R_['invocations'], cfg, uuid0, order)
    gate_rows = [r for r in R_['neighbour_gate']['rows'] if r['variant'] == R_['neighbour_gate']['gate_variant']]
    og = R_['neighbour_gate']['order_gate']
    print('NEIGHBOUR', og['passed'], og['pooled_paired_ratio'], og['median_after_over_main'], og['max_arm_paired_ratio'], flush=True)
    save()
    for rep in range(cfg['profile_reps']):
        for case in range(len(cases)):
            for i in order.permutation(len(subjects)):
                sub = subjects[int(i)]
                if sub['prof'] is None:
                    continue
                P3.burn(cfg['burn_seconds'])
                fj = jax.device_put(sources[case])
                fj.block_until_ready()
                fld, stages = sub['prof'](fj)
                fld = np.asarray(jax.device_get(fld), dtype=np.float64).reshape((n - 1,) * 3)
                R_['profile'].append(dict(name=sub['name'], case=case, rep=rep, **stages,
                                          field_relative_to_fused=P3.rel(fld, np.load(
                                              out / 'fields' / f"{sub['name']}_case{case}.npy"))))
        print('PROFILE', rep, round(time.perf_counter() - begin, 1), flush=True)
    save()
    for q in cfg['parity_q']:
        pairs = [(f'R{Rw}_q{q}', f'orig_q{q}', cfg['parity_limit'])]
        if dense:
            pairs.append((f'orig_q{q}', f'engine_q{q}', cfg['engine_parity_limit']))
        for cand, base, limit in pairs:
            worst = max(P3.rel(np.load(out / 'fields' / f'{cand}_case{c}.npy'),
                               np.load(out / 'fields' / f'{base}_case{c}.npy')) for c in range(len(cases)))
            R_['parity'].append(dict(candidate=cand, baseline=base, worst_field_relative=worst, limit=limit,
                                     passed=bool(worst <= limit)))
            print('PARITY', R_['parity'][-1], flush=True)
    stats = jax.devices()[0].memory_stats() or {}
    R_['device_memory'] = {k: int(v) for k, v in stats.items() if k in ('peak_bytes_in_use', 'bytes_limit')}
    allinv = R_['invocations'] + R_['slow_invocations'] + R_['neighbour']
    R_['stationarity_by_arm'] = {s_['name']: dict(
        stationary=sum(bool(x.get('stationary', True)) for x in allinv if x['name'] == s_['name']),
        invocations=sum(1 for x in allinv if x['name'] == s_['name'])) for s_ in subjects}
    R_['gates'] = dict(
        parity=bool(R_['parity']) and all(x['passed'] for x in R_['parity']),
        deterministic=all(x['matches_saved_field'] for x in allinv),
        cg_converged=all(x.get('cg_converged', True) for x in allinv),
        rom_stationary_parent_and_full_R=all(x.get('stationary', True) for x in allinv if x['Rp'] == Rw),
        neighbour=R_['neighbour_gate']['passed'],
        profile_matches_fused=all(x['field_relative_to_fused'] <= 1e-10 for x in R_['profile']),
        device_guard=P3.gpu_uuid() == uuid0)
    print('GATES', R_['gates'], flush=True)
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / ('COMPLETE' if all(R_['gates'].values()) else 'COMPLETE-WITH-FAILED-GATES')).write_text(
        json.dumps(R_['gates']) + '\n')
    print('PBK3 CUBE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
