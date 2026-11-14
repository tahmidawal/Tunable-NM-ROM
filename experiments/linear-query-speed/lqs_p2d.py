"""linear-query-speed, Poisson 2D: the paper's two Table-1 arms next to the compressed-reconstruction
variants, in one process on one GPU (PROTOCOL.md).

Paper path: `p2t_core.build_banks_host`, `pbk_core.make_trunc_linear`, `pbk_core.trunc_linear_query`
and the stage profile `p2t_solve.make_linear_stages`, imported unmodified from `paper_code/`.

--mode select : development cases (dev12) only; every variant of the protocol.
--mode test   : development gate re-check on dev12, reproduction gate on the test cohort against the
                stored paper values, then A-B-A timing on the test cohort for the frozen variants.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'paper_code'))

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import pbh_core as K_
import sep_common as sc
import hp_core as H
import pbk_core as P
import p2t_core
import p2t_solve as PS
import lqs_core as Q

ARMS = (('accurate', 512), ('fast', 128))
DEV_GATE = 1e-5


def vname(tau, prec, m, n):
    return f"T_tau{tau:g}_{prec}_{'full' if m == n else f'sub{m}'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--intervals', type=int, default=None)
    ap.add_argument('--mode', choices=('select', 'test'), required=True)
    ap.add_argument('--frozen', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=None)
    ap.add_argument('--cases', type=int, default=None, help='debug only: truncate the timed cohort')
    ap.add_argument('--allow-cpu', action='store_true', help='local debugging only; never a result')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    n = int(a.intervals or cfg['intervals'])
    backend = jax.default_backend()
    print(f'jax_backend={backend} x64=True precision={os.environ.get("JAX_DEFAULT_MATMUL_PRECISION")}', flush=True)
    assert a.allow_cpu or backend == 'gpu', backend
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    begin = time.perf_counter()
    try:
        smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                       '--format=csv,noheader'], text=True).strip()
    except Exception:
        smi = None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R_ = dict(lane='linear-query-speed', problem='poisson2d', mode=a.mode, intervals=n, config=cfg,
              commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
              host=os.uname().nodename, backend=backend, nvidia_smi=smi, jax_version=jax.__version__,
              debug_cpu=bool(a.allow_cpu), debug_case_limit=a.cases,
              timing_contract='paper scope: host f64 source in -> host f64 nodal field out; GPU-query = '
                              'fused_device_seconds (device source -> device f64 field, synced); A-B-A; '
                              '0.1 s burn before every invocation; random order within a case',
              complete=False)
    save = lambda: K_.dump(out / 'result.json', R_)

    # ---------------------------------------------------------------- frozen model and cohorts
    code = HERE / 'paper_code'
    m_ = cfg['model']
    params, codes, _ = sc.load_pkl(code / Path(m_['checkpoint_path']).name)
    codes = np.asarray(codes)
    K, Rw = codes.shape[1], int(np.asarray(params['h_lin']).shape[1])
    z = np.load(code / Path(cfg['prep_path']).name, allow_pickle=True)
    Tm = z['T']
    R_['prep_sha256'] = hashlib.sha256((code / Path(cfg['prep_path']).name).read_bytes()).hexdigest()
    R_['checkpoint_sha256'] = hashlib.sha256((code / Path(m_['checkpoint_path']).name).read_bytes()).hexdigest()
    dev12 = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                            C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    tc = cfg['test_cohort']
    cohorts = dict(dev12=dev12)
    if a.mode == 'test':
        cohorts['test'] = C.source_params(tc['seed'], tc['count'])
    R_['cohorts'] = {k: dict(count=len(v), sha256=H.sha_array(v)) for k, v in cohorts.items()}
    stored = None
    sp = HERE / 'paper_reference' / f'p2t{n}-stored.json'
    if sp.exists():
        stored = json.loads(sp.read_text())
        assert stored['dev12_sha256'] == R_['cohorts']['dev12']['sha256']
        if a.mode == 'test':
            assert stored['cohort_sha256'] == R_['cohorts']['test']['sha256']
    timed_name = 'test' if a.mode == 'test' else 'dev12'
    timed = cohorts[timed_name][:a.cases] if a.cases else cohorts[timed_name]
    same = {k: [PS.reference(p_, n) for p_ in v] for k, v in cohorts.items()}
    sources = {k: [C.full_source(n, p_) for p_ in v] for k, v in cohorts.items()}

    # ---------------------------------------------------------------- the paper bank and arms
    edges = [0] + sorted(cfg['R_ladder'])
    assert edges[-1] == Rw
    arms = [dict(label=label, Rp=Rp, name=f'R{Rp}_linear', M=4 * (K + Rp)) for label, Rp in ARMS]
    maxmode = 0
    for arm in arms:
        I, J, _ = H.mode_set(n, arm['M'])
        maxmode = max(maxmode, int(max(I.max(), J.max())) + 1)
    truth = np.stack([s_[1:-1, 1:-1] for s_ in same[timed_name]])
    rows_per_chunk = min(int(cfg['rows_per_chunk']), 128 if n >= 4096 else 256)
    bank = p2t_core.build_banks_host(params, n, rows_per_chunk, cfg['eval_rows'], maxmode, Tm, edges, truth,
                                     keep_orig=False)
    del truth
    assert bank['info']['retained_bank_rank'] == Rw, bank['info']
    R_['bank'] = bank['info']
    R_['floors'] = {str(k): K_.summarise(v) for k, v in bank['floors'].items()}
    print('BANK', round(bank['info']['build_seconds'], 1), flush=True)

    U0, V0, C0, col2, cinfo = Q.compress_bank(bank['rot'], n)
    R_['compression'] = cinfo
    print('COMPRESS', round(cinfo['seconds'], 1), cinfo['range_finder_relative_residual'], flush=True)

    if a.mode == 'select':
        grid = [(tau, prec, m) for tau in (1e-4, 1e-6, 1e-8) for prec in ('f64', 'f32')
                for m in (n, 256, 512, 1024) if m <= n]
        grid = list(dict.fromkeys(grid))
    else:
        frozen = json.loads(Path(a.frozen).read_text())
        grid = []
        for key in ('F', 'S'):
            f = frozen[key]
            grid.append((float(f['tau']), f['precision'], n if f['input'] == 'full' else min(int(f['input']), n)))
        R_['frozen'] = frozen
        R_['frozen_names'] = {key: vname(*grid[i], n) for i, key in enumerate(('F', 'S'))}
        grid += [(1e-8, 'f64', n), (1e-8, 'f32', n)]
        grid = list(dict.fromkeys(grid))

    subjects, paper, solve_only = [], [], []
    for arm in arms:
        ops = H.make_ops(bank, params, codes, n, arm['M'])
        Rp = arm['Rp']
        nb = P.nblocks(edges, Rp)
        Bp = ops['B'] @ jnp.asarray(Tm[:, :Rp])
        kern, Qt, Rr, linfo = P.make_trunc_linear(Bp, n, nb)
        arm.update(linfo)

        def pcall(s, ops=ops, kern=kern, Qt=Qt, Rr=Rr):
            field, row = P.trunc_linear_query(s, ops, kern, Qt, Rr, bank['rot'])
            row.pop('linear_coefficients')
            return field, row
        dec = jax.jit(lambda a_, rot, nb=nb: P.decode_blocks(rot, a_, n, nb))
        pprof = PS.make_linear_stages(ops, Qt, Rr, lambda a_, dec=dec: dec(a_, bank['rot']))
        paper.append(dict(name=arm['name'], arm=arm['label'], kind='paper', call=pcall, prof=pprof))

        Pm = Q.reduced_operator(Qt, Rr, ops['W'])
        Ih, Jh, Sh = np.asarray(ops['I']), np.asarray(ops['J']), np.asarray(ops['S'])
        tks = {}
        for tau, prec, m in grid:
            if tau not in tks:
                tks[tau] = Q.truncate(U0, V0, C0, col2, Rp, tau)
            tk = tks[tau]
            dtype = jnp.float64 if prec == 'f64' else jnp.float32
            args, st = Q.device_arrays(tk, Sh, Ih, Jh, Pm, n, m, dtype)
            kernel = Q.make_kernel(tk['p'], tk['q'], st, dtype)
            stages = Q.make_stages(tk['p'], tk['q'], st, dtype)
            name = f"{arm['name']}__{vname(tau, prec, m, n)}"

            def vcall(s, kernel=kernel, args=args):
                field, _, row = Q.query(s, kernel, args)
                return field, row
            subjects.append(dict(name=name, arm=arm['label'], kind='variant', variant=vname(tau, prec, m, n),
                                 tau=tau, precision=prec, input=('full' if m == n else m), call=vcall,
                                 prof=lambda src, stages=stages, args=args: stages(src, *args),
                                 info={k: v for k, v in tk.items() if k not in ('U', 'V', 'C')}))
        for m in sorted({m for _, _, m in grid}):
            Sx, st = Q.sine_rows(Sh, n, m)
            so_args = (jnp.asarray(Sx), jnp.asarray(Ih), jnp.asarray(Jh), jnp.asarray(Pm))
            so_k = Q.make_solve_only(st)

            def scall(s, so_k=so_k, so_args=so_args):
                _, _, row = Q.query(s, so_k, so_args, has_field=False)
                return None, row
            solve_only.append(dict(name=f"{arm['name']}__solve_only_{'full' if m == n else f'sub{m}'}",
                                   arm=arm['label'], kind='solve_only', input=('full' if m == n else m), call=scall))
        R_.setdefault('arms', []).append(arm)
    del U0, V0, C0
    R_['subjects'] = [dict({k: v for k, v in s.items() if k not in ('call', 'prof')}) for s in paper + subjects + solve_only]
    save()

    # ---------------------------------------------------------------- warm-up, development gate
    t = time.perf_counter()
    for sub in paper + subjects + solve_only:
        sub['call'](sources['dev12'][0])
    R_['compile_warmup_seconds'] = time.perf_counter() - t
    print('WARMUP', round(R_['compile_warmup_seconds'], 1), flush=True)

    def fields_of(sub, srcs):
        return [sub['call'](s)[0] for s in srcs]

    dev_rows = []
    for arm, psub in zip(arms, paper):
        pf = fields_of(psub, sources['dev12'])
        perr = [C.relative(f, r) for f, r in zip(pf, same['dev12'])]
        dev_rows.append(dict(name=psub['name'], arm=arm['label'], kind='paper', error=K_.summarise(np.array(perr))))
        for sub in [s for s in subjects if s['arm'] == arm['label']]:
            vf = fields_of(sub, sources['dev12'])
            devi = [C.relative(v, p_) for v, p_ in zip(vf, pf)]
            verr = [C.relative(f, r) for f, r in zip(vf, same['dev12'])]
            dev_rows.append(dict(name=sub['name'], arm=arm['label'], kind='variant', variant=sub['variant'],
                                 max_deviation_from_paper=float(max(devi)), deviations=devi,
                                 error=K_.summarise(np.array(verr)),
                                 max_abs_error_change=float(np.max(np.abs(np.array(verr) - np.array(perr)))),
                                 passes=bool(max(devi) <= DEV_GATE)))
            del vf
        del pf
    R_['dev_gate'] = dict(limit=DEV_GATE, rows=dev_rows)
    for r in dev_rows:
        if r['kind'] == 'variant':
            print('DEV', r['name'], f"{r['max_deviation_from_paper']:.3e}", r['passes'], flush=True)
    save()

    # ---------------------------------------------------------------- reproduction gate (test mode)
    pfields = {}
    for psub in paper:
        pfields[psub['name']] = fields_of(psub, sources[timed_name][:len(timed)])
    if a.mode == 'test':
        rows = []
        for psub in paper:
            err = [C.relative(f, r) for f, r in zip(pfields[psub['name']], same['test'])]
            row = dict(name=psub['name'], errors=err, worst=float(max(err)), mean=float(np.mean(err)))
            if stored is not None:
                ref = np.array(stored['arms'][psub['name']]['same_grid_error_by_case'][:len(err)])
                row.update(stored_worst=float(ref.max()), max_abs_difference=float(np.max(np.abs(np.array(err) - ref))),
                           field_hash_matches=int(sum(H.sha_array(f) == h for f, h in zip(
                               pfields[psub['name']], stored['arms'][psub['name']]['field_sha256_by_case']))),
                           passed=bool(np.max(np.abs(np.array(err) - ref)) <= 1e-9))
            rows.append(row)
        R_['repro_gate'] = dict(stored_available=stored is not None, limit=1e-9, rows=rows,
                                passed=(all(r['passed'] for r in rows) if stored is not None else None))
        print('REPRO', [(r['name'], r['worst'], r.get('max_abs_difference'), r.get('passed')) for r in rows], flush=True)
        save()

    # ---------------------------------------------------------------- A-B-A timing
    reps = int(a.reps or (cfg['repetitions'] if a.mode == 'test' else 3))
    order = np.random.default_rng(cfg['order_seed'])
    R_['invocations'] = []
    first = {}

    def phase_break(label):
        jax.block_until_ready(jnp.zeros(()))
        time.sleep(cfg['cooldown_seconds'])
        C.burn(cfg['phase_dummy_seconds'])
        R_.setdefault('phase_breaks', []).append(dict(before=label, at_seconds=time.perf_counter() - begin))

    def run_phase(label, subs, reps):
        prev = None
        for rep in range(reps):
            for case in range(len(timed)):
                for i in order.permutation(len(subs)):
                    sub = subs[int(i)]
                    C.burn(cfg['burn_seconds'])
                    field, row = sub['call'](sources[timed_name][case])
                    rec = dict(name=sub['name'], kind=sub['kind'], case=case, rep=rep, phase=label, previous=prev, **row)
                    if field is not None:
                        h = H.sha_array(field)
                        key = (sub['name'], case)
                        if key not in first:
                            first[key] = (h, C.relative(field, same[timed_name][case]),
                                          C.relative(field, pfields[sub['name'].split('__')[0]][case]))
                        rec.update(deterministic=bool(h == first[key][0]), same_grid_error=first[key][1],
                                   deviation_from_paper=first[key][2])
                    R_['invocations'].append(rec)
                    prev = sub['name']
            print('PHASE', label, rep, round(time.perf_counter() - begin, 1), flush=True)
            save()

    phase_break('paperA1')
    run_phase('paperA1', paper, reps)
    phase_break('variants')
    run_phase('variants', subjects + solve_only, reps)
    phase_break('paperA2')
    run_phase('paperA2', paper, reps)

    inv = R_['invocations']
    med = lambda name, phases, key='fused_device_seconds': float(np.median(
        [x[key] for x in inv if x['name'] == name and x['phase'] in phases]))
    summary = []
    for sub in paper + subjects + solve_only:
        rows = [x for x in inv if x['name'] == sub['name']]
        phases = ('paperA1', 'paperA2') if sub['kind'] == 'paper' else ('variants',)
        s = dict(name=sub['name'], arm=sub['arm'], kind=sub['kind'], variant=sub.get('variant'),
                 invocations=len(rows), median_fused_device_ms=1e3 * med(sub['name'], phases),
                 median_total_ms=1e3 * med(sub['name'], phases, 'total_seconds'),
                 median_input_ms=1e3 * med(sub['name'], phases, 'input_seconds'),
                 median_output_ms=1e3 * med(sub['name'], phases, 'output_seconds'),
                 q25_q75_fused_ms=[1e3 * float(v) for v in np.quantile(
                     [x['fused_device_seconds'] for x in rows], [0.25, 0.75])])
        if sub['kind'] == 'paper':
            a1, a2 = med(sub['name'], ('paperA1',)), med(sub['name'], ('paperA2',))
            s.update(A1_ms=1e3 * a1, A2_ms=1e3 * a2, drift_ratio=a2 / a1)
        if sub['kind'] != 'solve_only':
            errs = [first[(sub['name'], c)][1] for c in range(len(timed))]
            s.update(worst_error=float(max(errs)), mean_error=float(np.mean(errs)),
                     max_deviation_from_paper=float(max(first[(sub['name'], c)][2] for c in range(len(timed)))),
                     deterministic=bool(all(x['deterministic'] for x in rows)))
        summary.append(s)
    R_['summary'] = summary
    R_['timed_cohort'] = dict(name=timed_name, cases=len(timed), repetitions=reps)

    # ---------------------------------------------------------------- stage profile
    prof = []
    for sub in paper + subjects:
        for case in range(min(3, len(timed))):
            src = jax.device_put(sources[timed_name][case])
            for rep in range(1 + int(cfg['profile_reps'])):
                C.burn(cfg['burn_seconds'])
                field, st = sub['prof'](src)
                if rep:
                    prof.append(dict(name=sub['name'], case=case, rep=rep, **{k: float(v) for k, v in st.items()}))
            del src
    R_['profile'] = prof
    keys = sorted({k for p_ in prof for k in p_ if k not in ('name', 'case', 'rep')})
    R_['profile_median_ms'] = {sub['name']: {k: 1e3 * float(np.median([p_[k] for p_ in prof if p_['name'] == sub['name'] and k in p_]))
                                             for k in keys if any(k in p_ for p_ in prof if p_['name'] == sub['name'])}
                               for sub in paper + subjects}
    try:
        R_['device_memory'] = {k: int(v) for k, v in jax.devices()[0].memory_stats().items()
                               if k in ('peak_bytes_in_use', 'bytes_limit')}
    except Exception:
        R_['device_memory'] = None
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    for s in summary:
        print('SUMMARY', s['name'], f"{s['median_fused_device_ms']:.3f} ms",
              f"err {s.get('worst_error', float('nan')):.6f}", f"dev {s.get('max_deviation_from_paper', float('nan')):.2e}", flush=True)
    (out / 'COMPLETE').write_text(json.dumps(dict(elapsed=R_['elapsed_seconds'], job=R_['job_id'])))


if __name__ == '__main__':
    main()
