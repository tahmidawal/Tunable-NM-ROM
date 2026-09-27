"""heat-bank-knob panel: one allocation, every mesh of the config in sequence. For each mesh: the nested rotated bank,
every (R', q, stepping) NM-ROM arm, the linear rung per R', the unrotated parent arms (parity), the CN-CG FOM grid and
the DST control, on a validation cohort (selection) and the held-out cohort (evaluation). See DESIGN.md.

Timing: GPU query scope (supplied field on device -> all output fields on device, blocked). Model arms and FOM arms
are timed in separate phases; inside each phase every repetition visits the arms in a fresh random order with a
GPU burn-in before every invocation. Neighbour (order-effect) gate: each model arm re-timed right after a CG solve.
"""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import core as C
import hbk_core as K


def gpu_identity():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        return f'unavailable: {exc}'


def timed(fn, *args):
    begin = time.perf_counter(); out = C.block(fn(*args)); return out, time.perf_counter() - begin


FP = jax.jit(lambda f: jnp.stack((jnp.sum(f.reshape(-1)[::997]), jnp.max(jnp.abs(f)), jnp.sum(f.reshape(-1)[5::1009] ** 2))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text()); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    smoke = cfg.get('local_smoke', False)
    assert jax.default_backend() == 'gpu' and jax.config.jax_enable_x64 and cfg['repetitions'] >= (1 if smoke else 5)
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    print('jax_backend=gpu x64=True precision=highest', flush=True)
    here = Path(__file__).resolve().parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob('*.py'))}
    sources[Path(args.config).name] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    model = C.load_model(cfg['model'], here / 'inputs'); d = model['d']; times = cfg['times']; nu = cfg['diffusivity']
    prep = np.load(here / cfg['prep']); T, L = prep['T'], prep['L']
    rot_info = json.loads(str(prep['rotation_info']))
    assert K.sha_array(T) == rot_info['T_sha256'] and K.sha_array(L) == rot_info['L_sha256']
    tj = json.loads((here / 'inputs' / Path(cfg['model']['bank']).parent / 'training.json').read_text())
    assert rot_info['training_draws_sha'] == tj['train_draws_sha'], 'rotation not built from the recorded training draws'
    result = dict(schema='heat-bank-knob-panel-v1', config=cfg, source_sha256=sources, model_sha256=model['sha256'],
                  prep_sha256=hashlib.sha256((here / cfg['prep']).read_bytes()).hexdigest(), rotation=rot_info,
                  source_commit=os.environ.get('SOURCE_COMMIT'), complete=False, meshes=[],
                  metadata=dict(job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'), gpu=gpu_identity(),
                                jax=jax.__version__, backend='gpu', x64=True, precision='highest', local_smoke=smoke),
                  query_contract='Supplied full initial field on the GPU to all six full output fields on the GPU, blocked. '
                                 'Offline bank/rotation/operator assembly and compilation excluded and recorded. Host transfers excluded for every method.')
    cohorts = []; draws = []
    for name, seed, count in cfg['cohorts']:
        dr = C.family(model['family'], seed, count); cohorts += [name] * len(dr); draws.append(dr)
    draws = np.concatenate(draws)
    train = C.family(model['family'], tj['config']['train_seed'], tj['config']['train_count'])
    assert not any(np.array_equal(a, b) for a in train for b in draws), 'cohort overlaps training draws'
    edges = list(cfg['edges']); Rfull = edges[-1]; assert Rfull == T.shape[0]
    prop = C.make_propagate(d); tjx = jnp.asarray(times)
    for n in cfg['meshes']:
        begin = time.perf_counter(); keep_orig = n in cfg['parity']['meshes']
        nested, orig = K.build_bank(model, n, T, edges, keep_orig)
        modes = C.mode_list(cfg['tests'], n, d); lam_m = C.mode_eigs(n, modes)
        rtri = K.nested_tsqr_r(nested); a_rot = K.nested_weak_matrix(nested, modes, n, d)
        sv = np.linalg.svd(rtri, compute_uv=False)
        mesh = dict(intervals=n, unknowns=(n - 1) ** d, bank_bytes=int(sum(b.size for row in nested for b in row) * 8),
                    rotated_bank_condition=float(sv[0] / sv[-1]),
                    truncated_bank_condition={str(r): float(np.linalg.cond(rtri[:r, :r])) for r in cfg['ladder']},
                    cases=[], warmup_seconds={}, profile={}, parity=[], neighbour={}, compile_seconds={})
        if keep_orig:
            rtri_o = C.tsqr_r(orig); a_o = C.weak_matrix(orig, modes, n, d)
            mesh['parent_bank_condition'] = float(np.linalg.cond(rtri_o))
            # the rotated operators are the parent's times T (exact algebra; round-off only)
            mesh['weak_matrix_rotation_identity'] = float(np.linalg.norm(a_o @ T - a_rot) / np.linalg.norm(a_rot))
        if (n - 1) ** d <= 70000:
            g = np.concatenate([np.concatenate([np.asarray(b) for b in row], axis=1) for row in nested])
            explicit = C.explicit_tests(n, d, modes).T @ g
            mesh['weak_matrix_relative_error'] = float(np.linalg.norm(explicit - a_rot) / np.linalg.norm(explicit))
            assert mesh['weak_matrix_relative_error'] < 1e-10, mesh; del g, explicit
        mesh['setup_seconds'] = time.perf_counter() - begin
        methods, stages, kind = {}, {}, {}
        fam_opts = {f: {**cfg['rom_defaults'], **o} for f, o in cfg['families'].items()}
        for Rp in cfg['ladder']:
            nb = edges.index(Rp)
            proj = (lambda b, v, nb=nb: K.nproject(b, v, nb)); expd = (lambda c, b, nb=nb: K.nexpand(c, b, nb))
            setup = dict(n=n, d=d, times=times, nu=nu, modes=modes, a=a_rot[:, :Rp], rtri=rtri[:Rp, :Rp], mode_lam=lam_m,
                         directions=L[:Rp] @ np.asarray(model['directions']))
            tm = K.truncated_model(model, L, Rp)
            for fam, opt in fam_opts.items():
                for q in cfg['q_by_Rp'][str(Rp)]:
                    name = f'nmrom_R{Rp}_q{q}_{fam}'
                    st = K.make_stages(tm, setup, q, opt, proj, expd); stages[name] = st; kind[name] = 'model'
                    methods[name] = (lambda u, st=st: st['query'](u, nested))
                lin = K.make_linear(setup, opt['init'], opt['stepping'], opt.get('dt'), proj, expd)
                name = f'lin_R{Rp}_{fam}'; stages[name] = lin; kind[name] = 'model'
                methods[name] = (lambda u, lin=lin: lin['query'](u, nested))
        if keep_orig:   # the unrotated parent model, parent code (core.make_stages / plain free coefficients), R' = R
            setup_o = dict(n=n, d=d, times=times, nu=nu, modes=modes, a=a_o, rtri=rtri_o, mode_lam=lam_m, directions=model['directions'])
            for fam, opt in fam_opts.items():
                for q in cfg['parity']['q']:
                    name = f'parent_q{q}_{fam}'; st = C.make_stages(model, setup_o, q, opt); kind[name] = 'model'
                    methods[name] = (lambda u, st=st: st['query'](u, orig))
                lin = K.make_linear(setup_o, opt['init'], opt['stepping'], opt.get('dt'), C.bank_project, C.bank_expand)
                name = f'parent_lin_{fam}'; kind[name] = 'model'; methods[name] = (lambda u, lin=lin: lin['query'](u, orig))
        for name, spec in cfg['cg_arms'].items():
            methods[name] = C.make_cg(n, times, spec, nu); kind[name] = 'fom'
        lam_d, lam_c = C.eig_grid(n, d), C.eig_grid(n, d, True)
        methods['dst_exact_CONTROL'] = lambda u: (prop(u, lam_d, tjx, nu),); kind['dst_exact_CONTROL'] = 'fom'
        names = list(methods); model_arms = [m for m in names if kind[m] == 'model']; fom_arms = [m for m in names if kind[m] == 'fom']
        pairs = [(f'parent_q{q}_{f}', f'nmrom_R{Rfull}_q{q}_{f}') for f in fam_opts for q in cfg['parity']['q']] + \
                [(f'parent_lin_{f}', f'lin_R{Rfull}_{f}') for f in fam_opts] if keep_orig else []
        reps = cfg['repetitions']; stride = max(1, n // cfg['audit_intervals']); sub = (slice(None),) + (slice(stride - 1, None, stride),) * d
        rows = {m: dict(method=m, kind=kind[m], same=[], physical=[], same_sub=[], physical_sub=[], device_ms=[], stats=[],
                        failures=[], fingerprint_mismatch=0) for m in names}
        neighbour_cg = methods[cfg['neighbour']['cg']]
        nb_cases = [i for i, c in enumerate(cohorts) if c == cfg['neighbour']['cohort']][:cfg['neighbour']['cases']]
        prof_case = next(i for i, c in enumerate(cohorts) if c == cfg['neighbour']['cohort'])
        for ci, draw in enumerate(draws[:cfg['cases_by_mesh'].get(str(n), len(draws))]):
            u0 = C.block(C.initial_grid(n, d, draw)); same = C.block(prop(u0, lam_d, tjx, nu)); phys = C.block(prop(u0, lam_c, tjx, nu))
            case = dict(case=ci, cohort=cohorts[ci], draw=draw.tolist(), discretisation_error=np.asarray(C.rel_errors(same, phys)).tolist())
            saved = {}; fps = {}
            rng = np.random.default_rng(cfg.get('audit_sample_seed', 20260923) + ci); sample = np.sort(rng.choice((n - 1) ** d, min(100000, (n - 1) ** d), replace=False))
            for m in names:   # untimed pass: compile/warm, errors, statistics, audit fields, fingerprint
                res, sec = timed(methods[m], u0); mesh['warmup_seconds'].setdefault(m, []).append(sec)
                f = res[0]; assert f.dtype == jnp.float64 and bool(jnp.isfinite(f).all()), m
                rows[m]['same'].append(np.asarray(C.rel_errors(f, same)).tolist()); rows[m]['physical'].append(np.asarray(C.rel_errors(f, phys)).tolist())
                rows[m]['same_sub'].append(np.asarray(C.rel_errors(f[sub], same[sub])).tolist()); rows[m]['physical_sub'].append(np.asarray(C.rel_errors(f[sub], phys[sub])).tolist())
                if m.startswith('nmrom') or (m.startswith('parent_q')):
                    fl, i0, sm, sx = K.lm_failures(res[1], res[2]); rows[m]['failures'].append(fl)
                    rows[m]['stats'].append(dict(init_attempts=i0, step_attempts_mean=sm, step_attempts_max=sx))
                elif kind[m] == 'fom' and len(res) > 1:
                    cg = np.asarray(res[1]).reshape(-1, 3); rows[m]['failures'].append(int((cg[:, 2] != 1).sum()))
                    rows[m]['stats'].append(dict(cg_iterations=float(cg[:, 0].sum())))
                else:
                    rows[m]['failures'].append(0); rows[m]['stats'].append({})
                fps[m] = np.asarray(FP(f)); saved[m] = np.asarray(f[sub])
                if ci < 2: saved['RAND_' + m] = np.asarray(f.reshape(len(times), -1)[:, jnp.asarray(sample)])
                if (n - 1) ** d <= cfg.get('full_audit_max_unknowns', 300000) and ci < cfg.get('full_audit_cases', 2): saved['FULL_' + m] = np.asarray(f)
                del res, f
            if ci < 2: saved['sample_indices'] = sample
            for pa, pb in pairs:   # parity: rotated R' = R vs the unrotated parent, full fields
                fa = methods[pa](u0)[0]; fb = methods[pb](u0)[0]
                mesh['parity'].append(dict(case=ci, parent=pa, rotated=pb, relative_difference=float(jnp.max(C.rel_errors(fb, fa)))))
                del fa, fb
            for rep in range(reps):   # phase 1: model arms, random order, burn-in before each
                order = list(np.random.default_rng([cfg.get('order_seed', 923), n, ci, rep, 1]).permutation(model_arms))
                for m in order:
                    C.burn(cfg['burn_seconds']); res, sec = timed(methods[m], u0)
                    if not np.array_equal(np.asarray(FP(res[0])), fps[m]): rows[m]['fingerprint_mismatch'] += 1
                    rows[m]['device_ms'].append(1e3 * sec); del res
            for rep in range(reps):   # phase 2: FOM arms and the transform control (slow arms, own phase)
                order = list(np.random.default_rng([cfg.get('order_seed', 923), n, ci, rep, 2]).permutation(fom_arms))
                for m in order:
                    C.burn(cfg['burn_seconds']); res, sec = timed(methods[m], u0)
                    if not np.array_equal(np.asarray(FP(res[0])), fps[m]): rows[m]['fingerprint_mismatch'] += 1
                    rows[m]['device_ms'].append(1e3 * sec); del res
            if ci in nb_cases:   # order-effect gate: model arm timed immediately after a CN-CG solve
                for rep in range(cfg['neighbour']['reps']):
                    order = list(np.random.default_rng([cfg.get('order_seed', 923), n, ci, rep, 3]).permutation(model_arms))
                    for m in order:
                        C.burn(cfg['burn_seconds']); C.block(neighbour_cg(u0)); _, sec = timed(methods[m], u0)
                        mesh['neighbour'].setdefault(m, []).append(dict(case=ci, ms=1e3 * sec))
            if ci == prof_case and n in cfg['profile']['meshes']:   # same-job stage profile
                for name, st in stages.items():
                    if cfg['profile'].get('arms') and name not in cfg['profile']['arms'] and not name.startswith('lin_'): continue
                    lin = name.startswith('lin_'); keys = ('encode', 'evolve', 'decode', 'query') if lin else ('encode', 'init', 'evolve', 'decode', 'query')
                    rec = {k: [] for k in keys}
                    for _ in range(reps + 1):
                        C.burn(cfg['burn_seconds'])
                        if lin:
                            (c0, m0), a0 = timed(st['encode'], u0, nested); coefs, a2 = timed(st['evolve'], c0, m0); _, a3 = timed(st['decode'], coefs, nested)
                            _, a4 = timed(st['query'], u0, nested); vals = (a0, a2, a3, a4)
                        else:
                            (t0, m0), a0 = timed(st['encode'], u0, nested); (z, c0, s0), a1 = timed(st['init'], t0)
                            (coefs, s1), a2 = timed(st['evolve'], z, c0, m0); allc = C.block(jnp.concatenate((c0[None], coefs))); _, a3 = timed(st['decode'], allc, nested)
                            _, a4 = timed(st['query'], u0, nested); vals = (a0, a1, a2, a3, a4)
                        for k, v in zip(keys, vals): rec[k].append(1e3 * v)
                    mesh['profile'][name] = {k: float(np.median(v[1:])) for k, v in rec.items()}
                print('profile', n, json.dumps(mesh['profile']), flush=True)
            np.savez_compressed(out / f'fields_n{n}_case{ci}.npz', draw=draw, stride=stride, **saved)
            mesh['cases'].append(case)
            print('case', n, ci, cohorts[ci], {m: (round(float(np.median(rows[m]['device_ms'][-reps:])), 3), round(100 * max(rows[m]['same'][-1]), 4)) for m in names}, flush=True)
        mesh['rows'] = list(rows.values()); result['meshes'].append(mesh); C.dump(out / 'results.json', result)
        del nested, orig, methods, stages; jax.clear_caches()
    result['complete'] = True; C.dump(out / 'results.json', result); print('HEAT BANK KNOB PANEL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
