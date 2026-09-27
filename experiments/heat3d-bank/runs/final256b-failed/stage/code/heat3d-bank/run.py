"""One-allocation paired ROM/baseline/FOM panel with stage profiling. See DESIGN.md."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import core as C


def gpu_identity():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        return f'unavailable: {exc}'


def timed(fn, *args):
    begin = time.perf_counter(); out = C.block(fn(*args)); return out, time.perf_counter() - begin


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
    result = dict(schema='heat3d-bank-panel-v1', config=cfg, source_sha256=sources, model_sha256=model['sha256'],
                  source_commit=os.environ.get('SOURCE_COMMIT'), complete=False, meshes=[],
                  metadata=dict(job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'), gpu=gpu_identity(),
                                jax=jax.__version__, backend='gpu', x64=True, precision='highest', local_smoke=smoke),
                  query_contract='Supplied full initial field on the GPU to all requested full output fields on the GPU, '
                                 'blocked. Offline bank/operator assembly and compilation excluded and recorded. Host transfers excluded for every method.')
    if model['directions'] is None:
        model['directions'], result['directions'] = C.sep2d_directions(model, cfg['sep2d_training'])
    if 'sealed_gate' in cfg:   # pre-registered: the sealed cohort is opened only if the trained bank's VALIDATION floor passes
        floor = json.loads((here / 'inputs' / Path(cfg['model']['bank']).parent / 'training.json').read_text())['bank']['validation_projection_worst']
        result['sealed_gate'] = dict(threshold=cfg['sealed_gate'], validation_projection_worst=floor, sealed_opened=floor <= cfg['sealed_gate'])
        if floor > cfg['sealed_gate']:
            cfg['cohorts'] = cfg['cohorts'][:1]; cfg['cohort_names'] = cfg['cohort_names'][:1]
            cfg['cases_by_mesh'] = {k: min(v, cfg['cohorts'][0][1]) for k, v in cfg['cases_by_mesh'].items()}; result['config'] = cfg
    draws = np.concatenate([C.family(model['family'], s, c) for s, c in cfg['cohorts']])
    if 'sep2d_training' in cfg:
        train = np.concatenate([C.family('mr2d', s, c) for s, c in cfg['sep2d_training']['train_draws']])
        assert not any(np.array_equal(a, b) for a in train for b in draws)
    prop = C.make_propagate(d); tj = jnp.asarray(times)
    for n in cfg['meshes']:
        begin = time.perf_counter(); bank = C.bank_at(model, n); rtri = C.tsqr_r(bank); modes = C.mode_list(cfg['tests'], n, d)
        a = C.weak_matrix(bank, modes, n, d); sv = np.linalg.svd(rtri, compute_uv=False)
        setup = dict(n=n, d=d, times=times, nu=nu, modes=modes, a=a, rtri=rtri, mode_lam=C.mode_eigs(n, modes), directions=model['directions'])
        mesh = dict(intervals=n, unknowns=(n - 1) ** d, setup_seconds=time.perf_counter() - begin, bank_bytes=int(bank.size * 8),
                    bank_condition=float(sv[0] / sv[-1]), cases=[], warmup_seconds={}, profile={})
        if (n - 1) ** d <= 70000:   # gate the DST weak matrix against explicit tests where that is affordable
            explicit = C.explicit_tests(n, d, modes).T @ np.asarray(bank)
            mesh['weak_matrix_relative_error'] = float(np.linalg.norm(explicit - a) / np.linalg.norm(explicit))
            assert mesh['weak_matrix_relative_error'] < 1e-10, mesh
        methods, stages = {}, {}
        for arm in cfg['rom_arms']:
            if n in arm.get('skip_meshes', []): continue
            st = C.make_stages(model, setup, arm['q'], {**cfg['rom_defaults'], **arm.get('opt', {})}); stages[arm['name']] = st
            methods[arm['name']] = (lambda u, st=st: st['query'](u, bank))
        for init in cfg.get('linear_bank_inits', []):
            lb = C.linear_bank(setup, init); methods[f'linear_bank_{init}_BASELINE'] = (lambda u, lb=lb: (lb(u, bank),))
            ld = C.linear_bank(setup, init, direct=True); methods[f'linear_bank_{init}_direct_BASELINE'] = (lambda u, ld=ld: (ld(u, bank),))
        for name, spec in cfg['cg_arms'].items():
            if n in spec.get('skip_meshes', []): continue
            methods[name] = C.make_cg(n, times, spec, nu)
        for nc in cfg.get('coarse_intervals', []):
            if nc < n: methods[f'coarse{nc}_{cfg["coarse_cg"]}'] = C.make_coarse(n, nc, d, times, cfg['cg_arms'][cfg['coarse_cg']], nu)
        lam_d, lam_c = C.eig_grid(n, d), C.eig_grid(n, d, True)
        methods['dst_exact_CONTROL'] = lambda u: (prop(u, lam_d, tj, nu),)
        count = cfg['cases_by_mesh'][str(n)]; reps = cfg['repetitions']; names = list(methods)
        stride = max(1, n // cfg['audit_intervals']); sub = (slice(None),) + (slice(stride - 1, None, stride),) * d
        rows = {m: dict(method=m, same=[], physical=[], same_sub=[], physical_sub=[], device_ms=[], stats=[]) for m in names}
        for ci, draw in enumerate(draws[:count]):
            u0 = C.block(C.initial_grid(n, d, draw)); same = C.block(prop(u0, lam_d, tj, nu)); phys = C.block(prop(u0, lam_c, tj, nu))
            case = dict(case=ci, draw=draw.tolist(), discretisation_error=np.asarray(C.rel_errors(same, phys)).tolist())
            if n % 2 == 0:   # empirical refinement evidence for the continuum-spectral reference
                h = n // 2; ph = prop(C.initial_grid(h, d, draw), C.eig_grid(h, d, True), tj, nu)
                case['reference_refinement'] = float(jnp.max(C.rel_errors(phys[(slice(None),) + (slice(1, None, 2),) * d], ph)))
                case['reference_refinement_ok'] = case['reference_refinement'] < cfg['reference_budget']   # heat3d-bank: recorded + flagged, not fatal (last job)
                if not case['reference_refinement_ok']: print('WARNING reference refinement above budget', n, ci, case['reference_refinement'], flush=True)
            saved = {}; rng = np.random.default_rng(cfg.get('audit_sample_seed', 20260920) + ci); sample = np.sort(rng.choice((n - 1) ** d, min(100000, (n - 1) ** d), replace=False))
            for m in names:   # untimed pass: compile/warm, errors, solver statistics, saved audit fields
                res, sec = timed(methods[m], u0); mesh['warmup_seconds'].setdefault(m, []).append(sec)
                f = res[0]; assert f.dtype == jnp.float64 and bool(jnp.isfinite(f).all()), m
                rows[m]['same'].append(np.asarray(C.rel_errors(f, same)).tolist()); rows[m]['physical'].append(np.asarray(C.rel_errors(f, phys)).tolist())
                rows[m]['same_sub'].append(np.asarray(C.rel_errors(f[sub], same[sub])).tolist()); rows[m]['physical_sub'].append(np.asarray(C.rel_errors(f[sub], phys[sub])).tolist())
                rows[m]['stats'].append([np.asarray(s).tolist() for s in res[1:]]); saved[m] = np.asarray(f[sub])
                if ci < 2: saved['RAND_' + m] = np.asarray(f.reshape(len(times), -1)[:, jnp.asarray(sample)])
                del res, f
            if ci < 2: saved['sample_indices'] = sample
            timed_count = cfg.get('timed_cases_by_mesh', {}).get(str(n), count)   # heat3d-bank: later validation cases are errors-only
            for rep in range(reps if ci < timed_count else 0):   # timed passes contain nothing but burn-in and synchronised queries
                for m in (names if rep % 2 == 0 else names[::-1]):
                    C.burn(cfg['burn_seconds']); res, sec = timed(methods[m], u0); rows[m]['device_ms'].append(1e3 * sec); del res
            full = (n - 1) ** d <= 300000 and ci < cfg.get('full_audit_cases', 2)
            if full:
                for m in names: saved['FULL_' + m] = np.asarray(methods[m](u0)[0])
            # heat3d-bank: every arm saved for a prefix of cases; beyond it only the selection-driving arms (Codex finding 5)
            if ci < cfg.get('save_fields_cases', count): case['fields_saved'] = 'all'
            elif cfg.get('save_selection_arms'):
                case['fields_saved'] = 'selection'; saved = {k: v for k, v in saved.items() if cfg['save_selection_arms'] in k}
            else: case['fields_saved'] = False
            if case['fields_saved']: np.savez_compressed(out / f'fields_n{n}_case{ci}.npz', draw=draw, stride=stride, **saved)
            case['timed'] = ci < timed_count; mesh['cases'].append(case)
            print('case', n, ci, 'timed' if case['timed'] else 'errors-only', {m: (round(float(np.median(rows[m]['device_ms'][-reps:])), 3) if case['timed'] else None, round(100 * max(rows[m]['same'][-1]), 4)) for m in names}, flush=True)
            if ci == 0:   # stage profile on the first case: same-job breakdown of the ROM query
                for name, st in stages.items():
                    rec = {k: [] for k in ('encode', 'init', 'evolve', 'decode', 'query')}
                    for _ in range(reps + 1):
                        C.burn(cfg['burn_seconds']); (t0, m0), a0 = timed(st['encode'], u0, bank); (z, c0, s0), a1 = timed(st['init'], t0)
                        (coefs, s1), a2 = timed(st['evolve'], z, c0, m0); allc = C.block(jnp.concatenate((c0[None], coefs))); _, a3 = timed(st['decode'], allc, bank)
                        _, a4 = timed(st['query'], u0, bank)
                        for k, v in zip(rec, (a0, a1, a2, a3, a4)): rec[k].append(1e3 * v)
                    mesh['profile'][name] = {k: float(np.median(v[1:])) for k, v in rec.items()}
                print('profile', n, json.dumps(mesh['profile']), flush=True)
        mesh['rows'] = list(rows.values()); result['meshes'].append(mesh); C.dump(out / 'results.json', result)
        del bank, methods, stages; jax.clear_caches()
    result['complete'] = True; C.dump(out / 'results.json', result); print('HEAT3D BANK PANEL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
