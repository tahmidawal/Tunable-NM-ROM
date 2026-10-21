"""One Table-1 cell (Poisson 3D or Heat 3D at one mesh), one allocation, one process:
the Table-1 NM-ROM accurate and fast settings (frozen models of the source lanes, their own code, unchanged),
the Table-1 full-order grid (the source lane's CG / CN-CG ladder), and the trained operators, on the Table-1
cohort of that row. Accuracy pass, reproduction gate against the source lane's per-case errors, then
interleaved timing (GPU query: device-resident input -> device-resident output fields, synchronised).

Usage: panel_l3d.py --config configs/panel_<problem><n>.json --out output [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault('JAX_ENABLE_X64', '1')
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
jax.config.update('jax_enable_x64', True)

GATE = 1.10


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=float) + '\n')
    tmp.replace(path)


# ------------------------------------------------------------------------------------------ Poisson
def build_poisson(cfg, n, smoke):
    sys.path.insert(0, str(HERE / 'deps' / 'poisson'))
    import common as C
    import poisson as PP
    import iterative_cg as CG
    import pbk3_core as P3
    from pbk3_cube import reference, make_linear
    d = HERE / 'deps' / 'poisson'
    bankck = pickle.loads((d / 'bank.pkl').read_bytes())
    shas = dict(bank=sha_file(d / 'bank.pkl'), head=sha_file(d / 'head_K16.pkl'))
    assert shas == cfg['checkpoint_sha256'], shas
    prep = np.load(d / 'prep_cube.npz', allow_pickle=False)
    Tm = prep['T']
    assert json.loads(str(prep['rotation_info']))['checkpoint_sha256'] == shas
    mcfg = bankck['cfg']
    if cfg['cohort'] == 'development':
        cases = C.family(mcfg['validation_seed'], mcfg['validation_count'])
        coh = json.loads((d / 'cohorts.json').read_text())
        np.testing.assert_array_equal(cases, np.asarray(coh['validation_parameters']))
    else:
        cases = C.family(mcfg['reserved_final_seed'], cfg['final_count'])
        assert sha_array(cases) == cfg['final_parameters_sha256']
    if smoke:
        cases = cases[:3]
    train = C.family(mcfg['train_seed'], mcfg['train_count'])
    assert not any(np.allclose(t, s) for t in train for s in cases)
    t0 = time.perf_counter()
    sources = [np.asarray(PP.source(n, p)) for p in cases]
    same = [reference(n, f) for f in sources]
    bank = C.bank_at(bankck['params'], n, cfg['field_chunk']) @ np.asarray(bankck['rotation'])
    bankj = jnp.asarray(bank)
    del bank
    Rw = int(bankj.shape[1])
    triples = C.modes(cfg['weak_tests'], n)
    lam = np.asarray(C.mode_eigenvalues(n, triples))
    ti, tj, tk = (jnp.asarray(triples[:, ax] - 1) for ax in range(3))
    col = jax.jit(lambda g: C.dst3(g.reshape((n - 1,) * 3))[ti, tj, tk] / n ** 1.5)
    operator = np.stack([np.asarray(col(bankj[:, r])) for r in range(Rw)], axis=1)
    edges = [0] + sorted(cfg['R_ladder'])
    rot = P3.split_blocks(bankj @ jnp.asarray(Tm), edges)
    jax.block_until_ready(rot)
    bank_sha = sha_array(np.asarray(bankj[::997]))
    del bankj
    fj = [jax.device_put(f) for f in sources]
    jax.block_until_ready(fj)
    arms = {}
    for label, Rp in (('nmrom_accurate', cfg['accurate_Rp']), ('nmrom_fast', cfg['fast_Rp'])):
        Qm, Rr = np.linalg.qr(operator @ Tm[:, :Rp], mode='reduced')
        kern, _ = make_linear(n, triples, lam)
        blocks = rot[:P3.nblocks(edges, Rp)]
        Qt, Rrj = jnp.asarray(Qm.T), jnp.asarray(Rr)
        arms[f'{label}_R{Rp}_linear'] = dict(kind='nmrom', source_name=f'R{Rp}_linear', framework='jax',
                                             call=(lambda c, kern=kern, Qt=Qt, Rrj=Rrj, blocks=blocks: kern(fj[c], Qt, Rrj, blocks)))
    cap = int(cfg['cg_iterations_per_interval']) * n
    for tol in cfg['cg_tolerances']:
        eng = CG.engine(n, float(tol), cap, retain_history=False)
        name = f'fom_cg_rtol{tol:g}'
        arms[name] = dict(kind='fom', source_name=f'cg_{tol:g}', framework='jax', tolerance=float(tol),
                          call=(lambda c, eng=eng: eng(fj[c])),
                          check=(lambda out, tol=tol: CG.counters(out[1], float(tol), cap)['cg_converged']))
    setup = dict(seconds=time.perf_counter() - t0, bank_sample_sha256=bank_sha, rotation_edges=edges, weak_tests=int(len(triples)),
                 cg_iteration_cap=cap, checkpoint_sha256=shas)
    info = dict(cohort=cfg['cohort'], count=int(len(cases)), parameters=cases.tolist(), parameters_sha256=sha_array(cases))
    return dict(arms=arms, inputs=fj, truth=lambda c: same[c], setup=setup, cohort=info, ncase=len(cases),
                field=lambda out: out[0] if isinstance(out, tuple) else out)


# ------------------------------------------------------------------------------------------ Heat
def build_heat(cfg, n, smoke):
    sys.path.insert(0, str(HERE / 'deps' / 'heat'))
    import core as C
    import hbk_core as K
    d = HERE / 'deps' / 'heat'
    model = C.load_model(cfg['model'], d / 'inputs')
    assert model['sha256'] == cfg['model_sha256'], model['sha256']
    times, nu = cfg['times'], cfg['diffusivity']
    prep = np.load(d / 'prep_3d.npz')
    T, L = prep['T'], prep['L']
    rinfo = json.loads(str(prep['rotation_info']))
    assert K.sha_array(T) == rinfo['T_sha256'] and K.sha_array(L) == rinfo['L_sha256']
    tj = json.loads((d / 'inputs' / 'vp_R320' / 'training.json').read_text())
    assert rinfo['training_draws_sha'] == tj['train_draws_sha']
    draws = C.family(model['family'], cfg['heldout_seed'], cfg['heldout_count'])
    if smoke:
        draws = draws[:3]
    train = C.family(model['family'], tj['config']['train_seed'], tj['config']['train_count'])
    assert not any(np.array_equal(a, b) for a in train for b in draws)
    t0 = time.perf_counter()
    edges = list(cfg['edges'])
    nested, _ = K.build_bank(model, n, T, edges, False)
    modes = C.mode_list(cfg['tests'], n, 3)
    lam_m = C.mode_eigs(n, modes)
    rtri = K.nested_tsqr_r(nested)
    a_rot = K.nested_weak_matrix(nested, modes, n, 3)
    u0s = [C.block(C.initial_grid(n, 3, dr)) for dr in draws]
    prop = C.make_propagate(3)
    lam_d = C.eig_grid(n, 3)
    tjx = jnp.asarray(times)
    arms = {}
    for label, Rp in (('nmrom_accurate', cfg['accurate_Rp']), ('nmrom_fast', cfg['fast_Rp'])):
        nb = edges.index(Rp)
        proj = (lambda b, v, nb=nb: K.nproject(b, v, nb))
        expd = (lambda c_, b, nb=nb: K.nexpand(c_, b, nb))
        setup = dict(n=n, d=3, times=times, nu=nu, modes=modes, a=a_rot[:, :Rp], rtri=rtri[:Rp, :Rp], mode_lam=lam_m,
                     directions=L[:Rp] @ np.asarray(model['directions']))
        lin = K.make_linear(setup, 'field', 'cn', cfg['rom_dt'], proj, expd)
        arms[f'{label}_R{Rp}_linear_cn'] = dict(kind='nmrom', source_name=f'lin_R{Rp}_cn', framework='jax',
                                                call=(lambda c, lin=lin: lin['query'](u0s[c], nested)))
    for name, spec in cfg['cg_arms'].items():
        q = C.make_cg(n, times, spec, nu)
        arms[name] = dict(kind='fom', source_name=name, framework='jax', dt=spec['dt'], tolerance=spec['tolerance'],
                          call=(lambda c, q=q: q(u0s[c])),
                          check=(lambda out: bool(np.all(np.asarray(out[1])[:, 2] == 1))))
    setup = dict(seconds=time.perf_counter() - t0, rotation_edges=edges, weak_tests=int(len(modes)),
                 model_sha256=model['sha256'], bank_bytes=int(sum(b.size for row in nested for b in row) * 8))
    info = dict(cohort='heldout_sealed_921099', count=int(len(draws)), parameters=draws.tolist(), parameters_sha256=sha_array(draws))
    return dict(arms=arms, inputs=u0s, truth=lambda c: prop(u0s[c], lam_d, tjx, nu), setup=setup, cohort=info,
                ncase=len(draws), field=lambda out: out[0])


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(cfg.get('smoke', {}))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError(f'need a GPU backend, got {jax.default_backend()}')
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    problem, n = cfg['problem'], int(cfg['n'])
    t_job = time.time()
    report = dict(schema='ops-all-l3d-panel-v1', problem=problem, n=n, config=cfg, config_sha256=sha_file(args.config),
                  smoke=bool(args.smoke), job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                  source_commit=os.environ.get('SOURCE_COMMIT'), jax=jax.__version__, device=str(jax.devices()),
                  sources={p.name: sha_file(p) for p in sorted(HERE.glob('*.py'))},
                  deps={str(p.relative_to(HERE)): sha_file(p) for p in sorted((HERE / 'deps').rglob('*')) if p.is_file() and '__pycache__' not in str(p)})
    try:
        report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                                 '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report['gpu'] = 'unavailable'

    def save():
        report['elapsed_seconds'] = time.time() - t_job
        dump(out / 'summary.json', report)

    log(f'building {problem} {n}')
    B = (build_poisson if problem == 'poisson' else build_heat)(cfg, n, args.smoke)
    report['setup'], report['cohort'] = B['setup'], B['cohort']
    save()
    arms, ncase = B['arms'], B['ncase']

    # ---------------------------------------------------------------- operators (PyTorch, same process/GPU)
    import torch
    import ops_l3d as O
    import problems as PB
    report['torch_env'] = O.configure()
    prob = PB.Problem(problem, n)
    xin = [torch.from_dlpack(x) for x in B['inputs']]     # zero-copy views of the JAX inputs
    for c in range(min(ncase, 2)):
        assert float((xin[c] - torch.as_tensor(np.asarray(B['inputs'][c]), device='cuda')).abs().max()) == 0.0
    op_meta = {}
    for op in cfg['operators']:
        name = f"op_{op['family']}_{op['arm']}"
        if op.get('failure'):
            op_meta[name] = dict(op, status='not trained')
            continue
        path = HERE / op['checkpoint']
        digest = sha_file(path)
        if digest != op['sha256']:
            raise RuntimeError(f"checkpoint {name} hash {digest} != training record {op['sha256']}")
        try:
            model, norm, ckd = O.load_checkpoint(path)
            assert int(ckd['mesh']) == n and ckd['problem'] == problem and ckd['arm'] == op['arm'], ckd.get('arm')
            O.check_dtypes(model)
        except Exception as exc:   # recorded, never worked around
            op_meta[name] = dict(op, status=f'load failed: {type(exc).__name__}: {exc}'[:500])
            continue

        def op_call(c, m_=model, nm_=norm):
            with torch.no_grad():
                return prob.predict(m_, xin[c][None], nm_)[0]
        op_meta[name] = dict(op, status='loaded', parameter_dtype=str(getattr(model, 'parameter_dtype', torch.float64)),
                             real_parameter_count=O.parameter_count(model))
        arms[name] = dict(kind='operator', framework='torch', call=op_call)
    report['operators'] = op_meta

    def sync(arm, outv):
        if arm['framework'] == 'torch':
            torch.cuda.synchronize()
        else:
            jax.block_until_ready(outv)

    def as_torch(x):
        return x if isinstance(x, torch.Tensor) else torch.from_dlpack(x)

    def errors(field, truth):
        """float64 relative L2 per output time (heat, 6) or scalar (Poisson), on the device."""
        a, b = as_torch(field).reshape(truth.shape), as_torch(truth)
        if problem == 'poisson':
            return [float(((a - b).square().sum() / b.square().sum()).sqrt())]
        return [float(v) for v in ((a - b).square().sum((1, 2, 3)) / b.square().sum((1, 2, 3))).sqrt()]

    # ---------------------------------------------------------------- accuracy pass (also compile / warm-up)
    names = list(arms)
    acc = {m: dict(errors=[], checks=[], finite=True, failure=None) for m in names}
    fingerprints = {m: [] for m in names}
    log(f'accuracy pass: {len(names)} arms x {ncase} cases')
    for c in range(ncase):
        truth = B['truth'](c)
        if problem == 'poisson':
            truth = jnp.asarray(truth)
        for m in names:
            if acc[m]['failure']:
                continue
            arm = arms[m]
            try:
                outv = arm['call'](c)
                sync(arm, outv)
                field = B['field'](outv) if arm['framework'] == 'jax' else outv
                ft = as_torch(field)
                if not bool(torch.isfinite(ft).all()):
                    acc[m]['finite'] = False
                acc[m]['errors'].append(errors(field, truth))
                if 'check' in arm:
                    acc[m]['checks'].append(bool(arm['check'](outv)))
                fingerprints[m].append(float(ft.double().abs().sum()))
                del outv, field, ft
            except Exception as exc:
                acc[m]['failure'] = f'{type(exc).__name__}: {exc}'[:500]
                log(f'arm {m} failed: {acc[m]["failure"][:200]}')
                if 'op_' in m:
                    torch.cuda.empty_cache()
        del truth
        if c % 8 == 0:
            log(f'case {c}: ' + ', '.join(f'{m}={100 * max(acc[m]["errors"][-1]):.4f}%' for m in names if acc[m]['errors']))
    torch.cuda.empty_cache()
    report['accuracy'] = acc
    save()

    # ---------------------------------------------------------------- reproduction gate vs the source lane
    exp = json.loads((HERE / cfg['expected']).read_text())
    rep = {}
    for m, arm in arms.items():
        src = arm.get('source_name')
        if src is None or src not in exp['per_case_error'] or acc[m]['failure']:
            continue
        mine = np.asarray(acc[m]['errors'], dtype=float)
        theirs = np.asarray(exp['per_case_error'][src], dtype=float)[:ncase].reshape(mine.shape)
        gap = float(np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-14)))
        gated = arm['kind'] == 'nmrom' or m == cfg['table1_fom']
        tol = cfg['reproduction_tolerance'] if arm['kind'] == 'nmrom' else cfg['reproduction_tolerance_fom']
        rep[m] = dict(source=src, max_relative_gap=gap, tolerance=tol, gated=gated, passed=bool(gap <= tol))
    report['reproduction'] = dict(source=exp['source'], source_sha256=exp['source_sha256'], source_job=exp['job_id'], arms=rep,
                                  passed=bool(rep) and all(v['passed'] for v in rep.values() if v['gated']),
                                  note='gated: NM-ROM arms (<=1e-6 rel) and the Table-1 FOM setting (<=1e-3 rel); other FOM settings informational (loose-tolerance CG stopping is rounding-sensitive across GPUs)')
    log(f"reproduction gate passed={report['reproduction']['passed']} max gap "
        f"{max((v['max_relative_gap'] for v in rep.values()), default=None)}")
    save()

    # ---------------------------------------------------------------- timing
    live = [m for m in names if not acc[m]['failure'] and acc[m]['finite']]
    burn_fn = jax.jit(lambda x: jnp.tanh(x @ x / 512))
    burn_a = jnp.eye(512, dtype=jnp.float64) + 1e-4

    def burn(seconds):
        a = burn_fn(burn_a)
        t = time.perf_counter()
        while time.perf_counter() - t < seconds:
            a = burn_fn(a)
        jax.block_until_ready(a)

    for m in live:           # 2 untimed burn-in calls per arm
        for _ in range(2):
            o = arms[m]['call'](0)
            sync(arms[m], o)
            del o
    samples = {m: [] for m in live}
    rng = np.random.default_rng(int(cfg['order_seed']))
    rounds = int(cfg['timing_rounds'])
    log(f'timing: {len(live)} arms x {ncase} cases x {rounds} rounds')
    for r in range(rounds):
        for c in range(ncase):
            for i in rng.permutation(len(live)):
                m = live[int(i)]
                arm = arms[m]
                burn(float(cfg['burn_seconds']))
                if arm['framework'] == 'torch':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                o = arm['call'](c)
                sync(arm, o)
                dt = time.perf_counter() - t0
                samples[m].append(dict(round=r, case=c, ms=1e3 * dt))
                del o
        log(f'timing round {r} done')
        report['timing_samples'] = samples
        save()

    # ---------------------------------------------------------------- summary + FOM rule
    rows = {}
    for m in names:
        a = acc[m]
        row = dict(kind=arms[m]['kind'], failure=a['failure'], finite=a['finite'])
        if a['errors'] and not a['failure']:
            e = np.asarray(a['errors'])
            per = e.max(axis=1)
            row.update(worst_pct=100 * float(per.max()), median_pct=100 * float(np.median(per)),
                       evolved_worst_pct=(100 * float(e[:, 1:].max()) if e.shape[1] > 1 else None),
                       checks_all_passed=(all(a['checks']) if a['checks'] else None))
        if m in samples and samples[m]:
            ms = np.asarray([s['ms'] for s in samples[m]])
            r0 = np.median([s['ms'] for s in samples[m] if s['round'] == 0])
            rl = np.median([s['ms'] for s in samples[m] if s['round'] == rounds - 1])
            row.update(median_ms=float(np.median(ms)), p10_ms=float(np.percentile(ms, 10)), p90_ms=float(np.percentile(ms, 90)),
                       samples=int(len(ms)), drift_ratio=float(rl / r0), drift_ok=bool(1 / GATE <= rl / r0 <= GATE))
        rows[m] = row
    acc_name = next(m for m in names if m.startswith('nmrom_accurate'))
    fast_name = next(m for m in names if m.startswith('nmrom_fast'))
    fom_pool = [m for m in names if arms[m]['kind'] == 'fom' and 'median_ms' in rows[m] and rows[m].get('checks_all_passed')
                and rows[m]['finite'] and rows[m]['worst_pct'] <= rows[acc_name]['worst_pct']]
    fom = min(fom_pool, key=lambda m: rows[m]['median_ms']) if fom_pool else None
    report['rows'] = rows
    report['fom_rule'] = dict(rule='fastest full-order setting (all cases converged, finite) whose worst error <= the NM-ROM accurate worst error',
                              chosen=fom, eligible=fom_pool)
    if fom:
        for m in names:
            if 'median_ms' in rows[m]:
                rows[m]['speedup_vs_fom'] = rows[fom]['median_ms'] / rows[m]['median_ms']
    report['gates'] = dict(reproduction=report['reproduction']['passed'],
                           drift_selected=all(rows[m].get('drift_ok', True) for m in [acc_name, fast_name, fom] + [x for x in names if x.startswith('op_')] if m),
                           fom_found=fom is not None)
    report['status'] = 'final' if all(report['gates'].values()) else 'PROVISIONAL (a gate failed; see gates)'
    report['complete'] = True
    save()
    log('PANEL DONE ' + json.dumps({m: (round(rows[m].get('worst_pct', float('nan')), 4), round(rows[m].get('median_ms', float('nan')), 3))
                                     for m in names}))
    log(f"FOM {fom} gates {report['gates']}")


if __name__ == '__main__':
    main()
