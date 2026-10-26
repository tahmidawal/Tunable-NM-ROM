"""Frozen POD/ECSW replay. New admission evidence; inherited numerical kernels unchanged."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'jcp-hr-baselines'
sys.path.insert(0, str(OLD))
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import hrcore as H
import rom as R
import build as B
import pipeline as P
import score as S

ARMS = ('k8', 'k16', 'none')


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def quotient(num, den):
    """Exact zero handling, with nonfinite inputs rejected by an infinity result."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(den == 0, np.where(num == 0, 0., np.inf), num / den)
    return np.where(np.isfinite(num) & np.isfinite(den) & (den >= 0), v, np.inf)


def make_metrics(L, r, steps):
    """Whole residual gradient in common unnormalised objective units, every live step."""
    sf, fj = R.make_arm('none', 3, L)
    se, ej = R.make_arm('ecsw', 3, L)

    def metrics(C, nu, full, sampled, sch):
        def one(k):
            co = jnp.where(k < sch['nstart'], jnp.asarray(H.BE_CO), sch['co'])
            cn, cm, c1 = C[k], C[jnp.maximum(k - 1, 0)], C[k + 1]
            rf, jf = fj(c1, sf(cn, cm, nu, co, sch['dt'], full), nu, co, sch['dt'], full)
            re, je = ej(c1, se(cn, cm, nu, co, sch['dt'], sampled), nu, co, sch['dt'], sampled)
            gf, ge = jf.T @ rf, je.T @ re
            den = jnp.linalg.norm(jf) * jnp.linalg.norm(rf)
            u = full['Phi'] @ c1
            nf = H.adv_central(u.reshape(H.shape_of(3, L)), L).reshape(-1)
            us = jnp.einsum('esr,r->es', sampled['Pst'], c1)
            _, ne = H.stencil_F(us, 0., L, 3)
            af = full['Phi'].T @ nf
            ae = sampled['Pst'][:, 0, :].T @ (sampled['sw'] ** 2 * ne)
            return dict(gradient_difference=jnp.linalg.norm(ge - gf), denominator=den,
                        full_gradient=jnp.linalg.norm(gf), sampled_gradient=jnp.linalg.norm(ge),
                        nonlinear_difference=jnp.linalg.norm(ae - af), nonlinear_denominator=jnp.linalg.norm(af),
                        finite=jnp.all(jnp.isfinite(rf)) & jnp.all(jnp.isfinite(jf)) &
                               jnp.all(jnp.isfinite(re)) & jnp.all(jnp.isfinite(je)))
        return jax.lax.map(one, jnp.arange(steps))
    return jax.jit(metrics)


def metric_summary(a):
    a = {k: np.asarray(v) for k, v in a.items()}
    dg = quotient(a['gradient_difference'], a['denominator'])
    gf = quotient(a['full_gradient'], a['denominator'])
    rho = quotient(a['nonlinear_difference'], a['nonlinear_denominator'])
    return dict(**{k: v.tolist() for k, v in a.items()}, delta_g=dg.tolist(), full_gamma=gf.tolist(),
                nonlinear_rho_diagnostic=rho.tolist())


def scalar_check(a, b, tol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                and np.max(np.abs(a - b)) <= tol * max(1., float(np.max(np.abs(b)))))


def validate_config(cfg):
    assert cfg['cohort_seed'] == 923801 and cfg['cohort_count'] == 64
    ids = cfg['cases']
    assert ids and len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < 64 for i in ids)
    assert ids == sorted(ids) and cfg['tighter_cases'] == ids[:2]
    assert all(i in ids for i in cfg['timing_cases']) and cfg['timing_reps'] >= 2
    assert cfg['arms'] == list(ARMS) and cfg['rank'] == 512 and cfg['L'] == 64
    assert cfg['dt_numerator'] == 1 and cfg['dt_denominator'] == 60 and cfg['scheme'] == 'BDF2'
    assert cfg['admission'] == dict(gradient=.001, full_stationarity=.001, nonstationary_fraction=.01,
                                    field_distance=.00025, tight_tighter=.000025)
    return ids


def q1(records, tight, tighter, n0, subset, expected):
    """Historical numerical Q1, preserving explicit numeric case order for subset pilots."""
    complete = (list(records) == expected and list(tight) == expected and list(tighter) == subset
                and subset == expected[:2] and all(np.isfinite(n0[c]) and n0[c] > 0 for c in expected))
    if not complete:
        return dict(q1=False, reason='incomplete or misordered case inventory')
    dist = {c: S.lattice_distance(records[c]['fields_lat'], tight[c]['fields_lat'], n0[c]) for c in expected}
    dtt = {c: S.lattice_distance(tight[c]['fields_lat'], tighter[c]['fields_lat'], n0[c]) for c in subset}
    steps = sum(records[c]['cert']['steps'] for c in expected)
    nonstat = sum(records[c]['cert']['nonstat'] for c in expected)
    valid = all(records[c]['cert']['valid'] for c in expected)
    anchors = all(tight[c]['cert']['anchor_ok'] for c in expected) and all(tighter[c]['cert']['anchor_ok'] for c in subset)
    finite = all(np.isfinite(x) for x in [*dist.values(), *dtt.values()])
    return dict(q1=bool(finite and valid and anchors and steps > 0 and nonstat <= .01*steps
                        and max(dist.values()) <= .00025 and max(dtt.values()) <= .000025),
                all_valid=valid, anchors_ok=anchors, nonstat=nonstat, steps=steps,
                max_dist=max(dist.values()), max_anchor=max(dtt.values()), dist=dist, anchor=dtt,
                expected_case_ids=expected, tighter_case_ids=subset)


def run(cfg, artifacts, out):
    start = time.perf_counter()
    ids = validate_config(cfg)
    assert jax.default_backend() == 'gpu'
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    assert not out.exists(), 'a job may not overwrite an existing output directory'
    out.mkdir(parents=True)
    (out / 'arrays').mkdir()
    manifest = json.loads((HERE / 'inputs/MANIFEST.json').read_text())
    for name, want in manifest['files'].items():
        assert digest(artifacts / name) == want, name
    historic_meta = json.loads((HERE / 'inputs/historical_k8.json').read_text())
    historic_path = HERE / 'inputs/historical_k8.npz'
    assert digest(historic_path) == historic_meta['fixture_sha256']
    historic = np.load(historic_path)
    historic_ids = [str(x) for x in historic['case_ids']]
    assert historic_ids == [f'val64|{i}' for i in range(64)] == historic_meta['cases']
    print('jax_backend=gpu x64=True precision=highest', flush=True)
    meta = dict(config=cfg, config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                job=os.environ.get('SLURM_JOB_ID'), gpu=subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], text=True).strip(),
                jax=jax.__version__, artifacts=manifest, complete=False, cases={}, timing=[], phases={},
                qualification=None)

    def save():
        P.dump(out / 'result.json.tmp', meta)
        os.replace(out / 'result.json.tmp', out / 'result.json')

    def check_budget():
        if time.perf_counter() - start >= cfg['budget_hours'] * 3600:
            save()
            raise TimeoutError('registered measured wall budget exhausted; saved partial evidence')

    save()
    setup = time.perf_counter()
    z = np.load(artifacts / 'Phi.npz')
    assert 512 in set(int(x) for x in z['constructible'])
    trust = float(z['trust'][list(z['ranks']).index(512)])
    ph = np.asarray(z['Phi'][:, :512], np.float64)
    orth = float(np.max(np.abs(ph.T @ ph - np.eye(512))))
    assert orth <= 1e-10 and np.isfinite(ph).all()
    Phi = jax.block_until_ready(jnp.asarray(ph))
    del z, ph
    cell = P.Cell(3, 64)
    cell.set_basis(Phi, {512: trust})
    cell.constructible_r = {512}
    data = {'none': cell.arm_data('none', 512, '-')[0]}
    for arm in ('k8', 'k16'):
        w = np.load(artifacts / f'HR~ecsw~512~{arm}.npz')
        assert len(w['nodes']) == 512 * int(arm[1:])
        assert np.isfinite(w['weights']).all() and np.all(w['weights'] > 0)
        dat, _ = B.ecsw_data(Phi, w['nodes'], w['weights'], 3, 64)
        data[arm] = jax.block_until_ready(dict(dat, Phi=Phi))
    fam = H.Family(3, 'val64', count=64)  # regenerate unsealed cohort from seed on this machine
    refs = S.Refs(3, 'val64', fam, ids)
    meta.update(cohort_sha256=fam.sha256, orthonormality=orth, trust_radius=trust,
                reference=dict(store_version=S.STORE_VERSION, status_sha256=S.STATUS_SHA, kind='ST',
                               uncertainties={i: refs.st[i] for i in ids}),
                phases={'setup_seconds': time.perf_counter() - setup})
    opt = ('BDF2', 1 / 60)
    sch = R.schedule(3, *opt, 'prod')
    assert int(sch['steps']) == 15 and int(sch['nstart']) == 1
    assert np.array_equal(np.flatnonzero(np.asarray(sch['slot']) >= 0), np.arange(0, 16, 3))
    metrics = make_metrics(64, 512, 15)
    records, tight, tighter = {a: {} for a in ARMS}, {a: {} for a in ARMS}, {a: {} for a in ARMS}
    n0 = {f'val64|{i}': refs.n0(i) for i in ids}
    # One case is checkpointed at a time. The full field is released after scoring.
    for i in ids:
        check_budget()
        case_start = time.perf_counter()
        ck = f'val64|{i}'
        u0, nu = fam.ic(64, i), fam.nu(i)
        meta['cases'][str(i)] = {}
        for arm in ARMS:
            check_budget()
            kind = 'none' if arm == 'none' else 'ecsw'
            dat = data[arm]
            runs = {}
            for level in ('prod', 'tight') + (('tighter',) if i in cfg['tighter_cases'] else ()):
                solver = 'common' if level == 'prod' else 'native'
                rec = cell.run(kind, 512, dat, solver, opt, level, u0, nu)
                runs[level] = rec
                np.savez(out / 'arrays' / f'{arm}_c{i:03}_{level}.npz', case=i, W=rec['W'], C=rec['C'],
                         gamma=rec['gamma'], reason=rec['reason'], it=rec['it'], lam=rec['lam'],
                         scale=rec['scale'], dt=1/60, times=np.arange(6)*.05)
            rec = runs['prod']
            score = refs.score(rec['fields_lat'], i)
            assert score['eligible']
            sk = dict(e0=score, cert=rec['cert'], tight=runs['tight']['cert'],
                      tight_distance=S.lattice_distance(rec['fields_lat'], runs['tight']['fields_lat'], refs.n0(i)),
                      W_sha256=array_hash(rec['W']), seconds_first_includes_compile=rec['seconds'])
            if arm == 'k8':
                old = historic['W'][historic_ids.index(ck)]
                discrepancy = float(np.max(np.abs(rec['W'] - old))) / max(1., float(np.max(np.abs(old))))
                sk['historical_replay_relative_max'] = discrepancy
                assert np.isfinite(discrepancy) and discrepancy <= historic_meta['replay_tolerance'], 'historical k8 replay failed'
            if 'tighter' in runs:
                sk['tighter'] = runs['tighter']['cert']
                sk['tight_tighter_distance'] = S.lattice_distance(runs['tight']['fields_lat'], runs['tighter']['fields_lat'], refs.n0(i))
            if arm != 'none':
                check_budget()
                a = jax.block_until_ready(metrics(jnp.asarray(rec['C']), jnp.float64(nu), data['none'], dat, sch))
                sk['whole_residual'] = metric_summary(a)
            meta['cases'][str(i)][arm] = sk
            records[arm][ck] = dict(cert=rec['cert'], fields_lat=rec['fields_lat'])
            tight[arm][ck] = dict(cert=runs['tight']['cert'], fields_lat=runs['tight']['fields_lat'])
            if 'tighter' in runs:
                tighter[arm][ck] = dict(cert=runs['tighter']['cert'], fields_lat=runs['tighter']['fields_lat'])
            del runs, rec
            save()
        meta['cases'][str(i)]['field_distance'] = {
            'k8_to_k16': S.lattice_distance(records['k8'][ck]['fields_lat'], records['k16'][ck]['fields_lat'], refs.n0(i)),
            'k16_to_none': S.lattice_distance(records['k16'][ck]['fields_lat'], records['none'][ck]['fields_lat'], refs.n0(i)),
            'k8_to_none': S.lattice_distance(records['k8'][ck]['fields_lat'], records['none'][ck]['fields_lat'], refs.n0(i))}
        meta['cases'][str(i)]['case_seconds_includes_compile'] = time.perf_counter() - case_start
        save()
        print('completed_case', i, meta['cases'][str(i)]['field_distance'], flush=True)
    # Preserve the registered numeric order; historical lexicographic sorting is inappropriate for subset pilots.
    expected = [f'val64|{i}' for i in ids]
    subset = [f'val64|{i}' for i in cfg['tighter_cases']]
    meta['Q1'] = {a: q1(records[a], tight[a], tighter[a], n0, subset, expected) for a in ARMS}
    # Whole-residual admission and field chains are a separately labelled, new contract.
    whole = {}
    for a in ('k8', 'k16'):
        vals = [meta['cases'][str(i)][a]['whole_residual'] for i in ids]
        dg = np.concatenate([v['delta_g'] for v in vals])
        gf = np.concatenate([v['full_gamma'] for v in vals])
        finite = all(all(v['finite']) for v in vals) and np.isfinite(dg).all() and np.isfinite(gf).all()
        whole[a] = dict(finite=finite, delta_g_max=float(dg.max()), full_gamma_max=float(gf.max()),
                        full_nonstat=int(np.sum(gf > .001)), steps=len(gf),
                        pass_=bool(finite and np.all(dg <= .001) and np.sum(gf > .001) <= .01*len(gf)))
    chain = all(max(meta['cases'][str(i)]['field_distance'].values()) <= .00025 for i in ids)
    meta['qualification'] = dict(scope=cfg['scope'], full_population=ids == list(range(64)),
                               whole_residual=whole, field_chain=chain,
                               passes_observed_population=bool(chain and all(x['pass_'] for x in whole.values())
                                                                and all(q['q1'] for q in meta['Q1'].values())))
    save()
    # Timed fused queries: every output is checked and scored from that actual invocation.
    schedule = R.schedule(3, *opt, 'prod')
    queries = {a: cell.query('none' if a == 'none' else 'ecsw', 512, 'common') for a in ARMS}
    rng = np.random.default_rng(20261026)
    panels = [(i, rep, b) for i in cfg['timing_cases'] for rep in range(cfg['timing_reps']) for b in ('k16', 'none')]
    rng.shuffle(panels)
    for i, rep, b in panels:
        check_budget()
        u, nu = jnp.asarray(fam.ic(64, i)), jnp.float64(fam.nu(i))
        for retry in range(3):
            row = dict(case=i, rep=rep, B=b, retry=retry, invocations=[])
            for arm in ('k8', b, 'k8'):
                P.burn(cfg['burn_seconds'])
                t = time.perf_counter()
                o = jax.block_until_ready(queries[arm](u, nu, data[arm], schedule))
                elapsed = time.perf_counter() - t
                W = np.asarray(o['W'])
                pinned = np.load(out / 'arrays' / f'{arm}_c{i:03}_prod.npz')['W']
                identity = scalar_check(W, pinned)
                score = refs.score(cell.lattice(o['fields']), i)
                row['invocations'].append(dict(arm=arm, seconds=elapsed, W_sha256=array_hash(W),
                                               identity=identity, e0=score))
                assert identity, 'timed/scored output identity failure'
            ts = [v['seconds'] for v in row['invocations']]
            row.update(drift=ts[0]/ts[2], ratio=ts[1]/((ts[0]+ts[2])/2),
                       drift_accepted=abs(ts[0]/ts[2]-1) <= .25)
            meta['timing'].append(row)
            save()
            if row['drift_accepted']:
                break
    meta.update(complete=True, elapsed_seconds=time.perf_counter()-start, peak_device_bytes=P.device_peak(),
                peak_host_bytes=P.host_rss())
    save()
    print('complete', json.dumps(meta['qualification']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--artifacts', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    run(json.loads(Path(a.config).read_text()), Path(a.artifacts), Path(a.out))
