"""jcp-time2 3D driver (DESIGN section 10, amendment A10): second-order time stepping for the off-mesh Burgers 3D ROM.

The 3D lane's model (vendor/quad3d: model_M2 bank, rotation, sine tests completed to their eigenvalue shell, off-mesh
point rules from rules.npz) with the generic LMM stepper of t2core.make_evolve (adaptive LM on every step) in place of
the deployed fixed-sweep backward Euler. The tested advection is the vendored off-mesh point form

    N(c) = P^T ((B c) * (D c)),   dN/dc = P^T (diag(D c) B + diag(B c) D)        (offmesh.adv_offmesh_batch / contract_offmesh)

Per rule (Gauss 24^3 at R' = 512, lattice 4096 at R' = 256) and case of the validation cohort 923801 (first
`cohort_count` cases): the grid of the config (forms x schemes x dt x tolerance level), each an end-to-end query
(L2 projection of the initial field, LMM stepping, decode of six fields). Metrics against the 3D lane's refined
reference (513 nodes, dt = 0.0025 = dt0/4, BE; 65-node lattice): e_ref; anchor discrepancy (GAL-BDF2 at dt0/16);
self-differences d_half; production-vs-tight; tight-vs-tighter (s_h). The deployed fixed-sweep BE query
(offmesh.make_fsc_rule) is run on every case as the deployed comparator. Timing: paired A-B-A with A = the deployed
fixed-sweep BE at dt0. Outputs result.json and W_<rule>.npz.
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
import t2core as T2
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import offmesh as OM  # noqa: E402
from offmesh import C, TB  # noqa: E402
from qpanel import lattice65_index  # noqa: E402

DT0 = .01


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


_BURN = jax.jit(lambda a: a @ a / 512 + .01)


def burn(seconds):
    a = jnp.ones((512, 512), jnp.float64) * .01
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = jax.block_until_ready(_BURN(a))


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def nl3(c, d):
    return ((d['B'] @ c) * (d['D'] @ c)) @ d['P']


def nlJ3(c, d, Tm):
    Bc, Dc = d['B'] @ c, d['D'] @ c
    return (Bc * Dc) @ d['P'], d['P'].T @ (Dc[:, None] * d['B'] + Bc[:, None] * d['D'])


def tol_of(form, level):
    if form == 'LSPG':
        return {'prod': (1e-3, 1e-9), 'tight': (1e-7, 0.), 'tighter': (1e-8, 0.)}[level]
    return {'prod': (0., 1e-10), 'tight': (0., 1e-13), 'tighter': (0., 1e-15)}[level]


def run_list(g):
    dts = [DT0 * f for f in g['dt_factors']]
    tig = [DT0 * f for f in g['tight_factors']]
    out = []
    for form in g['forms']:
        for sc in g['schemes']:
            out += [(form, sc, d, 'prod') for d in dts] + [(form, sc, d, 'tight') for d in tig] + \
                   [(form, sc, d, 'tighter') for d in tig]
        for sc in g['control_schemes']:
            out += [(form, sc, d, 'tight') for d in tig] + [(form, sc, d, 'tighter') for d in tig]
        if form == 'GAL':
            for sc in g['anchor_schemes']:
                out += [(form, sc, DT0 / 16, 'tight'), (form, sc, DT0 / 16, 'tighter')]
    return out


def key(r):
    form, sc, dt, lev = r
    return f'{form}|{sc}|{dt:.8g}|{lev}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu'
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    try:
        smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    n = int(cfg['mesh'])
    Q3 = HERE / 'vendor' / 'quad3d'
    mdir = Q3 / 'inputs' / 'model_M2'
    b = pickle.loads((mdir / 'bank.pkl').read_bytes())
    model = dict(bank=jax.tree_util.tree_map(jnp.asarray, b['params']), T=np.asarray(b['rotation']),
                 spread=b['coefficient_rms_spread'], sha=sha_file(mdir / 'bank.pkl'))
    assert model['sha'] == cfg['expected_model_sha256'], model['sha']
    rpath = Q3 / 'rules' / 'rules.npz'
    assert sha_file(rpath) == cfg['expected_rules_sha256']
    rz = np.load(rpath)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi, mesh=n,
               model_sha256=model['sha'], setup={}, rows=[], vendor_rows=[], timing={}, complete=False)
    save = lambda: C.dump(out / 'result.json', C.clean(rep))
    # ---- cohort and refined reference
    tab = C.table(cfg['cohort_seed'], cfg['cohort_count'])
    rep['cohort'] = dict(seed=cfg['cohort_seed'], count=cfg['cohort_count'], table_sha256=tab['sha256'])
    ref_path = Path(cfg['refined_ref'])
    dn = json.loads(ref_path.with_suffix('.done').read_text())
    assert dn['sha256'] == sha_file(ref_path) and dn['accepted'] and dn['seed'] == cfg['cohort_seed'], dn
    rz_ref = np.load(ref_path)
    assert int(rz_ref['seed']) == cfg['cohort_seed'] and int(rz_ref['count']) >= cfg['cohort_count']
    rep['reference'] = dict(path=str(ref_path), sha256=dn['sha256'], n=int(rz_ref['n']), dt=float(rz_ref['dt']))
    REF = {j: np.asarray(rz_ref[f'c{j}']) for j in range(cfg['cohort_count'])}
    i65 = jnp.asarray(lattice65_index(n))
    # ---- tables (no tensor) and arms
    R = int(model['T'].shape[1])
    Rps = sorted({x['Rp'] for x in cfg['rules']})
    order_, lam_all = C.mode_order(n)
    M_of = lambda r: C.complete_M(n, 4 * r, order_, lam_all)
    tb = TB.build_tables(n, model['bank'], model['T'], sorted(set(Rps) | {R}), M_of, log=lambda s: print(s, flush=True),
                         tensor=False)
    kx = tb['kxyz']
    arms = {}
    for x in cfg['rules']:
        Rp, nm = x['Rp'], x['rule']
        M = int(M_of(Rp))
        X, w = np.asarray(rz[f'{nm}_X']), np.asarray(rz[f'{nm}_w'])
        om = OM.offmesh_tables(n, model['bank'], model['T'], X, w, kx)
        base = dict(GTb=TB.rows_prefix(tb['GTb'], Rp), L=tb['L'][:Rp, :Rp], A=tb['A'][:M, :Rp], lam=tb['lam'][:M],
                    B=om['B'][:, :Rp], D=om['D'][:, :Rp], P=om['P'][:, :M])
        base = jax.tree_util.tree_map(lambda v: jax.block_until_ready(jnp.asarray(v)), base)
        del om
        trust = float(cfg['trust_fraction'] * model['spread'][str(Rp)])
        name = f'{nm}_R{Rp}'
        arms[name] = dict(Rp=Rp, M=M, trust=trust, data={f: T2.form_data(base, {}, f) for f in ('LSPG', 'GAL')},
                          vendor=OM.make_fsc_rule(n, Rp, M, OM.contract_offmesh, dt=DT0, gtol=cfg['gtol'], trust=trust),
                          vdata=base)
        ev = {f: T2.make_evolve(nl3, nlJ3, f, Rp, trust) for f in ('LSPG', 'GAL')}

        def mk(evolve):
            def query(u0, nu, data, sch):
                w0, s0 = TB.project(u0, data)
                W, st = evolve(w0, nu, jnp.linalg.norm(s0), data, sch)
                return dict(fields=TB.decode(W, data), W=W, w0=w0, stats=st)
            return jax.jit(query)
        arms[name]['q'] = {f: mk(ev[f]) for f in ev}
        rep['setup'][name] = dict(Rp=Rp, M=M, m=int(len(w)), trust=trust, A=T2.a_conditioning(base['A']))
    print('SETUP', el(), flush=True)
    runs = run_list(cfg['grid'])
    rep['runs'] = [key(r) for r in runs]
    anchor = key(('GAL', 'BDF2', DT0 / 16, 'tight'))
    order = sorted(range(len(runs)), key=lambda i: key(runs[i]) != anchor)
    cases = list(range(cfg['cohort_count']))
    tc = cfg['timing']
    acc_sha = {}
    for name, arm in arms.items():
        Wset = np.full((len(cases), len(runs), 6, arm['Rp']), np.nan)
        for j in cases:
            u0 = jnp.asarray(C.initial_interior(n, tab, j))
            nu = float(tab['nu'][j])
            R0 = REF[j]
            n0 = float(np.linalg.norm(R0[0]))
            assert float(np.abs(np.asarray(u0)[np.asarray(i65)] - R0[0]).max()) <= 1e-12 * max(1., np.abs(R0[0]).max())
            F = {}
            base_i = len(rep['rows'])
            for i in order:
                form, sc, dt, lev = runs[i]
                g, t_ = tol_of(form, lev)
                t1 = time.perf_counter()
                v = arm['q'][form](u0, nu, arm['data'][form], T2.sched(sc, dt, g, t_))
                jax.block_until_ready(v)
                fr = v['fields'][:, i65]
                F[key(runs[i])] = fr
                Wset[j, i] = np.asarray(v['W'])
                st = {k_: np.asarray(x) for k_, x in v['stats'].items()}
                pe = [float(x) for x in jnp.linalg.norm(fr - jnp.asarray(R0), axis=1) / n0]
                row = dict(arm=name, run=key(runs[i]), form=form, scheme=sc, dt=dt, level=lev, case=j, e_ref_per_time=pe,
                           e_ref=max(pe[1:]), stats=st, seconds_first=time.perf_counter() - t1,
                           verified=bool(int(st['nfail']) == 0 and np.all(np.isfinite(np.asarray(v['W'])))))
                if lev == 'prod' and j < tc['cases']:
                    acc_sha[(name, key(runs[i]), j)] = sha(np.asarray(fr))
                rep['rows'].append(row)
            dist = lambda k1, k2: max(float(x) for x in (jnp.linalg.norm(F[k1] - F[k2], axis=1) / n0)[1:])
            for jj, i in enumerate(order):
                form, sc, dt, lev = runs[i]
                k_ = key(runs[i])
                m = {}
                if anchor in F:
                    m['anchor'] = dist(k_, anchor)
                for other, tag in ((key((form, sc, dt / 2, lev)), 'd_half'), (key((form, sc, dt, 'tight')), 'prod_vs_tight'),
                                   (key((form, sc, dt, 'tighter')), 's_h')):
                    if other in F and other != k_ and (tag != 's_h' or lev == 'tight') and (tag != 'prod_vs_tight' or lev == 'prod'):
                        m[tag] = dist(k_, other)
                rep['rows'][base_i + jj].update(m)
            vo = arm['vendor'](u0, nu, arm['vdata'], {})
            jax.block_until_ready(vo)
            frv = vo[0][:, i65]
            pe = [float(x) for x in jnp.linalg.norm(frv - jnp.asarray(R0), axis=1) / n0]
            dv = lambda Fx: max(float(x) for x in (jnp.linalg.norm(frv - Fx, axis=1) / n0)[1:])
            vrow = dict(arm=name, case=j, e_ref=max(pe[1:]), e_ref_per_time=pe, anchor=dv(F[anchor]),
                        vs_generic_BE=dv(F[key(('LSPG', 'BE', DT0, 'prod'))]),
                        it_sum=int(np.sum(np.asarray(vo[2]))), reasons=np.bincount(np.asarray(vo[3]), minlength=5).tolist())
            if j < tc['cases']:
                acc_sha[(name, 'vendor', j)] = sha(np.asarray(frv))
            rep['vendor_rows'].append(vrow)
            del F
            print('CASE', name, j, el(), flush=True)
            if j % 4 == 3:
                save()
        np.savez_compressed(out / f'W_{name}.npz', W=Wset, runs=np.array([key(r) for r in runs]))
        save()
        # ---- timing (A = deployed fixed-sweep BE at dt0)
        inputs = {j: (jnp.asarray(C.initial_interior(n, tab, j)), float(tab['nu'][j])) for j in range(tc['cases'])}
        cands = [r for r in runs if r[3] == 'prod' and r[2] >= DT0 / 2 - 1e-15]

        def call(B, j):
            u0, nu = inputs[j]
            if B == 'vendor':
                return arm['vendor'](u0, nu, arm['vdata'], {})[0]
            form, sc, dt, lev = B
            g, t_ = tol_of(form, lev)
            return arm['q'][form](u0, nu, arm['data'][form], T2.sched(sc, dt, g, t_))['fields']

        def timed(B, j):
            burn(tc['burn'])
            t1 = time.perf_counter()
            o = call(B, j)
            jax.block_until_ready(o)
            return time.perf_counter() - t1, sha(np.asarray(o[:, i65]))
        for B in ['vendor'] + cands:
            for j in range(tc['cases']):
                timed(B, j)
        rng = np.random.default_rng(int(tc.get('seed', 20261008)))
        inv = []
        for r_ in range(tc['reps']):
            orderT = [(b_, j) for b_ in range(len(cands)) for j in range(tc['cases'])]
            rng.shuffle(orderT)
            for b_, j in orderT:
                B = cands[b_]
                ta1, h1 = timed('vendor', j)
                tb_, hb = timed(B, j)
                ta2, h2 = timed('vendor', j)
                inv.append(dict(arm=name, B=key(B), case=j, rep=r_, tA1=ta1, tB=tb_, tA2=ta2, ratio=tb_ / (.5 * (ta1 + ta2)),
                                drift=ta2 / ta1, A_sha=[h1, h2], A_expected_sha=acc_sha.get((name, 'vendor', j)), B_sha=hb,
                                B_expected_sha=acc_sha.get((name, key(B), j))))
        rep['timing'][name] = dict(A='vendor fixed-sweep BE dt0', invocations=inv, candidates=[key(r) for r in cands])
        save()
        print('TIMING', name, el(), flush=True)
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('T3RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
