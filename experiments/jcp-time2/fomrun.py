"""jcp-time2 fair full-order comparison (DESIGN section 9, A1.10, A2.21, A3.5): the LMM FOM (fom2.py) against the ROM.

Phases
  O  order check at L = 256 on dev6 cases 0, 2: every scheme at dt0 * {2, 1, 1/2, 1/4, 1/8}, ntol 1e-10, ltol 1e-12;
     self-differences d(h) and observed orders (257^2 nodes).
  C  tolerance calibration at the job mesh, dev6 only, per (scheme, dt): the loosest ntol in {1e-4, 1e-6, 1e-8} whose
     fields differ from the ntol 1e-10 solve by < 1 % of that case's ST error in every case; fallback 1e-10 (labelled).
  E  evaluation: every case of the cohorts at the calibrated ntol: e_ST, e_S, e_TX (257^2), verification.
     Also BE at dt0 with ntol 1e-6 / ltol 1e-8 for the reproduction gate against the 2D lane's FOM rows.
  T  timing, paired A-B-A on one GPU: A = FOM BE dt0 at its calibrated ntol; B = every calibrated FOM configuration and
     the ROM arms named in the config (built exactly as in t2run.py); hashes of timed outputs vs the evaluation phase.

    python fomrun.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import t2core as T2
import jax
import jax.numpy as jnp
import fom2

Q = T2.load_qcore()
import qstudy as QS  # noqa: E402

DT0 = .005
LTOL = 1e-8


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def fsched(sc, dt):
    return T2.sched(sc, dt, 0., 0.)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu' or cfg.get('allow_cpu_smoke')
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    L = int(cfg['mesh'])
    s256 = L // 256
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), mesh=L, order=[], calibration={}, rows=[],
               repro=[], timing={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    allc = {k: Q.cohort(k) for k in ('dev6', 'val32')}
    cases = [(coh, c, allc[coh][c]) for coh in cfg['cohorts']
             for c in (cfg.get('case_subset', {}).get(coh) or range(len(allc[coh])))]
    # references (as t2run.py)
    rdir = Path(cfg['refs'])
    man = json.loads((rdir / 'result.json').read_text())
    assert man.get('complete') and man.get('all_accepted')
    idx = {(x['cohort'], x['case'], x['ref']): x for x in man['cases']}
    refs = {}
    for coh, c, ph in cases:
        if not cfg.get('local_smoke_waives_cohort_hash'):
            assert man['cohort_sha256'][coh] == sha(allc[coh]), coh
        for tag in ('ST', 'S'):
            r = np.load(rdir / f'ref_{tag}_{coh}_{c:03d}.npz')['f257']
            assert sha(r) == idx[(coh, c, tag)]['f257_sha256'] and idx[(coh, c, tag)]['accepted']
            refs[(coh, c, tag)] = r

    def errs(fr, coh, c, n0r):
        rST, rS = refs[(coh, c, 'ST')], refs[(coh, c, 'S')]
        o = {}
        for tag, R_ in (('ST', rST), ('S', rS), ('TX', (16 * rST - rS) / 15)):
            o[f'e_{tag}'] = max(float(np.linalg.norm(fr[j] - R_[j])) / n0r for j in range(1, 6))
        return o

    # ---------------------------------------------------------------- O: order check at L = 256 ----
    f256 = fom2.make_fom_lmm(256)
    for coh, c in cfg['order_cases']:
        ph = allc[coh][c]
        u0 = jnp.asarray(Q.e.initial(256, ph))
        n0 = float(np.linalg.norm(np.asarray(u0)))
        for sc in cfg['schemes']:
            F = {}
            for f in (2, 1, .5, .25, .125):
                v = f256(u0, float(ph[4]), fsched(sc, DT0 * f), 1e-10, 1e-12)
                F[f] = (np.asarray(v['fields']), int(v['stats']['nfail']))
            d = {f: max(float(np.linalg.norm(F[f][0][j] - F[f / 2][0][j])) / n0 for j in range(1, 6)) for f in (2, 1, .5, .25)}
            rep['order'].append(dict(cohort=coh, case=c, scheme=sc, d=d, nfail={str(f): F[f][1] for f in F},
                                     orders={str(f): float(np.log2(d[f] / d[f / 2])) for f in (2, 1, .5)}))
    print('ORDER', el(), flush=True)
    save()

    # ---------------------------------------------------------------- C: calibration ----
    fom = fom2.make_fom_lmm(L)
    dev = [x for x in cases if x[0] == 'dev6']
    chosen = {}
    for sc in cfg['schemes']:
        for f in cfg['dt_factors']:
            dt = DT0 * f
            ok = {nt: True for nt in cfg['ntols']}
            ent = dict(cases=[])
            for coh, c, ph in dev:
                u0 = jnp.asarray(Q.e.initial(L, ph))
                n0r = float(np.linalg.norm(np.asarray(u0)[::s256, ::s256]))
                v = fom(u0, float(ph[4]), fsched(sc, dt), 1e-10, 1e-12)
                fr0 = np.asarray(v['fields'][:, ::s256, ::s256])
                ref_ok = int(v['stats']['nfail']) == 0
                eST = errs(fr0, coh, c, n0r)['e_ST']
                row = dict(case=f'{coh}{c}', ref_verified=ref_ok, e_ST_ref=eST, diffs={})
                for nt in cfg['ntols']:
                    w = fom(u0, float(ph[4]), fsched(sc, dt), nt, LTOL)
                    fr = np.asarray(w['fields'][:, ::s256, ::s256])
                    dd = max(float(np.linalg.norm(fr[j] - fr0[j])) / n0r for j in range(1, 6))
                    row['diffs'][str(nt)] = dict(diff=dd, verified=int(w['stats']['nfail']) == 0)
                    ok[nt] = ok[nt] and ref_ok and int(w['stats']['nfail']) == 0 and dd < .01 * eST
                ent['cases'].append(row)
            passing = [nt for nt in sorted(cfg['ntols'], reverse=True) if ok[nt]]
            chosen[(sc, f)] = passing[0] if passing else 1e-10
            ent['chosen_ntol'] = chosen[(sc, f)]
            ent['fallback'] = not passing
            rep['calibration'][f'{sc}|{f:g}'] = ent
    print('CALIBRATION', el(), flush=True)
    save()

    # ---------------------------------------------------------------- E: evaluation ----
    eval_sha = {}
    for ci, (coh, c, ph) in enumerate(cases):
        u0 = jnp.asarray(Q.e.initial(L, ph))
        n0r = float(np.linalg.norm(np.asarray(u0)[::s256, ::s256]))
        for sc in cfg['schemes']:
            for f in cfg['dt_factors']:
                nt = chosen[(sc, f)]
                t1 = time.perf_counter()
                v = fom(u0, float(ph[4]), fsched(sc, DT0 * f), nt, LTOL)
                jax.block_until_ready(v)
                fr = np.asarray(v['fields'][:, ::s256, ::s256])
                st = {k_: np.asarray(x) for k_, x in v['stats'].items()}
                kk = f'fom|{sc}|{f:g}'
                if ci < cfg['timing']['cases'] and coh == 'dev6':
                    eval_sha[(kk, ci)] = sha(fr)
                rep['rows'].append(dict(run=kk, scheme=sc, dt=DT0 * f, ntol=nt, cohort=coh, case=c, seconds_first=time.perf_counter() - t1,
                                        verified=int(st['nfail']) == 0, stats=st, **errs(fr, coh, c, n0r)))
        w = fom(u0, float(ph[4]), fsched('BE', DT0), 1e-6, 1e-8)
        fr = np.asarray(w['fields'][:, ::s256, ::s256])
        rep['repro'].append(dict(cohort=coh, case=c, **errs(fr, coh, c, n0r)))
        print('EVAL', ci, el(), flush=True)
    save()

    # ---------------------------------------------------------------- T: timing ----
    tc = cfg['timing']
    subj = {f'fom|{sc}|{f:g}': ('fom', sc, f) for sc in cfg['schemes'] for f in cfg['dt_factors']}
    rom = {}
    if cfg.get('rom_arms'):
        mdl = Q.Model(ckpt=T2.CKPT)
        settings = sorted({x['setting'] for x in cfg['rom_arms']})
        mesh = QS.Mesh(mdl, L, settings)
        Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
        Hsub = np.asarray(jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))(jnp.asarray(Zsub)))
        for x in cfg['rom_arms']:
            s, form = x['setting'], x['form']
            Rp, M = Q.SETTINGS[s]['Rp'], Q.SETTINGS[s]['M']
            if (s, 'cold') not in rom:
                trust, _ = mdl.trust_linear(Rp)
                o = mesh.ops[M]
                X, w = Q.offmesh_rule(x['rule'])
                rom[(s, 'cold')] = (trust, Q.build_cold_linear(mdl, Rp, Hsub @ mdl.Lrot[:Rp].T),
                                    Q.offmesh_data(mdl, Rp, X, w, L, o['kx'], o['ky'], 'point'))
            trust, cold, blk = rom[(s, 'cold')]
            if (s, form) not in rom:
                rom[(s, form)] = (T2.make_query(Q, 'point', form, Rp, L, trust), T2.form_data(mesh.base(s), blk, form))
            kk = f"rom|{s}|{form}|{x['scheme']}|{x['dt_factor']:g}"
            subj[kk] = ('rom', x, None)
    tcases = [(ci, x) for ci, x in enumerate(cases) if x[0] == 'dev6'][:tc['cases']]
    inputs = {ci: (jnp.asarray(Q.e.initial(L, ph)), float(ph[4])) for ci, (coh, c, ph) in tcases}

    def call(kk, ci):
        kind, x, f = subj[kk]
        u0, nu = inputs[ci]
        if kind == 'fom':
            return fom(u0, nu, fsched(x, DT0 * f), chosen[(x, f)], LTOL)
        qf, dat = rom[(x['setting'], x['form'])]
        trust, cold, _ = rom[(x['setting'], 'cold')]
        return qf(u0, nu, dat, cold, T2.sched(x['scheme'], DT0 * x['dt_factor'], 1e-3 if x['form'] == 'LSPG' else 0.,
                                                1e-9 if x['form'] == 'LSPG' else 1e-10))

    def timed(kk, ci):
        QS.burn(tc['burn'])
        t1 = time.perf_counter()
        o = call(kk, ci)
        jax.block_until_ready(o)
        return time.perf_counter() - t1, sha(np.asarray(o['fields'][:, ::s256, ::s256]))

    A_ = 'fom|BE|1'
    for kk in subj:
        for ci, _ in tcases:
            _, h = timed(kk, ci)
            if kk.startswith('rom|'):
                eval_sha[(kk, ci)] = h          # ROM arms: the warm-up output is the reference hash
    rng = np.random.default_rng(int(tc.get('seed', 20261008)))
    inv = []
    for r_ in range(tc['reps']):
        order = [(kk, ci) for kk in subj if kk != A_ for ci, _ in tcases]
        rng.shuffle(order)
        for kk, ci in order:
            ta1, h1 = timed(A_, ci)
            tb, hb = timed(kk, ci)
            ta2, h2 = timed(A_, ci)
            inv.append(dict(B=kk, case_index=ci, rep=r_, tA1=ta1, tB=tb, tA2=ta2, ratio=tb / (.5 * (ta1 + ta2)),
                            drift=ta2 / ta1, A_sha=[h1, h2], A_expected_sha=eval_sha.get((A_, ci)), B_sha=hb,
                            B_expected_sha=eval_sha.get((kk, ci))))
    rep['timing'] = dict(A=A_, subjects=list(subj), invocations=inv, chosen_ntol={f'{a_}|{b_:g}': v for (a_, b_), v in chosen.items()})
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('FOMRUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
