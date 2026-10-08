"""jcp-time2 fair full-order comparison (DESIGN section 9, A1.10, A2.21, A3.5, A9): the LMM FOM (fom2.py) and the ROM.

Phases (result.json is saved after each phase and every few cases)
  O  order check at L = 256 on the config's order cases: every FOM scheme at dt0 * {2, 1, 1/2, 1/4, 1/8},
     (ntol, ltol) = (1e-10, 1e-12); restricted fields saved (output/order_fields.npz).
  C  tolerance calibration at the job mesh on dev6, per (scheme, dt): the tight solve (1e-10, 1e-12) must verify in all
     six cases, else the configuration is UNRESOLVED and excluded; otherwise the loosest ntol in {1e-4, 1e-6, 1e-8}
     (ltol 1e-8) whose restricted fields differ from the tight solve by < 1 % of the tight solve's ST error in every
     case, else the tight pair itself (fallback, labelled).
  E  evaluation on every case: each resolved FOM configuration at its chosen (ntol, ltol), and each ROM arm of the config
     (production tolerances, as t2run.py): e_ST, e_S, e_TX, verification, restricted-field sha256. Saved: FOM restricted
     fields (float32, all cases; float64 for the timing cases) and ROM coefficients (all cases). Also FOM BE at dt0 with
     (1e-6, 1e-8) for the reproduction gate against the 2D lane's FOM rows.
  T  timing, paired A-B-A on one GPU: A = FOM BE dt0 at its chosen pair; B = every resolved FOM configuration and every
     ROM arm; every timed output's restricted sha256 must equal its evaluation-phase output (same compiled function).

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
TIGHT = (1e-10, 1e-12)
LTOL = 1e-8


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def fsched(sc, dt):
    return T2.sched(sc, dt, 0., 0.)


def rom_key(x):
    return f"rom|{x['setting']}|{x['rule']}|{x['form']}|{x['scheme']}|{x['dt_factor']:g}"


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
               gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), mesh=L, order=[], calibration={},
               rows=[], rom_rows=[], repro=[], timing={}, references={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    allc = {k: Q.cohort(k) for k in ('dev6', 'val32')}
    cases = [(coh, c, allc[coh][c]) for coh in cfg['cohorts']
             for c in (cfg.get('case_subset', {}).get(coh) or range(len(allc[coh])))]
    rep['cohort_sha256'] = {k: sha(v) for k, v in allc.items()}
    rdir = Path(cfg['refs'])
    man = json.loads((rdir / 'result.json').read_text())
    assert man.get('complete') and man.get('all_accepted')
    idx = {(x['cohort'], x['case'], x['ref']): x for x in man['cases']}
    refs = {}
    for coh, c, ph in cases:
        if not cfg.get('local_smoke_waives_cohort_hash'):        # A5.2
            assert man['cohort_sha256'][coh] == rep['cohort_sha256'][coh], coh
        for tag in ('ST', 'S'):
            r = np.load(rdir / f'ref_{tag}_{coh}_{c:03d}.npz')['f257']
            assert sha(r) == idx[(coh, c, tag)]['f257_sha256'] and idx[(coh, c, tag)]['accepted']
            refs[(coh, c, tag)] = r
            rep['references'][f'{tag}|{coh}|{c}'] = idx[(coh, c, tag)]['f257_sha256']

    def errs(fr, coh, c, n0r):
        rST, rS = refs[(coh, c, 'ST')], refs[(coh, c, 'S')]
        o = {}
        for tag, R_ in (('ST', rST), ('S', rS), ('TX', (16 * rST - rS) / 15)):
            o[f'e_{tag}'] = max(float(np.linalg.norm(fr[j] - R_[j])) / n0r for j in range(1, 6))
        return o

    stats_np = lambda st: {k_: np.asarray(x) for k_, x in st.items()}

    # ---------------------------------------------------------------- O: order check at L = 256 ----
    f256 = fom2.make_fom_lmm(256)
    ofields = {}
    for coh, c in cfg['order_cases']:
        ph = allc[coh][c]
        u0 = jnp.asarray(Q.e.initial(256, ph))
        n0 = float(np.linalg.norm(np.asarray(u0)))
        for sc in cfg['schemes']:
            F = {}
            for f in (2, 1, .5, .25, .125):
                v = f256(u0, float(ph[4]), fsched(sc, DT0 * f), *TIGHT)
                F[f] = (np.asarray(v['fields']), int(v['stats']['nfail']))
                ofields[f'{sc}__{coh}__{c}__{f:g}'] = F[f][0]
            d = {f: max(float(np.linalg.norm(F[f][0][j] - F[f / 2][0][j])) / n0 for j in range(1, 6)) for f in (2, 1, .5, .25)}
            rep['order'].append(dict(cohort=coh, case=c, scheme=sc, d={f'{f:g}': v for f, v in d.items()},
                                     nfail={f'{f:g}': F[f][1] for f in F},
                                     orders={f'{f:g}': float(np.log2(d[f] / d[f / 2])) for f in (2, 1, .5)}))
    np.savez_compressed(out / 'order_fields.npz', **ofields)
    print('ORDER', el(), flush=True)
    save()

    # ---------------------------------------------------------------- C: calibration ----
    fom = fom2.make_fom_lmm(L)
    dev = [x for x in cases if x[0] == 'dev6']
    assert len(dev) == 6
    chosen = {}
    CAL = {}                                     # calibration evidence (float32 restricted fields) for the audit
    for sc in cfg['schemes']:
        for f in cfg['dt_factors']:
            dt = DT0 * f
            ok = {nt: True for nt in cfg['ntols']}
            ent = dict(cases=[], tight=list(TIGHT))
            ref_all = True
            for coh, c, ph in dev:
                u0 = jnp.asarray(Q.e.initial(L, ph))
                n0r = float(np.linalg.norm(np.asarray(u0)[::s256, ::s256]))
                v = fom(u0, float(ph[4]), fsched(sc, dt), *TIGHT)
                fr0 = np.asarray(v['fields'][:, ::s256, ::s256])
                CAL[f'{sc}__{f:g}__tight__{coh}__{c}'] = fr0.astype(np.float32)
                ref_ok = int(v['stats']['nfail']) == 0
                ref_all = ref_all and ref_ok
                eST = errs(fr0, coh, c, n0r)['e_ST']
                row = dict(case=f'{coh}{c}', tight_verified=ref_ok, tight_e_ST=eST, diffs={})
                for nt in cfg['ntols']:
                    w = fom(u0, float(ph[4]), fsched(sc, dt), nt, LTOL)
                    fr = np.asarray(w['fields'][:, ::s256, ::s256])
                    CAL[f'{sc}__{f:g}__{nt:g}__{coh}__{c}'] = fr.astype(np.float32)
                    dd = max(float(np.linalg.norm(fr[j] - fr0[j])) / n0r for j in range(1, 6))
                    row['diffs'][f'{nt:g}'] = dict(diff=dd, verified=int(w['stats']['nfail']) == 0)
                    ok[nt] = ok[nt] and int(w['stats']['nfail']) == 0 and dd < .01 * eST
                ent['cases'].append(row)
            if not ref_all:
                ent['status'], chosen[(sc, f)] = 'unresolved', None
            else:
                passing = [nt for nt in sorted(cfg['ntols'], reverse=True) if ok[nt]]
                chosen[(sc, f)] = (passing[0], LTOL) if passing else TIGHT
                ent['status'] = 'calibrated' if passing else 'fallback_tight'
            ent['chosen'] = chosen[(sc, f)]
            rep['calibration'][f'{sc}|{f:g}'] = ent
            save()
    np.savez_compressed(out / 'calibration_fields_f32.npz', **CAL)
    print('CALIBRATION', el(), flush=True)

    # ---------------------------------------------------------------- ROM arms (as t2run.py) ----
    rom = {}
    rom_arms = cfg.get('rom_arms', [])
    if rom_arms:
        mdl = Q.Model(ckpt=T2.CKPT)
        settings = sorted({x['setting'] for x in rom_arms})
        mesh = QS.Mesh(mdl, L, settings)
        Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
        Hsub = np.asarray(jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))(jnp.asarray(Zsub)))
        for x in rom_arms:
            s, rule, form = x['setting'], x['rule'], x['form']
            Rp, M = Q.SETTINGS[s]['Rp'], Q.SETTINGS[s]['M']
            if (s, rule, 'cold') not in rom:
                trust, _ = mdl.trust_linear(Rp)
                o = mesh.ops[M]
                X, w = Q.offmesh_rule(rule)
                rom[(s, rule, 'cold')] = (trust, Q.build_cold_linear(mdl, Rp, Hsub @ mdl.Lrot[:Rp].T),
                                          Q.offmesh_data(mdl, Rp, X, w, L, o['kx'], o['ky'], 'point'))
            trust, cold, blk = rom[(s, rule, 'cold')]
            if (s, rule, form) not in rom:
                rom[(s, rule, form)] = (T2.make_query(Q, 'point', form, Rp, L, trust), T2.form_data(mesh.base(s), blk, form))

    def rom_call(x, u0, nu):
        qf, dat = rom[(x['setting'], x['rule'], x['form'])]
        _, cold, _ = rom[(x['setting'], x['rule'], 'cold')]
        lspg = x['form'] == 'LSPG'
        return qf(u0, nu, dat, cold, T2.sched(x['scheme'], DT0 * x['dt_factor'], 1e-3 if lspg else 0., 1e-9 if lspg else 1e-10))

    # ---------------------------------------------------------------- E: evaluation ----
    eval_sha, F32, F64, Wrom = {}, {}, {}, {}
    nt_cases = cfg['timing']['cases']
    tidx = [ci for ci, x in enumerate(cases) if x[0] == 'dev6'][:nt_cases]     # explicit timing-case indices
    rep['timing_case_indices'] = tidx

    def dump_arrays():
        np.savez_compressed(out / 'fom_fields_f32.npz', **F32)
        np.savez_compressed(out / 'fom_fields_timing_f64.npz', **F64)
        np.savez_compressed(out / 'rom_W.npz', **Wrom)
    for ci, (coh, c, ph) in enumerate(cases):
        u0 = jnp.asarray(Q.e.initial(L, ph))
        nu = float(ph[4])
        n0r = float(np.linalg.norm(np.asarray(u0)[::s256, ::s256]))
        timing_case = ci in tidx
        for sc in cfg['schemes']:
            for f in cfg['dt_factors']:
                pair = chosen[(sc, f)]
                if pair is None:
                    continue
                t1 = time.perf_counter()
                v = fom(u0, nu, fsched(sc, DT0 * f), *pair)
                jax.block_until_ready(v)
                fr = np.asarray(v['fields'][:, ::s256, ::s256])
                kk = f'fom|{sc}|{f:g}'
                F32[f'{kk}|{coh}|{c}'.replace('|', '__')] = fr.astype(np.float32)
                if timing_case:
                    eval_sha[(kk, ci)] = sha(fr)
                    F64[f'{kk}|{coh}|{c}'.replace('|', '__')] = fr
                rep['rows'].append(dict(run=kk, scheme=sc, dt=DT0 * f, ntol=pair[0], ltol=pair[1], cohort=coh, case=c,
                                        seconds_first=time.perf_counter() - t1, verified=int(v['stats']['nfail']) == 0,
                                        restricted_sha256=sha(fr), stats=stats_np(v['stats']), **errs(fr, coh, c, n0r)))
        for x in rom_arms:
            v = rom_call(x, u0, nu)
            jax.block_until_ready(v)
            fr = np.asarray(v['fields'][:, ::s256, ::s256])
            kk = rom_key(x)
            Wrom[f'{kk}|{coh}|{c}'.replace('|', '__')] = np.asarray(v['W'])
            if timing_case:
                eval_sha[(kk, ci)] = sha(np.asarray(v['W']))       # ROM outputs are bound by their coefficients
            st = stats_np(v['stats'])
            rep['rom_rows'].append(dict(run=kk, **x, cohort=coh, case=c, verified=bool(int(st['nfail']) == 0),
                                        restricted_sha256=sha(fr), stats=st, **errs(fr, coh, c, n0r)))
        w = fom(u0, nu, fsched('BE', DT0), 1e-6, 1e-8)
        rep['repro'].append(dict(cohort=coh, case=c, **errs(np.asarray(w['fields'][:, ::s256, ::s256]), coh, c, n0r)))
        print('EVAL', ci, el(), flush=True)
        if ci % 8 == 7:
            save()
            dump_arrays()
    dump_arrays()
    save()

    # ---------------------------------------------------------------- T: timing ----
    tc = cfg['timing']
    subj = {f'fom|{sc}|{f:g}': ('fom', sc, f) for sc in cfg['schemes'] for f in cfg['dt_factors'] if chosen[(sc, f)]}
    subj.update({rom_key(x): ('rom', x, None) for x in rom_arms})
    A_ = 'fom|BE|1'
    assert A_ in subj, 'baseline FOM unresolved'
    tcases = [(ci, cases[ci]) for ci in tidx]
    inputs = {ci: (jnp.asarray(Q.e.initial(L, ph)), float(ph[4])) for ci, (coh, c, ph) in tcases}

    def call(kk, ci):
        kind, x, f = subj[kk]
        u0, nu = inputs[ci]
        if kind == 'fom':
            return fom(u0, nu, fsched(x, DT0 * f), *chosen[(x, f)])
        return rom_call(x, u0, nu)

    def timed(kk, ci):
        QS.burn(tc['burn'])
        t1 = time.perf_counter()
        o = call(kk, ci)
        jax.block_until_ready(o)
        secs = time.perf_counter() - t1
        return secs, (sha(np.asarray(o['W'])) if kk.startswith('rom|') else sha(np.asarray(o['fields'][:, ::s256, ::s256])))

    for kk in subj:                      # warm every subject (compiled already in phase E)
        for ci, _ in tcases:
            timed(kk, ci)
    n_inv = 0
    rng = np.random.default_rng(int(tc.get('seed', 20261008)))
    inv = []
    rep['timing'] = dict(A=A_, subjects=list(subj), invocations=inv,
                         chosen={f'{a_}|{b_:g}': v for (a_, b_), v in chosen.items()})
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
            n_inv += 1
            if n_inv % 50 == 0:
                save()
        save()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('FOMRUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
