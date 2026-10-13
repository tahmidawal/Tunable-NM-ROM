"""jcp-combined 2D grid job `c2grid` (DESIGN 3.1, 7, 8; A1.0, A1.6, A2.10, A2.11, A4). One job, one GPU:

  0  setup: cohorts (dev6, val32; disjoint from training), references (manifest + sha256), frozen model, rotated bank
     (edges at every R'), operators at M_max (nested prefixes asserted).
  1  per setting (name, R', M = 4R', main rule), per case: every production run of the grid (forms x schemes x
     dt factors), the deployed-rule anchor GAL-BDF2 at dt0/8 and dt0/16 (tight and tighter, A1.6) and the vendor
     backward-Euler query (G1a: generic LSPG-BE must equal it in the same job). Saved per run: the six output
     coefficient vectors W (6, R') and w0 (W_<setting>.npz), statistics, e_ST / e_S / e_TX per time (in-job scoring;
     offline score.py recomputes them through an independent decode path).
  2  FOM (C2 fom2.make_fom_lmm): order check at L=256 on two dev6 cases; tolerance calibration on dev6 (C2 A9.2);
     evaluation of every resolved configuration on every case (fields saved float64 on the 257^2 nodes, A4.2: the
     scoring lattice of any later reference is a subset of every compared mesh, including 256^2, plan E0).
  3  timing on one GPU: (a) per-setting A-B-A panels (A = that setting's LSPG-BE dt0, B = every production arm with
     dt >= dt0/2 of that setting and the vendor BE query); (b) one common panel (A = the `acc` setting's LSPG-BE dt0;
     B = every production arm with dt >= dt0/2 of every setting and every resolved FOM configuration). Every timed
     ROM output's W sha256 must equal the accuracy-phase W of the same (run, case); FOM outputs are bound by their
     restricted-field sha256. 0.1 s burn before each call; randomised order; every invocation saved.
Outputs: result.json, W_<setting>.npz, fom_f257.npz, calibration_fields.npz, order_fields.npz, COMPLETE.

    python c2grid.py --config configs/<attempt>.json --out <dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import c2common as CC
from c2common import Q, QS, T2, DT0, sha
import fom2
import timing as TM
import jax
import jax.numpy as jnp

TIGHT = (1e-10, 1e-12)
LTOL = 1e-8


def run_list(g):
    """(form, scheme, dt, level) tuples: production grid then the anchors (A1.6)."""
    runs = []
    for form, sc in g['forms_schemes']:
        runs += [(form, sc, DT0 * f, 'prod') for f in g['dt_factors']]
    for f in g['anchor_factors']:
        for lev in ('tight', 'tighter'):
            runs.append(('GAL', 'BDF2', DT0 * f, lev))
    return runs


def key(r):
    form, sc, dt, lev = r
    return f'{form}|{sc}|{dt:.8g}|{lev}'


def fsched(sc, dt):
    return T2.sched(sc, dt, 0., 0.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true', help='print the run inventory and exit')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    runs = run_list(cfg['grid'])
    if a.dry_run:
        inv = dict(settings=[s['name'] for s in cfg['settings']], runs_per_case=len(runs),
                   steps_per_case=sum(int(round(.25 / r[2])) for r in runs), runs=[key(r) for r in runs])
        print(json.dumps(inv, indent=1))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    CC.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    L = int(cfg['mesh'])
    s256 = max(1, L // 256)
    rep = dict(config=cfg, commit=CC.os.environ.get('SOURCE_COMMIT'), job_id=CC.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=CC.gpu_info(),
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(T2.CKPT), cohorts={}, references={},
               setup={}, rows=[], g1a=[], fom={}, timing={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    cases = CC.load_cases(cfg, rep)
    refs = CC.load_refs(cfg, cases, rep)
    print('CASES', len(cases), 'REFS', len(refs), el(), flush=True)
    bank = CC.Bank(cfg, L)
    rep['setup']['bank'] = dict(edges=bank.edges, seconds=bank.seconds, M_max=bank.Mmax)
    U0 = {}
    for coh, c, ph in cases:
        u0n = Q.e.initial(L, ph)
        U0[(coh, c)] = (jnp.asarray(u0n), float(ph[4]), float(np.linalg.norm(u0n[::s256, ::s256])))
    rep['run_keys'] = [key(r) for r in runs]
    tc = cfg['timing']
    tci = [ci for ci, x in enumerate(cases) if x[0] == tc['cohort']][:tc['cases']]
    rep['timing_case_indices'] = tci
    Wsha = {}                                    # (setting, run key, case index) -> sha of W (timing binding)
    settings = {}
    print('BANK', el(), flush=True)

    # ------------------------------------------------------------------ 1: accuracy per setting ----
    for spec in cfg['settings']:
        st = settings[spec['name']] = CC.Setting(bank, spec)
        rep['setup'][st.name] = dict(Rp=st.Rp, M=st.M, rule=spec['rule'], m=st.m['main'], trust=st.trust,
                                     A=T2.a_conditioning(st.base['A']))
        Wset = np.full((len(cases), len(runs), 6, st.Rp), np.nan)
        w0set = np.full((len(cases), st.Rp), np.nan)
        for ci, (coh, c, ph) in enumerate(cases):
            u0, nu, n0r = U0[(coh, c)]
            for i, r in enumerate(runs):
                form, sc, dt, lev = r
                t1 = time.perf_counter()
                v = st.call('main', form, sc, dt, lev, u0, nu)
                jax.block_until_ready(v)
                secs = time.perf_counter() - t1
                W = np.asarray(v['W'])
                Wset[ci, i] = W
                w0 = np.asarray(v['w0'])
                if i == 0:
                    w0set[ci] = w0
                assert np.array_equal(w0, w0set[ci]), 'initial coefficients differ across runs'
                stt = {k: np.asarray(x) for k, x in v['stats'].items()}
                fr = np.asarray(v['fields'][:, ::s256, ::s256])
                row = dict(setting=st.name, run=key(r), form=form, scheme=sc, dt=dt, level=lev, cohort=coh, case=c,
                           seconds_first=secs, stats=stt, finite=bool(np.isfinite(fr).all()),
                           verified=bool(int(stt['nfail']) == 0 and np.isfinite(W).all()), **CC.errs(fr, refs, coh, c, n0r))
                if ci in tci:
                    Wsha[(st.name, key(r), ci)] = sha(W)
                rep['rows'].append(row)
                del v
            # G1a: vendor BE query vs generic LSPG-BE dt0 (same job)
            vv = st.vendor(u0, nu)
            jax.block_until_ready(vv)
            ig = runs.index(('LSPG', 'BE', DT0, 'prod'))
            Wv = np.asarray(vv['internal'])[::int(round(.05 / DT0))]
            rep['g1a'].append(dict(setting=st.name, cohort=coh, case=c,
                                   w0_rel=float(np.linalg.norm(w0set[ci] - Wv[0]) / np.linalg.norm(Wv[0])),
                                   W_rel=float(np.max(np.linalg.norm(Wset[ci, ig] - Wv, axis=1)) / np.linalg.norm(Wv[0])),
                                   it_vendor=int(np.sum(np.asarray(vv['it'])))))
            if ci in tci:
                Wsha[(st.name, 'vendor', ci)] = sha(Wv)
            g_ = rep['g1a'][-1]
            g_['passed'] = bool(g_['W_rel'] <= 1e-8 and g_['w0_rel'] <= 1e-12)     # C2 A2.9 (same-job gate, A1.10)
            if not g_['passed']:
                save()
                print('G1a FAILED: generic LSPG-BE differs from the vendor query; stopping before any verdict', g_, flush=True)
                raise SystemExit(3)
            del vv
            print('CASE', st.name, ci, coh, c, el(), flush=True)
            if ci % 8 == 7:
                save()
        try:
            rep['setup'][st.name]['peak_bytes'] = int(jax.devices()[0].memory_stats().get('peak_bytes_in_use', -1))
        except Exception:  # noqa: BLE001
            rep['setup'][st.name]['peak_bytes'] = -1
        np.savez_compressed(out / f'W_{st.name}.npz', W=Wset, w0=w0set, runs=np.array([key(r) for r in runs]),
                            cases=np.array([f'{coh}|{c}' for coh, c, _ in cases]))
        save()
        gc.collect()

    # ------------------------------------------------------------------ 2: FOM ----
    fc = cfg['fom']
    f256 = fom2.make_fom_lmm(256)
    allc = {k: Q.cohort(k) for k in cfg['cohorts']}
    order, ofields = [], {}
    for coh, c in fc['order_cases']:
        ph = allc[coh][c]
        u0 = jnp.asarray(Q.e.initial(256, ph))
        n0 = float(np.linalg.norm(np.asarray(u0)))
        for sc in fc['schemes']:
            F = {}
            secs_o = {}
            for f in (2, 1, .5, .25, .125):
                t1 = time.perf_counter()
                v = f256(u0, float(ph[4]), fsched(sc, DT0 * f), *TIGHT)
                jax.block_until_ready(v)
                secs_o[f'{f:g}'] = time.perf_counter() - t1
                F[f] = (np.asarray(v['fields']), int(v['stats']['nfail']))
                ofields[f'{sc}__{coh}__{c}__{f:g}'] = F[f][0]
            d = {f: max(float(np.linalg.norm(F[f][0][j] - F[f / 2][0][j])) / n0 for j in range(1, 6)) for f in (2, 1, .5, .25)}
            order.append(dict(cohort=coh, case=c, scheme=sc, seconds=secs_o, d={f'{f:g}': v for f, v in d.items()},
                              nfail={f'{f:g}': F[f][1] for f in F},
                              orders={f'{f:g}': float(np.log2(d[f] / d[f / 2])) for f in (2, 1, .5)}))
    np.savez_compressed(out / 'order_fields.npz', **ofields)
    rep['fom']['order'] = order
    del f256, ofields
    rep.setdefault('phase_marks', {})['fom_order_done'] = el()
    print('FOM ORDER', el(), flush=True)
    fom = fom2.make_fom_lmm(L)
    dev = [x for x in cases if x[0] == 'dev6']
    chosen, CAL, calrep = {}, {}, {}
    for sc in fc['schemes']:
        for f in fc['dt_factors']:
            dt = DT0 * f
            ok = {nt: True for nt in fc['ntols']}
            ent = dict(cases=[], tight=list(TIGHT))
            ref_all = True
            for coh, c, ph in dev:
                u0, nu, n0r = U0[(coh, c)]
                t1 = time.perf_counter()
                v = fom(u0, nu, fsched(sc, dt), *TIGHT)
                jax.block_until_ready(v)
                t_tight = time.perf_counter() - t1
                fr0 = np.asarray(v['fields'][:, ::s256, ::s256])
                CAL[f'{sc}__{f:g}__tight__{coh}__{c}'] = fr0.astype(np.float32)
                ref_ok = int(v['stats']['nfail']) == 0
                ref_all = ref_all and ref_ok
                eST = CC.errs(fr0, refs, coh, c, n0r)['e_ST']
                row = dict(case=f'{coh}{c}', tight_verified=ref_ok, tight_e_ST=eST, diffs={}, seconds_tight=t_tight)
                for nt in fc['ntols']:
                    t1 = time.perf_counter()
                    w = fom(u0, nu, fsched(sc, dt), nt, LTOL)
                    jax.block_until_ready(w)
                    row.setdefault('seconds', {})[f'{nt:g}'] = time.perf_counter() - t1
                    fr = np.asarray(w['fields'][:, ::s256, ::s256])
                    dd = max(float(np.linalg.norm(fr[j] - fr0[j])) / n0r for j in range(1, 6))
                    row['diffs'][f'{nt:g}'] = dict(diff=dd, verified=int(w['stats']['nfail']) == 0)
                    ok[nt] = ok[nt] and int(w['stats']['nfail']) == 0 and dd < .01 * eST
                ent['cases'].append(row)
            if not ref_all:
                ent['status'], chosen[(sc, f)] = 'unresolved', None
            else:
                passing = [nt for nt in sorted(fc['ntols'], reverse=True) if ok[nt]]
                chosen[(sc, f)] = (passing[0], LTOL) if passing else TIGHT
                ent['status'] = 'calibrated' if passing else 'fallback_tight'
            ent['chosen'] = chosen[(sc, f)]
            calrep[f'{sc}|{f:g}'] = ent
    rep['fom']['calibration'] = calrep
    np.savez_compressed(out / 'calibration_fields.npz', **CAL)
    del CAL
    save()
    rep.setdefault('phase_marks', {})['fom_calibration_done'] = el()
    print('FOM CALIBRATION', el(), flush=True)
    F64, frows, fsha = {}, [], {}
    for ci, (coh, c, ph) in enumerate(cases):
        u0, nu, n0r = U0[(coh, c)]
        for sc in fc['schemes']:
            for f in fc['dt_factors']:
                pair = chosen[(sc, f)]
                if pair is None:
                    continue
                t1 = time.perf_counter()
                v = fom(u0, nu, fsched(sc, DT0 * f), *pair)
                jax.block_until_ready(v)
                secs_e = time.perf_counter() - t1
                fr = np.asarray(v['fields'][:, ::s256, ::s256])
                kk = f'fom|{sc}|{f:g}'
                F64[f'{kk}|{coh}|{c}'.replace('|', '__')] = fr
                if ci in tci:
                    fsha[(kk, ci)] = sha(fr)
                frows.append(dict(run=kk, scheme=sc, dt=DT0 * f, ntol=pair[0], ltol=pair[1], cohort=coh, case=c, seconds=secs_e,
                                  verified=int(v['stats']['nfail']) == 0,
                                  stats={k_: np.asarray(x) for k_, x in v['stats'].items()}, **CC.errs(fr, refs, coh, c, n0r)))
        print('FOM EVAL', ci, el(), flush=True)
    rep['fom']['rows'] = frows
    rep.setdefault('phase_marks', {})['fom_eval_done'] = el()
    F64['__coords_x'] = np.arange(257) / 256.          # nodes x = i/256 (both axes, 'ij' order), boundary included
    F64['__times'] = np.arange(6) * .05
    np.savez_compressed(out / 'fom_f257.npz', **F64)
    del F64
    save()

    # ------------------------------------------------------------------ 3: timing ----
    inputs = {ci: (U0[(cases[ci][0], cases[ci][1])][0], U0[(cases[ci][0], cases[ci][1])][1]) for ci in tci}
    timed_runs = [r for r in runs if r[3] == 'prod' and r[2] >= DT0 / 2 - 1e-15]

    def call(subj, ci):
        u0, nu = inputs[ci]
        if subj[0] == 'fom':
            sc, f = subj[1], subj[2]
            o = fom(u0, nu, fsched(sc, DT0 * f), *chosen[(sc, f)])
            return o, lambda: sha(np.asarray(o['fields'][:, ::s256, ::s256]))
        sname, r = subj[1], subj[2]
        st = settings[sname]
        if r == 'vendor':
            o = st.vendor(u0, nu)
            return o, lambda: sha(np.asarray(o['internal'])[::int(round(.05 / DT0))])
        o = st.call('main', *r, u0, nu)
        return o, lambda: sha(np.asarray(o['W']))

    def expected(subj, ci):
        if subj[0] == 'fom':
            return fsha.get((f'fom|{subj[1]}|{subj[2]:g}', ci))
        return Wsha.get((subj[1], 'vendor' if subj[2] == 'vendor' else key(subj[2]), ci))

    def name(subj):
        if subj[0] == 'fom':
            return f'fom|{subj[1]}|{subj[2]:g}'
        return f"{subj[1]}|{'vendor' if subj[2] == 'vendor' else key(subj[2])}"

    def timed(subj, ci):
        QS.burn(tc['burn'])
        t1 = time.perf_counter()
        o, h = call(subj, ci)
        jax.block_until_ready(o)
        secs = time.perf_counter() - t1
        return secs, h()

    def panel(label, A, Bs, seed):
        for s_ in [A] + Bs:                       # warm every subject on every timing case
            for ci in tci:
                timed(s_, ci)
        rng = np.random.default_rng(seed)
        inv = []
        for rr in range(tc['reps']):
            order_ = [(j, ci) for j in range(len(Bs)) for ci in tci]
            rng.shuffle(order_)
            for j, ci in order_:
                ta1, h1 = timed(A, ci)
                tb, hb = timed(Bs[j], ci)
                ta2, h2 = timed(A, ci)
                eA, eB = expected(A, ci), expected(Bs[j], ci)
                inv.append(dict(B=name(Bs[j]), case_index=ci, rep=rr, tA1=ta1, tB=tb, tA2=ta2, ratio=tb / (.5 * (ta1 + ta2)),
                                drift=ta2 / ta1, A_sha=[h1, h2], A_expected=eA, B_sha=hb, B_expected=eB,
                                hashes_ok=bool(eA is not None and eB is not None and h1 == eA and h2 == eA and hb == eB)))
        rep['timing'][label] = dict(A=name(A), subjects=[name(b) for b in Bs], invocations=inv, gpu=CC.gpu_info(),
                                    cases=tci, reps=tc['reps'], burn=tc['burn'], seed=seed, **TM.panel_validity(inv))
        save()
        print('TIMING', label, len(inv), el(), flush=True)

    for k_, spec in enumerate(cfg['settings']):
        sname = spec['name']
        A = ('rom', sname, ('LSPG', 'BE', DT0, 'prod'))
        Bs = [('rom', sname, r) for r in timed_runs if r != A[2]] + [('rom', sname, 'vendor')]
        panel(f'setting|{sname}', A, Bs, int(tc['seed']) + 1 + k_)
        rep.setdefault('phase_marks', {})[f'timing_{sname}_done'] = el()
    A = ('rom', cfg['common_baseline'], ('LSPG', 'BE', DT0, 'prod'))
    Bs = [('rom', s['name'], r) for s in cfg['settings'] for r in timed_runs if ('rom', s['name'], r) != A]
    Bs += [('fom', sc, f) for sc in fc['schemes'] for f in fc['dt_factors'] if chosen[(sc, f)] is not None]
    panel('common', A, Bs, int(tc['common_seed']))
    try:
        rep['peak_bytes_job'] = int(jax.devices()[0].memory_stats().get('peak_bytes_in_use', -1))
    except Exception:  # noqa: BLE001
        rep['peak_bytes_job'] = -1
    rep['phase_seconds'] = dict(total=el())
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('C2GRID COMPLETE', flush=True)


if __name__ == '__main__':
    main()
