"""jcp-p1-scaling timing job (P1c; DESIGN 4.1, 6, 7; A1.10, A2.2, A2.3, A3.2). One job, one GPU, every mesh co-resident.

  0  banks for every mesh of the config (explicit edges; 4096^2 prefix-retaining if configured), dev6 inputs per mesh
  1  per subject (mesh, r, rule, policy, scheme) and case: an untimed scoring run through the split query (init ->
     evolve -> decode) of the RECORDING program (nrec = 51, the P1d production program). Its trajectory is certified
     in-job (autodiff gamma, G-cert, validity); tight / tighter / successor / check replays and own-state rho give the
     timing-job Q1/Q2 label (A2.3, A3.2). Split vs fused equivalence on the first case.
  2  A-B-A triplets in randomised order: burn, A evolve | burn, B init | burn, B evolve | burn, B decode | burn, A evolve;
     drift t_A2/t_A1 outside [1/1.1, 1.1] -> the triplet is repeated (<= 2 retries), else marked invalid; then a fused
     triplet (burn, A evolve, burn, B fused query, burn, A evolve), descriptive. Every timed evolution must reproduce
     its scoring run's identity hash: W, every internal state, per-step exit reasons and iteration counts.
  3  doubled work (AM-8): single-double-single, two separately dispatched synchronised evolutions from the same state
  4  component-sum vs evolution (A1.2, A2.2): reported diagnostic only
  5  optional recording-overhead probe (A3.2): recording vs non-recording evolve, paired
  6  memory per subject: live table bytes, compiled evolve memory_analysis, process peak (process-lifetime)

    python p1time.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import p1common as P
from p1common import Q, QS, T2, DT0, sha
import p1comp as PC
import jax
import jax.numpy as jnp

BAND = (1 / 1.1, 1.1)
NREC = int(round(.25 / DT0)) + 1


def sync_time(fn):
    t = time.perf_counter()
    o = fn()
    jax.block_until_ready(o)
    return time.perf_counter() - t, o


def ident(W, st_):
    """Identity hash of one evolution: W, every internal state, per-step exit reasons and iteration counts (A3.2)."""
    return sha(np.concatenate([np.asarray(W).ravel(), np.asarray(st_['states']).ravel(),
                               np.asarray(st_['step_reason'], float), np.asarray(st_['step_it'], float)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    burn = float(cfg['burn'])
    rep = dict(config=cfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, checkpoint_sha256=QS.sha_file(T2.CKPT), cohorts=P.cohort_hashes(),
               banks={}, subjects={}, invocations=[], fused=[], doubled=[], components=[], equivalence=[],
               recording_overhead=[], memory={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    cases = P.case_list(cfg)
    tci = P.select(cases, cfg['cases'])
    banks, U0 = {}, {}
    for Ls, mc in cfg['meshes'].items():
        L = int(Ls)
        b = banks[L] = P.Bank(L, mc['edges'], nrb=mc.get('bank_blocks'))
        rep['banks'][Ls] = dict(edges=b.edges, seconds=b.build_seconds, bytes=b.bytes, mem=P.mem())
        s = L // 256
        for ci in tci:
            u0n = Q.e.initial(L, cases[ci][2])
            U0[(L, ci)] = (jnp.asarray(u0n), float(cases[ci][2][4]), float(np.linalg.norm(u0n[::s, ::s])))
        print('BANK', L, el(), P.mem(), flush=True)
    settings = {}

    def label(sp):
        return f"{sp['mesh']}|R{sp['Rp']}|{sp['rule']}|{sp.get('policy', 'frozen')}|{sp['scheme']}"

    def setting(sp):
        k = (int(sp['mesh']), int(sp['Rp']), sp['rule'], sp.get('policy', 'frozen'))
        if k not in settings:
            b0 = P.mem()['in_use']
            roles = {kk: sp[kk] for kk in ('qp', 'qpp') if sp.get(kk)}
            settings[k] = P.Setting(banks[k[0]], f'{k}', k[1], k[2], k[3], roles=roles, nrec=NREC)
            settings[k].live_bytes = P.mem()['in_use'] - b0
        st_ = settings[k]
        for kk in ('qp', 'qpp'):                      # a later subject of the same setting may ask for Q2 roles
            if sp.get(kk) and kk not in st_.rules:
                b0 = P.mem()['in_use']
                st_.add_role(banks[k[0]], kk, sp[kk])
                st_.live_bytes += P.mem()['in_use'] - b0
            elif sp.get(kk):
                assert st_.rules[kk] == sp[kk], (kk, st_.rules[kk], sp[kk])
        return st_

    subs = [cfg['A']] + cfg['subjects']
    W0, ref = {}, {}
    for sp in subs:
        lb = label(sp)
        if lb in rep['subjects']:
            continue
        st = setting(sp)
        L = int(sp['mesh'])
        bank = banks[L]
        q = st.q['LSPG']
        d = st.data[('main', 'LSPG')]
        sch = st.sched('LSPG', sp['scheme'], DT0, 'prod')
        cert = T2.make_certifier(q['nl'], st.Rp)
        nl = q['nl']
        n = len(tci)
        C = dict(W=np.full((n, 6, st.Rp), np.nan), w0=np.full((n, st.Rp), np.nan), states=np.full((n, NREC, st.Rp), np.nan),
                 gamma=np.full((n, NREC - 1), np.nan), gn_rec=np.full((n, NREC - 1), np.nan),
                 it=np.full((n, NREC - 1), -1), reason=np.full((n, NREC - 1), -1), n0=np.full(n, np.nan),
                 W_tight=np.full((n, 6, st.Rp), np.nan), W_tighter=np.full((n, 6, st.Rp), np.nan),
                 W_qp=np.full((n, 6, st.Rp), np.nan), W_qpp=np.full((n, 6, st.Rp), np.nan), rho=np.full((n, 6), np.nan),
                 f257=np.full((n, 6, 257, 257), np.nan))
        rows = []
        for j, ci in enumerate(tci):
            st.check('main', bank)
            u0, nu, n0 = U0[(L, ci)]
            w0, scale = jax.block_until_ready(q['init'](u0, st.cold))
            W, s_ = jax.block_until_ready(q['evolve'](w0, nu, scale, d, sch))
            F = jax.block_until_ready(q['decode'](W, d))
            W0[(lb, ci)] = (w0, scale)
            ref[(lb, ci)] = ident(W, s_)
            if lb in cfg.get('inject_identity_fault', []):     # smoke control (A4.1): a wrong reference must invalidate
                ref[(lb, ci)] = 'injected-' + ref[(lb, ci)]
            f = P.restrict(F, L)
            g, _ = cert(s_['states'], nu, d, sch)
            S_ = NREC - 1
            C['W'][j], C['w0'][j], C['states'][j], C['gamma'][j], C['n0'][j] = (np.asarray(W), np.asarray(w0),
                                                                                np.asarray(s_['states']), np.asarray(g), n0)
            C['gn_rec'][j], C['it'][j], C['reason'][j] = (np.asarray(s_[k])[:S_] for k in ('step_gn', 'step_it', 'step_reason'))
            assert np.asarray(s_['states']).shape[0] == int(round(.25 / DT0)) + 1
            C['f257'][j] = f
            row = dict(case=f'{cases[ci][0]}|{cases[ci][1]}', valid=P.run_valid(dict(W=W, w0=w0, fields=F, stats=s_)),
                       budget_exits=int(np.asarray(s_['exits'])[0]), nonstationary=int(np.sum(~(np.asarray(g) <= 1e-3))),
                       gcert_ok=P.gcert_ok(g, C['gn_rec'][j]), ident=ref[(lb, ci)])
            if j == 0:
                fq = jax.block_until_ready(q['fused'](u0, nu, d, st.cold, sch))
                fr = float(jnp.max(jnp.abs(fq['fields'] - F)) / jnp.max(jnp.abs(fq['fields'])))
                same_it = bool(np.array_equal(np.asarray(fq['stats']['step_it']), np.asarray(s_['step_it'])))
                rep['equivalence'].append(dict(subject=lb, fields_rel=fr, same_iterations=same_it,
                                               passed=bool(fr <= 1e-12 and same_it)))
            for lev, key in (('tight', 'W_tight'), ('tighter', 'W_tighter')):
                w = jax.block_until_ready(P.run_query(bank, st, 'main', 'LSPG', sp['scheme'], DT0, lev, u0, nu, record=False))
                C[key][j] = np.asarray(w['W'])
                row[f'{lev}_valid'] = P.run_valid(w)
                row[f'_f_{lev}'] = P.restrict(w['fields'], L)
            row['dist_tight'] = P.fdist(f, row['_f_tight'], n0)
            row['dist_tight_tighter'] = P.fdist(row.pop('_f_tight'), row.pop('_f_tighter'), n0)
            if 'qp' in st.rules:
                q_ = jax.block_until_ready(P.run_query(bank, st, 'qp', 'LSPG', sp['scheme'], DT0, 'prod', u0, nu, record=False))
                C['W_qp'][j] = np.asarray(q_['W'])
                fq_ = P.restrict(q_['fields'], L)
                row.update(dist_qp=P.fdist(f, fq_, n0), qp_valid=P.run_valid(q_))
                for kk in range(6):
                    c_ = jnp.asarray(C['W'][j, kk])
                    C['rho'][j, kk] = P.rho(np.asarray(nl(c_, d)), np.asarray(nl(c_, st.data[('qp', 'LSPG')])))
                if 'qpp' in st.rules:
                    qq = jax.block_until_ready(P.run_query(bank, st, 'qpp', 'LSPG', sp['scheme'], DT0, 'prod', u0, nu, record=False))
                    C['W_qpp'][j] = np.asarray(qq['W'])
                    row.update(dist_qp_qpp=P.fdist(fq_, P.restrict(qq['fields'], L), n0), qpp_valid=P.run_valid(qq))
            rows.append(row)
        try:
            ma = q['evolve'].lower(W0[(lb, tci[0])][0], U0[(L, tci[0])][1], W0[(lb, tci[0])][1], d, sch).compile().memory_analysis()
            mrep = {k_: int(getattr(ma, k_)) for k_ in ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes',
                                                       'generated_code_size_in_bytes') if hasattr(ma, k_)}
        except Exception as e_:  # noqa: BLE001
            mrep = dict(error=str(e_))
        rep['subjects'][lb] = dict(spec=sp, m=st.m, M=st.M, rules=st.rules, mode_hash=st.mode_hash, live_bytes=st.live_bytes,
                                   compiled=mrep, mem_after=P.mem(), case_ids=[f'{cases[ci][0]}|{cases[ci][1]}' for ci in tci],
                                   certification=rows)
        cids = np.array([f'{cases[ci][0]}|{cases[ci][1]}' for ci in tci])
        V = {k: np.array([bool(r_.get(k, False)) for r_ in rows]) for k in ('valid', 'tight_valid', 'tighter_valid', 'qp_valid', 'qpp_valid')}
        np.savez_compressed(out / f"S_{lb.replace('|', '_')}.npz", case_ids=cids, dev_case_ids=cids,
                            q2_case_ids=cids if 'qp' in st.rules else np.array([], str), valid=V['valid'],
                            valid_tight=V['tight_valid'], valid_tighter=V['tighter_valid'], valid_qp=V['qp_valid'],
                            valid_qpp=V['qpp_valid'], time_tag='t0,t1,t2,t3,t4,t5', scheme=sp['scheme'], dt=DT0,
                            kx=np.asarray(st.kx), ky=np.asarray(st.ky), A=np.asarray(st.base['A']), lam=np.asarray(st.base['lam']), **C)
        print('SUBJECT', lb, el(), flush=True)
        save()

    A = cfg['A']
    la = label(A)

    ARGS = {}

    def evolve_args(sp, ci):
        """Prebuilt (function, arguments) of one evolution: nothing but the dispatch is inside a timed interval."""
        lb = label(sp)
        if (lb, ci) not in ARGS:
            st = setting(sp)
            w0, scale = W0[(lb, ci)]
            ARGS[(lb, ci)] = (st.q['LSPG']['evolve'], (w0, U0[(int(sp['mesh']), ci)][1], scale, st.data[('main', 'LSPG')],
                                                       st.sched('LSPG', sp['scheme'], DT0, 'prod')))
        return ARGS[(lb, ci)]

    def run_evolve(sp, ci):
        fn, args = evolve_args(sp, ci)
        return fn(*args)

    def timed_A(ci):
        fn, args = evolve_args(A, ci)
        QS.burn(burn)
        t, o = sync_time(lambda: fn(*args))
        return t, ident(*o)

    rng = np.random.default_rng(int(cfg['seed']))
    order = [(j, ci, r_) for j in range(len(cfg['subjects'])) for ci in tci for r_ in range(int(cfg['reps']))]
    rng.shuffle(order)
    for j, ci, r_ in order:
        sp = cfg['subjects'][j]
        lb = label(sp)
        st = setting(sp)
        L = int(sp['mesh'])
        q = st.q['LSPG']
        d = st.data[('main', 'LSPG')]
        sch = st.sched('LSPG', sp['scheme'], DT0, 'prod')
        u0, nu, _ = U0[(L, ci)]
        cold = st.cold
        evolve_args(A, ci)
        tries = []
        for attempt in range(3):
            ta1, ha1 = timed_A(ci)
            QS.burn(burn)
            ti, (w0, scale) = sync_time(lambda: q['init'](u0, cold))
            QS.burn(burn)
            te, (W, s_) = sync_time(lambda: q['evolve'](w0, nu, scale, d, sch))
            QS.burn(burn)
            td, F = sync_time(lambda: q['decode'](W, d))
            ta2, ha2 = timed_A(ci)
            drift = ta2 / ta1
            tries.append(dict(B=lb, case=f'{cases[ci][0]}|{cases[ci][1]}', rep=r_, attempt=attempt, tA1=ta1, tA2=ta2, t_init=ti,
                              it=np.asarray(s_['step_it'])[:NREC - 1].tolist(),
                              t_evolve=te, t_decode=td, drift=drift, ratio=te / (.5 * (ta1 + ta2)),
                              hashes_ok=bool(ha1 == ref[(la, ci)] and ha2 == ref[(la, ci)] and ident(W, s_) == ref[(lb, ci)])))
            if BAND[0] <= drift <= BAND[1]:
                break
        all_ok = all(t_['hashes_ok'] for t_ in tries)
        x = dict(tries[-1], valid=bool(BAND[0] <= tries[-1]['drift'] <= BAND[1] and all_ok), retries=len(tries) - 1,
                 drifts=[t_['drift'] for t_ in tries], all_hashes_ok=all_ok,
                 drift_ok=bool(BAND[0] <= tries[-1]['drift'] <= BAND[1]), attempts=tries)
        rep['invocations'].append(x)
        ta1, hf1 = timed_A(ci)
        QS.burn(burn)
        tf, fo = sync_time(lambda: q['fused'](u0, nu, d, cold, sch))
        ta2, hf2 = timed_A(ci)
        hok = bool(ident(fo['W'], fo['stats']) == ref[(lb, ci)] and hf1 == ref[(la, ci)] and hf2 == ref[(la, ci)])
        rep['fused'].append(dict(B=lb, case=x['case'], rep=r_, tA1=ta1, tA2=ta2, t_fused=tf, ratio=tf / (.5 * (ta1 + ta2)),
                                 it=np.asarray(fo['stats']['step_it'])[:NREC - 1].tolist(),
                                 drift=ta2 / ta1, hash_ok=hok, valid=bool(hok and BAND[0] <= ta2 / ta1 <= BAND[1])))
        if len(rep['invocations']) % 24 == 0:
            save()
            print('TRIPLETS', len(rep['invocations']), el(), flush=True)
    save()
    print('TIMING DONE', el(), flush=True)

    for sp in cfg.get('doubled', []):                    # AM-8
        lb = label(sp)
        for ci in P.select(cases, sp.get('cases', cfg['cases'])):
            fn, args = evolve_args(sp, ci)
            for r_ in range(int(cfg['reps'])):
                atts = []
                for attempt in range(3):                  # same drift retry rule as the triplets (<= 2 retries)
                    QS.burn(burn)
                    t1, o1 = sync_time(lambda: fn(*args))
                    QS.burn(burn)
                    t0 = time.perf_counter()
                    oa = fn(*args)
                    jax.block_until_ready(oa)
                    ob = fn(*args)
                    jax.block_until_ready(ob)
                    tb = time.perf_counter() - t0
                    QS.burn(burn)
                    t2, o2 = sync_time(lambda: fn(*args))
                    atts.append(dict(t_single1=t1, t_double=tb, t_single2=t2, drift=t2 / t1,
                                     it=[np.asarray(o_[1]['step_it'])[:NREC - 1].tolist() for o_ in (o1, oa, ob, o2)],
                                     identity_ok=bool(all(ident(*o_) == ref[(lb, ci)] for o_ in (o1, oa, ob, o2))),
                                     same_iterations=bool(np.array_equal(np.asarray(oa[1]['step_it']), np.asarray(ob[1]['step_it'])))))
                    if BAND[0] <= t2 / t1 <= BAND[1]:
                        break
                rep['doubled'].append(dict(subject=lb, case=f'{cases[ci][0]}|{cases[ci][1]}', rep=r_, t_single1=t1, t_double=tb,
                                           t_single2=t2, ratio=tb / (.5 * (t1 + t2)), drift=t2 / t1, attempt=attempt,
                                           drift_ok=bool(BAND[0] <= t2 / t1 <= BAND[1]),
                                           it=[np.asarray(o_[1]['step_it'])[:NREC - 1].tolist() for o_ in (o1, oa, ob, o2)],
                                           same_iterations=all(x_['same_iterations'] for x_ in atts),
                                           identity_ok=all(x_['identity_ok'] for x_ in atts), attempts=atts))
    save()

    for sp in cfg.get('components', []):                 # A1.2 / A2.2 diagnostic
        lb = label(sp)
        st = setting(sp)
        ci = P.select(cases, [sp['case']])[0]
        res = PC.component_times(st, sp['scheme'], U0[(int(sp['mesh']), ci)][1], W0[(lb, ci)][0], W0[(lb, ci)][1],
                                 reps=int(sp.get('reps', 30)), burn=burn)
        ev = [x['t_evolve'] for x in rep['invocations'] if x['B'] == lb and x['case'] == f'{cases[ci][0]}|{cases[ci][1]}' and x['valid']]
        Ss = np.load(out / f"S_{lb.replace('|', '_')}.npz")
        it = Ss['it'][list(Ss['case_ids']).index(f'{cases[ci][0]}|{cases[ci][1]}')]
        nstart, b1 = T2.SCHEMES[sp['scheme']][5], T2.SCHEMES[sp['scheme']][4]
        hist = sum(res['hist_be'] if (k < nstart or b1 == 0) else res['hist_cn'] for k in range(NREC - 1))
        model = hist + (NREC - 1) * (res['pred'] + res['init']) + float(np.sum(it)) * res['iter']
        t_ev = float(np.median(ev)) if ev else float('nan')
        rep['components'].append(dict(subject=lb, case=f'{cases[ci][0]}|{cases[ci][1]}', components=res, iterations=it.tolist(),
                                      model_seconds=model, evolve_seconds=t_ev, ratio=model / t_ev,
                                      note='diagnostic only (A2.2); loop bookkeeping not accounted'))
    save()

    for sp in cfg.get('recording_overhead', []):          # A3.2 probe
        lb = label(sp)
        st = setting(sp)
        q0 = st.q0['LSPG']
        d = st.data[('main', 'LSPG')]
        sch = st.sched('LSPG', sp['scheme'], DT0, 'prod')
        for ci in P.select(cases, sp['cases']):
            w0, scale = W0[(lb, ci)]
            nu = U0[(int(sp['mesh']), ci)][1]
            jax.block_until_ready(q0['evolve'](w0, nu, scale, d, sch))
            fn, args = evolve_args(sp, ci)
            for r_ in range(int(cfg['reps'])):
                QS.burn(burn)
                t1, _ = sync_time(lambda: fn(*args))
                QS.burn(burn)
                tb, ob = sync_time(lambda: q0['evolve'](w0, nu, scale, d, sch))
                QS.burn(burn)
                t2, _ = sync_time(lambda: fn(*args))
                Wr = np.asarray(run_evolve(sp, ci)[0])
                rep['recording_overhead'].append(dict(subject=lb, case=f'{cases[ci][0]}|{cases[ci][1]}', rep=r_, t_rec1=t1,
                                                      t_norec=tb, t_rec2=t2, ratio_norec_over_rec=tb / (.5 * (t1 + t2)),
                                                      W_rel=float(np.max(np.abs(np.asarray(ob[0]) - Wr)) / np.max(np.abs(Wr)))))
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('P1TIME COMPLETE', el(), flush=True)


if __name__ == '__main__':
    main()
