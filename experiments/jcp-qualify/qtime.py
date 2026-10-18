"""jcp-qualify per-mesh timing job (B5; DESIGN 6, A1.2), generalising P1's p1time.py to subjects of any form (LSPG, GAL),
scheme and step factor. One job, one GPU: the mesh's bank plus the 1024^2 bank of the reference A (P1's A: r = 384,
1024^2, LSPG-CN-R, dt0, frozen list, gauss128).

  1  per subject (mesh, r, rule, form, scheme, factor) and dev6 case: an untimed scoring run through the split query
     (init -> evolve -> decode) of the RECORDING program (nrec = S + 1). Its trajectory is certified in-job (LSPG gamma /
     GAL eta, autodiff, G-cert, validity); tight / tighter / successor / check replays and own-state rho give the
     timing-job Q1/Q2 label (P1 A2.3, A3.2). Split vs fused equivalence on the first case.
  2  A-B-A triplets in randomised order: burn, A evolve | burn, B init | burn, B evolve | burn, B decode | burn, A evolve;
     drift t_A2/t_A1 outside [1/1.1, 1.1] -> repeated (<= 2 retries), else invalid; then a fused triplet (A | B fused | A).
     Every timed evolution must reproduce its scoring run's identity hash (W, every internal state, per-step exit reasons
     and iteration counts).
  3  doubled work (P1 AM-8) for the listed subjects.

    python qtime.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import p1common as P
from p1common import Q, QS, T2, DT0, sha
import qacc as QA
import qcert as QC
import jax
import jax.numpy as jnp

BAND = (1 / 1.1, 1.1)


def sync_time(fn):
    t = time.perf_counter()
    o = fn()
    jax.block_until_ready(o)
    return time.perf_counter() - t, o


def ident(W, st_):
    return sha(np.concatenate([np.asarray(W).ravel(), np.asarray(st_['states']).ravel(),
                               np.asarray(st_['step_reason'], float), np.asarray(st_['step_it'], float)]))


def label(sp):
    return f"{sp['mesh']}|R{sp['Rp']}|{sp['rule']}|{sp['form']}|{sp['scheme']}|{float(sp['factor']):g}"


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
               banks={}, subjects={}, invocations=[], fused=[], doubled=[], equivalence=[], memory={}, complete=False)
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

    def setting(sp, with_roles=True):
        k = (int(sp['mesh']), int(sp['Rp']), sp['rule'])
        if k not in settings:
            b0 = P.mem()['in_use']
            roles = {kk: sp[kk] for kk in ('qp', 'qpp') if sp.get(kk) and with_roles}
            settings[k] = P.Setting(banks[k[0]], f'{k}', k[1], k[2], 'frozen', roles=roles, forms=('LSPG', 'GAL'), nrec=0)
            settings[k].live_bytes = P.mem()['in_use'] - b0
        st_ = settings[k]
        for kk in ('qp', 'qpp'):
            if not with_roles:
                continue
            if sp.get(kk) and kk not in st_.rules:
                b0 = P.mem()['in_use']
                st_.add_role(banks[k[0]], kk, sp[kk], forms=('LSPG', 'GAL'))
                st_.live_bytes += P.mem()['in_use'] - b0
            elif sp.get(kk):
                assert st_.rules[kk] == sp[kk], (kk, st_.rules[kk], sp[kk])
        return st_

    def parts(sp, with_roles=True):
        st = setting(sp, with_roles)
        dt = DT0 * float(sp['factor'])
        S = QA.steps_of(dt)
        q = QA.QQ.get(st, sp['form'], S + 1, nstep=max(T2.NSTEP, S + 1))
        g, t_ = P.tol_of(sp['form'], 'prod')
        return st, dt, S, q, st.data[('main', sp['form'])], T2.sched(sp['scheme'], dt, g, t_), t_

    subs = [cfg['A']] + cfg['subjects']
    W0, ref = {}, {}
    for sp in subs:
        lb = label(sp)
        if lb in rep['subjects']:
            continue
        st, dt, S, q, d, sch, tolf = parts(sp)
        L = int(sp['mesh'])
        bank = banks[L]
        form = sp['form']
        nl = q['nl']
        cert = T2.make_certifier(nl, st.Rp) if form == 'LSPG' else QC.make_gal_certifier(nl, st.Rp)
        n = len(tci)
        C = dict(W=np.full((n, 6, st.Rp), np.nan), w0=np.full((n, st.Rp), np.nan), states=np.full((n, S + 1, st.Rp), np.nan),
                 cert=np.full((n, S), np.nan), rec=np.full((n, S), np.nan), raw=np.full((n, S), np.nan),
                 scale=np.full((n, S), np.nan), tolratio=np.full((n, S), np.nan),
                 it=np.full((n, S), -1), reason=np.full((n, S), -1), n0=np.full(n, np.nan),
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
            if lb in cfg.get('inject_identity_fault', []):     # smoke control (P1 A4.1)
                ref[(lb, ci)] = 'injected-' + ref[(lb, ci)]
            f = P.restrict(F, L)
            o = cert(s_['states'], nu, d, sch)
            c_ = np.asarray(o[0])
            if form == 'LSPG':
                rec = np.asarray(s_['step_gn'])[:S]
                gok = P.gcert_ok(c_, rec)
                nonst = int(np.sum(~(c_ <= QA.GAMMA)))
            else:
                tr = np.asarray(s_['step_tolratio'])[:S]
                rec = QC.recorded_eta(tr, np.asarray(o[4]), tolf)
                C['raw'][j], C['scale'][j], C['tolratio'][j] = np.asarray(o[3]), np.asarray(o[4]), tr
                gok = QC.gcert_gal(c_, np.asarray(o[3]), np.asarray(o[4]), tr, tolf)
                nonst = int(np.sum(~QC.certified_gal(c_)))
            assert np.asarray(s_['states']).shape[0] == S + 1
            C['W'][j], C['w0'][j], C['states'][j], C['cert'][j], C['rec'][j], C['n0'][j] = (
                np.asarray(W), np.asarray(w0), np.asarray(s_['states']), c_, rec, n0)
            C['it'][j], C['reason'][j] = (np.asarray(s_[k])[:S] for k in ('step_it', 'step_reason'))
            C['f257'][j] = f
            row = dict(case=f'{cases[ci][0]}|{cases[ci][1]}', valid=P.run_valid(dict(W=W, w0=w0, fields=F, stats=s_)),
                       budget_exits=int(np.asarray(s_['exits'])[0]), noncertified=nonst, gcert_ok=gok, ident=ref[(lb, ci)])
            if j == 0:
                fq = jax.block_until_ready(q['fused'](u0, nu, d, st.cold, sch))
                fr = float(jnp.max(jnp.abs(fq['fields'] - F)) / jnp.max(jnp.abs(fq['fields'])))
                same_it = bool(np.array_equal(np.asarray(fq['stats']['step_it']), np.asarray(s_['step_it'])))
                rep['equivalence'].append(dict(subject=lb, fields_rel=fr, same_iterations=same_it,
                                               passed=bool(fr <= 1e-12 and same_it)))
            for lev, key in (('tight', 'W_tight'), ('tighter', 'W_tighter')):
                w = jax.block_until_ready(QA.run(bank, st, 'main', form, sp['scheme'], dt, lev, u0, nu))
                C[key][j] = np.asarray(w['W'])
                row[f'{lev}_valid'] = P.run_valid(w)
                row[f'_f_{lev}'] = P.restrict(w['fields'], L)
            row['dist_tight'] = P.fdist(f, row['_f_tight'], n0)
            row['dist_tight_tighter'] = P.fdist(row.pop('_f_tight'), row.pop('_f_tighter'), n0)
            if 'qp' in st.rules:
                q_ = jax.block_until_ready(QA.run(bank, st, 'qp', form, sp['scheme'], dt, 'prod', u0, nu))
                C['W_qp'][j] = np.asarray(q_['W'])
                fq_ = P.restrict(q_['fields'], L)
                row.update(dist_qp=P.fdist(f, fq_, n0), qp_valid=P.run_valid(q_))
                for kk in range(6):
                    cv = jnp.asarray(C['W'][j, kk])
                    C['rho'][j, kk] = P.rho(np.asarray(nl(cv, st.data[('main', 'LSPG')])), np.asarray(nl(cv, st.data[('qp', 'LSPG')])))
                if 'qpp' in st.rules:
                    qq = jax.block_until_ready(QA.run(bank, st, 'qpp', form, sp['scheme'], dt, 'prod', u0, nu))
                    C['W_qpp'][j] = np.asarray(qq['W'])
                    row.update(dist_qp_qpp=P.fdist(fq_, P.restrict(qq['fields'], L), n0), qpp_valid=P.run_valid(qq))
            rows.append(row)
        rep['subjects'][lb] = dict(spec=sp, m=st.m, M=st.M, rules=st.rules, mode_hash=st.mode_hash, live_bytes=st.live_bytes,
                                   meta=st.meta, trust=st.trust, steps=S, dt=dt, mem_after=P.mem(),
                                   case_ids=[f'{cases[ci][0]}|{cases[ci][1]}' for ci in tci], certification=rows)
        cids = np.array([f'{cases[ci][0]}|{cases[ci][1]}' for ci in tci])
        V = {k: np.array([bool(r_.get(k, False)) for r_ in rows]) for k in ('valid', 'tight_valid', 'tighter_valid', 'qp_valid', 'qpp_valid')}
        np.savez_compressed(out / f"S_{lb.replace('|', '_')}.npz", case_ids=cids, dev_case_ids=cids,
                            q2_case_ids=cids if 'qp' in st.rules else np.array([], str), valid=V['valid'],
                            valid_tight=V['tight_valid'], valid_tighter=V['tighter_valid'], valid_qp=V['qp_valid'],
                            valid_qpp=V['qpp_valid'], time_tag='t0,t1,t2,t3,t4,t5', scheme=sp['scheme'], form=form,
                            factor=float(sp['factor']), dt=dt, steps=S, rule=sp['rule'], Rp=st.Rp, mesh=L, tol=np.array(P.tol_of(form, 'prod')),
                            kx=np.asarray(st.kx), ky=np.asarray(st.ky), A=np.asarray(st.base['A']), lam=np.asarray(st.base['lam']), **C)
        print('SUBJECT', lb, el(), flush=True)
        k_ = (int(sp['mesh']), int(sp['Rp']), sp['rule'])     # A2.4: drop this setting's Q2 tables as soon as no later
        rest = subs[subs.index(sp) + 1:]                      # subject of the same setting needs them
        if not any((int(x['mesh']), int(x['Rp']), x['rule']) == k_ and label(x) not in rep['subjects'] for x in rest):
            for kk in ('qp', 'qpp'):
                st.drop_role(kk)
                st.rules.pop(kk, None)
            import gc
            gc.collect()
            rep['memory'][f'after_certification|{k_}'] = P.mem()
        save()

    for st_ in settings.values():                      # A2.4: Q2 tables only during certification
        for kk in ('qp', 'qpp'):
            st_.drop_role(kk)
            st_.rules.pop(kk, None)
    import gc
    gc.collect()
    rep['memory']['after_certification'] = P.mem()
    A = cfg['A']
    la = label(A)
    ARGS = {}

    def evolve_args(sp, ci):
        lb = label(sp)
        if (lb, ci) not in ARGS:
            st, dt, S, q, d, sch, _ = parts(sp, False)
            w0, scale = W0[(lb, ci)]
            ARGS[(lb, ci)] = (q['evolve'], (w0, U0[(int(sp['mesh']), ci)][1], scale, d, sch), S)
        return ARGS[(lb, ci)]

    def timed_A(ci):
        fn, args, _ = evolve_args(A, ci)
        QS.burn(burn)
        t, o = sync_time(lambda: fn(*args))
        return t, ident(*o)

    rng = np.random.default_rng(int(cfg['seed']))
    order = [(j, ci, r_) for j in range(len(cfg['subjects'])) for ci in tci for r_ in range(int(cfg['reps']))]
    rng.shuffle(order)
    for j, ci, r_ in order:
        sp = cfg['subjects'][j]
        lb = label(sp)
        st, dt, S, q, d, sch, _ = parts(sp, False)
        L = int(sp['mesh'])
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
                              it=np.asarray(s_['step_it'])[:S].tolist(), t_evolve=te, t_decode=td, drift=drift,
                              ratio=te / (.5 * (ta1 + ta2)),
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
                                 it=np.asarray(fo['stats']['step_it'])[:S].tolist(), drift=ta2 / ta1, hash_ok=hok,
                                 valid=bool(hok and BAND[0] <= ta2 / ta1 <= BAND[1])))
        if len(rep['invocations']) % 24 == 0:
            save()
            print('TRIPLETS', len(rep['invocations']), el(), flush=True)
    save()
    print('TIMING DONE', el(), flush=True)

    for sp in cfg.get('doubled', []):                    # P1 AM-8
        lb = label(sp)
        for ci in P.select(cases, sp.get('cases', cfg['cases'])):
            fn, args, S = evolve_args(sp, ci)
            for r_ in range(int(cfg['reps'])):
                atts = []
                for attempt in range(3):
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
                                     identity_ok=bool(all(ident(*o_) == ref[(lb, ci)] for o_ in (o1, oa, ob, o2))),
                                     same_iterations=bool(np.array_equal(np.asarray(oa[1]['step_it']), np.asarray(ob[1]['step_it'])))))
                    if BAND[0] <= t2 / t1 <= BAND[1]:
                        break
                rep['doubled'].append(dict(subject=lb, case=f'{cases[ci][0]}|{cases[ci][1]}', rep=r_, t_single1=t1, t_double=tb,
                                           t_single2=t2, ratio=tb / (.5 * (t1 + t2)), drift=t2 / t1, attempt=attempt,
                                           drift_ok=bool(BAND[0] <= t2 / t1 <= BAND[1]),
                                           same_iterations=all(x_['same_iterations'] for x_ in atts),
                                           identity_ok=all(x_['identity_ok'] for x_ in atts), attempts=atts))
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('QTIME COMPLETE', el(), flush=True)


if __name__ == '__main__':
    main()
