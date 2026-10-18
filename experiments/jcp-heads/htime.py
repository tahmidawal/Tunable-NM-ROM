"""jcp-heads per-mesh timing job (DESIGN 6, A1.8, A2.4): qualify's qtime.py protocol (copied as base/qtime.py) with
subjects that are heads or linear rungs (harm arms), LSPG only. One job, one GPU: the mesh's bank plus the 1024^2 bank
of the reference A (P1's A: linear r = 384, 1024^2, LSPG-CN-R, dt0, frozen list, gauss128).

  1  per subject and certification case (dev6): an untimed scoring run of the RECORDING program, certified in-job
     (autodiff gamma, G-cert, validity incl. the head initial-fit veto), tight / tighter anchors (verified, A1.3),
     q+ / q++ replays and own-state rho -> the timing-job Q1/Q2 label; split vs fused equivalence on the first case.
     Every other timed case (val32 at 1024^2) gets an untimed prod run for its identity hash.
  2  A-B-A triplets in randomised order (seeded): burn, A evolve | burn, B init | burn, B evolve | burn, B decode | burn,
     A evolve; drift t_A2/t_A1 outside [1/1.1, 1.1] -> repeated (<= 2 retries), else invalid; then a fused triplet
     (A | B fused | A). Every timed evolution reproduces its scoring run's identity hash (W, every internal state, per-step
     exit reasons and iteration counts). Repeats per case: cfg['reps'][cohort].
  3  doubled work (P1 AM-8) for the listed subjects.

    python htime.py --config configs/<attempt>.json --out <dir>
config: {"meshes": {"1024": {"edges": [...]}, ...}, "A": {...}, "subjects": [{"mesh", "name", "kind", "head" | "Rp",
         "M"?, "rule", "qp", "qpp", "scheme"}], "cases": "all38", "cert_cases": "dev6", "reps": {"dev6": 3, "val32": 1},
         "burn", "seed", "doubled": [...], "inject_identity_fault": [...]}
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import p1common as P
from p1common import Q, QS, T2, DT0, sha
import harm as HA
import hacc as HAC
import jax
import jax.numpy as jnp

BAND = (1 / 1.1, 1.1)


def sync_time(fn):
    t = time.perf_counter()
    o = fn()
    jax.block_until_ready(o)
    return time.perf_counter() - t, o


def ident(W, st_):
    return sha(np.concatenate([np.asarray(W).ravel(), np.asarray(st_['states']).ravel() if 'states' in st_ else np.zeros(0),
                               np.asarray(st_['step_reason'], float), np.asarray(st_['step_it'], float)]))


def label(sp):
    return f"{sp['mesh']}|{sp['name']}|{sp['rule']}|{sp['scheme']}"


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
    cci = [i for i in P.select(cases, cfg.get('cert_cases', 'dev6')) if i in tci]
    cid = lambda ci: f'{cases[ci][0]}|{cases[ci][1]}'
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
    S = HA.steps_of(DT0)
    heads, arms = {}, {}

    def arm_of(sp):
        k = (int(sp['mesh']), sp['name'], sp['rule'])
        if k not in arms:
            b0 = P.mem()['in_use']
            arms[k] = HA.make_arm(banks[k[0]], dict(sp), heads, S)
            arms[k].live_bytes = P.mem()['in_use'] - b0
        return arms[k]

    subs = [cfg['A']] + list(cfg['subjects'])
    W0, ref = {}, {}
    for sp in subs:
        lb = label(sp)
        if lb in rep['subjects']:
            continue
        arm = arm_of(sp)
        L = int(sp['mesh'])
        scheme = sp['scheme']
        sch, sch_t, sch_tt = (HA.sched(scheme, DT0, lev) for lev in ('prod', 'tight', 'tighter'))
        for role in ('qp', 'qpp'):
            if sp.get(role) and role not in arm.rules:
                arm.add_role(role, sp[role])
        q = arm.parts('prod', True)
        d = arm.data('main')
        n, dim = len(cci), arm.dim
        C = dict(W=np.full((n, 6, dim), np.nan), w0=np.full((n, dim), np.nan), states=np.full((n, S + 1, dim), np.nan),
                 cert=np.full((n, S), np.nan), rec=np.full((n, S), np.nan), it=np.full((n, S), -1),
                 reason=np.full((n, S), -1), n0=np.full(n, np.nan), ic=np.full((n, 5), np.nan),
                 W_tight=np.full((n, 6, dim), np.nan), cert_tight=np.full((n, S), np.nan),
                 W_tighter=np.full((n, 6, dim), np.nan), cert_tighter=np.full((n, S), np.nan),
                 W_qp=np.full((n, 6, dim), np.nan), W_qpp=np.full((n, 6, dim), np.nan), rho=np.full((n, 6), np.nan),
                 f257=np.full((n, 6, 257, 257), np.nan), C=np.full((n, 6, 512 if arm.kind == 'head' else dim), np.nan))
        rows = []
        for ci in tci:
            arm.check('main')
            u0, nu, n0 = U0[(L, ci)]
            o_init = jax.block_until_ready(q['init'](u0, arm.cold))
            w0, scale = o_init[0], o_init[1]
            W, s_ = jax.block_until_ready(q['evolve'](w0, nu, scale, d, sch))
            F = jax.block_until_ready(q['decode'](W, d))
            W0[(lb, ci)] = (w0, scale)
            ref[(lb, ci)] = ident(W, s_)
            if lb in cfg.get('inject_identity_fault', []):
                ref[(lb, ci)] = 'injected-' + ref[(lb, ci)]
            if ci not in cci:
                continue
            j = cci.index(ci)
            f = P.restrict(F, L)
            v = dict(W=W, w0=w0, fields=F, stats=dict(s_, **(o_init[2] if len(o_init) > 2 else {})))
            c_, rec, gok, nonst = HAC.certify(arm, v, nu, 'main', sch, S)
            C['W'][j], C['w0'][j], C['states'][j], C['cert'][j], C['rec'][j], C['n0'][j] = (
                np.asarray(W), np.asarray(w0), np.asarray(s_['states']), c_, rec, n0)
            C['it'][j], C['reason'][j] = (np.asarray(s_[k])[:S] for k in ('step_it', 'step_reason'))
            C['f257'][j], C['C'][j] = f, arm.coef(W)
            C['ic'][j] = [HAC.ic_info(v)[k] for k in ('ic_reason', 'ic_it', 'ic_gn', 'ic_rel', 'ic_idx')]
            row = dict(case=cid(ci), valid=HA.run_valid(v), budget_exits=int(np.asarray(s_['exits'])[0]), noncertified=nonst,
                       gcert_ok=gok, ident=ref[(lb, ci)], **HAC.ic_info(v))
            if j == 0:
                fq = jax.block_until_ready(q['fused'](u0, nu, d, arm.cold, sch))
                fr = float(jnp.max(jnp.abs(fq['fields'] - F)) / jnp.max(jnp.abs(fq['fields'])))
                same_it = bool(np.array_equal(np.asarray(fq['stats']['step_it']), np.asarray(s_['step_it'])))
                rep['equivalence'].append(dict(subject=lb, fields_rel=fr, same_iterations=same_it,
                                               passed=bool(fr <= 1e-12 and same_it)))
            ft = {}
            for lev, key, bar, sc_ in (('tight', 'tight', HAC.ANCHOR_T, sch_t), ('tighter', 'tighter', HAC.ANCHOR_TT, sch_tt)):
                w = jax.block_until_ready(arm.run('main', scheme, lev, u0, nu, DT0, record=True))
                ct, _, _, _ = HAC.certify(arm, w, nu, 'main', sc_, S)
                C[f'W_{key}'][j], C[f'cert_{key}'][j] = np.asarray(w['W']), ct
                row[f'{lev}_valid'] = HA.run_valid(w)
                row[f'{lev}_verified'] = HAC.anchor_verified(w, ct, bar)
                ft[lev] = P.restrict(w['fields'], L)
            row['dist_tight'] = P.fdist(f, ft['tight'], n0)
            row['dist_tight_tighter'] = P.fdist(ft['tight'], ft['tighter'], n0)
            if 'qp' in arm.rules:
                q_ = jax.block_until_ready(arm.run('qp', scheme, 'prod', u0, nu, DT0, record=False))
                C['W_qp'][j] = np.asarray(q_['W'])
                fq_ = P.restrict(q_['fields'], L)
                row.update(dist_qp=P.fdist(f, fq_, n0), qp_valid=HA.run_valid(q_))
                for kk in range(6):
                    C['rho'][j, kk] = P.rho(arm.nl(C['C'][j, kk], 'main'), arm.nl(C['C'][j, kk], 'qp'))
                if 'qpp' in arm.rules:
                    qq = jax.block_until_ready(arm.run('qpp', scheme, 'prod', u0, nu, DT0, record=False))
                    C['W_qpp'][j] = np.asarray(qq['W'])
                    row.update(dist_qp_qpp=P.fdist(fq_, P.restrict(qq['fields'], L), n0), qpp_valid=HA.run_valid(qq))
            rows.append(row)
        rep['subjects'][lb] = dict(spec=sp, describe=arm.describe(), live_bytes=arm.live_bytes, steps=S, dt=DT0,
                                   mem_after=P.mem(), cert_case_ids=[cid(ci) for ci in cci],
                                   timed_case_ids=[cid(ci) for ci in tci], certification=rows)
        np.savez_compressed(out / f"S_{lb.replace('|', '_')}.npz", case_ids=np.array([cid(ci) for ci in cci]),
                            time_tag='t0,t1,t2,t3,t4,t5', scheme=scheme, form='LSPG', dt=DT0, steps=S, rule=sp['rule'],
                            kind=arm.kind, dim=dim, M=arm.M, mesh=L, tol=np.array(P.tol_of('LSPG', 'prod')),
                            A=np.asarray(d['A']), lam=np.asarray(d['lam']), **C)
        for role in ('qp', 'qpp'):                            # Q2 tables only during certification
            if role in arm.rules:
                arm.drop_role(role)
        gc.collect()
        print('SUBJECT', lb, el(), flush=True)
        save()
    rep['memory']['after_certification'] = P.mem()
    A = cfg['A']
    la = label(A)
    ARGS = {}

    def evolve_args(sp, ci):
        lb = label(sp)
        if (lb, ci) not in ARGS:
            arm = arm_of(sp)
            q = arm.parts('prod', True)
            w0, scale = W0[(lb, ci)]
            ARGS[(lb, ci)] = (q['evolve'], (w0, U0[(int(sp['mesh']), ci)][1], scale, arm.data('main'),
                                            HA.sched(sp['scheme'], DT0, 'prod')))
        return ARGS[(lb, ci)]

    def timed_A(ci):
        fn, args = evolve_args(A, ci)
        QS.burn(burn)
        t, o = sync_time(lambda: fn(*args))
        return t, ident(*o)

    reps = cfg['reps']
    rng = np.random.default_rng(int(cfg['seed']))
    order = [(j, ci, r_) for j in range(len(cfg['subjects'])) for ci in tci for r_ in range(int(reps[cases[ci][0]]))]
    rng.shuffle(order)
    for j, ci, r_ in order:
        sp = cfg['subjects'][j]
        lb = label(sp)
        arm = arm_of(sp)
        q = arm.parts('prod', True)
        d = arm.data('main')
        sch = HA.sched(sp['scheme'], DT0, 'prod')
        L = int(sp['mesh'])
        u0, nu, _ = U0[(L, ci)]
        cold = arm.cold
        evolve_args(A, ci)
        tries = []
        for attempt in range(3):
            ta1, ha1 = timed_A(ci)
            QS.burn(burn)
            ti, o_init = sync_time(lambda: q['init'](u0, cold))
            w0, scale = o_init[0], o_init[1]
            QS.burn(burn)
            te, (W, s_) = sync_time(lambda: q['evolve'](w0, nu, scale, d, sch))
            QS.burn(burn)
            td, F = sync_time(lambda: q['decode'](W, d))
            ta2, ha2 = timed_A(ci)
            drift = ta2 / ta1
            tries.append(dict(B=lb, case=cid(ci), rep=r_, attempt=attempt, tA1=ta1, tA2=ta2, t_init=ti,
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
        ftries = []
        for attempt in range(3):                       # DESIGN A3.2: fused triplets get the split triplets' retry rule
            ta1, hf1 = timed_A(ci)
            QS.burn(burn)
            tf, fo = sync_time(lambda: q['fused'](u0, nu, d, cold, sch))
            ta2, hf2 = timed_A(ci)
            hok = bool(ident(fo['W'], fo['stats']) == ref[(lb, ci)] and hf1 == ref[(la, ci)] and hf2 == ref[(la, ci)])
            ftries.append(dict(tA1=ta1, tA2=ta2, t_fused=tf, drift=ta2 / ta1, hash_ok=hok))
            if BAND[0] <= ta2 / ta1 <= BAND[1]:
                break
        fall = all(t_['hash_ok'] for t_ in ftries)
        rep['fused'].append(dict(B=lb, case=cid(ci), rep=r_, tA1=ta1, tA2=ta2, t_fused=tf, ratio=tf / (.5 * (ta1 + ta2)),
                                 it=np.asarray(fo['stats']['step_it'])[:S].tolist(), drift=ta2 / ta1, hash_ok=hok,
                                 all_hashes_ok=fall, retries=len(ftries) - 1, attempts=ftries,
                                 valid=bool(fall and BAND[0] <= ta2 / ta1 <= BAND[1])))
        if len(rep['invocations']) % 24 == 0:
            save()
            print('TRIPLETS', len(rep['invocations']), el(), flush=True)
    save()
    print('TIMING DONE', el(), flush=True)

    for sp in cfg.get('doubled', []):                    # P1 AM-8
        lb = label(sp)
        for ci in P.select(cases, sp.get('cases', 'dev6')):
            fn, args = evolve_args(sp, ci)
            for r_ in range(3):
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
                rep['doubled'].append(dict(subject=lb, case=cid(ci), rep=r_, t_single1=t1, t_double=tb, t_single2=t2,
                                           ratio=tb / (.5 * (t1 + t2)), drift=t2 / t1, attempt=attempt,
                                           drift_ok=bool(BAND[0] <= t2 / t1 <= BAND[1]),
                                           same_iterations=all(x_['same_iterations'] for x_ in atts),
                                           identity_ok=all(x_['identity_ok'] for x_ in atts), attempts=atts))
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('HTIME COMPLETE', el(), flush=True)


if __name__ == '__main__':
    main()
