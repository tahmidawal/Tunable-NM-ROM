"""jcp-speed3d gates on real data at small size (local GB10 through jaxrun, or a cluster smoke):

  G0  s3common fs query == fs3.make_fs_query (Step 2's end-to-end query), bitwise W, on the given cases
  G1  fs3p with extra=False == fs3 bitwise (W_all and records); fs3p extra=True runs, finite
  G2  certify(fs run) == fs3's recorded gn on fixed steps, |d| <= 1e-9 g + 1e-12
  QN  QN-1/4/inf run finite; distance to ADT tight; certified non-stationary count; K2 (q0) and K3 (noJuT) values
  G4  batched QN (Bt = len(cases)) == single queries (lattice distance); K6 wrong-viscosity association must differ
  mf  mesh-free projection runs (rank / condition reported)

    python test_s3.py --setting M2_512 --rule gl24 --cases 0 5 --out checks/test_s3-local.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

import c3common as C3
import fs3
import qn3
import s3common as S

BANKS = {'M2_512': ('M2', 512, 3), 'W256': ('W1024', 256, 6), 'W512': ('W1024', 512, 6), 'W768': ('W1024', 768, 6),
         'W1024': ('W1024', 1024, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', default='M2_512')
    ap.add_argument('--rule', default='gl24')
    ap.add_argument('--cases', type=int, nargs='+', default=[0, 5])
    ap.add_argument('--out', required=True)
    ap.add_argument('--quick', action='store_true', help='G0-G2 only')
    a = ap.parse_args()
    t0 = time.perf_counter()
    import os
    rep = dict(backend=jax.default_backend(), device=jax.devices()[0].device_kind, setting=a.setting, rule=a.rule,
               cases=a.cases, checks={}, job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'))
    print(f'jax_backend={jax.default_backend()}', flush=True)
    bname, Rp, aa = BANKS[a.setting]
    bank = C3.load_bank(bname)
    tb = C3.Tables(65, bank, [Rp], log=lambda s: print(s, flush=True))
    rules = S.load_rules([a.rule, 'gl12'])
    st = S.Setting(dict(name=a.setting, bank=bname, Rp=Rp, a=aa), {bname: bank}, {bname: tb}, rules)
    d = st.data(a.rule)
    tab = C3.C.table(923801, 64)
    U0 = jnp.stack([jnp.asarray(C3.C.initial_interior(65, tab, j)) for j in a.cases])
    NU = jnp.asarray([float(tab['nu'][j]) for j in a.cases])
    n0 = np.linalg.norm(np.asarray(U0), axis=1)
    G = np.concatenate([np.asarray(g) for g in st.lean['GTb']], 0)          # (R', 63^3)
    chk = rep['checks']
    print('setup', round(time.perf_counter() - t0, 1), flush=True)

    # G0: s3common fs query vs Step 2's fs3.make_fs_query
    sp = ('fs', 'BE', .01, 1)
    fq = S.make_query(sp, st)
    q2 = fs3.make_fs_query(65, Rp, C3.OM.contract_offmesh, 'BE', .01, 1, 1e-3, st.trust, st.a, dt0=.01)
    eq = [bool(np.array_equal(np.asarray(q2(U0[i], NU[i], d)['W']), np.asarray(fq(U0[i:i + 1], NU[i:i + 1], d)['W'][:, 0])))
          for i in range(len(a.cases))]
    chk['G0_bitwise'] = eq
    print('G0', eq, flush=True)

    # G1: fs3p extra=False == fs3; extra=True finite
    for sch, dt in (('CN', .025), ('BDF2', .0125)):
        sa, sb = ('fs', sch, dt, 1), ('fsp', sch, dt, 1)
        fa = S.make_query(sa, st)
        ol = [fa(U0[i:i + 1], NU[i:i + 1], d) for i in range(len(a.cases))]
        oa = {k: jnp.concatenate([x[k] for x in ol], 1) for k in ('W_all', 'reason', 'it', 'rej', 'pred', 'gn')}
        import fs3p
        steps, keep = S.nsteps(dt)
        a_eff = fs3.a_effective(sch, dt, .01, st.a, steps)
        evp = fs3p.make_fs_evolve(Rp, C3.OM.contract_offmesh, sch, dt, 1, 1e-3, st.trust, a_eff, steps, extra=False)
        obl = []
        for i in range(len(a.cases)):
            w0, sc = S.project_mesh(U0[i:i + 1], d)
            obl.append(jax.jit(lambda w, v, t, dd: evp(w, v, t, dd))(w0[0], NU[i], 1e-12 * sc[0], d))
        ob = {k: jnp.stack([x[k] for x in obl]) for k in obl[0]}
        same = bool(np.array_equal(np.asarray(ob['W_all']), np.moveaxis(np.asarray(oa['W_all']), 1, 0)) and
                    all(np.array_equal(np.asarray(ob[k]), np.moveaxis(np.asarray(oa[k]), 1, 0))
                        for k in ('reason', 'it', 'rej', 'pred', 'gn')))
        oc = S.make_query(sb, st)(U0, NU, d)
        chk[f'G1_{sch}_{dt}'] = dict(bitwise=same, extra_finite=bool(np.isfinite(np.asarray(oc['W'])).all()),
                                     extra_pred_counts=np.bincount(np.asarray(oc['pred']).ravel(), minlength=5).tolist())
        print('G1', sch, dt, chk[f'G1_{sch}_{dt}'], flush=True)

        # G2: certification of the fs run vs its recorded gn on fixed steps
        cert = qn3.make_certify(sch, dt, steps)
        worst = 0.
        for i in range(len(a.cases)):
            gc = np.asarray(cert(oa['W_all'][:, i], NU[i], d))
            gr = np.asarray(oa['gn'][:, i])
            if not (np.isfinite(gc).all() and np.isfinite(gr).all()):
                worst = float('inf')
                continue
            dlt = np.abs(gc - gr)[a_eff:] - (1e-9 * np.abs(gr[a_eff:]) + 1e-12)
            worst = max(worst, float(dlt.max()))
        chk[f'G2_{sch}_{dt}'] = dict(worst_excess=worst, ok=bool(np.isfinite(worst) and worst <= 0))
        print('G2', sch, dt, worst, flush=True)

    if a.quick:
        finish(rep, a.out)
        return
    # QN arms + ADT + controls
    adt = {}
    for sch, dt in (('CN', .025), ('BE', .01)):
        oad = S.make_query(('ad', sch, dt, 'tight'), st)(U0, NU, d)
        adt[(sch, dt)] = np.moveaxis(np.asarray(oad['W']), 1, 0)
        chk[f'ADT_{sch}_{dt}_valid'] = [S.run_valid(('ad', sch, dt, 'tight'), oad, i) for i in range(len(a.cases))]
    steps, _ = S.nsteps(.025)
    cert = qn3.make_certify('CN', .025, steps)
    for sp in [('fs', 'CN', .025, 2), ('qn', 'CN', .025, 1), ('qn', 'CN', .025, 4), ('qn', 'CN', .025, 0),
               ('qn', 'CN', .025, 1, 'q0'), ('qn', 'CN', .025, 1, 'noJuT'), ('qn', 'BE', .01, 1, 'noJuT'), ('qn', 'BE', .01, 1)]:
        f = S.make_query(sp, st)
        o = f(U0, NU, d)
        jax.block_until_ready(o)
        t1 = time.perf_counter()
        jax.block_until_ready(f(U0, NU, d))
        tt = time.perf_counter() - t1
        W = np.moveaxis(np.asarray(o['W']), 1, 0)
        dist = S.lattice_dist(W, adt[(sp[1], sp[2])], G, n0)
        stp, _ = S.nsteps(sp[2])
        ce = qn3.make_certify(sp[1], sp[2], stp)
        nonst = [int(np.sum(~(np.asarray(ce(o['W_all'][:, i], NU[i], d)) <= 1e-3))) for i in range(len(a.cases))]
        rec = dict(dist=dist.tolist(), nonstationary=nonst, valid=[S.run_valid(sp, o, i) for i in range(len(a.cases))],
                   seconds_batch=tt)
        if sp[0] == 'qn':
            rec.update(n_ref=int(np.asarray(o['n_ref']).sum()), q=int(np.asarray(o['q']).sum()),
                       q_max_step=int(np.asarray(o['q']).max()), reasons=np.bincount(np.asarray(o['reason']).ravel(), minlength=5).tolist())
        chk[S.spec_key(sp)] = rec
        print(S.spec_key(sp), json.dumps(rec), flush=True)

    # G4 batching + K6
    sp = ('qn', 'CN', .025, 4)
    f = S.make_query(sp, st)
    ob = np.moveaxis(np.asarray(f(U0, NU, d)['W']), 1, 0)
    os_ = np.stack([np.asarray(f(U0[i:i + 1], NU[i:i + 1], d)['W'])[:, 0] for i in range(len(a.cases))])
    chk['G4_dist'] = S.lattice_dist(ob, os_, G, n0).tolist()
    ow = np.moveaxis(np.asarray(f(U0, jnp.roll(NU, 1), d)['W']), 1, 0)
    chk['K6_dist'] = S.lattice_dist(ow, os_, G, n0).tolist()
    print('G4', chk['G4_dist'], 'K6', chk['K6_dist'], flush=True)

    # mesh-free projection
    X, w = rules[a.rule]
    U0X = jnp.stack([jnp.asarray(C3.C.b3.blob_ic_3d(65, tab, j, coords=X)) for j in a.cases])
    om = S.make_query(('qn', 'CN', .025, 4), st, path='mf')(U0X, NU, d)
    Wm = np.moveaxis(np.asarray(om['W']), 1, 0)
    chk['mf'] = dict(cond=st.mf_cond(a.rule), dist_to_mesh=S.lattice_dist(Wm, os_, G, n0).tolist(),
                     w0_dist=(np.linalg.norm((np.asarray(om['w0']) - np.asarray(S.project_mesh(U0, d)[0])) @ G, axis=1) / n0).tolist())
    print('mf', chk['mf'], flush=True)
    rep['seconds'] = time.perf_counter() - t0
    finish(rep, a.out)


def finish(rep, out):
    """Mandatory gates (audit code-1 #11): G0, G1 bitwise, G2; and, in the full test, QN-1 CN 2.5 within the bar, K2 / K3
    fire, G4 <= 1e-8, K6 fires. Non-zero exit if any fails."""
    c = rep['checks']
    fails = [k for k in ('G0_bitwise',) if not all(c.get(k, [False]))]
    fails += [k for k, v in c.items() if k.startswith('G1_') and not (v['bitwise'] and v['extra_finite'])]
    fails += [k for k, v in c.items() if k.startswith('G2_') and not v['ok']]
    if 'qn|CN|0.025|k1' in c:
        if not max(c['qn|CN|0.025|k1']['dist']) <= 2.5e-4:
            fails.append('QN1 CN 2.5 distance')
        if not all(all(c.get(k, [False])) for k in ('ADT_CN_0.025_valid', 'ADT_BE_0.01_valid')):
            fails.append('ADT validity')
        k2 = c['qn|CN|0.025|k1|q0']                  # configuration-level ineligibility (DESIGN §5): E-a, E-b or E-c fails
        steps = 10 * len(k2['nonstationary'])
        if not (sum(k2['nonstationary']) > 0.01 * steps or max(k2['dist']) > 2.5e-4 or not all(k2['valid'])):
            fails.append('K2')
        if not max(c['qn|CN|0.025|k1|noJuT']['dist'] + c['qn|BE|0.01|k1|noJuT']['dist']) > 2.5e-4:
            fails.append('K3')
        if not max(c['G4_dist']) <= 2.5e-5:          # A1.8: exact (<= 1e-8) or within tolerance (<= 2.5e-5)
            fails.append('G4')
        if not max(c['K6_dist']) > 2.5e-5:
            fails.append('K6')
    rep['mandatory_failures'] = fails
    json.dump(rep, open(out, 'w'), indent=1, default=float)
    print('DONE', 'FAILURES' if fails else 'all mandatory gates pass', fails, flush=True)
    if fails:
        raise SystemExit(5)


if __name__ == '__main__':
    main()
