"""jcp-speed3d s3run modes fom (f1), prof (p1), mesh (m1), rep (r1); DESIGN H3, H4, A1.5, A1.8, A1.9, A1.11, A1.13, A2.

Config blocks (all keyed by the mode name):
  fom   fd: [[fam, L, [dt...]], ...], ps: {Ns, dts}, calib_cases, save_cases, refined_ref (first-order 923801 file or null)
  prof  configs: [{label, setting, rule, spec}], mf: [labels], k5_rule, profile: [labels], batch: {labels, sizes},
        confirm: [[A_label, B_label], ...], conv: {label: rule}, reduced_adt: [labels]
  mesh  Ls, roms: [{label, setting, rule, spec}], same: [[fam, dt]], matched: [[rom_label, fom_key]], calib_small,
        calib_large, acc_cases
  rep   roms: [{label, setting, rule, spec, paths}], foms: [fom_key], confirm: [[A_label, B_label]]
"""
from __future__ import annotations

import gc
import math
import time

import numpy as np
import jax
import jax.numpy as jnp

import bigtables as BT
import c3common as C3
import fomrun as FR
import psjax
import qn3
import s3common as S
from s3ctx import COLD, COLD_SPEC


def inventory(cfg):
    m = cfg['mode']
    blk = cfg.get(m, {})
    if m == 'fom':
        return dict(fd_configs=sum(len(x[2]) for x in blk['fd']), ps_configs=len(blk['ps']['Ns']) * len(blk['ps']['dts']))
    if m == 'prof':
        return dict(configs=[c['label'] for c in blk['configs']], confirm=blk.get('confirm', []))
    if m == 'mesh':
        return dict(Ls=blk['Ls'], roms=[r['label'] for r in blk['roms']], same=blk['same'], matched=blk.get('matched', []))
    if m == 'rep':
        return dict(roms=[r['label'] for r in blk['roms']], foms=blk.get('foms', []))
    return {}


# ------------------------------------------------------------------------------------------------- helpers
def fd_input(cx, L, j, cache):
    k = (L, j)
    if k not in cache:
        t = time.perf_counter()
        cache[k] = jax.block_until_ready(jnp.asarray(C3.C.initial_interior(L + 1, cx.tab, j)))
        cx.rep.setdefault('input_prep_seconds', {}).setdefault(f'fd|L{L}', []).append(time.perf_counter() - t)
    return cache[k]


def ps_input(cx, N, j, cache):
    k = ('ps', N, j)
    if k not in cache:
        t = time.perf_counter()
        cache[k] = jax.block_until_ready(psjax.ic(cx.tab, j, N))
        cx.rep.setdefault('input_prep_seconds', {}).setdefault(f'ps|N{N}', []).append(time.perf_counter() - t)
    return cache[k]


def first_order_ref(cx, path):
    """The first-order 3D reference (513 nodes, BE dt0/4; PROVISIONAL, continuity), cohort 923801 only; the staged copy is
    at $REF1_PATH (sha256 checked)."""
    import os
    path = path or os.environ.get('REF1_PATH')
    if not path or cx.seed != 923801:
        return None
    assert C3.sha_file(path) == '36e84e3dbac2f5cf1b3350c311c1f16d17ecfcd4dc217ded836ba30613c6cc8b', path
    return np.load(path)


def score_fom(cx, F, j, R1):
    s = cx.score_fields(F, j)
    if R1 is not None:
        r = np.asarray(R1[f'c{j}'])
        Fn = np.asarray(F)
        e1 = [float(np.linalg.norm(Fn[k] - r[k]) / np.linalg.norm(r[0])) for k in range(6)]
        s['first_order'] = dict(e_t=e1, e=max(e1[1:]))
    return s


def fom_calibrate(cx, foms, fam, L, dt, ccases, inputs):
    """A1.9 / A2.5 tolerance qualification: level 0 must converge on every calibration case; the loosest level that
    converges everywhere and differs from level 0 by <= 1e-6 n0 at every output time is chosen; else unavailable."""
    out = dict(levels={})
    F0 = {}
    for lev in range(3):
        sp = ('fd', fam, L, dt, lev)
        rec = {}
        for j in ccases:
            u = fd_input(cx, L, j, inputs)
            t = time.perf_counter()
            F, info = foms.run(sp, u, float(cx.NU[j][0]))
            F = np.asarray(jax.block_until_ready(F))
            ok, det = foms.converged(sp, info, F)
            rec[j] = dict(converged=ok, seconds=time.perf_counter() - t, **det)
            if lev == 0:
                F0[j] = F
            else:
                n0 = float(np.linalg.norm(F0[j][0]))
                rec[j]['diff_to_level0'] = float(max(np.linalg.norm(F[k] - F0[j][k]) for k in range(6)) / n0)
        out['levels'][lev] = rec
    if not all(r['converged'] for r in out['levels'][0].values()):
        out['chosen'] = None
        out['status'] = 'unavailable (level 0 did not converge)'
        return out
    ok = [0] + [lev for lev in (1, 2) if all(r['converged'] and r['diff_to_level0'] <= 1e-6 for r in out['levels'][lev].values())]
    out['chosen'] = max(ok)
    out['status'] = 'qualified'
    return out


def fom_subject(cx, foms, sp, inputs, hashes, name=None):
    def run(ci):
        j = cx.cases[ci]
        u = ps_input(cx, sp[1], j, inputs) if sp[0] == 'ps' else fd_input(cx, sp[2], j, inputs)
        return foms.run(sp, u, float(cx.NU[j][0]))[0]
    return dict(name=name or FR.key(sp), run=run, expect=lambda ci: hashes.get((FR.key(sp), cx.cases[ci])))


def fom_accuracy(cx, foms, sp, ids, inputs, hashes, R1, save_cases=()):
    rec = {}
    fields = {}
    for j in ids:
        u = ps_input(cx, sp[1], j, inputs) if sp[0] == 'ps' else fd_input(cx, sp[2], j, inputs)
        t = time.perf_counter()
        F, info = foms.run(sp, u, float(cx.NU[j][0]))
        F = jax.block_until_ready(F)
        secs = time.perf_counter() - t
        Fn = np.asarray(F)
        ok, det = foms.converged(sp, info, Fn)
        hashes[(FR.key(sp), j)] = C3.sha(Fn)
        rec[j] = dict(converged=ok, seconds_first=secs, **det, **(score_fom(cx, Fn, j, R1) if np.isfinite(Fn).all() else {}))
        if j in save_cases:
            fields[j] = Fn
    for j, Fn in fields.items():
        cx.Wstore[f'fomfield|{FR.key(sp)}|{j}'] = Fn
    print('FOM', FR.key(sp), 'converged', all(r['converged'] for r in rec.values()),
          'median e', np.median([r.get('e', np.nan) for r in rec.values()]), cx.el(), flush=True)
    return rec


def fom_warm_times(cx, foms, sp, inputs, hashes, ncases, reps):
    """Screening cost of a FOM configuration (A1.9; audit code-1 #6): after one warm call per case, `reps` timed calls per
    case (burn-in before each), output hash checked against the accuracy phase. Same job, same GPU."""
    sub = fom_subject(cx, foms, sp, inputs, hashes)
    out = []
    for ci in range(ncases):
        jax.block_until_ready(sub['run'](ci))
    for _ in range(reps):
        for ci in range(ncases):
            C3.burn(cx.cfg['timing']['burn'])
            t = time.perf_counter()
            F = jax.block_until_ready(sub['run'](ci))
            out.append(dict(case=cx.cases[ci], seconds=time.perf_counter() - t, hash_ok=C3.sha(np.asarray(F)) == sub['expect'](ci)))
    return dict(times=out, median=float(np.median([x['seconds'] for x in out])), hashes_ok=all(x['hash_ok'] for x in out))


# ------------------------------------------------------------------------------------------------------ fom
def mode_fom(cx):
    blk = cx.cfg['fom']
    foms, inputs, hashes = FR.FOMs(), {}, {}
    R1 = first_order_ref(cx, blk.get('refined_ref'))
    res = cx.rep['fom'] = dict(fd={}, ps={})
    # G7 (PS port) and G6 pieces that need the GPU job are run in the smoke (test_s3fom.py)
    for fam, L, dts in blk['fd']:
        for dt in dts:
            cal = fom_calibrate(cx, foms, fam, L, dt, blk['calib_cases'], inputs)
            ent = dict(calibration=cal)
            if cal['chosen'] is not None:
                sp = ('fd', fam, L, dt, cal['chosen'])
                ent['key'] = FR.key(sp)
                ent['cases'] = fom_accuracy(cx, foms, sp, cx.cases, inputs, hashes, R1, blk.get('save_cases', ()))
                ent['warm'] = fom_warm_times(cx, foms, sp, inputs, hashes, cx.cfg['timing']['cases'], cx.cfg['timing']['reps'])
            res['fd'][f'{fam}|L{L}|{dt:.8g}'] = ent
            cx.save()
            cx.dump()
        for k in [k for k in inputs if k[0] == L]:
            inputs.pop(k)
        foms.fd.pop((L, FR.FAM[fam][0]), None)
        gc.collect()
    for N in blk['ps']['Ns']:
        for dt in blk['ps']['dts']:
            sp = ('ps', N, dt)
            res['ps'][FR.key(sp)] = dict(key=FR.key(sp), cases=fom_accuracy(cx, foms, sp, cx.cases, inputs, hashes, R1,
                                                                             blk.get('save_cases', ())))
            res['ps'][FR.key(sp)]['warm'] = fom_warm_times(cx, foms, sp, inputs, hashes, cx.cfg['timing']['cases'],
                                                           cx.cfg['timing']['reps'])
            cx.save()
            cx.dump()


# ----------------------------------------------------------------------------------------------------- prof
def _cfg_of(blk, label):
    c = next(x for x in blk['configs'] if x['label'] == label)
    return c['setting'], c['rule'], S.parse_key(c['spec'])


def mode_prof(cx):
    blk, tc = cx.cfg['prof'], cx.cfg['timing']
    cases = cx.cases
    P = cx.rep['prof'] = dict(mesh={}, mf={}, k5={}, profile={}, batch={}, conv={}, reduced_adt={})
    # (a) mesh path accuracy of every configuration (hashes for the panels; equals the grid job by construction)
    for c in blk['configs']:
        sn, rule, sp = _cfg_of(blk, c['label'])
        cx.rollouts(sn, rule, [sp], cases)
    # (b) mesh-free path + ADT initialised by the same projection; K5 (gl12 projection, rank / condition checked)
    for lab in blk.get('mf', []):
        sn, rule, sp = _cfg_of(blk, lab)
        st = cx.ST[sn]
        cnd = st.mf_cond(rule)
        if not (np.isfinite(cnd['cond']) and cnd['cond'] < 1e10 and cnd['m'] >= st.Rp):
            P['mf'][lab] = dict(invalid='mesh-free Gram rank / condition check failed (A1.8)', cond=cnd)
            continue
        r = cx.rollouts(sn, rule, [sp], cases, path='mf')
        ad = ('ad', sp[1], sp[2], 'tight')
        ra = cx.rollouts(sn, rule, [ad], cases, path='mf')
        P['mf'][lab] = dict(dist_to_adt=cx.dist_cases(sn, r[S.spec_key(sp)]['W'], ra[S.spec_key(ad)]['W'], cases).tolist(),
                            adt_valid=ra[S.spec_key(ad)]['valid'], cond=st.mf_cond(rule))
        k5 = blk.get('k5_rule', 'gl12')
        cond = st.mf_cond(k5)
        dk = dict(st.data(rule), Bw=st.data(k5)['Bw'], Lq=st.data(k5)['Lq'])
        ok_rank = bool(np.isfinite(cond['cond']) and cond['cond'] < 1e10 and cond['m'] >= st.Rp)
        P['k5'][lab] = dict(rule=k5, cond=cond, rank_ok=ok_rank)
        if ok_rank:
            cx.rollouts(sn, rule, [sp], cases, path='mf', data=dk, tag=f'{sn}|{rule}|mfproj_{k5}', in_rule=k5)
        cx.save()
        cx.dump()
    # (c) H3.1 profile (indicative): components timed against the fused query
    for lab in blk.get('profile', []):
        sn, rule, sp = _cfg_of(blk, lab)
        if sp[0] == 'ad':
            P['profile'][lab] = dict(skipped='ADT configurations are not profiled')
            continue
        st, d = cx.ST[sn], cx.ST[sn].data(rule)
        q = cx.q_of(sn, sp)
        memo = {}

        def subj(name, fn):
            def run(ci):
                return fn(cases[ci])

            def expect(ci):
                k = (name, ci)
                if k not in memo:
                    memo[k] = C3.sha(np.asarray(jax.block_until_ready(fn(cases[ci]))))
                return memo[k]
            return dict(name=name, run=run, expect=expect)
        w0s = {j: q.project(cx.U0[j], d) for j in cases[:tc['cases']]}
        ev = {j: q.evolve(w0s[j][0], cx.NU[j], w0s[j][1], d) for j in cases[:tc['cases']]}
        Wo = {j: ev[j]['W_all'][::q.keep] if 'W_all' in ev[j] else ev[j]['W'] for j in cases[:tc['cases']]}
        # stage inputs from the configuration's own trajectory (audit code-2 #5): mid step k, history (c_k, c_{k-1}), start
        # state = the linear predictor 2 c_k - c_{k-1}; ADT configurations have no internal trajectory and are not profiled
        steps_, _ = S.nsteps(sp[2])
        kmid = steps_ // 2
        Wall = {j: (ev[j]['W_all'] if 'W_all' in ev[j] else None) for j in cases[:tc['cases']]}
        cn = {j: Wall[j][kmid] for j in Wall}
        cm = {j: Wall[j][kmid - 1] for j in Wall}
        cp = {j: 2. * cn[j] - cm[j] for j in Wall}
        contract = jax.jit(lambda c, dd: C3.OM.contract_offmesh(dd, c))
        Ju = {j: contract(cp[j][0], d) for j in Wall}
        a0, a1, a2, b0, b1, _ns = qn3.SCHEMES[sp[1]]
        dtp = sp[2]

        def stage_S(nu, dd):
            return 1.0 / (a0 + dtp * b0 * nu[:, None] * dd['lam'][None, :])

        def nrm_chol(J, nu, dd):
            """The production QN refresh after the contraction: J~ = A + dt b0 S Ju, H = J~^T J~, chol(H + mu0 diag H)."""
            S_ = stage_S(nu, dd)[0]
            Jt = dd['A'] + (dtp * b0 * S_)[:, None] * J
            H = Jt.T @ Jt
            return jnp.linalg.cholesky(H + qn3.MU0 * jnp.diag(jnp.diag(H)))
        jt = jax.jit(nrm_chol)
        Lf = {j: jt(Ju[j], cx.NU[j], d) for j in Wall}

        def qn_iter(c, c_n, c_m, nu, L_, dd):
            """One production QN iteration at a real stage: exact residual, exact gradient (matrix-free), Cholesky solve,
            trust clip, the trial residual (the accepted-step path; backtracking adds trial residuals of the same cost)."""
            S_ = stage_S(nu, dd)
            lamA = lambda x: nu[:, None] * dd['lam'][None, :] * (x @ dd['A'].T)
            hist = a1 * (c_n @ dd['A'].T) + a2 * (c_m @ dd['A'].T) + dtp * b1 * (qn3.adv(c_n, dd) + lamA(c_n))
            res = lambda x: S_ * (a0 * (x @ dd['A'].T) + hist + dtp * b0 * (qn3.adv(x, dd) + lamA(x)))
            r = res(c)
            Bc, Dc = qn3.bd(c, dd)
            g = r @ dd['A'] + dtp * b0 * qn3.jut(Bc, Dc, S_ * r, dd)
            dlt = -jax.scipy.linalg.cho_solve((L_, True), g[0])
            dlt = dlt * jnp.minimum(1., st.trust / (jnp.linalg.norm(dlt) + 1e-300))
            return res(c + dlt[None])
        qi = jax.jit(qn_iter)
        kr = sp[3] if (sp[0] == 'qn' and sp[3] != 0) else (1 if sp[0] != 'qn' else math.inf)
        evq = qn3.make_qn_evolve(st.Rp, sp[1], sp[2], 1, st.trust, kr)
        qstep = jax.jit(lambda c_n, h, nu, dd: evq(c_n, nu, dd, hist=h, k0=kmid)['W_all'])
        floor = jax.jit(lambda x: x + 1.)
        subs = [subj('project', lambda j: q.project(cx.U0[j], d)[0]),
                subj('evolve', lambda j: q.evolve(w0s[j][0], cx.NU[j], w0s[j][1], d)['W']
                     if sp[0] == 'ad' else q.evolve(w0s[j][0], cx.NU[j], w0s[j][1], d)['W_all']),
                subj('decode', lambda j: q.decode(Wo[j], d)),
                subj('contraction', lambda j: contract(cp[j][0], d)),
                subj('normal+cholesky', lambda j: jt(Ju[j], cx.NU[j], d)),
                subj('qn_iteration', lambda j: qi(cp[j], cn[j], cm[j], cx.NU[j], Lf[j], d)),
                subj('qn_step_with_refresh', lambda j: qstep(cn[j], (cm[j], Wall[j][kmid - 2], Wall[j][kmid - 3]), cx.NU[j], d)),
                subj('dispatch_floor', lambda j: floor(jnp.float64(j)))]
        if sp[0] in ('fs', 'fsp'):
            steps, _ = S.nsteps(sp[2])
            a_eff = C3.fs3.a_effective(sp[1], sp[2], S.DT0, st.a, steps)
            evs = C3.fs3.make_fs_evolve(st.Rp, C3.OM.contract_offmesh, sp[1], sp[2], sp[3], 1e-3, st.trust, a_eff, a_eff)
            evsj = jax.jit(lambda w, v, t, dd: evs(w, v, t, dd)['W_all'])
            subs.append(subj('adaptive_start', lambda j: evsj(w0s[j][0][0], cx.NU[j][0], 1e-12 * w0s[j][1][0], d)))
        A = cx.rom_subject(sn, rule, sp)
        cx.time_panel(f'profile|{lab}', A, subs, int(tc['seed']) + 50, tc['cases'], tc['reps'])
        P['profile'][lab] = dict(note='components separately jitted (indicative, A1.8); qn_iteration and normal+cholesky use the '
                                      'configured scheme and step with the production arithmetic; G3 and the post-start split are '
                                      'computed in the analysis')
    # (d) H3.3 batching: latency and throughput per batch size; G4, K6 (wrong viscosity), K6+ (whole-query permutation)
    for lab in blk.get('batch', {}).get('labels', []):
        sn, rule, sp = _cfg_of(blk, lab)
        st, d = cx.ST[sn], cx.ST[sn].data(rule)
        q = cx.q_of(sn, sp)
        tag = f'{sn}|{rule}'
        single = cx.Wstore[f'{tag}|{S.spec_key(sp)}']
        rec = {}
        for B in blk['batch']['sizes']:
            ids = cases[:B]
            U = jnp.concatenate([cx.U0[j] for j in ids])
            NUb = jnp.concatenate([cx.NU[j] for j in ids])
            try:
                o = jax.block_until_ready(q(U, NUb, d))
                ts, hok = [], []
                h0 = C3.sha(np.asarray(o['W']))
                for _ in range(3):
                    C3.burn(tc['burn'])
                    t = time.perf_counter()
                    Wt = jax.block_until_ready(q(U, NUb, d)['W'])
                    ts.append(time.perf_counter() - t)
                    hok.append(C3.sha(np.asarray(Wt)) == h0)
                Wb = np.moveaxis(np.asarray(o['W']), 1, 0)
                g4 = cx.dist_cases(sn, Wb, single[:B], ids)[:, 1:].max(1)
                ow = np.moveaxis(np.asarray(q(U, jnp.roll(NUb, 1), d)['W']), 1, 0)
                k6 = cx.dist_cases(sn, ow, single[:B], ids)[:, 1:].max(1) if B > 1 else None
                perm = np.arange(B)[::-1]
                op = np.moveaxis(np.asarray(q(U[perm], NUb[perm], d)['W']), 1, 0)
                k6p = cx.dist_cases(sn, op, single[:B][perm], [ids[i] for i in perm])[:, 1:].max(1)
                outc = 'exact' if g4.max() <= 1e-8 else ('within tolerance' if g4.max() <= 2.5e-5 else 'fail')
                rec[B] = dict(seconds=ts, hashes_ok=all(hok), median_seconds=float(np.median(ts)), throughput=B / float(np.median(ts)),
                              G4=outc, G4_max=float(g4.max()), G4_count_above_1e8=int((g4 > 1e-8).sum()),
                              K6_max=None if k6 is None else float(k6.max()), K6_fires=None if k6 is None else bool(k6.max() > 2.5e-5),
                              K6plus_max=float(k6p.max()), K6plus_pass=bool(k6p.max() <= 2.5e-5))
            except Exception as e:  # noqa: BLE001  (out of memory etc.: recorded as infeasible)
                rec[B] = dict(infeasible=repr(e)[:400])
            print('BATCH', lab, B, rec[B].get('throughput'), rec[B].get('G4'), cx.el(), flush=True)
            gc.collect()
        P['batch'][lab] = rec
        cx.save()
    # (e) confirmation panels (A1.5): A = comparator, B = winner (one panel per pair; cases 0-7, 3 repetitions)
    for i, (la, lb) in enumerate(blk.get('confirm', [])):
        A = cx.rom_subject(*_cfg_of(blk, la), name=la)
        B = cx.rom_subject(*_cfg_of(blk, lb), name=lb)
        cx.time_panel(f'confirm|{la}|{lb}', A, [B], int(tc['seed']) + 100 + i, blk.get('confirm_cases', 8),
                      blk.get('confirm_reps', 3))
    # (f) A1.13 converged-rule sensitivity of finalists
    for lab, crule in blk.get('conv', {}).items():
        sn, rule, sp = _cfg_of(blk, lab)
        r = cx.rollouts(sn, crule, [sp], cases)
        base = cx.Wstore[f'{sn}|{rule}|{S.spec_key(sp)}']
        P['conv'][lab] = dict(rule=crule, delta_quad=cx.dist_cases(sn, base, r[S.spec_key(sp)]['W'], cases).tolist())
        cx.ST[sn].drop(crule)
        gc.collect()
        cx.save()
    # (g) A2.7(d) ADT tight vs tighter at finalists' reduced rules
    for lab in blk.get('reduced_adt', []):
        sn, rule, sp = _cfg_of(blk, lab)
        t1, t2 = ('ad', sp[1], sp[2], 'tight'), ('ad', sp[1], sp[2], 'tighter')
        ra = cx.rollouts(sn, rule, [t1, t2], cases[:8])
        P['reduced_adt'][lab] = dict(worst=float(cx.dist_cases(sn, ra[S.spec_key(t1)]['W'], ra[S.spec_key(t2)]['W'],
                                                               cases[:8])[:, 1:].max()))
    cx.dump()
    cx.save()


# ----------------------------------------------------------------------------------------------------- mesh
def mode_mesh(cx):
    blk, tc = cx.cfg['mesh'], cx.cfg['timing']
    cases = cx.cases
    M_ = cx.rep['mesh'] = dict(G5={}, per_L={})
    foms_all = {}
    # G5 / K7 for every ROM bank and for the registered population (M2_512, W256; A1.9) when loaded
    need = {(cx.ST[r['setting']].bank, cx.ST[r['setting']].Rp, r['setting']) for r in blk['roms']}
    need |= {(cx.ST[s].bank, cx.ST[s].Rp, s) for s in ('M2_512', 'W256') if s in cx.ST}
    for bank, Rp, sname in sorted(need):
        st = cx.ST[sname]
        k = f'{bank}|{Rp}'
        if k not in M_['G5']:
            M_['G5'][k] = BT.check65(st.bk['params'], st.bk['T'], st.Rp, st.tb, log=lambda s: print(s, flush=True))
            M_['G5'][k]['gate'] = bool(M_['G5'][k]['pass'] and M_['G5'][k]['K7_fires'])
            print('G5', k, M_['G5'][k], flush=True)
            cx.save()
    g5ok = all(v['gate'] for v in M_['G5'].values())
    M_['G5_all'] = g5ok
    if not g5ok:
        print('G5/K7 FAILED: no mesh n > 65 tables are used', flush=True)
    for L in blk['Ls']:
        n = L + 1
        E = M_['per_L'][L] = dict(rom={}, fom={})
        foms, inputs, hashes = FR.FOMs(), {}, {}
        datas = {}
        for r in blk['roms']:
            sn, rule, sp = r['setting'], r['rule'], S.parse_key(r['spec'])
            st = cx.ST[sn]
            if n == 65:
                d = st.data(rule)
                info = dict(M=st.M, seconds_A=None)
            else:
                if not g5ok:
                    continue
                X, w = cx.rules[rule]
                try:
                    d, info = BT.rom_data(n, st.bk['params'], st.bk['T'], st.Rp, X, w, st.lean, log=lambda s: print(s, flush=True))
                except Exception as e:  # noqa: BLE001
                    E['rom'][r['label']] = dict(infeasible=repr(e)[:400])
                    continue
            datas[r['label']] = d
            tag = f'{sn}|{rule}|mf|n{n}'
            rr = cx.rollouts(sn, rule, [sp], cases, path='mf', data=d, n=n, tag=tag)
            ad = ('ad', sp[1], sp[2], 'tight')
            ra = cx.rollouts(sn, rule, [ad], cases, path='mf', data=d, n=n, tag=tag)
            rt = cx.rollouts(sn, rule, [('ad', sp[1], sp[2], 'tighter')], cases[:8], path='mf', data=d, n=n, tag=tag)
            E['rom'][r['label']] = dict(tables=info, dist_to_adt=cx.dist_cases(sn, rr[S.spec_key(sp)]['W'], ra[S.spec_key(ad)]['W'],
                                                                               cases).tolist(),
                                        adt_valid=ra[S.spec_key(ad)]['valid'],
                                        adt_tighter_valid=rt[S.spec_key(('ad', sp[1], sp[2], 'tighter'))]['valid'],
                                        adt_tight_tighter=float(cx.dist_cases(sn, ra[S.spec_key(ad)]['W'][:8],
                                                                              rt[S.spec_key(('ad', sp[1], sp[2], 'tighter'))]['W'],
                                                                              cases[:8])[:, 1:].max()))
            cx.save()
            cx.dump()
        # FOMs at this mesh: same-mesh diagnostics (calibrated) and matched configurations
        R1 = first_order_ref(cx, blk.get('refined_ref'))
        subj = []
        for fam, dt in blk['same']:
            cc = blk['calib_small'] if L <= 128 else blk['calib_large']
            try:
                cal = fom_calibrate(cx, foms, fam, L, dt, cc, inputs)
            except Exception as e:  # noqa: BLE001
                E['fom'][f'{fam}|{dt:.8g}'] = dict(infeasible=repr(e)[:400])
                continue
            ent = E['fom'][f'{fam}|{dt:.8g}'] = dict(calibration=cal)
            if cal['chosen'] is None:
                continue
            sp = ('fd', fam, L, dt, cal['chosen'])
            ent['key'] = FR.key(sp)
            ent['cases'] = fom_accuracy(cx, foms, sp, cases[:blk['acc_cases']], inputs, hashes, R1)
            subj.append(sp)
            cx.save()
        for rl, fk in blk.get('matched', []):            # timed against the ROM at every mesh (audit code-2 #4)
            sp = FR.parse(fk)
            if fk not in E['fom']:
                E['fom'][fk] = dict(key=fk, matched_for=[rl], cases=fom_accuracy(cx, foms, sp, cases[:tc['cases']], inputs,
                                                                                  hashes, R1))
            else:
                E['fom'][fk].setdefault('matched_for', []).append(rl)
            if sp not in subj:
                subj.append(sp)
        # timing: A = each ROM configuration at this mesh, Bs = the FOMs above
        for i, r in enumerate(blk['roms']):
            if r['label'] not in datas:
                continue
            sn, rule, sp = r['setting'], r['rule'], S.parse_key(r['spec'])
            A = cx.rom_subject(sn, rule, sp, path='mf', data=datas[r['label']], n=n, tag=f'{sn}|{rule}|mf|n{n}',
                               name=f"{r['label']}|n{n}")
            Bs = [fom_subject(cx, foms, x, inputs, hashes) for x in subj]
            if not Bs:
                continue
            cx.time_panel(f'mesh|L{L}|{r["label"]}', A, Bs, int(tc['seed']) + 200 + 10 * L + i, tc['cases'], tc['reps'])
        E['peak_bytes'] = C3.peak_bytes()
        del datas, foms, inputs
        gc.collect()
        cx.save()
        cx.dump()


# ------------------------------------------------------------------------------------------------------ rep
def mode_rep(cx):
    blk, tc = cx.cfg['rep'], cx.cfg['timing']
    cases = cx.cases
    Rr = cx.rep['replication'] = dict(roms={}, foms={})
    for r in blk['roms']:
        sn, rule, sp = r['setting'], r['rule'], S.parse_key(r['spec'])
        for path in r.get('paths', ['mesh']):
            rr = cx.rollouts(sn, rule, [sp], cases, path=path)
            ad = ('ad', sp[1], sp[2], 'tight')
            ra = cx.rollouts(sn, rule, [ad], cases, path=path)
            Rr.setdefault('dist_to_adt', {})[f"{r['label']}|{path}"] = cx.dist_cases(
                sn, rr[S.spec_key(sp)]['W'], ra[S.spec_key(ad)]['W'], cases).tolist()
        Rr['roms'][r['label']] = dict(setting=sn, rule=rule, spec=r['spec'], paths=r.get('paths', ['mesh']))
    foms, inputs, hashes = FR.FOMs(), {}, {}
    for fk in blk.get('foms', []):
        sp = FR.parse(fk)
        Rr['foms'][fk] = fom_accuracy(cx, foms, sp, cases, inputs, hashes, None)
        cx.save()
    for i, (la, lb) in enumerate(blk.get('confirm', [])):
        ra = next(x for x in blk['roms'] if x['label'] == la)
        rb = next(x for x in blk['roms'] if x['label'] == lb)
        A = cx.rom_subject(ra['setting'], ra['rule'], S.parse_key(ra['spec']), name=la)
        B = cx.rom_subject(rb['setting'], rb['rule'], S.parse_key(rb['spec']), name=lb)
        cx.time_panel(f'rep-confirm|{la}|{lb}', A, [B], int(tc['seed']) + 300 + i, blk.get('confirm_cases', 8),
                      blk.get('confirm_reps', 3))
    cx.dump()
    cx.save()
