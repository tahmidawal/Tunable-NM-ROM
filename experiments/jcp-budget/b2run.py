"""jcp-budget 2D job driver (DESIGN 4, 5, 6, 7 with amendment A1): one mesh per job, one GPU.

Blocks, each optional in the config:
  b2        B2 corners per member (pairwise or full factorial), the shared ideal end u_star and its U_star replays per r,
            the converged-rule check replays (every Q-containing corner type and the u_star family, check cases),
            M is the sixth ingredient (A3.1); the LS-init records (G-B5)
  b7        long-horizon rollouts per member: prod (recorded + certified in-job), tight (all), tighter (check cases),
            q+ (q2 cases), q++ (check cases), rho on own states
  b8        fine-route scoring and physical diagnostics of the b7 prod runs against this lane's references (in-job;
            the fine reference fields never leave the cluster), controls C-A and C-E
  controls  C-Q, C-S, C-I, sham G-B1, certifier controls G-Q
  pilot     memory pilot: the combined path at this mesh (A1.8)

    python b2run.py --config configs/<attempt>.json --out <dir>
Outputs: result.json, B2_<member>.npz, B2CHK_<member>.npz, STAR_R<r>.npz, STARCHK_R<r>.npz, B7_<member>.npz, B8_<member>.npz,
CTRL_*.npz, COMPLETE.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import p1common as P
from p1common import Q, QS, DT0, sha
import budcore as BC
import budcommon as B
import qcert as QC

GAMMA = 1e-3


def mkey(m):
    return f"R{m['Rp']}|{m['form']}|{m['scheme']}|{float(m['factor']):g}"


def subsets(design):
    """A3.1: pairwise over {Q, T, F, S, I, M} (22 corners); full = the 2^5 factorial over {Q, T, F, S, I} plus the M
    main effect and its five pairs (38 corners)."""
    S1 = [frozenset()] + [frozenset([x]) for x in B.ING]
    S2 = [frozenset(c) for c in itertools.combinations(B.ING, 2)]
    if design == 'pairwise':
        return S1 + S2
    assert design == 'full'
    return S1 + S2 + [frozenset(c) for n in (3, 4, 5) for c in itertools.combinations(B.FULL, n)]


def skey(S):
    return ''.join(x for x in B.ING if x in S) or '0'


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
    L = int(cfg['mesh'])
    cases = P.case_list(cfg)
    rep = dict(config=cfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(BC.CKPT),
               rotation_sha256=QS.sha_file(Q.ROTATION), cohorts=P.cohort_hashes(), setup={}, b2={}, b7={}, b8={},
               controls={}, memory={}, elapsed={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    bank = P.Bank(L, cfg['edges'], nrb=cfg.get('bank_blocks'))
    rep['setup']['bank'] = dict(edges=bank.edges, seconds=bank.build_seconds, bytes=bank.bytes, mem=P.mem())
    s256 = L // 256
    class LazyIC(dict):
        """Mesh ICs built on first use only (code audit 1: never all 38 when a job uses a subset)."""
        def __missing__(self, i):
            u0n = Q.e.initial(L, cases[i][2])
            v = (jnp.asarray(u0n), float(cases[i][2][4]), float(np.linalg.norm(u0n[::s256, ::s256])), u0n)
            self[i] = v
            return v
    U0 = LazyIC()
    cid = lambda i: f'{cases[i][0]}|{cases[i][1]}'
    save()
    print('SETUP', el(), P.mem(), flush=True)
    members = cfg['members']
    byR = {}
    for m in members:
        byR.setdefault(int(m['Rp']), []).append(m)
    for Rp, mem_ in sorted(byR.items()):
        t_r = time.perf_counter()
        st = B.BSetting(bank, Rp)
        rules = {m['rule'] for m in mem_}
        assert len(rules) == 1, ('one deployed rule per r in FRONT-2D', rules)
        st.add_role('main', rules.pop())
        w0ls, lsrec = {}, {}
        LS = None
        if 'b2' in cfg or 'controls' in cfg or 'pilot' in cfg:
            lsi = P.select(cases, cfg.get('ls_cases', cfg['cases']))
            LS = B.LSInit(bank, Rp, [U0[i][3] for i in lsi])
            for k, i in enumerate(lsi):
                lsrec[cid(i)] = LS.recs[k]
                if LS.recs[k]['accepted']:          # A3.4: a rejected LS init makes the I ingredient unavailable
                    w0ls[i] = LS.C[k]
            rep['setup'][f'LS_R{Rp}'] = dict(cond=LS.cond, seconds=LS.seconds, worst_orth=max(r['orth'] for r in lsrec.values()),
                                             records=lsrec, G_B5=bool(all(r['accepted'] for r in lsrec.values())))
            np.savez_compressed(out / f'LS_R{Rp}.npz', case_ids=np.array([cid(i) for i in w0ls]),
                                C=np.array([w0ls[i] for i in w0ls]) if w0ls else np.zeros((0, Rp)))
        if 'pilot' in cfg:
            run_pilot(cfg, rep, bank, st, mem_, U0, w0ls, cases, out)
            save()
            del st
            gc.collect()
            continue
        if 'b2' in cfg:
            run_b2(cfg, rep, bank, st, Rp, mem_, U0, w0ls, cases, cid, out, save, el)
        if 'controls' in cfg and Rp == int(cfg['controls'].get('Rp', 384)):
            run_controls(cfg, rep, bank, st, U0, w0ls, cases, cid, out, save)
        if 'b7' in cfg:
            run_b7(cfg, rep, bank, st, Rp, mem_, U0, cases, cid, out, save, el)
        rep['memory'][f'R{Rp}'] = P.mem()
        rep['elapsed'][f'R{Rp}'] = time.perf_counter() - t_r
        save()
        del st, LS
        gc.collect()
    if 'b8' in cfg:
        run_b8(cfg, rep, bank, members, cases, cid, out, save)
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('B2RUN COMPLETE', el(), flush=True)


# ------------------------------------------------------------------------------------------------------- B2 ---

def run_b2(cfg, rep, bank, st, Rp, mem_, U0, w0ls, cases, cid, out, save, el):
    """B2 (DESIGN 4, A1, A2, A3): corners on the settings {4r: st, 2048: sM} (the M ingredient selects the setting)."""
    b = cfg['b2']
    ids = P.select(cases, cfg['cases'])
    chk = [i for i in P.select(cases, b.get('check_cases', 'dev6')) if i in ids]
    sM = B.BSetting(bank, Rp, M=B.M_SWITCH)
    sM.add_role('main', st.rules['main'])
    SET = {st.M: st, sM.M: sM}
    for s_ in SET.values():
        s_.add_role('qinf', b['qinf'])
    R = rep['b2'].setdefault(f'R{Rp}', dict(members={}, star=None))
    NaW = np.full((6, Rp), np.nan)

    def go(sp, i, anchor=None):
        """One corner run; a corner needing the LS init of a case whose LS init was rejected is unavailable (A3.4)."""
        if sp['init'] == 'ls' and i not in w0ls:
            return dict(W=NaW, valid=False, seconds=np.nan, nfail=-1, budget_exits=-1)
        u0, nu, n0, _ = U0[i]
        return B.run_corner(SET, sp, u0, nu, w0_ls=w0ls.get(i), anchor=anchor)

    # ---- the shared ideal end and its U_star replays (never deferred, A1.1)
    star_specs = {'star': ('tight', DT0 / 8), 'star16': ('tight', DT0 / 16), 'starx': ('tighter', DT0 / 8),
                  'star16x': ('tighter', DT0 / 16)}
    sspec = lambda role, lev, dt: dict(role=role, form='GAL', scheme='BDF2', dt=dt, level=lev, init='ls', M=4 * Rp)
    Ws = {k: np.full((len(ids), 6, Rp), np.nan) for k in star_specs}
    Vs = {k: np.zeros(len(ids), bool) for k in star_specs}
    for j, i in enumerate(ids):
        for k, (lev, dt) in star_specs.items():
            r = go(sspec('qinf', lev, dt), i)
            Ws[k][j], Vs[k][j] = r['W'], r['valid']
        print('STAR', Rp, cid(i), el(), flush=True)
    np.savez_compressed(out / f'STAR_R{Rp}.npz', case_ids=np.array([cid(i) for i in ids]),
                        **{f'W_{k}': v for k, v in Ws.items()}, **{f'valid_{k}': v for k, v in Vs.items()})
    R['star'] = dict(specs={k: list(v) for k, v in star_specs.items()}, valid={k: int(v.sum()) for k, v in Vs.items()})
    save()
    # ---- corners per member
    design = b.get('design', 'pairwise')
    Sl = subsets(design)
    qS = [s_ for s_, S in enumerate(Sl) if 'Q' in S]
    for m in mem_:
        t_m = time.perf_counter()
        mk = mkey(m)
        spec0 = B.corner_spec(m, frozenset())
        W = np.full((len(ids), len(Sl), 6, Rp), np.nan)
        V = np.zeros((len(ids), len(Sl)), bool)
        secs = np.full((len(ids), len(Sl)), np.nan)
        NF = np.full((len(ids), len(Sl)), -1)
        BX = np.full((len(ids), len(Sl)), -1)
        tuples = []
        for s_, S in enumerate(Sl):
            sp = B.corner_spec(m, S)
            B.assert_tuple(spec0, sp, S)                               # G-B6 (logical ingredients)
            tuples.append(dict(S=skey(S), **sp, rule=b['qinf'] if 'Q' in S else st.rules['main']))
        for j, i in enumerate(ids):
            for s_, S in enumerate(Sl):
                r = go(B.corner_spec(m, S), i)
                W[j, s_], V[j, s_], secs[j, s_] = r['W'], r['valid'], r['seconds']
                NF[j, s_], BX[j, s_] = r['nfail'], r['budget_exits']
            print('B2', mk, cid(i), el(), flush=True)
        np.savez_compressed(out / f"B2_{mk.replace('|', '_')}.npz", case_ids=np.array([cid(i) for i in ids]),
                            corners=np.array([skey(S) for S in Sl]), W=W, valid=V, seconds=secs, nfail=NF, budget_exits=BX)
        R['members'][mk] = dict(member=m, design=design, corners=[skey(S) for S in Sl], tuples=tuples,
                                valid=int(V.sum()), total=int(V.size), seconds=time.perf_counter() - t_m, mem=P.mem())
        save()
    # ---- converged-rule check (G-B4, A1.7, A3.5): every Q-containing corner type of every member and the u_star family,
    # on the check cases, with the check rule (built once per setting, then dropped: memory lifetime)
    for M_, s_ in SET.items():
        s_.drop_role('qinf')
    gc.collect()
    for M_, s_ in SET.items():
        s_.add_role('qcheck', b['qcheck'])
    for m in mem_:
        mk = mkey(m)
        Wc = np.full((len(chk), len(qS), 6, Rp), np.nan)
        Vc = np.zeros((len(chk), len(qS)), bool)
        for jj, i in enumerate(chk):
            for kk, s_ in enumerate(qS):
                r = go(dict(B.corner_spec(m, Sl[s_]), role='qcheck'), i, anchor=True)
                Wc[jj, kk], Vc[jj, kk] = r['W'], r['valid']
        np.savez_compressed(out / f"B2CHK_{mk.replace('|', '_')}.npz", case_ids=np.array([cid(i) for i in chk]),
                            corners=np.array([skey(Sl[s_]) for s_ in qS]), W=Wc, valid=Vc)
        R['members'][mk].update(check_valid=int(Vc.sum()), check_total=int(Vc.size))
    Wsc = {k: np.full((len(chk), 6, Rp), np.nan) for k in star_specs}
    Vsc = {k: np.zeros(len(chk), bool) for k in star_specs}
    for jj, i in enumerate(chk):
        for k, (lev, dt) in star_specs.items():
            r = go(sspec('qcheck', lev, dt), i, anchor=True)
            Wsc[k][jj], Vsc[k][jj] = r['W'], r['valid']
    np.savez_compressed(out / f'STARCHK_R{Rp}.npz', case_ids=np.array([cid(i) for i in chk]),
                        **{f'W_{k}': v for k, v in Wsc.items()}, **{f'valid_{k}': v for k, v in Vsc.items()})
    for M_, s_ in SET.items():
        s_.drop_role('qcheck')
    del sM, SET
    gc.collect()
    save()


# ------------------------------------------------------------------------------------------------------- B7 ---

def run_b7(cfg, rep, bank, st, Rp, mem_, U0, cases, cid, out, save, el):
    b = cfg['b7']
    T = float(b['horizon'])
    nout = B.nout_of(T)
    ids = P.select(cases, b.get('cases', cfg['cases']))
    q2 = [i for i in P.select(cases, b.get('q2_cases', 'sub16')) if i in ids]
    dev = [i for i in P.select(cases, b.get('tighter_cases', 'dev6')) if i in ids]
    roles = {m['qp'] for m in mem_}, {m['qpp'] for m in mem_}
    assert len(roles[0]) == len(roles[1]) == 1
    st.add_role('qp', roles[0].pop())
    st.add_role('qpp', roles[1].pop())
    nl = st.q('LSPG')['nl']
    certs = dict(LSPG=BC.make_certifier(nl, Rp), GAL=QC.make_gal_certifier(nl, Rp))
    for m in mem_:
        t_m = time.perf_counter()
        mk = mkey(m)
        form, sc, dt = m['form'], m['scheme'], DT0 * float(m['factor'])
        S = B.steps_keep(dt, T)
        sp = B.corner_spec(m, frozenset())
        g_p, t_p = P.tol_of(form, 'prod')
        sch = BC.sched(sc, dt, g_p, t_p, horizon=T)
        n = len(ids)
        R = dict(case_ids=np.array([cid(i) for i in ids]), W=np.full((n, nout, Rp), np.nan), cert=np.full((n, S), np.nan),
                 rec=np.full((n, S), np.nan), raw=np.full((n, S), np.nan), scale=np.full((n, S), np.nan),
                 tolratio=np.full((n, S), np.nan), it=np.full((n, S), -1), reason=np.full((n, S), -1),
                 valid=np.zeros(n, bool), gcert=np.zeros(n, bool), noncert=np.zeros(n, int), budget_exits=np.zeros(n, int),
                 W_tight=np.full((n, nout, Rp), np.nan), valid_tight=np.zeros(n, bool),
                 dev_case_ids=np.array([cid(i) for i in dev]), W_tighter=np.full((len(dev), nout, Rp), np.nan),
                 valid_tighter=np.zeros(len(dev), bool), q2_case_ids=np.array([cid(i) for i in q2]),
                 W_qp=np.full((len(q2), nout, Rp), np.nan), valid_qp=np.zeros(len(q2), bool), rho=np.full((len(q2), nout), np.nan),
                 W_qpp=np.full((len(dev), nout, Rp), np.nan), valid_qpp=np.zeros(len(dev), bool), secs=np.full(n, np.nan))
        for j, i in enumerate(ids):
            u0, nu, n0, _ = U0[i]
            r = B.run_corner(st, sp, u0, nu, nout=nout, nrec=S + 1, horizon=T)
            stt = r['stats']
            R['W'][j], R['valid'][j], R['secs'][j], R['budget_exits'][j] = r['W'], r['valid'], r['seconds'], r['budget_exits']
            R['it'][j], R['reason'][j] = (np.asarray(stt[k])[:S] for k in ('step_it', 'step_reason'))
            if form == 'LSPG':
                gam, _ = certs['LSPG'](stt['states'], nu, st.data[('main', 'LSPG')], sch)
                R['cert'][j] = np.asarray(gam)
                R['rec'][j] = np.asarray(stt['step_gn'])[:S]
                R['gcert'][j] = P.gcert_ok(R['cert'][j], R['rec'][j]) and np.isfinite(R['cert'][j]).all()
                R['noncert'][j] = int(np.sum(~(R['cert'][j] <= GAMMA)))
            else:
                eta, gam, jc, rn_, scl = certs['GAL'](stt['states'], nu, st.data[('main', 'GAL')], sch)
                R['cert'][j], R['raw'][j], R['scale'][j] = np.asarray(eta), np.asarray(rn_), np.asarray(scl)
                R['tolratio'][j] = np.asarray(stt['step_tolratio'])[:S]
                R['rec'][j] = QC.recorded_eta(R['tolratio'][j], R['scale'][j], t_p)
                R['gcert'][j] = QC.gcert_gal(R['cert'][j], R['raw'][j], R['scale'][j], R['tolratio'][j], t_p)
                R['noncert'][j] = int(np.sum(~QC.certified_gal(R['cert'][j])))
            del r, stt
            rt = B.run_corner(st, dict(sp, level='tight'), u0, nu, nout=nout, horizon=T)
            R['W_tight'][j], R['valid_tight'][j] = rt['W'], rt['valid']
            if i in dev:
                k = dev.index(i)
                rtt = B.run_corner(st, dict(sp, level='tighter'), u0, nu, nout=nout, horizon=T)
                R['W_tighter'][k], R['valid_tighter'][k] = rtt['W'], rtt['valid']
                rqq = B.run_corner(st, dict(sp, role='qpp'), u0, nu, nout=nout, horizon=T, anchor=True)
                R['W_qpp'][k], R['valid_qpp'][k] = rqq['W'], rqq['valid']
            if i in q2:
                k = q2.index(i)
                rq = B.run_corner(st, dict(sp, role='qp'), u0, nu, nout=nout, horizon=T, anchor=True)
                R['W_qp'][k], R['valid_qp'][k] = rq['W'], rq['valid']
                for kk in range(nout):
                    cv = jnp.asarray(R['W'][j, kk])
                    R['rho'][k, kk] = P.rho(np.asarray(nl(cv, st.data[('main', 'LSPG')])), np.asarray(nl(cv, st.data[('qp', 'LSPG')])))
            print('B7', mk, cid(i), el(), flush=True)
        R.update(horizon=T, nout=nout, steps=S, dt=dt, form=form, scheme=sc, rule=st.rules['main'], qp=st.rules['qp'],
                 qpp=st.rules['qpp'], Rp=Rp, mesh=bank.L, tol=np.array([g_p, t_p]))
        np.savez_compressed(out / f"B7_{mk.replace('|', '_')}.npz", **R)
        rep['b7'][mk] = dict(member=m, horizon=T, n=n, valid=int(R['valid'].sum()), noncert=int(R['noncert'].sum()),
                             steps=int(S * n), seconds=time.perf_counter() - t_m)
        save()
    st.drop_role('qp')
    st.drop_role('qpp')


# ------------------------------------------------------------------------------------------------------- B8 ---

def load_ref(cfg, coh, c):
    """This lane's reference of (cohort, case) at the B8 horizon: (meta, finest lattice field, fine field) or None.
    Every file is checked against the manifest in the config."""
    b = cfg['b8']
    stem = Path(b['refs_dir']) / '2d' / coh / f"T{float(b['horizon']):g}" / f'c{int(c):03d}'
    pj, pz, pf = Path(str(stem) + '.json'), Path(str(stem) + '.npz'), Path(str(stem) + '_fine.npz')
    if not (pj.exists() and pz.exists() and pf.exists()):
        return None
    if 'manifest' in b:
        man = b['manifest']
    elif 'manifest_file' in b:              # REFS-MANIFEST.sha256 of the refs/ data directory ('<sha>  ./2d/...')
        man = {}
        for line in (Path(b['refs_dir']) / b['manifest_file']).read_text().splitlines():
            if line.strip():
                h, f = line.split(None, 1)
                man[f.strip().lstrip('./')] = h
    else:                                   # smoke: the producing refrun job's own manifest (same job, earlier step)
        man = {m_['file']: m_['sha256'] for m_ in json.loads(Path(b['manifest_from']).read_text())['manifest']}
    for p in (pj, pz, pf):
        rel = str(p.relative_to(Path(b['refs_dir'])))
        assert man.get(rel) == B.file_sha(p), ('reference file not in the manifest or changed', rel)
    meta = json.loads(pj.read_text())
    T = float(b['horizon'])
    want = dict(dim=2, cohort=coh, case=int(c), kind='ST', horizon=T)
    got = {k: meta.get(k) for k in want}
    assert got == want, ('reference identity differs from the request', got, want)
    assert np.allclose(meta['times'], np.arange(int(round(T / .05)) + 1) * .05, rtol=0, atol=1e-12), 'reference time list'
    assert meta.get('fine_file') == pf.name and meta.get('fine_sha256') == B.file_sha(pf), 'fine file identity'
    assert meta.get('lattice') == 256 and meta.get('metric_id') == f'2|lattice=256|times=0:0.05:{T:g}|norm=fixed_t0|max_k>=1', \
        'reference lattice / metric'
    assert meta.get('cohort_sha256') == P.sha(Q.cohort(coh)), 'reference cohort hash'
    z, zf = np.load(pz), np.load(pf)
    fin = meta['final'][2]
    return meta, np.asarray(z[fin]), np.asarray(zf['fine_' + fin]), np.asarray(zf['t0_fine'])


def run_b8(cfg, rep, bank, members, cases, cid, out, save):
    b = cfg['b8']
    T = float(b['horizon'])
    nout = B.nout_of(T)
    fb = bank if bank.L == B.FINE_L else P.Bank(B.FINE_L, cfg['edges'])
    band, inter = B.band_mask()
    phi = jnp.asarray(B.alias_field())
    nphi = float(jnp.linalg.norm(phi))
    chk = set(P.select(cases, b.get('check_cases', 'dev6')))
    ids = P.select(cases, b.get('cases', cfg['cases']))
    refs = {}
    for i in ids:
        r = load_ref(cfg, *cases[i][:2])
        if r is not None:
            refs[i] = r
    rep['b8']['refs_found'] = [cid(i) for i in refs]
    FQ = {}
    s4 = B.FINE_L // 256
    for m in members:
        mk = mkey(m)
        f7 = out / f"B7_{mk.replace('|', '_')}.npz"
        if not f7.exists():
            continue
        z7 = np.load(f7)
        Rp = int(m['Rp'])
        G = Q.BK.prefix(fb.Grot, fb.edges, Rp)
        if Rp not in FQ:                                     # fine-lattice span basis (interior nodes) for the floor
            Gf = jnp.concatenate([B.prefix_blocks(rb, fb.edges, Rp) for rb in fb.Grot], axis=0)
            FQ[Rp] = jnp.linalg.qr(Gf, mode='reduced')[0]
            del Gf
        Qf = FQ[Rp]
        lsf = out / f'LS_R{Rp}.npz'
        LSC = {}
        if lsf.exists():
            zl = np.load(lsf)
            LSC = {str(c_): zl['C'][k] for k, c_ in enumerate(zl['case_ids'])}
        n = len(ids)
        K = lambda *s: np.full((n,) + s, np.nan)
        D = dict(case_ids=np.array([cid(i) for i in ids]), has_ref=np.zeros(n, bool), e_fine=K(nout), e_257=K(nout),
                 n0_fine=K(), n0_257=K(), E_rom=K(nout), D_rom=K(nout), E_ref=K(nout), D_ref=K(nout), umax_rom=K(nout),
                 umin_rom=K(nout), umax_ref=K(nout), umin_ref=K(nout), wall_max_rom=K(), band_err2=K(nout), all_err2=K(nout),
                 alias_e257=K(nout), alias_efine=K(nout), CE_zero=K(nout - 1), CE_pred=K(nout - 1), CE_meas=K(nout - 1),
                 e0_ic=K(), e0_ic_ls=K(), eP_fine=K(nout))
        b7ids = [str(x) for x in z7['case_ids']]
        for j, i in enumerate(ids):
            if cid(i) not in b7ids or i not in refs:
                continue
            Wr = z7['W'][b7ids.index(cid(i))]
            if not np.isfinite(Wr).all():
                continue
            meta, lat, Rf, t0f = refs[i]
            nu = float(cases[i][2][4])
            F = B.decode(G, Wr, B.FINE_L)
            Rf = jnp.asarray(Rf)
            E = F - Rf
            n0f = float(jnp.linalg.norm(Rf[0]))
            n0l = float(np.linalg.norm(lat[0]))
            D['has_ref'][j] = True
            D['n0_fine'][j], D['n0_257'][j] = n0f, n0l
            D['e_fine'][j] = np.asarray(jnp.linalg.norm(E.reshape(nout, -1), axis=1)) / n0f
            D['e_257'][j] = np.linalg.norm((np.asarray(F[:, ::s4, ::s4]) - lat).reshape(nout, -1), axis=1) / n0l
            D['E_rom'][j], D['D_rom'][j] = B.energy_diss(F, nu)
            D['E_ref'][j], D['D_ref'][j] = B.energy_diss(Rf, nu)
            D['umax_rom'][j], D['umin_rom'][j] = np.asarray(F.max(axis=(1, 2))), np.asarray(F.min(axis=(1, 2)))
            D['umax_ref'][j], D['umin_ref'][j] = np.asarray(Rf.max(axis=(1, 2))), np.asarray(Rf.min(axis=(1, 2)))
            D['wall_max_rom'][j] = float(jnp.max(jnp.abs(jnp.concatenate([F[:, 0, :], F[:, -1, :], F[:, :, 0], F[:, :, -1]], axis=1))))
            D['band_err2'][j] = np.asarray(jnp.sum(jnp.where(jnp.asarray(band), E ** 2, 0.), axis=(1, 2)))
            D['all_err2'][j] = np.asarray(jnp.sum(jnp.where(jnp.asarray(inter), E ** 2, 0.), axis=(1, 2)))
            D['e0_ic'][j] = float(jnp.linalg.norm(E[0])) / n0f
            RfI = Rf[:, 1:-1, 1:-1].reshape(nout, -1)
            D['eP_fine'][j] = np.asarray(jnp.linalg.norm(RfI - (RfI @ Qf) @ Qf.T, axis=1)) / n0f     # fine floor per time
            if cid(i) in LSC:                                   # IC projection: LS init decoded on the fine route (6.4)
                Fl = B.decode(G, LSC[cid(i)][None], B.FINE_L)
                D['e0_ic_ls'][j] = float(jnp.linalg.norm(Fl[0] - Rf[0])) / n0f
            if i in chk:                                         # C-A (A1.6): guaranteed alias, e_257 unchanged
                ef = np.asarray(jnp.linalg.norm(E.reshape(nout, -1), axis=1))
                for k in range(1, nout):
                    w_ = 'W1' if k <= 5 else ('W2' if k <= 10 else 'W3')
                    eps_l = (meta['judge_windows'] or {}).get(w_, {}).get('eps_ref') or 0.
                    eps_f = (meta['judge_fine_windows'] or {}).get(w_, {}).get('eps_ref') or 0.
                    # A2.2 C-A: amplitude against the actual rejection threshold (guaranteed 'inconsistent')
                    ak = (ef[k] + n0f * (1.2 * D['e_257'][j][k] + 2 * (eps_f + eps_l) + 1e-6)) / nphi
                    Fa = F[k] + ak * phi
                    D['alias_efine'][j, k] = float(jnp.linalg.norm(Fa - Rf[k])) / n0f
                    D['alias_e257'][j, k] = float(np.linalg.norm(np.asarray(Fa[::s4, ::s4]) - lat[k])) / n0l
                # C-E (A1.6): ROM := reference -> zero defect; ROM := scaled reference -> closed-form defect
                Er, Dr = D['E_ref'][j], D['D_ref'][j]
                Bref = B.balance(Er, Dr)
                D['CE_zero'][j] = B.balance(Er, Dr) - Bref
                fac = 1 + .01 * np.arange(nout)
                Es, Ds = B.energy_diss(Rf * jnp.asarray(fac)[:, None, None], nu)
                D['CE_meas'][j] = B.balance(Es, Ds) - Bref
                g = fac ** 2 - 1
                D['CE_pred'][j] = g[1:] * Er[1:] - g[:-1] * Er[:-1] + .025 * (g[1:] * Dr[1:] + g[:-1] * Dr[:-1])
        np.savez_compressed(out / f"B8_{mk.replace('|', '_')}.npz", **D, horizon=T, fine_L=B.FINE_L)
        rep['b8'][mk] = dict(n_scored=int(D['has_ref'].sum()), fine_route='decode at the 1025^2 nodes (Bank(1024))')
        save()


# ------------------------------------------------------------------------------------------------- controls ---

def run_controls(cfg, rep, bank, st, U0, w0ls, cases, cid, out, save):
    c = cfg['controls']
    ids = P.select(cases, c.get('cases', 'dev6'))
    res = {}
    # every control run keeps its validity (code audit 1: comparators must be valid / converged anchors)
    VAL = {}

    def rc(name, spec, i, **kw):
        u0, nu, n0, _ = U0[i]
        r = B.run_corner(st, spec, u0, nu, **kw)
        VAL.setdefault(name, []).append(bool(r['valid']))
        return r['W']

    # C-Q: fixture R384 LSPG-BE dt0 with the deployed rule replaced by an under-resolved rule; I_Q = u(qinf) - u(ctrl)
    st.add_role('ctrlq', c.get('under_resolved', 'gauss32'))
    st.add_role('qinf', cfg['b2']['qinf'])
    fx = dict(Rp=st.Rp, form='LSPG', scheme='BE', factor=1)
    Wq = {k: [] for k in ('ctrl', 'qinf')}
    for i in ids:
        Wq['ctrl'].append(rc('Q_ctrl', dict(B.corner_spec(fx, frozenset()), role='ctrlq'), i))
        Wq['qinf'].append(rc('Q_qinf', B.corner_spec(fx, frozenset('Q')), i))
    st.drop_role('ctrlq')
    # C-S: R384 LSPG-CNR dt0 with LM budget 0 vs the tight corner (a converged anchor)
    fs = dict(Rp=st.Rp, form='LSPG', scheme='CNR', factor=1)
    Ws = {k: [] for k in ('b0', 'tight', 'prod', 'prod2')}
    Wi = {k: [] for k in ('pert', 'pert_ls')}
    sham = []
    st.add_role('sham', st.rules['main'])
    for i in ids:
        Ws['b0'].append(rc('S_b0', B.corner_spec(fs, frozenset()), i, budget=0))
        Ws['tight'].append(rc('S_tight', B.corner_spec(fs, frozenset('S')), i))
        Ws['prod'].append(rc('S_prod', B.corner_spec(fs, frozenset()), i))
        Ws['prod2'].append(rc('S_prod2', B.corner_spec(fs, frozenset()), i))                   # G-B1 prod -> prod
        sham.append(rc('sham', dict(B.corner_spec(fs, frozenset()), role='sham'), i))           # G-B1 q -> q (rebuilt)
        # C-I: perturbed deployed init (x 1.01) and its LS-switched corner
        Wi['pert'].append(rc('I_pert', B.corner_spec(fs, frozenset()), i, w0_scale=1.01))
        Wi['pert_ls'].append(rc('I_pert_ls', B.corner_spec(fs, frozenset('I')), i, w0_ls=w0ls[i]))
    st.drop_role('sham')
    st.drop_role('qinf')
    np.savez_compressed(out / 'CTRL_b2.npz', case_ids=np.array([cid(i) for i in ids]),
                        **{f'Q_{k}': np.array(v) for k, v in Wq.items()}, **{f'S_{k}': np.array(v) for k, v in Ws.items()},
                        **{f'I_{k}': np.array(v) for k, v in Wi.items()}, sham=np.array(sham),
                        w0_ls=np.array([w0ls[i] for i in ids]), **{f'valid_{k}': np.array(v) for k, v in VAL.items()})
    res['b2_controls'] = dict(cases=[cid(i) for i in ids], under_resolved=c.get('under_resolved', 'gauss32'),
                              sham_rule_bitwise=bool(np.array_equal(np.array(sham), np.array(Ws['prod']))),
                              prod_rerun_bitwise=bool(np.array_equal(np.array(Ws['prod2']), np.array(Ws['prod']))))
    # G-Q: certifier controls (premature exit must be detected), LSPG and GAL, and the wrong-table guard
    nl = st.q('LSPG')['nl']
    GQE = {}
    for form, scheme in (('LSPG', 'CNR'), ('GAL', 'CN')):
        S = B.steps_keep(DT0, .25)
        g_, t_ = P.tol_of(form, 'prod')
        sch = BC.sched(scheme, DT0, g_, t_)
        cert = BC.make_certifier(nl, st.Rp) if form == 'LSPG' else QC.make_gal_certifier(nl, st.Rp)
        nc, tot = 0, 0
        CE = []
        for i in ids[:2]:
            u0, nu, n0, _ = U0[i]
            r = B.run_corner(st, dict(role='main', form=form, scheme=scheme, dt=DT0, level='prod', init='deployed'), u0, nu,
                             nrec=S + 1, budget=0)
            o = np.asarray(cert(r['stats']['states'], nu, st.data[('main', form)], sch)[0])
            CE.append(o)
            nc += int(np.sum(~(o <= (GAMMA if form == 'LSPG' else QC.ETA_BAR))))
            tot += len(o)
        GQE[form] = np.array(CE)                                  # the certificate evidence, recomputed offline (code audit 3)
        res[f'GQ_budget0_{form}'] = dict(noncertified=nc, steps=tot, fraction=nc / tot, fired=bool(nc / tot > .01))
    saved = st.data[('main', 'LSPG')]
    st.data[('main', 'LSPG')] = dict(saved, lam=saved['lam'] * (1 + 1e-6))
    try:
        st.check('main')
        res['GQ_wrong_table'] = dict(raised=False)
    except ValueError as e_:
        res['GQ_wrong_table'] = dict(raised=True, msg=str(e_))
    st.data[('main', 'LSPG')] = saved
    np.savez_compressed(out / 'CTRL_gq.npz', cert_LSPG=GQE['LSPG'], cert_GAL=GQE['GAL'], gamma_bar=GAMMA, eta_bar=QC.ETA_BAR,
                        case_ids=np.array([cid(i) for i in ids[:2]]),
                        wrong_table_raised=bool(res['GQ_wrong_table']['raised']))
    rep['controls'] = res
    save()


# --------------------------------------------------------------------------------------------------- pilot ---

def run_pilot(cfg, rep, bank, st, mem_, U0, w0ls, cases, out):
    """A1.8 memory pilot: every table of the combined path resident at once, one corner run, peak recorded."""
    p = cfg['pilot']
    i = P.select(cases, p.get('cases', 'dev2'))[0]
    u0, nu, n0, _ = U0[i]
    rec = dict(start=P.mem())
    st.add_role('qinf', p['qinf'])
    rec['qinf'] = P.mem()
    st.add_role('qcheck', p['qcheck'])
    rec['qcheck'] = P.mem()
    sM = B.BSetting(bank, st.Rp, M=P.MMAX)
    for role, rl in (('main', st.rules['main']), ('qinf', p['qinf']), ('qcheck', p['qcheck'])):
        sM.add_role(role, rl)                                  # the production M-table path (code audit 1)
    rec['m2048'] = P.mem()
    fb = bank if bank.L == B.FINE_L else P.Bank(B.FINE_L, cfg['edges'])
    rec['fine_bank'] = P.mem()
    m = mem_[0]
    r = B.run_corner(st, B.corner_spec(m, frozenset('QTSI')), u0, nu, w0_ls=w0ls[i])
    r2 = B.run_corner(st, dict(B.corner_spec(m, frozenset('QS')), role='qcheck'), u0, nu, w0_ls=w0ls[i])
    r3 = B.run_corner({st.M: st, sM.M: sM}, dict(B.corner_spec(m, frozenset('QSM')), role='qcheck'), u0, nu, w0_ls=w0ls[i])
    F = B.decode(Q.BK.prefix(fb.Grot, fb.edges, st.Rp), r['W'], B.FINE_L)
    jax.block_until_ready(F)
    rec['after_runs'] = P.mem()
    rec['valid'] = [r['valid'], r2['valid'], r3['valid']]
    rep.setdefault('pilot', {})[f'R{st.Rp}'] = rec
    del sM, fb, F


if __name__ == '__main__':
    main()
