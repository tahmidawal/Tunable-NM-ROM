"""jcp-robust 2D rollout / qualification driver (DESIGN 2, 3.2; A1.2-A1.4, A2.1-A2.4, A3.2, A3.3, A3.6).

Derived from jcp-qualify's qacc.py / jcp-dims's dacc.py (copied unchanged in this lane for provenance). One mesh per job,
one GPU. Serves E1 (rule draws), E2 selection (candidate ladders) and E2 verdict runs (selected rules) with ONE code path.

Config:
  {"attempt", "mesh", "edges", "cases": "all38" | [[coh, c], ...],
   "bank": {"ckpt", "ckpt_sha256", "rotation", "rotation_sha256"}          (optional; default = incumbent dn256b),
   "groups": [{"name", "Rp", "target": "gauss384" | null, "check": "gauss320" | null, "check_cases": "dev6",
               "tighter_cases": "dev6",
               "arms": [{"form": "LSPG"|"GAL", "scheme", "factor"}],
               "rules": [{"label", "rule", "succ": rule | null, "level": "full" | "prod"}],
               "fixture": {"path", "sha256", "rules": [labels]}   (optional: in-job rho_tgt at frozen states, C-E1.1),
               "permute": {"label", "arm": index}                 (optional: G-E1.3 permuted-case replay),
               "identity": {"label_s1", "claimed"}                (optional: C-E1.2 draw-identity control)}]}

Per group: one Setting (r, frozen 1024^2 test modes, M = 4r) whose fixed roles are 'tgt' (target) and 'chk' (target
check). Phase T: for every arm, target runs on every case and check runs on check_cases (target certification A3.3);
target fields stay in host memory. Phase R: for every rule entry, roles 'd' (the rule) and 'ds' (its successor) are
added, and for every arm and case:
  prod     fused query, internal trajectory recorded, certified in-job (LSPG gamma / GAL eta, qualify certifiers),
           G-cert against the solver's own record
  tight    (level full) tight tolerance, every case                       -> Q1 distance
  tighter  (level full) tighter tolerance on tighter_cases                -> Q1 tight/tighter agreement
  succ     (level full, succ given) successor-rule replay, every case     -> Q2 rollout distance, rho_succ at own states
  always   rho_tgt at own states and the rollout distance to the target  (cross-family gate)
The six output coefficient vectors of every run are saved (E0 scoring and every distance are recomputed offline with the
independent NumPy decode); f257 fields are not saved (size; DESIGN 8 and brief section 16).

    python r2acc.py --config configs/<attempt>.json --out <dir> [--dry-run]
Outputs: result.json, T_<group>.npz, R_<group>_<label>.npz, CTRL_<group>.json, COMPLETE.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import p1common as P
from p1common import Q, QS, T2, DT0, sha
import qcert as QC
import jax
import jax.numpy as jnp

GAMMA = 1e-3
HERE = Path(__file__).resolve().parent


def steps_of(dt):
    s, k = int(round(.25 / dt)), int(round(.05 / dt))
    assert abs(s * dt - .25) < 1e-12 and abs(k * dt - .05) < 1e-12, dt
    return s


def arm_key(arm):
    return f"{arm['form']}_{arm['scheme']}_{float(arm['factor']):g}"


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Queries:
    def __init__(self):
        self.c = {}

    def get(self, st, form, nrec, nstep):
        k = (id(st), form, nrec, nstep)
        if k not in self.c:
            self.c[k] = T2.make_split_query(Q, 'point', form, st.Rp, st.L, st.trust, nrec=nrec, budget=600, nstep=nstep)
        return self.c[k]


QQ = Queries()


def run(bank, st, role, form, scheme, dt, level, u0, nu, record=False):
    st.check(role, bank)
    S = steps_of(dt)
    nstep = max(T2.NSTEP, S + 1)
    q = QQ.get(st, form, S + 1 if record else 0, nstep)
    g, t_ = P.tol_of(form, level)
    return q['fused'](u0, nu, st.data[(role, form)], st.cold, T2.sched(scheme, dt, g, t_))


def rule_identity(st, role, claimed):
    """C-E1.2 / G-E1.1: the table under `role` must have been built from the points of `claimed` (sha256)."""
    want = P.rule_points(claimed)[2]
    if st.meta[role]['rule_sha256'] != want:
        raise ValueError(f'rule identity mismatch: role {role} holds {st.meta[role]["rule"]}, claimed {claimed}')
    return True


def load_bank_cfg(cfg):
    b = cfg.get('bank')
    if not b:
        return None, None, None
    ck, rot = HERE / b['ckpt'], HERE / b['rotation']
    if b.get('ckpt_sha256') == 'SMOKE-RUNTIME':      # a bank trained earlier in the same SMOKE job: hashed at load, recorded
        assert str(cfg.get('attempt', '')).startswith('smk'), 'runtime-hashed banks are allowed in smokes only'
        b = dict(b, ckpt_sha256=sha_file(ck), rotation_sha256=sha_file(rot), runtime_hashed=True)
    assert sha_file(ck) == b['ckpt_sha256'], ('checkpoint hash', ck)
    assert sha_file(rot) == b['rotation_sha256'], ('rotation hash', rot)
    return ck, rot, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cases = P.case_list(cfg)
    ids_all = P.select(cases, cfg.get('cases', 'all38'))
    if a.dry_run:
        inv = dict(prod=0, tight=0, tighter=0, succ=0, target=0, check=0)
        for g in cfg['groups']:
            nt = len(P.select(cases, g.get('tighter_cases', 'dev6')))
            for arm in g['arms']:
                if g.get('target'):
                    inv['target'] += len(ids_all)
                if g.get('check'):
                    inv['check'] += len(P.select(cases, g.get('check_cases', 'dev6')))
                for r in g['rules']:
                    inv['prod'] += len(ids_all)
                    if r.get('level', 'full') == 'full':
                        inv['tight'] += len(ids_all)
                        inv['tighter'] += nt
                        inv['succ'] += len(ids_all) if r.get('succ') else 0
        inv['total'] = sum(inv.values())
        print(json.dumps(inv))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    L = int(cfg['mesh'])
    ck, rot, bcfg = load_bank_cfg(cfg)
    rep = dict(config=cfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, mesh=L, bank=bcfg or 'incumbent',
               checkpoint_sha256=sha_file(ck or T2.CKPT), rotation_sha256=sha_file(rot or Q.ROTATION),
               cohorts=P.cohort_hashes(), setup={}, rows=[], groups={}, memory={}, elapsed={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    bank = P.Bank(L, cfg['edges'], nrb=cfg.get('bank_blocks'), ckpt=ck, rotation=rot)
    rep['setup']['bank'] = dict(edges=bank.edges, seconds=bank.build_seconds, bytes=bank.bytes, mem=P.mem())
    ops = bank.ops('frozen')
    rep['setup']['ops_frozen'] = dict(seconds=ops['seconds'], list_hash_2048=P.list_hash(ops['kx'], ops['ky']))
    s256 = L // 256
    U0 = {}
    for i in ids_all:
        coh, c, ph = cases[i]
        u0n = Q.e.initial(L, ph)
        U0[i] = (jnp.asarray(u0n), float(ph[4]), float(np.linalg.norm(u0n[::s256, ::s256])))
    cid = lambda i: f'{cases[i][0]}|{cases[i][1]}'
    save()

    def go(st, role, arm, level, ci, record=False):
        u0, nu, n0 = U0[ci]
        dt = DT0 * float(arm['factor'])
        t1 = time.perf_counter()
        v = run(bank, st, role, arm['form'], arm['scheme'], dt, level, u0, nu, record=record)
        jax.block_until_ready(v)
        return v, P.restrict(v['fields'], L), time.perf_counter() - t1

    for g in cfg['groups']:
        t_g = time.perf_counter()
        fixed = {k: g[k] for k in ('target', 'check') if g.get(k)}
        roles = {'tgt': fixed['target']} if 'target' in fixed else {}
        if 'check' in fixed:
            roles['chk'] = fixed['check']
        main_rule = g['rules'][0]['rule']           # 'main' is never run; the cheapest table keeps memory low
        st = P.Setting(bank, g['name'], g['Rp'], main_rule, 'frozen', roles=roles, forms=('LSPG', 'GAL'), nrec=0)
        nl = st.q['LSPG']['nl']
        certs = dict(LSPG=T2.make_certifier(nl, st.Rp), GAL=QC.make_gal_certifier(nl, st.Rp))
        grep = rep['groups'][g['name']] = dict(Rp=st.Rp, M=st.M, mode_hash=st.mode_hash, meta_fixed=dict(st.meta),
                                               rules={}, target={}, controls={})
        ids = ids_all
        tight_ids = [i for i in P.select(cases, g.get('tighter_cases', 'dev6')) if i in ids]
        chk_ids = [i for i in P.select(cases, g.get('check_cases', 'dev6')) if i in ids]
        # ---------------- phase T: target and target check (A3.3)
        FT = {}
        if 'tgt' in roles:
            TW = {}
            for arm in g['arms']:
                ak = arm_key(arm)
                Wt = np.full((len(ids), 6, st.Rp), np.nan)
                vt_ = np.zeros(len(ids), bool)
                FT[ak] = {}
                for j, ci in enumerate(ids):
                    v, f, secs = go(st, 'tgt', arm, 'prod', ci)
                    Wt[j], vt_[j] = np.asarray(v['W']), P.run_valid(v)
                    FT[ak][ci] = f
                    row = dict(group=g['name'], label='TARGET', rule=roles['tgt'], arm=ak, case=cid(ci), kind='target',
                               valid=bool(vt_[j]), seconds=secs, W_sha=sha(v['W']))
                    if ci in chk_ids and 'chk' in roles:
                        vc, fc, _ = go(st, 'chk', arm, 'prod', ci)
                        row.update(check_rule=roles['chk'], check_valid=P.run_valid(vc), dist_check=P.fdist(f, fc, U0[ci][2]))
                        TW[(ak, ci)] = np.asarray(vc['W'])
                        del vc
                    rep['rows'].append(row)
                    del v
                np.savez_compressed(out / f"T_{g['name']}_{ak}.npz", case_ids=np.array([cid(i) for i in ids]), W=Wt, valid=vt_,
                                    check_case_ids=np.array([cid(i) for i in chk_ids] if 'chk' in roles else [], dtype='<U16'),
                                    W_check=np.array([TW[(ak, i)] for i in chk_ids]) if chk_ids and 'chk' in roles else np.zeros((0, 6, st.Rp)),
                                    rule=roles['tgt'], check_rule=roles.get('chk', ''), Rp=st.Rp, mesh=L,
                                    form=arm['form'], scheme=arm['scheme'], dt=DT0 * float(arm['factor']))
                grep['target'][ak] = dict(n=len(ids), seconds=time.perf_counter() - t_g)
                print('TARGET', g['name'], ak, el(), flush=True)
                save()
        # ---------------- phase R: rules (draws / candidates / controls)
        for r in g['rules']:
            t_r = time.perf_counter()
            lab, full = r['label'], r.get('level', 'full') == 'full'
            st.add_role(bank, 'd', r['rule'], forms=('LSPG', 'GAL'))
            rule_identity(st, 'd', r['rule'])
            if full and r.get('succ'):
                st.add_role(bank, 'ds', r['succ'], forms=('LSPG', 'GAL'))
            grep['rules'][lab] = dict(rule=r['rule'], succ=r.get('succ'), level=r.get('level', 'full'), meta=st.meta['d'],
                                      meta_succ=st.meta.get('ds') if full and r.get('succ') else None, arms={})
            for arm in g['arms']:
                ak, form, scheme = arm_key(arm), arm['form'], arm['scheme']
                dt = DT0 * float(arm['factor'])
                S = steps_of(dt)
                n, nt = len(ids), len(tight_ids)
                sch = T2.sched(scheme, dt, *P.tol_of(form, 'prod'))
                R = dict(case_ids=np.array([cid(i) for i in ids]), W=np.full((n, 6, st.Rp), np.nan),
                         w0=np.full((n, st.Rp), np.nan), n0=np.full(n, np.nan), cert=np.full((n, S), np.nan),
                         rec=np.full((n, S), np.nan), raw=np.full((n, S), np.nan), scale=np.full((n, S), np.nan),
                         tolratio=np.full((n, S), np.nan), it=np.full((n, S), -1), reason=np.full((n, S), -1),
                         valid=np.zeros(n, bool), secs=np.full(n, np.nan),
                         W_tight=np.full((n, 6, st.Rp), np.nan), valid_tight=np.zeros(n, bool),
                         tighter_case_ids=np.array([cid(i) for i in tight_ids]),
                         W_tighter=np.full((nt, 6, st.Rp), np.nan), valid_tighter=np.zeros(nt, bool),
                         W_succ=np.full((n, 6, st.Rp), np.nan), valid_succ=np.zeros(n, bool),
                         rho_succ=np.full((n, 6), np.nan), rho_tgt=np.full((n, 6), np.nan))
                for j, ci in enumerate(ids):
                    u0, nu, n0 = U0[ci]
                    v, f, secs = go(st, 'd', arm, 'prod', ci, record=True)
                    stt = v['stats']
                    R['W'][j], R['w0'][j], R['n0'][j], R['secs'][j] = np.asarray(v['W']), np.asarray(v['w0']), n0, secs
                    R['it'][j], R['reason'][j] = (np.asarray(stt[k])[:S] for k in ('step_it', 'step_reason'))
                    if form == 'LSPG':
                        gm, _ = certs['LSPG'](stt['states'], nu, st.data[('d', 'LSPG')], sch)
                        R['cert'][j] = np.asarray(gm)
                        R['rec'][j] = np.asarray(stt['step_gn'])[:S]
                        gok = bool(P.gcert_ok(R['cert'][j], R['rec'][j]) and np.isfinite(R['cert'][j]).all())
                        nonst = int(np.sum(~(R['cert'][j] <= GAMMA)))
                    else:
                        _, tprod = P.tol_of(form, 'prod')
                        eta, gam, jc, rn_, scl = certs['GAL'](stt['states'], nu, st.data[('d', 'GAL')], sch)
                        R['cert'][j], R['raw'][j], R['scale'][j] = np.asarray(eta), np.asarray(rn_), np.asarray(scl)
                        R['tolratio'][j] = np.asarray(stt['step_tolratio'])[:S]
                        R['rec'][j] = QC.recorded_eta(R['tolratio'][j], R['scale'][j], tprod)
                        gok = QC.gcert_gal(R['cert'][j], R['raw'][j], R['scale'][j], R['tolratio'][j], tprod)
                        nonst = int(np.sum(~QC.certified_gal(R['cert'][j])))
                    R['valid'][j] = P.run_valid(v)
                    row = dict(group=g['name'], label=lab, rule=r['rule'], arm=ak, case=cid(ci), kind='prod', seconds=secs,
                               valid=bool(R['valid'][j]), it_sum=int(stt['it_sum']), noncertified=nonst, steps=S,
                               gcert_ok=gok, budget_exits=P.budget_exits(v), W_sha=sha(v['W']))
                    del v
                    for kk in range(6):           # own-state indicators: vs successor (Q2) and vs target (cross-family)
                        cvec = jnp.asarray(R['W'][j, kk])
                        Nd = np.asarray(nl(cvec, st.data[('d', 'LSPG')]))
                        if 'tgt' in roles:
                            R['rho_tgt'][j, kk] = P.rho(Nd, np.asarray(nl(cvec, st.data[('tgt', 'LSPG')])))
                        if full and r.get('succ'):
                            R['rho_succ'][j, kk] = P.rho(Nd, np.asarray(nl(cvec, st.data[('ds', 'LSPG')])))
                    if 'tgt' in roles:
                        row.update(dist_target=P.fdist(f, FT[ak][ci], n0), rho_tgt_max=float(np.nanmax(R['rho_tgt'][j])))
                    if full:
                        vt, ft, st_ = go(st, 'd', arm, 'tight', ci)
                        R['W_tight'][j], R['valid_tight'][j] = np.asarray(vt['W']), P.run_valid(vt)
                        row.update(dist_tight=P.fdist(f, ft, n0), tight_valid=bool(R['valid_tight'][j]), seconds_tight=st_)
                        del vt
                        if ci in tight_ids:
                            k = tight_ids.index(ci)
                            vtt, ftt, _ = go(st, 'd', arm, 'tighter', ci)
                            R['W_tighter'][k], R['valid_tighter'][k] = np.asarray(vtt['W']), P.run_valid(vtt)
                            row.update(dist_tight_tighter=P.fdist(ft, ftt, n0), tighter_valid=bool(R['valid_tighter'][k]))
                            del vtt
                        if r.get('succ'):
                            vq, fq, _ = go(st, 'ds', arm, 'prod', ci)
                            R['W_succ'][j], R['valid_succ'][j] = np.asarray(vq['W']), P.run_valid(vq)
                            row.update(dist_succ=P.fdist(f, fq, n0), succ_valid=bool(R['valid_succ'][j]),
                                       rho_succ_max=float(np.nanmax(R['rho_succ'][j])))
                            del vq
                    rep['rows'].append(row)
                R.update(rule=r['rule'], succ=r.get('succ') or '', target=roles.get('tgt', ''), label=lab, Rp=st.Rp, mesh=L,
                         form=form, scheme=scheme, factor=float(arm['factor']), dt=dt, steps=S, policy=st.policy,
                         mode_hash=st.mode_hash, level=r.get('level', 'full'), tol=np.array(P.tol_of(form, 'prod')),
                         time_tag='t0,t1,t2,t3,t4,t5', rule_sha256=st.meta['d']['rule_sha256'])
                np.savez_compressed(out / f"R_{g['name']}_{lab}_{ak}.npz", **R)
                grep['rules'][lab]['arms'][ak] = dict(n=n, seconds=time.perf_counter() - t_r)
                print('RULE', g['name'], lab, ak, el(), flush=True)
                save()
            # ------------ controls bound to this rule
            ctl = grep['controls']
            fx = g.get('fixture')
            if fx and lab in fx['rules'] and 'tgt' in roles:          # C-E1.1 in-job advection-level check (A1.3)
                fp = HERE / fx['path']
                assert sha_file(fp) == fx['sha256'], 'fixture hash'
                z = np.load(fp)
                C = np.asarray(z['W'])
                vals = []
                for kk in range(len(C)):
                    cvec = jnp.asarray(C[kk])
                    vals.append(P.rho(np.asarray(nl(cvec, st.data[('d', 'LSPG')])),
                                      np.asarray(nl(cvec, st.data[('tgt', 'LSPG')]))))
                ctl.setdefault('fixture', {})[lab] = dict(rule=r['rule'], rho_tgt=vals, fixture=fx['path'], fixture_sha256=fx['sha256'])
            pm = g.get('permute')
            if pm and pm['label'] == lab:                               # G-E1.3 permuted-case replay
                arm = g['arms'][pm['arm']]
                ak = arm_key(arm)
                Rsaved = np.load(out / f"R_{g['name']}_{lab}_{ak}.npz")
                perm = list(reversed(ids))
                dev = []
                for ci in perm:
                    v, _f, _ = go(st, 'd', arm, 'prod', ci, record=True)
                    j = ids.index(ci)
                    Wv = np.asarray(v['W'])
                    dev.append(float(np.max(np.abs(Wv - Rsaved['W'][j])) / max(float(np.max(np.abs(Rsaved['W'][j]))), 1e-300)))
                    del v
                ctl['permute'] = dict(label=lab, arm=ak, order=[cid(i) for i in perm], max_rel_dev=max(dev), per_case=dev,
                                      passed=bool(max(dev) <= 1e-12))
            idn = g.get('identity')
            if idn and idn['label_s1'] == lab:                          # C-E1.2: s1 table claimed as s0 must be refused
                try:
                    rule_identity(st, 'd', idn['claimed'])
                    res = dict(raised=False)
                except ValueError as e_:
                    res = dict(raised=True, message=str(e_))
                res['fired'] = res['raised']
                ctl['identity'] = dict(res, holder=r['rule'], claimed=idn['claimed'])
            st.drop_role('d')
            st.drop_role('ds')
            for k in ('d', 'ds'):
                st.meta.pop(k, None)
                st.rules.pop(k, None)
            gc.collect()
            save()
        rep['memory'][g['name']] = P.mem()
        rep['elapsed'][g['name']] = time.perf_counter() - t_g
        QQ.c.clear()
        del st
        gc.collect()
        save()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('R2ACC COMPLETE', el(), flush=True)


if __name__ == '__main__':
    main()
