"""burgers-quadrature-plateau, 3D driver: one mesh, one allocation (DESIGN.md is the specification).

The rollout of every arm is the quadrature-burgers3d panel path, unchanged: vendored tables.build_tables, the off-mesh tables
(offmesh.offmesh_tables) and offmesh.make_fsc_rule at R'=512 (dt 0.01, gtol 1e-3, trust 0.05 x spread), with qpanel's
evaluator gates G1-G4 copied verbatim in logic. What is new: the rule ladder (precomputed rules3d.npz, sha-checked), scoring
against the E0 references (refstore metric, test32 = seed 923901), the FOM comparators of DESIGN §7 on the same cases, and
one timing panel in this allocation (lattice contract, A1.2) plus a confirmation panel (A1.7, A2.4).
python plateau3d.py --config configs/<job>.json --out output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / 'vendor' / 'tree' / 'experiments'
for _p in (V / 'jcp-speed3d', V / 'jcp-references', V / 'quadrature-burgers3d'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# order: quadrature-burgers3d, jcp-references, jcp-speed3d. `common` resolves to the quadrature-burgers3d vendor copy
# (byte-identical to the E0 and speed3d copies, vendor/PROVENANCE.json); pscore to E0 (identical to the speed3d copy).

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

import offmesh as OM  # noqa: E402
from offmesh import C, TB  # noqa: E402
import qpanel as QP  # noqa: E402     (lattice65_coords / lattice65_index / rel_max helpers, unchanged)
import refstore as RS  # noqa: E402
import verdict as VD  # noqa: E402
import fomrun as FR  # noqa: E402
import psjax  # noqa: E402
import qlat3d  # noqa: E402
import plateau as PL  # noqa: E402

block = jax.block_until_ready
COHORT = {'test32': (923901, 32, '85cef1e7e736c8203a4161d5e7dc575f1291befe744086955d0823c6cbf673ea'),
          'dev4': (923651, 4, None)}
NONCONV = (0, 2, 3)                   # DESIGN A2.2


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg_path(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def clean(x):
    return C.clean(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    if not cfg.get('local_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    log = lambda s: print(f'[{el()}s] {s}', flush=True)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    n = int(cfg['mesh'])
    model = QP.load_model(cfg_path(cfg['model']).parent if cfg_path(cfg['model']).is_file() else cfg_path(cfg['model']))
    assert model['sha'] == cfg['expected_model_sha256'], model['sha']
    rep = dict(config=cfg, mesh=n, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi,
               x64=True, precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
               model_sha256=model['sha'], store_root=os.environ.get('JCPREF_STORE'), gates={}, setup={}, rows=[],
               fom_rows=[], timing={}, complete=False)
    save = lambda: C.dump(out / 'result.json', clean(rep))
    save()

    # ------------------------------------------------------------------ rules ----
    rp = cfg_path(cfg['rules_npz'])
    man = json.loads(cfg_path(cfg['rules_manifest']).read_text())
    rep['rules_npz_sha256'] = sha_file(rp)
    assert rep['rules_npz_sha256'] == man['npz_sha256'], 'rules npz differs from the manifest'
    z = np.load(rp)
    names = [s['rule'] for s in cfg['arms']]
    rules = {}
    bad = []
    for nm in names:
        X, w = np.asarray(z[f'{nm}_X']), np.asarray(z[f'{nm}_w'])
        if hashlib.sha256(X.tobytes() + w.tobytes()).hexdigest() != man['rules'][nm]['sha256']:
            bad.append(nm)
        rules[nm] = (X, w)
    rep['gates']['G_R'] = dict(checked=len(names), mismatched=bad, passed=not bad)
    assert not bad, bad

    # ------------------------------------------------------------- cohort/refs ----
    coh = cfg['cohort']
    seed, count, want = COHORT[coh]
    tab = C.table(seed, count)
    rep['gates']['G_coh'] = dict(cohort=coh, sha256=tab['sha256'], expected=want, bitwise=want in (None, tab['sha256']))
    cases = list(cfg.get('cases', range(count)))
    rep['cases'] = cases
    rep['store_version'] = RS.store_version()
    assert rep['store_version'] == cfg['expected_store_version'], rep['store_version']
    REF, META, STAT = {}, {}, {}
    for c in cases:
        for kind in ('ST', 'S'):
            f, m = RS.load(3, coh, c, kind)
            assert f.shape == (6, 63 ** 3) and np.isfinite(f).all()
            REF[(c, kind)] = np.asarray(f, np.float64)
            META[(c, kind)] = dict(n0=float(m['judge']['n0']), eps_ref=m['judge']['eps_ref'], outcome=m['outcome'])
            STAT[(c, kind)] = RS.reference_status(3, coh, c, kind)
    rep['references'] = {f'{c}|{k}': dict(META[(c, k)], eligible=STAT[(c, k)]['eligible']) for (c, k) in META}
    rep['gates']['G_ref'] = dict(store_version=rep['store_version'], loaded=len(REF), passed=True)
    log(f'references loaded: {len(REF)}')
    # G-coh (amendment A5): the drawn parameters equal the E0 reference descriptors exactly, s_star within 1e-12 rel.
    worst_d, worst_s, nb = 0.0, 0.0, True
    rl = lambda a, b: float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)) / np.maximum(np.abs(np.asarray(b, float)), 1e-300)))
    for c in cases:
        dsc = RS.load(3, coh, c, 'ST')[1]['desc']
        B = int(tab['B'][c])
        nb &= B == int(dsc['B'])
        worst_d = max(worst_d, rl(tab['A'][c], dsc['A']), rl(tab['nu'][c], dsc['nu']),
                      *(rl(np.asarray(tab[k][c])[:B], np.asarray(dsc[k])[:B]) for k in ('c', 'w', 'rho')))
        worst_s = max(worst_s, rl(tab['s_star'][c], dsc['s_star']))
    rep['gates']['G_coh'].update(blob_counts_equal=bool(nb), draws_max_rel_vs_E0=worst_d, s_star_max_rel_vs_E0=worst_s,
                                 passed=bool(nb and worst_d <= 1e-14 and worst_s <= 1e-12))
    assert rep['gates']['G_coh']['passed'], rep['gates']['G_coh']

    def score(F, c, kind):
        e, et = VD.rel_err(F, REF[(c, kind)], META[(c, kind)]['n0'])
        return dict(e=e, e_t=et, eligible=STAT[(c, kind)]['eligible'])

    # ================================================================ 1. tables (qpanel phase 1, unchanged calls)
    R = int(model['T'].shape[1])
    Rp = int(cfg['Rp'])
    ladder = sorted({Rp, R} | set(cfg.get('table_ladder_extra', [])))   # 257: [256] splits the 68 GB bank build as in October
    order_, lam_all = C.mode_order(n)
    M_of = lambda r: C.complete_M(n, 4 * r, order_, lam_all)
    t0 = time.perf_counter()
    tb = TB.build_tables(n, model['bank'], model['T'], ladder, M_of, log=log)
    rep['setup']['table_seconds'] = time.perf_counter() - t0
    rep['gates']['gram_condition'] = tb['gram_cond'] ** 2
    assert rep['gates']['gram_condition'] <= 1e8, rep['gates']
    kx = tb['kxyz']
    rng = np.random.default_rng(20261001)
    cvec = rng.normal(size=R) / np.sqrt(R)
    u = TB.decode(jnp.asarray(cvec)[None], tb)[0]
    idx = tuple(jnp.asarray(kx[:, j]) for j in range(3))
    direct = np.asarray(C.phiT(C.backward_adv(u, n)[None], n, idx)[0])
    via = np.asarray(0.5 * jnp.einsum('mij,i,j->m', tb['Tsym'], jnp.asarray(cvec), jnp.asarray(cvec)))
    rep['gates']['tensor_vs_direct'] = float(np.linalg.norm(via - direct) / np.linalg.norm(direct))
    assert rep['gates']['tensor_vs_direct'] < 1e-10, rep['gates']
    del u
    save()

    # ================================================================ 2. off-mesh tables + evaluator gates (qpanel G1-G3)
    om = {nm: OM.offmesh_tables(n, model['bank'], model['T'], *rules[nm], kx) for nm in names}
    log('off-mesh tables built')
    Xg = rules[names[0]][0][:32]
    _, Dg = OM.point_blocks(model['bank'], model['T'], Xg)
    h = 1e-5
    fp, _ = OM.point_blocks(model['bank'], model['T'], Xg + h)
    fm, _ = OM.point_blocks(model['bank'], model['T'], Xg - h)
    rep['gates']['G1_derivative_vs_fd'] = QP.rel_max((fp - fm) / (2 * h), Dg)
    Rs, Ms = min(32, R), 64
    Xn = C.interior_coords(n)
    rows_ = jnp.array(tb['GTb'][0][:Rs])
    Drows = C.dminus(rows_, n)
    cs = jnp.asarray(rng.normal(size=Rs) / np.sqrt(Rs))
    Ju_mesh = 0.
    for s in range(0, Xn.shape[0], 1 << 20):
        Pq = OM.test_block(n, Xn[s:s + (1 << 20)], np.full(min(1 << 20, Xn.shape[0] - s), float(n - 1) ** -3), kx[:Ms])
        Ju_mesh = Ju_mesh + OM.contract_offmesh(dict(B=rows_[:, s:s + (1 << 20)].T, D=Drows[:, s:s + (1 << 20)].T, P=Pq), cs)
    Ju_t = OM.contract_tensor(dict(Ts=tb['Tsym'][:Ms, :Rs, :Rs]), cs)
    rep['gates']['G2_meshnodes_offmesh_vs_tensor'] = QP.rel_max(Ju_mesh, Ju_t)
    del rows_, Drows, Xn, Pq, Ju_mesh
    M = int(M_of(Rp))
    d3 = dict(B=om[names[0]]['B'][:, :Rp], D=om[names[0]]['D'][:, :Rp], P=om[names[0]]['P'][:, :M])
    c3 = jnp.asarray(rng.normal(size=Rp) / np.sqrt(Rp))
    J_ad = jax.jacfwd(lambda cc: OM.adv_offmesh_batch(d3, cc[None])[0])(c3)
    J_f = OM.contract_offmesh(d3, c3)
    rep['gates']['G3_jacobian_vs_jacfwd'] = QP.rel_max(J_f, J_ad)
    rep['gates']['G3_adv_half_Jc'] = QP.rel_max(0.5 * J_f @ c3, OM.adv_offmesh_batch(d3, c3[None])[0])
    del d3, J_ad
    assert rep['gates']['G1_derivative_vs_fd'] < 1e-6, rep['gates']
    assert rep['gates']['G2_meshnodes_offmesh_vs_tensor'] < 1e-12, rep['gates']
    assert rep['gates']['G3_jacobian_vs_jacfwd'] < 1e-12 and rep['gates']['G3_adv_half_Jc'] < 1e-12, rep['gates']
    log(f"evaluator gates {rep['gates']}")

    # ================================================================ arms
    G65 = block(jnp.asarray(TB.feature_rows(model['bank'], model['T'], QP.lattice65_coords()))[:Rp])   # (R', 63^3)
    trust = float(cfg['trust_fraction'] * model['spread'][str(Rp)])
    base = TB.arm_data(tb, Rp, M)
    lean = {k: v for k, v in base.items() if k != 'Ts'}
    arms = {}

    def add_arm(name, family, rule, data, contract):
        q = OM.make_fsc_rule(n, Rp, M, contract, dt=cfg['dt'], gtol=cfg['gtol'], trust=trust)
        ql = qlat3d.make_fsc_rule_lat(n, Rp, M, contract, dt=cfg['dt'], gtol=cfg['gtol'], trust=trust)
        dl = dict(data, G65=G65)
        arms[name] = dict(family=family, rule=rule, data=data, q=q, ql=ql, dl=dl)

    add_arm('tensor', 'tensor', 'tensor', base, OM.contract_tensor)
    for s_ in cfg['arms']:
        nm = s_['rule']
        d = dict(lean, B=om[nm]['B'][:, :Rp], D=om[nm]['D'][:, :Rp], P=om[nm]['P'][:, :M])
        d = jax.tree_util.tree_map(lambda x: block(jnp.asarray(x)), d)
        add_arm(s_['name'], s_['family'], nm, d, OM.contract_offmesh)
    del om
    rep['setup'].update(R_prime=Rp, M=M, trust=trust, dt=cfg['dt'], gtol=cfg['gtol'],
                        arms={k: dict(family=v['family'], rule=v['rule'],
                                      m=(int(v['data']['B'].shape[0]) if 'B' in v['data'] else None)) for k, v in arms.items()})
    # G4 (qpanel): tensor rule through make_fsc_rule == vendor make_fsc (one dev case)
    tab_g = C.table(923811, 1)
    u0g = jnp.asarray(C.initial_interior(n, tab_g, 0))
    qv, _ = TB.make_fsc(n, Rp, M, dt=cfg['dt'], gtol=cfg['gtol'], trust=trust)
    o_v = qv(u0g, float(tab_g['nu'][0]), base, {})
    o_r = arms['tensor']['q'](u0g, float(tab_g['nu'][0]), base, {})
    rep['gates']['G4_solver_vs_vendor'] = dict(fields=QP.rel_max(o_r[0], o_v[0]), coefs=QP.rel_max(o_r[1], o_v[1]),
                                               its_equal=all(bool(jnp.all(o_r[i] == o_v[i])) for i in (2, 3, 6)))
    g4 = rep['gates']['G4_solver_vs_vendor']
    assert g4['fields'] <= 1e-12 and g4['coefs'] <= 1e-12 and g4['its_equal'], g4
    del o_v, o_r, qv
    log('arms built; G4 passed')
    save()

    # ================================================================ 4. cohort accuracy
    U0 = {j: block(jnp.asarray(C.initial_interior(n, tab, j))) for j in cases}
    NU = {j: float(tab['nu'][j]) for j in cases}
    keep = int(round(0.05 / cfg['dt']))
    conv = cfg['conv_arm']
    tcases = cases[:cfg['timing']['cases']]
    COEF, FCONV, KEEP_T = {}, {}, {}
    names_ord = [conv] + [k for k in arms if k != conv]
    for nm in names_ord:
        s_ = arms[nm]
        t_c = time.perf_counter()
        for j in cases:
            t1 = time.perf_counter()
            o = s_['q'](U0[j], NU[j], s_['data'], {})
            block(o)
            secs = time.perf_counter() - t1
            Cout = np.asarray(o[1])[::keep]
            F = np.asarray(jnp.asarray(Cout) @ G65)
            rs = np.asarray(o[3])
            row = dict(arm=nm, rule=s_['rule'], family=s_['family'],
                       m=(int(s_['data']['B'].shape[0]) if 'B' in s_['data'] else None), case=j,
                       finite=bool(np.isfinite(np.asarray(o[0])).all() and np.isfinite(Cout).all()), seconds_first=secs,
                       iterations=int(np.asarray(o[2]).sum()), reasons={str(k): int((rs == k).sum()) for k in range(5)},
                       steps=int(rs.size), nonconv_steps=int(np.isin(rs, NONCONV).sum()), hard_fail=bool((rs == 3).any()))
            for kind in ('ST', 'S'):
                row[kind] = score(F, j, kind)
            if nm == conv:
                FCONV[j] = F
            else:
                row['dist_conv'] = float(np.max(VD.tnorm(F - FCONV[j])[1:] / META[(j, 'ST')]['n0']))
            rep['rows'].append(row)
            COEF[(nm, j)] = Cout
            if j in tcases:
                KEEP_T[(nm, j)] = (F, Cout)
            del o
        log(f'ARM {nm} done ({time.perf_counter() - t_c:.1f}s)')
        save()
    np.savez(out / 'fields' / 'coefficients.npz', **{f'{nm}__c{j}': v for (nm, j), v in COEF.items()})
    np.save(out / 'fields' / 'G65_head.npy', np.asarray(G65[:, :64]))
    rep['phases'] = dict(accuracy=el())
    save()

    # ================================================================ 5. FOMs
    i65 = jnp.asarray(QP.lattice65_index(n))
    foms = FR.FOMs()
    hist = {}
    FKEEP, PREP = {}, {}
    for d in cfg['foms']:
        name = d['name']
        t1 = time.perf_counter()
        if d['kind'] == 'hist':
            hist[name] = C.make_fom(n, d['dt'], d['ntol'], d['ltol'])
        for j in cases:
            if d['kind'] == 'hist':
                f, it, rn = hist[name](U0[j], NU[j])
                F = np.asarray(block(f)[:, i65])
                ok = bool(np.isfinite(F).all() and int(jnp.max(it)) < C.MAX_NEWTON and float(jnp.max(rn)) <= d['ntol'] * (1 + 1e-9))
                ev = dict(max_newton=int(jnp.max(it)), worst_res=float(jnp.max(rn)))
                x = None
            else:
                sp = ('ps', int(d['N']), FR.S_canon(d['dt'])) if d['kind'] == 'ps' else \
                     ('fd', d['fam'], int(d['L']), FR.S_canon(d['dt']), int(d['lev']))
                t_p = time.perf_counter()
                x = psjax.ic(tab, j, sp[1]) if sp[0] == 'ps' else jnp.asarray(C.initial_interior(sp[2] + 1, tab, j))
                x = block(x)
                prep = time.perf_counter() - t_p
                Fd, info = foms.run(sp, x, NU[j])
                F = np.asarray(block(Fd))
                ok, ev = FR.FOMs.converged(sp, info, F)
                ev = dict(ev, prepare_seconds=prep)
            row = dict(fom=name, case=j, converged=bool(ok), evidence=ev)
            for kind in ('ST', 'S'):
                row[kind] = score(F, j, kind)
            rep['fom_rows'].append(row)
            if j in tcases:
                FKEEP[(name, j)] = F
                PREP[(name, j)] = x
        log(f'FOM {name} done ({time.perf_counter() - t1:.1f}s)')
        save()
    rep['phases']['foms'] = el()

    # ================================================================ 6. timing
    tc = cfg['timing']
    prng = np.random.default_rng(int(tc.get('seed', 20261105)))
    xb = jnp.ones((2048, 2048))

    def burn(sec):
        t_ = time.perf_counter()
        while time.perf_counter() - t_ < sec:
            block(xb @ xb)
    fdef = {d['name']: d for d in cfg['foms']}

    def fn_of(sj, j):
        kind, nm = sj
        if kind == 'rom_lat':
            s_ = arms[nm]
            return lambda: s_['ql'](U0[j], NU[j], s_['dl'], {})
        if kind == 'rom_full':
            s_ = arms[nm]
            return lambda: s_['q'](U0[j], NU[j], s_['data'], {})
        if kind == 'decode':
            Wd = jnp.asarray(COEF[(conv, j)])
            return lambda: TB.decode(Wd, arms[conv]['data'])
        d = fdef[nm]
        if d['kind'] == 'hist':
            return lambda: hist[nm](U0[j], NU[j])
        sp = ('ps', int(d['N']), FR.S_canon(d['dt'])) if d['kind'] == 'ps' else \
             ('fd', d['fam'], int(d['L']), FR.S_canon(d['dt']), int(d['lev']))
        x = PREP[(nm, j)]
        return lambda: foms.run(sp, x, NU[j])[0]

    def check(sj, j, o):
        kind, nm = sj
        if kind == 'rom_lat':
            F = np.asarray(o[0])
            return max(QP.rel_max(F, KEEP_T[(nm, j)][0]), QP.rel_max(np.asarray(o[1])[::keep], KEEP_T[(nm, j)][1]))
        if kind == 'rom_full':
            return QP.rel_max(np.asarray(o[1])[::keep], KEEP_T[(nm, j)][1])
        if kind == 'fom':
            F = np.asarray(o[0][:, i65]) if fdef[nm]['kind'] == 'hist' else np.asarray(o)
            return QP.rel_max(F, FKEEP[(nm, j)])
        return None

    glat = []
    for nm in arms:
        for j in tcases:
            glat.append(check(('rom_lat', nm), j, fn_of(('rom_lat', nm), j)()))
    rep['gates']['G_lat'] = dict(compared=len(glat), max_rel=max(glat), passed=bool(max(glat) <= 1e-12))
    log(f"G_lat {rep['gates']['G_lat']}")

    def panel(subjects, reps, label):
        for sj in subjects:
            for j in tcases:
                block(fn_of(sj, j)())
        burn(tc.get('initial_burn', 30.0))
        inv = []
        for r_ in range(reps):
            ordr = [(sj, j) for sj in subjects for j in tcases]
            prng.shuffle(ordr)
            for sj, j in ordr:
                fn = fn_of(tuple(sj), j)
                burn(tc['burn'])
                t1 = time.perf_counter()
                o_ = fn()
                block(o_)
                secs = time.perf_counter() - t1
                inv.append(dict(panel=label, kind=sj[0], name=sj[1], case=j, rep=r_, seconds=secs,
                                diff=check(tuple(sj), j, o_)))
                del o_
            log(f'timing {label} rep {r_} done')
        summ = {}
        for kind, nm in subjects:
            ts = np.array([d['seconds'] for d in inv if d['kind'] == kind and d['name'] == nm])
            q1, q3 = np.quantile(ts, [.25, .75])
            blocks = [np.median([d['seconds'] for d in inv if d['kind'] == kind and d['name'] == nm and d['rep'] == r_])
                      for r_ in range(reps)]
            med = float(np.median(ts))
            summ[f'{kind}|{nm}'] = dict(median_ms=1e3 * med, iqr_ms=1e3 * float(q3 - q1), min_ms=1e3 * float(ts.min()),
                                        max_ms=1e3 * float(ts.max()), n=int(len(ts)),
                                        outliers=int(np.sum((ts < q1 - 1.5 * (q3 - q1)) | (ts > q3 + 1.5 * (q3 - q1)))),
                                        block_drift=float(max(max(blocks) / med, med / min(blocks))))
        return inv, summ

    main_subj = [('rom_lat', k) for k in arms] + [('fom', d['name']) for d in cfg['foms']]
    inv, summ = panel(main_subj, tc['reps'], 'main')
    rep['timing'] = dict(cases=tcases, reps=tc['reps'], burn=tc['burn'], initial_burn=tc.get('initial_burn', 30.0),
                         invocations=inv, summary=summ, median_ms={k: v['median_ms'] for k, v in summ.items()})
    save()
    sel = PL.cheapest_plateau_rule(rep)
    rep['timing']['selected_in_job'] = sel
    conf_arms = [conv] + ([sel['arm']] if sel and sel['arm'] != conv else [])
    conf_subj = ([('rom_lat', k) for k in conf_arms] + [('rom_full', k) for k in conf_arms] + [('decode', conv)] +
                 [('fom', d['name']) for d in cfg['foms']])
    inv2, summ2 = panel(conf_subj, tc['confirm_reps'], 'confirm')
    rep['timing'].update(confirm_invocations=inv2, confirm_summary=summ2)
    diffs = [d['diff'] for d in inv + inv2 if d['diff'] is not None]
    worst_drift = max(v['block_drift'] for v in list(summ.values()) + list(summ2.values()))
    rep['gates']['G_time'] = dict(max_output_diff=max(diffs), worst_block_drift=worst_drift,
                                  outputs_pass=bool(max(diffs) <= 1e-12), drift_pass=bool(worst_drift <= 1.10))
    rep['phases']['timing'] = el()
    log(f"timing gates {rep['gates']['G_time']}; selected {sel}")

    # ================================================================ in-job checks
    rep['gates']['G_fin'] = dict(nonfinite=[(r['arm'], r['case']) for r in rep['rows'] if not r['finite']])
    rep['gates']['G_fom'] = dict(nonconverged=sorted({r['fom'] for r in rep['fom_rows'] if not r['converged']}))
    rs = []
    for j in cases[:2]:
        for nm in (conv, cfg['anchor_check_arm']):
            F = np.asarray(jnp.asarray(COEF[(nm, j)]) @ G65)
            s = RS.score(F, 3, coh, j, 'ST')
            mine = next(r['ST']['e'] for r in rep['rows'] if r['arm'] == nm and r['case'] == j)
            rs.append(abs(s['e'] - mine) / max(abs(s['e']), 1e-300))
    rep['gates']['G_score_vs_refstore'] = dict(compared=len(rs), max_rel=max(rs), passed=bool(max(rs) <= 1e-12))
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('PLATEAU3D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
