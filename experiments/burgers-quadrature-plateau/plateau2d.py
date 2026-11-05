"""burgers-quadrature-plateau, 2D driver: one mesh, one allocation (DESIGN.md is the specification).

The rollout of every rule arm is the quadrature-study phase-1 path, unchanged: vendored qcore.make_linear_query('point', ...)
with the `acc` setting (R'=384, M=1536, dt 0.005, gtol 1e-3, step budget 600), the vendored qstudy.Mesh, the same cold-start
fit. What is new: a dense ladder of off-mesh rules; scoring against the E0 second-order references (refstore ST and S, the
loader's own metric, references loaded once per case and sha-checked by refstore); the FOM comparators of DESIGN section 7
(vendored jcp-fom2d runner) on the same cases; and one timing panel (ROM arms, decode, FOMs) in this allocation.

Phases: setup -> accuracy (case-major; conv arm first) -> FOMs -> timing -> audit hooks. Output: --out/result.json and
--out/fields/*.npz. python plateau2d.py --config configs/<job>.json --out output
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / 'vendor' / 'tree' / 'experiments'
for _p in (V / 'jcp-fom2d', V / 'jcp-references', V / 'mr-burgers2d', V / 'separable-decoder',
           V / 'quadrature-study' / 'vendor', V / 'quadrature-study'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# resulting order: quadrature-study, its vendor, separable-decoder, mr-burgers2d, jcp-references, jcp-fom2d
# (refcore / pscore / cohorts resolve to jcp-references; the jcp-fom2d copies are byte-identical, vendor/PROVENANCE.json)

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

import qcore as Q  # noqa: E402
import qstudy as QS  # noqa: E402   (Mesh, burn, sha, clean: unchanged helpers of the October driver)
import refstore as RS  # noqa: E402
import verdict as VD  # noqa: E402
import fomrun2 as F2  # noqa: E402
import qlat  # noqa: E402
import plateau as PL  # noqa: E402

REASONS = QS.REASONS
NONCONV = (0, 2, 3)          # DESIGN A2.2
SETTING = 'acc'
DT = 0.005
COHORT_SHA = {'test64': 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rule_sha(X, w):
    return hashlib.sha256(np.ascontiguousarray(X, np.float64).tobytes() + np.ascontiguousarray(w, np.float64).tobytes()).hexdigest()


def fom_spec(d):
    """config dict -> fomrun2 spec tuple."""
    if d['kind'] == 'ps':
        return ('ps', int(d['N']), F2.canon_dt(d['dt']))
    return ('fd', d['fam'], int(d['L']), F2.canon_dt(d['dt']), str(d['lev']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    local = bool(cfg.get('local_smoke'))
    if not local:
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest' and jax.config.jax_enable_x64
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    log = lambda s: print(f'[{el()}s] {s}', flush=True)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    L = int(cfg['mesh'])
    s256 = L // 256
    assert s256 * 256 == L
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               nvidia_smi=smi, jax_version=jax.__version__, mesh=L, x64=True,
               precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], store_root=os.environ.get('JCPREF_STORE'),
               checkpoint_sha256=sha_file(cfg_path(cfg['checkpoint'])), rotation_sha256=sha_file(cfg_path(cfg['rotation'])),
               gates={}, setup={}, rows=[], fom_rows=[], timing={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)

    # ------------------------------------------------------------------ cohort ----
    coh = cfg['cohort']
    ph_all = Q.cohort(coh)
    h = QS.sha(ph_all)
    if coh in COHORT_SHA:
        assert h == COHORT_SHA[coh], (coh, h)
    rep['gates']['G_coh'] = dict(cohort=coh, sha256=h, expected=COHORT_SHA.get(coh), passed=COHORT_SHA.get(coh) in (None, h))
    cases = list(cfg.get('cases', range(len(ph_all))))
    rep['cases'] = cases

    # -------------------------------------------------------------- references ----
    rep['store_version'] = RS.store_version()
    assert rep['store_version'] == cfg['expected_store_version'], rep['store_version']
    REF, META, STAT = {}, {}, {}
    for c in cases:
        for kind in ('ST', 'S'):
            f, m = RS.load(2, coh, c, kind)                         # sha256-checked against the vendored manifest
            assert f.shape == (6, 257, 257) and np.isfinite(f).all()
            REF[(c, kind)] = np.asarray(f, np.float64)
            META[(c, kind)] = dict(n0=float(m['judge']['n0']), eps_ref=m['judge']['eps_ref'], outcome=m['outcome'])
            STAT[(c, kind)] = RS.reference_status(2, coh, c, kind)
    rep['references'] = {f'{c}|{k}': dict(META[(c, k)], eligible=STAT[(c, k)]['eligible']) for (c, k) in META}
    rep['gates']['G_ref'] = dict(store_version=rep['store_version'], loaded=len(REF), passed=True)
    log(f'references loaded: {len(REF)}')

    def score(F, c, kind):
        e, et = VD.rel_err(F, REF[(c, kind)], META[(c, kind)]['n0'])     # identical to refstore.score (audited)
        return dict(e=e, e_t=et, eligible=STAT[(c, kind)]['eligible'])

    # ----------------------------------------------------------- model, mesh ----
    mdl = Q.Model(ckpt=cfg_path(cfg['checkpoint']), rotation=cfg_path(cfg['rotation']))
    st = Q.SETTINGS[SETTING]
    Rp, M = st['Rp'], st['M']
    mesh = QS.Mesh(mdl, L, [SETTING])
    rep['setup']['bank'] = dict(edges=mesh.edges, seconds=mesh.bank_seconds, bytes=mesh.bank_bytes)
    Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
    hv = jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))
    Hsub = np.asarray(hv(jnp.asarray(Zsub)))
    trust, _ = mdl.trust_linear(Rp)
    cold = Q.build_cold_linear(mdl, Rp, Hsub @ mdl.Lrot[:Rp].T)
    rep['setup'].update(R_prime=Rp, M=M, trust=trust, dt=DT, gtol=cfg['gtol'], step_budget=cfg['step_budget'])
    log('bank + operators + cold start built')

    # ------------------------------------------------------------------- arms ----
    o = mesh.ops[M]
    man = json.loads(cfg_path(cfg['rules_manifest']).read_text())['rules'] if cfg.get('rules_manifest') else {}
    Glat = qlat.lattice_bank(mdl, Rp)
    arms = {}
    rule_rec = {}
    for spec in cfg['arms']:
        t1 = time.perf_counter()
        X, w = Q.offmesh_rule(spec['rule'])
        hsh = rule_sha(X, w)
        rule_rec[spec['name']] = dict(rule=spec['rule'], m=int(len(w)), sha256=hsh, weight_sum=float(np.sum(w)),
                                      manifest=man.get(spec['rule']))
        d = Q.offmesh_data(mdl, Rp, X, w, L, o['kx'], o['ky'], 'point')
        data = dict(mesh.base(SETTING), **d)
        fq, dec = Q.make_linear_query('point', Rp, L, DT, trust, step_budget=cfg['step_budget'], gtol=cfg['gtol'])
        fl, _ = qlat.make_linear_query_lat('point', Rp, L, DT, trust, step_budget=cfg['step_budget'], gtol=cfg['gtol'])
        dlat = dict(data, Glat=Glat)
        arms[spec['name']] = dict(spec=spec, m=int(len(w)), data=data,
                                  call=(lambda u, nu, fq=fq, data=data: fq(u, nu, data, cold)),
                                  call_lat=(lambda u, nu, fl=fl, d=dlat: fl(u, nu, d, cold)),
                                  dcall=(lambda W, dec=dec, data=data: dec(W, data)),
                                  build_seconds=time.perf_counter() - t1)
    rep['setup']['rules'] = rule_rec
    # G-R (amendment A3): in-job rule (generated as in October) vs the committed local arrays, within round-off
    ra = np.load(cfg_path(cfg['rules_arrays']))
    for k, v in rule_rec.items():
        X, w = Q.offmesh_rule(v['rule'])
        X0, w0 = ra[f"{v['rule']}_X"], ra[f"{v['rule']}_w"]
        v['bitwise'] = v['manifest'] == v['sha256']
        v['max_dX'] = float(np.abs(X - X0).max()) if X.shape == X0.shape else None
        v['max_dw_rel'] = float(np.abs(w - w0).max() / np.abs(w0).max()) if w.shape == w0.shape else None
    bad = [k for k, v in rule_rec.items() if v['max_dX'] is None or v['max_dX'] > 1e-14 or v['max_dw_rel'] > 1e-12]
    rep['gates']['G_R'] = dict(checked=len(rule_rec), bitwise=sum(v['bitwise'] for v in rule_rec.values()),
                               mismatched=bad, worst_dX=max(v['max_dX'] or 0 for v in rule_rec.values()),
                               worst_dw_rel=max(v['max_dw_rel'] or 0 for v in rule_rec.values()), passed=not bad)
    assert not bad, bad
    log(f'arms built: {len(arms)}')
    save()

    # --------------------------------------------------------------- accuracy ----
    conv = cfg['conv_arm']
    order = [conv] + [n for n in arms if n != conv]
    tcases = cases[:cfg['timing']['cases']]
    audit = set(cfg.get('audit_cases', []))
    KEEP_T = {}                       # (arm, case) -> f257 of the timing cases (timed-output check)
    COEF = {}
    keep = int(round(.05 / DT))
    for ci, c in enumerate(cases):
        ph = ph_all[c]
        u0 = jnp.asarray(Q.e.initial(L, ph))
        nu = float(ph[4])
        fconv = None
        n0 = META[(c, 'ST')]['n0']
        for name in order:
            arm = arms[name]
            t2 = time.perf_counter()
            v = arm['call'](u0, nu)
            jax.block_until_ready(v['fields'])
            secs = time.perf_counter() - t2
            fr = np.asarray(v['fields'][:, ::s256, ::s256])
            it, reason, rej = (np.asarray(v[k_]) for k_ in ('it', 'reason', 'rej'))
            row = dict(arm=name, rule=arm['spec']['rule'], family=arm['spec']['family'], m=arm['m'], case=c,
                       finite=bool(np.isfinite(fr).all()), seconds_first=secs, iterations_total=int(it.sum()),
                       iterations_max=int(it.max()), exits={REASONS[k_]: int(np.sum(reason == k_)) for k_ in REASONS},
                       rejected_total=int(rej.sum()), f257_sha256=QS.sha(fr),
                       steps=int(reason.size), nonconv_steps=int(np.isin(reason, NONCONV).sum()),
                       hard_fail=bool((reason == 3).any()))
            for kind in ('ST', 'S'):
                row[kind] = score(fr, c, kind)
            if name == conv:
                fconv = fr
            else:
                d_ = VD.tnorm(fr - fconv)[1:] / n0
                row['dist_conv'] = float(np.max(d_))
            rep['rows'].append(row)
            COEF[(name, c)] = np.asarray(v['internal'])[::keep]
            if c in tcases:
                KEEP_T[(name, c)] = fr
            if c in audit:
                np.savez_compressed(out / 'fields' / f'f257_{name}_c{c:03d}.npz', f257=fr, coef=COEF[(name, c)])
            del v
        log(f'CASE {ci} ({coh}{c}) done')
        if ci % 8 == 7:
            save()
    np.savez_compressed(out / 'fields' / 'coefficients.npz', **{f'{n}__c{c}': v for (n, c), v in COEF.items()})
    rep['phases'] = dict(accuracy=el())
    save()

    # ------------------------------------------------------------------- FOMs ----
    foms = F2.FOMs()
    fspecs = [(d['name'], fom_spec(d)) for d in cfg['foms']]
    FKEEP = {}
    PREP = {}
    for fname, sp in fspecs:
        t1 = time.perf_counter()
        for c in cases:
            ph = ph_all[c]
            x, prep_s = F2.prepare(sp, ph)
            F, info = foms.run(sp, x, float(ph[4]))
            F = np.asarray(jax.block_until_ready(F))
            ok, ev = F2.FOMs.converged(sp, info, F)
            row = dict(fom=fname, key=F2.key(sp), case=c, converged=ok, prepare_seconds=prep_s,
                       evidence={k: ev.get(k) for k in ('worst_res', 'max_newton', 'at_newton_limit', 'complete', 'finite', 'umax')})
            for kind in ('ST', 'S'):
                row[kind] = score(F, c, kind)
            rep['fom_rows'].append(row)
            if c in tcases:
                FKEEP[(fname, c)] = F
                PREP[(fname, c)] = x
            if c in audit:
                np.savez_compressed(out / 'fields' / f'fom_{fname}_c{c:03d}.npz', f257=F)
            del F, info
        log(f'FOM {fname} done ({time.perf_counter() - t1:.1f}s)')
        save()
    rep['phases']['foms'] = el()

    # ----------------------------------------------------------------- timing ----
    tc = cfg['timing']
    rng = np.random.default_rng(int(tc.get('seed', 20261105)))
    inputs = [(jnp.asarray(Q.e.initial(L, ph_all[c])), float(ph_all[c][4]), c) for c in tcases]
    spd = dict(fspecs)
    Wd = [jnp.asarray(.1 * np.random.default_rng(x).normal(size=(6, Rp))) for x in range(len(tcases))]

    def fn_of(sj, x):
        kind, name = sj
        u0, nu, c = inputs[x]
        if kind == 'rom_lat':
            return lambda: arms[name]['call_lat'](u0, nu)
        if kind == 'rom_full':
            return lambda: arms[name]['call'](u0, nu)
        if kind == 'decode':
            return lambda: arms[name]['dcall'](Wd[x])
        return foms.callable(spd[name], PREP[(name, c)], nu, 'lattice')

    def check(sj, x, o):
        kind, name = sj
        c = inputs[x][2]
        if kind == 'rom_full':
            f, ref = np.asarray(o['fields'][:, ::s256, ::s256]), KEEP_T[(name, c)]
        elif kind == 'rom_lat':
            f, ref = np.asarray(o['fields']), KEEP_T[(name, c)]
        elif kind == 'fom':
            f, ref = np.asarray(o), FKEEP[(name, c)]
        else:
            return None
        return float(np.abs(f - ref).max() / max(np.abs(ref).max(), 1e-300))

    # G-lat (A1.2): the lattice-contract query equals the vendored query on every timing case
    glat = []
    for name in arms:
        for x in range(len(tcases)):
            u0, nu, c = inputs[x]
            o1, o2 = arms[name]['call'](u0, nu), arms[name]['call_lat'](u0, nu)
            i1, i2 = np.asarray(o1['internal']), np.asarray(o2['internal'])
            glat.append(max(float(np.abs(i1 - i2).max() / max(np.abs(i1).max(), 1e-300)),
                            check(('rom_lat', name), x, o2)))
            del o1, o2
    rep['gates']['G_lat'] = dict(compared=len(glat), max_rel=max(glat), passed=bool(max(glat) <= 1e-12))
    log(f"G_lat {rep['gates']['G_lat']}")

    def panel(subjects, reps, label):
        for sj in subjects:                   # compile + warm every subject on every timing case
            for x in range(len(tcases)):
                jax.block_until_ready(fn_of(sj, x)())
        QS.burn(tc.get('initial_burn', 30.0))
        inv = []
        for r_ in range(reps):
            ordr = [(j, x) for j in range(len(subjects)) for x in range(len(tcases))]
            rng.shuffle(ordr)
            for j, x in ordr:
                sj = subjects[j]
                fn = fn_of(sj, x)
                QS.burn(tc['burn'])
                t1 = time.perf_counter()
                o_ = fn()
                jax.block_until_ready(o_)
                secs = time.perf_counter() - t1
                inv.append(dict(panel=label, kind=sj[0], name=sj[1], case=inputs[x][2], rep=r_, seconds=secs,
                                diff=check(sj, x, o_)))
                del o_
            log(f'timing {label} rep {r_} done')
        summ = {}
        for kind, name in subjects:
            ts = np.array([d['seconds'] for d in inv if d['kind'] == kind and d['name'] == name])
            q1, q3 = np.quantile(ts, [.25, .75])
            blocks = [np.median([d['seconds'] for d in inv if d['kind'] == kind and d['name'] == name and d['rep'] == r_])
                      for r_ in range(reps)]
            med = float(np.median(ts))
            summ[f'{kind}|{name}'] = dict(median_ms=1e3 * med, iqr_ms=1e3 * float(q3 - q1), min_ms=1e3 * float(ts.min()),
                                          max_ms=1e3 * float(ts.max()), n=int(len(ts)),
                                          outliers=int(np.sum((ts < q1 - 1.5 * (q3 - q1)) | (ts > q3 + 1.5 * (q3 - q1)))),
                                          block_drift=float(max(max(blocks) / med, med / min(blocks))))
        return inv, summ

    main_subj = [('rom_lat', n) for n in arms] + [('fom', fn) for fn, _ in fspecs]
    inv, summ = panel(main_subj, tc['reps'], 'main')
    rep['timing'] = dict(cases=tcases, reps=tc['reps'], burn=tc['burn'], initial_burn=tc.get('initial_burn', 30.0),
                         invocations=inv, summary=summ, median_ms={k: v['median_ms'] for k, v in summ.items()})
    save()
    # in-job selection (A1.3 / A1.7), same code as the report; then the confirmation panel
    sel = PL.cheapest_plateau_rule(rep)
    rep['timing']['selected_in_job'] = sel
    conf_arms = [conv] + ([sel['arm']] if sel and sel['arm'] != conv else [])
    conf_subj = ([('rom_lat', n) for n in conf_arms] + [('rom_full', n) for n in conf_arms] + [('decode', conv)] +
                 [('fom', fn) for fn, _ in fspecs])
    inv2, summ2 = panel(conf_subj, tc['confirm_reps'], 'confirm')
    rep['timing'].update(confirm_invocations=inv2, confirm_summary=summ2)
    diffs = [d['diff'] for d in inv + inv2 if d['diff'] is not None]
    worst_drift = max(v['block_drift'] for v in list(summ.values()) + list(summ2.values()))
    rep['gates']['G_time'] = dict(max_output_diff=max(diffs), worst_block_drift=worst_drift,
                                  outputs_pass=bool(max(diffs) <= 1e-12), drift_pass=bool(worst_drift <= 1.10))
    rep['phases']['timing'] = el()
    log(f"timing gates {rep['gates']['G_time']}; selected {sel}")
    save()

    # ------------------------------------------------------------ in-job gates ----
    rows = rep['rows']
    rep['gates']['G_fin'] = dict(nonfinite=[(r['arm'], r['case']) for r in rows if not r['finite']])
    rep['gates']['G_fom'] = dict(nonconverged=sorted({r['fom'] for r in rep['fom_rows'] if not r['converged']}))
    a_ = [c for c in audit if c in cases]
    rs = []
    for c in a_:                              # refstore.score itself on the saved audit fields == in-job score
        for name in arms:
            fr = np.load(out / 'fields' / f'f257_{name}_c{c:03d}.npz')['f257']
            s = RS.score(fr, 2, coh, c, 'ST')
            mine = next(r['ST']['e'] for r in rows if r['arm'] == name and r['case'] == c)
            rs.append(abs(s['e'] - mine) / max(abs(s['e']), 1e-300))
    rep['gates']['G_score_vs_refstore'] = dict(compared=len(rs), max_rel=max(rs) if rs else None,
                                               passed=bool(rs and max(rs) <= 1e-12))
    rep['elapsed_seconds'] = el()
    rep['checkpoint_sha256_after'] = sha_file(cfg_path(cfg['checkpoint']))
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('PLATEAU2D COMPLETE', flush=True)


def cfg_path(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


if __name__ == '__main__':
    main()
