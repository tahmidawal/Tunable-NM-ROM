"""Independent NumPy audit, acceptance decision and summary of one collected quadrature-study attempt
(DESIGN.md G1-G8, B1, B3, B4; Codex design audits 1 and 2).

    python audit_qs.py runs/<attempt>/archive [--refs runs/<refattempt>/archive/output] [--rho-sample 12]
                       --out checks/<attempt>-summary.json

No JAX. Exits non-zero if any applicable gate fails (the summary is written either way). Gate applicability depends on
the job's role (config 'role': dev / test) and mesh:
  G3 parity        dev jobs at 256, 1024, 4096 (every registered target must be present and pass)
  G6 rollout       dev job at 1024 (the Gauss-768 rollout must exist for every setting)
  G7 controls      every job (both controls, every setting, B3 failed AND continuum rho > 0.116)
Required evidence (G8) is enumerated from the rows: for each audit case every arm that ran must have its saved field
file; full fields where the config says so. Missing evidence or non-finite metrics of a required arm fail the gate.
Injected controls are pushed through the same validation functions that make the acceptance decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
import npbank  # noqa: E402
import hari_quadrature as HQ  # noqa: E402  (pure NumPy / SciPy)

CKPT = HERE.parents[1] / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
EXPECTED = {'test64': 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'}
B3 = dict(acc=5e-4, fast=2e-3, head=5e-3)        # DESIGN B3 as fractions (0.05 % / 0.2 % / 0.5 %)
SETTINGS = dict(acc=(384, 1536), fast=(128, 512), head=(512, 64))
TOL_ERR = 1e-10                                   # absolute tolerance of recomputed errors (fractions)
TOL_RHO = 1e-6                                    # relative tolerance of recomputed rho
FIB_INDEX = {987: 16, 1597: 17, 2584: 18, 4181: 19, 6765: 20, 10946: 21, 17711: 22}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def evolved(per_time):
    return float(np.max(np.asarray(per_time)[1:]))


def rel_rows(F, G, n0):
    return [float(np.linalg.norm(a - b)) / n0 for a, b in zip(F, G)]


def modes_lean(L, M):
    k = np.arange(1, L)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)
    ind = np.argsort(lam.ravel(), kind='stable')[:M]
    return kx.ravel()[ind], ky.ravel()[ind]


def np_rule(name):
    if name.startswith('gauss'):
        return HQ.gauss_tensor(int(name[5:]))
    if name.startswith('fib'):
        return HQ.fibonacci_lattice(FIB_INDEX[int(name[3:])], seed=0)
    if name.startswith('sobol'):
        return HQ.sobol(int(name[5:]), scramble=True, seed=0)
    if name.startswith('smolyak'):
        return HQ.smolyak(int(name[7:]), 'cc')
    raise ValueError(name)


def np_offmesh_value(P, T, C, X, w, L, kx, ky, form='point', chunk=8192):
    """L sum_q w_q psi(x_q) u (u_x+u_y) (or the flux form) for every row c of C (independent NumPy path)."""
    out = np.zeros((len(C), len(kx)))
    a, b = kx[None].astype(float), ky[None].astype(float)
    for s in range(0, len(X), chunk):
        Xs, ws = X[s:s + chunk], w[s:s + chunk]
        v, gx, gy = npbank.features_grad(P, Xs)
        Gq, Gs = v @ T, (gx + gy) @ T
        u, us = C @ Gq.T, C @ Gs.T
        x, y = Xs[:, 0:1], Xs[:, 1:2]
        if form == 'point':
            out += (u * us) @ (L * ws[:, None] * 2 * np.sin(a * np.pi * x) * np.sin(b * np.pi * y))
        else:
            pxy = 2 * np.pi * (a * np.cos(a * np.pi * x) * np.sin(b * np.pi * y) + b * np.sin(a * np.pi * x) * np.cos(b * np.pi * y))
            out -= (.5 * u * u) @ (L * ws[:, None] * pxy)
    return out


def np_upwind(c, xm, xp, ym, yp, L):
    return c * L * (np.where(c > 0, c - xm, xp - c) + np.where(c > 0, c - ym, yp - c))


def np_mesh_target(P, T, C, L, kx, ky):
    """Phi^T a_h(G c) on the full L-mesh (NumPy upwind stencil of engines.spatial)."""
    x = np.arange(1, L) / L
    XY = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    G = np.concatenate([npbank.features_grad(P, XY[s:s + 8192])[0] for s in range(0, len(XY), 8192)]) @ T
    sx, sy = np.sin(np.pi * x[:, None] * kx), np.sin(np.pi * x[:, None] * ky)
    out = []
    for c in C:
        p = np.pad((G @ c).reshape(L - 1, L - 1), 1)
        adv = np_upwind(p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:], L)
        out.append((2. / L) * np.sum(sx * (adv @ sy), axis=0))
    return np.array(out)


def np_lattice_value(P, T, C, L, s_, kx, ky):
    """lat<s> mesh rule: upwind stencil at the (s-1)^2 sub-lattice nodes, weights (L/s)^2, Phi rows (NumPy)."""
    k = np.arange(1, s_) * (L // s_)
    I, J = [v.ravel() for v in np.meshgrid(k, k, indexing='ij')]
    off = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])          # c, x+, x-, y+, y-
    pts = ((np.stack((I, J), 1)[:, None, :] + off[None]) / L).reshape(-1, 2)
    g5 = npbank.features_grad(P, pts)[0].reshape(len(I), 5, -1) @ T
    us = np.einsum('msr,nr->nms', g5, C)
    adv = np_upwind(us[..., 0], us[..., 2], us[..., 1], us[..., 4], us[..., 3], L)
    phi = (2. / L) * np.sin(np.pi * (I / L)[:, None] * kx) * np.sin(np.pi * (J / L)[:, None] * ky) * (L // s_) ** 2
    return adv @ phi


def check_errors(fr, row, tr, refs, saved, n0r, T, full):
    """The validation used for acceptance: recompute every recorded error of one (arm, case); returns the worst
    absolute discrepancy (inf if a required comparand is missing), the per-metric discrepancies and the sha check."""
    d = {'same_grid_restricted': abs(evolved(rel_rows(fr, tr, n0r)) - row['same_grid_restricted_evolved'])}
    for tag, rf in refs.items():
        d[f'ref_{tag}'] = (abs(evolved(rel_rows(fr, rf, n0r)) - row[f'ref_{tag}_evolved'])
                           if row.get(f'ref_{tag}_evolved') is not None else np.inf)
    for other in ('dense', 'gref'):
        k = f'vs_{other}_restricted_evolved'
        if k in row:
            d[f'vs_{other}'] = (abs(evolved(rel_rows(fr, saved[other]['f257'], n0r)) - row[k])
                                if other in saved and row[k] is not None else np.inf)
    if T is not None and full is not None:
        d['same_grid_full'] = abs(evolved(rel_rows(full, T['full'], float(np.linalg.norm(T['full'][0])))) - row['same_grid_evolved'])
    return max(d.values()), d, sha(fr) == row['restricted_sha256']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('archive')
    p.add_argument('--refs', default=None)
    p.add_argument('--rho-sample', type=int, default=12)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    arc = Path(a.archive)
    out_dir = arc / 'output'
    R = json.loads((out_dir / 'result.json').read_text())
    cfg = R['config']
    role = cfg.get('role', 'dev')
    L = int(R['mesh'])
    logs = ''.join(f.read_text() for f in (arc / 'logs').glob('*.out'))
    gates = {}
    gates['G1_backend'] = dict(passed=bool('jax_backend=gpu' in logs and 'precision=highest' in logs and
                                           R['backend'] == 'gpu' and R['complete']), gpu=R['gpu'], complete=R['complete'])
    coh_sha = {}
    ok2 = True
    for k, v in R['cohorts'].items():
        h = sha(np.array(v['physical']))
        coh_sha[k] = h
        ok2 &= h == v['physical_sha256'] and (k not in EXPECTED or h == EXPECTED[k] or bool(cfg.get('local_smoke_waives_cohort_hash')))
    gates['G2_cohort'] = dict(passed=bool(ok2), sha=coh_sha, cases=sorted({f"{x['cohort']}{x['case']}" for x in R['fom_rows']}))

    # ------------------------------------------------------------ aggregates ----
    rows = R['rows']
    agg = {}
    for r in rows:
        agg.setdefault(r['setting'], {}).setdefault(r['arm'], []).append(r)

    def stats(rs, key):
        sub = [x for x in rs if key in x]
        if not sub:
            return None
        v = [x[key] if x[key] is not None else np.inf for x in sub]      # None = non-finite -> inf
        i = int(np.argmax(v))
        return dict(n=len(v), worst=float(max(v)), median=float(np.median(v)), argmax=f"{sub[i]['cohort']}{sub[i]['case']}",
                    nonfinite=int(sum(not np.isfinite(t) for t in v)))

    keys = ['same_grid_evolved', 'same_grid_restricted_evolved', 'ref_ST_evolved', 'ref_S_evolved', 'vs_dense_evolved',
            'vs_dense_restricted_evolved', 'vs_gref_evolved', 'vs_gref_restricted_evolved']
    arms = {}
    for s, d in agg.items():
        for name, rs in d.items():
            ent = dict(kind=rs[0]['kind'], rule=rs[0]['rule'], m=rs[0]['m'], gtol=rs[0]['gtol'],
                       control=name.startswith('ctrl'))
            for scope in ['all'] + sorted({x['cohort'] for x in rs}):
                sub = rs if scope == 'all' else [x for x in rs if x['cohort'] == scope]
                e = {k: stats(sub, k) for k in keys}
                e['iterations_total_mean'] = float(np.mean([x['iterations_total'] for x in sub]))
                e['iterations_total_max'] = int(max(x['iterations_total'] for x in sub))
                e['iterations_max_per_step'] = int(max(x['iterations_max'] for x in sub))
                e['exits'] = {k: int(sum(x['exits'][k] for x in sub)) for k in sub[0]['exits']}
                e['rejected_total'] = int(sum(x['rejected_total'] for x in sub))
                e['finite'] = bool(all(x['finite'] for x in sub))
                e['cases'] = len(sub)
                e['seconds_first_median'] = float(np.median([x['seconds_first'] for x in sub]))
                ent[scope] = e
            arms.setdefault(s, {})[name] = ent

    # B1 on the dense-matched case set (exact equality required), B3
    for s, d in agg.items():
        dense_rows = d.get('dense', [])
        dcases = {(x['cohort'], x['case']) for x in dense_rows}
        dref = {(x['cohort'], x['case']): x.get('ref_ST_evolved') for x in dense_rows}
        for name, rs in d.items():
            sub = {(x['cohort'], x['case']): x for x in rs}
            if dcases and dcases <= set(sub):
                va = [sub[k_].get('ref_ST_evolved') for k_ in dcases]
                vd = [dref[k_] for k_ in dcases]
                if all(v is not None and np.isfinite(v) for v in va + vd):
                    wa, wd = max(va), max(vd)
                    arms[s][name]['B1'] = dict(cases=len(dcases), worst_ST=wa, dense_worst_ST=wd, diff_pp=100 * (wa - wd),
                                               passed=bool(abs(wa - wd) <= max(2e-4, .02 * wd)))
                else:
                    arms[s][name]['B1'] = dict(cases=len(dcases), passed=False, reason='missing or non-finite')
            if name in ('gref', 'dense'):
                continue
            vk = 'vs_gref_evolved' if rs[0]['kind'] in ('point', 'flux') else 'vs_dense_evolved'
            v = [x[vk] if x.get(vk) is not None else np.inf for x in rs if vk in x]
            if v:
                arms[s][name]['B3'] = dict(metric=vk, worst=float(max(v)), bar=B3[s], cases=len(v),
                                           passed=bool(max(v) <= B3[s]))

    # ------------------------------------------------------------- G3 parity ----
    pt = json.loads((HERE / 'configs/parity_targets.json').read_text()).get(str(L), {})
    if role == 'dev' and pt and 'dev6' in R['cohorts'] and not cfg.get('case_subset'):
        par = {}
        for s, (arm, target) in pt.items():
            rs = [x for x in agg.get(s, {}).get(arm, []) if x['cohort'] == 'dev6']
            if len(rs) != 6:
                par[s] = dict(arm=arm, passed=False, reason=f'{len(rs)} dev6 rows')
                continue
            got = 100 * max(x['same_grid_evolved'] for x in rs)
            par[s] = dict(arm=arm, target_pct=target, got_pct=got, rel=abs(got - target) / target,
                          passed=bool(abs(got - target) / target <= 1e-6))
        gates['G3_parity'] = dict(passed=bool(all(v['passed'] for v in par.values()) and set(par) == set(pt)), arms=par)

    # --------------------------------------------------------------- G4 / FOM ----
    fr_ = R['fom_rows']
    ntol = cfg['fom']['truth']['ntol']
    gates['G4_truth_converged'] = dict(passed=bool(all(x['max_relative_residual'] <= ntol * (1 + 1e-9) for x in fr_)),
                                       worst=float(max(x['max_relative_residual'] for x in fr_)), truth=cfg['fom']['truth']['name'])
    fom = {}
    for scope in ['all'] + sorted({x['cohort'] for x in fr_}):
        sub = [x for x in fr_ if scope in ('all', x['cohort'])]
        fom[scope] = {k: stats(sub, k) for k in ('ref_ST_evolved', 'ref_S_evolved')}

    # ------------------------------------------------------- G5 references ----
    refdir = Path(a.refs) if a.refs else (arc.parent.parent / Path(cfg['refs']).parent.name / 'archive' / 'output'
                                          if cfg.get('refs') else None)
    rr = None
    if refdir is not None and (refdir / 'result.json').exists():
        rr = json.loads((refdir / 'result.json').read_text())
        need = {(x['cohort'], x['case']) for x in fr_}
        have = {(c['cohort'], c['case'], c['ref']): c for c in rr['cases']}
        ok5 = bool(rr.get('complete') and rr.get('all_accepted') and int(rr['config']['mesh']) == 8192 and
                   all(have.get((co, ca, t), {}).get('accepted') for co, ca in need for t in ('ST', 'S')))
        gates['G5_references'] = dict(passed=ok5, job_id=rr.get('job_id'), cases=len(rr['cases']),
                                      worst_residual=float(max(c['max_relative_residual'] for c in rr['cases'])))
        mism = []
        for tag, d in R.get('references', {}).items():
            for key_, v in d.items():
                f = refdir / Path(v['file']).name
                if not f.exists() or sha(np.load(f)['f257']) != v['f257_sha256']:
                    mism.append(key_)
        nref = sum(len(d) for d in R.get('references', {}).values())
        gates['G5_reference_identity'] = dict(passed=bool(not mism and nref == 2 * len(need)), checked=nref, mismatches=mism)
    else:
        refdir = None
        gates['G5_references'] = dict(passed=False, reason='no references')

    # --------------------------------------------------------------- G6 ----
    g6 = {k: v for k, v in R['gates'].items() if k.startswith('continuum_target')}
    gates['G6_continuum_target'] = dict(passed=bool(g6 and len(g6) == len(R['rho']) and all(v['passed'] for v in g6.values())),
                                        detail=g6)
    if role == 'dev' and L == 1024:
        g6r = {}
        for s in arms:
            rs = agg[s].get('gref_check', [])
            v = [x.get('vs_gref_evolved') for x in rs]
            ok_ = len(rs) == 6 and all(t is not None for t in v)
            g6r[s] = dict(cases=len(rs), worst_vs_gref=float(max(v)) if ok_ else None, bar=.1 * B3[s],
                          passed=bool(ok_ and max(v) <= .1 * B3[s]))
        gates['G6_rollout_convergence'] = dict(passed=bool(g6r and all(v['passed'] for v in g6r.values())), detail=g6r)

    # --------------------------------------------------------------- G7 ----
    g7 = {}
    for s in arms:
        for ctrl, rule in (('ctrl_smolyak8', 'smolyak8'), ('ctrl_gauss8', 'gauss8')):
            b3 = arms[s].get(ctrl, {}).get('B3', {})
            rho = R['rho'].get(s, {}).get('rules', {}).get(rule, {}).get('cont', {}).get('max')
            g7[f'{s}/{ctrl}'] = dict(B3_failed=(b3.get('passed') is False), rho_cont_max=rho,
                                     rho_failed=bool(rho is not None and rho > .116))
    gates['G7_controls_fail'] = dict(passed=bool(g7 and len(g7) == 2 * len(arms) and
                                                 all(v['B3_failed'] and v['rho_failed'] for v in g7.values())), detail=g7)

    # ------------------------------------------------- G8 NumPy recompute ----
    ac = cfg['audit']
    coh = ac['cohort']
    rec, missing, worst, sha_ok = [], [], 0., True
    fullspec = lambda s, name, c: name in ac.get('full_arms', []) and (
        L <= ac.get('full_max_mesh', 256) or (L <= ac.get('full_case0_max_mesh', 0) and c == ac['cases'][0] and s == 'acc'))
    loaded = {}
    for c in ac['cases']:
        if not any(x['cohort'] == coh and x['case'] == c for x in fr_):
            continue                                   # audit case not part of this job (smoke subsets)
        tf = out_dir / f'audit_truth_{coh}{c}.npz'
        if not tf.exists():
            missing.append(tf.name)
            continue
        T = np.load(tf)
        tr = T['f257']
        refs = {}
        for tag in ('ST', 'S'):
            f = (refdir / f'ref_{tag}_{coh}_{c:03d}.npz') if refdir else None
            if f is not None and f.exists():
                refs[tag] = np.load(f)['f257']
        n0r = float(np.linalg.norm(tr[0]))
        for s in arms:
            saved = {}
            for x in [x for x in rows if x['setting'] == s and x['cohort'] == coh and x['case'] == c]:
                f = out_dir / f"audit_{s}_{x['arm']}_{coh}{c}.npz"
                if not f.exists():
                    missing.append(f.name)
                    continue
                saved[x['arm']] = (np.load(f), x)
            savedf = {k: v[0] for k, v in saved.items()}
            for name, (z, row) in saved.items():
                need_full = fullspec(s, name, c)
                if need_full and ('full' not in z.files or 'full' not in T.files):
                    missing.append(f'{s}:{name}:{coh}{c}:full')
                full = z['full'] if need_full and 'full' in z.files else None
                w_, d_, ok_ = check_errors(z['f257'], row, tr, refs, savedf, n0r, T if full is not None else None, full)
                rec.append(dict(setting=s, arm=name, case=c, worst=w_, sha_ok=ok_, **{k: float(v) for k, v in d_.items()}))
                worst, sha_ok = max(worst, w_), sha_ok and ok_
            loaded[(s, c)] = (saved, tr, refs, n0r)
    gates['G8_error_recompute'] = dict(passed=bool(rec and not missing and worst <= TOL_ERR and sha_ok),
                                       checked=len(rec), missing=missing[:20], n_missing=len(missing), worst_abs_diff=float(worst))

    # injected controls pushed through check_errors (the acceptance path); two real audit cases for the swap
    ctl = {}
    s0 = next(iter(arms))
    cs_ = [c for c in ac['cases'] if (s0, c) in loaded]
    if cs_:
        sv0, tr0, rf0, n00 = loaded[(s0, cs_[0])]
        nm = 'lat64' if 'lat64' in sv0 else next(iter(sv0))
        z, row = sv0[nm]
        sv0f = {k: v[0] for k, v in sv0.items()}
        w_p, _, okp = check_errors(z['f257'] * (1 + 1e-6), row, tr0, rf0, sv0f, n00, None, None)
        ctl['perturbed_field_rejected'] = bool(not okp or w_p > TOL_ERR)
        if len(cs_) >= 2:
            sv1 = loaded[(s0, cs_[1])][0]
            w_s, _, oks = check_errors(sv1[nm][0]['f257'], row, tr0, rf0, sv0f, n00, None, None)
            ctl['swapped_case_rejected'] = bool(not oks or w_s > TOL_ERR)

    # NumPy rho recompute: rule families + argmax states (continuum target = NumPy Gauss-640), mesh rules at 256^2
    P = npbank.load(CKPT)
    Trot = np.asarray(np.load(HERE / 'inputs/rotation_R512.npz')['T'])
    Xg, wg = HQ.gauss_tensor(int(cfg['gref'][5:]))
    rho_chk = []
    fam = ['gauss64', 'gauss128', 'fib6765', 'fib17711', 'sobol4096', 'smolyak8', 'flux_gauss64']
    for s, (Rp, M) in SETTINGS.items():
        if s not in R['rho']:
            continue
        pop = np.load(out_dir / f'population_{s}.npz')
        C = pop['C']
        per = np.load(out_dir / f'rho_per_state_{s}.npz')
        rules = [r for r in fam if r in per.files]
        pick = sorted({int(np.argmax(per[r][0])) for r in rules} |
                      set(np.random.default_rng(7).choice(len(C), min(a.rho_sample, len(C)), replace=False).tolist()))
        kx, ky = modes_lean(L, M)
        Tp = Trot[:, :Rp]
        Cs = C[pick]
        tgt = np_offmesh_value(P, Tp, Cs, Xg, wg, L, kx, ky)
        for rule in rules:
            form = 'flux' if rule.startswith('flux_') else 'point'
            X, w = np_rule(rule.replace('flux_', ''))
            v = np_offmesh_value(P, Tp, Cs, np.asarray(X), np.asarray(w), L, kx, ky, form)
            rnp = np.linalg.norm(v - tgt, axis=1) / np.linalg.norm(tgt, axis=1)
            rj = per[rule][0][pick]
            rho_chk.append(dict(setting=s, rule=rule, target='continuum', states=len(pick),
                                max_rel_diff=float(np.max(np.abs(rnp - rj) / rj))))
        if L <= 256:
            mt = np_mesh_target(P, Tp, Cs, L, kx, ky)
            for rule, V, tg, idx in (('dense', mt, tgt, 0), ('lat64', np_lattice_value(P, Tp, Cs, L, 64, kx, ky), mt, 1)):
                if rule not in per.files:
                    continue
                rnp = np.linalg.norm(V - tg, axis=1) / np.linalg.norm(tg, axis=1)
                rj = per[rule][idx][pick]
                rho_chk.append(dict(setting=s, rule=rule, target='continuum' if idx == 0 else 'mesh', states=len(pick),
                                    max_rel_diff=float(np.max(np.abs(rnp - rj) / np.maximum(rj, 1e-300)))))
    gates['G8_rho_recompute'] = dict(passed=bool(rho_chk and max(x['max_rel_diff'] for x in rho_chk) <= TOL_RHO),
                                     worst=float(max([x['max_rel_diff'] for x in rho_chk] or [np.inf])), detail=rho_chk)

    # timing: recompute medians, timed-output identity, solve times, same-GPU B4
    inv = R['timing'].get('invocations', [])
    key = lambda d: f"{d['setting']}|{d['kind']}|{d['mesh']}|{d['name']}"

    def medians(invs, scale=1.):
        m = {}
        for d in invs:
            m.setdefault(key(d), []).append(d['seconds'] * scale)
        return {k: 1e3 * float(np.median(v)) for k, v in m.items()}

    def timing_ok(med_):
        rec_ = R['timing'].get('median_ms', {})
        return bool(med_ and set(med_) == set(rec_) and max(abs(med_[k] - rec_[k]) / rec_[k] for k in med_) <= 1e-12)

    med = medians(inv)
    gates['G8_timing_recompute'] = dict(passed=timing_ok(med), subjects=len(med))
    if inv:
        k0 = key(inv[0])
        ctl['time_x1.2_rejected'] = not timing_ok({**med, k0: medians([d for d in inv if key(d) == k0], 1.2)[k0]})
    need_ctl = 3 if len(cs_) >= 2 else 2
    gates['G8_injected_controls'] = dict(passed=bool(len(ctl) == need_ctl and all(ctl.values())), detail=ctl)
    rom_inv = [d for d in inv if d['kind'] == 'rom']
    fom_inv = [d for d in inv if d['kind'] == 'fom']
    bad_w = sum(not d.get('output_matches_warmup', False) for d in rom_inv)
    bad_p = sum(not d['output_matches_phase1'] for d in rom_inv if 'output_matches_phase1' in d)
    gates['G8_timed_outputs'] = dict(passed=bool(rom_inv and bad_w == 0 and bad_p == 0 and all(d['converged'] for d in fom_inv)),
                                     rom_checked=len(rom_inv), warmup_mismatch=bad_w, phase1_mismatch=bad_p,
                                     fom_checked=len(fom_inv), fom_unconverged=sum(not d['converged'] for d in fom_inv))
    timing = dict(median_ms=med, solve_ms={}, query_ms={}, decode_ms={}, fom={})
    for k, v in med.items():
        st_, kind, mesh_, name = k.split('|')
        if kind == 'rom' and f'{st_}|decode|{mesh_}|None' in med:
            timing['solve_ms'][f'{st_}|{mesh_}|{name}'] = v - med[f'{st_}|decode|{mesh_}|None']
            timing['query_ms'][f'{st_}|{mesh_}|{name}'] = v
        if kind == 'decode':
            timing['decode_ms'][f'{st_}|{mesh_}'] = v
    for d in fom_inv:
        e_ = timing['fom'].setdefault(d['name'], dict(worst_same_grid_evolved=0., invocations=0))
        e_['worst_same_grid_evolved'] = max(e_['worst_same_grid_evolved'], d['same_grid_evolved'])
        e_['invocations'] += 1
    for nm_ in timing['fom']:
        timing['fom'][nm_]['median_ms'] = 1e3 * float(np.median([d['seconds'] for d in fom_inv if d['name'] == nm_]))
    b4 = {}
    meshes = sorted({int(k.split('|')[1]) for k in timing['solve_ms']})
    lo_mesh = min(meshes) if meshes else None
    if lo_mesh is not None and L in meshes and L > lo_mesh:
        for k, v in timing['solve_ms'].items():
            st_, mesh_, name = k.split('|')
            if int(mesh_) == L:
                lo = timing['solve_ms'].get(f'{st_}|{lo_mesh}|{name}')
                if lo is not None:
                    valid = bool(np.isfinite(v) and np.isfinite(lo) and v > 0 and lo > 0)
                    b4[f'{st_}|{name}'] = dict(solve_ms={m_: timing['solve_ms'].get(f'{st_}|{m_}|{name}') for m_ in meshes},
                                               low_mesh=lo_mesh, ratio=(v / lo) if valid else None,
                                               passed=bool(valid and .8 <= v / lo <= 1.25))
    timing['B4_same_gpu'] = b4

    # -------------------------------------------- question (iv): worst states ----
    iv = {}
    for s in R['rho']:
        per = np.load(out_dir / f'rho_per_state_{s}.npz')
        labels = [tuple(x.split('|')) for x in np.load(out_dir / f'population_{s}.npz')['labels']]
        cases = sorted({(x[0], int(x[1])) for x in labels})
        cidx = {cc: [j for j, x in enumerate(labels) if (x[0], int(x[1])) == cc] for cc in cases}
        wv = np.array([R['cohorts'][cc[0]]['physical'][cc[1]][2] for cc in cases])
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)
        ent = {}
        for rule in per.files:
            rc = per[rule][0]
            i = int(np.argmax(rc))
            co, c, k = labels[i][0], int(labels[i][1]), int(labels[i][2])
            ph = R['cohorts'][co]['physical'][c]
            cmax = np.array([rc[cidx[cc]].max() for cc in cases])
            top = np.argsort(rc)[::-1][:max(1, len(rc) // 100)]
            ent[rule] = dict(argmax=dict(cohort=co, case=c, k=k, width=ph[2], amplitude=ph[3], nu=ph[4], rho=float(rc[i])),
                             spearman_casemax_vs_width=(float(np.corrcoef(rk(cmax), rk(wv))[0, 1]) if len(cases) > 2 else None),
                             top1pct_share_k_le_5=float(np.mean([int(labels[j][2]) <= 5 for j in top])),
                             top1pct_states=len(top))
        iv[s] = ent

    failed = [k for k, v in gates.items() if v.get('passed') is False]
    summ = dict(attempt=cfg['attempt'], role=role, job_id=R['job_id'], commit=R['commit'], gpu=R['gpu'], mesh=L,
                cohorts=coh_sha, elapsed_seconds=R.get('elapsed_seconds'), gates=gates, failed_gates=failed,
                accepted=not failed, arms=arms, fom=fom, rho=R['rho'], timing=timing, question_iv=iv,
                setup=R['setup'], result_sha256=hashlib.sha256((out_dir / 'result.json').read_bytes()).hexdigest(),
                references_job=(rr or {}).get('job_id'), rho_recompute=rho_chk)
    Path(a.out).write_text(json.dumps(summ, indent=1, default=float) + '\n')
    print(json.dumps({k: v.get('passed') for k, v in gates.items()}, indent=1))
    print('failed:', failed)
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
