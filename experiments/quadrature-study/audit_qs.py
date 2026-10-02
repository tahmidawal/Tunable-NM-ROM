"""Independent NumPy audit and summary of one collected quadrature-study attempt (DESIGN.md G1-G8, B1-B5).

    python audit_qs.py <archive_dir> --refs <ref archive output dir> [--rho-sample 12] --out checks/<attempt>-summary.json

No JAX. Recomputes from the saved files: restricted same-grid / refined-reference / vs-dense / vs-gref errors of the
audit cases; full-mesh same-grid errors where full fields were saved (256^2); continuum rho of sampled states with an
independent NumPy bank + gradient (npbank.py) and NumPy sine tests, against a NumPy Gauss-640 target; mesh rho at 256^2
with a NumPy upwind stencil on the full mesh; every timing median from raw invocations. Injected controls (a swapped
case, a 1e-6-perturbed field, a x1.2 time) must be detected. Writes the gate table and the aggregated metrics the
report reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
import npbank  # noqa: E402
import hari_quadrature as HQ  # noqa: E402  (pure NumPy / SciPy)

CKPT = HERE.parents[1] / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
TEST64_SHA = 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'
B3 = dict(acc=5e-4, fast=2e-3, head=5e-3)        # DESIGN B3 (fractions, i.e. 0.05 % / 0.2 % / 0.5 %)
SETTINGS = dict(acc=(384, 1536), fast=(128, 512), head=(512, 64))


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


def np_offmesh_value(P, T, C, X, w, L, kx, ky, form='point', chunk=8192):
    """L sum_q w_q psi(x_q) u (u_x+u_y) for every row c of C (independent NumPy path)."""
    out = np.zeros((len(C), len(kx)))
    for s in range(0, len(X), chunk):
        Xs, ws = X[s:s + chunk], w[s:s + chunk]
        v, gx, gy = npbank.features_grad(P, Xs)
        Gq, Gs = v @ T, (gx + gy) @ T
        u, us = C @ Gq.T, C @ Gs.T                     # (S, m)
        a, b = kx[None].astype(float), ky[None].astype(float)
        x, y = Xs[:, 0:1], Xs[:, 1:2]
        if form == 'point':
            psi = 2 * np.sin(a * np.pi * x) * np.sin(b * np.pi * y)
            out += (u * us) @ (L * ws[:, None] * psi)
        else:
            pxy = 2 * np.pi * (a * np.cos(a * np.pi * x) * np.sin(b * np.pi * y) + b * np.sin(a * np.pi * x) * np.cos(b * np.pi * y))
            out -= (.5 * u * u) @ (L * ws[:, None] * pxy)
    return out


def np_mesh_target(P, T, C, L, kx, ky):
    """Phi^T a_h(G c) on the full L-mesh with the upwind stencil of engines.spatial (NumPy)."""
    x = np.arange(1, L) / L
    XY = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    G = np.concatenate([npbank.features_grad(P, XY[s:s + 8192])[0] for s in range(0, len(XY), 8192)]) @ T
    sx, sy = np.sin(np.pi * x[:, None] * kx), np.sin(np.pi * x[:, None] * ky)
    out = []
    for c in C:
        u = (G @ c).reshape(L - 1, L - 1)
        p = np.pad(u, 1)
        cc, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
        adv = cc * L * (np.where(cc > 0, cc - xm, xp - cc) + np.where(cc > 0, cc - ym, yp - cc))
        out.append((2. / L) * np.sum(sx * (adv @ sy), axis=0))
    return np.array(out)


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
    L = int(R['mesh'])
    s256 = max(1, L // 256)
    logs = ''.join(f.read_text() for f in (arc / 'logs').glob('*.out'))
    gates = {}
    gates['G1_backend'] = dict(passed=bool('jax_backend=gpu' in logs and R['backend'] == 'gpu' and R['complete']),
                               gpu=R['gpu'], complete=R['complete'])
    coh_sha = {k: v['physical_sha256'] for k, v in R['cohorts'].items()}
    gates['G2_cohort'] = dict(passed=bool(coh_sha.get('test64', TEST64_SHA) == TEST64_SHA), sha=coh_sha)

    # ------------------------------------------------------------ aggregates ----
    rows = R['rows']
    agg = {}
    for r in rows:
        agg.setdefault(r['setting'], {}).setdefault(r['arm'], []).append(r)

    def stats(rs, key):
        v = [x[key] for x in rs if x.get(key) is not None]
        if not v:
            return None
        return dict(n=len(v), worst=float(max(v)), median=float(np.median(v)), argmax=f"{rs[int(np.argmax([x.get(key, -1) if x.get(key) is not None else -1 for x in rs]))]['cohort']}{rs[int(np.argmax([x.get(key, -1) if x.get(key) is not None else -1 for x in rs]))]['case']}")

    keys = ['same_grid_evolved', 'same_grid_restricted_evolved', 'ref_ST_evolved', 'ref_S_evolved', 'vs_dense_evolved',
            'vs_dense_restricted_evolved', 'vs_gref_evolved', 'vs_gref_restricted_evolved']
    arms = {}
    for s, d in agg.items():
        for name, rs in d.items():
            ent = dict(kind=rs[0]['kind'], rule=rs[0]['rule'], m=rs[0]['m'], gtol=rs[0]['gtol'])
            for scope in ['all'] + sorted({x['cohort'] for x in rs}):
                sub = rs if scope == 'all' else [x for x in rs if x['cohort'] == scope]
                e = {k: stats(sub, k) for k in keys}
                e['iterations_total_mean'] = float(np.mean([x['iterations_total'] for x in sub]))
                e['iterations_max'] = int(max(x['iterations_max'] for x in sub))
                e['exits'] = {k: int(sum(x['exits'][k] for x in sub)) for k in sub[0]['exits']}
                e['rejected_total'] = int(sum(x['rejected_total'] for x in sub))
                e['finite'] = bool(all(x['finite'] for x in sub))
                e['cases'] = len(sub)
                ent[scope] = e
            arms.setdefault(s, {})[name] = ent

    # dense-matched comparison set for B1: the arm restricted to the cases where dense ran
    for s, d in agg.items():
        dcases = {(x['cohort'], x['case']) for x in d.get('dense', [])}
        dref = {(x['cohort'], x['case']): x.get('ref_ST_evolved') for x in d.get('dense', [])}
        for name, rs in d.items():
            sub = [x for x in rs if (x['cohort'], x['case']) in dcases and x.get('ref_ST_evolved') is not None]
            if sub and dref and all(v is not None for v in dref.values()):
                wa = max(x['ref_ST_evolved'] for x in sub)
                wd = max(dref.values())
                arms[s][name]['B1'] = dict(cases=len(sub), worst_ST=wa, dense_worst_ST=wd, diff_pp=100 * (wa - wd),
                                           passed=bool(abs(wa - wd) <= max(2e-4, .02 * wd)))
            vk = 'vs_gref_evolved' if rs[0]['kind'] in ('point', 'flux') else 'vs_dense_evolved'
            v = [x[vk] for x in rs if x.get(vk) is not None]
            if v:
                arms[s][name]['B3'] = dict(metric=vk, worst=float(max(v)), bar=B3[s], cases=len(v),
                                           passed=bool(max(v) <= B3[s]))

    # ------------------------------------------------------------- G3 parity ----
    pt = json.loads((HERE / 'configs/parity_targets.json').read_text()).get(str(L), {})
    par = {}
    for s, (arm, target) in pt.items():
        rs = [x for x in agg.get(s, {}).get(arm, []) if x['cohort'] == 'dev6']
        if rs:
            got = 100 * max(x['same_grid_evolved'] for x in rs)
            par[s] = dict(arm=arm, target_pct=target, got_pct=got, rel=abs(got - target) / target,
                          passed=bool(abs(got - target) / target <= 1e-6))
    gates['G3_parity'] = dict(passed=bool(par and all(v['passed'] for v in par.values())) if pt else None, arms=par)

    # --------------------------------------------------------------- G4 / FOM ----
    fr = R['fom_rows']
    ntol = cfg['fom']['truth']['ntol']
    gates['G4_truth_converged'] = dict(passed=bool(all(x['max_relative_residual'] <= ntol * (1 + 1e-9) for x in fr)),
                                       worst=float(max(x['max_relative_residual'] for x in fr)))
    fom = {}
    for scope in ['all'] + sorted({x['cohort'] for x in fr}):
        sub = fr if scope == 'all' else [x for x in fr if x['cohort'] == scope]
        fom[scope] = {k: stats(sub, k) for k in ('ref_ST_evolved', 'ref_S_evolved')}

    # ------------------------------------------------------- G5 references ----
    refdir = Path(a.refs) if a.refs else None
    if refdir is None and cfg.get('refs'):
        refdir = arc.parent.parent / Path(cfg['refs']).parent.name / 'archive' / 'output'
        a.refs = str(refdir)
    refdir = refdir or HERE / '__no_refs__'
    rr = json.loads((refdir / 'result.json').read_text()) if a.refs else dict(complete=False, cases=[dict(accepted=False, max_relative_residual=np.inf)])
    gates['G5_references'] = dict(passed=bool(rr.get('complete') and all(c['accepted'] for c in rr['cases'])),
                                  cases=len(rr['cases']), worst_residual=float(max(c['max_relative_residual'] for c in rr['cases'])))
    used = R.get('references', {})
    mism = []
    for tag, d in used.items():
        for key, v in d.items():
            f = refdir / Path(v['file']).name
            if not f.exists() or sha(np.load(f)['f257']) != v['f257_sha256']:
                mism.append(key)
    gates['G5_reference_identity'] = dict(passed=not mism and bool(used), checked=sum(len(d) for d in used.values()),
                                          mismatches=mism)

    # --------------------------------------------------------------- G6 ----
    g6 = {k: v for k, v in R['gates'].items() if k.startswith('continuum_target')}
    gates['G6_continuum_target'] = dict(passed=bool(g6 and all(v['passed'] for v in g6.values())), detail=g6)

    # --------------------------------------------------------------- G7 ----
    g7 = {}
    for s in arms:
        for ctrl in ('ctrl_smolyak8', 'ctrl_gauss8'):
            b3 = arms[s].get(ctrl, {}).get('B3', {})
            rule = {'ctrl_smolyak8': 'smolyak8', 'ctrl_gauss8': 'gauss8'}[ctrl]
            rho = R['rho'].get(s, {}).get('rules', {}).get(rule, {}).get('cont', {}).get('max')
            g7[f'{s}/{ctrl}'] = dict(B3_failed=(b3.get('passed') is False), rho_cont_max=rho,
                                     rho_failed=bool(rho is not None and rho > .116))
    gates['G7_controls_fail'] = dict(passed=bool(g7 and all(v['B3_failed'] and v['rho_failed'] for v in g7.values())),
                                     detail=g7)

    # ------------------------------------------------- G8 NumPy recompute ----
    rec = dict(errors=[], full=[], rho=[], timing=None, controls={})
    ac = cfg['audit']
    for c in ac['cases']:
        coh = ac['cohort']
        tf = out_dir / f'audit_truth_{coh}{c}.npz'
        if not tf.exists():
            continue
        T = np.load(tf)
        tr = T['f257']
        ph = np.array(R['cohorts'][coh]['physical'][c])
        refs = {}
        for tag in ('ST', 'S'):
            f = refdir / f'ref_{tag}_{coh}_{c:03d}.npz'
            if f.exists():
                refs[tag] = np.load(f)['f257']
        n0r = float(np.linalg.norm(tr[0]))
        for s in arms:
            saved = {}
            for name in arms[s]:
                f = out_dir / f'audit_{s}_{name}_{coh}{c}.npz'
                if f.exists():
                    saved[name] = np.load(f)
            for name, z in saved.items():
                row = next(x for x in rows if x['setting'] == s and x['arm'] == name and x['cohort'] == coh and x['case'] == c)
                fr_ = z['f257']
                chk = dict(setting=s, arm=name, case=c, sha_ok=sha(fr_) == row['restricted_sha256'])
                chk['same_grid_restricted'] = abs(evolved(rel_rows(fr_, tr, n0r)) - row['same_grid_restricted_evolved'])
                for tag, rf in refs.items():
                    if row.get(f'ref_{tag}_evolved') is not None:
                        chk[f'ref_{tag}'] = abs(evolved(rel_rows(fr_, rf, n0r)) - row[f'ref_{tag}_evolved'])
                for other in ('dense', 'gref'):
                    k = f'vs_{other}_restricted_evolved'
                    if row.get(k) is not None and other in saved:
                        chk[f'vs_{other}'] = abs(evolved(rel_rows(fr_, saved[other]['f257'], n0r)) - row[k])
                if 'full' in z.files and 'full' in T.files:
                    n0 = float(np.linalg.norm(T['full'][0]))
                    chk['same_grid_full'] = abs(evolved(rel_rows(z['full'], T['full'], n0)) - row['same_grid_evolved'])
                rec['errors'].append(chk)
            # injected controls on the first saved arm
            if saved and not rec['controls']:
                name0 = next(iter(saved))
                row = next(x for x in rows if x['setting'] == s and x['arm'] == name0 and x['cohort'] == coh and x['case'] == c)
                fr_ = saved[name0]['f257']
                pert = fr_ * (1 + 1e-6)
                rec['controls']['perturbed_detected'] = bool(sha(pert) != row['restricted_sha256'])
                other = ([x for x in rows if x['setting'] == s and x['arm'] == name0 and x['case'] != c] or
                         [x for x in rows if x['setting'] == s and x['arm'] != name0 and x['case'] == c])
                rec['controls']['swapped_case_detected'] = bool(other and other[0]['restricted_sha256'] != row['restricted_sha256'])
    worst_err = max([max(v for k, v in e.items() if k not in ('setting', 'arm', 'case', 'sha_ok')) for e in rec['errors']] or [np.inf])
    gates['G8_error_recompute'] = dict(passed=bool(rec['errors'] and worst_err <= 1e-10 and all(e['sha_ok'] for e in rec['errors'])),
                                       checked=len(rec['errors']), worst_abs_diff=float(worst_err))

    # rho recompute (continuum target, NumPy Gauss-640) for a deterministic sample incl. each setting's argmax states
    P = npbank.load(CKPT)
    rot = np.load(HERE / 'inputs/rotation_R512.npz')
    Trot = np.asarray(rot['T'])
    Xg, wg = HQ.gauss_tensor(int(cfg['gref'][5:]))
    rho_chk = []
    for s, (Rp, M) in SETTINGS.items():
        if s not in R['rho']:
            continue
        pop = np.load(out_dir / f'population_{s}.npz')
        C, labels = pop['C'], list(pop['labels'])
        per = np.load(out_dir / f'rho_per_state_{s}.npz')
        rules_chk = [r for r in ('gauss64', 'fib6765', 'fib4181') if r in per.files][:2]
        pick = sorted({int(np.argmax(per[r][0])) for r in rules_chk} |
                      set(np.random.default_rng(7).choice(len(C), min(a.rho_sample, len(C)), replace=False).tolist()))
        kx, ky = modes_lean(L, M)
        Tp = Trot[:, :Rp]
        Cs = C[pick]
        tgt = np_offmesh_value(P, Tp, Cs, Xg, wg, L, kx, ky)
        for rule in rules_chk:
            X, w = (HQ.gauss_tensor(64) if rule == 'gauss64' else
                    HQ.fibonacci_lattice({'fib6765': 20, 'fib4181': 19}[rule], seed=0))
            assert len(X) == {'gauss64': 4096, 'fib6765': 6765, 'fib4181': 4181}[rule]
            v = np_offmesh_value(P, Tp, Cs, X, w, L, kx, ky)
            rnp = np.linalg.norm(v - tgt, axis=1) / np.linalg.norm(tgt, axis=1)
            rj = per[rule][0][pick]
            rho_chk.append(dict(setting=s, rule=rule, states=len(pick), max_rel_diff=float(np.max(np.abs(rnp - rj) / rj))))
        if L <= 256:
            mt = np_mesh_target(P, Tp, Cs, L, kx, ky)
            rnp = np.linalg.norm(mt - tgt, axis=1) / np.linalg.norm(tgt, axis=1)
            rj = per['dense'][0][pick]
            rho_chk.append(dict(setting=s, rule='dense(mesh target)', states=len(pick),
                                max_rel_diff=float(np.max(np.abs(rnp - rj) / rj))))
    rec['rho'] = rho_chk
    gates['G8_rho_recompute'] = dict(passed=bool(rho_chk and max(x['max_rel_diff'] for x in rho_chk) <= 1e-6),
                                     worst=float(max([x['max_rel_diff'] for x in rho_chk] or [np.inf])), detail=rho_chk)

    # timing recompute (keys setting|kind|mesh|name) and timed-output identity
    inv = R['timing'].get('invocations', [])
    med = {}
    for d in inv:
        med.setdefault(f"{d['setting']}|{d['kind']}|{d['mesh']}|{d['name']}", []).append(d['seconds'])
    med = {k: 1e3 * float(np.median(v)) for k, v in med.items()}
    tdiff = max([abs(med[k] - R['timing']['median_ms'][k]) / R['timing']['median_ms'][k] for k in med] or [np.inf])
    k0 = next(iter(med)) if med else None
    ctrl_t = None
    if k0:
        vv = [d['seconds'] * 1.2 for d in inv if f"{d['setting']}|{d['kind']}|{d['mesh']}|{d['name']}" == k0]
        ctrl_t = bool(abs(1e3 * np.median(vv) - med[k0]) > 1e-9)
    rec['controls']['time_x1.2_detected'] = ctrl_t
    gates['G8_timing_recompute'] = dict(passed=bool(med and tdiff <= 1e-12), subjects=len(med), worst_rel=float(tdiff))
    gates['G8_injected_controls'] = dict(passed=bool(rec['controls'] and all(rec['controls'].values())), detail=rec['controls'])
    rom_inv = [d for d in inv if 'output_matches_phase1' in d]
    fom_inv = [d for d in inv if d['kind'] == 'fom']
    gates['G8_timed_outputs'] = dict(passed=bool(rom_inv and all(d['output_matches_phase1'] for d in rom_inv)
                                                 and all(d['converged'] for d in fom_inv)),
                                     rom_checked=len(rom_inv), rom_mismatch=sum(not d['output_matches_phase1'] for d in rom_inv),
                                     fom_checked=len(fom_inv), fom_unconverged=sum(not d['converged'] for d in fom_inv))
    timing = dict(median_ms=med, solve_ms={}, fom={})
    for k, v in med.items():
        st_, kind, mesh_, name = k.split('|')
        if kind == 'rom':
            dk = f'{st_}|decode|{mesh_}|None'
            if dk in med:
                timing['solve_ms'][f'{st_}|{mesh_}|{name}'] = v - med[dk]
    for d in fom_inv:
        e_ = timing['fom'].setdefault(d['name'], dict(worst_same_grid_evolved=0.))
        e_['worst_same_grid_evolved'] = max(e_['worst_same_grid_evolved'], d['same_grid_evolved'])
    for nm in timing['fom']:
        timing['fom'][nm]['median_ms_all_blocks'] = 1e3 * float(np.median([d['seconds'] for d in fom_inv if d['name'] == nm]))
    # B4 (cross-mesh on this GPU): solve(L) / solve(smallest timing mesh)
    b4 = {}
    meshes = sorted({int(k.split('|')[1]) for k in timing['solve_ms']})
    if len(meshes) > 1:
        for k, v in timing['solve_ms'].items():
            st_, mesh_, name = k.split('|')
            if int(mesh_) == max(meshes):
                lo = timing['solve_ms'].get(f'{st_}|{min(meshes)}|{name}')
                if lo:
                    r_ = v / lo
                    b4[f'{st_}|{name}'] = dict(solve_ms={m_: timing['solve_ms'].get(f'{st_}|{m_}|{name}') for m_ in meshes},
                                               ratio=r_, passed=bool(.8 <= r_ <= 1.25))
    timing['B4_same_gpu'] = b4

    # G6 rollout level: the Gauss-768 rollout within 0.1 x B3 of the Gauss-640 rollout (where run)
    g6r = {}
    for s in arms:
        b = arms[s].get('gref_check', {}).get('B3')
        if b:
            g6r[s] = dict(worst_vs_gref=b['worst'], bar=.1 * B3[s], passed=bool(b['worst'] <= .1 * B3[s]))
    if g6r:
        gates['G6_rollout_convergence'] = dict(passed=bool(all(v['passed'] for v in g6r.values())), detail=g6r)

    # -------------------------------------------- question (iv): worst states ----
    iv = {}
    for s in R['rho']:
        per = np.load(out_dir / f'rho_per_state_{s}.npz')
        labels = [tuple(x.split('|')) for x in np.load(out_dir / f'population_{s}.npz')['labels']]
        ent = {}
        for rule in per.files:
            rc = per[rule][0]
            i = int(np.argmax(rc))
            coh, c, k = labels[i][0], int(labels[i][1]), int(labels[i][2])
            ph = R['cohorts'][coh]['physical'][c]
            cases = sorted({(x[0], int(x[1])) for x in labels})
            cmax = np.array([max(rc[j] for j, x in enumerate(labels) if (x[0], int(x[1])) == cc) for cc in cases])
            wv = np.array([R['cohorts'][cc[0]]['physical'][cc[1]][2] for cc in cases])
            rk = lambda v: np.argsort(np.argsort(v)).astype(float)
            sp = float(np.corrcoef(rk(cmax), rk(wv))[0, 1]) if len(cases) > 2 else None
            top = np.argsort(rc)[::-1][:max(1, len(rc) // 100)]
            ent[rule] = dict(argmax=dict(cohort=coh, case=c, k=k, width=ph[2], amplitude=ph[3], nu=ph[4], rho=float(rc[i])),
                             spearman_casemax_vs_width=sp,
                             top1pct_share_k_le_5=float(np.mean([int(labels[j][2]) <= 5 for j in top])))
        iv[s] = ent

    summ = dict(attempt=cfg['attempt'], job_id=R['job_id'], commit=R['commit'], gpu=R['gpu'], mesh=L,
                cohorts=coh_sha, elapsed_seconds=R.get('elapsed_seconds'), gates=gates,
                failed_gates=[k for k, v in gates.items() if v.get('passed') is False],
                arms=arms, fom=fom, rho=R['rho'], timing=timing, question_iv=iv,
                result_sha256=hashlib.sha256((out_dir / 'result.json').read_bytes()).hexdigest(),
                references_job=rr.get('job_id'))
    Path(a.out).write_text(json.dumps(summ, indent=1) + '\n')
    print(json.dumps({k: v.get('passed') for k, v in gates.items()}, indent=1))
    print('failed:', summ['failed_gates'])


if __name__ == '__main__':
    main()
