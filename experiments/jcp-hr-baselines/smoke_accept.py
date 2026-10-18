"""Offline acceptance of the phase-I smoke (DESIGN.md 10; A1.9, A2.9-A2.11, A3.2-A3.3, A3.9, A4.1-A4.4, A4.9, A5.1, A5.4,
A6, A7). Every predicate is recomputed from raw evidence: per-step certification lists (gamma, exit reason, LM damping,
trajectory finiteness), saved lattice fields (distances), raw timing triples and per-call discrepancies, projection
errors, and the CPU gate file bound to source hashes. The frozen A5.4 inventory is validated, not read from the output.
A required item that is missing, skipped, errored, non-finite or unavailable is NOT accepted; unavailable fixtures are
'unfulfilled'. Writes ACCEPTANCE.json next to the outputs.

Usage: JAX_PLATFORMS=cpu python smoke_accept.py <out_dir> <job log> <staged lane dir or this lane dir>
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REQUIRED_PHASES = ('refs', 'G-fom', 'basis', 'adequacy', 'generator_probe_cost', 'adv_pod', 'qdeim_max_shape',
                   'ecsw_states', 'ecsw_smoke_rank', 'ecsw_max_shape', 'none_max_shape', 'grid', 'G-fd', 'G-all',
                   'N-under', 'N-grad', 'N-swap', 'N-time+ours')
KINDS_WANT = {'3|S|BE|0.005': 'raised', '3|S|BE|0.01': 'raised', '2|S|CNR|0.005': 'raised', '2|S|BE|0.01': 'raised',
              '2|S|BE|0.005': 'allowed', '2|ST|CNR|0.005': 'allowed'}
LEVEL_GTOL = {'prod': 1e-3, 'tight': 1e-7, 'tighter': 1e-8}
CPU_GATES = ('provenance-train-draw-2d', 'disjoint-train-cohorts', 'G-fd-a operators vs refcore',
             'G-fd-b online Jacobians (FD + jacfwd), every arm', 'N-fd mutant Jacobian fails the FD check',
             'G-fd-c ECSW rows and Q-DEIM objective vs independent assembly', 'FOM (hrcore) vs refcore.make_solver, BE and CN',
             'G-lmm step equations (refcore operators) and temporal order, every scheme',
             'G-full complete basis reproduces the FOM (every arm, option, solver)',
             'G-nnls greedy NNLS vs scipy (objective, KKT, tau stop)', 'G-qdeim pivots vs scipy, exact interpolation on span(U)',
             'G-pod orthonormality, snapshot reproduction and POD optimality',
             'N-dom decision table on synthetic data (A2.11, A3.3, A4.2; hand-derived labels)')
# A5.4 frozen smoke inventory
FROZEN = {2: dict(L=1024, n_traj=192, rmax=512, pmax=2048, rs=128, ps=256, taus=[1e-2, 1e-3], cases=[0, 1],
                  under_fixtures=[2, 3], n_adequacy=8, n_probe=4, options=[['BE', 0.005], ['CNR', 0.005]]),
          3: dict(L=64, n_traj=342, rmax=1024, pmax=4096, rs=256, ps=512, taus=[1e-2, 1e-3], cases=[0, 1],
                  under_fixtures=[5, 0], n_adequacy=8, n_probe=4, options=[['BE', 0.01], ['CN', 0.01], ['BDF2', 0.025]])}


def fin(*xs):
    try:
        return all(x is not None and math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def cert_ok(v):
    """A6.1 certificate recomputed from (frob_rel, pod_tail_sq)."""
    t2, e = v.get('pod_tail_sq'), v.get('frob_rel')
    if not fin(t2, e) or t2 < -1e-12:
        return False
    tau = math.sqrt(max(t2, 0.))
    return e <= 1e-6 if tau < 1e-6 else abs(e - tau) <= 0.01 * tau + 1e-6


def run_evidence(cert, level):
    """Recompute (valid, nonstat, steps, anchor_ok) from the per-step lists (A2.7, A6.5)."""
    g = np.asarray(cert['gamma_list'], float)
    rsn = np.asarray(cert['reason_list'], int)
    lam = np.asarray(cert['lam_list'], float)
    steps = len(g)
    ok_len = steps == cert['steps'] == len(rsn) == len(lam) and steps > 0
    valid = bool(ok_len and cert['C_finite'] and np.all(rsn >= 0) and not np.any(rsn == 3)
                 and np.all(np.isfinite(lam)) and np.all(lam < 1e14))
    stationary = np.isfinite(g) & (g <= 1e-3)
    anchor_ok = bool(valid and np.all(np.isfinite(g) & (g <= LEVEL_GTOL[level])))
    return valid, int(np.sum(~stationary)), steps, anchor_ok


def q1_from_evidence(prod, tight, tighter, dist, adist, cases, subset):
    if sorted(prod) != sorted(cases) or sorted(tight) != sorted(cases) or sorted(tighter) != sorted(subset):
        return False, 'coverage'
    ev = [run_evidence(prod[c], 'prod') for c in cases]
    valid = all(e[0] for e in ev)
    ns, st = sum(e[1] for e in ev), sum(e[2] for e in ev)
    a_ok = all(run_evidence(tight[c], 'tight')[3] for c in cases) and all(run_evidence(tighter[c], 'tighter')[3] for c in subset)
    ok = bool(valid and st > 0 and ns <= 0.01 * st and a_ok and all(fin(v) and v <= 2.5e-4 for v in dist.values())
              and all(fin(v) and v <= 2.5e-5 for v in adist.values()))
    return ok, dict(valid=valid, nonstat=ns, steps=st, anchors=a_ok, max_dist=max(dist.values()) if dist else None)


def triple_ok(t):
    vals = (t['tA1'], t['tB'], t['tA2'])
    return bool(all(fin(v) and v > 0 for v in vals) and abs(t['tA1'] / t['tA2'] - 1.) <= 0.25)


def main():
    out = Path(sys.argv[1])
    logp = Path(sys.argv[2])
    lane = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE
    S = json.loads((out / 'smoke.json').read_text())
    acc = {}

    def put(k, ok, **ev):
        acc[k] = dict(ok=bool(ok), **ev)

    txt = logp.read_text() if logp.exists() else ''
    put('backend=gpu', 'jax_backend=gpu' in txt)
    put('disjoint', S.get('disjoint') == [True, True])
    # CPU gates bound to the exact sources that ran in the smoke
    try:
        cg = json.loads((lane / 'checks' / 'test_core.json').read_text())
        names_ok = sorted(cg['gates']) == sorted(CPU_GATES) and all(cg['gates'][n][0] == 'PASS' for n in CPU_GATES)
        src_ok = all(hashlib.sha256((lane / f).read_bytes()).hexdigest() == h for f, h in cg['source_sha256'].items())
        put('cpu-gates', cg.get('all_pass') is True and names_ok and src_ok, names_ok=names_ok, sources_match=src_ok)
    except Exception as e:  # noqa: BLE001
        put('cpu-gates', False, error=str(e))
    cfg_cells = {c['d']: c for c in S['config']['cells']}
    if os.environ.get('HRB_ACCEPT_DRY') == '1':      # local CPU dry run only: never a binding acceptance
        for d_ in (2, 3):
            FROZEN[d_] = {k: cfg_cells[d_][k] for k in FROZEN[d_]}
        acc['DRY-RUN (not binding)'] = dict(ok=False)
    put('frozen-inventory', sorted(cfg_cells) == [2, 3] and all(
        all(cfg_cells[d].get(k) == v for k, v in FROZEN[d].items()) for d in (2, 3)),
        diffs={d: {k: (cfg_cells.get(d, {}).get(k), v) for k, v in FROZEN[d].items() if cfg_cells.get(d, {}).get(k) != v}
               for d in (2, 3)})
    import hrcore as H
    import score as SC
    import stats as ST
    for d in (2, 3):
        c = dict(FROZEN[d], d=d)
        L = c['L']
        key = f'{d}d_L{L}'
        R = S.get(key)
        if R is None:
            put(f'{key}|present', False)
            continue
        n = (L - 1) ** d
        ph = R.get('phases', {})
        miss = [p for p in REQUIRED_PHASES if ph.get(p, {}).get('status') != 'ok']
        put(f'{key}|phases', not miss, missing_or_failed={p: ph.get(p) for p in miss},
            peaks={k: (v.get('device_peak'), v.get('host_maxrss'), v.get('seconds')) for k, v in ph.items()})
        g = R.get('G-fom') or {}
        put(f'{key}|G-fom', set(g) == {'BE', 'CN'} and all(fin(v['max_diff'], v['norm0']) and v['norm0'] > 0 and
                                                          v['max_diff'] <= 1e-10 * v['norm0'] for v in g.values()), ev=g)
        b = R.get('basis') or {}
        proj = b.get('proj') or {}
        ranks = [str(r) for r in (c['rs'], c['rmax'])]
        dev = (b.get('gate') or {}).get('orthonormality')
        put(f'{key}|G-pod', all(r in proj and cert_ok(proj[r]) for r in ranks) and fin(dev) and dev <= 1e-10,
            orthonormality=dev, ranks={r: proj.get(r) for r in ranks})
        a = R.get('adv') or {}
        ap = a.get('proj') or {}
        ladder, p = [], c['rs'] // 8
        while p >= 1:
            ladder.append(p)
            p //= 2
        plist = [str(p) for p in sorted(set([c['ps'], c['pmax']] + ladder))]
        put(f'{key}|G-pod(adv)', all(p in ap and cert_ok(ap[p]) for p in plist) and fin(a.get('orthonormality'))
            and a['orthonormality'] <= 1e-10, p=plist)
        # maximal-shape pilots: recorded shapes equal the registered maxima; finite positive device peaks
        q = R.get('qdeim_pilot') or {}
        put(f'{key}|qdeim-max-shape', fin(q.get('solve_backward_error')) and q['solve_backward_error'] <= 1e-12
            and q.get('B_shape') == [n, 2 * c['rmax'] + c['pmax']] and fin(q.get('device_peak')) and q['device_peak'] > 0,
            ev={k: q.get(k) for k in ('B_shape', 'interp_const', 'solve_backward_error', 'seconds', 'device_peak')})
        e = R.get('ecsw_rs') or {}
        put(f'{key}|ecsw-smoke-rank', all(e.get(str(t)) is not None and fin(e[str(t)]['relres']) and e[str(t)]['relres'] <= t
                                          for t in c['taus']), taus=e)
        ep = R.get('ecsw_pilot') or {}
        nst = (R.get('ecsw_states') or {}).get('n_states')
        shp = list(ep.get('shape', [0, 0]))
        put(f'{key}|ecsw-max-shape', fin(ep.get('relres'), ep.get('device_peak')) and nst is not None and
            shp[0] > 0 and shp[0] % c['rmax'] == 0 and shp[0] <= nst * c['rmax'] and shp[1] == min(32768, n),
            ev=ep, n_states=nst)
        npil = R.get('none_pilot') or {}
        put(f'{key}|none-max-shape', fin(npil.get('seconds'), npil.get('device_peak')) and npil.get('J_shape') == [n, c['rmax']],
            ev={k: npil.get(k) for k in ('seconds', 'device_peak', 'J_shape')})
        z = np.load(out / f'arrays_{key}.npz') if (out / f'arrays_{key}.npz').exists() else None
        n0 = {int(k): v for k, v in (R.get('n0') or {}).items()}
        cases, subset = c['cases'], c['cases'][:2]
        grid = R.get('grid') or {}
        expect = {f'{k}|{c["rs"]}|{h}|{o[0]}|{o[1]:.8g}|{s}|central' for k, h in (('none', '-'), ('ecsw', c['taus'][-1]),
                  ('qdeim', c['ps'])) for o in c['options'] for s in ('native', 'common')}
        q1s, bad, tvs = {}, [], []
        for cid in sorted(expect):
            gg = grid.get(cid)
            try:
                F = z[cid.replace('|', '~') + '~F']
                Ft = z[cid.replace('|', '~') + '~Ftight']
                Ftt = z[cid.replace('|', '~') + '~Ftighter']
                dist = {j: SC.lattice_distance(F[i], Ft[i], n0[j]) for i, j in enumerate(cases)}
                adist = {j: SC.lattice_distance(Ft[i], Ftt[i], n0[j]) for i, j in enumerate(subset)}
                prod = {int(k): v['cert'] for k, v in gg['runs'].items()}
                anc = {kk: {int(j): vv for j, vv in v.items()} for kk, v in gg['anchors'].items()}
                ok, ev = q1_from_evidence(prod, anc['tight'], anc['tighter'], dist, adist, cases, subset)
                q1s[cid] = ok
                if ok != gg['q1']['q1']:
                    bad.append(f'{cid}: in-job q1 {gg["q1"]["q1"]} != recomputed {ok}')
                tvs += [gg['runs'][str(j)]['timed_vs_scored'] if str(j) in gg['runs'] else gg['runs'][j]['timed_vs_scored']
                        for j in cases]
            except Exception as ex:  # noqa: BLE001
                bad.append(f'{cid}: {type(ex).__name__}: {ex}')
        put(f'{key}|grid-Q1-machinery', not bad and len(q1s) == len(expect) and all(fin(v) and v <= 1e-12 for v in tvs),
            expected=len(expect), q1=q1s, problems=bad)
        # G-fd (A1.9 thresholds)
        fd = R.get('G-fd') or {}
        ga = fd.get('G-fd-a') or {}
        okfd = fin(ga.get('rows'), ga.get('jac')) and ga['rows'] <= 1e-12 and ga['jac'] <= 1e-11
        want_arms = {f'{k}|{m}' for k in ('none', 'ecsw', 'qdeim') for m in ('None', 'drop_adv_jac')}
        okfd &= {k for k in fd if k != 'G-fd-a'} == want_arms
        for k in want_arms & set(fd):
            v = fd[k]
            if k.endswith('|None'):
                okfd &= fin(v.get('fd'), v.get('jacfwd')) and v['fd'] <= 1e-6 and v['jacfwd'] <= 1e-11
                if k.startswith('ecsw'):
                    okfd &= fin(v.get('rows')) and v['rows'] <= 1e-10
                if k.startswith('qdeim'):
                    okfd &= all(fin(v.get(f'{m}_{t}')) and v[f'{m}_{t}'] <= 1e-10 for m in ('norm', 'grad')
                                for t in ('generic', 'cancel'))
            else:
                okfd &= fin(v.get('fd')) and v['fd'] > 1e-3
        put(f'{key}|G-fd+N-fd', okfd, ev=fd)
        gal = R.get('G-all') or {}
        put(f'{key}|G-all', fin(gal.get('dist')) and gal['dist'] <= 1e-10 and gal.get('valid') == [True, True], ev=gal)
        # N-under: Q1 recomputed from evidence and saved fields; ECSW credit needs support >= r (A6.3)
        nu_ = R.get('N-under') or {}
        fxs = sorted(c['under_fixtures'])
        levels = {}
        base = None
        for tag, v in nu_.items():
            if 'construction_failure' in v:
                levels[tag] = dict(construction_failure=v['construction_failure'], degraded=False)
                continue
            try:
                dist = {j: SC.lattice_distance(z[f'NU~{tag}~{j}~F'], z[f'NU~{tag}~{j}~Ftight'], n0[j]) for j in fxs}
                adist = {j: SC.lattice_distance(z[f'NU~{tag}~{j}~Ftight'], z[f'NU~{tag}~{j}~Ftighter'], n0[j]) for j in fxs}
                prod = {int(j): x for j, x in v['cert'].items()}
                tg = {int(j): x for j, x in v['tight'].items()}
                tgg = {int(j): x for j, x in v['tighter'].items()}
                ok, ev = q1_from_evidence(prod, tg, tgg, dist, adist, fxs, fxs)
                if tag.startswith('ecsw') and not (v.get('support') is not None and v['support'] >= c['rs']):
                    ok = False
                es = [v['e'][str(j)] if str(j) in v['e'] else v['e'][j] for j in fxs]
                med = float(np.median(es)) if all(fin(x) for x in es) else float('nan')
                levels[tag] = dict(q1=ok, median=med, support=v.get('support'))
                if tag == 'none':
                    base = med if ok else None
            except Exception as ex:  # noqa: BLE001
                levels[tag] = dict(error=f'{type(ex).__name__}: {ex}', degraded=False)
        for f in ('ecsw', 'qdeim'):
            lv = {k: x for k, x in levels.items() if k.startswith(f)}
            fired = base is not None and any(x.get('q1') and fin(x.get('median')) and x['median'] >= 2 * base for x in lv.values())
            put(f'{key}|N-under[{f}]', fired, status='fired' if fired else 'unfulfilled', none_median=base, levels=lv)
        # N-grad (A4.1; code audit 2 item 10)
        ng = R.get('N-grad') or {}
        if ng.get('status') == 'run':
            tp = [t for t in ng['tuples'] if t.get('oracle') is not None]
            tu = [t for t in ng['tuples'] if t.get('oracle') is None]
            agree = all(fin(t['cert'], t['oracle']) and abs(t['cert'] - t['oracle']) <= 1e-9 * t['oracle'] + 1e-12 and
                        ((t['cert'] > 1e-3) == (t['oracle'] > 1e-3)) for t in tp)
            same = all(fin(t['cert'], t['cert_traj']) and abs(t['cert'] - t['cert_traj']) <= 1e-9 * abs(t['cert_traj']) + 1e-12
                       for t in tu)
            cls = ng.get('classify') or []
            cls_ok = len(cls) == 14 and all(x['nonstat_production'] == x['nonstat_oracle'] for x in cls)
            disc = [t for t in tp if t['oracle'] > 1e-3]
            status = 'pass' if (agree and same and cls_ok and disc) else (
                'fixture unavailable' if (agree and same and cls_ok) else 'fail')
            put(f'{key}|N-grad(a)', status == 'pass', status=status, n_perturbed=len(tp), n_discriminating=len(disc),
                shared_vs_trajectory=same, classification=cls_ok)
            acc[f'{key}|N-grad(b) reported'] = dict(ok=None, q1=(ng.get('b') or {}).get('q1'))
        else:
            put(f'{key}|N-grad(a)', False, status=ng.get('status', 'missing'), reason=ng.get('reason'))
        # N-swap
        sw = R.get('N-swap') or {}
        if sw:
            eps = sw['eps_max']
            rf = sw['ref_fixture']
            ok_ref = fin(eps, rf['true'], rf['case_swap'], rf['time_shift']) and \
                abs(rf['case_swap'] - rf['true']) > 10 * eps and abs(rf['time_shift'] - rf['true']) > 10 * eps
            rom = sw.get('rom_fixture')
            if isinstance(rom, dict):
                ok_rom = fin(rom['true'], rom['case_swap'], rom['time_shift'], rom['refstore']) and \
                    abs(rom['case_swap'] - rom['true']) > 10 * eps and abs(rom['time_shift'] - rom['true']) > 10 * eps \
                    and abs(rom['refstore'] - rom['true']) <= 1e-12 * max(rom['true'], 1e-300)
                rs_ = 'pass' if ok_rom else 'fail'
            else:
                ok_rom, rs_ = True, 'fixture unavailable (A5.1: scoring acceptance rests on the reference fixture)'
            put(f'{key}|N-swap', ok_ref and ok_rom and sw.get('kinds') == KINDS_WANT, ref=rf, rom=rom, rom_status=rs_,
                kinds=sw.get('kinds'))
        else:
            put(f'{key}|N-swap', False, status='missing')
        # N-time + ours' adapter, from raw triples (drift recomputed) and per-call discrepancies
        tm = R.get('timing') or {}
        for tagk, label, lo_hi in (('N-time', 'N-time', (1.8, 2.2)), ('ours', 'ours-adapter (phase-I execution check; not G-ours)', None)):
            cs_ = (tm.get(tagk) or {}).get('cases', {})
            rho, dmax = {}, 0.
            okd = True
            for j, v in cs_.items():
                for t in v['triples']:
                    for kk in ('dA1', 'dB', 'dA2'):
                        okd &= fin(t.get(kk)) and t[kk] <= 1e-12
                rho[int(j)] = [t['tB'] / (0.5 * (t['tA1'] + t['tA2'])) for t in v['triples'] if triple_ok(t)]
            full = sorted(rho) == list(range(6)) and all(rho.values())
            if full:
                ts = ST.time_stats(rho)
                ok = okd and (lo_hi is None or (lo_hi[0] <= ts['median'] <= lo_hi[1] and ts['ci95'][0] > 1.))
                if tagk == 'ours':
                    ok = ok and (tm.get('ours') or {}).get('e0') is not None and fin(tm['ours']['e0'].get('e'))
                put(f'{key}|{label}', ok, T=ts['median'], ci95=ts['ci95'], outputs_match=okd,
                    attempts=sum(len(v['triples']) for v in cs_.values()))
            else:
                put(f'{key}|{label}', False, reason='timing coverage incomplete', cases=sorted(rho))
        # G-score: re-decode the saved coefficients and rescore (finite checks first)
        if z is None:
            put(f'{key}|G-score', False, reason='arrays missing')
            continue
        try:
            Pl = z['Phi_lat']
            famc = H.Family(d, 'dev6' if d == 2 else 'val64', count=None if d == 2 else max(cases) + 1)
            refs = SC.Refs(d, 'dev6' if d == 2 else 'val64', famc, cases)
            worst_dec, worst_sc, nchk, finite = 0., 0., 0, bool(np.all(np.isfinite(Pl)))
            for cid in sorted(expect):
                W = z[cid.replace('|', '~') + '~W']
                F = z[cid.replace('|', '~') + '~F']
                gg = grid[cid]
                for i, j in enumerate(cases):
                    if not (np.all(np.isfinite(W[i])) and np.all(np.isfinite(F[i]))):
                        finite = False
                        continue
                    dec = H.to_lattice(W[i] @ Pl.T, d)
                    worst_dec = max(worst_dec, float(np.max(np.abs(dec - F[i]))) / max(float(np.max(np.abs(F[i]))), 1e-300))
                    ej = refs.score(dec, j)['e']
                    rec = (gg['e0'][str(j)] if str(j) in gg['e0'] else gg['e0'][j])['e']
                    if not fin(ej, rec):
                        finite = False
                        continue
                    worst_sc = max(worst_sc, abs(ej - rec) / max(rec, 1e-300))
                    nchk += 1
            put(f'{key}|G-score', finite and nchk == len(expect) * len(cases) and worst_dec <= 1e-12 and worst_sc <= 1e-12,
                decode=worst_dec, score=worst_sc, n=nchk, finite=finite)
        except Exception as ex:  # noqa: BLE001
            put(f'{key}|G-score', False, error=f'{type(ex).__name__}: {ex}')
    req = {k: v for k, v in acc.items() if v['ok'] is not None}
    acc['ALL'] = dict(ok=all(v['ok'] for v in req.values()), failed=[k for k, v in req.items() if not v['ok']])
    (out / 'ACCEPTANCE.json').write_text(json.dumps(acc, indent=1, default=str))
    for k, v in acc.items():
        extra = {kk: vv for kk, vv in v.items() if kk in ('status', 'reason', 'problems', 'missing_or_failed', 'error')}
        print(f"{k}: {v['ok']}" + ('' if v['ok'] or k == 'ALL' else f'  {json.dumps(extra, default=str)[:400]}'))


if __name__ == '__main__':
    main()
