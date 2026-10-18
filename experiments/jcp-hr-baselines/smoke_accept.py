"""Offline acceptance of the phase-I smoke (DESIGN.md 10; A1.9, A2.9-A2.11, A3.2-A3.3, A3.9, A4.1-A4.4, A4.9, A5.1, A5.4,
A6). Every predicate is recomputed from numerical evidence (smoke.json + arrays_*.npz + the CPU gate file), on the CPU.
A required item that is missing, skipped, errored, non-finite or unavailable is NOT accepted (fail-closed); unavailable
fixtures are reported as 'unfulfilled'. Writes ACCEPTANCE.json next to the outputs.

Usage: JAX_PLATFORMS=cpu python smoke_accept.py <out_dir> <job log> [<checks/test_core.json>]
"""
from __future__ import annotations

import json
import math
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


def fin(*xs):
    try:
        return all(x is not None and math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def cert_ok(v):
    """A5.2 / A6.1 certificate recomputed from (frob_rel, pod_tail_sq)."""
    t2, e = v.get('pod_tail_sq'), v.get('frob_rel')
    if not fin(t2, e) or t2 < -1e-12:
        return False
    tau = math.sqrt(max(t2, 0.))
    return e <= 1e-6 if tau < 1e-6 else abs(e - tau) <= 0.01 * tau + 1e-6


def q1_recompute(runs, anchors, dist, anchor_dist, subset):
    """Q1 (A2.7) from per-run certification evidence and recomputed distances."""
    valid = all(r['cert']['valid'] for r in runs.values())
    ns = sum(r['cert']['nonstat'] for r in runs.values())
    st = sum(r['cert']['steps'] for r in runs.values())
    a_ok = all(anchors['tight'][c]['valid'] and anchors['tight'][c]['worst_gamma'] <= LEVEL_GTOL['tight'] for c in runs) \
        and all(anchors['tighter'][c]['valid'] and anchors['tighter'][c]['worst_gamma'] <= LEVEL_GTOL['tighter']
                for c in subset)
    return bool(valid and st > 0 and ns <= 0.01 * st and a_ok and all(fin(v) and v <= 2.5e-4 for v in dist.values())
                and all(fin(v) and v <= 2.5e-5 for v in anchor_dist.values()))


def main():
    out = Path(sys.argv[1])
    logp = Path(sys.argv[2])
    cpu = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / 'checks' / 'test_core.json'
    S = json.loads((out / 'smoke.json').read_text())
    acc = {}

    def put(k, ok, **ev):
        acc[k] = dict(ok=bool(ok), **ev)

    txt = logp.read_text() if logp.exists() else ''
    put('backend=gpu', 'jax_backend=gpu' in txt)
    put('disjoint', S.get('disjoint') == [True, True])
    try:
        cg = json.loads(cpu.read_text())
        put('cpu-gates', cg.get('all_pass') is True and len(cg.get('gates', {})) >= 13, gates={k: v[0] for k, v in cg['gates'].items()})
    except Exception as e:  # noqa: BLE001
        put('cpu-gates', False, error=str(e))
    import hrcore as H
    import score as SC
    import stats as ST
    for c in S['config']['cells']:
        d, L = c['d'], c['L']
        key = f'{d}d_L{L}'
        R = S.get(key)
        if R is None:
            put(f'{key}|present', False)
            continue
        ph = R.get('phases', {})
        miss = [p for p in REQUIRED_PHASES if ph.get(p, {}).get('status') != 'ok']
        put(f'{key}|phases', not miss, missing_or_failed={p: ph.get(p) for p in miss},
            peaks={k: (v.get('device_peak'), v.get('host_maxrss'), v.get('seconds')) for k, v in ph.items()})
        g = R.get('G-fom') or {}
        put(f'{key}|G-fom', set(g) == {'BE', 'CN'} and all(fin(v['max_diff'], v['norm0']) and
                                                          v['max_diff'] <= 1e-10 * v['norm0'] for v in g.values()), ev=g)
        b = R.get('basis') or {}
        proj = b.get('proj') or {}
        ranks = [str(r) for r in c['ranks']]
        dev = (b.get('gate') or {}).get('orthonormality')
        put(f'{key}|G-pod', bool(proj) and all(r in proj and cert_ok(proj[r]) for r in ranks) and fin(dev) and dev <= 1e-10,
            orthonormality=dev, ranks={r: proj.get(r) for r in ranks})
        a = R.get('adv') or {}
        ap = a.get('proj') or {}
        plist = [str(p) for p in R.get('p_list', [])]
        put(f'{key}|G-pod(adv)', bool(plist) and all(p in ap and cert_ok(ap[p]) for p in plist)
            and fin(a.get('orthonormality')) and a['orthonormality'] <= 1e-10, p=plist)
        q = R.get('qdeim_pilot') or {}
        put(f'{key}|qdeim-max-shape', fin(q.get('solve_backward_error')) and q['solve_backward_error'] <= 1e-12,
            ev={k: q.get(k) for k in ('p', 'interp_const', 'solve_backward_error', 'seconds', 'device_peak')})
        e = R.get('ecsw_rs') or {}
        put(f'{key}|ecsw-smoke-rank', len(e) == len(c['taus']) and all(
            e.get(str(t)) is not None and fin(e[str(t)]['relres']) and e[str(t)]['relres'] <= t for t in c['taus']), taus=e)
        ep = R.get('ecsw_pilot') or {}
        put(f'{key}|ecsw-max-shape', fin(ep.get('relres'), ep.get('contrib_seconds')), ev=ep)
        npil = R.get('none_pilot') or {}
        put(f'{key}|none-max-shape', fin(npil.get('seconds')), ev=npil)
        # grid evidence (Q1 recomputed; distances recomputed from saved fields)
        z = np.load(out / f'arrays_{key}.npz') if (out / f'arrays_{key}.npz').exists() else None
        grid = R.get('grid') or {}
        n0 = {int(k): v for k, v in (R.get('n0') or {}).items()}
        n_expect = len([1 for k in ('none|-', f"ecsw|{c['taus'][-1]}", f"qdeim|{c['ps']}")
                        if k in (R.get('arm_info') or {})]) * len(c['options']) * 2
        cases = c['cases']
        subset = cases[:2]
        q1s, tvs, bad = {}, 0., []
        for cid, gg in grid.items():
            try:
                F = z[cid.replace('|', '~') + '~F']
                Ft = z[cid.replace('|', '~') + '~Ftight']
                Ftt = z[cid.replace('|', '~') + '~Ftighter']
                dist = {j: SC.lattice_distance(F[i], Ft[i], n0[j]) for i, j in enumerate(cases)}
                adist = {j: SC.lattice_distance(Ft[i], Ftt[i], n0[j]) for i, j in enumerate(subset)}
                runs = {int(k): v for k, v in gg['runs'].items()}
                anchors = {kk: {int(j): vv for j, vv in v.items()} for kk, v in gg['anchors'].items()}
                q1s[cid] = q1_recompute(runs, anchors, dist, adist, subset)
                if q1s[cid] != gg['q1']['q1']:
                    bad.append(cid)
                tvs = max(tvs, max(r['timed_vs_scored'] for r in runs.values()))
            except Exception as ex:  # noqa: BLE001
                bad.append(f'{cid}: {ex}')
        put(f'{key}|grid-Q1-machinery', len(grid) == n_expect and n_expect > 0 and not bad and tvs <= 1e-12,
            n=len(grid), expected=n_expect, q1=q1s, mismatches=bad, timed_vs_scored=tvs)
        # G-fd
        fd = R.get('G-fd') or {}
        ga = fd.get('G-fd-a') or {}
        okfd = fin(ga.get('rows'), ga.get('jac')) and ga['rows'] <= 1e-12 and ga['jac'] <= 1e-11
        arms = [k for k in fd if k != 'G-fd-a']
        for k in arms:
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
        put(f'{key}|G-fd+N-fd', okfd and len(arms) == 6, ev=fd)
        gal = R.get('G-all') or {}
        put(f'{key}|G-all', fin(gal.get('dist')) and gal['dist'] <= 1e-10 and gal.get('valid') == [True, True], ev=gal)
        # N-under (fires per family only with a Q1-qualified, finite, degraded level)
        nu_ = R.get('N-under') or {}
        fam = {}
        if 'none' in nu_:
            base = float(np.median(list(nu_['none']['e'].values())))
            for k, v in nu_.items():
                if k == 'none':
                    continue
                f = k.split('_')[0]
                if 'construction_failure' in v:
                    fam.setdefault(f, []).append(dict(level=k, construction_failure=v['construction_failure']))
                    continue
                es = list(v['e'].values())
                qok = bool(v['q1']['q1'])
                med = float(np.median(es)) if all(fin(x) for x in es) else float('nan')
                fam.setdefault(f, []).append(dict(level=k, median=med, q1=qok, degraded=bool(qok and fin(med) and med >= 2 * base)))
            for f in ('ecsw', 'qdeim'):
                levels = fam.get(f, [])
                put(f'{key}|N-under[{f}]', any(x.get('degraded') for x in levels),
                    status='fired' if any(x.get('degraded') for x in levels) else 'unfulfilled', none_median=base,
                    levels=levels)
        else:
            put(f'{key}|N-under[ecsw]', False, status='missing')
            put(f'{key}|N-under[qdeim]', False, status='missing')
        # N-grad (A4.1): agreement with the oracle on perturbed tuples; shared certifier = trajectory certifier
        ng = R.get('N-grad') or {}
        if ng.get('status') == 'run':
            tp = [t for t in ng['tuples'] if t.get('oracle') is not None]
            tu = [t for t in ng['tuples'] if t.get('oracle') is None]
            agree = all(fin(t['cert'], t['oracle']) and abs(t['cert'] - t['oracle']) <= 1e-9 * t['oracle'] + 1e-12 and
                        ((t['cert'] > 1e-3) == (t['oracle'] > 1e-3)) for t in tp)
            same = all(fin(t['cert'], t['cert_traj']) and abs(t['cert'] - t['cert_traj']) <= 1e-9 * abs(t['cert_traj']) + 1e-12
                       for t in tu)
            disc = [t for t in tp if t['oracle'] > 1e-3]
            status = 'pass' if (agree and same and disc) else ('fixture unavailable' if (agree and same) else 'fail')
            put(f'{key}|N-grad(a)', status == 'pass', status=status, n_perturbed=len(tp), n_discriminating=len(disc),
                shared_vs_trajectory=same)
        else:
            put(f'{key}|N-grad(a)', False, status=ng.get('status', 'missing'), reason=ng.get('reason'))
        acc[f'{key}|N-grad(b) reported'] = dict(ok=None, ev=ng.get('b'))
        # N-swap
        sw = R.get('N-swap') or {}
        if sw:
            eps = sw['eps_max']
            rf = sw['ref_fixture']
            ok_ref = fin(rf['true'], rf['case_swap'], rf['time_shift']) and \
                abs(rf['case_swap'] - rf['true']) > 10 * eps and abs(rf['time_shift'] - rf['true']) > 10 * eps
            rom = sw.get('rom_fixture')
            if isinstance(rom, dict):
                ok_rom = abs(rom['case_swap'] - rom['true']) > 10 * eps and abs(rom['time_shift'] - rom['true']) > 10 * eps \
                    and abs(rom['refstore'] - rom['true']) <= 1e-12 * max(rom['true'], 1e-300)
                rom_status = 'pass' if ok_rom else 'fail'
            else:
                ok_rom, rom_status = True, 'fixture unavailable (A5.1: scoring acceptance rests on the reference fixture)'
            put(f'{key}|N-swap', ok_ref and ok_rom and sw.get('kinds') == KINDS_WANT, ref=rf, rom=rom, rom_status=rom_status,
                kinds=sw.get('kinds'))
        else:
            put(f'{key}|N-swap', False, status='missing')
        # N-time and ours' adapter (recomputed from raw triples)
        tm = R.get('timing') or {}
        nt = (tm.get('N-time') or {}).get('cases', {})
        rho = {int(j): [t['tB'] / (0.5 * (t['tA1'] + t['tA2'])) for t in v['triples'] if t['valid']] for j, v in nt.items()}
        if len(rho) == 6 and all(rho.values()):
            ts = ST.time_stats(rho)
            put(f'{key}|N-time', 1.8 <= ts['median'] <= 2.2 and ts['ci95'][0] > 1. and
                max(v['timed_vs_scored'] for v in nt.values()) <= 1e-12, T=ts['median'], ci95=ts['ci95'],
                retries=sum(len(v['triples']) for v in nt.values()))
        else:
            put(f'{key}|N-time', False, reason='timing coverage incomplete', cases=sorted(rho))
        ours = tm.get('ours') or {}
        oc = ours.get('cases', {})
        rx = {int(j): [t['tB'] / (0.5 * (t['tA1'] + t['tA2'])) for t in v['triples'] if t['valid']] for j, v in oc.items()}
        ok_o = len(rx) == 6 and all(rx.values()) and ours.get('e0') is not None and \
            max(v['timed_vs_scored_X'] for v in oc.values()) <= 1e-12
        put(f'{key}|ours-adapter+A-B-A', ok_o, id=ours.get('id'), e0=ours.get('e0'),
            T=ST.time_stats(rx)['median'] if ok_o else None)
        # G-score: re-decode coefficients and rescore against E0
        if z is None:
            put(f'{key}|G-score', False, reason='arrays missing')
            continue
        Pl = z['Phi_lat']
        worst_dec, worst_sc, n_checked = 0., 0., 0
        try:
            famc = H.Family(d, 'dev6' if d == 2 else 'val64', count=None if d == 2 else max(cases) + 1)
            refs = SC.Refs(d, 'dev6' if d == 2 else 'val64', famc, cases)
            for cid, gg in grid.items():
                W = z[cid.replace('|', '~') + '~W']
                F = z[cid.replace('|', '~') + '~F']
                for i, j in enumerate(cases):
                    dec = H.to_lattice(W[i] @ Pl.T, d)
                    worst_dec = max(worst_dec, float(np.max(np.abs(dec - F[i]))) / max(float(np.max(np.abs(F[i]))), 1e-300))
                    ej = refs.score(dec, j)['e']
                    rec = gg['e0'][str(j)]['e'] if str(j) in gg['e0'] else gg['e0'][j]['e']
                    worst_sc = max(worst_sc, abs(ej - rec) / max(rec, 1e-300))
                    n_checked += 1
            put(f'{key}|G-score', n_checked == len(grid) * len(cases) and n_checked > 0 and worst_dec <= 1e-12
                and worst_sc <= 1e-12, decode=worst_dec, score=worst_sc, n=n_checked)
        except Exception as ex:  # noqa: BLE001
            put(f'{key}|G-score', False, error=f'{type(ex).__name__}: {ex}')
    req = {k: v for k, v in acc.items() if v['ok'] is not None}
    acc['ALL'] = dict(ok=all(v['ok'] for v in req.values()), failed=[k for k, v in req.items() if not v['ok']])
    (out / 'ACCEPTANCE.json').write_text(json.dumps(acc, indent=1, default=str))
    for k, v in acc.items():
        print(f"{k}: {v['ok']}" + ('' if v['ok'] or k == 'ALL' else f"  {json.dumps({kk: vv for kk, vv in v.items() if kk in ('status', 'reason', 'missing_or_failed', 'mismatches')}, default=str)[:300]}"))


if __name__ == '__main__':
    main()
