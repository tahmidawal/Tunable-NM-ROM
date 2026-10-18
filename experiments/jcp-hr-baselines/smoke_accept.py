"""Offline acceptance of the phase-I smoke (DESIGN.md 10; A1.9, A2.9-A2.11, A3.2-A3.3, A4.1-A4.4, A4.9): every predicate
is recomputed from the raw outputs (smoke.json + arrays_*.npz), on the CPU. Writes ACCEPTANCE.json next to them.

Usage: JAX_PLATFORMS=cpu python smoke_accept.py <out_dir> [<job log>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    out = Path(sys.argv[1])
    logp = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    S = json.loads((out / 'smoke.json').read_text())
    acc = {}

    def put(k, ok, **ev):
        acc[k] = dict(ok=bool(ok) if ok is not None else None, **ev)

    if logp is not None:
        txt = logp.read_text()
        put('backend', 'jax_backend=gpu' in txt, found='jax_backend=gpu' in txt)
    put('disjoint', S.get('disjoint') == [True, True])
    import hrcore as H
    import score as SC
    for c in S['config']['cells']:
        key = f"{c['d']}d_L{c['L']}"
        R = S.get(key)
        if R is None:
            put(f'{key}|present', False)
            continue
        ph = R['phases']
        put(f'{key}|phases', all(v['status'] == 'ok' for v in ph.values()),
            failed={k: v.get('error', v.get('reason')) for k, v in ph.items() if v['status'] != 'ok'},
            peaks={k: (v.get('device_peak'), v.get('host_maxrss')) for k, v in ph.items()})
        g = R.get('G-fom')
        put(f'{key}|G-fom', g is not None and all(v['rel'] <= 1e-10 for v in g.values()), ev=g)
        b = R.get('basis', {}).get('gate', {})
        put(f'{key}|G-pod', b.get('orth_ok') and b.get('optimality_ok'), gate=b)
        a = R.get('adv', {})
        put(f'{key}|G-pod(adv)', a.get('optimality_ok'), numerical_rank=a.get('numerical_rank'), fallback=a.get('fallback'))
        q = (R.get('qdeim_pilot') or {}).get('max', {})
        put(f'{key}|qdeim-max-shape', q.get('solve_ok'), ev={k: q.get(k) for k in ('p', 'interp_const', 'solve_backward_error',
                                                                                    'seconds', 'device_peak')})
        e = R.get('ecsw_rs') or {}
        put(f'{key}|ecsw-smoke-rank', all(v is not None for v in e.values()) and len(e) > 0, taus=e)
        put(f'{key}|ecsw-max-shape', R.get('ecsw_pilot') is not None, ev=R.get('ecsw_pilot'))
        fd = R.get('G-fd') or {}
        okfd = bool(fd) and all((v['fd'] <= 1e-6 and v['jacfwd'] <= 1e-11 and v.get('rows', 0.) <= 1e-10) if k.endswith('|None')
                                else v['fd'] > 1e-3 for k, v in fd.items())
        put(f'{key}|G-fd+N-fd', okfd, ev=fd)
        ga = R.get('G-all')
        put(f'{key}|G-all', ga is not None and ga['dist'] <= 1e-10, ev=ga)
        nu = R.get('N-under')
        if nu:
            base = np.median(list(nu['none'].values()))
            fam = {}
            for k, v in nu.items():
                if k == 'none':
                    continue
                f = k.split('_')[0]
                es = [x['e'] for x in v.values()]
                fin = all(np.isfinite(es))
                fam.setdefault(f, []).append(dict(level=k, median=float(np.median(es)), finite=fin,
                                                  degraded=bool(fin and np.median(es) >= 2 * base)))
            put(f'{key}|N-under', all(any(x['degraded'] for x in v) for v in fam.values()) and set(fam) == {'ecsw', 'qdeim'},
                none_median=float(base), families=fam)
        else:
            put(f'{key}|N-under', False, reason='missing')
        ng = R.get('N-grad')
        if ng:
            t = ng['a']['tuples']
            agree = all(abs(x['cert'] - x['oracle']) <= 1e-9 * x['oracle'] + 1e-12 and ((x['cert'] > 1e-3) == (x['oracle'] > 1e-3))
                        for x in t)
            disc = [x for x in t if x['oracle'] > 1e-3]
            status = 'pass' if (agree and disc) else ('fixture unavailable' if agree else 'fail')
            put(f'{key}|N-grad(a)', status == 'pass', status=status, n_tuples=len(t), n_discriminating=len(disc))
            put(f'{key}|N-grad(b) reported', None, ev=ng['b'])
        else:
            put(f'{key}|N-grad(a)', False, reason='missing')
        sw = R.get('N-swap')
        if sw:
            eps = sw['eps_max']
            rf = sw['ref_fixture']
            ok_ref = abs(rf['case_swap'] - rf['true']) > 10 * eps and abs(rf['time_shift'] - rf['true']) > 10 * eps
            rom = sw.get('rom_fixture')
            ok_rom = rom is None or (abs(rom['case_swap'] - rom['true']) > 10 * eps and
                                     abs(rom['time_shift'] - rom['true']) > 10 * eps and
                                     abs(rom['refstore_score'] - rom['true']) <= 1e-12 * max(rom['true'], 1e-300))
            kinds = {k: v for k, v in sw.items() if k.startswith('kind|')}
            want = {k: ('allowed' if ("|2|" in k and "'BE', 0.005" in k) else 'raised') for k in kinds}
            put(f'{key}|N-swap', ok_ref and ok_rom and kinds == want, ref=rf, rom=rom, kinds=kinds, want=want)
        tm = R.get('timing') or {}
        nt = tm.get('N-time')
        put(f'{key}|N-time', nt is not None and 1.8 <= nt['ratio'] <= 2.2, ratio=None if nt is None else nt['ratio'])
        put(f'{key}|ours-adapter', tm.get('ours') is not None and tm['ours'].get('e0') is not None,
            ev=None if tm.get('ours') is None else dict(e0=tm['ours']['e0'], ratio=tm['ours']['ratio_B_over_ours']))
        # G-score: re-decode coefficients offline and rescore
        z = np.load(out / f'arrays_{key}.npz')
        Pl = z['Phi_lat']
        cases = list(z['cases'])
        worst_dec, worst_sc = 0., 0.
        n_checked = 0
        refs = None
        try:
            fam = H.Family(c['d'], 'dev6' if c['d'] == 2 else 'val64', count=None if c['d'] == 2 else max(cases) + 1)
            refs = SC.Refs(c['d'], 'dev6' if c['d'] == 2 else 'val64', fam, cases)
        except Exception as ex:  # noqa: BLE001
            put(f'{key}|G-score refs', False, error=str(ex))
        for k in z.files:
            if not k.endswith('~W'):
                continue
            cid = k[:-2].replace('~', '|')
            W = z[k]
            F = z[k[:-2] + '~F']
            for i, j in enumerate(cases):
                dec = H.to_lattice((W[i] @ Pl.T), c['d'])
                worst_dec = max(worst_dec, float(np.max(np.abs(dec - F[i]))) / max(float(np.max(np.abs(F[i]))), 1e-300))
                if refs is not None and R['grid'].get(cid, {}).get('e0'):
                    ej = refs.score(dec, j)['e']
                    rec = R['grid'][cid]['e0'][str(j)]['e'] if str(j) in R['grid'][cid]['e0'] else R['grid'][cid]['e0'][j]['e']
                    worst_sc = max(worst_sc, abs(ej - rec) / max(rec, 1e-300))
                n_checked += 1
        put(f'{key}|G-score', n_checked > 0 and worst_dec <= 1e-12 and worst_sc <= 1e-12, decode=worst_dec, score=worst_sc,
            n=n_checked)
    acc['ALL'] = dict(ok=all(v['ok'] for v in acc.values() if v['ok'] is not None))
    (out / 'ACCEPTANCE.json').write_text(json.dumps(acc, indent=1, default=str))
    for k, v in acc.items():
        print(f"{k}: {v['ok']}")


if __name__ == '__main__':
    main()
