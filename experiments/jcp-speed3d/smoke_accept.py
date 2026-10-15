"""jcp-speed3d smoke acceptance (audit code-2 #1): every mandatory gate of the cluster smoke, read from the saved outputs;
writes <out>/ACCEPTANCE.json with one entry per gate (pass / fail / missing, with the evidence) and the overall verdict.

    python smoke_accept.py --out <smoke output dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

import analyze as AN


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    O = Path(a.out)
    G = {}

    def gate(name, ok, **ev):
        G[name] = dict(status='pass' if ok else 'fail', **ev)
    for s in ('M2_512', 'W1024'):
        t = load(O / f'test_s3-{s}.json')
        gate(f'test_s3_{s}', t is not None and t.get('mandatory_failures') == [], failures=None if t is None else t.get('mandatory_failures'))
    t = load(O / 'test_fom.json')
    gate('test_fom (G6v, K8, G7, G5 M2, FOM runs converge)', t is not None and t.get('mandatory_failures') == [],
         failures=None if t is None else t.get('mandatory_failures'))
    # grid
    r = load(O / 'smkgrid' / 'result.json')
    if r is None:
        gate('smkgrid', False, missing=True)
    else:
        gate('smkgrid complete', bool(r.get('complete')))
        for lab, tm in r['timing'].items():
            gate(f'smkgrid panel {lab} valid + hashes', bool(tm.get('valid')), missing_expected=tm.get('missing_expected'))
            gate(f'smkgrid panel {lab} T1 in [1.8, 2.2]', AN.t1_ok(tm))
        ctl = AN.controls(r, None)
        for sn, c in ctl.items():
            for k in ('K1_fires', 'K2_fires', 'K3_fires'):
                gate(f'smkgrid {sn} {k}', bool(c.get(k)))
        for sn, sv in r['settings'].items():
            dep = next(iter(sv['rules']))
            worst = max(v['worst'] for v in sv['rules'][dep]['adt_tight_tighter'].values())
            gate(f'smkgrid {sn} ADT tight/tighter <= 2.5e-5', worst <= 2.5e-5, worst=worst)
            rp = sv['rules'][dep].get('h11_replay', {})
            gate(f'smkgrid {sn} replay ran without error', all(v.get('label') not in ('replay error',) for v in rp.values()),
                 labels={k: v.get('label') for k, v in rp.items()})
    # fom
    r = load(O / 'smkfom' / 'result.json')
    if r is None:
        gate('smkfom', False, missing=True)
    else:
        gate('smkfom complete', bool(r.get('complete')))
        for k, v in r['fom']['fd'].items():
            gate(f'smkfom {k} calibrated', v['calibration'].get('chosen') is not None, status=v['calibration'].get('status'))
            if 'cases' in v:
                gate(f'smkfom {k} converged + scored', all(c['converged'] and np.isfinite(c.get('e', np.nan)) for c in v['cases'].values()))
                gate(f'smkfom {k} warm hashes', bool(v['warm']['hashes_ok']))
        for k, v in r['fom']['ps'].items():
            gate(f'smkfom {k} finite + scored', all(c['converged'] and np.isfinite(c.get('e', np.nan)) for c in v['cases'].values()))
            gate(f'smkfom {k} warm hashes', bool(v['warm']['hashes_ok']))
    # prof
    r = load(O / 'smkprof' / 'result.json')
    if r is None:
        gate('smkprof', False, missing=True)
    else:
        gate('smkprof complete', bool(r.get('complete')))
        for lab, tm in r['timing'].items():
            gate(f'smkprof panel {lab} valid', bool(tm.get('valid')), missing_expected=tm.get('missing_expected'))
        P = r['prof']
        for lab, b in P['batch'].items():
            for B, x in b.items():
                if 'infeasible' in x:
                    gate(f'smkprof batch {lab} B={B} feasible', False, error=x['infeasible'])
                    continue
                gate(f'smkprof batch {lab} B={B} G4', x['G4'] in ('exact', 'within tolerance'), G4_max=x['G4_max'])
                gate(f'smkprof batch {lab} B={B} hashes', bool(x['hashes_ok']))
                gate(f'smkprof batch {lab} B={B} K6+ (permutation)', bool(x['K6plus_pass']))
                if x['K6_fires'] is not None:
                    gate(f'smkprof batch {lab} B={B} K6 fires', bool(x['K6_fires']))
        for lab, m in P['mf'].items():
            gate(f'smkprof mf {lab} ran', 'dist_to_adt' in m, info=m.get('invalid'))
        for lab, k in P['k5'].items():
            gate(f'smkprof K5 {lab} gl12 full rank / conditioned', bool(k['rank_ok']), cond=k['cond'])
    # mesh
    r = load(O / 'smkmesh' / 'result.json')
    if r is None:
        gate('smkmesh', False, missing=True)
    else:
        gate('smkmesh complete', bool(r.get('complete')))
        M = r['mesh']
        gate('smkmesh G5/K7 all banks incl. W256', bool(M.get('G5_all')) and any(k.startswith('W1024|256') for k in M['G5']),
             G5={k: v.get('gate') for k, v in M['G5'].items()})
        for L, E in M['per_L'].items():
            for lab, v in E['rom'].items():
                gate(f'smkmesh L{L} ROM {lab}', 'dist_to_adt' in v, info=v.get('infeasible'), tables=v.get('tables'))
            for k, v in E['fom'].items():
                gate(f'smkmesh L{L} FOM {k}', 'cases' in v and all(c['converged'] for c in v['cases'].values()),
                     info=v.get('infeasible') or (v.get('calibration') or {}).get('status'))
        for lab, tm in r['timing'].items():
            gate(f'smkmesh panel {lab} valid', bool(tm.get('valid')), missing_expected=tm.get('missing_expected'))
    ok = all(v['status'] == 'pass' for v in G.values())
    out = dict(verdict='PASS' if ok else 'FAIL', failing=[k for k, v in G.items() if v['status'] != 'pass'], gates=G)
    (O / 'ACCEPTANCE.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(out['verdict'], out['failing'])


if __name__ == '__main__':
    main()
