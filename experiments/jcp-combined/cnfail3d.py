"""Descriptive investigation of the fixed-sweep LSPG-CN K=2 failures at 2.5 dt0 in the 3D escalation jobs (DESIGN A10.3).
Selects nothing. Per setting and rule: per case, the Gram-metric field distance of the fixed-sweep CN K=2 run to the
verified adaptive tight CN run of the same step (per output time, relative to the lattice ||u0||), the per-step solver
evidence of the fixed-sweep run (exit reasons, final gradient norms, rejected steps, damping saturation) and the
adaptive run's iteration counts; the failing set (distance > 2.5e-4) and whether it is identical across rules.

    python cnfail3d.py --run runs/c3descW768/archive/output [--out checks/cnfail3d-c3descW768.json]
"""
from __future__ import annotations

import offline  # noqa: F401  (A8.1: CPU, highest, x64)

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FS, AD = 'fs|CN|0.025|K2', 'ad|LSPG|CN|0.025|tight'
TAU = 2.5e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    import c3common as C3
    rp = Path(a.run)
    res = json.loads((rp / 'result.json').read_text())
    cfg = res['config']
    cases = cfg.get('case_subset') or list(range(cfg['cohort_count']))
    tab = C3.C.table(cfg['cohort_seed'], cfg['cohort_count'])
    i65 = C3.lattice65_index(65)
    X65 = np.stack(np.meshgrid(*(np.arange(1, 64) / 64.,) * 3, indexing='ij'), -1).reshape(-1, 3)
    n0 = {j: float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j))[i65])) for j in cases}
    Wz = np.load(rp / 'W_esc.npz')
    out = dict(run=str(rp), job_id=res.get('job_id'), tau=TAU, fixed=FS, adaptive=AD, settings={})
    for s in cfg['settings']:
        sn = s['name']
        bk = C3.load_bank(s['bank'])
        G = np.asarray(C3.TB.feature_rows(bk['params'], bk['T'][:, :int(s['Rp'])], X65))
        rules = sorted({k.split('|')[1] for k in Wz.files if k.startswith(sn + '|') and k.endswith(FS)})
        per_rule = {}
        for rule in rules:
            kf, ka = f'{sn}|{rule}|{FS}', f'{sn}|{rule}|{AD}'
            if ka not in Wz.files:
                continue
            ids = [int(x) for x in Wz['cases|' + kf]] if 'cases|' + kf in Wz.files else cases
            assert 'cases|' + ka not in Wz.files or [int(x) for x in Wz['cases|' + ka]] == ids, ka
            d = np.linalg.norm(np.einsum('cki,ip->ckp', Wz[kf] - Wz[ka], G), axis=2) / np.array([n0[j] for j in ids])[:, None]
            rows = {(r['case']): r for r in res['rows'] if r['setting'] == sn and r['rule'] == rule and r['spec'] == FS}
            arow = {(r['case']): r for r in res['rows'] if r['setting'] == sn and r['rule'] == rule and r['spec'] == AD}
            cs = {}
            for ci, j in enumerate(ids):
                st = rows[j]['steps']
                gn = np.asarray(st['gn'], float)
                cs[int(j)] = dict(dist_per_time=d[ci].tolist(), worst=float(d[ci, 1:].max()),
                                  reasons=np.bincount(np.asarray(st['reason']), minlength=5).tolist(),
                                  gn_max=float(np.nanmax(gn)), gn_first_above_1e-3=int(np.argmax(gn > 1e-3)) if (gn > 1e-3).any() else None,
                                  rej=int(np.sum(st['rej'])), sat=int(np.sum(st['sat'])), valid=rows[j]['valid'],
                                  e_ref_fixed=rows[j].get('e_ref'), e_ref_adaptive=arow[j].get('e_ref') if j in arow else None,
                                  adaptive_it_total=int(np.sum(arow[j]['steps']['it'])) if j in arow else None)
            fail = sorted(j for j, v in cs.items() if not (np.isfinite(v['worst']) and v['worst'] <= TAU))
            w = np.array([v['worst'] for v in cs.values()])
            per_rule[rule] = dict(failing_cases=fail, n_failing=len(fail), median_worst=float(np.median(w)),
                                  max_worst=float(np.max(w)), cases=cs)
        sets = {r: tuple(v['failing_cases']) for r, v in per_rule.items()}
        out['settings'][sn] = dict(rules=per_rule, failing_set_identical_across_rules=len(set(sets.values())) <= 1,
                                   failing_union=sorted({j for v in sets.values() for j in v}))
    dest = Path(a.out or HERE / 'checks' / f'cnfail3d-{cfg["attempt"]}.json')
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest)
    for sn, v in out['settings'].items():
        print(sn, 'identical' if v['failing_set_identical_across_rules'] else 'varies', v['failing_union'])
        for r, x in v['rules'].items():
            print('  ', r, x['n_failing'], f"{x['median_worst']:.2e} {x['max_worst']:.2e}")


if __name__ == '__main__':
    main()
