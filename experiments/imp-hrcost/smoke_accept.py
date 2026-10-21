"""Smoke acceptance (DESIGN 6, A1.6, A2.6, A6): every control predicate recomputed from the raw job outputs.
python smoke_accept.py <run dir> <analysis.json from hscore> -> checks/smoke-accept-<attempt>.json; exit 1 if any fails."""
import json, sys
from pathlib import Path
import numpy as np
import hlabel as HL

run, ana = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
R = json.loads((run / 'result.json').read_text())
C, Q2 = R['controls'], R['q2']
out = {}


def relabel(name):
    q = Q2[name]
    z = np.load(run / 'arrays' / f'q2_{name}.npz')
    ids, dev = [str(x) for x in z['case_ids']], [str(x) for x in z['dev_ids']]
    rows = q['rows']
    return HL.q2_label(list(z['own_valid']), list(z['succ_valid']), [rows[c]['dist_succ'] for c in ids],
                       np.asarray([rows[c]['rho'] for c in ids]), [bool(z['succ_valid'][ids.index(c)]) for c in dev],
                       list(z['chk_valid']), [q['chk'][c] for c in dev])


for p in ('sine', 'bank'):
    k = f'C1_{p}'
    if k in Q2:
        lab = relabel(k)
        out[k] = dict(fired=bool(lab.get('fail_rollout') and lab.get('fail_rho')), recomputed=lab, job=C[k])
if 'C2' in Q2:
    lab = relabel('C2')
    out['C2'] = dict(fired=lab['label'] == 'fail' and C['C2']['fixture_valid'], recomputed=lab['label'])
c3 = C['C3']
ok3 = bool(c3['per_case'])
for cid, v in c3['per_case'].items():
    for tag in ('a', 'b', 'c'):
        z = np.load(run / 'arrays' / f'C3_{cid.replace("|", "_")}_{tag}.npz')
        flagged = sorted(np.nonzero(~HL.certified(z['eta']))[0].tolist())
        ok3 = ok3 and v['base_certified'] and flagged == v[tag]['expected'] and v[tag]['ok']
out['C3'] = dict(fired=bool(ok3 and c3['label_c'] == 'fail' and c3['pooled_c'] > .01))
out['C4'] = dict(fired=bool((not C['C4']['valid']) and C['C4']['q1_label'] == 'fail'))
r5 = float(np.median(C['C5']['reps_dup']) / np.median(C['C5']['reps_R']))
out['C5'] = dict(fired=bool(C['C5']['max_rel_diff'] <= 1e-12 and r5 >= 1.5), ratio=r5)
aa = R['timing'].get('AA', {})
r6 = float(np.median([v['e2e']['ratio'] for v in aa['per_case'].values()])) if aa else float('nan')
out['C6'] = dict(fired=bool(aa and .9 <= r6 <= 1.1), e2e_median=r6)
out['C7'] = dict(fired=bool(C['C7']['fired']))
out['C8'] = dict(fired=bool(ana['controls']['C8']['fired']))
out['P1'] = dict(passed=bool(C['P1']['max_rel_diff'] <= 1e-12), value=C['P1']['max_rel_diff'])
if 'P3' in C:
    p3 = C['P3']
    out['P3'] = dict(passed=bool(max(p3['dM'], p3['dK']) <= 2.5e-6 and p3['mass_min_eig'] > 0 and p3['stiff_min_eig'] > 0),
                     dM=p3['dM'], dK=p3['dK'])
p4 = C['P4']
out['P4'] = dict(passed=bool(p4['rhs_max'] <= 1e-12 and max(v for _, v in p4['jac']) <= 1e-10))
if 'P5' in C:
    p5 = C['P5']
    out['P5'] = dict(passed=bool(max(p5['rhs_np']) <= 1e-10 and max(p5['lap_identity']) <= 1e-5 and max(p5['energy']) <= 1e-6
                                 and p5['host_mass'] <= 1e-11 and p5['host_stiff'] <= 1e-11),
                     mesh_mass_rel=p5['mesh_mass_rel'])
p6 = C['P6']
out['P6'] = dict(passed=bool(p6['a_missed_tol'] and p6['a_label'] == 'fail' and p6['b_label'].startswith('unverified')
                             and p6['c_label'].startswith('unverified')))
REQUIRED = ['C1_sine', 'C1_bank', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'P1', 'P3', 'P4', 'P5', 'P6']
missing = [k for k in REQUIRED if k not in out]
for k in missing:
    out[k] = dict(fired=False, missing=True)
ok = all(v.get('fired', v.get('passed')) for v in out.values()) and not missing
res = dict(attempt=R['config'].get('attempt'), accepted=ok, items=out,
           offline_disagreements={k: v['disagreements'] for k, v in {**ana['q1'], **ana['q2']}.items() if v.get('disagreements')})
res['accepted'] = ok and not res['offline_disagreements']
Path(__file__).resolve().parent.joinpath('checks', f"smoke-accept-{res['attempt']}.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: v.get('fired', v.get('passed')) for k, v in out.items()}), 'accepted', res['accepted'])
sys.exit(0 if res['accepted'] else 1)
