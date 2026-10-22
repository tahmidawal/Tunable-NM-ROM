"""Smoke acceptance (DESIGN 6, A1.6, A2.6, A6): every control predicate recomputed from the raw job outputs.
python smoke_accept.py <run dir> <analysis.json from hscore> -> checks/smoke-accept-<attempt>.json; exit 1 if any fails."""
import json, sys
from pathlib import Path
import numpy as np
import hlabel as HL

run, ana = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
R = json.loads((run / 'result.json').read_text())
# binding (code audit 2 / job audit smk1-1): the analysis must belong to this run and cover every scored arm
assert Path(ana['source']).resolve() == run.resolve(), ('analysis belongs to another run', ana['source'])
assert (run / 'COMPLETE').exists(), 'run not complete'
assert set(ana['q1']) == {n for n in R['q1'] if not n.startswith('C4')} and set(ana['q2']) == set(R['q2']), 'analysis coverage'
assert ana['q1'] and ana['q2'] and 'R' in ana['q1'], 'empty analysis'
PROVENANCE = {'recomputed': ['C1_sine', 'C1_bank', 'C2', 'C3', 'C5', 'C6', 'C4', 'C8'],
              'job-reported (in-job assertion or comparison)': ['C7', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']}
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
        S_ = len(z['eta'])
        want = sorted({int(k) - 1 for k in z['ks']} | {int(k) for k in z['ks'] if k < S_})
        ok3 = ok3 and v['base_certified'] and flagged == want and v[tag]['ok']
out['C3'] = dict(fired=bool(ok3 and c3['label_c'] == 'fail' and c3['pooled_c'] > .01))
z4 = np.load(run / 'arrays' / 'q1_C4.npz')
out['C4'] = dict(fired=bool(np.any(z4['reason'] == 2) and C['C4']['q1_label'] == 'fail' and not C['C4']['valid']))   # A9.2
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
out['P4'] = dict(passed=bool(p4['passed'] and p4['rhs_max'] <= 1e-12 and max(p4['jac_max'].values()) <= 1e-10), fixtures=p4['fixtures'])
if 'P5' in C:
    p5 = C['P5']
    out['P5'] = dict(passed=bool(max(p5['rhs_np']) <= 1e-10 and max(p5['lap_identity']) <= 1e-5 and max(p5['energy']) <= 1e-6
                                 and p5['host_mass'] <= 1e-11 and p5['host_stiff'] <= 1e-11),
                     mesh_mass_rel=p5['mesh_mass_rel'])
p6 = C['P6']
out['P6'] = dict(passed=bool(p6['a_missed_tol'] and p6['a_label'] == 'fail' and p6['b_label'].startswith('unverified')
                             and p6['c_label'].startswith('unverified')))
if 'P2' in C:
    out['P2'] = dict(passed=bool(C['P2']['passed'] and C['P2']['max'] <= 1e-10), max=C['P2']['max'])
REQUIRED = ['C1_sine', 'C1_bank', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']
missing = [k for k in REQUIRED if k not in out]
for k in missing:
    out[k] = dict(fired=False, missing=True)
BLOCKABLE = {'C5': 'timer resolution: speed predicates are evaluated only if C5-prime fires in the production job (A9.1)'}
blocked = {k: BLOCKABLE[k] for k, v in out.items() if k in BLOCKABLE and not v.get('fired', v.get('passed'))}
ok = all(v.get('fired', v.get('passed')) for k, v in out.items() if k not in blocked) and not missing
res = dict(attempt=R['config'].get('attempt'), accepted=ok, items=out, blocked_claims=blocked, predicate_provenance=PROVENANCE,
           offline_disagreements={f'{ns}:{k}': v['disagreements'] for ns in ('q1', 'q2') for k, v in ana[ns].items()
                                  if v.get('disagreements')},
           gcert_failures=[k for k, v in ana['q1'].items() if any('G-cert' in r_ for r_ in v['reasons'])],
           run=str(run), config=R['config'], source_commit=(run.parent.parent / 'COMMIT.txt').read_text().strip() if (run.parent.parent / 'COMMIT.txt').exists() else None,
           stepper_sha256=R['setup'].get('stepper_sha256'), fscore_sha256=R['setup'].get('fscore_sha256'))
tc = R['config'].get('timing', {})
res['timing_settings_ok'] = bool(tc.get('reps', 5) == 5 and tc.get('burn', 3) == 3)
res['accepted'] = bool(ok and not res['offline_disagreements'] and not res['gcert_failures'] and res['timing_settings_ok'])
Path(__file__).resolve().parent.joinpath('checks', f"smoke-accept-{res['attempt']}.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: v.get('fired', v.get('passed')) for k, v in out.items()}), 'accepted', res['accepted'])
sys.exit(0 if res['accepted'] else 1)
