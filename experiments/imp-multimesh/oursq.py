"""J5 checker (DESIGN A1.4, A2.5): E0 errors and in-job Q1 + Q2 verdicts of our configurations from byte-identical
jcp-p1-scaling p1acc outputs (result.json rows + R_<config>_<scheme>.npz). CPU / numpy; runs inside the job.

    python oursq.py <p1acc outdir> <mesh> <out.json>
Bars (COMMON-BRIEF 3): every run valid; non-stationary steps (certified gamma > 1e-3 or non-finite) <= 1 % pooled;
distance to tight <= 2.5e-4 on every case; tight/tighter <= 2.5e-5 on dev6; Q2: own-state rho <= 0.116 and successor
rollout distance <= 2.5e-4 on every Q2 case (all 38 here), successor runs valid.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'jcp-hr-baselines'))
import hrcore as H     # noqa: E402
import score as SC     # noqa: E402

Q1_DIST, Q1_ANCHOR, Q1_NONSTAT, Q2_RHO, Q2_DIST = 2.5e-4, 2.5e-5, 0.01, 0.116, 2.5e-4


def main():
    src, L, outp = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    rep = json.loads((src / 'result.json').read_text())
    assert rep['complete'], 'p1acc run incomplete'
    fams = {c: H.Family(2, c) for c in ('dev6', 'val32')}
    refs = {c: SC.Refs(2, c, fams[c], range(fams[c].n)) for c in fams}
    rows = rep['rows']
    out = dict(mesh=L, job=rep.get('job_id'), gpu=rep.get('gpu'), backend=rep.get('backend'), commit=rep.get('commit'),
               checkpoint_sha256=rep.get('checkpoint_sha256'), configs={})
    for cc in rep['config']['configs']:
        for sc in cc['schemes']:
            key = f"{cc['name']}|{sc}"
            R = np.load(src / f"R_{cc['name']}_{sc}.npz")
            rr = [x for x in rows if x['config'] == cc['name'] and x['scheme'] == sc]
            per = []
            for i, cid in enumerate(R['case_ids']):
                coh, j = str(cid).split('|')
                j = int(j)
                s_ = refs[coh].score_safe(R['f257'][i], j)
                e0 = float(np.linalg.norm((R['f257'][i][0] - refs[coh].f[j][0]).ravel()) / refs[coh].n0(j))
                per.append(dict(cohort=coh, case=j, e=s_['e'], eps_ref=s_['eps_ref'], eligible=s_['eligible'], e0=e0))
            el = np.array([p['e'] for p in per if p['eligible']])
            steps = 50 * len(rr)
            nonstat = sum(x['nonstationary'] for x in rr)
            q1 = dict(n=len(rr), all_valid=all(x['valid'] for x in rr), tight_valid=all(x.get('tight_valid') for x in rr),
                      nonstat=nonstat, steps=steps, max_dist_tight=max(x['dist_tight'] for x in rr),
                      dev_anchor=[x.get('dist_tight_tighter') for x in rr if x['cohort'] == 'dev6'],
                      tighter_valid=all(x.get('tighter_valid') for x in rr if x['cohort'] == 'dev6'),
                      gcert_ok=all(x.get('gcert_ok') for x in rr), budget_exits=sum(x.get('budget_exits', 0) for x in rr))
            q1['q1'] = bool(len(rr) == 38 and q1['all_valid'] and q1['tight_valid'] and nonstat <= Q1_NONSTAT * steps
                            and q1['max_dist_tight'] <= Q1_DIST and len(q1['dev_anchor']) == 6
                            and all(v is not None and v <= Q1_ANCHOR for v in q1['dev_anchor']) and q1['tighter_valid'])
            q2r = [x for x in rr if 'rho_max' in x]
            q2 = dict(n=len(q2r), max_rho=max((x['rho_max'] for x in q2r), default=float('nan')),
                      max_dist_qp=max((x['dist_qp'] for x in q2r), default=float('nan')),
                      qp_valid=all(x.get('qp_valid') for x in q2r),
                      max_dist_qp_qpp=max((x.get('dist_qp_qpp', float('nan')) for x in q2r if 'dist_qp_qpp' in x),
                                          default=float('nan')))
            q2['q2'] = bool(len(q2r) == 38 and q2['qp_valid'] and q2['max_rho'] <= Q2_RHO and q2['max_dist_qp'] <= Q2_DIST)
            out['configs'][key] = dict(per_case=per, worst=float(el.max()), median=float(np.median(el)),
                                       n_eligible=int(len(el)), q1=q1, q2=q2, qualified=bool(q1['q1'] and q2['q2']),
                                       W_sha=[x['W_sha'] for x in rr])
            print(key, 'worst', out['configs'][key]['worst'], 'median', out['configs'][key]['median'], 'Q1', q1['q1'],
                  'Q2', q2['q2'], flush=True)
    outp.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
