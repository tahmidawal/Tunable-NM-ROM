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
    assert int(rep['mesh']) == L, ('mesh mismatch', rep['mesh'], L)
    expect = [f'dev6|{j}' for j in range(6)] + [f'val32|{j}' for j in range(32)]

    def fin(v):
        return v is not None and isinstance(v, (int, float)) and np.isfinite(v)

    def tri(ok_pred, values):
        """pass / fail / unverified (missing or non-finite evidence)."""
        if not values or not all(fin(v) for v in values):
            return 'UNVERIFIED'
        return 'PASS' if ok_pred(values) else 'FAIL'

    for cc in rep['config']['configs']:
        for sc in cc['schemes']:
            key = f"{cc['name']}|{sc}"
            R = np.load(src / f"R_{cc['name']}_{sc}.npz")
            rr = [x for x in rows if x['config'] == cc['name'] and x['scheme'] == sc]
            ids = [str(c) for c in R['case_ids']]
            rid = [f"{x['cohort']}|{x['case']}" for x in rr]
            full = cc.get('cases') == 'all38'
            coverage_ok = (sorted(ids) == sorted(expect) and sorted(rid) == sorted(expect)) if full else \
                (len(set(ids)) == len(ids) and sorted(ids) == sorted(rid))
            per = []
            for i, cid in enumerate(ids):
                coh, j = cid.split('|')
                j = int(j)
                s_ = refs[coh].score_safe(R['f257'][i], j)
                e0 = float(np.linalg.norm((R['f257'][i][0] - refs[coh].f[j][0]).ravel()) / refs[coh].n0(j))
                per.append(dict(cohort=coh, case=j, e=s_['e'], eps_ref=s_['eps_ref'], eligible=s_['eligible'], e0=e0))
            el = np.array([p['e'] for p in per if p['eligible']])
            steps = 50 * len(rr)
            nonstat = sum(x['nonstationary'] for x in rr)
            dev = [x for x in rr if x['cohort'] == 'dev6']
            q1 = dict(n=len(rr), coverage_ok=coverage_ok, all_valid=all(x['valid'] for x in rr),
                      tight_valid=all(x.get('tight_valid') for x in rr), gcert_ok=all(x.get('gcert_ok') for x in rr),
                      nonstat=nonstat, steps=steps,
                      dist_tight=tri(lambda v: max(v) <= Q1_DIST, [x.get('dist_tight') for x in rr]),
                      anchor=tri(lambda v: len(v) == 6 and max(v) <= Q1_ANCHOR, [x.get('dist_tight_tighter') for x in dev]),
                      tighter_valid=all(x.get('tighter_valid') for x in dev) and len(dev) == 6,
                      budget_exits=sum(x.get('budget_exits', 0) for x in rr))
            q1['state'] = 'PASS' if (coverage_ok and q1['all_valid'] and q1['tight_valid'] and q1['gcert_ok']
                                     and nonstat <= Q1_NONSTAT * steps and q1['dist_tight'] == 'PASS'
                                     and q1['anchor'] == 'PASS' and q1['tighter_valid']) else \
                ('UNVERIFIED' if 'UNVERIFIED' in (q1['dist_tight'], q1['anchor']) else 'FAIL')
            q2r = [x for x in rr if 'rho_max' in x or 'dist_qp' in x]
            q2 = dict(n=len(q2r), qp_valid=all(x.get('qp_valid') for x in q2r),
                      rho=tri(lambda v: max(v) <= Q2_RHO, [x.get('rho_max') for x in q2r]),
                      dist_qp=tri(lambda v: max(v) <= Q2_DIST, [x.get('dist_qp') for x in q2r]),
                      qpp_runs=sum(1 for x in dev if 'qpp_valid' in x), qpp_valid=all(x.get('qpp_valid') for x in dev if 'qpp_valid' in x),
                      max_dist_qp_qpp=max((x['dist_qp_qpp'] for x in dev if fin(x.get('dist_qp_qpp'))), default=None))
            q2['state'] = 'PASS' if (len(q2r) == 38 and q2['qp_valid'] and q2['rho'] == 'PASS' and q2['dist_qp'] == 'PASS'
                                     and q2['qpp_runs'] == 6 and q2['qpp_valid']) else \
                ('UNVERIFIED' if 'UNVERIFIED' in (q2['rho'], q2['dist_qp']) or len(q2r) != 38 or q2['qpp_runs'] != 6
                 else 'FAIL')
            status = 'QUALIFIED' if (q1['state'] == 'PASS' and q2['state'] == 'PASS') else \
                ('NOT-QUALIFIED' if 'FAIL' in (q1['state'], q2['state']) else 'UNRESOLVED')
            out['configs'][key] = dict(per_case=per, worst=float(el.max()) if len(el) else None,
                                       median=float(np.median(el)) if len(el) else None,
                                       n_eligible=int(len(el)), q1=q1, q2=q2, status=status,
                                       W_sha=[x['W_sha'] for x in rr])
            print(key, 'worst', out['configs'][key]['worst'], 'median', out['configs'][key]['median'], 'Q1', q1['state'],
                  'Q2', q2['state'], status, flush=True)
    outp.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
