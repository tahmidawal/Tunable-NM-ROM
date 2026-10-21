"""J5 checker (DESIGN A1.4, A2.5): E0 errors and in-job Q1 + Q2 verdicts of our configurations from byte-identical
jcp-p1-scaling p1acc outputs (result.json rows + R_<config>_<scheme>.npz). CPU / numpy; runs inside the job.

    python oursq.py <p1acc outdir> <mesh> <out.json>
Bars (COMMON-BRIEF 3): every run valid; non-stationary steps (certified gamma > 1e-3 or non-finite) <= 1 % pooled;
distance to tight <= 2.5e-4 on every case; tight/tighter <= 2.5e-5 on dev6; Q2: own-state rho <= 0.116 and successor
rollout distance <= 2.5e-4 on every Q2 case (all 38 here), successor runs valid.
"""
import hashlib
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
    sys.path.insert(0, str(HERE.parent / 'jcp-p1-scaling'))
    import p1common as P1
    for coh in ('dev6', 'val32'):                        # descriptor identity: p1 cohort tables == hrcore tables
        qc = np.asarray(P1.Q.cohort(coh))
        # the job's hash is recorded (it matches only under the job's NumPy build: last-bit RNG differences across
        # NumPy versions, jcp-budget A6); the binding check is the value identity of the two tables in this process
        out.setdefault('cohort_hash_same_build', {})[coh] = rep['cohorts'][coh] == P1.sha(qc)
        assert np.allclose(qc, np.asarray(fams[coh].D), rtol=0, atol=1e-12), ('cohort descriptor table differs', coh)
    expect = [f'dev6|{j}' for j in range(6)] + [f'val32|{j}' for j in range(32)]

    def fin(v):
        return v is not None and isinstance(v, (int, float)) and np.isfinite(v)

    def tri(ok_pred, values):
        """pass / fail / unverified (missing or non-finite evidence)."""
        if not values or not all(fin(v) for v in values):
            return 'UNVERIFIED'
        return 'PASS' if ok_pred(values) else 'FAIL'

    sys.path.insert(0, str(HERE.parent / 'jcp-p1-scaling'))
    import p1score as SCP          # byte-identical jcp-p1-scaling offline checker (NumPy decode, independent rho)
    cases_all = [f'dev6|{j}' for j in range(6)] + [f'val32|{j}' for j in range(32)]
    for cc in rep['config']['configs']:
        for sc in cc['schemes']:
            key = f"{cc['name']}|{sc}"
            R = dict(np.load(src / f"R_{cc['name']}_{sc}.npz"))
            rr = [x for x in rows if x['config'] == cc['name'] and x['scheme'] == sc]
            ids = [str(c) for c in R['case_ids']]
            meta_ok = bool(int(R['mesh']) == L and str(R['scheme']) == sc and int(R['Rp']) == int(cc['Rp']))
            wsha = {f"{x['cohort']}|{x['case']}": x['W_sha'] for x in rr}
            sha_ok = all(hashlib.sha256(np.ascontiguousarray(R['W'][i]).tobytes()).hexdigest() == wsha.get(c)
                         for i, c in enumerate(ids))
            cases = SCP.case_set(cc['cases']) if isinstance(cc['cases'], str) else cases_all
            expect = dict(cases=cases, dev=[x for x in SCP.case_set('dev6') if x in cases],
                          q2=[x for x in SCP.case_set(cc.get('q2_cases', 'sub16')) if x in cases] if cc.get('qp') else [],
                          qpp=bool(cc.get('qpp')))
            tabs = (SCP.rule_tables(cc['rule'], cc['Rp'], R['kx'], R['ky'], L),
                    SCP.rule_tables(cc['qp'], cc['Rp'], R['kx'], R['ky'], L)) if cc.get('qp') else None
            q = SCP.qualify(R, rr, cc['Rp'], expect, rho_tables=tabs, rho_cases=expect['dev'])
            per = []
            for i, cid in enumerate(ids):
                coh, j = cid.split('|')
                j = int(j)
                s_ = refs[coh].score_safe(R['f257'][i], j)
                e0 = float(np.linalg.norm((R['f257'][i][0] - refs[coh].f[j][0]).ravel()) / refs[coh].n0(j))
                per.append(dict(cohort=coh, case=j, e=s_['e'], eps_ref=s_['eps_ref'], eligible=s_['eligible'], e0=e0))
            el = np.array([p['e'] for p in per if p['eligible']])
            ident_ok = bool(meta_ok and sha_ok and ids == cases)
            q2l = q['Q2'].get('label', '')
            if not ident_ok or q['disagreements'] or q2l.startswith('unverified') or 'offline disagreement' in q2l:
                status = 'UNRESOLVED' if (q['Q1'].get('pass') or not ident_ok or q['disagreements']) else 'NOT-QUALIFIED'
            elif q['Q1']['pass'] and q2l == 'pass':
                status = 'QUALIFIED'
            else:
                status = 'NOT-QUALIFIED'           # measured Q1 failure or Q2 'fail' (rollout / rho bar)
            if q2l.startswith('unverified') and q['Q1']['pass']:
                status = 'UNRESOLVED'
            out['configs'][key] = dict(per_case=per, worst=float(el.max()) if len(el) else None,
                                       median=float(np.median(el)) if len(el) else None, n_eligible=int(len(el)),
                                       identity=dict(meta_ok=meta_ok, W_sha_ok=sha_ok, population_ok=ids == cases),
                                       Q1=q['Q1'], Q2={k: v for k, v in q['Q2'].items()},
                                       disagreements=[str(d) for d in q['disagreements']][:20], status=status,
                                       pooled_nonstationary=q['pooled_nonstationary'], pooled_steps=q['pooled_steps'])
            print(key, 'worst', out['configs'][key]['worst'], 'Q1', q['Q1']['pass'], 'Q2', q2l, status, flush=True)
    outp.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
