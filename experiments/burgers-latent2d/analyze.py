"""Recompute paired cost gate and retained timing witnesses; no PDE/model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('result'); ap.add_argument('--out',required=True); a=ap.parse_args()
    path=Path(a.result); x=json.loads(path.read_text()); assert x['complete']
    arms=('direct','physics','transfer')
    times={(t['arm'],t['case'],t['block']):t for t in x['timings']}
    assert len(times)==126 and len(x['timings'])==126
    for t in x['timings']:
        z=np.load(path.parent/t['array_witness'])
        assert sha_array(z['W'])==t['W_sha256'] and sha_array(z['fields'])==t['fields_sha256']
        assert int(z['it_sum'])==t['fine_corrections']
    summary=dict(scope=x['scope'],source_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 job_id=x['job_id'],commit=x['commit'],gpu=x['gpu'],settings=x['settings'],arms={},
                 all_fresh_certificates=all(r['valid'] for r in x['records']),
                 all_baseline_controls=all(r['passed'] for r in x['parity']),
                 all_timed_identities=all(t['identity'] and t['online_valid'] for t in x['timings']),
                 strong_accuracy_checks=x['strong_accuracy_checks'],continue_accuracy=x['continue_accuracy'],
                 elapsed_program_seconds=x['elapsed_program_seconds'],setup=x['setup'],checks=x['checks'])
    for arm in arms:
        ts=np.array([[times[(arm,c,b)]['seconds'] for b in range(7)] for c in range(6)])
        base=np.array([[times[('direct',c,b)]['seconds'] for b in range(7)] for c in range(6)])
        ratios=ts/base; med=np.median(ratios,axis=1)
        rec=[r for r in x['records'] if r['arm']==arm]
        zrec=[np.load(path.parent/f'{arm}_case{c}.npz') for c in range(6)]
        for z in zrec:
            assert np.all(np.isfinite(z['gamma_fresh']))
            assert np.all((z['gamma_fresh']<=1e-3)|(z['step_tolratio']<=1))
            assert np.all(np.abs(z['gamma_fresh']-z['step_gn'])<=1e-7*np.abs(z['step_gn'])+1e-12)
        if arm!='direct':
            np.testing.assert_allclose(med,x['gate'][arm]['per_case_paired_median_ratios'],rtol=1e-14,atol=1e-14)
        summary['arms'][arm]=dict(median_query_ms=float(np.median(np.median(ts,axis=1))*1e3),
            median_paired_ratio=float(np.median(med)),per_case_ratios=med.tolist(),
            slower_pairs=int(np.sum(ratios>1)),total_pairs=42,
            median_fine_corrections=float(np.median([r['fine_corrections'] for r in rec])),
            total_fine_corrections=sum(r['fine_corrections'] for r in rec),
            total_fine_J=sum(r['fine_J'] for r in rec),
            total_coarse_J=sum(r['coarse_J'] for r in rec),
            total_down_J=sum(r['auxiliary_sums']['down_J'] for r in rec),
            total_up_J=sum(r['auxiliary_sums']['up_J'] for r in rec),
            accepted_candidates=sum(r['auxiliary_sums']['accepted'] for r in rec),
            auxiliary_failures=sum(r['auxiliary_failures'] for r in rec),
            cost_gate=None if arm=='direct' else x['gate'][arm]['passed'])
        if arm!='direct':
            for z in zrec:
                info=z['seed_info']
                assert np.all(info[:,1]<=4) and np.all(info[:,5]<=2) and np.all(info[:,8]<=4)
                for row in info:
                    if row[0]:
                        assert all(row[j]!=3 for j in (2,6,9)) and row[13]<row[12] and row[13]<=(1-1e-8)*row[12]
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
