"""Results-audit check (coefficient -> field -> E0): rebuild lattice fields as to_lattice(W Phi_lat^T) from the saved
coefficients and an archived, sha256-verified lattice basis, score every case against E0 and compare with the
target-job rows.   python coefcheck.py <job> <L> <arm> <Phi_lat.npy> <out.json>"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'jcp-hr-baselines'))
import hrcore as H   # noqa: E402
import score as SC   # noqa: E402


def main():
    job, L, arm, plat, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    T = json.loads((HERE / 'runs' / job / 'pulled' / 'output' / f'target{L}.json').read_text())
    P = np.load(plat)
    fams = {c: H.Family(2, c) for c in ('dev6', 'val32')}
    refs = {c: SC.Refs(2, c, fams[c], range(fams[c].n)) for c in fams}
    out, worst = {}, 0.
    for cid, c in T['arms'][arm]['configs'].items():
        if c.get('status') not in ('MEASURED-Q1', 'MEASURED-FAIL'):
            continue
        r, k = int(cid.split('|')[0][1:]), int(cid.split('|')[1][1:])
        z = np.load(HERE / 'runs' / job / 'pulled' / 'output' / f'W_{arm}_r{r}_k{k}.npz')
        d = []
        for row in c['rows']:
            W = z[f"W|{row['cohort']}|{row['case']}"]
            F = H.to_lattice(W @ P[:, :r].T, 2)
            e = refs[row['cohort']].score(F, row['case'])['e']
            d.append(abs(e - row['e']) / row['e'])
        out[cid] = dict(n=len(d), max_rel_diff=float(max(d)))
        worst = max(worst, max(d))
    Path(outp).write_text(json.dumps(dict(job=job, L=L, arm=arm, configs=out, max_rel_diff=worst), indent=1))
    print(arm, L, 'max relative difference', worst)


if __name__ == '__main__':
    main()
