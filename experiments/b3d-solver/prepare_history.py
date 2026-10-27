"""Copy a small, explicitly unsealed historical coefficient inventory; CPU only."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims')
CASES = [0, 1, 2, 3, 4, 5, 6, 7, 30, 43]
PICKS = {
    'm3a': ['M2_512|gl32|qn|BDF2|0.025|k4'],
    'm3b': ['W768|gl40|qn|BDF2|0.025|k4', 'W1024|gl40|qn|BDF2|0.025|k4',
            'W768|gl40|fs|CN|0.01|K1', 'W1024|gl56|fs|CN|0.01|K1',
            'W1024|gl64|fs|CN|0.01|K1',
            'W768|gl40|ad|LSPG|BDF2|0.025|tight'] + [f'W1024|{rl}|ad|LSPG|BDF2|0.025|{level}' for rl in ('gl40','gl48','gl56') for level in ('tight','tighter')],
    'k3b': ['W512k8|lat16384|fs|CN|0.01|K1'],
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    arrays, meta = {}, {'cases': CASES, 'entries': {}, 'sources': {}}
    for att, keys in PICKS.items():
        root = SOURCE / 'runs' / att / 'slim' / 'output' / f'd3run_{att}'
        rp, wp, sp = root/'result.json', root/f'W_{att}.npz', SOURCE/'checks'/f'scores-{att}.json'
        result, scores = json.loads(rp.read_text()), json.loads(sp.read_text())['scores']
        z = np.load(wp, allow_pickle=False)
        meta['sources'][att] = dict(result=str(rp), result_sha256=sha(rp), coefficients=str(wp),
                                    coefficients_sha256=sha(wp), score=str(sp), score_sha256=sha(sp),
                                    job_id=result['job_id'], commit=result['commit'], cohort=result['cohort'])
        for key in keys:
            ids = [int(x) for x in z['cases|'+key]]
            chosen = [j for j in CASES if j in ids]
            arrays[key] = np.asarray(z[key])[[ids.index(j) for j in chosen]]
            arrays['cases|'+key] = np.asarray(chosen, dtype=np.int64)
            rows = [r for r in result['rows'] if r['tag']== '|'.join(key.split('|')[:2])
                    and r['spec']=='|'.join(key.split('|')[2:]) and r['case'] in chosen]
            meta['entries'][key] = dict(cases=chosen, scores={str(j): scores.get(key,{}).get(str(j),{}) for j in chosen},
                                        rows=rows, source=att)
    dst=HERE/'inputs'; dst.mkdir(exist_ok=True)
    np.savez_compressed(dst/'historical.npz', **arrays)
    meta['subset_sha256']=sha(dst/'historical.npz')
    (dst/'historical.json').write_text(json.dumps(meta, indent=1)+'\n')
    print('historical entries',len(meta['entries']), 'bytes', (dst/'historical.npz').stat().st_size)

if __name__=='__main__':
    main()
