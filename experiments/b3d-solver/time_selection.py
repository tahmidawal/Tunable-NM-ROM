"""Generate the post-diagnostic temporal endpoint note from preserved numerical outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
    p=HERE/'results/tdiag02/raw/output/result.json';z=json.loads(p.read_text());assert z['complete']
    out=dict(source=dict(path=str(p.relative_to(HERE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
        scope='four exposed validation cases; post-diagnostic choice frozen before full64',
        selection='coarse0.025 repaired policy plus finest0.00625 best-tested-median representative, each conditional on own full64 Q1/Q2',
        abandoned_rule='Worst-only interval selection was not in staged pre-data design and is not used to exclude the fine representative.',
        cost_scope='No warmed production timings in D-T. First-compile anchor seconds are not deployment costs.',
        caution='Median and worst trade off. These two final temporal endpoints do not exhaust the tested diagnostic tradeoff or the temporal frontier.',rows=[])
    for dt in z['config']['dts']:
        rows=[z['runs'][f'W1024|gl56|BDF2|{dt:.8g}|candidate|case{j}'] for j in z['config']['cases']]
        e=np.asarray([r['score']['e'] for r in rows]);eps=np.asarray([r['score']['eps_ref'] for r in rows])
        out['rows'].append(dict(dt=dt,cases=z['config']['cases'],errors=e.tolist(),median=float(np.median(e)),worst=float(e.max()),
            median_lower=float(np.median(np.maximum(e-eps,0))),median_upper=float(np.median(e+eps)),
            worst_lower=float(np.max(np.maximum(e-eps,0))),worst_upper=float(np.max(e+eps)),
            corrections=sum(r['iterations'] for r in rows),jacobian_contractions=sum(r['jacobian_contractions'] for r in rows),
            total_steps=sum(r['steps'] for r in rows),fresh_bad_steps=sum(r['nonstationary'] for r in rows),
            all_target_anchors_usable=all(r['anchor_usable'] for r in rows),max_distance_tight=max(r['distance_tight'] for r in rows),
            all_own_chains_pass=all(v['quadrature_subset_pass'] for k,v in z['quadrature'].items() if k.startswith(f'{dt:.8g}|'))))
    (HERE/'results/time-selection.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(out['rows']))
if __name__=='__main__':main()
