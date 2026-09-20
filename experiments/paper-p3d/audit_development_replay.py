"""Check frozen selected assets and deterministic fields across development jobs.

CG is independently tolerance/stopping audited; its finite-precision trajectories
are intentionally not required to be identical across different GPU allocations.
No runtimes from separate jobs are compared here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit(before,after):
    a=json.loads((before/'result.json').read_text())
    b=json.loads((after/'result.json').read_text())
    assert a['complete'] and b['complete']
    assert not a['final_cohort_opened'] and not b['final_cohort_opened']
    assets=['bank.pkl']+[f'head_K{k}.pkl' for k in b['config']['latent_dimensions']]
    assets += [f"operators/{x['name']}/best.pkl" for x in b['config']['operators']]
    hashes={name:digest(after/name) for name in assets}
    assert hashes=={name:digest(before/name) for name in assets}
    def rows(record):
        return {(r['intervals'],r['case'],r['method']):r for r in record['invocations']
                if r['repetition']==0 and 'field_file' in r and not r['method'].startswith('cg_')}
    old,new=rows(a),rows(b)
    assert old.keys()==new.keys()
    comparisons=[]
    for key,row in new.items():
        x=np.load(before/old[key]['field_file'])['prediction']
        y=np.load(after/row['field_file'])['prediction']
        relative=float(np.linalg.norm((x-y).ravel())/max(np.linalg.norm(x.ravel()),1e-300))
        assert np.isfinite(relative) and relative<1e-10,(key,relative)
        comparisons.append(dict(intervals=key[0],case=key[1],method=key[2],relative_field_difference=relative))
    return dict(passed=True,selection_job=a['job_id'],replay_job=b['job_id'],
        selection_result_sha256=digest(before/'result.json'),replay_result_sha256=digest(after/'result.json'),
        checkpoint_sha256=hashes,checked_fields=len(comparisons),
        max_relative_field_difference=max(x['relative_field_difference'] for x in comparisons),
        field_tolerance=1e-10,comparisons=comparisons,
        cg_policy='Independently audited per invocation; no cross-GPU trajectory identity assumed.',
        timing_policy='No cross-job runtime comparisons.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');p.add_argument('--output',required=True)
    a=p.parse_args();Path(a.output).write_text(json.dumps(audit(Path(a.before),Path(a.after)),indent=2)+'\n')
