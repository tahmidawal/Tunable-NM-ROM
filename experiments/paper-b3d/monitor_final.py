"""Save a source-derived monitoring snapshot, never an accepted paper summary."""
import argparse
import datetime
import hashlib
import json
import shlex
import statistics
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    run=Path(__file__).resolve().parent/'runs'/args.attempt
    submission=json.loads((run/'submission.json').read_text())
    def read(relative):
        return subprocess.check_output(['ssh','tufts-login',shlex.join(['cat',submission['remote']+'/'+relative])])
    raw=read('out/seed0/result.json');result=json.loads(raw)
    calls=result['invocations']+result.get('operator_invocations',[]);rows=[]
    for name in sorted({value['method'] for value in calls}):
        selected=[value for value in calls if value['method']==name]
        accepted=[value.get('stationary',value.get('converged')) for value in selected]
        rows.append(dict(method=name,invocations=len(selected),
            worst_evolved=max(value['worst_evolved'] for value in selected),
            median_gpu_ms=statistics.median(value['gpu_ms'] for value in selected),
            stationary_or_converged=None if all(value is None for value in accepted) else sum(bool(value) for value in accepted)))
    status=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        job_id=submission['job_id'],source_commit=submission['source_commit'],remote=submission['remote'],
        state='PROVISIONAL monitoring snapshot only; acceptance requires both seed panels, all references, independent local audits, checksum collection, verified Git retention and remote cleanup',
        scope=result['comparison_scope'],source_result_sha256=hashlib.sha256(raw).hexdigest(),
        primary_seed_index=0,primary_panel_complete=result['complete'],operator_complete=result.get('operator_complete',False),
        cluster_audit=json.loads(read('out/seed0/audit.json')),rows=rows)
    (run/'PROVISIONAL-STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({key:value for key,value in status.items() if key not in ('rows','cluster_audit')},indent=2))


if __name__=='__main__':main()
