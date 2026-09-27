"""Verify staged source by Git content and reused assets by archived bytes."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();exp=Path(__file__).resolve().parent;root=exp.parents[1]
    run=exp/'runs'/a.attempt;submission=json.loads((run/'submission.json').read_text());archive=run/'collected';source=submission['source_commit']
    assert (archive/'EXIT_CODE').read_text().strip()=='0'
    log=(archive/f"slurm.{submission['job_id']}.out").read_text();assert 'jax_backend=gpu' in log
    bad=[]
    for path in archive.rglob('*'):
        if path.is_file() and path.suffix in ['.log','.err']:
            text=path.read_text(errors='replace')
            for token in ['ResourceExhausted','CUDA_ERROR_OUT_OF_MEMORY','No space left on device','Captured consts']:
                if token in text:bad.append(dict(path=str(path.relative_to(archive)),token=token))
    assert not bad,bad
    reuse=None
    if (archive/'code/reuse/REUSE.json').exists():reuse=json.loads((archive/'code/reuse/REUSE.json').read_text())
    assets={v['destination']:v for v in submission['config'].get('assets',[])};checks=[]
    for line in (archive/'code/SOURCE.sha256').read_text().splitlines():
        expected,name=line.split('  ',1);path=archive/'code'/name;assert digest(path)==expected
        if name=='reuse/REUSE.json':checks.append(dict(path=name,sha256=expected,proof='generated immutable reuse manifest'));continue
        if name.startswith('reuse/'):
            original=next(v for v in reuse['files'] if v['path']==name)
            entry='experiments/paper-b3d/runs/'+reuse['attempt']+'/collected/'+original['source_path']
            assert expected==original['sha256']
        elif name in assets:entry=assets[name]['source'];assert expected==assets[name]['sha256']
        else:entry='experiments/paper-b3d/'+name
        blob=subprocess.check_output(['git','-C',str(root),'show',source+':'+entry])
        assert hashlib.sha256(blob).hexdigest()==expected,(name,entry)
        checks.append(dict(path=name,git_path=entry,sha256=expected,proof='Git content at pinned source'))
    result=dict(passed=True,source_commit=source,job_id=submission['job_id'],files=len(checks),checks=checks,
        backend_log_verified=True,invalid_resource_warnings=bad)
    (archive/'source-audit-local.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
