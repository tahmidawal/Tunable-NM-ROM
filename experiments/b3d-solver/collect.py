"""Collect finished job output/logs with checksums before removing only the remote duplicate."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
NS='/cluster/tufts/paralab/tawal01/b3d-solver-20261026'

def ssh(cmd):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','tufts-login',cmd],text=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('job_id',type=int);a=p.parse_args()
    assert re.fullmatch('[a-z0-9-]+',a.attempt)
    remote=NS+'/'+a.attempt; qr=shlex.quote(remote)
    acct=ssh(f'sacct -X -j {a.job_id} -n -P --format=JobIDRaw,State,ElapsedRaw,AllocTRES,NodeList')
    row=next(x.split('|') for x in acct.splitlines() if x.split('|')[0]==str(a.job_id))
    assert row[1] in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED'),row
    # Fail closed on every source check and snapshot only scientifically relevant new output/log files.
    ssh(f'set -euo pipefail; cd {qr}; sha256sum -c MANIFEST.sha256 --quiet; '
        'find output logs -type f -print0 | sort -z | xargs -0 sha256sum > COLLECTION.sha256; '
        'tar -czf collection.tar output logs COLLECTION.sha256 COMMIT.txt MANIFEST.sha256 SOURCE.json run.sbatch')
    stage=HERE/'runs'/a.attempt; dest=stage/'collected';dest.mkdir(exist_ok=False)
    tarpath=stage/'collection.tar'
    subprocess.run(['scp','-q',f'tufts-login:{remote}/collection.tar',str(tarpath)],check=True)
    with tarfile.open(tarpath) as tf:tf.extractall(dest,filter='data')
    for line in (dest/'COLLECTION.sha256').read_text().splitlines():
        digest,rel=line.split(None,1);f=dest/rel.strip()
        assert f.is_relative_to(dest) and hashlib.sha256(f.read_bytes()).hexdigest()==digest,rel
    assert (dest/'COMMIT.txt').read_text()==(stage/'COMMIT.txt').read_text()
    evidence=dict(job_id=a.job_id,attempt=a.attempt,slurm=row,sha256_verified=True,
                  archive_sha256=hashlib.sha256(tarpath.read_bytes()).hexdigest(),remote=remote)
    (stage/'COLLECTED.json').write_text(json.dumps(evidence,indent=1)+'\n')
    # Only now all output and logs have a verified local duplicate. Keep unique failed diagnostics too.
    ssh(f'set -euo pipefail; test -d {qr}; rm -rf -- {qr}')
    evidence['remote_duplicate_deleted']=True
    (stage/'COLLECTED.json').write_text(json.dumps(evidence,indent=1)+'\n')
    print(json.dumps(evidence))

if __name__=='__main__':main()
