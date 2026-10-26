"""Submit a frozen, independently approved package once; every check fails closed."""
from pathlib import Path
import argparse
import hashlib
import re
import subprocess
HERE=Path(__file__).resolve().parent
NS='/cluster/tufts/paralab/tawal01/b3d-solver-20261026'
def ssh(s):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','tufts-login',s],text=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--approved-manifest',required=True);a=p.parse_args()
    assert re.fullmatch('[a-z0-9-]+',a.attempt)
    stage=HERE/'runs'/a.attempt
    assert hashlib.sha256((stage/'MANIFEST.sha256').read_bytes()).hexdigest()==a.approved_manifest
    name='b3ds_'+a.attempt;remote=NS+'/'+a.attempt
    assert not ssh(f'squeue -u tawal01 -n {name} -h').strip(),'matching active job exists'
    archive=HERE/'runs'/(a.attempt+'.tar')
    subprocess.run(['tar','-czf',str(archive),'-C',str(stage),'.'],check=True)
    ssh(f'set -euo pipefail; test ! -e {remote}; mkdir -p {remote}')
    subprocess.run(['scp','-q',str(archive),f'tufts-login:{remote}/source.tar'],check=True)
    ssh(f'set -euo pipefail; cd {remote}; tar -xzf source.tar; sha256sum -c MANIFEST.sha256 --quiet; test ! -d output')
    before=ssh('squeue -u tawal01 -o "%.18i %.32j %.8T %.10M %.20R"');print('QUEUE-BEFORE\n'+before)
    job=ssh(f'set -euo pipefail; cd {remote}; test ! -e SUBMITTED.jobid; '
            f'test -z "$(squeue -u tawal01 -n {name} -h)"; '
            'jid=$(sbatch --parsable run.sbatch); printf "%s\\n" "$jid" > SUBMITTED.jobid; printf "%s\\n" "$jid"').strip()
    assert job.isdigit(),job
    (stage/'SUBMITTED.jobid').write_text(job+'\n')
    print('JOB-ID '+job)
    print('QUEUE-AFTER\n'+ssh('squeue -u tawal01 -o "%.18i %.32j %.8T %.10M %.20R"'))
if __name__=='__main__':main()
