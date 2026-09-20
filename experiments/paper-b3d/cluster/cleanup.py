"""Remove one exact completed Burgers attempt after accepted durable retention."""
import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

EXP=Path(__file__).resolve().parents[1]
REMOTE='/cluster/tufts/paralab/tawal01/paper_b3d_20260920'


def ssh(args):
    return subprocess.check_output(['ssh','tufts-login',shlex.join(args)],text=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    assert args.attempt.isalnum() and args.attempt.startswith('b3d')
    run=EXP/'runs'/args.attempt;record=json.loads((run/'COLLECTED.json').read_text())
    submission=json.loads((run/'submission.json').read_text());proof=json.loads((run/'scientific-archive/git-restore-audit.json').read_text())
    assert record['checksums_passed'] and record['independent_audit_passed'] and proof['passed']
    assert not record['remote_directory_removed']
    job=str(submission['job_id']);assert job.isdecimal() and record['job_id']==job
    remote=REMOTE+'/'+args.attempt;assert submission['remote']==record['remote']==remote
    before=ssh(['squeue','-u','tawal01','-h','-o','%i|%j|%T'])
    assert job not in [v.split('|')[0] for v in before.splitlines()]
    assert ssh(['cat',remote+'/EXIT_CODE']).strip()=='0'
    for name in ('OUTPUTS.sha256','FINISHED_UTC'):
        actual=ssh(['sha256sum',remote+'/'+name]).split()[0]
        assert actual==hashlib.sha256((run/'collected'/name).read_bytes()).hexdigest()
    ssh(['rm','-rf','--',remote])
    subprocess.run(['ssh','tufts-login','test ! -e '+shlex.quote(remote)],check=True)
    record.update(remote_directory_removed=True,remote_cleanup_queue_before=before,
                  remote_cleanup_queue_after=ssh(['squeue','-u','tawal01','-h','-o','%i|%j|%T']))
    (run/'COLLECTED.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
