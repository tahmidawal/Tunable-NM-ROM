"""Return only this lane's completed artifacts, verify bytes, optionally clean up."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE=Path(__file__).resolve().parents[1]
NS='/cluster/tufts/paralab/tawal01/b3d-recon-20261027'


def run(args,**kw):
    return subprocess.check_output(args,text=True,**kw)


def remote(command):return run(['ssh','tufts-login',command])


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(root,manifest):
    count=size=0
    for line in manifest.read_text().splitlines():
        digest,relative=line.split(None,1)
        p=root/relative.strip()
        assert p.resolve().is_relative_to(root.resolve()),p
        assert checksum(p)==digest,p
        count+=1;size+=p.stat().st_size
    return dict(files=count,bytes=size,manifest_sha256=checksum(manifest))


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--job',required=True,type=int)
    p.add_argument('--cleanup',action='store_true');a=p.parse_args()
    assert re.fullmatch('[a-zA-Z0-9]+',a.attempt)
    remote_dir=f'{NS}/{a.attempt}';root=HERE/'results'/a.attempt
    root.mkdir(parents=True,exist_ok=True)
    accounting=remote(f'sacct -j {a.job} --parsable2 --format=JobID,State,ExitCode,ElapsedRaw,AllocTRES -n')
    lines=[line.split('|') for line in accounting.splitlines()]
    master=next(line for line in lines if line[0]==str(a.job))
    assert master[1]=='COMPLETED' and master[2]=='0:0',master
    assert 'gres/gpu=1' in master[4],master
    (root/'SACCT.txt').write_text(accounting)
    remote(f'cd {shlex.quote(remote_dir)} && sha256sum logs/* > LOGS.sha256')
    for name in ('output','logs'):
        subprocess.check_call(['rsync','-a','--exclude=regenerable/',f'tufts-login:{remote_dir}/{name}',str(root)+'/'])
    for name in ('OUTPUTS.sha256','LOGS.sha256','MANIFEST.sha256','COMMIT.txt','run.sbatch'):
        subprocess.check_call(['scp',f'tufts-login:{remote_dir}/{name}',str(root/name)])
    outputs=verify(root,root/'OUTPUTS.sha256');logs=verify(root,root/'LOGS.sha256')
    expected=HERE/'runs'/a.attempt/'MANIFEST.sha256'
    assert expected.read_bytes()==(root/'MANIFEST.sha256').read_bytes()
    receipt=dict(job_id=a.job,source_commit=(root/'COMMIT.txt').read_text().strip(),
        gpu_seconds=int(master[3]),gpu_hours=int(master[3])/3600,
        source_manifest_sha256=checksum(expected),outputs=outputs,logs=logs,
        verified=True,raw_training_matrices_excluded=True,remote_cleaned=False)
    (root/'RETURN.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if a.cleanup:
        queue=remote('squeue -u "$USER" -h -o "%i %j"')
        assert not any(' br3_' in row for row in queue.splitlines()),'own active consumer remains'
        # Exact path is constructed solely from a validated lane-owned attempt.
        remote(f'test -f {shlex.quote(remote_dir+"/output/COMPLETE.json")} && rm -rf -- {shlex.quote(remote_dir)}')
        receipt['remote_cleaned']=True
        (root/'RETURN.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
