"""Checksum-collect an exact completed attempt; cleanup is a separate explicit step."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('attempt');p.add_argument('--kind',choices=['calibration','followup','tuning'],required=True);a=p.parse_args()
    assert a.attempt.isalnum()
    remote=f'/cluster/tufts/paralab/tawal01/no_burgers_20260914/{a.attempt}'
    out=ROOT/'experiments/neural-operator-burgers/runs'/a.attempt/'archive';out.mkdir(parents=True,exist_ok=False)
    members=['experiments','COMMIT.txt','PROVENANCE.json','MANIFEST.sha256','run.sbatch','logs','OUTPUTS.sha256']
    members+={'calibration':['calibration','smoke'],'followup':['output','CALIBRATION.sha256'],'tuning':['output']}[a.kind]
    # The completed batch's own manifest must exist and verify before packaging.
    cmd=f'cd {shlex.quote(remote)} && sha256sum -c OUTPUTS.sha256 --quiet && sha256sum -c MANIFEST.sha256 --quiet && tar -czf collection.tar.gz '+ ' '.join(map(shlex.quote,members))+' && sha256sum collection.tar.gz > collection.tar.gz.sha256'
    subprocess.run(['ssh','tufts-login',cmd],check=True)
    for name in ['collection.tar.gz','collection.tar.gz.sha256']:
        subprocess.run(['scp',f'tufts-login:{remote}/{name}',str(out/name)],check=True)
    subprocess.run(['sha256sum','-c','collection.tar.gz.sha256'],cwd=out,check=True)
    subprocess.run(['tar','-xzf','collection.tar.gz'],cwd=out,check=True)
    subprocess.run(['sha256sum','-c','OUTPUTS.sha256','--quiet'],cwd=out,check=True)
    subprocess.run(['sha256sum','-c','MANIFEST.sha256','--quiet'],cwd=out,check=True)
    print(out)
    print('Checksums verified; exact remote cleanup remains explicit after audits/cached-reference handoff.')


if __name__=='__main__':main()
