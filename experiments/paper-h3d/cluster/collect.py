"""Collect one exact attempt, verify remote checksums, and optionally remove it."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
NS='/cluster/tufts/paralab/tawal01/paper_h3d_20260920'


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--remove-verified',action='store_true');a=p.parse_args()
    assert a.attempt.isalnum()
    remote=f'{NS}/{a.attempt}';local=ROOT/'experiments/paper-h3d/runs'/a.attempt/'archive'
    local.mkdir(parents=True,exist_ok=False)
    subprocess.run(['scp','-r',f'tufts-login:{remote}/.',str(local)],check=True)
    subprocess.run(['sha256sum','-c','SOURCE.sha256'],cwd=local,check=True)
    subprocess.run(['sha256sum','-c','OUTPUTS.sha256'],cwd=local,check=True)
    subprocess.run(['/home/tahmid/Dev/.venv/bin/python',str(ROOT/'experiments/paper-h3d/audit.py'),str(local/'out')],check=True)
    if a.remove_verified:
        # Only the literal namespace + validated alphanumeric attempt is removed.
        subprocess.run(['ssh','tufts-login',f'rm -rf -- {remote}'],check=True)
    (local.parent/'COLLECTED.json').write_text(json.dumps(dict(remote=remote,checksums_verified=True,
              removed=a.remove_verified,archive=str(local)),indent=2)+'\n')


if __name__=='__main__':main()
