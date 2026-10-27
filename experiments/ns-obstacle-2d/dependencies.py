"""Download exact hashed binary wheels; install only with explicit --install flag.

No dependency resolution and no shared-environment mutation. Caller must supply a new
lane-local target. Downloaded wheels are execution inputs, not generated training data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--wheel-dir',required=True)
    ap.add_argument('--install',action='store_true')
    ap.add_argument('--target')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    meta=json.loads((root/'DEPENDENCIES.json').read_text())
    wheel_dir=Path(args.wheel_dir).resolve();wheel_dir.mkdir(parents=True,exist_ok=True)
    wheels=[]
    for name,package in meta['candidates'].items():
        choices=package['cluster_candidate_wheels']
        if len(choices)!=1:raise ValueError(f'{name}: expected one pinned wheel, got {len(choices)}')
        wheel=choices[0];path=wheel_dir/wheel['filename']
        if not path.exists():
            partial=path.with_suffix('.partial')
            with urllib.request.urlopen(wheel['url'],timeout=60) as src,partial.open('xb') as dst:
                while block:=src.read(1024*1024):dst.write(block)
            partial.rename(path)
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        if digest!=wheel['digests']['sha256'] or path.stat().st_size!=wheel['size']:
            raise ValueError(f'{path.name}: pinned byte hash/size mismatch')
        wheels.append(str(path))
        print(json.dumps({'wheel':path.name,'sha256':digest,'bytes':path.stat().st_size}),flush=True)
    if args.install:
        if not args.target:raise ValueError('--target required with --install')
        target=Path(args.target).resolve()
        if target.exists():raise FileExistsError('target exists: refusing to mix installations')
        allowed='/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/'
        if not str(target).startswith(allowed):raise ValueError('target must be inside this lane cluster namespace')
        if sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python':
            raise RuntimeError('installation requires designated cluster interpreter')
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--no-compile','--target',str(target),*wheels],check=True,timeout=180)

if __name__=='__main__':main()
