"""Install the obstacle campaign's pinned NGSolve wheels into a job-owned prefix.

Same wheels, hashes, pip flags and RECORD verification as vendor/dependencies.py
(byte copy from exp/2026-10-27-ns-obstacle-2d); only the namespace binding differs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
from dependencies import sha256_file, verify_prefix  # noqa: E402

NAMESPACE = Path('/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103')
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wheel-dir', required=True)
    ap.add_argument('--target', required=True)
    args = ap.parse_args()
    meta = json.loads((Path(__file__).resolve().parent / 'vendor/DEPENDENCIES.json').read_text())
    target = Path(args.target).resolve()
    if not target.is_relative_to(NAMESPACE) or target.exists():
        raise ValueError('target must be a fresh path inside this campaign namespace')
    if sys.executable != PY:
        raise RuntimeError('installation requires the designated cluster interpreter')
    wheel_dir = Path(args.wheel_dir).resolve()
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheels = []
    for name, package in meta['candidates'].items():
        (wheel,) = package['cluster_candidate_wheels']
        path = wheel_dir / wheel['filename']
        if not path.exists():
            partial = path.with_suffix('.partial')
            with urllib.request.urlopen(wheel['url'], timeout=60) as src, partial.open('xb') as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
            partial.rename(path)
        digest = sha256_file(path).hexdigest()
        if digest != wheel['digests']['sha256'] or path.stat().st_size != wheel['size']:
            raise ValueError(f'{path.name}: pinned byte hash/size mismatch')
        wheels.append(str(path))
        print(json.dumps({'wheel': path.name, 'sha256': digest}), flush=True)
    subprocess.run([sys.executable, '-m', 'pip', '--isolated', 'install', '--no-deps', '--no-index',
                    '--no-compile', '--no-cache-dir', '--disable-pip-version-check', '--ignore-installed',
                    '--prefix', str(target), *wheels], check=True, timeout=300)
    verify_prefix(target, meta, wheels)


if __name__ == '__main__':
    main()
