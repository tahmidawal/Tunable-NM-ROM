"""Prepare an isolated, hashed cluster package; does not submit jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NAMESPACE='/cluster/tufts/paralab/tawal01/b3d-pod-20261026'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('attempt');a=ap.parse_args()
    cfg=HERE/'configs'/f'{a.attempt}.json'
    j=json.loads(cfg.read_text())
    assert j['attempt']==a.attempt and a.attempt in ('pilot','validation')
    d=HERE/'runs'/a.attempt
    assert not d.exists(), 'inspect existing job state; do not overwrite a staged attempt'
    d.mkdir()
    sources=list((ROOT/'experiments/jcp-hr-baselines').glob('*.py'))
    sources+=list((ROOT/'experiments/jcp-hr-baselines/vendor').rglob('*.py'))
    sources+=list((ROOT/'experiments/jcp-hr-baselines/e0').rglob('*.py'))
    sources += [ROOT/'experiments/jcp-hr-baselines/e0/store/MANIFEST.sha256',
                ROOT/'experiments/jcp-hr-baselines/e0/store/STATUS.json',
                ROOT/'experiments/mr-burgers2d/engines.py',ROOT/'experiments/separable-decoder/sep_common.py']
    sources += [p for p in HERE.rglob('*') if p.is_file() and p.name != 'Phi.npz'
                and not any(x in p.parts for x in ('runs','__pycache__'))]
    sources=sorted(set(sources))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  fork='93f399faad67f94ea2b56fe25f13003143e64694',files=hashes,
                  config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest())
    (d/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(d/'source.tar','w') as t:
        for p in sources:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        t.add(d/'PROVENANCE.json',arcname='PROVENANCE.json')
    archive=hashlib.sha256((d/'source.tar').read_bytes()).hexdigest()
    (d/'source.tar.sha256').write_text(archive+'  source.tar\n')
    remote=f'{NAMESPACE}/{a.attempt}'
    # All runtime outputs and caches stay inside this unique job directory.
    sb=f'''#!/usr/bin/env bash
#SBATCH --job-name=b3dpod_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time={int(j['wall_hours']):02}:00:00
#SBATCH --output={remote}/slurm-%j.log
set -euo pipefail
cd {remote}
export PYTHONDONTWRITEBYTECODE=1
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
export TMPDIR={remote}/tmp
export XDG_CACHE_HOME={remote}/cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}'); sys.exit(0 if b=='gpu' else 42)"
sha256sum -c source.tar.sha256
tar -xf source.tar
$PY - <<'PY'
from pathlib import Path
import hashlib,json
p=json.loads(Path('PROVENANCE.json').read_text())
for f,h in p['files'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
print('source_provenance=verified',flush=True)
PY
mkdir artifacts
cp /cluster/tufts/paralab/tawal01/burgers-archive/jcp-hr-baselines/t3d64b/jobdir/output/artifacts/Phi.npz artifacts/
cp /cluster/tufts/paralab/tawal01/burgers-archive/jcp-hr-baselines/t3d64b/jobdir/output/artifacts/HR~ecsw~512~k8.npz artifacts/
cp /cluster/tufts/paralab/tawal01/burgers-archive/jcp-hr-baselines/t3d64b/jobdir/output/artifacts/HR~ecsw~512~k16.npz artifacts/
$PY experiments/b3d-pod/replay.py --config experiments/b3d-pod/configs/{a.attempt}.json --artifacts artifacts --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
'''
    (d/'run.sbatch').write_text(sb)
    print(json.dumps(dict(local=str(d),remote=remote,source_sha256=archive,files=len(sources)),indent=2))


if __name__=='__main__':main()
