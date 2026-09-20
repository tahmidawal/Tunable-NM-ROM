"""Stage committed NS3D code for one isolated, bounded GPU attempt. Does not submit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[3]
REMOTE_ROOT='/cluster/tufts/paralab/tawal01/paper_ns3d_20260920'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('attempt')
    parser.add_argument('--config',default='pilot01.json')
    parser.add_argument('--gpu',choices=('a100','h200'),default='a100')
    parser.add_argument('--driver',choices=('pilot.py','comparison.py'),default='pilot.py')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9]{1,30}',args.attempt):
        raise ValueError('attempt must be a bounded alphanumeric name')
    if not re.fullmatch(r'[a-z][a-z0-9]*\.json',args.config):
        raise ValueError('config must be an existing named JSON')
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    files=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'experiments/ns3d').glob('*.py'))]
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'experiments/ns3d/operators').glob('*')) if p.is_file()]
    files += ['experiments/ns3d/DESIGN.md','experiments/ns3d/cluster/stage.py',
              'experiments/ns3d/configs/'+args.config,
              'experiments/ns2d/ns2d_decoder.py','experiments/ns2d/ns2d_rom.py',
              'experiments/ns2d/ns2d_fom.py','experiments/separable-decoder/sep_common.py']
    out=ROOT/'experiments/ns3d/runs'/args.attempt
    out.mkdir(parents=True,exist_ok=False)
    provenance=[]
    for name in files:
        path=ROOT/name
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',source+':'+name])
        if blob != path.read_bytes():
            raise RuntimeError('uncommitted source: '+name)
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
        provenance.append(dict(path=name,sha256=digest(dest),bytes=len(blob),source_commit=source))
    remote=REMOTE_ROOT+'/'+args.attempt
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'COMMIT.txt').write_text(source+'\n')
    script=f'''#!/bin/bash
#SBATCH --job-name=ctol_ns3d_920_{args.attempt}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:{args.gpu}:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/ns3d:$TASK_ROOT/experiments/ns2d:$TASK_ROOT/experiments/separable-decoder"
set +e
"$PY" experiments/ns3d/{args.driver} --config experiments/ns3d/configs/{args.config} --out output
NS3D_EXIT=$?
set -e
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo "PILOT_EXIT=$NS3D_EXIT"
exit "$NS3D_EXIT"
'''
    (out/'run.sbatch').write_text(script)
    (out/'logs').mkdir()
    stage=dict(attempt=args.attempt,source_commit=source,local=str(out),remote=remote,
               config=args.config,gpu=args.gpu,wall_limit_hours=2,driver=args.driver)
    (out/'stage.json').write_text(json.dumps(stage,indent=2)+'\n')
    manifest=[f'{digest(p)}  {p.relative_to(out)}' for p in sorted(out.rglob('*')) if p.is_file()]
    (out/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
    print(json.dumps(stage,indent=2))


if __name__=='__main__':
    main()
