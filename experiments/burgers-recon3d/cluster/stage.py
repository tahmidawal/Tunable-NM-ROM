"""Stage one committed reconstruction attempt; does not submit it."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
LANE='experiments/burgers-recon3d'
NS='/cluster/tufts/paralab/tawal01/b3d-recon-20261027'
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--mode',choices=['smoke','prepare','screen'],required=True)
p.add_argument('--prepared',default='prep01');a=p.parse_args()
assert a.attempt.isalnum() and a.prepared.isalnum()
commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files',LANE],text=True).splitlines()
files=[f for f in tracked if not any(s in f for s in ['/runs/','/results/','/__pycache__/'])]
files += ['experiments/jcp-combined/inputs/model_W1024/bank.pkl','experiments/jcp-combined/inputs/model_W1024/training.json']
out=ROOT/LANE/'runs'/a.attempt;out.mkdir(parents=True,exist_ok=False)
proof=[]
for rel in files:
 content=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{rel}'])
 assert (ROOT/rel).read_bytes()==content,rel
 dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
 proof.append({'path':rel,'commit':commit,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
(out/'PROVENANCE.json').write_text(json.dumps(proof,indent=2)+'\n');(out/'COMMIT.txt').write_text(commit+'\n')
(out/'logs').mkdir()
remote=f'{NS}/{a.attempt}'
limits={'smoke':('00:10:00','128G'),'prepare':('02:00:00','384G'),'screen':('01:30:00','192G')}
wall,mem=limits[a.mode]
cmd=(f'"$PY" -u {LANE}/prepare3d.py --bank experiments/jcp-combined/inputs/model_W1024/bank.pkl '
 f'--training experiments/jcp-combined/inputs/model_W1024/training.json --out "$TASK_ROOT/output"'+(' --smoke' if a.mode=='smoke' else ''))
if a.mode=='smoke': cmd=f'\"$PY\" -u {LANE}/vendor/check_core.py && \"$PY\" -u {LANE}/check_prepare.py && '+cmd
if a.mode=='screen': cmd=f'"$PY" -u {LANE}/screen3d.py --prepared {NS}/{a.prepared}/output --out "$TASK_ROOT/output"'
script=f'''#!/bin/bash
#SBATCH --job-name=br3_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={wall}
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
export JCPREF_META="$JCPREF_STORE"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
{cmd}
find output -type f ! -path 'output/regenerable/*' -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
(out/'run.sbatch').write_text(script)
manifest=[]
for file in sorted(out.rglob('*')):
 if file.is_file():manifest.append(f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(out)}')
(out/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
print(json.dumps({'local':str(out),'remote':remote,'commit':commit,'mode':a.mode,'wall':wall,'memory':mem},indent=2))
