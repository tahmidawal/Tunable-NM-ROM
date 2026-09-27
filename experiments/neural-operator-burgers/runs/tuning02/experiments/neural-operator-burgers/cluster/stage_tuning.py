"""Stage the fixed-checkpoint tuning job from committed source bytes into its own attempt.

One attempt directory per submitted job.  The verified cluster-generated dataset and
reference cache is read in place and never written to or removed by this job.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
attempt = sys.argv[1]
phase = sys.argv[2]
minutes = int(sys.argv[3])
assert attempt.isalnum() and phase in ('calibration', 'validation') and 20 <= minutes <= 180
extra = sys.argv[4:]
out = root / 'experiments/neural-operator-burgers/runs' / attempt
out.mkdir(parents=True, exist_ok=False)
remote = f'/cluster/tufts/paralab/tawal01/no_burgers_20260914/{attempt}'
cache = '/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01'
commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
files = ['experiments/neural-operator-burgers/' + n for n in
         ['data.py', 'protocol.json', 'protocol-refined.json', 'refine.py', 'diagnose.py',
          'tuning.py', 'tuning-config.json', 'cluster/stage_tuning.py']]
if phase == 'validation':
    files.append('experiments/neural-operator-burgers/checks/tuning02-shortlist.json')
files += ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/accuracy_paths.py',
          'experiments/separable-decoder/sep_common.py',
          'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
proof = []
for name in files:
    content = (root / name).read_bytes()
    assert content == subprocess.check_output(['git', '-C', str(root), 'show', f'{commit}:{name}']), name
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    proof.append(dict(source=name, sha256=hashlib.sha256(content).hexdigest(), commit=commit))
(out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
(out / 'COMMIT.txt').write_text(commit + '\n')
(out / 'logs').mkdir()
script = '''#!/bin/bash
#SBATCH --job-name=ctol_nob_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=MINUTES
#SBATCH --output=REMOTE/logs/%j.out
#SBATCH --error=REMOTE/logs/%j.err
set -euo pipefail
TASK_ROOT=REMOTE
CACHE=CACHEDIR
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
export PYTHONPATH="$TASK_ROOT/experiments/neural-operator-burgers:$TASK_ROOT/experiments/mr-burgers2d"
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" experiments/neural-operator-burgers/tuning.py --phase PHASE \\
  --reference "$CACHE/refinement/index.json" --dataset "$CACHE/validation/index.json" \\
  --out "$TASK_ROOT/output" --seconds SECONDS SHORTLIST EXTRA
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''.replace('ATTEMPT', attempt).replace('MINUTES', f'{minutes // 60:02d}:{minutes % 60:02d}:00') \
   .replace('REMOTE', remote).replace('CACHEDIR', cache).replace('PHASE', phase) \
   .replace('SECONDS', str(minutes * 60 - 600)) \
   .replace('SHORTLIST', '--shortlist "$TASK_ROOT/experiments/neural-operator-burgers/checks/tuning02-shortlist.json"'
            if phase == 'validation' else '').replace('EXTRA', ' '.join(extra))
(out / 'run.sbatch').write_text(script)
manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
            for p in sorted(out.rglob('*')) if p.is_file()]
(out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
print(out)
print(remote)
