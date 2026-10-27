from pathlib import Path
import shutil,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2];LANE=ROOT/'experiments/burgers-multilevel';stage=LANE/'runs/pilot01'
assert not stage.exists();stage.mkdir(parents=True)
paths=[]
for pattern in ['experiments/jcp-dims/*.py','experiments/jcp-dims/vendor/quad2d/*.py','experiments/jcp-dims/vendor/quad2d/vendor/*.py','experiments/separable-decoder/*.py','experiments/mr-burgers2d/*.py','experiments/burgers-multilevel/*.py']:
    paths+=list(ROOT.glob(pattern))
paths += [ROOT/'experiments/jcp-dims/vendor/quad2d/inputs/rotation_R512.npz',ROOT/'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
manifest=[]
for p in paths:
    rel=p.relative_to(ROOT);out=stage/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    manifest.append(dict(path=str(rel),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(LANE/'SOURCE-MANIFEST.json').write_text(json.dumps(manifest,indent=2))
ns='/cluster/tufts/paralab/tawal01/burgers-ml-20261027/pilot01'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
script=f'''#!/bin/bash
#SBATCH --job-name=ctol_bml_pilot01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output={ns}/stdout.log
#SBATCH --error={ns}/stderr.log
set -euo pipefail
cd {ns}
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export SOURCE_COMMIT={commit}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}'); sys.exit(0 if b=='gpu' else 42)"
sha256sum -c MANIFEST.sha256 --quiet
$PY experiments/burgers-multilevel/pilot.py output
'''
(stage/'run.sbatch').write_text(script)
(stage/'RESERVATION.json').write_text(json.dumps(dict(hours=1,cap=4,previous_spent=0,max_concurrent_lane_jobs=1,namespace=ns)))
files=sorted(p for p in stage.rglob('*') if p.is_file())
(stage/'MANIFEST.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(stage))+'\n' for p in files))
print(stage)
