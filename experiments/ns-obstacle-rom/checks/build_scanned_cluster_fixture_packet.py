"""Prepare an exact, source-only cluster fallback packet; never submits."""
from pathlib import Path
import ast,hashlib,json,shutil
B=Path(__file__).resolve().parents[1]
P=B/'staging/scanned-composite-fixture-cluster01'
JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/scanned-composite-fixture-cluster01'
R=B.parents[1].parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    prior=R/'SCANNED-COORDINATE-COMPOSITE-FIXTURE-RELEASE.json'
    assert sha(prior)=='4aed0c77efad955b2507dea010b4719654ebf483c16c31516f7afc45c48c40cd'
    r=json.loads(prior.read_text());P.mkdir(exist_ok=False)
    for n,s in r['source_files'].items():assert sha(B/n)==s
    model=Path(r['model_path']);review=Path(r['model_review_path'])
    assert sha(model)==r['model_sha256'] and model.stat().st_size==477824
    assert sha(review)==r['model_review_sha256'] and json.loads(review.read_text())['passed'] is True
    pending=list(r['source_files']);found=set()
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name)
        if not name.endswith('.py'):continue
        for node in ast.walk(ast.parse((B/name).read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[v.name.replace('.','/')+'.py' for v in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module.replace('.','/')+'.py']
            elif isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py'):names=[node.value]
            pending.extend(n for n in names if not Path(n).is_absolute() and '..' not in Path(n).parts and (B/n).is_file())
    plan={'schema':'ns-obstacle-scanned-cluster-fixture-plan-v1','job_directory':JOB,'model_sha256':sha(model),'model_bytes':477824,'model_review_sha256':sha(review),'points_sha256':r['points_fixture_sha256'],'prior_failed_local_release_sha256':sha(prior),'prior_failed_execution_sha256':sha(B/'checks/scanned-coordinate-composite-fixture01-EXECUTION.json'),'fixture_source_sha256':r['fixture_source_sha256'],'allocation_seconds':120,'outer_timeout_seconds':105,'kill_grace_seconds':5,'memory_bytes':16*2**30,'raw_plus_archive_reservation_bytes':256*2**20,'automatic_retry':False,'scope':'same339 points and unchanged numerical fixture; no TRAIN/PDE/fit or timing comparison'}
    (P/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    shutil.copyfile(review,P/'MODEL-REVIEW.json')
    shell='''#!/bin/bash
#SBATCH --job-name=ctol_nsobsrom_scanfixture
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
exec /usr/bin/timeout --signal=TERM --kill-after=5s 105s /bin/bash ./native-body.sh
'''
    body='''#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?batch allocation required}"
[ "$(pwd -P)" = JOB_LITERAL ] || exit64
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_CACHE_MAXSIZE=16777216 JAX_ENABLE_COMPILATION_CACHE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/cache" CUDA_CACHE_PATH="$PWD/cache/cuda"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" results
sha256sum -c SOURCE-SHA256SUMS
"$PY" -c 'import jax,json,sys; b=jax.default_backend(); print("jax_backend="+b); r={"backend":b,"x64":jax.config.x64_enabled,"matmul_precision":str(jax.config.jax_default_matmul_precision),"devices":[d.device_kind for d in jax.devices()]};open("results/runtime.json","w").write(json.dumps(r));sys.exit(0 if b=="gpu" and r["x64"] and r["matmul_precision"]=="highest" else42)'
exec "$PY" -m accuracy_v2.residual_composite_fixture --model model.npz --model-sha256 MODEL_SHA --points-fixture points.npz --points-sha256 POINTS_SHA --output results/fixture
'''.replace('JOB_LITERAL',JOB).replace('exit64','exit 64').replace('else42','else 42').replace('MODEL_SHA',sha(model)).replace('POINTS_SHA',r['points_fixture_sha256'])
    (P/'native.sh').write_text(shell);(P/'native-body.sh').write_text(body)
    rows=[{'source':str(B/n),'destination':n,'bytes':(B/n).stat().st_size,'sha256':sha(B/n)} for n in sorted(found)]
    for source,dest in [(model,'model.npz'),(B/'checks/fast-traction-bank-kernel/arrays.npz','points.npz')]+[(P/n,n) for n in ['PLAN.json','MODEL-REVIEW.json','native.sh','native-body.sh']]:
        rows.append({'source':str(source),'destination':dest,'bytes':source.stat().st_size,'sha256':sha(source)})
    (P/'RUNTIME-INVENTORY.json').write_text(json.dumps({'files':rows,'builder_sha256':sha(__file__)},indent=2)+'\n')
    print(json.dumps({'runtime_files':len(rows),'runtime_bytes':sum(x['bytes'] for x in rows),'plan_sha256':sha(P/'PLAN.json'),'inventory_sha256':sha(P/'RUNTIME-INVENTORY.json')}))
if __name__=='__main__':main()
