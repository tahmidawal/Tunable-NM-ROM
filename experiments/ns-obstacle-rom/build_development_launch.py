"""Prepare only frozen-model development launch source; actual data/release are separate."""
from pathlib import Path
import argparse, json
from build_preparation_launch import source_closure, ROOT
from fe_adapter import sha_file
import stage2d


def development_sources():
    return sorted(set(source_closure()) | {'stage2d.py','paired_model2d.py','validation2d.py','development_launch2d.py'})


def build(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitignore').write_text('payload/\n')
    outer = (ROOT/'staging/fine-preparation01/native.sh').read_text()
    outer = outer.replace('ctol_nsobsrom_prepare','ctol_nsobsrom_development').replace('00:22:30','00:56:00').replace('1335s','3345s')
    (out/'native.sh').write_text(outer)
    body = '''#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?reviewed batch allocation required}"
JOB_DIR=$(pwd -P)
[ "$JOB_DIR" = /cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/model-development01 ] || exit 64
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" CUDA_CACHE_PATH="$JOB_DIR/cache/cuda"
export MPLCONFIGDIR="$JOB_DIR/cache/matplotlib"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" results
JOB_PHASE=source_verification
finish() {
  task_status=$?
  trap - EXIT
  set +e
  "$PY" - "$task_status" "$JOB_PHASE" <<'PY'
import json,sys
from pathlib import Path
p=Path('results/development-stage/STAGE-RESULT.json');r=json.loads(p.read_text()) if p.exists() else {}
Path('results/batch_summary.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'last_stage':sys.argv[2],
'development_completed':r.get('status')=='execution_completed_pending_independent_review',
'physical_admission_created':False,'automatic_retry':False},indent=2)+'\\n')
PY
  exit "$task_status"
}
trap finish EXIT
trap 'JOB_PHASE=allocation_deadline; exit 143' TERM
sha256sum -c SOURCE-SHA256SUMS
cat SOURCE_COMMIT.txt
"$PY" development_launch2d.py --candidate STAGE-CANDIDATE.json --release STAGE-RELEASE.json --inventory RUNTIME-INVENTORY.json --verify-only
JOB_PHASE=gpu_preflight
"$PY" - <<'PY'
import json,sys,platform,hashlib,jax,jaxlib,optax,numpy,scipy
from pathlib import Path
r={'python':platform.python_version(),'jax':jax.__version__,'jaxlib':jaxlib.__version__,'optax':optax.__version__,
'numpy':numpy.__version__,'scipy':scipy.__version__,'jax_backend':jax.default_backend(),
'jax_x64':jax.config.x64_enabled,'matmul_precision':str(jax.config.jax_default_matmul_precision),
'python_executable_sha256':hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
Path('results/runtime.json').write_text(json.dumps(r,indent=2)+'\\n')
print('jax_backend='+r['jax_backend'],flush=True);print(json.dumps(r),flush=True)
sys.exit(0 if r['jax_backend']=='gpu' and r['jax_x64'] and r['matmul_precision']=='highest' else 42)
PY
lscpu
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
JOB_PHASE=frozen_model_development
"$PY" development_launch2d.py --candidate STAGE-CANDIDATE.json --release STAGE-RELEASE.json --inventory RUNTIME-INVENTORY.json
JOB_PHASE=completed
'''
    (out/'native-body.sh').write_text(body)
    rows = [{'source':name,'destination':name,'bytes':(ROOT/name).stat().st_size,'sha256':sha_file(ROOT/name)} for name in development_sources()]
    for name in ('native.sh','native-body.sh'):
        path = out/name
        rows.append({'source':str(path.relative_to(ROOT)),'destination':name,'bytes':path.stat().st_size,'sha256':sha_file(path)})
    inventory = {'schema':'ns-obstacle-development-runtime-inventory-v1','files':rows,
                 'stage_source_sha256':stage2d.source_digest(),'builder_sha256':sha_file(__file__)}
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    plan = {'schema':'ns-obstacle-development-source-plan-v1','execution_released':False,
        'actual_stage_candidate_pending':True,'actual_preparation_and_replay_review_pending':False,
        'job_directory':'/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/model-development01',
        'allocation_seconds':3360,'outer_timeout_seconds':3345,'outer_kill_grace_seconds':10,
        'original_development_reservation_seconds':3600,'prior_materialization_reservation_seconds':240,
        'training_executed_by_this_job':False,'bank_recipe_unchanged':True,
        'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),
        'launcher_sha256':sha_file(ROOT/'development_launch2d.py'),'stage_source_sha256':stage2d.source_digest(),
        'input_contract':'Existing stage2d candidate binds every file in actual accepted prepared tree and separately reviewed consumer release.',
        'prepared_storage':'A hash-bound immutable warm PointVolumeStore is required; compact primary alone is not a consumer input.',
        'reconstruction_if_needed':'Use existing primary_volume.materialize with exact source/runtime; charge all reconstruction and setup within released resources.',
        'validation_data_access':False,'automatic_retry':False,'resume_authorized':False,
        'pending_before_execution':['actual model review and complete two-case development reconstruction review',
            'actual consumer release including measured-capacity and native-result review bindings',
            'actual development stage candidate from existing stage2d builder; all data paths must exist on execution host',
            'fresh stage-specific memory/disk/global quota and unique job-directory checks',
            'exact separate development release plus launch/runtime/candidate bindings and SOURCE-SHA256SUMS']}
    (out/'SOURCE-PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    print(json.dumps(build(p.parse_args().out),indent=2))
